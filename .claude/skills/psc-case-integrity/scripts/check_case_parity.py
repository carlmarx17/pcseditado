#!/usr/bin/env python3
"""Verifica la integridad estructural de la matriz de casos PSC.

Uso:
    python3 check_case_parity.py <ruta-a-src>

Compara los .cxx de anisotropia ignorando kappa / beta / anisotropia y
reporta cualquier OTRA diferencia, que es por definicion una divergencia
estructural. Sale con codigo 1 si encuentra hallazgos.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

HEADER = "psc_anisotropy_case.hxx"
RECON_HEADER = "psc_reconnection_case.hxx"
HEADERS = (HEADER, RECON_HEADER)

IDENTITY = ["PSC_CASE_LABEL", "PSC_DISTRIBUTION_LABEL", "PSC_OUTPUT_BASENAME"]
REGIME = [
    "PSC_BETA_I_PAR",
    "PSC_BETA_E_PAR",
    "PSC_TI_PERP_OVER_TI_PAR",
    "PSC_TE_PERP_OVER_TE_PAR",
]
DISTRIBUTION = ["PSC_USE_KAPPA", "PSC_KAPPA"]
BOX = ["PSC_DOMAIN_DI"]
WHITELIST = set(IDENTITY + REGIME + DISTRIBUTION + BOX)

# Defines que solo deben vivir en el header compartido.
STRUCTURAL_HINT = {
    "PSC_NGRID_DEFAULT": "grilla",
    "PSC_NICELL_DEFAULT": "particulas por celda",
    "PSC_MASS_RATIO": "razon de masas",
    "PSC_VA_OVER_C": "velocidad de Alfven",
    "PSC_LAMBDA0": "lambda0",
    "PSC_NMAX_DEFAULT": "pasos maximos",
    "PSC_FIELDS_EVERY_DEFAULT": "intervalo de campos",
    "PSC_PARTICLES_EVERY_DEFAULT": "intervalo de particulas",
    "PSC_CHECKPOINT_EVERY_DEFAULT": "intervalo de checkpoint",
    "PSC_BALANCE_INTERVAL": "balanceo de carga",
    "PSC_NP_Y_DEFAULT": "descomposicion en y",
    "PSC_NP_Z_DEFAULT": "descomposicion en z",
    "PSC_STATS_EVERY": "intervalo de stats",
    "PSC_ENERGIES_EVERY_DEFAULT": "intervalo de energias",
    "PSC_CONTINUITY_EVERY_DEFAULT": "intervalo de continuidad",
    # invariantes de la hoja de Harris (setup B)
    "PSC_L_DI": "espesor de hoja",
    "PSC_TI_TE": "razon Ti/Te",
    "PSC_DBY_B0": "amplitud de perturbacion",
    "PSC_BG": "campo guia",
    "PSC_LPERT_LZ": "longitud de onda de perturbacion",
    "PSC_WPE_WCE": "razon wpe/wce",
    "PSC_LX_DI": "caja en x",
    "PSC_LY_DI": "caja en y",
    "PSC_LZ_DI": "caja en z",
}

# Cantidades del fondo que el header DERIVA de los betas anclados.
# Escritas a mano en un caso, desincronizan el beta real del beta declarado.
DERIVED_FROM_ANCHOR = {
    "PSC_NB_N0": "densidad de fondo",
    "PSC_TIB_TI": "temperatura de fondo ionica",
    "PSC_TEB_TE": "temperatura de fondo electronica",
}

FAMILIES = ("mirror", "firehose", "whistler")
REGIMES = ("strong", "moderate", "weak")
RECON_PREFIX = "reconnection_"

DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)\s*(.*?)\s*$")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+(.+?)\s*$')
CASE_GLOB = "psc_*.cxx"

# Valores citados en el comentario de encabezado.
COMMENT_VALUES = [
    (re.compile(r"beta_i_parallel\s*=\s*([\d.]+)"), "PSC_BETA_I_PAR"),
    (re.compile(r"beta_e_parallel\s*=\s*([\d.]+)"), "PSC_BETA_E_PAR"),
    (re.compile(r"\bAi\s*=\s*(?:Ti_perp/Ti_parallel\s*=\s*)?([\d.]+)"),
     "PSC_TI_PERP_OVER_TI_PAR"),
    (re.compile(r"\bAe\s*=\s*(?:Te_perp/Te_parallel\s*=\s*)?([\d.]+)"),
     "PSC_TE_PERP_OVER_TE_PAR"),
]
COMMENT_STRUCTURAL = [
    (re.compile(r"(\d+)\s*ppc"), "PSC_NICELL_DEFAULT", "particulas por celda"),
    (re.compile(r"(\d+)\s*x\s*\1|(\d+)\^?2\b"), None, None),  # manejado aparte
]
GRID_IN_COMMENT = re.compile(r"(\d{3,4})\s*[x×]\s*(\d{3,4})")
PPC_IN_COMMENT = re.compile(r"(\d{2,5})\s*ppc")
MR_IN_COMMENT = re.compile(r"mass_ratio\s*=\s*([\d.]+)")


def num(value: str):
    """Normaliza un literal numerico para comparar 5 con 5.0."""
    try:
        return float(value.rstrip("f"))
    except ValueError:
        return None


class Case:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem  # psc_mirror_bimaxwellian_moderate
        self.label = self.name[4:] if self.name.startswith("psc_") else self.name
        self.defines: dict[str, str] = {}
        self.includes: list[str] = []
        self.stray_lines: list[tuple[int, str]] = []
        self.comment = ""
        self._parse()

    def _parse(self) -> None:
        comment_lines = []
        in_leading_comment = True
        for lineno, raw in enumerate(
            self.path.read_text(errors="replace").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                if in_leading_comment:
                    comment_lines.append(line)
                continue
            in_leading_comment = False
            m = DEFINE_RE.match(raw)
            if m:
                self.defines[m.group(1)] = m.group(2).strip()
                continue
            m = INCLUDE_RE.match(raw)
            if m:
                self.includes.append(m.group(1).strip())
                continue
            self.stray_lines.append((lineno, line))
        self.comment = "\n".join(comment_lines)

    @property
    def is_case(self) -> bool:
        return any(h in inc for inc in self.includes for h in HEADERS)

    @property
    def setup(self) -> str:
        """'B' = reconexion (hoja de Harris), 'A' = anisotropia uniforme."""
        if self.label.startswith(RECON_PREFIX) or any(
            RECON_HEADER in inc for inc in self.includes
        ):
            return "B"
        return "A"

    @property
    def is_isotropic_control(self) -> bool:
        return self.label.endswith("_isotropic")

    @property
    def uses_kappa(self) -> bool:
        return num(self.defines.get("PSC_USE_KAPPA", "0")) == 1.0

    def regime_signature(self) -> tuple:
        return tuple(num(self.defines.get(k, "")) for k in REGIME)

    def non_regime_defines(self) -> dict[str, str]:
        """Todo lo que NO es identidad, kappa, beta ni anisotropia."""
        skip = set(IDENTITY) | set(REGIME) | set(DISTRIBUTION)
        return {k: v for k, v in self.defines.items() if k not in skip}

    def family(self):
        """Inestabilidad: mirror / firehose / whistler (None en controles)."""
        for fam in FAMILIES:
            if self.label.startswith(fam + "_") or f"_{fam}_" in self.label:
                return fam
        return None

    def regime(self):
        for reg in REGIMES:
            if self.label.endswith("_" + reg) or f"_{reg}_" in self.label:
                return reg
        return None

    def anchor_label(self):
        """Para un caso de reconexion, el caso de anisotropia pura que lo ancla."""
        if self.setup != "B" or self.is_isotropic_control:
            return None
        fam, reg, dist = self.family(), self.regime(), self.distribution_key()
        if not (fam and reg and dist):
            return None
        return f"{fam}_{dist}_{reg}"

    def distribution_key(self):
        """bimaxwellian, bikappa3, ... tal como aparece en el nombre."""
        m = re.search(r"_(bimaxwellian|bikappa\d+)", self.label)
        return m.group(1) if m else None

    def base_label(self):
        """Label sin el sufijo de caja, para emparejar gemelos bigbox."""
        return re.sub(r"_bigbox\d+$", "", self.label)


class Report:
    def __init__(self):
        self.findings: list[tuple[str, str, str]] = []

    def add(self, section: str, where: str, msg: str) -> None:
        self.findings.append((section, where, msg))

    def emit(self) -> int:
        if not self.findings:
            print("OK: la matriz de casos es estructuralmente consistente.")
            return 0
        by_section: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for section, where, msg in self.findings:
            by_section[section].append((where, msg))
        for section in sorted(by_section):
            print(f"\n## {section}")
            for where, msg in by_section[section]:
                print(f"  [{where}] {msg}")
        print(f"\n{len(self.findings)} hallazgo(s).")
        return 1


def check_structure(case: Case, rep: Report) -> None:
    own_header = RECON_HEADER if case.setup == "B" else HEADER
    for lineno, line in case.stray_lines:
        rep.add(
            "Codigo en un archivo de caso",
            case.name,
            f"linea {lineno}: los casos deben ser solo #define + include "
            f"del header compartido, pero hay -> {line[:70]}",
        )

    extra_includes = [i for i in case.includes
                      if not any(h in i for h in HEADERS)]
    for inc in extra_includes:
        rep.add(
            "Codigo en un archivo de caso",
            case.name,
            f"include extra {inc}: la fisica va en {own_header}, no en el caso",
        )

    if case.setup == "B" and any(HEADER in i for i in case.includes):
        rep.add(
            "Setup equivocado",
            case.name,
            f"un caso de reconexion incluye {HEADER}; debe usar "
            f"{RECON_HEADER}, que tiene la geometria de hoja de Harris",
        )

    for key in case.defines:
        if key in WHITELIST:
            if key == "PSC_DOMAIN_DI" and case.setup == "B":
                rep.add(
                    "Define fuera de la lista blanca",
                    case.name,
                    "PSC_DOMAIN_DI es una excepcion solo del setup de "
                    "anisotropia uniforme; en reconexion la caja la fija "
                    "el header (Lx_di/Ly_di/Lz_di)",
                )
            continue
        derived = DERIVED_FROM_ANCHOR.get(key)
        if derived:
            rep.add(
                "Cantidad derivada escrita a mano",
                case.name,
                f"{key} ({derived}) lo deriva {own_header} a partir de los "
                f"betas anclados; escrito a mano, el beta real del fondo deja "
                f"de coincidir con el beta declarado y el ancla cruzada se "
                f"vuelve mentira",
            )
            continue

        hint = STRUCTURAL_HINT.get(key)
        if hint:
            rep.add(
                "Define estructural fuera del header",
                case.name,
                f"{key} ({hint}) esta definido en el caso; pertenece a "
                f"{own_header} y ponerlo aca lo separa de sus hermanos",
            )
        else:
            rep.add(
                "Define fuera de la lista blanca",
                case.name,
                f"{key} no esta en la lista blanca de defines por caso",
            )

    for key in IDENTITY + REGIME:
        if key not in case.defines:
            rep.add("Define obligatorio faltante", case.name, f"falta {key}")


def check_identity(case: Case, rep: Report) -> None:
    label = case.defines.get("PSC_CASE_LABEL", "").strip('"')
    basename = case.defines.get("PSC_OUTPUT_BASENAME", "").strip('"')
    dist = case.defines.get("PSC_DISTRIBUTION_LABEL", "").strip('"')

    if label and label != case.label:
        rep.add(
            "Identidad",
            case.name,
            f'PSC_CASE_LABEL "{label}" no coincide con el nombre del '
            f'archivo ("{case.label}")',
        )
    if basename and label and basename != f"prt_{label}":
        rep.add(
            "Identidad",
            case.name,
            f'PSC_OUTPUT_BASENAME "{basename}" deberia ser "prt_{label}"',
        )
    expected = "Bi-Kappa" if case.uses_kappa else "Bi-Maxwellian"
    if dist and dist != expected:
        rep.add(
            "Identidad",
            case.name,
            f'PSC_DISTRIBUTION_LABEL "{dist}" contradice PSC_USE_KAPPA '
            f"(esperado \"{expected}\")",
        )
    if "PSC_KAPPA" in case.defines and not case.uses_kappa:
        rep.add(
            "Identidad",
            case.name,
            "define PSC_KAPPA pero no activa PSC_USE_KAPPA 1: kappa se ignora",
        )
    if case.uses_kappa and "PSC_KAPPA" not in case.defines:
        rep.add(
            "Identidad",
            case.name,
            "activa PSC_USE_KAPPA 1 pero no define PSC_KAPPA (usa el default)",
        )
    if "bikappa" in case.label and not case.uses_kappa:
        rep.add(
            "Identidad",
            case.name,
            "el nombre dice bikappa pero el caso corre bi-Maxwelliano",
        )


def check_comment(case: Case, rep: Report, header_defaults: dict) -> None:
    if not case.comment:
        return
    for pattern, key in COMMENT_VALUES:
        m = pattern.search(case.comment)
        if not m:
            continue
        commented = num(m.group(1))
        actual = num(case.defines.get(key, ""))
        if commented is not None and actual is not None and commented != actual:
            rep.add(
                "Comentario desactualizado",
                case.name,
                f"el encabezado dice {key.replace('PSC_', '').lower()}="
                f"{m.group(1)} pero el define es {case.defines[key]}",
            )

    m = GRID_IN_COMMENT.search(case.comment)
    if m:
        commented = num(m.group(1))
        actual = num(case.defines.get("PSC_NGRID_DEFAULT",
                                      header_defaults.get("PSC_NGRID_DEFAULT", "")))
        if commented is not None and actual is not None and commented != actual:
            rep.add(
                "Comentario desactualizado",
                case.name,
                f"el encabezado dice grilla {m.group(1)}x{m.group(2)} pero el "
                f"valor efectivo es {int(actual)}",
            )

    m = PPC_IN_COMMENT.search(case.comment)
    if m:
        commented = num(m.group(1))
        actual = num(case.defines.get("PSC_NICELL_DEFAULT",
                                      header_defaults.get("PSC_NICELL_DEFAULT", "")))
        if commented is not None and actual is not None and commented != actual:
            rep.add(
                "Comentario desactualizado",
                case.name,
                f"el encabezado dice {m.group(1)} ppc pero el valor efectivo "
                f"es {int(actual)}",
            )

    m = MR_IN_COMMENT.search(case.comment)
    if m:
        commented = num(m.group(1))
        actual = num(case.defines.get("PSC_MASS_RATIO",
                                      header_defaults.get("PSC_MASS_RATIO", "")))
        if commented is not None and actual is not None and commented != actual:
            rep.add(
                "Comentario desactualizado",
                case.name,
                f"el encabezado dice mass_ratio={m.group(1)} pero el valor "
                f"efectivo es {actual}",
            )


def check_regime_siblings(cases: list[Case], rep: Report) -> None:
    """Hermanos strong/moderate/weak: solo pueden diferir en beta/anisotropia."""
    groups: dict[tuple, list[Case]] = defaultdict(list)
    for c in cases:
        fam, dist, reg = c.family(), c.distribution_key(), c.regime()
        if fam and dist and reg and "bigbox" not in c.label:
            groups[(c.setup, fam, dist)].append(c)

    for (setup, fam, dist), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ref = members[0]
        ref_extra = ref.non_regime_defines()
        for other in members[1:]:
            other_extra = other.non_regime_defines()
            for key in sorted(set(ref_extra) | set(other_extra)):
                a, b = ref_extra.get(key), other_extra.get(key)
                if a != b:
                    rep.add(
                        "Paridad de hermanos de regimen",
                        f"{fam}/{dist}",
                        f"{ref.name} y {other.name} difieren en {key} "
                        f"({a!r} vs {b!r}); entre regimenes solo deberian "
                        f"cambiar los betas y las anisotropias",
                    )


def check_distribution_twins(cases: list[Case], rep: Report) -> None:
    """Mismo regimen fisico, distinta distribucion -> mismos beta/anisotropia."""
    groups: dict[tuple, list[Case]] = defaultdict(list)
    for c in cases:
        fam, reg = c.family(), c.regime()
        if fam and reg:
            box = c.defines.get("PSC_DOMAIN_DI", "20.0")
            groups[(c.setup, fam, reg, box)].append(c)
        elif c.is_isotropic_control:
            groups[(c.setup, "isotropic", "control", "-")].append(c)

    for (setup, fam, reg, box), members in sorted(groups.items()):
        dists = {c.distribution_key() for c in members}
        if len(members) < 2 or len(dists) < 2:
            continue
        ref = members[0]
        for other in members[1:]:
            if ref.distribution_key() == other.distribution_key():
                continue
            if ref.regime_signature() != other.regime_signature():
                diffs = [
                    f"{k}: {ref.defines.get(k)} vs {other.defines.get(k)}"
                    for k in REGIME
                    if num(ref.defines.get(k, "")) != num(other.defines.get(k, ""))
                ]
                rep.add(
                    "Paridad de gemelos de distribucion",
                    f"{fam}/{reg}",
                    f"{ref.name} y {other.name} deberian tener identico "
                    f"regimen fisico para aislar el efecto de la "
                    f"distribucion, pero difieren en -> " + "; ".join(diffs),
                )


def check_box_twins(cases: list[Case], rep: Report) -> None:
    """bigbox40 debe ser identico a su base salvo PSC_DOMAIN_DI."""
    by_label = {c.label: c for c in cases}
    for c in cases:
        if "bigbox" not in c.label:
            continue
        base = by_label.get(c.base_label())
        if base is None:
            rep.add(
                "Paridad de gemelos de caja",
                c.name,
                f"no existe el caso base psc_{c.base_label()}.cxx contra el "
                f"cual comparar el estudio de tamano de caja",
            )
            continue
        domain = num(c.defines.get("PSC_DOMAIN_DI", ""))
        if domain is None:
            rep.add(
                "Paridad de gemelos de caja",
                c.name,
                "el nombre dice bigbox pero no define PSC_DOMAIN_DI",
            )
        a = {k: v for k, v in c.defines.items() if k not in IDENTITY + BOX}
        b = {k: v for k, v in base.defines.items() if k not in IDENTITY + BOX}
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                rep.add(
                    "Paridad de gemelos de caja",
                    c.name,
                    f"difiere de {base.name} en {key} "
                    f"({a.get(key)!r} vs {b.get(key)!r}); la unica diferencia "
                    f"permitida es PSC_DOMAIN_DI",
                )


def check_harris_invariants(cases: list[Case], rep: Report) -> None:
    """Los casos de reconexion solo pueden diferir en kappa/beta/anisotropia."""
    recon = [c for c in cases if c.setup == "B"]
    if len(recon) < 2:
        return
    ref = recon[0]
    ref_extra = ref.non_regime_defines()
    for other in recon[1:]:
        other_extra = other.non_regime_defines()
        for key in sorted(set(ref_extra) | set(other_extra)):
            a, b = ref_extra.get(key), other_extra.get(key)
            if a != b:
                rep.add(
                    "Invariantes de la hoja de Harris",
                    "reconexion",
                    f"{ref.name} y {other.name} difieren en {key} "
                    f"({a!r} vs {b!r}); los ocho casos de reconexion "
                    f"comparten la misma hoja y solo pueden cambiar en "
                    f"kappa/beta/anisotropia",
                )


def check_cross_anchor(cases: list[Case], rep: Report) -> None:
    """Cada caso de reconexion anisotropo ancla en su caso de anisotropia pura."""
    by_label = {c.label: c for c in cases}
    for c in cases:
        anchor_label = c.anchor_label()
        if not anchor_label:
            continue
        anchor = by_label.get(anchor_label)
        fallback = False
        if anchor is None:
            # el gemelo de distribucion comparte los betas, sirve de ancla
            alt = re.sub(r"_bikappa\d+_", "_bimaxwellian_", anchor_label)
            anchor = by_label.get(alt)
            fallback = anchor is not None
        if anchor is None:
            rep.add(
                "Ancla cruzada anisotropia-reconexion",
                c.name,
                f"no existe psc_{anchor_label}.cxx, el caso de anisotropia "
                f"uniforme contra el cual este caso de reconexion tiene que "
                f"estar anclado",
            )
            continue
        diffs = [
            f"{k}: {c.defines.get(k)} vs {anchor.defines.get(k)}"
            for k in REGIME
            if num(c.defines.get(k, "")) != num(anchor.defines.get(k, ""))
        ]
        if diffs:
            via = " (via su gemelo bi-Maxwelliano)" if fallback else ""
            rep.add(
                "Ancla cruzada anisotropia-reconexion",
                c.name,
                f"deberia llevar exactamente el mismo plasma que "
                f"{anchor.name}{via} para poder decir 'este plasma, ahora "
                f"sobre una hoja de corriente', pero difiere en -> "
                + "; ".join(diffs),
            )


def check_isotropic_controls(cases: list[Case], rep: Report) -> None:
    controls = [c for c in cases if c.is_isotropic_control]
    for c in controls:
        for key in ("PSC_TI_PERP_OVER_TI_PAR", "PSC_TE_PERP_OVER_TE_PAR"):
            v = num(c.defines.get(key, ""))
            if v is not None and v != 1.0:
                rep.add(
                    "Controles isotropicos",
                    c.name,
                    f"{key}={c.defines[key]}: un control con anisotropia no "
                    f"es un control",
                )
    recon_controls = [c for c in controls if c.setup == "B"]
    if recon_controls:
        kinds = {c.uses_kappa for c in recon_controls}
        if len(kinds) < 2:
            falta = "bi-Kappa" if False in kinds else "bi-Maxwelliano"
            rep.add(
                "Controles isotropicos",
                "reconexion",
                f"falta el control isotropico {falta}; sin los dos, el efecto "
                f"de las colas suprathermicas queda confundido con el de la "
                f"anisotropia en los casos kappa",
            )


def check_legacy_reconnection(src: Path, cases: list[Case], rep: Report) -> None:
    """Archivos de reconexion monoliticos que todavia no estan en el esquema."""
    migrated = {c.name for c in cases if c.setup == "B"}
    for path in sorted(src.glob("psc_reconnection*.cxx")):
        if path.stem in migrated:
            continue
        n = len(path.read_text(errors="replace").splitlines())
        rep.add(
            "Reconexion pendiente de migrar",
            path.stem,
            f"{n} lineas monoliticas: todavia no usa {RECON_HEADER}, asi que "
            f"queda fuera de los chequeos de paridad de la matriz",
        )


def check_cmake(cases: list[Case], src: Path, rep: Report) -> None:
    cmake = src / "CMakeLists.txt"
    if not cmake.exists():
        return
    text = cmake.read_text(errors="replace")
    registered = set(re.findall(r"add_psc_executable\(\s*(\w+)", text))
    for c in cases:
        if c.name not in registered:
            rep.add(
                "Registro en CMakeLists",
                c.name,
                "el caso existe pero no tiene add_psc_executable(); no se "
                "compila y va a quedar fuera de la matriz",
            )


def read_header_defaults(src: Path) -> dict:
    header = src / HEADER
    defaults: dict[str, str] = {}
    if not header.exists():
        return defaults
    for line in header.read_text(errors="replace").splitlines():
        m = DEFINE_RE.match(line)
        if m and m.group(1).startswith("PSC_"):
            defaults.setdefault(m.group(1), m.group(2).strip())
    return defaults


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    src = Path(argv[1]).expanduser().resolve()
    if not src.is_dir():
        print(f"No es un directorio: {src}")
        return 2

    header_defaults = read_header_defaults(src)
    cases = []
    for path in sorted(src.glob(CASE_GLOB)):
        case = Case(path)
        if case.is_case:
            cases.append(case)

    rep = Report()
    check_legacy_reconnection(src, cases, rep)

    if not cases:
        print(f"No se encontraron casos que incluyan {HEADER} o "
              f"{RECON_HEADER} en {src}")
        return rep.emit() or 2

    for case in cases:
        check_structure(case, rep)
        check_identity(case, rep)
        check_comment(case, rep, header_defaults)
    check_regime_siblings(cases, rep)
    check_distribution_twins(cases, rep)
    check_box_twins(cases, rep)
    check_harris_invariants(cases, rep)
    check_cross_anchor(cases, rep)
    check_isotropic_controls(cases, rep)
    check_cmake(cases, src, rep)

    n_a = sum(1 for c in cases if c.setup == "A")
    n_b = len(cases) - n_a
    print(f"Casos analizados: {n_a} de anisotropia uniforme, "
          f"{n_b} de reconexion")
    return rep.emit()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
