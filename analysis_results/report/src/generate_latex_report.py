#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


# This script is invoked from the analysis_results/ root (so that image paths
# embedded in the .tex stay root-relative, matching how pdflatex resolves
# them when compiled with `-output-directory=report/build` from that same
# root). Only the .tex/.md source and the generated .tex output live under
# report/src/ — the case directories with the actual PNG/CSV data stay
# untouched at the root.
ROOT = Path(".")
SRC = ROOT / "report" / "src"
MD_FILE = SRC / "ANALISIS_ESTRUCTURA.md"
OUT_TEX = SRC / "analysis_results_report.tex"


UNICODE_REPLACEMENTS = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "č": "c",
    "Č": "C",
    "ñ": "n",
    "Ñ": "N",
    "ü": "u",
    "Ü": "U",
    "¿": "?",
    "¡": "!",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
    "│": "|",
    "├": "+",
    "└": "+",
    "┌": "+",
    "┐": "+",
    "┴": "+",
    "┬": "+",
    "─": "-",
    "▼": "v",
    "←": "<-",
    "→": "->",
    "⇒": "=>",
    "×": "x",
    "≈": "approx.",
    "≃": "approx.",
    "≤": "<=",
    "≥": ">=",
    "∞": "infty",
    "√": "sqrt",
    "∥": "parallel",
    "⊥": "perp",
    "∇": "nabla",
    "⟨": "<",
    "⟩": ">",
    "β": "beta",
    "γ": "gamma",
    "κ": "kappa",
    "ω": "omega",
    "Ω": "Omega",
    "δ": "delta",
    "Δ": "Delta",
    "π": "pi",
    "₀": "_0",
    "ᵢ": "_i",
    "ₑ": "_e",
    "ₚ": "_p",
    "⁻": "-",
    "¹": "^1",
    "·": "*",
    "∈": "in",
    "½": "1/2",
    "²": "^2",
    "³": "^3",
}


LATEX_ESCAPES = {
    "\\": "backslash",
    "&": r"\&",
    "%": r"\%",
    "$": "",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"$\sim$",
    "^": r"\^{}",
}


def normalize_unicode(text: str) -> str:
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def escape_latex(text: str) -> str:
    text = normalize_unicode(text)
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def tex_path(path: Path) -> str:
    return path.as_posix().replace("\\", "/")


def caption_from_path(path: Path, case_name: str | None = None) -> str:
    stem = path.stem
    if case_name and stem.startswith(case_name + "_"):
        stem = stem[len(case_name) + 1 :]
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\bstep\s*(\d+)\b", r"paso \1", stem)
    return escape_latex(stem)


def paragraph_to_tex(text: str) -> str:
    # Preserve inline `code` verbatim and $...$ math as real LaTeX math
    # (the source markdown already writes valid LaTeX inside $...$), only
    # escaping/translating plain prose in between.
    parts = re.split(r"(`[^`]+`|\$[^$]+\$)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{\detokenize{" + normalize_unicode(part[1:-1]) + "}}")
        elif part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(normalize_unicode(part))
        else:
            out.append(escape_latex(part))
    return "".join(out)


def markdown_to_latex(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    in_code = False
    in_itemize = False
    in_enum = False
    in_table = False

    def close_lists() -> None:
        nonlocal in_itemize, in_enum
        if in_itemize:
            out.append(r"\end{itemize}")
            in_itemize = False
        if in_enum:
            out.append(r"\end{enumerate}")
            in_enum = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            out.append(r"\end{verbatim}")
            out.append(r"\endgroup")
            in_table = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            close_table()
            close_lists()
            if not in_code:
                out.append(r"\begingroup\footnotesize")
                out.append(r"\begin{verbatim}")
                in_code = True
            else:
                out.append(r"\end{verbatim}")
                out.append(r"\endgroup")
                in_code = False
            continue

        if in_code:
            out.append(normalize_unicode(line))
            continue

        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        if is_table_line:
            close_lists()
            if not in_table:
                out.append(r"\begingroup\footnotesize")
                out.append(r"\begin{verbatim}")
                in_table = True
            out.append(normalize_unicode(line))
            continue
        close_table()

        if not stripped:
            close_lists()
            out.append("")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            title = escape_latex(heading.group(2).replace("`", ""))
            if level == 1:
                out.append(r"\subsection{" + title + "}")
            elif level == 2:
                out.append(r"\subsubsection{" + title + "}")
            else:
                out.append(r"\paragraph{" + title + "}")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            if not in_itemize:
                close_lists()
                out.append(r"\begin{itemize}")
                in_itemize = True
            out.append(r"\item " + paragraph_to_tex(bullet.group(1)))
            continue

        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            if not in_enum:
                close_lists()
                out.append(r"\begin{enumerate}")
                in_enum = True
            out.append(r"\item " + paragraph_to_tex(numbered.group(1)))
            continue

        if stripped.startswith(">"):
            close_lists()
            quote = stripped.lstrip(">").strip()
            out.append(r"\begin{quote}" + paragraph_to_tex(quote) + r"\end{quote}")
            continue

        close_lists()
        out.append(paragraph_to_tex(line))

    close_table()
    close_lists()
    if in_code:
        out.append(r"\end{verbatim}")
        out.append(r"\endgroup")
    return "\n".join(out)


def step_number(path: Path) -> int | None:
    match = re.search(r"step[_-]?(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def quantile_pick(paths: list[Path], max_items: int = 2) -> list[Path]:
    unique = sorted(set(paths), key=lambda p: (step_number(p) is None, step_number(p) or -1, p.name))
    if len(unique) <= max_items:
        return unique
    if max_items == 1:
        return [unique[-1]]
    idxs = [round(i * (len(unique) - 1) / (max_items - 1)) for i in range(max_items)]
    return [unique[i] for i in sorted(set(idxs))]


def pngs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"))


# ---------------------------------------------------------------------------
# Per-section figure selectors. Each receives the case directory (not a
# sub-folder) so that a logical section (e.g. VDF evolution) can pull figures
# that live in more than one physical output folder.
# ---------------------------------------------------------------------------

VDF_PREFIXES = (
    "vdf_2d_ion",
    "vdf_2d_electron",
    "vdf_3d_ion",
    "vdf_3d_electron",
)
VDF_TIME_SERIES_NAMES = (
    "vdf_1d_parallel_evolution",
    "vdf_1d_perp_evolution",
    "distribution_evolution_electrons_parallel",
    "distribution_evolution_electrons_perp",
    "distribution_evolution_ions_parallel",
    "distribution_evolution_ions_perp",
)
KAPPA_NAMES_EXACT = ("kappa_fit_vs_time", "suprathermal_fraction_vs_time")
KAPPA_PREFIX_STEP = "kappa_vs_maxwellian"


def select_anisotropy(case_dir: Path) -> list[Path]:
    return pngs(case_dir / "01_anisotropy")


def select_fields(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "02_fields")
    if len(images) <= 30:
        return images
    selected: list[Path] = []
    for prefix in ("Bmag", "Bx", "By", "Bz"):
        matches = [p for p in images if p.name.startswith(prefix)]
        selected.extend(quantile_pick(matches, 3))
    return sorted(set(selected))


def select_particles(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "03_particles")
    return [p for p in images if not any(p.stem.endswith(name) for name in VDF_TIME_SERIES_NAMES)]


def select_vdf(case_dir: Path) -> list[Path]:
    particles = pngs(case_dir / "03_particles")
    time_series = sorted(
        p for p in particles if any(p.stem.endswith(name) for name in VDF_TIME_SERIES_NAMES)
    )
    snapshots = pngs(case_dir / "09_physical_diagnostics")
    selected = list(time_series)
    for prefix in VDF_PREFIXES:
        matches = [p for p in snapshots if p.name.startswith(prefix)]
        selected.extend(quantile_pick(matches, 4))
    return sorted(set(selected), key=lambda p: (p.parent.name, step_number(p) is None, step_number(p) or -1, p.name))


def select_kappa(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "09_physical_diagnostics")
    selected = [p for p in images if p.stem in KAPPA_NAMES_EXACT]
    matches = [p for p in images if p.name.startswith(KAPPA_PREFIX_STEP)]
    selected.extend(quantile_pick(matches, 4))
    return sorted(set(selected), key=lambda p: (p.stem not in KAPPA_NAMES_EXACT, step_number(p) or -1, p.name))


def select_spectra(case_dir: Path) -> list[Path]:
    return pngs(case_dir / "04_spectra")


def select_diamagnetic(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "05_diamagnetic")
    if len(images) <= 20:
        return images
    selected: list[Path] = []
    for prefix in ("jdia_total", "jdia_ions", "jdia_electrons"):
        matches = [p for p in images if p.name.startswith(prefix)]
        selected.extend(quantile_pick(matches, 3))
    return sorted(set(selected))


def select_heat_flux(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "06_heat_flux")
    selected = [p for p in images if "vs_time" in p.name]
    for prefix in ("heatflux_anisotropy", "heatflux_beta_parallel", "heatflux_q_parallel", "heatflux_q_perp"):
        matches = [p for p in images if p.name.startswith(prefix) and "step" in p.name]
        selected.extend(quantile_pick(matches, 2))
    return sorted(set(selected))


def select_validation(case_dir: Path) -> list[Path]:
    return pngs(case_dir / "08_validation")


def select_physical(case_dir: Path) -> list[Path]:
    images = pngs(case_dir / "09_physical_diagnostics")
    excluded_exact = set(KAPPA_NAMES_EXACT)
    excluded_prefixes = VDF_PREFIXES + (KAPPA_PREFIX_STEP,)
    images = [
        p
        for p in images
        if p.stem not in excluded_exact and not p.name.startswith(excluded_prefixes)
    ]
    selected = [p for p in images if "_step_" not in p.name]
    prefixes = (
        "A_i_map",
        "T_parallel_map",
        "T_perp_map",
        "deltaB_map",
        "mirror_holes_map",
        "J_dia_total_map",
        "q_parallel_map",
        "q_perp_map",
        "magnetic_spectrum",
    )
    for prefix in prefixes:
        matches = [p for p in images if p.name.startswith(prefix)]
        selected.extend(quantile_pick(matches, 2))
    return sorted(set(selected))


SECTION_SELECTORS = {
    "01_anisotropy": select_anisotropy,
    "vdf": select_vdf,
    "kappa": select_kappa,
    "02_fields": select_fields,
    "03_particles": select_particles,
    "04_spectra": select_spectra,
    "05_diamagnetic": select_diamagnetic,
    "06_heat_flux": select_heat_flux,
    "08_validation": select_validation,
    "09_physical_diagnostics": select_physical,
}


SECTION_LABELS = {
    "01_anisotropy": "Anisotropia de temperatura",
    "vdf": "Evolucion de la funcion de distribucion de velocidades (VDF)",
    "kappa": "Evolucion del parametro kappa",
    "02_fields": "Campos magneticos",
    "03_particles": "Particulas",
    "04_spectra": "Espectros",
    "05_diamagnetic": "Corriente diamagnetica",
    "06_heat_flux": "Flujo de calor",
    "08_validation": "Validacion",
    "09_physical_diagnostics": "Diagnosticos fisicos integrados",
}

SECTION_INTROS = {
    "vdf": (
        "Se muestra la evolucion temporal de la funcion de distribucion de "
        "velocidades (VDF) por especie. Los paneles 1D ($f(v_\\parallel)$ y "
        "$f(v_\\perp)$ superpuestos en varios pasos) permiten ver el "
        "desarrollo de colas supratermales y el achatamiento/relajacion de "
        "la anisotropia inicial; los mapas 2D $f(v_\\perp,v_\\parallel)$ y su "
        "version 3D muestran la deformacion completa de la VDF en pasos "
        "representativos (inicial, intermedio, saturacion). Ver "
        "\\S\\ref{sec:teoria-vdf} para el marco teorico bi-Maxwelliano/bi-kappa."
    ),
    "kappa": (
        "Ajuste de la VDF simulada a una distribucion kappa en funcion del "
        "tiempo. \\texttt{kappa\\_fit\\_vs\\_time} da la evolucion de "
        "$\\kappa$ (a menor $\\kappa$, colas mas pesadas / mas energia en "
        "particulas supratermales); \\texttt{suprathermal\\_fraction\\_vs\\_time} "
        "cuantifica la fraccion de particulas por fuera del nucleo "
        "Maxwelliano ajustado; los paneles \\texttt{kappa\\_vs\\_maxwellian} "
        "comparan directamente, en pasos puntuales, el mejor ajuste kappa "
        "contra el mejor ajuste Maxwelliano sobre la VDF medida. Ver "
        "\\S\\ref{sec:teoria-kappa}."
    ),
}


def case_dirs() -> list[Path]:
    dirs = []
    for manifest in sorted(ROOT.glob("*/*_analysis_manifest.json")):
        dirs.append(manifest.parent)
    return dirs


def manifest_table(case_dir: Path) -> str:
    manifest_path = next(case_dir.glob("*_analysis_manifest.json"), None)
    if manifest_path is None:
        return ""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    detected = data.get("detected", {})
    physics = data.get("physics", {})
    rows = [
        ("Etiqueta", data.get("label", case_dir.name)),
        ("Inestabilidad", data.get("instability", "")),
        ("Especie impulsora", data.get("driven_species", "")),
        ("Campos detectados", detected.get("field_files", "")),
        ("Momentos detectados", detected.get("moment_files", "")),
        ("Particulas detectadas", detected.get("particle_files", "")),
        ("Primer paso emparejado", detected.get("first_paired_step", "")),
        ("Ultimo paso emparejado", detected.get("last_paired_step", "")),
        ("Grid", " x ".join(map(str, physics.get("grid", detected.get("grid_from_hdf5", []))))),
        ("Dominio [d_i]", physics.get("domain_di", "")),
        ("B0", physics.get("B0", "")),
    ]
    out = [
        r"\begin{center}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}l l@{}}",
        r"\toprule",
    ]
    for key, value in rows:
        out.append(r"\textbf{" + escape_latex(str(key)) + "} & " + escape_latex(str(value)) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(out)


def figure_grid(images: list[Path], case_name: str) -> str:
    out: list[str] = []
    for i in range(0, len(images), 2):
        pair = images[i : i + 2]
        out.append(r"\begin{center}")
        for img in pair:
            out.append(r"\begin{minipage}{0.48\linewidth}")
            out.append(r"\centering")
            out.append(r"\fbox{\includegraphics[width=\linewidth,height=0.34\textheight,keepaspectratio]{" + tex_path(img) + "}}")
            out.append(r"\par{\footnotesize " + caption_from_path(img, case_name) + r"}")
            out.append(r"\par{\scriptsize\ttfamily\color{gray}\detokenize{" + tex_path(img) + r"}}")
            out.append(r"\end{minipage}")
            if img != pair[-1]:
                out.append(r"\hfill")
        out.append(r"\end{center}")
    return "\n".join(out)


def build_results() -> str:
    out = [r"\section{Resultados por simulacion}", r"\label{sec:resultados}"]
    total_selected = 0
    total_available = len(list(ROOT.glob("*/*/*.png")))
    out.append(
        "Se incluyen figuras representativas por simulacion y diagnostico. "
        "Los directorios con series densas de snapshots se reducen a pasos "
        "iniciales, intermedios y finales para mantener el PDF manejable; "
        "ninguna figura se elimina del disco, solo se selecciona un "
        "subconjunto representativo para su inclusion en este documento."
    )
    out.append("")
    out.append("Figuras PNG disponibles en \\texttt{analysis\\_results/}: " + str(total_available) + ".")
    for case_dir in case_dirs():
        out.append(r"\clearpage")
        out.append(r"\subsection{" + escape_latex(case_dir.name) + "}")
        out.append(manifest_table(case_dir))
        for section, selector in SECTION_SELECTORS.items():
            images = selector(case_dir)
            if not images:
                continue
            total_selected += len(images)
            out.append(r"\subsubsection{" + SECTION_LABELS[section] + "}")
            if section in SECTION_INTROS:
                out.append(SECTION_INTROS[section])
            out.append("Figuras incluidas: " + str(len(images)) + ".")
            out.append(figure_grid(images, case_dir.name))
            out.append(r"\clearpage")
    out.insert(4, "Figuras seleccionadas para este informe: " + str(total_selected) + ".")
    return "\n".join(out)


def theory_section() -> str:
    return r"""
\section{Marco teorico}
\label{sec:teoria}

\subsection{Anisotropia de temperatura y funciones de distribucion}
\label{sec:teoria-vdf}

En un plasma sin colisiones magnetizado ($\mathbf{B}_0 \parallel \hat{z}$), la
funcion de distribucion de velocidades (VDF) de cada especie $s$ no tiene por
que ser isotropa: al no haber colisiones que igualen $T_\parallel$ y
$T_\perp$ en la escala de tiempo de la simulacion, procesos como la
compresion/expansion adiabatica del plasma, la conveccion desde regiones con
distinta intensidad de campo, o el forzamiento inicial impuesto en estas
corridas, dejan una anisotropia de temperatura

\begin{equation}
A_s \equiv \frac{T_{\perp s}}{T_{\parallel s}}, \qquad
R_s \equiv \frac{1}{A_s} = \frac{T_{\parallel s}}{T_{\perp s}} .
\end{equation}

El caso de referencia es la VDF \textbf{bi-Maxwelliana}, producto de dos
Gaussianas independientes en las direcciones paralela y perpendicular a
$\mathbf{B}_0$:

\begin{equation}
f_{M}(v_\parallel, v_\perp) =
\frac{n_s}{\pi^{3/2}\, v_{th\parallel}\, v_{th\perp}^2}\,
\exp\!\left(-\frac{v_\parallel^2}{v_{th\parallel}^2}
             -\frac{v_\perp^2}{v_{th\perp}^2}\right),
\qquad v_{th\alpha} = \sqrt{2T_\alpha/m_s}.
\end{equation}

Las colas de $f_M$ decaen como una Gaussiana, de forma que la poblacion de
particulas rapidas (supratermales) es, por construccion, despreciable. Sin
embargo, tanto observaciones in situ del viento solar y la magnetofunda como
la turbulencia cinetica desarrollada en simulaciones PIC muestran de forma
sistematica colas mas pobladas que las de una Maxwelliana. La familia de
distribuciones \textbf{bi-kappa} generaliza el caso Gaussiano permitiendo un
exceso de particulas supratermales controlado por un unico parametro
$\kappa$:

\begin{equation}
f_{\kappa}(v_\parallel, v_\perp) =
\frac{n_s}{\pi^{3/2}\,\kappa^{3/2}\,\theta_\parallel\,\theta_\perp^2\,
B\!\left(\kappa-\tfrac{1}{2},\tfrac{3}{2}\right)}
\left[1 + \frac{1}{\kappa}\left(\frac{v_\parallel^2}{\theta_\parallel^2}
+ \frac{v_\perp^2}{\theta_\perp^2}\right)\right]^{-(\kappa+1)},
\end{equation}

donde $\theta_\alpha$ es la velocidad termica caracteristica y $B(\cdot,\cdot)$
es la funcion Beta. Para $v \gg \theta$, $f_\kappa \sim v^{-2(\kappa+1)}$: una
cola de ley de potencia, en contraste con el decaimiento exponencial de
$f_M$. En el limite $\kappa \to \infty$ se recupera exactamente la
bi-Maxwelliana, de modo que $\kappa$ finito mide, de forma continua, cuanto
se aparta la VDF simulada de la Maxwelliana de referencia. Los casos
\texttt{bikappa} de esta tesis inicializan la VDF directamente con
$f_\kappa$ (t\'{\i}picamente $\kappa=3$, una cola muy pesada) para comparar su
evolucion no lineal contra el caso bi-Maxwelliano equivalente.

\subsection{Inestabilidades por anisotropia: mirror y firehose}
\label{sec:teoria-inestabilidades}

Con $\beta_{\parallel s} = 8\pi n_s T_{\parallel s}/B_0^2$, la teoria lineal
de plasmas sin colisiones predice dos familias de inestabilidades
electromagneticas de baja frecuencia cuando la anisotropia de temperatura
ionica supera un umbral que depende de $\beta_{\parallel i}$:

\begin{itemize}
\item \textbf{Inestabilidad mirror} ($A_i>1$, exceso de presion
perpendicular): crece cuando
\begin{equation}
\beta_{\perp i}\left(A_i - 1\right) \;\gtrsim\; 1 ,
\end{equation}
y satura formando pozos de campo magnetico (\emph{mirror holes}) que
compensan localmente el exceso de $p_\perp$. Es la inestabilidad dominante
en los casos \texttt{mirror\_*} de esta tesis.

\item \textbf{Inestabilidad firehose} ($A_i<1$, exceso de presion
paralela): crece cuando
\begin{equation}
\beta_{\parallel i}\left(1 - A_i\right) \;\gtrsim\; 2 ,
\end{equation}
y satura generando fluctuaciones magneticas cuasi-paralelas a $B_0$ que
redistribuyen momento entre las direcciones paralela y perpendicular. Es la
inestabilidad dominante en los casos \texttt{firehose\_*}.
\end{itemize}

Ambos umbrales son los que se dibujan como curvas de referencia en los
\emph{brazil plots} (seccion ``Anisotropia de temperatura'' de cada caso,
\S\ref{sec:resultados}): la trayectoria del sistema en el plano
$(\beta_{\parallel i}, A_i)$ empieza en la zona inestable (forzada por las
condiciones iniciales) y relaja hacia el umbral marginal a medida que la
inestabilidad satura, lo cual es precisamente el mecanismo de regulacion de
anisotropia que se busca caracterizar.

\subsection{Fase lineal: tasa de crecimiento}

En la fase lineal, la energia de las fluctuaciones magneticas crece de
forma exponencial, $\delta B_{\rm rms}(t) \propto \delta B_{\rm rms}(0)\,
e^{\gamma t}$, con $\gamma$ la tasa de crecimiento lineal del modo mas
inestable. El pipeline ajusta $\ln \delta B_{\rm rms}(t)$ en la ventana
donde ese crecimiento es aproximadamente lineal (filtrando por bondad de
ajuste $R^2$) para extraer $\gamma$ por caso (\texttt{growth\_rate\_fit.png},
\texttt{growth\_rate\_summary.csv}), y de forma independiente estima
$\gamma(k)$ por modo a partir del espectro de campos
(\texttt{growth\_rate\_vs\_k}, \texttt{growth\_rate\_map}, seccion
Espectros). Comparar $\gamma$ entre el caso bi-Maxwelliano y su contraparte
bi-kappa a igual $\beta_{\parallel i}$ y $A_i$ nominal es una de las
preguntas centrales de la tesis: una cola supratermal mas pesada
(\emph{i.e.} $\kappa$ menor) modifica la fraccion resonante de particulas y,
por tanto, puede acelerar o suavizar el crecimiento lineal respecto al caso
Maxwelliano.

\subsection{Relajacion cuasi-lineal, saturacion y regulacion de la
anisotropia}

Una vez saturada la inestabilidad, el sistema se autorregula: el scattering
por las fluctuaciones de campo generadas (\emph{pitch-angle scattering}
cuasi-lineal) reduce $A_i$ (mirror) o la incrementa (firehose) hasta acercar
la trayectoria $(\beta_{\parallel i}, A_i)$ al umbral marginal. Esto se
cuantifica con la evolucion temporal de la anisotropia
(\texttt{anisotropy\_ratio\_vs\_time}), la profundidad y area ocupada por
las estructuras mirror saturadas (\texttt{mirror\_depth\_area\_vs\_time}) y,
para los casos bi-kappa, con la evolucion del propio $\kappa$ ajustado
(\S\ref{sec:teoria-kappa}): si la relajacion popula aun mas las colas
supratermales, $\kappa$ ajustado deberia \emph{disminuir} con el tiempo
incluso en corridas que inician bi-Maxwellianas ($\kappa \to \infty$ en
$t=0$).

\subsection{De la VDF simulada al parametro kappa ajustado}
\label{sec:teoria-kappa}

Para cada paso con datos de particulas, el diagnostico integrado ajusta por
minimos cuadrados tanto una bi-Maxwelliana como una bi-kappa
(Ec.~3) a la VDF medida, y reporta el error de ajuste global y en la cola
por separado (\texttt{fit\_metrics.csv}). De ese ajuste se extraen dos
series temporales complementarias:

\begin{itemize}
\item $\kappa(t)$ (\texttt{kappa\_fit\_vs\_time.png}): el valor de $\kappa$
que mejor reproduce la VDF simulada en cada paso. Valores grandes
($\kappa \gtrsim 10$) indican una VDF practicamente Gaussiana; valores
pequenos ($\kappa \sim 2$--$5$) indican colas fuertemente supratermales.
\item Fraccion supratermal (\texttt{suprathermal\_fraction\_vs\_time.png}):
la fraccion de particulas cuya velocidad excede varias veces la velocidad
termica del nucleo ajustado, una medida mas directa (no parametrica) del
mismo efecto.
\end{itemize}

Los paneles \texttt{kappa\_vs\_maxwellian\_step\_<paso>.png} muestran, para
un paso puntual, la VDF medida junto con ambos ajustes superpuestos: la
separacion visible entre el ajuste Maxwelliano y los datos en la zona de
colas (mientras que el ajuste kappa sigue a los datos) es la evidencia
directa de que la poblacion supratermal generada durante la inestabilidad
no es capturada por una descripcion Gaussiana.
"""


def preamble() -> str:
    # Note: this environment's TeX install has no titlesec/fancyhdr/enumitem
    # and no T1 (ec*) font metrics (no metafont binary to build them), so we
    # stick to default OT1 Computer Modern and implement colored section
    # titles with the native \@startsection kernel hook instead of titlesec.
    return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage[dvipsnames]{xcolor}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\setlength{\paperwidth}{8.5in}
\setlength{\paperheight}{11in}
\setlength{\textwidth}{6.7in}
\setlength{\textheight}{9.0in}
\setlength{\oddsidemargin}{-0.1in}
\setlength{\evensidemargin}{-0.1in}
\setlength{\topmargin}{-0.45in}
\setlength{\headheight}{14pt}
\setlength{\headsep}{18pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\labelitemi}{$\bullet$}
\renewcommand{\labelitemii}{$\circ$}
\sloppy
\emergencystretch=2em

\definecolor{plasmablue}{HTML}{1B3A6B}
\definecolor{plasmaorange}{HTML}{C1521E}

\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{0pt}%
  {18pt plus 4pt minus 2pt}{10pt}%
  {\color{plasmablue}\Large\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}%
  {14pt plus 3pt minus 2pt}{6pt}%
  {\color{plasmablue}\large\bfseries}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{0pt}%
  {12pt plus 2pt minus 2pt}{4pt}%
  {\color{plasmaorange}\normalsize\bfseries}}
\makeatother

\pagestyle{headings}
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

\title{\color{plasmablue}\textbf{Informe de resultados de simulaciones PSC}\\[4pt]
\large Anisotropia de temperatura y distribuciones supratermales
(bi-Maxwelliana vs.\ bi-kappa) en las inestabilidades mirror y firehose}
\author{Carlos Martinez \\ \small Maestria en Fisica, Universidad Nacional de Colombia}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\noindent
Este informe recopila los diagnosticos fisicos calculados sobre las
simulaciones PIC (codigo PSC) que sustentan la tesis: evolucion de la
anisotropia de temperatura ionica, tasas de crecimiento lineal, estructuras
magneticas de saturacion y, en particular, la evolucion de la funcion de
distribucion de velocidades (VDF) y su ajuste a una distribucion kappa, para
comparar corridas inicializadas con VDF bi-Maxwelliana frente a corridas
bi-kappa en los regimenes mirror y firehose. La \S\ref{sec:teoria} resume
el marco teorico necesario para interpretar las figuras; la
\S\ref{sec:resultados} presenta los resultados por simulacion; el apendice
documenta la pipeline de post-procesamiento usada para generarlos.
\end{abstract}

\tableofcontents
\clearpage
"""


def main() -> None:
    md_latex = markdown_to_latex(MD_FILE.read_text(encoding="utf-8"))
    report = preamble()
    report += theory_section()
    report += "\n\\clearpage\n"
    report += build_results()
    report += "\n\\clearpage\n"
    report += r"\appendix" + "\n"
    report += r"\section{Documentacion tecnica de la pipeline de analisis}" + "\n"
    report += (
        "Documentacion de referencia de los scripts, formatos de archivo y "
        "convenciones usados para producir los diagnosticos de la "
        "\\S\\ref{sec:resultados}. No es necesaria para interpretar las "
        "figuras fisicas, mantenida aqui como apendice reproducible.\n\n"
    )
    report += md_latex
    report += "\n\\end{document}\n"
    OUT_TEX.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
