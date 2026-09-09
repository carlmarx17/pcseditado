#!/usr/bin/env python3
"""
linear_theory.py — Relación de dispersión cinética lineal (bi-Maxwelliana / bi-kappa)
=====================================================================================
`polarization_dispersion.py` acepta `--theory-csv` y rellena las columnas
`gamma_theory` y `relative_difference_pct`, pero nunca existió un generador de
ese archivo: en las cuatro corridas analizadas esas columnas salen NaN, de modo
que no hay comparación PIC contra teoría cinética. Este módulo produce el CSV.

Alcance
-------
Modos de **propagación paralela** (k ∥ B0): firehose paralelo y ciclotrónico
iónico (EMIC), en polarización derecha (R) e izquierda (L). El modo mirror es
oblicuo y aperiódico y NO se obtiene de esta relación; pedir `--instability
mirror` es un error explícito en vez de devolver un número sin sentido.

Relación resuelta (unidades normalizadas: ω y γ en Ω_ci, k en 1/d_i):

    c²k²/ω² = 1 + Σ_s (ω_ps²/ω²) { (A_s − 1)
                                   + [A_s(ω ∓ Ω_s) ± Ω_s] / (k v_∥s) · Z(ζ_s^∓) }

con ζ_s^∓ = (ω ∓ Ω_s)/(k v_∥s), v_∥s = √(2T_∥s/m_s), Ω_s con signo, y el signo
superior para el modo R.

Validación
----------
La forma de arriba no se da por buena: `--self-test` la comprueba contra tres
límites con respuesta conocida.

  1. Z(ζ) de Fried–Conte cumple Z'(ζ) = −2(1 + ζZ(ζ)).
  2. Z_κ(ζ,κ) → Z(ζ) cuando κ → ∞.
  3. El umbral del firehose paralelo sale en β_∥ − β_⊥ = 2, que es el
     resultado analítico de fluido en el límite k → 0.

El test 3 es el que discrimina de verdad: una relación de dispersión con el
numerador mal puesto reproduce los límites frío e isótropo pero falla el umbral.

Función de dispersión kappa
---------------------------
Z_κ se evalúa integrando numéricamente su definición

    Z_κ(ζ) = (1/√(πκ)) · Γ(κ)/Γ(κ−1/2) · ∫ (1 + x²/κ)^(−κ) / (x − ζ) dx

que es válida sin continuación analítica mientras Im(ζ) > 0 — justamente el
régimen de los modos inestables, que es lo que buscamos. Para modos amortiguados
(Im ζ < 0) haría falta continuar analíticamente y el resultado se marca NaN.

Uso típico:
    python linear_theory.py --self-test
    python linear_theory.py --case mirror_bikappa3_moderate --out theory.csv
    python linear_theory.py --beta-par 10 --anisotropy 0.1 --kappa 3 --out t.csv
    # y después:
    python polarization_dispersion.py --data-dir ... --theory-csv theory.csv
"""

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gammaln
from scipy.special import wofz


# ── Funciones de dispersión de plasma ────────────────────────────────────────

def Z(zeta: complex) -> complex:
    """Función de dispersión de plasma de Fried–Conte, Z(ζ) = i√π w(ζ)."""
    return 1j * math.sqrt(math.pi) * wofz(zeta)


def Z_prime(zeta: complex) -> complex:
    """Derivada, vía la identidad Z'(ζ) = −2(1 + ζ Z(ζ))."""
    return -2.0 * (1.0 + zeta * Z(zeta))


def Z_kappa(zeta: complex, kappa: float, limit: int = 200) -> complex:
    """Función de dispersión modificada para una distribución kappa.

    Integración numérica directa de la definición. Sólo válida para
    Im(ζ) > 0 (modos crecientes); fuera de ahí se devuelve NaN en vez de un
    número silenciosamente incorrecto.
    """
    if kappa is None or kappa == np.inf:
        return Z(zeta)
    if not np.isfinite(kappa) or kappa <= 1.5:
        raise ValueError("A finite-temperature bi-Kappa requires kappa > 1.5")
    if np.imag(zeta) <= 0:
        return complex(np.nan, np.nan)

    norm = math.exp(gammaln(kappa) - gammaln(kappa - 0.5)) / math.sqrt(math.pi * kappa)

    def integrand(x, part):
        val = np.exp(-kappa * np.log1p(x * x / kappa)) / (x - zeta)
        return val.real if part == 0 else val.imag

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            re, _ = quad(integrand, -np.inf, np.inf, args=(0,), limit=limit,
                         epsabs=1e-10, epsrel=1e-10)
            im, _ = quad(integrand, -np.inf, np.inf, args=(1,), limit=limit,
                         epsabs=1e-10, epsrel=1e-10)
        except IntegrationWarning:
            return complex(np.nan, np.nan)
    return norm * complex(re, im)


# ── Relación de dispersión paralela ──────────────────────────────────────────

class ParallelDispersion:
    """D(ω, k) para modos EM de propagación paralela en un plasma bi-especie.

    Frequencies are in Omega_ci, k in 1/di, velocities in vA. For
    exp[i(kz-omega*t)], 'plus' is Bx+i*By (denominator omega-Omega_s),
    'minus' is Bx-i*By (omega+Omega_s). Omega_s is signed. At positive
    frequency the plus channel contains the ion-cyclotron resonance.
    Both species use the same kappa, as in the current PSC case loader.
    """

    def __init__(self, beta_par_i: float, A_i: float,
                 beta_par_e: float, A_e: float,
                 mass_ratio: float, c_over_va: float,
                 kappa: float | None = None):
        values = (beta_par_i, A_i, beta_par_e, A_e, mass_ratio, c_over_va)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Betas, anisotropies, mass ratio and c/vA must be positive")
        if kappa == np.inf:
            kappa = None
        if kappa is not None and (not np.isfinite(kappa) or kappa <= 1.5):
            raise ValueError("A finite-temperature bi-Kappa requires kappa > 1.5")
        self.beta_par_i = beta_par_i
        self.A_i = A_i
        self.beta_par_e = beta_par_e
        self.A_e = A_e
        self.mass_ratio = mass_ratio
        self.c_over_va = c_over_va
        self.kappa = kappa

        # Fixed second-moment temperature, matching createKappaMultivariate.
        # Lazar et al. (2011), Eqs. (5), (11), (12), bi-Kappa (not product).
        scale = 1.0 if kappa is None else math.sqrt((kappa - 1.5) / kappa)
        self.v_i = math.sqrt(beta_par_i) * scale
        self.v_e = math.sqrt(beta_par_e * mass_ratio) * scale
        # Giro-frecuencias con signo, en unidades de Ω_ci.
        self.omega_c_i = 1.0
        self.omega_c_e = -mass_ratio

    def _zfun(self, zeta: complex) -> complex:
        if self.kappa is None:
            return Z(zeta)
        return Z_kappa(zeta, self.kappa)

    def _species_term(self, omega: complex, k: float, sign: float,
                      A: float, v_th: float, omega_c: float) -> complex:
        """Contribución de una especie al lado derecho, sin el peso ω_ps²/ω²."""
        zeta = (omega - sign * omega_c) / (k * v_th)
        zfun = self._zfun(zeta)
        numerator = A * (omega - sign * omega_c) + sign * omega_c
        return (A - 1.0) + numerator / (k * v_th) * zfun

    def __call__(self, omega: complex, k: float,
                 polarization: str = "plus") -> complex:
        """Residuo D(ω,k); cero en una raíz del modo."""
        if polarization not in ("plus", "minus"):
            raise ValueError("polarization must be plus or minus")
        if not np.isfinite(k) or k <= 0:
            raise ValueError("This solver requires k_parallel > 0")
        sign = 1.0 if polarization == "plus" else -1.0
        ion = self._species_term(omega, k, sign, self.A_i, self.v_i,
                                 self.omega_c_i)
        ele = self._species_term(omega, k, sign, self.A_e, self.v_e,
                                 self.omega_c_e)
        # Dividido por (ω_pi/Ω_ci)²/ω² = (c/v_A)²/ω²: queda k² a la izquierda.
        displacement = omega**2 / self.c_over_va**2
        return k * k - (displacement + ion + self.mass_ratio * ele)

    # ── Búsqueda de raíces ──────────────────────────────────────────────────

    def solve(self, k: float, polarization: str = "plus",
              omega_guess: complex | None = None,
              tol: float = 1e-10, max_iter: int = 120,
              residual_tol: float = 1e-8) -> complex:
        """Complex secant iteration; return NaN unless the residual converges.

        residual_tol bounds |D|/k^2. A small step alone is not convergence,
        especially when a Kappa iteration approaches the Im(omega)=0 boundary.
        """
        if k <= 0 or not np.isfinite(k) or polarization not in ("plus", "minus"):
            raise ValueError("Require positive k and a plus/minus polarization")
        if tol <= 0 or residual_tol <= 0 or max_iter < 1:
            raise ValueError("Solver tolerances and max_iter must be positive")
        self.last_solve = {"converged": False, "residual": float("nan"), "iterations": 0}
        if omega_guess is None:
            omega_guess = complex(0.3 * k, 0.05)
        w0 = omega_guess
        w1 = omega_guess * 1.02 + 1e-4j
        try:
            f0 = self(w0, k, polarization)
            f1 = self(w1, k, polarization)
        except (ValueError, ZeroDivisionError):
            return complex(np.nan, np.nan)

        for iteration in range(max_iter):
            self.last_solve["iterations"] = iteration + 1
            if not (np.isfinite(f0) and np.isfinite(f1)):
                return complex(np.nan, np.nan)
            if abs(f1 - f0) < 1e-300:
                break
            w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
            if not np.isfinite(w2):
                return complex(np.nan, np.nan)
            # La integración de Z_kappa exige Im(ω) > 0; mantener la búsqueda
            # en el semiplano superior evita evaluar donde no es válida.
            if self.kappa is not None and np.imag(w2) <= 0:
                w2 = complex(np.real(w2), max(1e-6, 0.5 * np.imag(w1)))
            f2 = self(w2, k, polarization)
            residual = float(abs(f2) / (k * k))
            self.last_solve["residual"] = residual
            if (np.isfinite(residual) and residual <= residual_tol
                    and abs(w2 - w1) < tol * max(1.0, abs(w2))):
                self.last_solve["converged"] = True
                return w2
            w0, f0 = w1, f1
            w1 = w2
            try:
                f1 = self(w1, k, polarization)
            except (ValueError, ZeroDivisionError):
                return complex(np.nan, np.nan)
        return complex(np.nan, np.nan)

    def scan(self, k_values: np.ndarray, polarization: str = "plus",
             omega_guess: complex | None = None) -> list[dict]:
        """Recorre k arrastrando la raíz anterior como semilla."""
        rows = []
        guess = omega_guess
        for k in k_values:
            root = self.solve(k, polarization, guess)
            if np.isfinite(root) and abs(np.imag(root)) < 50:
                guess = root
            rows.append({
                "kdi": float(k),
                "omega_r_over_Omegai": float(np.real(root)),
                "gamma_over_Omegai": float(np.imag(root)),
                "polarization": polarization,
                **self.last_solve,
            })
        return rows


# ── Tests de validación ──────────────────────────────────────────────────────

def self_test(verbose: bool = True) -> bool:
    """Comprueba la implementación contra tres límites de respuesta conocida."""
    ok = True

    # 1. Identidad de la derivada de Z.
    worst = 0.0
    for zeta in (0.5 + 0.3j, -1.2 + 0.8j, 2.0 + 0.1j, 0.1 + 2.0j):
        num = (Z(zeta + 1e-6) - Z(zeta - 1e-6)) / 2e-6
        worst = max(worst, abs(num - Z_prime(zeta)))
    passed = worst < 1e-6
    ok &= passed
    if verbose:
        print(f"[{'OK ' if passed else 'FALLA'}] Z'(z) = -2(1+zZ(z)); "
              f"error max = {worst:.2e}")

    # 2. Z_kappa -> Z cuando kappa -> infinito, y además al ritmo correcto.
    #    La corrección de una kappa respecto a la Maxwelliana es O(1/kappa), así
    #    que error*kappa debe tender a una constante. Comprobar sólo que el
    #    error sea "pequeño" no distingue una implementación correcta de una que
    #    converge por la razón equivocada.
    zeta = 0.7 + 0.4j
    kappas = (4.0, 10.0, 40.0)
    errs = [abs(Z_kappa(zeta, kappa) - Z(zeta)) for kappa in kappas]
    scaled = [e * kappa for e, kappa in zip(errs, kappas)]
    monotone = errs[0] > errs[1] > errs[2]
    rate_ok = max(scaled) / min(scaled) < 1.15      # constante al 15 %
    passed = monotone and rate_ok
    ok &= passed
    if verbose:
        print(f"[{'OK ' if passed else 'FALLA'}] Z_kappa -> Z como O(1/kappa): "
              f"error*kappa = " +
              ", ".join(f"{s:.4f} (k={kappa:.0f})"
                        for s, kappa in zip(scaled, kappas)))

    # 3. Umbral del firehose paralelo: inestable si beta_par - beta_perp > 2.
    #    Se busca dónde gamma cruza cero variando beta_par a A fijo.
    threshold = _firehose_threshold_numeric()
    passed = np.isfinite(threshold) and abs(threshold - 2.0) < 0.35
    ok &= passed
    if verbose:
        print(f"[{'OK ' if passed else 'FALLA'}] umbral firehose paralelo: "
              f"beta_par - beta_perp = {threshold:.3f} (analitico: 2.0)")

    return bool(ok)


def _firehose_threshold_numeric(A: float = 0.2, k: float = 0.10,
                                gamma_tol: float = 1e-4) -> float:
    """Valor de (beta_par - beta_perp) donde gamma cruza cero, a k pequeño.

    Se barre beta_par en rejilla arrastrando la raíz anterior como semilla, en
    vez de bisecar: en la rama estable gamma vale -0.0 y un criterio de cambio
    de signo se vuelve degenerado. El umbral se interpola linealmente entre el
    último punto estable y el primero con gamma > gamma_tol.
    """
    betas = np.linspace(0.5, 8.0, 120)
    gammas = []
    guess = complex(0.02, 0.01)
    for beta_par in betas:
        disp = ParallelDispersion(beta_par_i=float(beta_par), A_i=A,
                                  beta_par_e=0.1, A_e=1.0,
                                  mass_ratio=1836.0, c_over_va=1.0e4)
        root = disp.solve(k, "plus", guess)
        if np.isfinite(root):
            guess = root if abs(np.imag(root)) < 1.0 else guess
            gammas.append(float(np.imag(root)))
        else:
            gammas.append(float("nan"))

    gammas = np.asarray(gammas)
    unstable = np.flatnonzero(np.isfinite(gammas) & (gammas > gamma_tol))
    if unstable.size == 0 or unstable[0] == 0:
        return float("nan")
    i = int(unstable[0])
    g0, g1 = gammas[i - 1], gammas[i]
    b0, b1 = betas[i - 1], betas[i]
    if not np.isfinite(g0) or g1 == g0:
        beta_crit = b1
    else:
        beta_crit = b0 + (gamma_tol - g0) * (b1 - b0) / (g1 - g0)
    return float(beta_crit * (1.0 - A))


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    if args.case:
        import os
        os.environ["PSC_PROFILE"] = args.case
    from psc_units import (
        BETA_I_PAR, BETA_I_PERP_OVER_PAR, BETA_E_PAR, BETA_E_PERP_OVER_PAR,
        KAPPA, MASS_RATIO, VA_OVER_C, INSTABILITY, PROFILE_LABEL,
    )

    if INSTABILITY == "mirror" and not args.force:
        print("ERROR: el modo mirror es oblicuo y aperiodico; esta relacion de "
              "dispersion es de propagacion paralela y no lo describe. Usa "
              "--force solo si sabes que quieres las ramas paralelas de este "
              "caso (firehose/EMIC), no el mirror.")
        return 2

    beta_par_i = args.beta_par if args.beta_par is not None else BETA_I_PAR
    A_i = args.anisotropy if args.anisotropy is not None else BETA_I_PERP_OVER_PAR
    kappa = args.kappa if args.kappa is not None else KAPPA

    disp = ParallelDispersion(
        beta_par_i=beta_par_i, A_i=A_i,
        beta_par_e=BETA_E_PAR, A_e=BETA_E_PERP_OVER_PAR,
        mass_ratio=MASS_RATIO, c_over_va=1.0 / VA_OVER_C,
        kappa=kappa,
    )

    print(f"Perfil:        {PROFILE_LABEL}")
    print(f"beta_par_i={beta_par_i:.3f}  A_i={A_i:.3f}  "
          f"kappa={kappa}  m_i/m_e={MASS_RATIO}  c/v_A={1.0 / VA_OVER_C:.1f}")

    k_values = np.linspace(args.k_min, args.k_max, args.n_k)
    rows = []
    for pol in args.polarizations:
        rows.extend(disp.scan(k_values, pol))

    finite = [r for r in rows if np.isfinite(r["gamma_over_Omegai"])]
    if finite:
        best = max(finite, key=lambda r: r["gamma_over_Omegai"])
        print(f"gamma_max = {best['gamma_over_Omegai']:.5f} Omega_ci "
              f"en k d_i = {best['kdi']:.4f} ({best['polarization']})")
    else:
        print("[WARN] ninguna raiz convergio; revisa el rango de k o la semilla.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["kdi", "omega_r_over_Omegai",
                            "gamma_over_Omegai", "polarization", "converged",
                            "residual", "iterations"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Escrito: {out}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Relacion de dispersion cinetica lineal para modos "
                    "de propagacion paralela (bi-Maxwelliana o bi-kappa).")
    p.add_argument("--self-test", action="store_true",
                   help="valida la implementacion contra limites conocidos")
    p.add_argument("--case", default=None,
                   help="perfil PSC del que tomar los parametros")
    p.add_argument("--beta-par", type=float, default=None)
    p.add_argument("--anisotropy", type=float, default=None,
                   help="A = T_perp / T_par de los iones")
    p.add_argument("--kappa", type=float, default=None,
                   help="indice kappa; omitir para bi-Maxwelliana")
    p.add_argument("--k-min", type=float, default=0.02)
    p.add_argument("--k-max", type=float, default=2.0)
    p.add_argument("--n-k", type=int, default=60)
    p.add_argument("--polarizations", nargs="*", default=["plus", "minus"],
                   choices=["plus", "minus"])
    p.add_argument("--force", action="store_true",
                   help="permite ejecutar en un caso mirror pese al aviso")
    p.add_argument("--out", default="linear_theory.csv")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
