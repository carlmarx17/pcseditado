"""
psc_units.py
============
Módulo central de unidades físicas para las simulaciones PSC con masa artificial.

En PSC (en unidades internas del código):
  c = 1,  q_e = 1 (carga del electrón),  mu_0 = 1,  epsilon_0 = 1

Masa artificial: mi/me = 200  =>  me = 1.0,  mi = 200.0

Parámetros comunes a las 4 simulaciones:
  - mass_ratio = 200,  vA/c = 0.05
  - Dominio: 32 d_i × 32 d_i,  grilla 128×128 celdas  (d_i = √200 ≈ 14.14 celdas)
  - nicell = 2000 ppc

CONFIGURACIONES DE INESTABILIDAD:
  Mirror   :  beta_i_par = 5,   Ti_perp/Ti_par = 3.0   (T_perp > T_par)
  Firehose :  beta_i_par = 10,  Ti_perp/Ti_par = 0.1   (T_par > T_perp)

  Cada una tiene variante Maxwellian (kappa=None) y Kappa (kappa=3).

Selección del perfil activo:  configurar SIM_PROFILE al inicio del análisis
  o pasar --profile mirror_kappa | mirror_maxwellian | firehose_kappa | firehose_maxwellian

Todas las fórmulas siguen la notación estándar de plasma PIC de campo magnético
orientado según z (dirección paralela = z).

Referencia:
  - Baumjohann & Treumann (1996) - Basic Space Plasma Physics
  - Birdsall & Langdon (1991) - Plasma Physics via Computer Simulation
"""

import os
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Simulation profiles — all 4 simulation configurations
# ══════════════════════════════════════════════════════════════════════════

_PROFILES = {
    "mirror_kappa": {
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
    },
    "mirror_maxwellian": {
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
    },
    "firehose_kappa": {
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
    },
    "firehose_maxwellian": {
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
    },
}

# ── Seleccionar perfil activo ────────────────────────────────────────────
# Puede ser sobreescrito con la variable de entorno PSC_PROFILE
SIM_PROFILE = os.environ.get("PSC_PROFILE", "mirror_kappa")

if SIM_PROFILE not in _PROFILES:
    raise ValueError(
        f"Perfil desconocido: '{SIM_PROFILE}'.  "
        f"Perfiles válidos: {list(_PROFILES.keys())}"
    )

_active = _PROFILES[SIM_PROFILE]

# ── Parámetros de la simulación (desde los .cxx) ────────────────────────
MASS_RATIO      = _active["mass_ratio"]       # mi/me = 200
ZI              = 1.0                          # carga iónica (en unidades de e)
VA_OVER_C       = _active["vA_over_c"]        # vA/c = 0.05
BETA_I_PAR      = _active["beta_i_par"]       # 5 (mirror) ó 10 (firehose)
BETA_I_PERP_OVER_PAR = _active["Ti_perp_over_Ti_par"]  # 3.0 ó 0.1
BETA_E_PAR      = _active["beta_e_par"]       # 1.0
BETA_E_PERP_OVER_PAR = _active["Te_perp_over_Te_par"]  # 1.0
KAPPA           = _active["kappa"]            # 3.0 (kappa) ó None (Maxwellian)
N0              = 1.0                          # densidad de referencia

# ── Masas en unidades de código ─────────────────────────────────────────
M_ELEC = 1.0                    # masa del electrón  [unidades código]
M_ION  = MASS_RATIO * ZI        # masa del ión       [unidades código]  = 200.0

# ── Campo magnético y velocidad de Alfvén ───────────────────────────────
# En los .cxx: g.B0 = g.vA_over_c = 0.05
# (simplificación PIC con c=1, n₀=1)
B0   = VA_OVER_C                # = 0.05  (campo de fondo, consistente con CXX)
VA   = VA_OVER_C                # velocidad Alfvén normalizada [c=1]

# ── Temperaturas (de las betas y B0) ────────────────────────────────────
# T = beta * B0² / 2  (con n=1, mu0=1)
TI_PAR  = BETA_I_PAR * B0**2 / (2.0 * N0)
TI_PERP = BETA_I_PERP_OVER_PAR * TI_PAR
TE_PAR  = BETA_E_PAR * B0**2 / (2.0 * N0)
TE_PERP = BETA_E_PERP_OVER_PAR * TE_PAR

# ── Frecuencias de plasma ─────────────────────────────────────────────
# omega_pi = sqrt(n0 * qi^2 / mi) = sqrt(N0 * ZI^2 / M_ION)
OMEGA_PI  = np.sqrt(N0 * ZI**2 / M_ION)   # freq. plasma iónica  [rad/t_code]
OMEGA_PE  = np.sqrt(N0 * 1.0**2 / M_ELEC) # freq. plasma electrónica [rad/t_code]

# ── Distancias inerciales (skin depth) ──────────────────────────────────
# d_i = c / omega_pi = 1 / omega_pi  (con c=1)
# En los .cxx: d_i = sqrt(mass_ratio / n) = sqrt(100) = 10
DI = 1.0 / OMEGA_PI    # longitud inercial iónica    [celdas] = 10
DE = 1.0 / OMEGA_PE    # longitud inercial electrónica [celdas] = 1.0

# ── Frecuencias de giro (ciclotrón) ─────────────────────────────────────
# Omega_ci = |qi| * B0 / mi
OMEGA_CI  = ZI * B0 / M_ION    # freq. ciclotrón iónica     [rad/t_code]
OMEGA_CE  = 1.0 * B0 / M_ELEC  # freq. ciclotrón electrónica [rad/t_code]

# ── Radios de Larmor térmicos ────────────────────────────────────────────
# rho_i = v_th_perp / Omega_ci,  v_th = sqrt(T_perp / m)
RHO_I = np.sqrt(TI_PERP / M_ION) / OMEGA_CI   # radio de Larmor iónico  [celdas]
RHO_E = np.sqrt(TE_PERP / M_ELEC) / OMEGA_CE  # radio de Larmor electrónico [celdas]

# ── Velocidades térmicas ─────────────────────────────────────────────────
VTH_I_PAR  = np.sqrt(TI_PAR  / M_ION)
VTH_I_PERP = np.sqrt(TI_PERP / M_ION)
VTH_E_PAR  = np.sqrt(TE_PAR  / M_ELEC)
VTH_E_PERP = np.sqrt(TE_PERP / M_ELEC)

# ── Betas de plasma ──────────────────────────────────────────────────────
# beta = 2 * mu0 * n * T / B^2  (con mu0=1, n=1)
BETA_I_PAR_COMPUTED  = 2.0 * N0 * TI_PAR  / B0**2
BETA_I_PERP_COMPUTED = 2.0 * N0 * TI_PERP / B0**2
BETA_E_PAR_COMPUTED  = 2.0 * N0 * TE_PAR  / B0**2

# ── Grilla y dominio ─────────────────────────────────────────────────────
# .cxx: domain_size = 32 * d_i,  gdims = 128x128
# d_i = sqrt(mi/n) = sqrt(200) ≈ 14.14  =>  domain_size ≈ 452.5 celdas
# 128 celdas => Δ ≈ 452.5/128 ≈ 3.54 celdas  ≈ 0.25 d_i  ✓
N_GRID_Y  = 128              # celdas en Y
N_GRID_Z  = 128              # celdas en Z
DOMAIN_DI = 32.0             # extensión física del dominio en d_i
DOMAIN_DE = DOMAIN_DI * DI   # extensión en d_e  (≈ 452.5)
NICELL    = 2000             # partículas por celda (ver setupGrid)
CORI      = 1.0 / NICELL

# Tamaño del dominio en d_i (calculado a partir de DI)
DOMAIN_DI_Y = DOMAIN_DI   # 32 d_i
DOMAIN_DI_Z = DOMAIN_DI   # 32 d_i

# ── Región de salida de partículas ───────────────────────────────────────
# Solo se guardan partículas de la región central 8×8 d_i (celdas 48–80)
# para optimizar almacenamiento (~180 MB/snapshot vs 2.88 GB del dominio completo)
PRT_OUTPUT_LO = (0, 48, 48)
PRT_OUTPUT_HI = (1, 80, 80)
PRT_OUTPUT_EVERY = 500       # cada 500 pasos (≈ 200 snapshots en nmax=100000)

# ── Constantes auxiliares para análisis ─────────────────────────────────
MU0 = 1.0
FIELD_FILE_PATTERN = "pfd.*.h5"
MOMENT_FILE_PATTERN = "pfd_moments.*.h5"
PARTICLE_FILE_PATTERN = "prt.*.h5"

# ── Funciones de conversión ───────────────────────────────────────────────

def cells_to_di(x_cells):
    """Convierte posición de celdas a longitudes inerciales iónicas (dᵢ)."""
    return x_cells / DI

def di_to_cells(x_di):
    """Convierte longitudes inerciales iónicas a celdas."""
    return x_di * DI

def time_to_omegaci(t_code):
    """Convierte tiempo de código a unidades de Ωcᵢ⁻¹  (tiempo giromagnético iónico)."""
    return t_code * OMEGA_CI

def time_to_omegapi(t_code):
    """Convierte tiempo de código a unidades de ωₚᵢ⁻¹."""
    return t_code * OMEGA_PI

def vel_to_va(v_code):
    """Convierte velocidad de código a unidades de vA (velocidad de Alfvén)."""
    return v_code / VA

def momentum_to_va(p_code, m):
    """Convierte momentum p = m*v a velocidad en unidades de vA."""
    return (p_code / m) / VA

def temp_to_va2mi(T_code, m=M_ION):
    """Convierte temperatura T [código] a T/(mi*vA²) (normalización estándar)."""
    return T_code / (m * VA**2)


def step_to_omegaci(step: int, dt_code: float = 1.0) -> float:
    """Convierte un step de simulación a tiempo normalizado por Ωci^-1."""
    return step * dt_code * OMEGA_CI


def cells_to_di_y(cells):
    """Convierte índice de celda en Y a posición en dᵢ."""
    return cells * DOMAIN_DE / (N_GRID_Y * DI)


def cells_to_di_z(cells):
    """Convierte índice de celda en Z a posición en dᵢ."""
    return cells * DOMAIN_DE / (N_GRID_Z * DI)


def get_profile_name() -> str:
    """Retorna el nombre del perfil activo."""
    return SIM_PROFILE


def list_profiles() -> list:
    """Retorna los perfiles disponibles."""
    return list(_PROFILES.keys())


def print_units_summary():
    """Imprime un resumen de todas las unidades físicas relevantes."""
    kappa_str = f"κ = {KAPPA}" if KAPPA is not None else "Maxwellian"
    print("=" * 68)
    print(f"  UNIDADES FÍSICAS — PSC  (mi/me={int(MASS_RATIO)}, vA/c={VA_OVER_C})")
    print(f"  Perfil activo: {SIM_PROFILE}  ({kappa_str})")
    print("=" * 68)
    print()
    print("  Masas:")
    print(f"    m_electron  = {M_ELEC:.4f}  [código]")
    print(f"    m_ion       = {M_ION:.4f}  [código]   =>  mi/me = {M_ION/M_ELEC:.0f}")
    print()
    print("  Campo magnético / Alfvén:")
    print(f"    B0          = {B0:.6f}  [código]  (= vA/c en CXX)")
    print(f"    vA          = {VA:.4f}      [código]   = {VA_OVER_C:.4f} c")
    print()
    print("  Distancias inerciales:")
    print(f"    dᵢ  = c/ωₚᵢ = {DI:.4f}  celdas/dᵢ  =>  1 dᵢ = {DI:.2f} celdas")
    print(f"    dₑ  = c/ωₚₑ = {DE:.4f}  celdas/dₑ  =>  1 dₑ = {DE:.2f} celda")
    print(f"    Dominio físico: {DOMAIN_DI:.0f} dᵢ × {DOMAIN_DI:.0f} dᵢ  "
          f"= {DOMAIN_DE:.0f} dₑ × {DOMAIN_DE:.0f} dₑ")
    print(f"    Grilla: {N_GRID_Y} × {N_GRID_Z} celdas  "
          f"(Δ = {DOMAIN_DI/N_GRID_Y:.2f} dᵢ per cell)")
    print()
    print("  Temperaturas (t=0):")
    print(f"    Tᵢ‖  = {TI_PAR:.6f}   βᵢ‖ = {BETA_I_PAR_COMPUTED:.2f}")
    print(f"    Tᵢ⊥  = {TI_PERP:.6f}   Aᵢ  = Tᵢ⊥/Tᵢ‖ = {BETA_I_PERP_OVER_PAR:.1f}")
    print(f"    Tₑ‖  = {TE_PAR:.6f}   βₑ‖ = {BETA_E_PAR_COMPUTED:.2f}")
    print()
    print("  Frecuencias de ciclotrón:")
    print(f"    Ωcᵢ = {OMEGA_CI:.6f}  [rad/t_código]   =>  1 Ωcᵢ⁻¹ = {1/OMEGA_CI:.1f} pasos")
    print(f"    Ωcₑ = {OMEGA_CE:.6f}  [rad/t_código]")
    print(f"    Ωcₑ / Ωcᵢ = {OMEGA_CE/OMEGA_CI:.1f}  (= mi/me, correcto)")
    print()
    print("  Radios de Larmor térmicos (t=0):")
    print(f"    ρᵢ  = {RHO_I:.4f}  celdas  = {RHO_I/DI:.4f} dᵢ")
    print(f"    ρₑ  = {RHO_E:.4f}  celdas  = {RHO_E/DE:.4f} dₑ")
    print()
    print("  Velocidades térmicas (t=0):")
    print(f"    vth_i‖  = {VTH_I_PAR:.5f}  [código]  = {VTH_I_PAR/VA:.3f} vA")
    print(f"    vth_i⊥  = {VTH_I_PERP:.5f}  [código]  = {VTH_I_PERP/VA:.3f} vA")
    print(f"    vth_e‖  = {VTH_E_PAR:.5f}  [código]  = {VTH_E_PAR/VA:.3f} vA")
    print()
    print(f"  Instabilidad: ", end="")
    if BETA_I_PERP_OVER_PAR > 1.0:
        print(f"MIRROR  (Aᵢ = {BETA_I_PERP_OVER_PAR:.1f} > 1)")
    elif BETA_I_PERP_OVER_PAR < 1.0:
        print(f"FIREHOSE  (Aᵢ = {BETA_I_PERP_OVER_PAR:.1f} < 1)")
    else:
        print("Isótropo")
    print("=" * 68)


if __name__ == "__main__":
    print_units_summary()
