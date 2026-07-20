"""
psc_units.py
============
Módulo central de unidades físicas para las simulaciones PSC con masa artificial.

En PSC (en unidades internas del código):
  c = 1,  q_e = 1 (carga del electrón),  mu_0 = 1,  epsilon_0 = 1

Masa artificial: mi/me = 200  =>  me = 1.0,  mi = 200.0

Parámetros comunes de los ejecutables actuales de anisotropía:
  - mass_ratio = 200,  vA/c = B0 = 0.08
  - dominio = 20 d_i x 20 d_i, grilla = 1024 x 1024, nicell = 1500

Los perfiles heredados `F_*_bM`, `M_*_bM`, `W_*_bM` y `*_lite` se mantienen
para analizar corridas antiguas.


CONFIGURACIONES:
  F_*_bM: Firehose iónico fuerte, medio y débil.
  M_*_bM: Mirror iónico fuerte, medio y débil.
  W_*_bM: Whistler electrónico fuerte, medio y débil.

Selección del perfil activo: configurar PSC_PROFILE antes del análisis.
El flujo usa por defecto M_S_bM; en producción el Makefile siempre debe pasar
el perfil correspondiente al valor de CASE.

Todas las fórmulas siguen la notación estándar de plasma PIC de campo magnético
orientado según z (dirección paralela = z).

Referencia:
  - Baumjohann & Treumann (1996) - Basic Space Plasma Physics
  - Birdsall & Langdon (1991) - Plasma Physics via Computer Simulation
"""

import os
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Simulation profiles
# ══════════════════════════════════════════════════════════════════════════

_PROFILES = {
    "mirror_bimaxwellian_strong": {
        "label": "Mirror Strong Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bimaxwellian_strong",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "mirror_bimaxwellian_moderate": {
        "label": "Mirror Moderate Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 2.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bimaxwellian_moderate",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "mirror_bimaxwellian_weak": {
        "label": "Mirror Weak Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 6.0,
        "Ti_perp_over_Ti_par": 1.5,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bimaxwellian_weak",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "firehose_bimaxwellian_strong": {
        "label": "Firehose Strong Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_bimaxwellian_strong",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "firehose_bimaxwellian_moderate": {
        "label": "Firehose Moderate Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 6.0,
        "Ti_perp_over_Ti_par": 0.3,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_bimaxwellian_moderate",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "firehose_bimaxwellian_weak": {
        "label": "Firehose Weak Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 3.0,
        "Ti_perp_over_Ti_par": 0.6,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_bimaxwellian_weak",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "whistler_bimaxwellian_strong": {
        "label": "Whistler Strong Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 1.0,
        "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5,
        "Te_perp_over_Te_par": 3.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_whistler_bimaxwellian_strong",
        "instability": "whistler",
        "driven_species": "electron",
    },
    "whistler_bimaxwellian_moderate": {
        "label": "Whistler Moderate Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 1.0,
        "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5,
        "Te_perp_over_Te_par": 2.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_whistler_bimaxwellian_moderate",
        "instability": "whistler",
        "driven_species": "electron",
    },
    "whistler_bimaxwellian_weak": {
        "label": "Whistler Weak Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 1.0,
        "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5,
        "Te_perp_over_Te_par": 1.5,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_whistler_bimaxwellian_weak",
        "instability": "whistler",
        "driven_species": "electron",
    },
    "mirror_bikappa3": {
        "label": "Mirror Bi-Kappa 3",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bikappa3",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "mirror_bikappa3_moderate": {
        "label": "Mirror Moderate Bi-Kappa 3",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 2.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bikappa3_moderate",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "mirror_bikappa5": {
        "label": "Mirror Bi-Kappa 5",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 5.0,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_bikappa5",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "firehose_bikappa3": {
        "label": "Firehose Bi-Kappa 3",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_bikappa3",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "firehose_bikappa5": {
        "label": "Firehose Bi-Kappa 5",
        "mass_ratio": 200.0,
        "vA_over_c": 0.08,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 5.0,
        "domain_di": 20.0,
        "ngrid": 576,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_bikappa5",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "M_S_bM": {
        "label": "Mirror Strong Bi-Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 30.0,
        "ngrid": 1408,
        "nmax": 1_650_000,
        "nicell": 1000,
        "particle_basename": "prt_M_S_bM",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "M_M_bM": {
        "label": "Mirror Moderate Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 5.0, "Ti_perp_over_Ti_par": 2.0,
        "beta_e_par": 1.0, "Te_perp_over_Te_par": 1.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_M_M_bM",
        "instability": "mirror", "driven_species": "ion",
    },
    "M_W_bM": {
        "label": "Mirror Weak Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 6.0, "Ti_perp_over_Ti_par": 1.5,
        "beta_e_par": 1.0, "Te_perp_over_Te_par": 1.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_M_W_bM",
        "instability": "mirror", "driven_species": "ion",
    },
    "F_S_bM": {
        "label": "Firehose Strong Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 10.0, "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0, "Te_perp_over_Te_par": 1.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_F_S_bM",
        "instability": "firehose", "driven_species": "ion",
    },
    "F_M_bM": {
        "label": "Firehose Moderate Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 6.0, "Ti_perp_over_Ti_par": 0.3,
        "beta_e_par": 1.0, "Te_perp_over_Te_par": 1.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_F_M_bM",
        "instability": "firehose", "driven_species": "ion",
    },
    "F_W_bM": {
        "label": "Firehose Weak Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 3.0, "Ti_perp_over_Ti_par": 0.6,
        "beta_e_par": 1.0, "Te_perp_over_Te_par": 1.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_F_W_bM",
        "instability": "firehose", "driven_species": "ion",
    },
    "W_S_bM": {
        "label": "Whistler Strong Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 1.0, "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5, "Te_perp_over_Te_par": 3.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_W_S_bM",
        "instability": "whistler", "driven_species": "electron",
    },
    "W_M_bM": {
        "label": "Whistler Moderate Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 1.0, "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5, "Te_perp_over_Te_par": 2.0,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_W_M_bM",
        "instability": "whistler", "driven_species": "electron",
    },
    "W_W_bM": {
        "label": "Whistler Weak Bi-Maxwellian",
        "mass_ratio": 200.0, "vA_over_c": 0.05,
        "beta_i_par": 1.0, "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5, "Te_perp_over_Te_par": 1.5,
        "kappa": None, "domain_di": 30.0, "ngrid": 1408,
        "nmax": 1_650_000, "nicell": 1000,
        "particle_basename": "prt_W_W_bM",
        "instability": "whistler", "driven_species": "electron",
    },
    "F_lite": {
        "label": "Firehose Lite",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 512,
        "nmax": 180_000,
        "nicell": 1000,
        "particle_basename": "prt_F_lite",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "M_lite": {
        "label": "Mirror Lite",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 512,
        "nmax": 180_000,
        "nicell": 1000,
        "particle_basename": "prt_M_lite",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "W_lite": {
        "label": "Whistler Lite",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 1.0,
        "Ti_perp_over_Ti_par": 1.0,
        "beta_e_par": 0.5,
        "Te_perp_over_Te_par": 3.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 512,
        "nmax": 180_000,
        "nicell": 1000,
        "particle_basename": "prt_W_lite",
        "instability": "whistler",
        "driven_species": "electron",
    },
    "mirror_kappa": {
        "label": "Mirror Kappa",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
        "domain_di": 20.0,
        "ngrid": 1536,
        "nmax": 1_800_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_kappa3",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "mirror_maxwellian": {
        "label": "Mirror Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 5.0,
        "Ti_perp_over_Ti_par": 3.0,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 1536,
        "nmax": 1_800_000,
        "nicell": 1000,
        "particle_basename": "prt_mirror_maxwellian",
        "instability": "mirror",
        "driven_species": "ion",
    },
    "firehose_kappa": {
        "label": "Firehose Kappa",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": 3.0,
        "domain_di": 20.0,
        "ngrid": 1024,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_kappa3",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "firehose_maxwellian": {
        "label": "Firehose Maxwellian",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 1024,
        "nmax": 1_200_000,
        "nicell": 1000,
        "particle_basename": "prt_firehose_maxwellian",
        "instability": "firehose",
        "driven_species": "ion",
    },
    "F_S_bM_local": {
        "label": "Firehose Strong Bi-Maxwellian (local)",
        "mass_ratio": 200.0,
        "vA_over_c": 0.05,
        "beta_i_par": 10.0,
        "Ti_perp_over_Ti_par": 0.1,
        "beta_e_par": 1.0,
        "Te_perp_over_Te_par": 1.0,
        "kappa": None,
        "domain_di": 20.0,
        "ngrid": 512,
        "nmax": 72_000,
        "nicell": 1000,
        "particle_basename": "prt_F_S_bM_local",
        "instability": "firehose",
        "driven_species": "ion",
    },
}

# ── Seleccionar perfil activo ────────────────────────────────────────────
# Puede ser sobreescrito con la variable de entorno PSC_PROFILE.
# El Makefile lo fija a CASE; este default solo aplica al ejecutar scripts
# directamente sin entorno.
SIM_PROFILE = os.environ.get("PSC_PROFILE", "mirror_bimaxwellian_strong")

if SIM_PROFILE not in _PROFILES:
    raise ValueError(
        f"Perfil desconocido: '{SIM_PROFILE}'.  "
        f"Perfiles válidos: {list(_PROFILES.keys())}"
    )

_active = _PROFILES[SIM_PROFILE]
PROFILE_LABEL = _active["label"]
INSTABILITY = _active["instability"]
DRIVEN_SPECIES = _active["driven_species"]

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
# Los parámetros de grilla dependen del perfil (mirror vs firehose).
#
# PSC 1vbec = full PIC (1st order Villasenor-Buneman Edge-Centered).
# Ambas especies (iones y electrones) son partículas cinéticas.
#
# Estos valores deben coincidir con setupGrid() y setupParameters() del .cxx.
N_GRID_Y  = _active["ngrid"]             # celdas en Y
N_GRID_Z  = _active["ngrid"]             # celdas en Z
DOMAIN_DI = _active["domain_di"]         # extensión del dominio en d_i
NMAX      = _active["nmax"]              # número máximo de pasos
DOMAIN_DE = DOMAIN_DI * DI              # extensión en d_e
DX_DI     = DOMAIN_DI / N_GRID_Y       # resolución [d_i / celda]
DX_DE     = DOMAIN_DE / N_GRID_Y       # resolución [d_e / celda]
NICELL    = _active["nicell"]           # partículas por celda (setupGrid)
CORI      = 1.0 / NICELL

# Tamaño del dominio en d_i
DOMAIN_DI_Y = DOMAIN_DI
DOMAIN_DI_Z = DOMAIN_DI

# ── Región de salida de partículas ───────────────────────────────────────
# Central 20% per resolved direction, matching the maintained C++ executables.
_prt_half = int(round(0.1 * N_GRID_Y))
_prt_center = N_GRID_Y // 2
PRT_OUTPUT_LO = (0, _prt_center - _prt_half, _prt_center - _prt_half)
PRT_OUTPUT_HI = (1, _prt_center + _prt_half, _prt_center + _prt_half)
PRT_OUTPUT_EVERY = 10000

# ── Constantes auxiliares para análisis ─────────────────────────────────
MU0 = 1.0
# dt del código: dt = CFL * courant_length(domain)
# courant_length(2D) = dx / sqrt(2),  dx = domain_de / N_grid
_dx_code = DOMAIN_DE / N_GRID_Y
DT_CODE = 0.95 * _dx_code / np.sqrt(2.0)
FIELD_FILE_PATTERN = "pfd.*.h5"
MOMENT_FILE_PATTERN = "pfd_moments.*.h5"
PARTICLE_BASENAME = _active["particle_basename"]
PARTICLE_FILE_PATTERN = f"{PARTICLE_BASENAME}.*.h5"


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


def step_to_omegaci(step: int, dt_code: float = DT_CODE) -> float:
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
