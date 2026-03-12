"""
psc_units.py
============
Módulo central de unidades físicas para la simulación PSC con masa artifical.

En PSC (en unidades internas del código):
  c = 1,  q_e = 1 (carga del electrón),  mu_0 = 1,  epsilon_0 = 1

Masa artificial: mi/me = 64  =>  me = 1.0,  mi = 64.0

Todas las fórmulas siguen la notación estándar de plasma PIC de campo magnético
orientado según z (dirección paralela = z).

Referencia:
  - Baumjohann & Treumann (1996) - Basic Space Plasma Physics
  - Birdsall & Langdon (1991) - Plasma Physics via Computer Simulation
"""

import numpy as np

# ── Parámetros de la simulación (de psc_temp_aniso.cxx) ─────────────────────
MASS_RATIO      = 64.0     # mi/me (artificial)
ZI              = 1.0      # carga iónica (en unidades de e)
VA_OVER_C       = 0.1      # vA/c
BETA_I_PAR      = 10.0
BETA_I_PERP_OVER_PAR = 3.5
BETA_E_PAR      = 1.0
BETA_E_PERP_OVER_PAR = 1.0
KAPPA           = 3.0      # parámetro kappa de la distribución
N0              = 1.0      # densidad de referencia (normalizada)

# ── Masas en unidades de código ─────────────────────────────────────────────
# NOTA: en PSC con mi/me = 64, se fija me = 1.0 y mi = 64.0 en código.
# Esto es correcto para masa artificial; no confundir con 1/MASS_RATIO.
M_ELEC = 1.0                    # masa del electrón  [unidades código]
M_ION  = MASS_RATIO * ZI        # masa del ión       [unidades código]  = 64.0

# ── Campo magnético y velocidad de Alfvén ───────────────────────────────────
# En PSC: c = 1  =>  B0 = vA/c * c = vA
B0  = VA_OVER_C          # B0 en unidades de código, donde c = 1
VA  = VA_OVER_C          # velocidad de Alfvén vA = B0 (con c=1, n=1, mu0=1)
                          # vA = B0 / sqrt(mu0 * n * mi) = 0.1/sqrt(64) = 0.0125c
                          # pero en unidades normalizadas de PSC: vA = B0 = 0.1

# ── Temperaturas (de las betas y B0) ────────────────────────────────────────
TI_PAR  = BETA_I_PAR * B0**2 / 2.0                  # = 0.050
TI_PERP = BETA_I_PERP_OVER_PAR * TI_PAR             # = 0.175
TE_PAR  = BETA_E_PAR * B0**2 / 2.0                  # = 0.005
TE_PERP = BETA_E_PERP_OVER_PAR * TE_PAR             # = 0.005

# ── Frecuencias de plasma ─────────────────────────────────────────────────
# omega_pi = sqrt(n0 * qi^2 / mi) = sqrt(N0 * ZI^2 / M_ION)
OMEGA_PI  = np.sqrt(N0 * ZI**2 / M_ION)   # freq. plasma iónica  [rad/t_code]
OMEGA_PE  = np.sqrt(N0 * 1.0**2 / M_ELEC) # freq. plasma electrónica [rad/t_code]

# ── Distancias inerciales (skin depth) ──────────────────────────────────────
# d_i = c / omega_pi = 1 / omega_pi  (con c=1)
DI = 1.0 / OMEGA_PI    # longitud inercial iónica    [celdas] = sqrt(64) = 8
DE = 1.0 / OMEGA_PE    # longitud inercial electrónica [celdas] = 1.0

# ── Frecuencias de giro (ciclotrón) ─────────────────────────────────────────
# Omega_ci = |qi| * B0 / mi
OMEGA_CI  = ZI * B0 / M_ION    # freq. ciclotrón iónica     [rad/t_code]
OMEGA_CE  = 1.0 * B0 / M_ELEC  # freq. ciclotrón electrónica [rad/t_code]

# ── Radios de Larmor térmicos ────────────────────────────────────────────────
# rho_i = v_th_perp / Omega_ci,  v_th = sqrt(T_perp / m)
RHO_I = np.sqrt(TI_PERP / M_ION) / OMEGA_CI   # radio de Larmor iónico  [celdas]
RHO_E = np.sqrt(TE_PERP / M_ELEC) / OMEGA_CE  # radio de Larmor electrónico [celdas]

# ── Velocidades térmicas ─────────────────────────────────────────────────────
VTH_I_PAR  = np.sqrt(TI_PAR  / M_ION)
VTH_I_PERP = np.sqrt(TI_PERP / M_ION)
VTH_E_PAR  = np.sqrt(TE_PAR  / M_ELEC)
VTH_E_PERP = np.sqrt(TE_PERP / M_ELEC)

# ── Betas de plasma ──────────────────────────────────────────────────────────
# beta = 2 * mu0 * n * T / B^2  (con mu0=1, n=1)
BETA_I_PAR_COMPUTED  = 2.0 * N0 * TI_PAR  / B0**2
BETA_I_PERP_COMPUTED = 2.0 * N0 * TI_PERP / B0**2
BETA_E_PAR_COMPUTED  = 2.0 * N0 * TE_PAR  / B0**2

# ── Grilla ───────────────────────────────────────────────────────────────────
N_GRID_Y = 384
N_GRID_Z = 512
NICELL   = 250
CORI     = 1.0 / NICELL

# ── Funciones de conversión ───────────────────────────────────────────────────

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


def print_units_summary():
    """Imprime un resumen de todas las unidades físicas relevantes."""
    print("=" * 65)
    print("  UNIDADES FÍSICAS — PSC  (mi/me = 64, vA/c = 0.1)")
    print("=" * 65)
    print()
    print("  Masas:")
    print(f"    m_electron  = {M_ELEC:.4f}  [código]")
    print(f"    m_ion       = {M_ION:.4f}  [código]   =>  mi/me = {M_ION/M_ELEC:.0f}")
    print()
    print("  Campo magnético / Alfvén:")
    print(f"    B0          = {B0:.4f}     [código]")
    print(f"    vA          = {VA:.4f}     [código]   = {VA_OVER_C:.4f} c")
    print()
    print("  Distancias inerciales:")
    print(f"    dᵢ  = c/ωₚᵢ = {DI:.4f}  [celdas/dᵢ]   =>  1 dᵢ = {DI:.1f} celdas")
    print(f"    dₑ  = c/ωₚₑ = {DE:.4f}  [celdas/dₑ]   =>  1 dₑ = {DE:.1f} celda")
    print(f"    Grilla: {N_GRID_Y/DI:.1f} dᵢ × {N_GRID_Z/DI:.1f} dᵢ")
    print()
    print("  Frecuencias de ciclotrón:")
    print(f"    Ωcᵢ = qᵢB₀/mᵢ = {OMEGA_CI:.6f}  [rad/t_código]")
    print(f"    Ωcₑ = qₑB₀/mₑ = {OMEGA_CE:.6f}  [rad/t_código]")
    print(f"    Ωcₑ / Ωcᵢ = {OMEGA_CE/OMEGA_CI:.1f}  (= mi/me, correcto)")
    print()
    print("  Frecuencias de plasma:")
    print(f"    ωₚᵢ = {OMEGA_PI:.6f}  [rad/t_código]   =>  1 Ωcᵢ⁻¹ = {1/OMEGA_CI:.1f} t_código")
    print(f"    ωₚₑ = {OMEGA_PE:.6f}  [rad/t_código]")
    print()
    print("  Radios de Larmor térmicos (t=0):")
    print(f"    ρᵢ  = vth_i_perp / Ωcᵢ = {RHO_I:.4f}  [celdas]  = {RHO_I/DI:.4f} dᵢ")
    print(f"    ρₑ  = vth_e_perp / Ωcₑ = {RHO_E:.4f}  [celdas]  = {RHO_E/DE:.4f} dₑ")
    print()
    print("  Velocidades térmicas (t=0):")
    print(f"    vth_i‖  = {VTH_I_PAR:.5f}  [código]  = {VTH_I_PAR/VA:.3f} vA")
    print(f"    vth_i⊥  = {VTH_I_PERP:.5f}  [código]  = {VTH_I_PERP/VA:.3f} vA")
    print(f"    vth_e‖  = {VTH_E_PAR:.5f}  [código]  = {VTH_E_PAR/VA:.3f} vA")
    print()
    print("  Betas de plasma:")
    print(f"    βᵢ‖ = {BETA_I_PAR_COMPUTED:.2f}   βᵢ⊥ = {BETA_I_PERP_COMPUTED:.2f}")
    print(f"    βₑ‖ = {BETA_E_PAR_COMPUTED:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    print_units_summary()
