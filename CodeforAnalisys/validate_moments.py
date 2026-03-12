#!/usr/bin/env python3
"""
validate_moments.py
====================
Valida que los momentos de la distribución (densidad, velocidad macroscópica,
temperatura) calculados de los datos de partículas coincidan con los parámetros
utiizados en psc_temp_aniso.cxx (createKappaMultivariate).

Unidades físicas usando mi/me = 64 (masa artificial):
  Espacial  →  dᵢ = c/ωₚᵢ = √(mᵢ/n₀) = 8  [celdas/dᵢ]
  Temporal  →  Ωcᵢ = qᵢB₀/mᵢ = 0.1/64    [rad/t_código]
  Velocidad →  vA = B₀ = 0.1              [en código con c=1]

Usage:
    python validate_moments.py [path_to_prt_file.h5]

If no path is given, defaults to ../build/src/prt.000000000.h5 (t=0).
"""

import sys
import os
import h5py
import numpy as np
from scipy.special import gamma as gamma_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# 1. EXPECTED VALUES — Extracted from psc_temp_aniso.cxx
#    These are the "ground truth" parameters used to generate the particles.
# ═══════════════════════════════════════════════════════════════════════

# Simulation parameters (from setupParameters())
BB = 1.0
Zi = 1.0
MASS_RATIO = 64.0
VA_OVER_C = 0.1
BETA_E_PAR = 1.0
BETA_I_PAR = 10.0
TI_PERP_OVER_TI_PAR = 3.5
TE_PERP_OVER_TE_PAR = 1.0
N_DENSITY = 1.0
KAPPA = 3.0

# Derived quantities (same formulas as C++ code)
B0 = VA_OVER_C                                     # = 0.1
TE_PAR = BETA_E_PAR * B0**2 / 2.0                  # = 0.005
TE_PERP = TE_PERP_OVER_TE_PAR * TE_PAR             # = 0.005
TI_PAR = BETA_I_PAR * B0**2 / 2.0                  # = 0.05
TI_PERP = TI_PERP_OVER_TI_PAR * TI_PAR             # = 0.175

# NOTA sobre masas con masa artificial mi/me = 64:
#   En PSC con masa artificial, el código fija:
#     m_electron = 1.0  (unidades código)
#     m_ion      = MASS_RATIO * Zi = 64.0
#   NO confundir con la masa real del electrón (1/1836 mp).
#   La simulación es válida porque mi/me = 64 en la simulación,
#   lo que reduce la separación de escala temporal ce/ci de forma controlada.
M_ION      = MASS_RATIO * Zi   # = 64.0  [unidades PSC]
M_ELECTRON = 1.0               # = 1.0   [unidades PSC, masa artificial]

# Unidades físicas (con c=1, n₀=1, μ₀=1 en PSC)
VA       = B0                                   # vA = B₀/√(n₀mᵢ) en código
DI       = np.sqrt(M_ION / N_DENSITY)           # dᵢ = c/ωₚᵢ = 8 celdas
DE       = np.sqrt(M_ELECTRON / N_DENSITY)      # dₑ = c/ωₚₑ = 1 celda
OMEGA_CI = Zi * B0 / M_ION                      # Ωcᵢ = 0.001563 rad/t
OMEGA_CE = 1.0 * B0 / M_ELECTRON               # Ωcₑ = 0.1 rad/t

# Grid parameters (from setupGrid())
NICELL = 250   # particles per cell per species at density = 1
N_GRID_Y = 384
N_GRID_Z = 512

# Normalization: dimensionless => beta_norm = 1, cori = 1/nicell
BETA_NORM = 1.0
CORI = 1.0 / NICELL

# Definición de especies PSC:
#   iones:      q = +Zi = +1.0,  m = MASS_RATIO * Zi = 64.0
#   electrones: q = -1.0,        m = 1.0  (masa artificial, mi/me = 64)
Q_ION      = Zi     # = +1.0
Q_ELECTRON = -1.0

# Expected number of particles per cell (with fractional_n_particles_per_cell = true)
# n_in_cell = n / cori + random ~ 250 per species
EXPECTED_PPC = N_DENSITY / CORI  # = 250

# Total cells in the 2D grid (X is invariant with 1 cell)
N_CELLS = 1 * N_GRID_Y * N_GRID_Z  # = 196608
EXPECTED_TOTAL_PER_SPECIES = EXPECTED_PPC * N_CELLS  # ~ 49,152,000


# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD PARTICLE DATA
# ═══════════════════════════════════════════════════════════════════════

def load_particles(filepath):
    """Load particle data from a PSC prt.*.h5 file."""
    print(f"Loading particles from: {filepath}")
    print(f"  (file size: {os.path.getsize(filepath) / 1e9:.2f} GB)")
    with h5py.File(filepath, 'r') as f:
        grp = f['particles']['p0']
        data = grp['1d'][:]
    print(f"  Total particles: {len(data):,}")
    print(f"  Fields: {data.dtype.names}")
    return data


def separate_species(data):
    """Separate particles by charge sign: ions (q>0), electrons (q<0)."""
    ions = data[data['q'] > 0]
    electrons = data[data['q'] < 0]
    print(f"  Ions:      {len(ions):,}")
    print(f"  Electrons: {len(electrons):,}")
    return ions, electrons


# ═══════════════════════════════════════════════════════════════════════
# 3. MOMENT COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_moments(species, species_name, expected_T_perp, expected_T_par, expected_m):
    """
    Compute the 0th, 1st, and 2nd moments of the distribution and compare
    with the expected values.

    PSC stores momenta as p_i (not velocities), so:
      - Velocity: v_i = p_i / m   (non-relativistic)
      - Temperature: T_i = m * Var(p_i/m) = Var(p_i) / m
        But in the sampling code: Var(p_i) = beta^2 * T_i / m
        So: T_measured = m * <(p_i - <p_i>)^2> / beta^2
        With beta=1 (dimensionless): T_measured = m * <p_i^2> (assuming <p_i>=0)

    Actually, the weight w matters! In PSC with fractional_n_particles_per_cell=true,
    all weights are 1.0. The density is the total weight per cell * cori.
    """
    m = np.abs(species['m'][0])
    q = species['q'][0]
    w = species['w']

    print(f"\n{'═'*70}")
    print(f"  MOMENT VALIDATION: {species_name}")
    print(f"{'═'*70}")

    # --- 0th Moment: Density ---
    # With fractional_n = true, weight = 1.0 for all particles.
    # Number of particles per cell ~ n / cori
    # So average density n_measured = N_particles / N_cells * cori
    N_particles = len(species)
    n_measured = N_particles * CORI / N_CELLS
    # Also compute from weights
    total_weight = np.sum(w)
    n_from_weight = total_weight * CORI / N_CELLS
    
    print(f"\n  ── 0th Moment: DENSITY ──")
    print(f"    Expected density n:           {N_DENSITY:.6f}")
    print(f"    Measured n (from count):       {n_measured:.6f}")
    print(f"    Measured n (from weights):     {n_from_weight:.6f}")
    print(f"    Particles per cell (expected): {EXPECTED_PPC:.1f}")
    print(f"    Particles per cell (measured): {N_particles / N_CELLS:.2f}")
    print(f"    Total particles (expected):    {EXPECTED_TOTAL_PER_SPECIES:,.0f}")
    print(f"    Total particles (measured):    {N_particles:,}")
    rel_err_n = abs(n_measured - N_DENSITY) / N_DENSITY * 100
    print(f"    ▸ Relative error:              {rel_err_n:.3f}%")
    status_n = "✓ PASS" if rel_err_n < 1.0 else "✗ FAIL"
    print(f"    ▸ Status: {status_n}")

    # --- 1st Moment: Macroscopic Velocity (Bulk Drift) ---
    # Expected: zero for all components (no bulk drift)
    px_mean = np.average(species['px'], weights=w)
    py_mean = np.average(species['py'], weights=w)
    pz_mean = np.average(species['pz'], weights=w)

    # Velocity = p / m
    vx_mean = px_mean / m
    vy_mean = py_mean / m
    vz_mean = pz_mean / m

    print(f"\n  ── 1st Moment: MACROSCOPIC VELOCITY (bulk drift) ──")
    print(f"    Expected: <v_x> = <v_y> = <v_z> = 0.0")
    print(f"    Measured <p_x>: {px_mean:+.6e}")
    print(f"    Measured <p_y>: {py_mean:+.6e}")
    print(f"    Measured <p_z>: {pz_mean:+.6e}")
    print(f"    Measured <v_x> = <p_x>/m: {vx_mean:+.6e}")
    print(f"    Measured <v_y> = <p_y>/m: {vy_mean:+.6e}")
    print(f"    Measured <v_z> = <p_z>/m: {vz_mean:+.6e}")
    
    # Compare to thermal speed to see if drift is negligible
    # v_th ~ sqrt(T/m)
    v_th_par = np.sqrt(expected_T_par / m)
    v_th_perp = np.sqrt(expected_T_perp / m)
    print(f"    Thermal speed v_th_perp = sqrt(T_perp/m): {v_th_perp:.6e}")
    print(f"    Thermal speed v_th_par  = sqrt(T_par/m):  {v_th_par:.6e}")
    print(f"    |<v_x>| / v_th_perp: {abs(vx_mean)/v_th_perp:.6e}")
    print(f"    |<v_y>| / v_th_perp: {abs(vy_mean)/v_th_perp:.6e}")
    print(f"    |<v_z>| / v_th_par:  {abs(vz_mean)/v_th_par:.6e}")

    drift_ok = all(abs(v) / v_th_par < 0.01 for v in [vx_mean, vy_mean, vz_mean])
    status_v = "✓ PASS" if drift_ok else "✗ FAIL"
    print(f"    ▸ Status (|<v>|/v_th < 1%): {status_v}")

    # --- 2nd Moment: Temperature ---
    # In the sampling: p_i = npt.p[i] + Z * S * beta * sqrt(T_i / m)
    # where Z ~ N(0,1), S has E[S^2] = 1 for kappa > 1.5
    # So Var(p_i) = beta^2 * T_i / m
    # => T_i = m * Var(p_i) / beta^2 = m * Var(p_i)  (beta=1)
    #
    # Note: PSC stores momentum (not velocity), and Var(p_i) = <p_i^2> - <p_i>^2
    
    px_var = np.average((species['px'] - px_mean)**2, weights=w)
    py_var = np.average((species['py'] - py_mean)**2, weights=w)
    pz_var = np.average((species['pz'] - pz_mean)**2, weights=w)

    T_x_measured = m * px_var / BETA_NORM**2
    T_y_measured = m * py_var / BETA_NORM**2
    T_z_measured = m * pz_var / BETA_NORM**2
    T_perp_measured = 0.5 * (T_x_measured + T_y_measured)

    print(f"\n  ── 2nd Moment: TEMPERATURE ──")
    print(f"    Formula: T_i = m * Var(p_i) / beta^2  (beta={BETA_NORM})")
    print(f"    Mass m = {m:.6f}")
    print()
    print(f"    {'Direction':<15} {'Expected T':<15} {'Measured T':<15} {'Rel.Err(%)':<12} {'Status'}")
    print(f"    {'─'*67}")
    
    temp_checks = [
        ("T_x (perp)",   expected_T_perp, T_x_measured),
        ("T_y (perp)",   expected_T_perp, T_y_measured),
        ("T_z (par)",    expected_T_par,  T_z_measured),
        ("T_perp (avg)", expected_T_perp, T_perp_measured),
    ]

    all_temp_ok = True
    for label, expected, measured in temp_checks:
        rel_err = abs(measured - expected) / expected * 100
        status = "✓" if rel_err < 2.0 else "✗"
        if rel_err >= 2.0:
            all_temp_ok = False
        print(f"    {label:<15} {expected:<15.6e} {measured:<15.6e} {rel_err:<12.4f} {status}")
    
    # Anisotropy ratio
    aniso_expected = expected_T_perp / expected_T_par
    aniso_measured = T_perp_measured / T_z_measured
    aniso_err = abs(aniso_measured - aniso_expected) / aniso_expected * 100

    print(f"\n    T_perp/T_par expected: {aniso_expected:.4f}")
    print(f"    T_perp/T_par measured: {aniso_measured:.4f}")
    print(f"    Anisotropy rel. error: {aniso_err:.4f}%")
    status_aniso = "✓ PASS" if aniso_err < 2.0 else "✗ FAIL"
    print(f"    ▸ Status: {status_aniso}")

    # --- Kurtosis (signature of kappa vs Maxwellian) ---
    # For a 1D kappa distribution with parameter kappa, the excess kurtosis is:
    # Kurt_excess = 6 / (2*kappa - 5)  for kappa > 5/2
    # For kappa=3: Kurt_excess = 6 / (6-5) = 6.0
    # For Maxwellian: Kurt_excess = 0
    print(f"\n  ── HIGHER MOMENT: KURTOSIS (kappa signature) ──")
    kappa_kurt_expected = 6.0 / (2.0 * KAPPA - 5.0) if KAPPA > 2.5 else float('inf')
    print(f"    Expected excess kurtosis for kappa={KAPPA}: {kappa_kurt_expected:.4f}")
    print(f"    (Maxwellian would give: 0.0)")

    for field, label in [('px', 'p_x (perp)'), ('py', 'p_y (perp)'), ('pz', 'p_z (par)')]:
        vals = species[field]
        mean = np.average(vals, weights=w)
        var = np.average((vals - mean)**2, weights=w)
        kurt = np.average((vals - mean)**4, weights=w) / var**2 - 3.0
        kurt_err = abs(kurt - kappa_kurt_expected) / kappa_kurt_expected * 100 if kappa_kurt_expected != float('inf') else float('inf')
        status_k = "✓" if kurt_err < 10.0 else "~"
        print(f"    {label:<15} excess kurtosis = {kurt:+.4f}  (err: {kurt_err:.1f}%)  {status_k}")

    print()
    return {
        'n': n_measured,
        'vx': vx_mean, 'vy': vy_mean, 'vz': vz_mean,
        'Tx': T_x_measured, 'Ty': T_y_measured, 'Tz': T_z_measured,
        'T_perp': T_perp_measured, 'T_par': T_z_measured,
        'anisotropy': aniso_measured,
        'all_temp_ok': all_temp_ok,
        'drift_ok': drift_ok,
        'density_ok': rel_err_n < 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def print_expected_parameters():
    """Print the expected simulation parameters for reference."""
    print("\n" + "═"*70)
    print("  EXPECTED PARAMETERS (from psc_temp_aniso.cxx)")
    print("═"*70)
    print(f"  B0 = vA/c              = {B0}")
    print(f"  n                      = {N_DENSITY}")
    print(f"  mass_ratio             = {MASS_RATIO}")
    print(f"  kappa                  = {KAPPA}")
    print(f"  beta_e_par             = {BETA_E_PAR}")
    print(f"  beta_i_par             = {BETA_I_PAR}")
    print(f"  Ti_perp/Ti_par         = {TI_PERP_OVER_TI_PAR}")
    print(f"  Te_perp/Te_par         = {TE_PERP_OVER_TE_PAR}")
    print()
    print(f"  Derived temperatures:")
    print(f"    Te_par  = beta_e_par * B0²/2 = {TE_PAR:.6f}")
    print(f"    Te_perp = Te_perp/Te_par * Te_par = {TE_PERP:.6f}")
    print(f"    Ti_par  = beta_i_par * B0²/2 = {TI_PAR:.6f}")
    print(f"    Ti_perp = Ti_perp/Ti_par * Ti_par = {TI_PERP:.6f}")
    print()
    print(f"  Species masses (in code units):")
    print(f"    m_ion      = {M_ION} (mass_ratio * Zi)")
    print(f"    m_electron = {M_ELECTRON}")
    print()
    print(f"  Grid: {N_GRID_Y} x {N_GRID_Z} (Y x Z), nicell = {NICELL}")
    print(f"  Total cells: {N_CELLS:,}")
    print(f"  Expected particles/species: ~{EXPECTED_TOTAL_PER_SPECIES:,.0f}")
    print(f"  Normalization: dimensionless (beta={BETA_NORM}, cori={CORI})")
    print()


def print_final_summary(ion_results, electron_results):
    """Print a final pass/fail summary table."""
    print("\n" + "═"*70)
    print("  FINAL VALIDATION SUMMARY")
    print("═"*70)
    print(f"  {'Check':<35} {'Ions':<15} {'Electrons':<15}")
    print(f"  {'─'*65}")
    
    checks = [
        ("Density (< 1% error)", 'density_ok'),
        ("Bulk drift (< 1% of v_th)", 'drift_ok'),
        ("Temperature (< 2% error)", 'all_temp_ok'),
    ]
    
    all_pass = True
    for label, key in checks:
        ion_s = "✓ PASS" if ion_results[key] else "✗ FAIL"
        ele_s = "✓ PASS" if electron_results[key] else "✗ FAIL"
        if not ion_results[key] or not electron_results[key]:
            all_pass = False
        print(f"  {label:<35} {ion_s:<15} {ele_s:<15}")
    
    print(f"  {'─'*65}")
    
    # Print measured vs expected summary
    print(f"\n  {'Parameter':<25} {'Expected':<15} {'Ion meas.':<15} {'Elec. meas.':<15}")
    print(f"  {'─'*70}")
    print(f"  {'n (density)':<25} {N_DENSITY:<15.6f} {ion_results['n']:<15.6f} {electron_results['n']:<15.6f}")
    print(f"  {'<v_x>':<25} {'0.0':<15} {ion_results['vx']:<+15.4e} {electron_results['vx']:<+15.4e}")
    print(f"  {'<v_y>':<25} {'0.0':<15} {ion_results['vy']:<+15.4e} {electron_results['vy']:<+15.4e}")
    print(f"  {'<v_z>':<25} {'0.0':<15} {ion_results['vz']:<+15.4e} {electron_results['vz']:<+15.4e}")
    print(f"  {'T_perp (ions)':<25} {TI_PERP:<15.6e} {ion_results['T_perp']:<15.6e} {'—':<15}")
    print(f"  {'T_par  (ions)':<25} {TI_PAR:<15.6e} {ion_results['T_par']:<15.6e} {'—':<15}")
    print(f"  {'T_perp (electrons)':<25} {TE_PERP:<15.6e} {'—':<15} {electron_results['T_perp']:<15.6e}")
    print(f"  {'T_par  (electrons)':<25} {TE_PAR:<15.6e} {'—':<15} {electron_results['T_par']:<15.6e}")
    print(f"  {'Ti_perp/Ti_par':<25} {TI_PERP_OVER_TI_PAR:<15.4f} {ion_results['anisotropy']:<15.4f} {'—':<15}")
    print(f"  {'Te_perp/Te_par':<25} {TE_PERP_OVER_TE_PAR:<15.4f} {'—':<15} {electron_results['anisotropy']:<15.4f}")

    print(f"\n  {'═'*70}")
    if all_pass:
        print("  ★ ALL CHECKS PASSED — Distribution moments are consistent! ★")
    else:
        print("  ✗ SOME CHECKS FAILED — Review the details above.")
    print(f"  {'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════
# 5. GRÁFICO RESUMEN VISUAL
# ═══════════════════════════════════════════════════════════════════════

def plot_validation_summary(ion_res, elec_res, outdir):
    """
    Gráfico resumen visual tipo tabla con colores PASS/FAIL.
    Temperaturas en mᵢvA² (iones) y mₑvA² (electrones).
    Velocidades en vA. Densidad en n/n₀.
    """
    from matplotlib.patches import FancyBboxPatch

    rows = [
        # (label, expected, ion_val, elec_val, unit, tol%)
        ("Density  n/n_0",
         N_DENSITY, ion_res['n'], elec_res['n'], "n_0", 1.0),
        ("Drift <vx>/vA",
         0.0, ion_res['vx']/VA, elec_res['vx']/VA, "vA", None),
        ("Drift <vy>/vA",
         0.0, ion_res['vy']/VA, elec_res['vy']/VA, "vA", None),
        ("Drift <vz>/vA",
         0.0, ion_res['vz']/VA, elec_res['vz']/VA, "vA", None),
        ("T_perp  [m_i vA^2]  (ions)",
         TI_PERP/(M_ION*VA**2), ion_res['T_perp']/(M_ION*VA**2), None, "m_i vA^2", 2.0),
        ("T_par   [m_i vA^2]  (ions)",
         TI_PAR/(M_ION*VA**2),  ion_res['T_par']/(M_ION*VA**2),  None, "m_i vA^2", 2.0),
        ("T_perp  [m_e vA^2]  (elec)",
         TE_PERP/(M_ELECTRON*VA**2), None, elec_res['T_perp']/(M_ELECTRON*VA**2), "m_e vA^2", 2.0),
        ("T_par   [m_e vA^2]  (elec)",
         TE_PAR/(M_ELECTRON*VA**2),  None, elec_res['T_par']/(M_ELECTRON*VA**2),  "m_e vA^2", 2.0),
        ("T_perp/T_par  (ions)",
         TI_PERP_OVER_TI_PAR, ion_res['anisotropy'], None, "", 2.0),
        ("T_perp/T_par  (elec)",
         TE_PERP_OVER_TE_PAR, None, elec_res['anisotropy'], "", 2.0),
    ]

    def rel_err(exp, meas):
        if meas is None or exp is None: return None
        if exp == 0.0: return abs(meas) * 100
        return abs(meas - exp) / abs(exp) * 100

    def pcolor(err, tol):
        if err is None: return '#555555'
        if err <= tol:      return '#27ae60'
        if err <= tol * 3:  return '#f39c12'
        return '#c0392b'

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(14, n_rows * 0.72 + 2.8))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.97,
             f'Validation Report — PSC  (mi/me={int(MASS_RATIO)}, '
             f'kappa={KAPPA}, beta_i||={BETA_I_PAR:.1f}, T_perp/T_par={TI_PERP_OVER_TI_PAR})',
             ha='center', va='top', fontsize=14, fontweight='bold', color='black')
    fig.text(0.5, 0.93,
             f'd_i={DI:.1f} cells  |  w_ci={OMEGA_CI:.5f} rad/t  |  '
             f'vA={VA:.3f}  |  Grid: {N_GRID_Y/DI:.0f}x{N_GRID_Z/DI:.0f} d_i',
             ha='center', va='top', fontsize=10, color='#444444')

    col_x = [0.01, 0.26, 0.40, 0.51, 0.63, 0.74, 0.86]
    hdrs  = ['Parameter','Expected','Ions','Err% (i)','Electrons','Err% (e)','Status']
    hy    = 0.88
    for cx, h in zip(col_x, hdrs):
        fig.text(cx, hy, h, transform=fig.transFigure,
                 fontsize=10.5, fontweight='bold', color='#111111', va='top')
    fig.add_artist(plt.Line2D([0.01,0.99],[hy-0.026, hy-0.026],
                               transform=fig.transFigure, color='#cccccc', lw=1.0))

    row_h = (0.85 - 0.07) / (n_rows + 0.5)
    for i, (param, expected, iv, ev, unit, tol) in enumerate(rows):
        y = hy - 0.040 - (i+1)*row_h
        fig.add_artist(FancyBboxPatch(
            (0.005, y-row_h*0.48), 0.99, row_h*0.96,
            boxstyle="round,pad=0.003",
            facecolor='#f8f9fa' if i%2==0 else '#e9ecef',
            edgecolor='none', transform=fig.transFigure, zorder=0))

        ei = rel_err(expected, iv)
        ee = rel_err(expected, ev)

        if tol is not None:
            errs = [e for e in [ei,ee] if e is not None]
            w = max(errs) if errs else None
            if w is None:    st, sc = '—', '#555555'
            elif w <= tol:   st, sc = 'PASS', '#27ae60'
            elif w <= tol*3: st, sc = 'WARN', '#e67e22'
            else:            st, sc = 'FAIL', '#c0392b'
        else:
            vals = [v for v in [iv,ev] if v is not None]
            wa = max(abs(v) for v in vals) if vals else 0
            if wa < 0.01:   st, sc = 'PASS', '#27ae60'
            elif wa < 0.1:  st, sc = 'WARN', '#e67e22'
            else:           st, sc = 'FAIL', '#c0392b'

        def fmt(v):
            if v is None: return '—'
            if abs(v) < 0.001 or abs(v) >= 1000: return f'{v:.4e}'
            return f'{v:.6f}'
        def ferr(e):
            return '—' if e is None else f'{e:.3f}%'

        cells = [
            (col_x[0], param,                '#000000'),
            (col_x[1], f'{fmt(expected)} {unit}','#555555'),
            (col_x[2], fmt(iv),              '#2980b9'),
            (col_x[3], ferr(ei),             pcolor(ei,tol) if tol else '#555555'),
            (col_x[4], fmt(ev),              '#c0392b'),
            (col_x[5], ferr(ee),             pcolor(ee,tol) if tol else '#555555'),
            (col_x[6], st,                   sc),
        ]
        for cx, txt, col in cells:
            fig.text(cx, y, txt, transform=fig.transFigure,
                     fontsize=9.5, color=col, va='center')

    for xl, col, lbl in [(0.01,'#27ae60','PASS (<2%)'),
                          (0.18,'#e67e22','WARN (2–6%)'),
                          (0.35,'#c0392b','FAIL (>6%)')]:
        fig.text(xl, 0.025, lbl, transform=fig.transFigure,
                 fontsize=9.5, color=col, fontweight='bold', va='bottom')

    out_path = os.path.join(outdir, "validation_summary.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  [Visual report] Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__),
                                "..", "build", "src", "prt.000000000.h5")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Usage: python validate_moments.py [path_to_prt_file.h5]")
        sys.exit(1)

    print_expected_parameters()

    data = load_particles(filepath)
    ions, electrons = separate_species(data)

    ion_results = compute_moments(
        ions, "IONS",
        expected_T_perp=TI_PERP,   # 0.175
        expected_T_par=TI_PAR,     # 0.05
        expected_m=M_ION           # 64.0
    )
    electron_results = compute_moments(
        electrons, "ELECTRONS",
        expected_T_perp=TE_PERP,   # 0.005
        expected_T_par=TE_PAR,     # 0.005
        expected_m=M_ELECTRON      # 1.0
    )

    print_final_summary(ion_results, electron_results)

    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), "validation_plots")
    os.makedirs(outdir, exist_ok=True)
    plot_validation_summary(ion_results, electron_results, outdir)


if __name__ == "__main__":
    main()

