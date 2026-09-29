"""Shared pressure projections, particle kinematics and reference thresholds.

Everything that more than one analysis script needs to compute *the same way*
lives here, so that two figures of the thesis can never disagree because two
scripts re-derived a formula with a different sign or convention.

Array layout: 2D maps are (Nz, Ny) -- axis 0 is z (along B0), axis 1 is y.
"""

import numpy as np


# ── Reference anisotropy curves ─────────────────────────────────────────────

def mirror_threshold(beta_parallel):
    """Cold-electron, bi-Maxwellian mirror reference in parallel-beta coordinates.

    beta_perp * (A - 1) = 1 with beta_perp = A * beta_parallel.
    This is not a finite-growth contour or a general hot-electron/Kappa threshold.
    See Hellinger (2007), doi:10.1063/1.2768310, Eq. (16).
    """
    beta = np.asarray(beta_parallel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 0, 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / beta)), np.nan)


def mirror_criterion(beta_i_parallel, anisotropy_i, beta_e_parallel, anisotropy_e):
    """Hellinger (2007) mirror criterion Gamma for bi-Maxwellian protons and electrons.

    Gamma = sum_s beta_perp,s (A_s - 1) - 1
            - (sum_s q_s n_s A_s)^2 / (2 sum_s (q_s n_s)^2 / beta_par,s),

    unstable for Gamma > 0 (Hellinger 2007, Phys. Plasmas 14, 082105, Eq. 16;
    long-wavelength, marginal-stability limit of linear kinetic theory). For a
    quasi-neutral proton-electron plasma q_i n_i = -q_e n_e, so the last term
    is (A_i - A_e)^2 / (2 (1/beta_par,i + 1/beta_par,e)).
    """
    bi, ai, be, ae = (np.asarray(v, dtype=float) for v in
                      (beta_i_parallel, anisotropy_i, beta_e_parallel, anisotropy_e))
    with np.errstate(divide="ignore", invalid="ignore"):
        charge_term = (ai - ae) ** 2 / (2.0 * (1.0 / bi + 1.0 / be))
        charge_term = np.where(be > 0, charge_term, 0.0)
    return bi * ai * (ai - 1.0) + be * ae * (ae - 1.0) - 1.0 - charge_term


def mirror_threshold_electrons(beta_parallel, beta_e_parallel, anisotropy_e=1.0):
    """Ion anisotropy A_i at which mirror_criterion vanishes, for given electrons.

    Why: mirror_threshold assumes cold electrons, but in these runs the
    electrons are heated numerically from beta_e = 1 to ~8 and need not stay
    isotropic. Both enter the threshold: hot isotropic electrons raise it
    slightly through the charge term (< 1 % at beta_i = 5), while an electron
    anisotropy adds beta_perp,e (A_e - 1) to the drive, which at beta_e ~ 8
    shifts the ion threshold far more than any other term. The ion trajectory
    must therefore be compared with the threshold of the electrons actually
    present at that time, not with the initial or the cold-electron one.

    Gamma = 0 is a quadratic in A_i; the positive root is returned. For
    beta_e -> 0 it reduces to mirror_threshold. Bi-Maxwellian theory: for
    bi-kappa ions it is a reference, not the kappa threshold.
    """
    b, be, ae = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in
                                       (beta_parallel, beta_e_parallel, anisotropy_e)))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = np.where(be > 0, b * be / (2.0 * (b + be)), 0.0)
        a = b - inv_d
        linear = -b + 2.0 * ae * inv_d
        const = be * ae * (ae - 1.0) - 1.0 - ae ** 2 * inv_d
        disc = linear ** 2 - 4.0 * a * const
        root = (-linear + np.sqrt(disc)) / (2.0 * a)
        return np.where((b > 0) & (disc >= 0) & (a > 0), root, np.nan)


def firehose_threshold(beta_parallel):
    """Fluid (CGL) parallel-firehose marginal curve A = 1 - 2/beta_parallel.

    Marginal stability of the k -> 0 limit; kinetic contours of finite growth
    sit at lower A. Undefined (NaN) for beta_parallel <= 2.
    """
    beta = np.asarray(beta_parallel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 2.0, 1.0 - 2.0 / beta, np.nan)


#: (S, alpha) of the empirical whistler contour A_e - 1 = S / beta_e_par^alpha
#: used across this repository (see src/SIMULACIONES_ANISOTROPIA.md). This is a
#: fitted contour of constant maximum growth rate in the form of Gary & Wang
#: (1996, JGR 101, 10749); quote it together with the growth level of the
#: source the constants are taken from.
WHISTLER_CONTOUR = (0.21, 0.6)


def whistler_threshold(beta_parallel, contour=WHISTLER_CONTOUR):
    """Empirical electron whistler-anisotropy contour A_e = 1 + S/beta^alpha."""
    s, alpha = contour
    beta = np.asarray(beta_parallel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 0, 1.0 + s / beta**alpha, np.nan)


def reference_threshold(instability: str, beta_parallel):
    """Reference curve for the declared instability family, with its label."""
    if instability == "mirror":
        return mirror_threshold(beta_parallel), "mirror reference (cold electrons)"
    if instability == "firehose":
        return firehose_threshold(beta_parallel), r"parallel firehose (CGL) $1-2/\beta_\parallel$"
    s, alpha = WHISTLER_CONTOUR
    return (whistler_threshold(beta_parallel),
            rf"whistler contour $1+{s:g}/\beta_{{e\parallel}}^{{{alpha:g}}}$")


# ── Pressure tensors ────────────────────────────────────────────────────────

def field_aligned_pressures(Pxx, Pyy, Pzz, Pxy, Pyz, Pzx, Bx, By, Bz):
    """Project thermal pressure onto local B; the direction is undefined at B=0."""
    B2 = Bx**2 + By**2 + Bz**2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_B = np.where(B2 > 0, 1.0 / np.sqrt(B2), np.nan)
    bx, by, bz = Bx * inv_B, By * inv_B, Bz * inv_B
    p_par = (
        Pxx * bx**2 + Pyy * by**2 + Pzz * bz**2
        + 2.0 * Pxy * bx * by + 2.0 * Pyz * by * bz
        + 2.0 * Pzx * bz * bx
    )
    return p_par, 0.5 * (Pxx + Pyy + Pzz - p_par), B2


def central_pressure_tensor(mom: dict, suffix: str, mass: float,
                            n_min: float = 1e-12) -> dict:
    """Thermal pressure from PSC's raw ``all_1st`` moments of one species.

    PSC deposits the momentum density p_a = n m <u_a> and the raw second
    moment t_ab = n m <u_a v_b>. The thermal (central) part removes the bulk
    flow, P_ab = t_ab - p_a p_b / (n m); using t_ab directly contaminates the
    temperature wherever a bulk drift develops. ``rho`` is a charge density,
    so electrons carry a negative sign.
    """
    n = np.abs(np.asarray(mom[f"rho_{suffix}"], dtype=float))
    safe_n = np.where(n > n_min, n, np.nan)
    px, py, pz = (np.asarray(mom[f"p{a}_{suffix}"], dtype=float) for a in "xyz")
    nm = safe_n * mass
    return {
        "n": n,
        "Pxx": mom[f"txx_{suffix}"] - px * px / nm,
        "Pyy": mom[f"tyy_{suffix}"] - py * py / nm,
        "Pzz": mom[f"tzz_{suffix}"] - pz * pz / nm,
        "Pxy": mom[f"txy_{suffix}"] - px * py / nm,
        "Pyz": mom[f"tyz_{suffix}"] - py * pz / nm,
        "Pzx": mom[f"tzx_{suffix}"] - pz * px / nm,
        "Vx": px / nm, "Vy": py / nm, "Vz": pz / nm,
    }


def diamagnetic_current_x(p_perp, by, bz, dy: float, dz: float):
    """Out-of-plane component of J_dia = B x grad(P_perp) / B^2 on a (Nz, Ny) map.

    For a yz simulation plane (d/dx = 0):
        (B x grad P)_x = B_y dP/dz - B_z dP/dy.
    ``dy``/``dz`` are the cell sizes in code lengths (d_e), so the current
    comes out in code units (e n0 c) instead of "per cell". Periodic
    central differences, matching the periodic box.
    """
    p = np.asarray(p_perp, dtype=float)
    dpdz = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2.0 * dz)
    dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * dy)
    b2 = np.asarray(by, dtype=float) ** 2 + np.asarray(bz, dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b2 > 0, (by * dpdz - bz * dpdy) / b2, np.nan)


# ── Particle kinematics ─────────────────────────────────────────────────────

def velocity_from_u(ux, uy, uz):
    """PSC stores u = gamma v (c = 1). Return (vx, vy, vz, gamma)."""
    ux, uy, uz = (np.asarray(a, dtype=float) for a in (ux, uy, uz))
    gamma = np.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
    return ux / gamma, uy / gamma, uz / gamma, gamma


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0 else float("nan")


def central_uv(ua, vb, weights):
    """Central mixed moment <u_a v_b> - <u_a><v_b>, PSC's pressure convention.

    PSC's moment deposit uses t_ab = n m <u_a v_b>; building particle
    temperatures the same way makes the particle and moment estimators
    comparable, and it is the relativistically consistent kinetic pressure
    (it reduces to the usual variance of v for |u| << 1).
    """
    ua = np.asarray(ua, dtype=float)
    vb = np.asarray(vb, dtype=float)
    return (weighted_mean(ua * vb, weights)
            - weighted_mean(ua, weights) * weighted_mean(vb, weights))
