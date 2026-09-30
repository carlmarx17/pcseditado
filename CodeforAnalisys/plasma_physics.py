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


# ── Model velocity distributions ────────────────────────────────────────────
#
# One definition for every VDF figure and fit. The PSC loader draws the kappa
# population as u_i = Z_i S sigma_i with Z_i ~ N(0, 1) and a single
# S = sqrt((kappa - 3/2) / Y), Y ~ Gamma(kappa - 1/2), shared by the three
# components: a multivariate Student t with nu = 2 kappa - 1 degrees of freedom
# and E[S^2] = 1, so each component has exactly the variance sigma_i^2 of the
# Maxwellian loaded at the same temperature. Its 3-D density is the bi-kappa
#   f = C [1 + v_par^2/((2k-3) s_par^2) + v_perp^2/((2k-3) s_perp^2)]^-(k+1),
# every marginal is again a Student t with the same nu, and kappa -> inf gives
# the bi-Maxwellian. sigma is the standard deviation of one component, not the
# kappa "thermal speed" theta = sigma sqrt((2k-3)/k): a kappa curve built with
# theta = sigma has a variance 2k/(2k-3) times the measured one (3x for
# kappa = 3), the classic normalisation error. A kappa curve is also only
# comparable with data of the same variance, i.e. the temperature measured at
# that time, not the initial one.

def _student(kappa):
    """nu of the Student t, or None for the Maxwellian limit."""
    if kappa is None or not np.isfinite(kappa):
        return None
    if kappa <= 1.5:
        raise ValueError("A finite-variance kappa distribution needs kappa > 3/2")
    return 2.0 * float(kappa) - 1.0


def kappa_marginal_pdf(v, sigma, kappa=None):
    """1-D density of one velocity component with standard deviation ``sigma``."""
    from scipy.special import gammaln
    v = np.asarray(v, dtype=float) / sigma
    nu = _student(kappa)
    if nu is None:
        return np.exp(-0.5 * v ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    s = np.sqrt((nu - 2.0) / nu)                     # unit-variance scale
    logc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return np.exp(logc - (nu + 1) / 2 * np.log1p((v / s) ** 2 / nu)) / (s * sigma)


def kappa_marginal_cdf(v, sigma, kappa=None):
    """CDF of kappa_marginal_pdf (exact Student t / normal)."""
    from scipy import stats
    v = np.asarray(v, dtype=float) / sigma
    nu = _student(kappa)
    if nu is None:
        return stats.norm.cdf(v)
    return stats.t.cdf(v / np.sqrt((nu - 2.0) / nu), df=nu)


def kappa_speed_pdf_2d(v_perp, sigma, kappa=None):
    """Density of |v_perp| for two components of standard deviation ``sigma`` each.

    Rayleigh (v/sigma^2) exp(-v^2/2 sigma^2) for the Maxwellian; for the kappa,
    the speed of a 2-D Student t: (v/s^2) (1 + v^2/(nu s^2))^-(nu+2)/2 with
    s^2 = sigma^2 (nu - 2)/nu.
    """
    v = np.asarray(v_perp, dtype=float)
    nu = _student(kappa)
    if nu is None:
        return v / sigma ** 2 * np.exp(-0.5 * (v / sigma) ** 2)
    s2 = sigma ** 2 * (nu - 2.0) / nu
    return v / s2 * (1.0 + v ** 2 / (nu * s2)) ** (-(nu + 2.0) / 2.0)


def bi_distribution_3d(v_par, v_perp, sigma_par, sigma_perp, kappa=None):
    """Gyrotropic f(v_par, v_perp) per d^3v, normalised to 1 (bi-kappa or bi-Maxwellian)."""
    from scipy.special import gammaln
    v_par = np.asarray(v_par, dtype=float)
    v_perp = np.asarray(v_perp, dtype=float)
    nu = _student(kappa)
    if nu is None:
        norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma_perp ** 2 * sigma_par)
        return norm * np.exp(-0.5 * (v_par / sigma_par) ** 2 - 0.5 * (v_perp / sigma_perp) ** 2)
    k = 0.5 * (nu + 1.0)
    a = 2.0 * k - 3.0
    logc = (gammaln(k + 1.0) - gammaln(k - 0.5) - 1.5 * np.log(np.pi * a)
            - np.log(sigma_perp ** 2 * sigma_par))
    q = v_par ** 2 / (a * sigma_par ** 2) + v_perp ** 2 / (a * sigma_perp ** 2)
    return np.exp(logc - (k + 1.0) * np.log1p(q))


def reduced_distribution_2d(v1, v2, sigma1, sigma2, kappa=None):
    """Density of two velocity components (the third integrated out), per dv1 dv2."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    nu = _student(kappa)
    if nu is None:
        return (np.exp(-0.5 * (v1 / sigma1) ** 2 - 0.5 * (v2 / sigma2) ** 2)
                / (2.0 * np.pi * sigma1 * sigma2))
    f = (nu - 2.0) / nu
    s1, s2 = sigma1 * np.sqrt(f), sigma2 * np.sqrt(f)
    q = (v1 / s1) ** 2 + (v2 / s2) ** 2
    return (1.0 + q / nu) ** (-(nu + 2.0) / 2.0) / (2.0 * np.pi * s1 * s2)


def _kappa_point(x, w, kappa_max):
    """1/kappa maximising the likelihood of the unit-variance sample x (weights w)."""
    def loglike_inv(inv):
        kappa = None if inv <= 1.0 / kappa_max else 1.0 / inv
        return float(np.sum(w * np.log(np.maximum(kappa_marginal_pdf(x, 1.0, kappa), 1e-300))))

    coarse = np.linspace(0.0, 1.0 / 1.55, 91)
    ll = np.array([loglike_inv(i) for i in coarse])
    best = int(np.argmax(ll))
    step = coarse[1] - coarse[0]
    fine = np.linspace(max(coarse[best] - 1.5 * step, 0.0), min(coarse[best] + 1.5 * step, coarse[-1]), 41)
    ll_fine = np.array([loglike_inv(i) for i in fine])
    return float(fine[int(np.argmax(ll_fine))])


def kappa_mle(values, weights=None, kappa_max: float = 200.0, max_samples: int = 400_000,
              n_boot: int = 40, boot_size: int = 20_000, rng=None) -> dict:
    """Maximum-likelihood kappa of a 1-D velocity sample whose variance is fixed.

    The variance is the measured one (the temperature), so only the tail shape
    is estimated: the likelihood of kappa_marginal_pdf over 1/kappa (0 is the
    Maxwellian, reached smoothly), maximised on a grid and refined. The
    uncertainty is a bootstrap of the whole estimator (variance included):
    the variance and the tail estimate of a heavy-tailed sample are
    correlated, and an interval that holds the variance fixed covers the true
    kappa only ~50 % of the time instead of 68 % (Monte Carlo, kappa = 3,
    n = 7200). The bootstrap runs on resamples of at most ``boot_size``
    points and is scaled to the sample behind the point estimate by
    sqrt(m / N), N = N_eff or ``max_samples`` when the sample was thinned.
    Returns kappa (inf when no tail is measurable), its 68 % interval, sigma
    and N_eff.
    """
    rng = rng or np.random.default_rng(20260930)
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size < 100:
        return {"kappa": np.nan, "kappa_lo": np.nan, "kappa_hi": np.nan,
                "sigma": np.nan, "n_eff": float(v.size)}
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    n_point = n_eff                    # sample size behind the point estimate
    if v.size > max_samples:
        pick = rng.choice(v.size, max_samples, replace=False, p=w / w.sum())
        v, w = v[pick], np.ones(max_samples)
        n_point = min(n_eff, float(max_samples))
    mean = np.sum(w * v) / w.sum()
    sigma = float(np.sqrt(np.sum(w * (v - mean) ** 2) / w.sum()))
    inv_hat = _kappa_point((v - mean) / sigma, w / w.sum(), kappa_max)

    m = min(boot_size, v.size)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(v.size, m, replace=True, p=w / w.sum())
        vb = v[idx]
        sb = np.std(vb)
        boot.append(_kappa_point((vb - vb.mean()) / sb, np.full(m, 1.0 / m), kappa_max))
    se = float(np.std(boot, ddof=1)) * np.sqrt(m / n_point) if n_boot > 1 else float("nan")
    to_kappa = lambda inv: float("inf") if inv <= 1.0 / kappa_max else 1.0 / inv
    return {"kappa": to_kappa(inv_hat),
            "kappa_lo": to_kappa(min(inv_hat + se, 1.0 / 1.55)),
            "kappa_hi": to_kappa(inv_hat - se),
            "sigma": sigma, "n_eff": n_eff}
