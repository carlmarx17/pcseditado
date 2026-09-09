"""Shared nonrelativistic pressure projections and reference thresholds."""

import numpy as np


def mirror_threshold(beta_parallel):
    """Cold-electron, bi-Maxwellian mirror reference in parallel-beta coordinates.

    beta_perp * (A - 1) = 1 with beta_perp = A * beta_parallel.
    This is not a finite-growth contour or a general hot-electron/Kappa threshold.
    See Hellinger (2007), doi:10.1063/1.2768310, Eq. (16).
    """
    beta = np.asarray(beta_parallel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 0, 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / beta)), np.nan)


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
