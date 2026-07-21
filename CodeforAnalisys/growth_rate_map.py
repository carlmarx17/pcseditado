#!/usr/bin/env python3
"""Growth-rate map in the (k_parallel, k_perp) plane from PIC field snapshots.

This diagnostic measures gamma(k_parallel, k_perp) directly from the simulation
by fitting the exponential growth of magnetic spectral power in each Fourier
cell during a selected linear-time window. Unlike radial E(k,t) binning, each
pixel remains a distinct wavevector geometry.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def _fold_positive(axis_vals: np.ndarray, arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Fold a shifted FFT axis onto |k| by adding +/- k power."""
    n = len(axis_vals)
    center = n // 2
    pos_vals = axis_vals[center:]
    arr_pos = np.take(arr, range(center, n), axis=axis).copy()
    arr_neg = np.flip(np.take(arr, range(0, center), axis=axis), axis=axis)

    n_match = min(arr_pos.shape[axis] - 1, arr_neg.shape[axis])
    if n_match > 0:
        idx_pos = [slice(None)] * arr_pos.ndim
        idx_neg = [slice(None)] * arr_neg.ndim
        idx_pos[axis] = slice(1, 1 + n_match)
        idx_neg[axis] = slice(0, n_match)
        arr_pos[tuple(idx_pos)] += arr_neg[tuple(idx_neg)]
    return pos_vals, arr_pos


def compute_growth_rate_map(
    field_series: np.ndarray,
    time_oci: np.ndarray,
    spacing: tuple[float, float],
    axes: tuple[str, str],
    parallel_axis: str = "z",
    kpar_max: float | None = None,
    kperp_max: float | None = None,
    min_rvalue: float = 0.0,
    fit_frac: tuple[float, float] = (0.1, 0.6),
    fold_negative_k: bool = False,
) -> dict:
    """Return gamma(k_parallel, k_perp) fitted from the linear growth phase.

    ``field_series`` has shape ``(n_components, nt, n0, n1)``. Time is
    ``Omega_ci t`` and spacing is measured in ``d_i``. The returned k axes are
    therefore in ``k d_i`` and gamma is in ``Omega_ci``.

    Modes separate by geometry: a gamma peak on the k_parallel axis
    (k_perp ~ 0) indicates a parallel mode (EMIC / parallel firehose), while
    an off-axis, oblique peak (~45 deg) indicates a mirror / oblique firehose
    mode.

    ``k_perp`` has no physically preferred sign for a gyrotropic background
    (flipping it is just a spatial reflection perpendicular to B0), so it is
    always folded onto |k_perp|. ``k_parallel`` is kept as a full signed axis
    by default (``fold_negative_k=False``): folding it too would sum the four
    quadrants (+k_par,+k_perp), (-k_par,+k_perp), (+k_par,-k_perp), and
    (-k_par,-k_perp) into a single pixel, silently conflating potentially
    distinct oblique branches (forward vs. backward propagation along B0)
    into one blended growth rate. Pass ``fold_negative_k=True`` to restore
    that first-quadrant-only view.
    """
    fields = np.asarray(field_series, dtype=float)
    times = np.asarray(time_oci, dtype=float)
    if fields.ndim != 4:
        raise ValueError("field_series must be (components, time, axis0, axis1)")
    if fields.shape[1] != len(times):
        raise ValueError("time axis mismatch")
    if len(times) < 4:
        raise ValueError("At least four snapshots are required")
    if axes[0] != parallel_axis and axes[1] != parallel_axis:
        raise ValueError(f"Plane axes {axes} do not contain parallel axis {parallel_axis!r}")
    if not (0.0 <= fit_frac[0] < fit_frac[1] <= 1.0):
        raise ValueError("fit_frac must satisfy 0 <= lo < hi <= 1")

    _, nt, n0, n1 = fields.shape
    fields = fields - np.mean(fields, axis=(2, 3), keepdims=True)

    window = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
    window_power = float(np.mean(window**2))
    if window_power <= 0:
        window = np.ones((n0, n1), dtype=float)
        window_power = 1.0

    k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi

    energy = np.zeros((nt, n0, n1), dtype=float)
    for component in fields:
        transformed = np.fft.fftshift(
            np.fft.fft2(component * window[None, :, :], axes=(1, 2)),
            axes=(1, 2),
        )
        energy += np.abs(transformed) ** 2 / (window_power * n0 * n1)

    if axes[0] == parallel_axis:
        kpar_axis, kperp_axis = k0, k1
        par_dim, perp_dim = 1, 2
    else:
        kpar_axis, kperp_axis = k1, k0
        par_dim, perp_dim = 2, 1

    energy = np.moveaxis(energy, (par_dim, perp_dim), (1, 2))
    if fold_negative_k:
        kpar_pos, energy = _fold_positive(kpar_axis, energy, axis=1)
    else:
        kpar_pos = kpar_axis
    kperp_pos, energy = _fold_positive(kperp_axis, energy, axis=2)

    if kpar_max is not None:
        keep = np.abs(kpar_pos) <= kpar_max
        kpar_pos = kpar_pos[keep]
        energy = energy[:, keep, :]
    if kperp_max is not None:
        keep = kperp_pos <= kperp_max
        kperp_pos = kperp_pos[keep]
        energy = energy[:, :, keep]

    t0, t1 = float(times[0]), float(times[-1])
    fit_lo = t0 + fit_frac[0] * (t1 - t0)
    fit_hi = t0 + fit_frac[1] * (t1 - t0)
    fit_mask = (times >= fit_lo) & (times <= fit_hi)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("Fit window contains fewer than three snapshots")
    tfit = times[fit_mask]
    design = np.vstack([tfit, np.ones_like(tfit)]).T

    n_par, n_perp = len(kpar_pos), len(kperp_pos)
    gamma = np.full((n_par, n_perp), np.nan)
    rvalue = np.full((n_par, n_perp), np.nan)
    final_power = np.full((n_par, n_perp), np.nan)

    for i in range(n_par):
        for j in range(n_perp):
            series = energy[:, i, j]
            final_power[i, j] = series[-1]
            y = series[fit_mask]
            if np.any(y <= 0.0) or not np.any(np.isfinite(y)) or np.allclose(y, y[0]):
                continue
            log_amp = 0.5 * np.log(y)
            (slope, intercept), *_ = np.linalg.lstsq(design, log_amp, rcond=None)
            fitted = design @ np.array([slope, intercept])
            ss_res = float(np.sum((log_amp - fitted) ** 2))
            ss_tot = float(np.sum((log_amp - np.mean(log_amp)) ** 2))
            r = np.sqrt(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
            gamma[i, j] = slope
            rvalue[i, j] = r

    return {
        "gamma": gamma,
        "rvalue": rvalue,
        "final_power": final_power,
        "kpar": kpar_pos,
        "kperp": kperp_pos,
        "fit_window": (fit_lo, fit_hi),
        "min_rvalue": min_rvalue,
        "fold_negative_k": fold_negative_k,
        "diagnostics": {
            "nt": int(nt),
            "n_axis0": int(n0),
            "n_axis1": int(n1),
            "axes": axes,
            "spacing": tuple(float(v) for v in spacing),
            "parallel_axis": parallel_axis,
            "fold_negative_k": fold_negative_k,
            "raw_kpar_modes": int(len(kpar_axis)),
            "raw_kperp_modes": int(len(kperp_axis)),
            "folded_kpar_modes": int(len(kpar_pos)),
            "folded_kperp_modes": int(len(kperp_pos)),
            "dkpar": float(kpar_pos[1] - kpar_pos[0]) if len(kpar_pos) > 1 else float("nan"),
            "dkperp": float(kperp_pos[1] - kperp_pos[0]) if len(kperp_pos) > 1 else float("nan"),
            "fit_points": int(np.count_nonzero(fit_mask)),
        },
    }


def _display_crop(result: dict, display_kpar_max: float | None, display_kperp_max: float | None):
    kpar = np.asarray(result["kpar"], dtype=float)
    kperp = np.asarray(result["kperp"], dtype=float)
    gamma = np.asarray(result["gamma"], dtype=float)
    rvalue = np.asarray(result["rvalue"], dtype=float)
    final_power = np.asarray(result["final_power"], dtype=float)

    keep_par = np.ones_like(kpar, dtype=bool)
    keep_perp = np.ones_like(kperp, dtype=bool)
    if display_kpar_max is not None:
        keep_par &= np.abs(kpar) <= display_kpar_max
    if display_kperp_max is not None:
        keep_perp &= kperp <= display_kperp_max
    return (
        kpar[keep_par],
        kperp[keep_perp],
        gamma[np.ix_(keep_par, keep_perp)],
        rvalue[np.ix_(keep_par, keep_perp)],
        final_power[np.ix_(keep_par, keep_perp)],
    )


def plot_growth_rate_map(
    result: dict,
    output: Path,
    component: str,
    mark_peak: bool = True,
    display_kpar_max: float | None = None,
    display_kperp_max: float | None = None,
    shading: str = "auto",
    contour_count: int = 6,
    angle_step: int = 15,
):
    kpar, kperp, gamma, rvalue, final_power = _display_crop(
        result, display_kpar_max, display_kperp_max
    )
    minr = float(result["min_rvalue"])

    floor = np.nanmax(final_power) * 1e-3 if np.any(np.isfinite(final_power)) else np.inf
    mask = ~np.isfinite(gamma) | (gamma <= 0.0) | (final_power < floor)
    if minr > 0:
        mask |= rvalue < minr
    display = np.ma.array(gamma, mask=mask)
    gmax = float(np.nanmax(gamma[~mask])) if np.any(~mask) else 1.0

    fig, axis = plt.subplots(figsize=(8.6, 7.2))
    image = axis.pcolormesh(
        kpar, kperp, display.T, shading=shading, cmap="inferno", vmin=0.0, vmax=gmax
    )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"growth rate $\gamma\ [\Omega_{ci}]$")

    if contour_count > 0 and np.any(~mask) and len(kpar) > 1 and len(kperp) > 1:
        levels = np.linspace(0.0, gmax, contour_count + 2)[1:-1]
        if len(levels):
            axis.contour(
                kpar,
                kperp,
                display.T,
                levels=levels,
                colors="white",
                linewidths=0.55,
                alpha=0.45,
            )

    if len(kpar) and len(kperp):
        angles = range(max(angle_step, 1), 90, max(angle_step, 1))
        kpar_extent = float(np.max(np.abs(kpar)))
        signs = (-1, 1) if kpar.min() < 0 else (1,)
        for angle in angles:
            main_angle = angle in {30, 45, 60}
            for sign in signs:
                kk = np.linspace(0.0, sign * kpar_extent, 32)
                yy = np.abs(kk) * np.tan(np.radians(angle))
                axis.plot(
                    kk,
                    yy,
                    color="cyan",
                    lw=0.9 if main_angle else 0.45,
                    ls=":",
                    alpha=0.65 if main_angle else 0.35,
                )
            yy_label = kpar_extent * np.tan(np.radians(angle))
            if yy_label <= kperp.max():
                axis.text(
                    kpar_extent * 0.92,
                    yy_label * 0.92,
                    f"{angle} deg",
                    color="cyan",
                    fontsize=8 if main_angle else 7,
                    alpha=0.85 if main_angle else 0.55,
                )

    if mark_peak and np.any(~mask):
        candidate = np.where(~mask, gamma, np.nan)
        max_gamma = float(np.nanmax(candidate))
        near_peak = np.isfinite(candidate) & (candidate >= max_gamma - max(1e-4, 0.01 * abs(max_gamma)))
        # Window side lobes can grow with the same slope as the true mode in a
        # clean synthetic signal. Break near-ties with final power so the marker
        # lands on the dominant physical Fourier cell, not a low-power leakage cell.
        peak_score = np.where(near_peak, final_power, -np.inf)
        peak_i, peak_j = np.unravel_index(int(np.argmax(peak_score)), peak_score.shape)
        # theta ~ 0 (on the k_parallel axis) => parallel mode (EMIC / parallel
        # firehose); theta ~ 45 deg, off-axis => mirror / oblique firehose.
        theta = np.degrees(np.arctan2(abs(kperp[peak_j]), abs(kpar[peak_i])))
        axis.plot(
            kpar[peak_i],
            kperp[peak_j],
            "x",
            color="lime",
            ms=14,
            mew=3,
            label=(
                fr"peak $\gamma$={gamma[peak_i, peak_j]:.3f} at "
                fr"$k_\parallel d_i$={kpar[peak_i]:.2f}, "
                fr"$k_\perp d_i$={kperp[peak_j]:.2f} "
                fr"($\theta$={theta:.0f}$\degree$)"
            ),
        )
        axis.legend(loc="upper right", fontsize=9)

    axis.set_xlabel(r"$k_\parallel\,d_i$")
    axis.set_ylabel(r"$k_\perp\,d_i$")
    axis.set_title(fr"Growth-rate map $\gamma(k_\parallel,k_\perp)$ - {component}")
    axis.grid(True, color="white", alpha=0.12, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def print_diagnostics(result: dict, metadata: dict, component: str):
    diag = result.get("diagnostics", {})
    print("Growth-rate map diagnostics:")
    print(f"  component: {component}")
    print(f"  plane: {metadata.get('plane')} axes={metadata.get('axes')} normal={metadata.get('normal_axis')}")
    print(f"  parallel axis: {diag.get('parallel_axis')}")
    print(f"  snapshots used: {diag.get('nt')}  fit points: {diag.get('fit_points')}")
    print(f"  grid on analyzed plane: {diag.get('n_axis0')} x {diag.get('n_axis1')}")
    print(f"  k_parallel folded to |k|: {diag.get('fold_negative_k')}")
    print(
        "  k modes retained: "
        f"k_parallel={diag.get('folded_kpar_modes')} (dk={diag.get('dkpar'):.4g}), "
        f"k_perp={diag.get('folded_kperp_modes')} (dk={diag.get('dkperp'):.4g})"
    )
    if metadata.get("normal_axis") == diag.get("parallel_axis"):
        print(
            "  [WARN] selected plane collapses the requested parallel axis; "
            "choose a plane containing B0/parallel direction."
        )
    gamma = np.asarray(result["gamma"], dtype=float)
    final_power = np.asarray(result["final_power"], dtype=float)
    rvalue = np.asarray(result["rvalue"], dtype=float)
    floor = np.nanmax(final_power) * 1e-3 if np.any(np.isfinite(final_power)) else np.inf
    good = np.isfinite(gamma) & (gamma > 0.0) & (final_power >= floor)
    minr = float(result["min_rvalue"])
    if minr > 0:
        good &= rvalue >= minr
    print(f"  display-quality growing cells: {int(np.count_nonzero(good))}/{gamma.size}")


def write_csv(result: dict, path: Path):
    kpar = result["kpar"]
    kperp = result["kperp"]
    gamma = result["gamma"]
    rvalue = result["rvalue"]
    final_power = result["final_power"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kpar_di", "kperp_di", "gamma_oci", "rvalue", "final_power"])
        for i in range(len(kpar)):
            for j in range(len(kperp)):
                writer.writerow(
                    [
                        f"{kpar[i]:.6f}",
                        f"{kperp[j]:.6f}",
                        f"{gamma[i, j]:.6e}",
                        f"{rvalue[i, j]:.4f}",
                        f"{final_power[i, j]:.6e}",
                    ]
                )


def main() -> int:
    from dispersion_analysis import load_series

    try:
        from psc_units import DX_DI, step_to_omegaci
    except (ImportError, ValueError):
        DX_DI = 1.0
        step_to_omegaci = lambda step: float(step)

    parser = argparse.ArgumentParser(
        description="gamma(k_parallel,k_perp) growth-rate map from PSC snapshots."
    )
    parser.add_argument("--fields", default="pfd.*.h5")
    parser.add_argument("--plane", choices=["auto", "xy", "xz", "yz"], default="yz")
    parser.add_argument("--parallel-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--component", choices=["perp", "parallel", "total"], default="parallel")
    parser.add_argument("--dx", type=float, default=DX_DI)
    parser.add_argument("--dy", type=float, default=DX_DI)
    parser.add_argument("--dz", type=float, default=DX_DI)
    parser.add_argument("--kpar-max", type=float, default=1.5)
    parser.add_argument("--kperp-max", type=float, default=1.5)
    parser.add_argument("--min-rvalue", type=float, default=0.7)
    parser.add_argument("--fit-lo", type=float, default=0.1)
    parser.add_argument("--fit-hi", type=float, default=0.6)
    parser.add_argument("--fold-negative-k", action="store_true",
                        help="Fold +/-k_parallel onto |k_parallel| too (in addition to the always-folded "
                             "k_perp). Off by default: folding k_parallel sums forward- and backward-"
                             "propagating branches into the same pixel, which can conflate distinct "
                             "oblique modes.")
    parser.add_argument("--display-kpar-max", type=float, default=None,
                        help="Plot-only k_parallel zoom; does not change the computed CSV.")
    parser.add_argument("--display-kperp-max", type=float, default=0.9,
                        help="Plot-only k_perp zoom; does not change the computed CSV.")
    parser.add_argument("--shading", choices=["auto", "nearest", "gouraud"], default="auto",
                        help="pcolormesh shading used only for display.")
    parser.add_argument("--contours", type=int, default=7,
                        help="Number of gamma contour levels overlaid on the image.")
    parser.add_argument("--angle-step", type=int, default=15,
                        help="Angle-grid spacing in degrees.")
    parser.add_argument("--outdir", default="spectral_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    series, times, metadata = load_series(
        args.fields,
        args.plane,
        args.parallel_axis,
        args.component,
        args.dx,
        args.dy,
        args.dz,
        step_to_omegaci,
    )
    result = compute_growth_rate_map(
        series,
        times,
        metadata["spacing"],
        metadata["axes"],
        parallel_axis=args.parallel_axis,
        kpar_max=args.kpar_max,
        kperp_max=args.kperp_max,
        min_rvalue=args.min_rvalue,
        fit_frac=(args.fit_lo, args.fit_hi),
        fold_negative_k=args.fold_negative_k,
    )
    png = outdir / f"growth_rate_map_{metadata['plane']}_{args.component}.png"
    csv_path = outdir / f"growth_rate_map_{metadata['plane']}_{args.component}.csv"
    print_diagnostics(result, metadata, args.component)
    plot_growth_rate_map(
        result,
        png,
        args.component,
        display_kpar_max=args.display_kpar_max,
        display_kperp_max=args.display_kperp_max,
        shading=args.shading,
        contour_count=args.contours,
        angle_step=args.angle_step,
    )
    write_csv(result, csv_path)
    print(f"Saved growth-rate map: {png}")
    print(f"Saved data: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
