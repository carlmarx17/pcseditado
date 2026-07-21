#!/usr/bin/env python3
"""Space-time spectral density in frequency--phase-velocity coordinates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_reader import PICDataReader
from spectral_analysis import SpectralAnalyzer

plt.switch_backend("Agg")
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})


def _smooth_2d(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small dependency-free Gaussian-like smoother for display density."""
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    result = np.asarray(values, dtype=float)
    for _ in range(max(passes, 0)):
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 0, result
        )
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, result
        )
    return result


def compute_phase_velocity_density(
    field_series: np.ndarray,
    time_oci: np.ndarray,
    spacing: tuple[float, float],
    axes: tuple[str, str],
    parallel_axis: str = "z",
    velocity_min: float = 0.5,
    velocity_max: float = 12.0,
    velocity_bins: int = 240,
    frequency_bins: int = 180,
    absolute_velocity: bool = True,
    max_spatial_mode: int = 128,
    temporal_fft_size: int = 128,
) -> dict:
    """Transform B(t, axis0, axis1) into weighted (omega, v_phase) density.

    ``field_series`` has shape ``(n_components, nt, n0, n1)``. Time is
    normalized as ``Omega_ci t`` and spacing is measured in ``d_i``. Therefore
    ``(omega/Omega_ci) / (k_parallel d_i)`` is directly ``v_phase/v_A``.
    """
    fields = np.asarray(field_series, dtype=float)
    times = np.asarray(time_oci, dtype=float)
    if fields.ndim != 4:
        raise ValueError(
            "field_series must have shape (components, time, axis0, axis1)"
        )
    if fields.shape[1] != len(times) or len(times) < 4:
        raise ValueError("At least four time-aligned snapshots are required")
    if parallel_axis not in axes:
        raise ValueError(
            f"Plane axes {axes} do not contain parallel axis '{parallel_axis}'"
        )

    delta_t = np.diff(times)
    if not np.allclose(delta_t, np.median(delta_t), rtol=1e-6, atol=1e-12):
        raise ValueError("Snapshot times must be uniformly spaced")
    dt = float(np.median(delta_t))

    fields = fields - np.mean(fields, axis=(2, 3), keepdims=True)
    fields = fields - np.mean(fields, axis=1, keepdims=True)
    nt, n0, n1 = fields.shape[1:]
    spatial_window = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
    temporal_window = np.hanning(nt)[:, None, None]

    full_k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    full_k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi
    half0 = min(max_spatial_mode, (n0 - 1) // 2)
    half1 = min(max_spatial_mode, (n1 - 1) // 2)
    center0 = n0 // 2
    center1 = n1 // 2
    slice0 = slice(center0 - half0, center0 + half0 + 1)
    slice1 = slice(center1 - half1, center1 + half1 + 1)
    k0 = full_k0[slice0]
    k1 = full_k1[slice1]

    nfft = max(int(temporal_fft_size), nt)
    power = np.zeros((nfft, len(k0), len(k1)), dtype=float)
    for component in fields:
        spatial_fft = np.fft.fftshift(
            np.fft.fft2(component * spatial_window, axes=(1, 2)),
            axes=(1, 2),
        )[:, slice0, slice1]
        transformed = np.fft.fftshift(
            np.fft.fft(spatial_fft * temporal_window, n=nfft, axis=0),
            axes=0,
        )
        power += np.abs(transformed) ** 2

    omega = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt)) * 2.0 * np.pi
    omega_grid, k0_grid, k1_grid = np.meshgrid(
        omega, k0, k1, indexing="ij"
    )
    k_parallel = k0_grid if axes[0] == parallel_axis else k1_grid

    positive_frequency = omega_grid > 0
    nonzero_k = np.abs(k_parallel) > 1e-12
    phase_velocity = np.divide(
        omega_grid,
        k_parallel,
        out=np.full_like(omega_grid, np.nan),
        where=nonzero_k,
    )
    if absolute_velocity:
        phase_velocity = np.abs(phase_velocity)

    if absolute_velocity:
        velocity_range = (velocity_min, velocity_max)
        velocity_mask = (
            (phase_velocity >= velocity_min) & (phase_velocity <= velocity_max)
        )
    else:
        velocity_range = (-velocity_max, velocity_max)
        velocity_mask = (
            (np.abs(phase_velocity) >= velocity_min)
            & (np.abs(phase_velocity) <= velocity_max)
        )

    valid = (
        positive_frequency
        & nonzero_k
        & velocity_mask
        & np.isfinite(phase_velocity)
        & np.isfinite(power)
        & (power > 0)
    )
    if not np.any(valid):
        raise ValueError("No finite spectral modes remain inside the requested range")

    omega_values = omega_grid[valid]
    velocity_values = phase_velocity[valid]
    # The FFT samples are uniform in k_parallel, but the displayed coordinate is
    # v_phase = omega / k_parallel. Compensate the sampling-density Jacobian so
    # low phase velocities are not over-weighted by the nonlinear mapping.
    jacobian_weight = np.divide(
        velocity_values**2,
        np.abs(omega_values),
        out=np.zeros_like(velocity_values),
        where=np.abs(omega_values) > 0,
    )
    weights = power[valid] * jacobian_weight
    omega_max = float(np.max(omega[omega > 0]))
    density, omega_edges, velocity_edges = np.histogram2d(
        omega_values,
        velocity_values,
        bins=(frequency_bins, velocity_bins),
        range=((0.0, omega_max), velocity_range),
        weights=weights,
    )
    # Each discrete k contributes one straight v_ph=omega/k ray; with too few
    # independent k's relative to the velocity/frequency bin count, those rays
    # show up as separate streaks instead of merging into a continuum. Using
    # more spatial modes (max_spatial_mode above) and a bit more smoothing
    # here closes most of the gaps between them.
    density = _smooth_2d(density, passes=3)
    # Conditional density P(v_phase | omega): this prevents frequencies with
    # larger total fluctuation power from hiding weaker but coherent branches.
    frequency_power = np.sum(density, axis=1, keepdims=True)
    density = np.divide(
        density,
        frequency_power,
        out=np.zeros_like(density),
        where=frequency_power > 0,
    )

    return {
        "density": density,
        "omega_edges": omega_edges,
        "velocity_edges": velocity_edges,
        "omega_samples": omega[omega > 0],
        "omega_grid": omega_grid,
        "phase_velocity": phase_velocity,
        "power": power,
        "k0": k0,
        "k1": k1,
        "omega": omega,
        "spacing": spacing,
        "axes": axes,
        "parallel_axis": parallel_axis,
        "jacobian_weight": np.divide(
            phase_velocity**2,
            np.abs(omega_grid),
            out=np.zeros_like(phase_velocity),
            where=np.isfinite(phase_velocity) & (np.abs(omega_grid) > 0),
        ),
        "valid": valid,
        "absolute_velocity": absolute_velocity,
        "independent_positive_frequencies": (nt - 1) // 2,
    }


def extract_ridges(result: dict, ridge_count: int = 2) -> list[dict]:
    """Find separated phase-velocity peaks at every resolved positive frequency."""
    rows: list[dict] = []
    omega_grid = result["omega_grid"]
    velocity = result["phase_velocity"]
    power = result["power"]
    jacobian_weight = result["jacobian_weight"]
    valid = result["valid"]
    velocity_edges = result["velocity_edges"]
    centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

    for omega in result["omega_samples"]:
        frequency_mask = valid & np.isclose(omega_grid, omega)
        if not np.any(frequency_mask):
            continue
        histogram, _ = np.histogram(
            velocity[frequency_mask],
            bins=velocity_edges,
            weights=power[frequency_mask] * jacobian_weight[frequency_mask],
        )
        work = histogram.astype(float)
        exclusion = max(2, len(centers) // 40)
        for rank in range(1, ridge_count + 1):
            index = int(np.argmax(work))
            peak_power = float(work[index])
            if peak_power <= 0:
                break
            rows.append(
                {
                    "omega_over_omega_ci": float(omega),
                    "phase_velocity_over_va": float(centers[index]),
                    "ridge_rank": rank,
                    "spectral_power": peak_power,
                }
            )
            lo = max(0, index - exclusion)
            hi = min(len(work), index + exclusion + 1)
            work[lo:hi] = 0.0
    return rows


def plot_density(result: dict, ridges: list[dict], output: Path, component: str):
    density = result["density"]
    normalized = density / max(float(np.max(density)), np.finfo(float).tiny)
    log_density = np.log10(normalized + 1e-8)

    fig, axis = plt.subplots(figsize=(9.2, 7.0))
    image = axis.pcolormesh(
        result["omega_edges"],
        result["velocity_edges"],
        log_density.T,
        shading="auto",
        cmap="turbo",
        vmin=-6,
        vmax=0,
    )
    levels = [-4.0, -3.0, -2.0, -1.0]
    omega_centers = 0.5 * (
        result["omega_edges"][:-1] + result["omega_edges"][1:]
    )
    velocity_centers = 0.5 * (
        result["velocity_edges"][:-1] + result["velocity_edges"][1:]
    )
    axis.contour(
        omega_centers,
        velocity_centers,
        log_density.T,
        levels=levels,
        colors="white",
        linewidths=0.7,
        linestyles="dotted",
        alpha=0.8,
    )

    for rank in sorted({row["ridge_rank"] for row in ridges}):
        selected = [row for row in ridges if row["ridge_rank"] == rank]
        axis.plot(
            [row["omega_over_omega_ci"] for row in selected],
            [row["phase_velocity_over_va"] for row in selected],
            "k.",
            markersize=5 if rank == 1 else 3.5,
            alpha=0.95 if rank == 1 else 0.65,
            label="dominant ridge" if rank == 1 else None,
        )

    velocity_label = (
        r"$|v_{\rm ph}|/v_A$"
        if result["absolute_velocity"]
        else r"$v_{\rm ph}/v_A$"
    )
    axis.set_xlabel(r"Angular frequency $\omega/\Omega_{ci}$")
    axis.set_ylabel(velocity_label)
    axis.set_title(
        f"Frequency-normalized mode density ({component} magnetic power)"
    )
    axis.grid(alpha=0.15)
    if ridges:
        axis.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}[P(v_{\rm ph}\mid\omega)/P_{\max}]$")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_omega_k_dispersion(
    result: dict,
    output: Path,
    component: str,
    va_over_c: float,
    kperp_reduction: str = "sum",
):
    """Dense omega-k dispersion diagram in log(omega/omega_p) vs log(k c/omega_p).

    Uses the full ``power`` cube (nfft, nk0, nk1) already computed in
    ``compute_phase_velocity_density``. Because time is Omega_ci t and spacing is
    in d_i, we have k c/omega_pi = k d_i exactly, and omega/omega_pi =
    (omega/Omega_ci) * (v_A/c). Pass ``va_over_c`` (= Omega_ci/omega_pi).
    """
    power = result["power"]                      # (nfft, nk0, nk1)
    omega = result["omega"]                      # omega/Omega_ci, shape (nfft,)
    k0 = result["k0"]                            # k*d_i along axis0
    k1 = result["k1"]                            # k*d_i along axis1
    axes = result["axes"]
    parallel_axis = result["parallel_axis"]

    # Pick k_parallel axis and reduce the perpendicular one.
    if axes[0] == parallel_axis:
        k_par = k0
        axis_perp = 2                            # reduce k1
    else:
        k_par = k1
        axis_perp = 1                            # reduce k0

    if kperp_reduction == "sum":
        p2d = np.sum(power, axis=axis_perp)      # (nfft, nk_par)
    elif kperp_reduction == "max":
        p2d = np.max(power, axis=axis_perp)
    else:                                        # perpendicular slice at k_perp~0
        kperp = k1 if axis_perp == 2 else k0
        j0 = int(np.argmin(np.abs(kperp)))
        p2d = np.take(power, j0, axis=axis_perp)

    # Keep positive omega and positive k_parallel (physical quadrant).
    pos_w = omega > 0
    pos_k = k_par > 0
    w = omega[pos_w]                             # omega/Omega_ci
    kk = k_par[pos_k]                            # k d_i = k c/omega_pi
    p2d = p2d[np.ix_(pos_w, pos_k)]

    # Convert to requested units.
    w_wp = w * va_over_c                         # omega/omega_pi
    kc_wp = kk                                   # k c/omega_pi = k d_i (identity)

    log_x = np.log10(kc_wp)
    log_y = np.log10(w_wp)
    pnorm = p2d / max(float(np.max(p2d)), np.finfo(float).tiny)
    log_p = np.log10(pnorm + 1e-8)

    fig, axis = plt.subplots(figsize=(9.6, 7.4))
    image = axis.pcolormesh(
        log_x, log_y, log_p,
        shading="auto", cmap="turbo", vmin=-6, vmax=0,
    )
    # Reference line: Alfven phase speed v_ph = v_A  ->  omega/Omega_ci = k d_i
    #   -> omega/omega_pi = (k d_i) * va_over_c
    kline = np.logspace(np.log10(kc_wp.min()), np.log10(kc_wp.max()), 50)
    axis.plot(
        np.log10(kline), np.log10(kline * va_over_c),
        color="white", ls="--", lw=1.2, alpha=0.7, label=r"$v_{ph}=v_A$",
    )
    axis.set_xlabel(r"$\log_{10}(k\,c/\omega_{pi})=\log_{10}(k\,d_i)$")
    axis.set_ylabel(r"$\log_{10}(\omega/\omega_{pi})$")
    axis.set_title(f"Dispersion diagram ({component} magnetic power)")
    axis.legend(loc="lower right")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}[P(\omega,k_\parallel)/P_{\max}]$")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_series(
    pattern: str,
    plane: str,
    parallel_axis: str,
    component: str,
    dx: float,
    dy: float,
    dz: float,
    step_to_time,
) -> tuple[np.ndarray, np.ndarray, dict]:
    files = PICDataReader.find_files(pattern)
    if len(files) < 4:
        raise ValueError(f"Need at least four snapshots; found {len(files)}")

    slicer = SpectralAnalyzer(
        dx=dx, dy=dy, dz=dz, parallel_axis=parallel_axis, outdir="/tmp/psc-dispersion"
    )
    snapshots = []
    times = []
    metadata = None
    perpendicular_axes = [axis for axis in ("x", "y", "z") if axis != parallel_axis]

    for step, filepath in files.items():
        fields = PICDataReader.read_multiple_fields_3d(
            filepath, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"]
        )
        plane_data = slicer._get_plane_slice(
            fields["hx_fc/p0/3d"],
            fields["hy_fc/p0/3d"],
            fields["hz_fc/p0/3d"],
            plane,
        )
        components = {
            "x": np.atleast_2d(plane_data["bx"]),
            "y": np.atleast_2d(plane_data["by"]),
            "z": np.atleast_2d(plane_data["bz"]),
        }
        if component == "perp":
            selected = [components[axis] for axis in perpendicular_axes]
        elif component == "parallel":
            selected = [components[parallel_axis]]
        else:
            selected = list(components.values())
        snapshots.append(np.stack(selected))
        times.append(step_to_time(step))
        metadata = plane_data

    # snapshots: time, component, axis0, axis1 -> component, time, axis0, axis1
    return (
        np.transpose(np.asarray(snapshots, dtype=np.float32), (1, 0, 2, 3)),
        np.asarray(times, dtype=float),
        metadata,
    )


def main() -> int:
    try:
        from psc_units import DX_DI, VA_OVER_C, step_to_omegaci
    except (ImportError, ValueError):
        DX_DI = 1.0
        VA_OVER_C = None
        step_to_omegaci = lambda step: float(step)

    parser = argparse.ArgumentParser(
        description="Frequency--phase-velocity density map from PSC field snapshots."
    )
    parser.add_argument("--fields", default="pfd.*.h5")
    parser.add_argument("--plane", choices=["auto", "xy", "xz", "yz"], default="auto")
    parser.add_argument("--parallel-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--component", choices=["perp", "parallel", "total"], default="perp")
    parser.add_argument("--dx", type=float, default=DX_DI)
    parser.add_argument("--dy", type=float, default=DX_DI)
    parser.add_argument("--dz", type=float, default=DX_DI)
    parser.add_argument("--velocity-min", type=float, default=0.5)
    parser.add_argument("--velocity-max", type=float, default=12.0)
    parser.add_argument("--max-spatial-mode", type=int, default=128)
    parser.add_argument("--temporal-fft-size", type=int, default=128)
    parser.add_argument("--signed-velocity", action="store_true")
    parser.add_argument("--ridges", type=int, default=2)
    parser.add_argument("--va-over-c", type=float, default=VA_OVER_C,
                        help="v_A/c = Omega_ci/omega_pi. If set, also emit the omega-k dispersion diagram. "
                             "Defaults to psc_units.VA_OVER_C when available.")
    parser.add_argument("--kperp-reduction", choices=["sum", "max", "slice"], default="sum",
                        help="How to collapse the perpendicular k axis for the omega-k diagram.")
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
    result = compute_phase_velocity_density(
        series,
        times,
        metadata["spacing"],
        metadata["axes"],
        parallel_axis=args.parallel_axis,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        absolute_velocity=not args.signed_velocity,
        max_spatial_mode=args.max_spatial_mode,
        temporal_fft_size=args.temporal_fft_size,
    )
    ridges = extract_ridges(result, ridge_count=args.ridges)
    suffix = "absolute" if result["absolute_velocity"] else "signed"
    image_path = outdir / f"dispersion_density_{metadata['plane']}_{args.component}_{suffix}.png"
    csv_path = outdir / f"dispersion_ridges_{metadata['plane']}_{args.component}_{suffix}.csv"
    plot_density(result, ridges, image_path, args.component)
    if args.va_over_c is not None:
        wk_path = outdir / f"dispersion_omega_k_{metadata['plane']}_{args.component}_{suffix}.png"
        plot_omega_k_dispersion(result, wk_path, args.component, args.va_over_c,
                                kperp_reduction=args.kperp_reduction)
        print(f"Saved omega-k dispersion diagram: {wk_path}")
    write_csv(csv_path, ridges)
    print(f"Processed {series.shape[1]} snapshots on plane {metadata['plane']}.")
    print(
        "Independent positive frequencies: "
        f"{result['independent_positive_frequencies']}; "
        f"displayed FFT bins after zero-padding: {len(result['omega_samples'])}."
    )
    print(f"Saved density map: {image_path}")
    print(f"Saved modal ridges: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
