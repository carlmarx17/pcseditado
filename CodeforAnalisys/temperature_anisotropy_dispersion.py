#!/usr/bin/env python3
"""Temperature-anisotropy fluctuation spectra and dispersion diagnostics.

This script combines the pressure-tensor reconstruction used by
``anisotropy_analysis.py`` with the time-space FFT machinery used by
``dispersion_analysis.py``.  The analyzed scalar is a field-aligned
temperature diagnostic such as A = T_perp / T_parallel, computed from central
second moments after subtracting bulk-flow pressure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anisotropy_analysis import field_aligned_pressures
from data_reader import PICDataReader
from dispersion_analysis import compute_phase_velocity_density, extract_ridges
from spectral_analysis import SpectralAnalyzer
from psc_units import (
    DX_DI,
    DRIVEN_SPECIES,
    FIELD_FILE_PATTERN,
    MASS_RATIO,
    M_ELEC,
    M_ION,
    MOMENT_FILE_PATTERN,
    PROFILE_LABEL,
    step_to_omegaci,
)

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

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#30363d"
EPS = 1e-30


def _style_axes(axis):
    axis.set_facecolor(PANEL_BG)
    axis.tick_params(
        which="both", colors=TEXT_CLR, direction="in", top=True, right=True
    )
    axis.grid(True, color=GRID_CLR, alpha=0.22, linestyle=":")
    for spine in axis.spines.values():
        spine.set_edgecolor(GRID_CLR)
    axis.xaxis.label.set_color(TEXT_CLR)
    axis.yaxis.label.set_color(TEXT_CLR)
    axis.title.set_color(TEXT_CLR)


def _style_colorbar(colorbar):
    colorbar.ax.yaxis.label.set_color(TEXT_CLR)
    colorbar.ax.tick_params(colors=TEXT_CLR)
    for label in colorbar.ax.get_yticklabels():
        label.set_color(TEXT_CLR)


def _slice_scalar_plane(data_3d: np.ndarray, plane: str, slice_idx: int | None = None) -> np.ndarray:
    """Return the same PSC-ordered plane convention as SpectralAnalyzer."""
    values = np.asarray(data_3d)
    if values.ndim != 3:
        raise ValueError(f"Expected PSC 3D array, received shape {values.shape}")

    nz, ny, nx = values.shape
    if plane == "xy":
        idx = nz // 2 if slice_idx is None else slice_idx
        return values[idx, :, :].squeeze()
    if plane == "xz":
        idx = ny // 2 if slice_idx is None else slice_idx
        return values[:, idx, :].squeeze()
    if plane == "yz":
        idx = nx // 2 if slice_idx is None else slice_idx
        return values[:, :, idx].squeeze()
    raise ValueError(f"Unknown plane {plane}")


def _fill_invalid(values: np.ndarray) -> np.ndarray:
    """Replace NaN/inf holes by the finite median before FFTs."""
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if not np.any(finite):
        raise ValueError("Scalar field contains no finite values")
    replacement = float(np.nanmedian(result[finite]))
    result[~finite] = replacement
    return result


def _quantity_label(quantity: str, species_symbol: str) -> str:
    labels = {
        "anisotropy": rf"$A_{species_symbol}=T_\perp/T_\parallel$",
        "log-anisotropy": rf"$\log(A_{species_symbol})$",
        "parallel-temperature": rf"$T_{{{species_symbol}\parallel}}$",
        "perp-temperature": rf"$T_{{{species_symbol}\perp}}$",
    }
    return labels[quantity]


def compute_temperature_quantity(
    moment_file: str,
    field_file: str,
    species: str = DRIVEN_SPECIES,
    quantity: str = "anisotropy",
) -> dict:
    """Compute a 3D field-aligned temperature quantity from one snapshot."""
    suffix = "i" if species == "ion" else "e"
    mass = M_ION if species == "ion" else M_ELEC

    moments = PICDataReader.read_multiple_fields_3d(
        moment_file,
        "all_1st",
        [
            f"txx_{suffix}/p0/3d",
            f"tyy_{suffix}/p0/3d",
            f"tzz_{suffix}/p0/3d",
            f"txy_{suffix}/p0/3d",
            f"tyz_{suffix}/p0/3d",
            f"tzx_{suffix}/p0/3d",
            f"px_{suffix}/p0/3d",
            f"py_{suffix}/p0/3d",
            f"pz_{suffix}/p0/3d",
            f"rho_{suffix}/p0/3d",
        ],
    )
    fields = PICDataReader.read_multiple_fields_3d(
        field_file,
        "jeh-",
        ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
    )

    rho = np.asarray(moments[f"rho_{suffix}/p0/3d"], dtype=float)
    n = rho if species == "ion" else -rho
    safe_n = np.where(n > 0.05, n, np.nan)

    px = np.asarray(moments[f"px_{suffix}/p0/3d"], dtype=float)
    py = np.asarray(moments[f"py_{suffix}/p0/3d"], dtype=float)
    pz = np.asarray(moments[f"pz_{suffix}/p0/3d"], dtype=float)

    pxx = np.asarray(moments[f"txx_{suffix}/p0/3d"], dtype=float) - px * px / (safe_n * mass)
    pyy = np.asarray(moments[f"tyy_{suffix}/p0/3d"], dtype=float) - py * py / (safe_n * mass)
    pzz = np.asarray(moments[f"tzz_{suffix}/p0/3d"], dtype=float) - pz * pz / (safe_n * mass)
    pxy = np.asarray(moments[f"txy_{suffix}/p0/3d"], dtype=float) - px * py / (safe_n * mass)
    pyz = np.asarray(moments[f"tyz_{suffix}/p0/3d"], dtype=float) - py * pz / (safe_n * mass)
    pzx = np.asarray(moments[f"tzx_{suffix}/p0/3d"], dtype=float) - pz * px / (safe_n * mass)

    bx = np.asarray(fields["hx_fc/p0/3d"], dtype=float)
    by = np.asarray(fields["hy_fc/p0/3d"], dtype=float)
    bz = np.asarray(fields["hz_fc/p0/3d"], dtype=float)
    p_par, p_perp, b2 = field_aligned_pressures(pxx, pyy, pzz, pxy, pyz, pzx, bx, by, bz)

    t_par = p_par / safe_n
    t_perp = p_perp / safe_n
    anisotropy = t_perp / (t_par + EPS)

    valid = (
        (n > 0.05)
        & (p_par > 0.0)
        & (p_perp > 0.0)
        & (b2 > 1e-10)
        & np.isfinite(anisotropy)
        & (anisotropy > 0.02)
        & (anisotropy < 50.0)
    )

    if quantity == "anisotropy":
        values = anisotropy
    elif quantity == "log-anisotropy":
        values = np.log(anisotropy)
    elif quantity == "parallel-temperature":
        values = t_par
    elif quantity == "perp-temperature":
        values = t_perp
    else:
        raise ValueError(f"Unknown quantity {quantity}")

    values = np.where(valid & np.isfinite(values), values, np.nan)
    return {
        "values": values,
        "valid_cells": int(np.count_nonzero(valid)),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
    }


def load_temperature_series(
    moment_pattern: str,
    field_pattern: str,
    plane: str,
    parallel_axis: str,
    species: str,
    quantity: str,
    dx: float,
    dy: float,
    dz: float,
    drop_first: bool = True,
) -> dict:
    """Read all paired snapshots and return a scalar time series on one plane."""
    moment_files = PICDataReader.find_files(moment_pattern)
    field_files = PICDataReader.find_files(field_pattern)
    steps = sorted(set(moment_files) & set(field_files))
    if drop_first and len(steps) > 1:
        steps = steps[1:]
    if len(steps) < 4:
        raise ValueError(f"Need at least four paired snapshots; found {len(steps)}")

    slicer = SpectralAnalyzer(
        dx=dx,
        dy=dy,
        dz=dz,
        parallel_axis=parallel_axis,
        outdir="/tmp/psc-temperature-anisotropy-dispersion",
    )

    snapshots: list[np.ndarray] = []
    stats: list[dict] = []
    metadata = None
    for index, step in enumerate(steps, start=1):
        print(f"Processing step {step:6d} ({index}/{len(steps)})")
        fields = PICDataReader.read_multiple_fields_3d(
            field_files[step],
            "jeh-",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )
        plane_data = slicer._get_plane_slice(
            fields["hx_fc/p0/3d"],
            fields["hy_fc/p0/3d"],
            fields["hz_fc/p0/3d"],
            plane,
        )
        quantity_data = compute_temperature_quantity(
            moment_files[step],
            field_files[step],
            species=species,
            quantity=quantity,
        )
        scalar = _slice_scalar_plane(
            quantity_data["values"],
            plane_data["plane"],
            slice_idx=plane_data["slice_idx"],
        )
        scalar = np.atleast_2d(_fill_invalid(scalar))
        snapshots.append(scalar.astype(np.float32))
        stats.append({
            "step": step,
            "omega_ci_t": step_to_omegaci(step),
            "mean": quantity_data["mean"],
            "median": quantity_data["median"],
            "valid_cells": quantity_data["valid_cells"],
        })
        metadata = plane_data

    series = np.asarray(snapshots, dtype=np.float32)
    return {
        "series": series,
        "times": np.asarray([row["omega_ci_t"] for row in stats], dtype=float),
        "steps": np.asarray(steps, dtype=int),
        "stats": stats,
        "metadata": metadata,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase_velocity_density(
    result: dict,
    ridges: list[dict],
    output: Path,
    quantity: str,
    species_symbol: str,
) -> None:
    density = result["density"]
    normalized = density / max(float(np.max(density)), np.finfo(float).tiny)
    log_density = np.log10(normalized + 1e-8)

    fig, axis = plt.subplots(figsize=(9.2, 7.0))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(axis)
    image = axis.pcolormesh(
        result["omega_edges"],
        result["velocity_edges"],
        log_density.T,
        shading="auto",
        cmap="turbo",
        vmin=-6,
        vmax=0,
    )

    omega_centers = 0.5 * (result["omega_edges"][:-1] + result["omega_edges"][1:])
    velocity_centers = 0.5 * (result["velocity_edges"][:-1] + result["velocity_edges"][1:])
    axis.contour(
        omega_centers,
        velocity_centers,
        log_density.T,
        levels=[-4.0, -3.0, -2.0, -1.0],
        colors="white",
        linewidths=0.7,
        linestyles="dotted",
        alpha=0.75,
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

    velocity_label = r"$|v_{\rm ph}|/v_A$" if result["absolute_velocity"] else r"$v_{\rm ph}/v_A$"
    axis.set_xlabel(r"Angular frequency $\omega/\Omega_{ci}$")
    axis.set_ylabel(velocity_label)
    axis.set_title("Temperature-anisotropy fluctuation density", fontsize=18, pad=10)
    axis.text(
        0.01,
        0.99,
        _quantity_label(quantity, species_symbol),
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=TEXT_CLR,
        fontsize=13,
    )
    if ridges:
        axis.legend(loc="upper right", fontsize=12, framealpha=0.75)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}[P(v_{\rm ph}\mid\omega)/P_{\max}]$")
    _style_colorbar(colorbar)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor=DARK_BG)
    plt.close(fig)


def compute_folded_kspace_density(
    scalar_2d: np.ndarray,
    spacing: tuple[float, float],
    axes: tuple[str, str],
    parallel_axis: str,
    bins: int = 240,
) -> dict:
    """Fold a 2D scalar spectrum into positive |k_parallel|, |k_perp| axes."""
    field = _fill_invalid(scalar_2d)
    field = field - np.mean(field)
    n0, n1 = field.shape
    window = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(field * window))
    power = np.abs(spectrum) ** 2

    k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi
    k0_grid, k1_grid = np.meshgrid(k0, k1, indexing="ij")
    if axes[0] == parallel_axis:
        k_parallel = np.abs(k0_grid)
        k_perp = np.abs(k1_grid)
        perpendicular_axis = axes[1]
    elif axes[1] == parallel_axis:
        k_parallel = np.abs(k1_grid)
        k_perp = np.abs(k0_grid)
        perpendicular_axis = axes[0]
    else:
        raise ValueError(f"Plane axes {axes} do not contain parallel axis '{parallel_axis}'")

    nonzero = (k_parallel > 1e-12) | (k_perp > 1e-12)
    density, kpar_edges, kperp_edges = np.histogram2d(
        k_parallel[nonzero].ravel(),
        k_perp[nonzero].ravel(),
        bins=bins,
        weights=power[nonzero].ravel(),
    )
    counts, _, _ = np.histogram2d(
        k_parallel[nonzero].ravel(),
        k_perp[nonzero].ravel(),
        bins=(kpar_edges, kperp_edges),
    )
    density = np.divide(density, counts, out=np.zeros_like(density), where=counts > 0)
    return {
        "density": density,
        "kpar_edges": kpar_edges,
        "kperp_edges": kperp_edges,
        "parallel_axis": parallel_axis,
        "perpendicular_axis": perpendicular_axis,
    }


def plot_kspace_density(
    result: dict,
    output: Path,
    quantity: str,
    species_symbol: str,
    step: int,
    time_oci: float,
) -> None:
    density = result["density"]
    normalized = density / max(float(np.max(density)), np.finfo(float).tiny)
    log_power = np.log10(normalized + 1e-8)

    fig, axis = plt.subplots(figsize=(8.4, 7.0))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(axis)
    image = axis.pcolormesh(
        result["kpar_edges"],
        result["kperp_edges"],
        log_power.T,
        shading="auto",
        cmap="turbo",
        vmin=-6,
        vmax=0,
    )
    axis.set_xlabel(rf"$|k_{{{result['parallel_axis']}}}|d_i$")
    axis.set_ylabel(rf"$|k_{{{result['perpendicular_axis']}}}|d_i$")
    axis.set_title("Temperature-anisotropy k-space spectrum", fontsize=18, pad=10)
    axis.text(
        0.01,
        0.99,
        rf"{_quantity_label(quantity, species_symbol)}  |  step {step}  |  "
        rf"$t\Omega_{{ci}}={time_oci:.2f}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=TEXT_CLR,
        fontsize=12,
        bbox={"facecolor": PANEL_BG, "edgecolor": "none", "alpha": 0.78, "pad": 3},
    )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}(P/P_{\max})$")
    _style_colorbar(colorbar)
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor=DARK_BG)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispersion-style diagnostics for PSC temperature anisotropy."
    )
    parser.add_argument("--data-dir", type=Path, help="PSC directory; auto-detect pfd and pfd_moments files.")
    parser.add_argument("--moments", default=MOMENT_FILE_PATTERN, help="Glob for pfd_moments snapshots.")
    parser.add_argument("--fields", default=FIELD_FILE_PATTERN, help="Glob for pfd field snapshots.")
    parser.add_argument("--species", choices=["ion", "electron"], default=DRIVEN_SPECIES)
    parser.add_argument(
        "--quantity",
        choices=["anisotropy", "log-anisotropy", "parallel-temperature", "perp-temperature"],
        default="anisotropy",
    )
    parser.add_argument("--plane", choices=["auto", "xy", "xz", "yz"], default="auto")
    parser.add_argument("--parallel-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--dx", type=float, default=DX_DI)
    parser.add_argument("--dy", type=float, default=DX_DI)
    parser.add_argument("--dz", type=float, default=DX_DI)
    parser.add_argument("--velocity-min", type=float, default=0.05)
    parser.add_argument("--velocity-max", type=float, default=12.0)
    parser.add_argument("--velocity-bins", type=int, default=240)
    parser.add_argument("--frequency-bins", type=int, default=180)
    parser.add_argument("--max-spatial-mode", type=int, default=128)
    parser.add_argument("--temporal-fft-size", type=int, default=128)
    parser.add_argument("--signed-velocity", action="store_true")
    parser.add_argument("--ridges", type=int, default=2)
    parser.add_argument("--keep-first", action="store_true", help="Keep the first quiet-start snapshot.")
    parser.add_argument(
        "--kspace-step",
        type=int,
        default=None,
        help="Step used for the linear k-space panel. Default: last analyzed step.",
    )
    parser.add_argument("--kspace-bins", type=int, default=240)
    parser.add_argument("--outdir", default="anisotropy_dispersion_plots")
    args = parser.parse_args()

    if args.data_dir:
        discovered = PICDataReader.discover_outputs(str(args.data_dir))
        moment_pattern = str(discovered["data_dir"] / "pfd_moments.*_p*.h5")
        field_pattern = str(discovered["data_dir"] / "pfd.*_p*.h5")
    else:
        moment_pattern = args.moments
        field_pattern = args.fields

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    species_symbol = "i" if args.species == "ion" else "e"

    loaded = load_temperature_series(
        moment_pattern,
        field_pattern,
        plane=args.plane,
        parallel_axis=args.parallel_axis,
        species=args.species,
        quantity=args.quantity,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        drop_first=not args.keep_first,
    )
    series = loaded["series"]
    times = loaded["times"]
    metadata = loaded["metadata"]

    result = compute_phase_velocity_density(
        series[None, ...],
        times,
        metadata["spacing"],
        metadata["axes"],
        parallel_axis=args.parallel_axis,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        velocity_bins=args.velocity_bins,
        frequency_bins=args.frequency_bins,
        absolute_velocity=not args.signed_velocity,
        max_spatial_mode=args.max_spatial_mode,
        temporal_fft_size=args.temporal_fft_size,
    )
    ridges = extract_ridges(result, ridge_count=args.ridges)

    suffix = "absolute" if result["absolute_velocity"] else "signed"
    prefix = f"temperature_{args.quantity.replace('-', '_')}_{args.species}_{metadata['plane']}"
    density_path = outdir / f"{prefix}_dispersion_density_{suffix}.png"
    ridges_path = outdir / f"{prefix}_dispersion_ridges_{suffix}.csv"
    stats_path = outdir / f"{prefix}_timeseries_summary.csv"
    plot_phase_velocity_density(result, ridges, density_path, args.quantity, species_symbol)
    write_csv(ridges_path, ridges)
    write_csv(stats_path, loaded["stats"])

    if args.kspace_step is None:
        kspace_index = len(loaded["steps"]) - 1
    else:
        matches = np.where(loaded["steps"] == args.kspace_step)[0]
        if len(matches) == 0:
            raise ValueError(
                f"kspace step {args.kspace_step} is not in the analyzed paired snapshots"
            )
        kspace_index = int(matches[0])

    kspace = compute_folded_kspace_density(
        series[kspace_index],
        metadata["spacing"],
        metadata["axes"],
        args.parallel_axis,
        bins=args.kspace_bins,
    )
    kspace_step = int(loaded["steps"][kspace_index])
    kspace_path = outdir / f"{prefix}_kspace_step{kspace_step:06d}.png"
    plot_kspace_density(
        kspace,
        kspace_path,
        args.quantity,
        species_symbol,
        kspace_step,
        float(times[kspace_index]),
    )

    print(f"Processed {series.shape[0]} paired snapshots on plane {metadata['plane']}.")
    print(f"Independent positive frequencies: {result['independent_positive_frequencies']}.")
    print(f"Saved phase-velocity density: {density_path}")
    print(f"Saved modal ridges: {ridges_path}")
    print(f"Saved time-series summary: {stats_path}")
    print(f"Saved k-space anisotropy spectrum: {kspace_path}")
    if metadata["normal_axis"] == args.parallel_axis:
        print(
            "[WARN] The selected plane collapses the parallel axis; choose a plane "
            "that contains the guide-field direction for a physical k_parallel."
        )
    print(f"Profile: {PROFILE_LABEL}; species={args.species}; mi/me={int(MASS_RATIO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
