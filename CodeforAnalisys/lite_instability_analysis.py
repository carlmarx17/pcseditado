#!/usr/bin/env python3
"""Compact evidence plot for seeded Mirror, Firehose, and Whistler runs."""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_reader import PICDataReader
from psc_units import (
    B0, DRIVEN_SPECIES, DT_CODE, INSTABILITY, M_ELEC, M_ION, OMEGA_CE,
    OMEGA_CI, PROFILE_LABEL,
)


def threshold(beta):
    beta = max(float(beta), 1e-12)
    if INSTABILITY == "firehose":
        return 1.0 - 2.0 / beta
    if INSTABILITY == "mirror":
        return 1.0 + 1.0 / beta
    return 1.0 + 0.21 / beta**0.6


def drive(anisotropy, marginal):
    if INSTABILITY == "firehose":
        return marginal - anisotropy
    return anisotropy - marginal


def read_snapshot(moment_file, field_file):
    suffix = "i" if DRIVEN_SPECIES == "ion" else "e"
    mass = M_ION if DRIVEN_SPECIES == "ion" else M_ELEC
    names = [
        f"rho_{suffix}", f"px_{suffix}", f"py_{suffix}", f"pz_{suffix}",
        f"txx_{suffix}", f"tyy_{suffix}", f"tzz_{suffix}",
        f"txy_{suffix}", f"tyz_{suffix}", f"tzx_{suffix}",
    ]
    with h5py.File(moment_file, "r") as handle:
        group = handle[PICDataReader.get_uid_group(handle, "all_1st")]
        values = {
            name: group[f"{name}/p0/3d"][()].astype(float).squeeze()
            for name in names
        }
    with h5py.File(field_file, "r") as handle:
        group = handle[PICDataReader.get_uid_group(handle, "jeh")]
        bx = group["hx_fc/p0/3d"][()].astype(float).squeeze()
        by = group["hy_fc/p0/3d"][()].astype(float).squeeze()
        bz = group["hz_fc/p0/3d"][()].astype(float).squeeze()

    rho = values[f"rho_{suffix}"]
    n = rho if suffix == "i" else -rho
    safe_n = np.where(n > 0.05, n, np.nan)
    px, py, pz = (values[f"p{axis}_{suffix}"] for axis in "xyz")
    pxx = values[f"txx_{suffix}"] - px * px / (safe_n * mass)
    pyy = values[f"tyy_{suffix}"] - py * py / (safe_n * mass)
    pzz = values[f"tzz_{suffix}"] - pz * pz / (safe_n * mass)
    pxy = values[f"txy_{suffix}"] - px * py / (safe_n * mass)
    pyz = values[f"tyz_{suffix}"] - py * pz / (safe_n * mass)
    pzx = values[f"tzx_{suffix}"] - pz * px / (safe_n * mass)

    b2 = bx * bx + by * by + bz * bz
    inv_b = 1.0 / np.sqrt(np.maximum(b2, 1e-30))
    ux, uy, uz = bx * inv_b, by * inv_b, bz * inv_b
    ppar = (
        pxx * ux**2 + pyy * uy**2 + pzz * uz**2
        + 2.0 * pxy * ux * uy + 2.0 * pyz * uy * uz
        + 2.0 * pzx * uz * ux
    )
    pperp = 0.5 * (pxx + pyy + pzz - ppar)
    valid = (
        (safe_n > 0) & (ppar > 0) & (pperp > 0) & (b2 > 1e-12)
        & np.isfinite(ppar) & np.isfinite(pperp)
    )
    anisotropy = float(np.mean(pperp[valid]) / np.mean(ppar[valid]))
    beta = float(2.0 * np.mean(ppar[valid]) / np.mean(b2[valid]))

    power = np.zeros((bx.shape[0], bx.shape[1] // 2 + 1))
    for component in (bx, by, bz):
        spectrum = np.fft.rfft2(component - np.mean(component))
        power += np.abs(spectrum) ** 2
    ky = np.fft.fftfreq(bx.shape[0]) * bx.shape[0]
    kz = np.fft.rfftfreq(bx.shape[1]) * bx.shape[1]
    mode = np.sqrt(ky[:, None] ** 2 + kz[None, :] ** 2)
    low_k = (mode >= 1.0) & (mode <= 16.0)
    weights = np.ones_like(power)
    weights[:, 1:-1] *= 2.0
    low_k_rms = float(
        np.sqrt(np.sum(power[low_k] * weights[low_k]))
        / (bx.shape[0] * bx.shape[1] * B0)
    )
    total_rms = float(
        np.sqrt(sum(np.var(component) for component in (bx, by, bz))) / B0
    )
    return anisotropy, beta, low_k_rms, total_rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-snapshots", type=int, default=100)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    moments = PICDataReader.find_files(
        str(args.data_dir / "pfd_moments.*_p*.h5")
    )
    fields = PICDataReader.find_files(str(args.data_dir / "pfd.*_p*.h5"))
    steps = sorted(set(moments) & set(fields))
    if not steps:
        raise SystemExit("No matching moment/field snapshots found.")
    if len(steps) > args.max_snapshots:
        indices = np.unique(
            np.linspace(0, len(steps) - 1, args.max_snapshots, dtype=int)
        )
        steps = [steps[index] for index in indices]

    active_omega = OMEGA_CE if DRIVEN_SPECIES == "electron" else OMEGA_CI
    rows = []
    for index, step in enumerate(steps, 1):
        anisotropy, beta, low_k, total = read_snapshot(
            moments[step], fields[step]
        )
        marginal = threshold(beta)
        rows.append({
            "step": step,
            "active_time": step * DT_CODE * active_omega,
            "omega_ci_t": step * DT_CODE * OMEGA_CI,
            "anisotropy": anisotropy,
            "beta_parallel": beta,
            "marginal_threshold": marginal,
            "instability_drive": drive(anisotropy, marginal),
            "low_k_delta_b_over_b0": low_k,
            "total_delta_b_over_b0": total,
        })
        print(f"{index}/{len(steps)} step={step} A={anisotropy:.5g}")

    csv_path = args.outdir / "lite_instability_evolution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    time = np.array([row["active_time"] for row in rows])
    anisotropy = np.array([row["anisotropy"] for row in rows])
    marginal = np.array([row["marginal_threshold"] for row in rows])
    instability = np.array([row["instability_drive"] for row in rows])
    low_k = np.array([row["low_k_delta_b_over_b0"] for row in rows])
    total = np.array([row["total_delta_b_over_b0"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(time, anisotropy, "o-", label=r"$T_\perp/T_\parallel$")
    axes[0].plot(time, marginal, "--", label="Marginal threshold")
    axes[0].axhline(1.0, color="0.4", linestyle=":")
    axes[0].set_ylabel("Anisotropy")
    axes[0].legend()
    axes[1].plot(time, instability, "o-", color="#c0392b")
    axes[1].axhline(0.0, color="0.4", linestyle=":")
    axes[1].set_ylabel("Unstable-side distance")
    axes[2].semilogy(time, np.maximum(low_k, 1e-12), "o-", label="Low-k RMS")
    axes[2].semilogy(time, np.maximum(total, 1e-12), "-", alpha=.6,
                    label="Total RMS")
    axes[2].set_ylabel(r"$\delta B/B_0$")
    axes[2].set_xlabel(
        r"$t\Omega_{ce}$" if DRIVEN_SPECIES == "electron"
        else r"$t\Omega_{ci}$"
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=.25)
    fig.suptitle(PROFILE_LABEL)
    fig.tight_layout()
    figure_path = args.outdir / "lite_instability_evidence.png"
    fig.savefig(figure_path, dpi=180)
    print(f"Saved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
