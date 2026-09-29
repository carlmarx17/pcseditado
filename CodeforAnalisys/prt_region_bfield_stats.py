#!/usr/bin/env python3
"""
prt_region_bfield_stats.py — Magnetic-field statistics inside the prt window
============================================================================
Time series and time average of the magnetic field restricted to the region
whose particles PSC writes to the prt files (the [lo, hi) cell box of
`OutputParticlesParams`, read from the prt file attributes), with the PIC
noise floor removed.

Quantities per snapshot, over the cells of the prt window:

    <|B|>          mean field magnitude
    B_rms          sqrt(<|B|^2>)
    dB_rms         sqrt(<|B - <B>_R|^2>), fluctuation about the window mean
    dB_par, dB_perp  same, split into B_z (along B0) and (B_x, B_y)
    <|B|^2>/2      mean magnetic energy density (code units)

Numerical-noise cleaning
------------------------
A PIC run carries a thermal fluctuation floor that scales as T/ppc, and
numerical (grid) heating makes T -- and hence the floor -- grow in time. Two
steps remove it:

  1. Spectral low-pass. B is filtered on the full periodic domain, keeping
     |k| <= k_c (default: half the grid Nyquist). The physical modes of the
     three instabilities live well below that (mirror/firehose at k d_i ~ 0.1-1,
     whistler at k d_e <~ 1); what is removed is grid-scale noise.

  2. Noise-floor subtraction in quadrature, dB^2_clean = dB^2 - dB^2_noise(t).
     `--floor tracked` (default) follows the floor in time: the band above
     k_c contains no physical signal, so its power P_hi(t) measures the noise
     level at every snapshot, heating included. The ratio
     r = P_lo / P_hi is calibrated in a quiet window before the instability
     grows, and the floor inside the kept band is r * P_hi(t).
     `--floor static` subtracts the constant quiet-window level instead.

The calibration window must end before linear growth starts. For the
whistler cases this is very early (the growth time is ~10-100 / Omega_ce,
i.e. a few snapshots); check the figure and pass --noise-window explicitly.

Usage:
    python prt_region_bfield_stats.py --data-dir ../build/src --outdir out
    python prt_region_bfield_stats.py --data-dir RUN --avg-window 100 158
    python prt_region_bfield_stats.py --data-dir RUN --noise-window 0 0.3 \\
        --kmax-de 2.0 --floor static
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps

ps.apply()

from data_reader import PICDataReader
from prt_region_field_cut import D_I, read_b_field, read_grid, read_prt_window
from psc_units import B0, PROFILE_LABEL, step_to_omegaci

COMPONENTS = ("par", "perp")


# ── Filtering ────────────────────────────────────────────────────────────────

def lowpass_mask(nz: int, ny: int, dz_de: float, dy_de: float,
                 kmax_de: float) -> np.ndarray:
    """Boolean mask of the kept Fourier modes, |k| d_e <= kmax_de.

    Arrays are (Nz, Ny), so axis 0 is k_z and axis 1 is k_y.
    """
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz_de)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy_de)
    kmag = np.sqrt(kz[:, None] ** 2 + ky[None, :] ** 2)
    return kmag <= kmax_de


def split_bands(arr: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(low-k, high-k) parts of a periodic 2D field."""
    spec = np.fft.fft2(arr)
    low = np.fft.ifft2(np.where(keep, spec, 0.0)).real
    return low, arr - low


# ── Per-snapshot statistics ──────────────────────────────────────────────────

def window_stats(b: dict, keep: np.ndarray, lo, hi) -> dict:
    """Field statistics over the prt window for one snapshot."""
    iy0, iy1 = int(lo[1]), int(hi[1])
    iz0, iz1 = int(lo[2]), int(hi[2])

    low, high = {}, {}
    for name in ("bx", "by", "bz"):
        low[name], high[name] = split_bands(b[name], keep)

    def crop(arr):
        return arr[iz0:iz1, iy0:iy1]

    def fluct_power(fields: dict) -> dict:
        # B0 lies along z: B_par = B_z, B_perp = (B_x, B_y).
        dz = crop(fields["bz"]) - crop(fields["bz"]).mean()
        dx = crop(fields["bx"]) - crop(fields["bx"]).mean()
        dy = crop(fields["by"]) - crop(fields["by"]).mean()
        return {"par": float(np.mean(dz**2)),
                "perp": float(np.mean(dx**2 + dy**2))}

    raw_b2 = crop(b["bx"])**2 + crop(b["by"])**2 + crop(b["bz"])**2
    low_b2 = crop(low["bx"])**2 + crop(low["by"])**2 + crop(low["bz"])**2

    return {
        "mean_B_raw": float(np.mean(np.sqrt(raw_b2))),
        "mean_B": float(np.mean(np.sqrt(low_b2))),
        "B_rms": float(np.sqrt(np.mean(low_b2))),
        "energy_density": float(0.5 * np.mean(low_b2)),
        "dB2_raw": fluct_power(b),
        "dB2_lo": fluct_power(low),
        "dB2_hi": fluct_power(high),
    }


# ── Noise floor ──────────────────────────────────────────────────────────────

def noise_floor(times: np.ndarray, records: list, window: tuple[float, float],
                mode: str) -> tuple[dict, dict]:
    """Noise floor dB^2_noise(t) per component, plus calibration info."""
    t0, t1 = window
    quiet = (times >= t0) & (times <= t1)
    if not np.any(quiet):
        raise ValueError(f"no snapshot inside the noise window [{t0}, {t1}]")

    floor, info = {}, {"n_snapshots": int(quiet.sum()), "mode": mode}
    for comp in COMPONENTS:
        lo_pow = np.array([r["dB2_lo"][comp] for r in records])
        hi_pow = np.array([r["dB2_hi"][comp] for r in records])
        if mode == "static":
            floor[comp] = np.full_like(lo_pow, lo_pow[quiet].mean())
        else:
            ratio = lo_pow[quiet].mean() / max(hi_pow[quiet].mean(), 1e-300)
            floor[comp] = ratio * hi_pow
            info[f"ratio_lo_over_hi_{comp}"] = float(ratio)
        # Growth leaking into the calibration window biases the floor high.
        drift = lo_pow[quiet].max() / max(lo_pow[quiet].min(), 1e-300)
        info[f"quiet_window_drift_{comp}"] = float(drift)
        if drift > 2.0:
            print(f"[WARN] dB2_{comp} changes by x{drift:.1f} inside the noise "
                  f"window: the instability is probably already growing there. "
                  f"Move --noise-window earlier.")
    return floor, info


# ── Time average ─────────────────────────────────────────────────────────────

def time_average(times: np.ndarray, series: dict, window: tuple[float, float]) -> dict:
    t0, t1 = window
    sel = (times >= t0) & (times <= t1)
    if not np.any(sel):
        raise ValueError(f"no snapshot inside the averaging window [{t0}, {t1}]")
    out = {"window_omegaci_t": [float(t0), float(t1)], "n_snapshots": int(sel.sum())}
    for name, values in series.items():
        vals = np.asarray(values)[sel]
        out[name] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                     "min": float(vals.min()), "max": float(vals.max())}
    return out


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_series(times, series, noise_window, avg_window, averages, b0,
                outname: Path) -> None:
    fig, (ax_db, ax_b) = plt.subplots(2, 1, figsize=(11.0, 9.0), sharex=True)

    ax_db.semilogy(times, series["dB_rms_raw"] / b0, color=ps.c("#999999"),
                   lw=1.2, label="raw")
    ax_db.semilogy(times, series["dB_rms_lowpass"] / b0, color=ps.c("#1f77b4"),
                   lw=1.4, label=r"low-pass $|k|\leq k_c$")
    ax_db.semilogy(times, np.sqrt(series["dB2_noise"]) / b0, color=ps.c("#d62728"),
                   lw=1.2, ls="--", label="noise floor")
    clean = series["dB_rms_clean"] / b0
    ax_db.semilogy(times, np.where(clean > 0, clean, np.nan),
                   color=ps.c("#111111"), lw=2.0, label="cleaned")
    ax_db.set_ylabel(r"$\delta B_{\rm rms}/B_0$")
    ax_db.legend(loc="lower right", framealpha=0.85, ncol=2)

    ax_b.plot(times, series["mean_B"] / b0, color=ps.c("#111111"), lw=1.8,
              label=r"$\langle|B|\rangle/B_0$")
    ax_b.plot(times, series["B_rms"] / b0, color=ps.c("#2ca02c"), lw=1.4,
              ls="--", label=r"$B_{\rm rms}/B_0$")
    ax_b.set_ylabel(r"$B/B_0$")
    ax_b.set_xlabel(r"$t\Omega_{ci}$")
    ax_b.legend(loc="best", framealpha=0.85)

    for ax in (ax_db, ax_b):
        ax.axvspan(*noise_window, color="red", alpha=0.10)
        ax.axvspan(*avg_window, color="gray", alpha=0.15)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    avg = averages["dB_rms_clean"]
    ax_db.axhline(avg["mean"] / b0, color=ps.c("#111111"), lw=1.0, ls=":")
    ax_db.set_title(
        rf"{PROFILE_LABEL} — prt window; "
        rf"$\overline{{\delta B_{{\rm rms}}}}/B_0 = {avg['mean'] / b0:.3g}"
        rf"\pm{avg['std'] / b0:.2g}$ (grey band)", pad=8)

    ps.save(fig, outname)


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = PICDataReader.discover_outputs(args.data_dir)
    field_map = outputs["fields"]
    if not field_map:
        print(f"ERROR: no pfd.*.h5 files found in {args.data_dir}")
        return 1

    particle_file = args.particles
    if particle_file is None and outputs["particles"]:
        series = next(iter(outputs["particles"].values()))
        particle_file = series[max(series)]
    lo, hi, source = read_prt_window(particle_file)

    steps = sorted(field_map)[::args.every]
    grid = read_grid(field_map[steps[0]])
    dz_de, dy_de = grid["dz_di"] * D_I, grid["dy_di"] * D_I
    k_nyquist = np.pi / max(dz_de, dy_de)
    kmax_de = args.kmax_de if args.kmax_de is not None else 0.5 * k_nyquist
    keep = lowpass_mask(grid["nz"], grid["ny"], dz_de, dy_de, kmax_de)

    print(f"Profile:        {PROFILE_LABEL}")
    print(f"prt window:     y cells [{lo[1]}, {hi[1]}), z cells [{lo[2]}, {hi[2]})  ({source})")
    print(f"Low-pass:       |k| d_e <= {kmax_de:.3f}  (Nyquist {k_nyquist:.3f}, "
          f"k_c d_i = {kmax_de * D_I:.1f})")
    print(f"Snapshots:      {len(steps)}")

    records = []
    for i, step in enumerate(steps):
        records.append(window_stats(read_b_field(field_map[step]), keep, lo, hi))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(steps)}")

    times = np.array([step_to_omegaci(s) for s in steps])

    # Step 0 is the exact uniform initial field (dB = 0), not noise.
    positive = times[times > 0]
    noise_window = (tuple(args.noise_window) if args.noise_window
                    else (positive[0], positive[min(2, positive.size - 1)]))
    avg_window = (tuple(args.avg_window) if args.avg_window
                  else (0.5 * times[-1], times[-1]))

    floor, floor_info = noise_floor(times, records, noise_window, args.floor)

    dB2_lo = sum(np.array([r["dB2_lo"][c] for r in records]) for c in COMPONENTS)
    dB2_noise = sum(floor[c] for c in COMPONENTS)
    clean = {c: np.clip(np.array([r["dB2_lo"][c] for r in records]) - floor[c], 0, None)
             for c in COMPONENTS}

    series = {
        "mean_B_raw": np.array([r["mean_B_raw"] for r in records]),
        "mean_B": np.array([r["mean_B"] for r in records]),
        "B_rms": np.array([r["B_rms"] for r in records]),
        "energy_density": np.array([r["energy_density"] for r in records]),
        "dB_rms_raw": np.sqrt(sum(np.array([r["dB2_raw"][c] for r in records])
                                  for c in COMPONENTS)),
        "dB_rms_lowpass": np.sqrt(dB2_lo),
        "dB2_noise": dB2_noise,
        "dB_rms_clean": np.sqrt(clean["par"] + clean["perp"]),
        "dB_par_clean": np.sqrt(clean["par"]),
        "dB_perp_clean": np.sqrt(clean["perp"]),
    }

    averages = time_average(times, series, avg_window)

    prefix = args.prefix
    # Field amplitudes are written normalised to B0; the two squared/energy
    # columns stay in code units.
    unscaled = ("energy_density", "dB2_noise")
    csv_path = outdir / f"{prefix}prt_region_bfield_timeseries.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "omegaci_t"] +
                        [n if n in unscaled else f"{n}_over_B0" for n in series])
        for i, step in enumerate(steps):
            row = [step, f"{times[i]:.6f}"]
            for n, values in series.items():
                scale = 1.0 if n in unscaled else args.B0
                row.append(f"{values[i] / scale:.8e}")
            writer.writerow(row)

    summary = {
        "profile": PROFILE_LABEL,
        "data_dir": str(Path(args.data_dir).resolve()),
        "prt_window": {"lo": [int(v) for v in lo], "hi": [int(v) for v in hi],
                       "source": source},
        "B0": args.B0,
        "lowpass_kmax_de": float(kmax_de),
        "k_nyquist_de": float(k_nyquist),
        "noise_window_omegaci_t": [float(v) for v in noise_window],
        "noise_floor": floor_info,
        "time_average": averages,
        "time_average_over_B0": {
            n: {k: v / args.B0 for k, v in averages[n].items()}
            for n in ("mean_B", "B_rms", "dB_rms_clean", "dB_par_clean",
                      "dB_perp_clean", "dB_rms_raw")
        },
    }
    json_path = outdir / f"{prefix}prt_region_bfield_summary.json"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_series(times, series, noise_window, avg_window, averages, args.B0,
                outdir / f"{prefix}prt_region_bfield_timeseries.png")

    t0, t1 = avg_window
    print(f"Time average over Omega_ci t in [{t0:.1f}, {t1:.1f}] "
          f"({averages['n_snapshots']} snapshots):")
    for n in ("mean_B", "B_rms", "dB_rms_clean", "dB_par_clean", "dB_perp_clean"):
        a = averages[n]
        print(f"  {n:<14} / B0 = {a['mean'] / args.B0:.5g} +- {a['std'] / args.B0:.2g}")
    print(f"Outputs: {csv_path.name}, {json_path.name}, prt_region_bfield_timeseries.png")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mean, rms and fluctuation of B inside the prt window, "
                    "time-averaged and with the PIC noise floor removed.")
    parser.add_argument("--data-dir", default="../build/src")
    parser.add_argument("--particles", default=None,
                        help="prt file to read lo/hi from (default: last one)")
    parser.add_argument("--outdir", default="prt_region_bfield")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--every", type=int, default=1,
                        help="use every N-th field snapshot")
    parser.add_argument("--kmax-de", type=float, default=None,
                        help="low-pass cutoff |k| d_e (default: half Nyquist)")
    parser.add_argument("--noise-window", type=float, nargs=2, metavar=("T0", "T1"),
                        help="quiet window in Omega_ci t used to calibrate the "
                             "noise floor (default: first three snapshots after t=0)")
    parser.add_argument("--avg-window", type=float, nargs=2, metavar=("T0", "T1"),
                        help="time-averaging window in Omega_ci t "
                             "(default: second half of the run)")
    parser.add_argument("--floor", choices=["tracked", "static"], default="tracked",
                        help="tracked: floor follows the high-k noise power "
                             "(accounts for numerical heating); static: constant")
    parser.add_argument("--B0", type=float, default=B0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
