#!/usr/bin/env python3
"""
field_residuals.py — div B, Gauss's law and charge continuity over a run
========================================================================
DiagEnergies (``energy_conservation.py``) measures the global energy budget;
it says nothing about the constraints the field solver must keep. This
script collects the three residuals that do:

* **div B** from the field snapshots. PSC writes B with its Yee staggering
  (``hx_fc`` etc., face-centred): B_x at (i, j+1/2, k+1/2), B_y at
  (i+1/2, j, k+1/2), B_z at (i+1/2, j+1/2, k). The discrete divergence at the
  cell centre is therefore the *forward* difference of each component along
  its own axis, which the Yee scheme keeps at round-off. It is reported as
  max|div B| dx / B0 and its RMS; a value far above the single-precision
  round-off of the output (~1e-6) means the snapshot is not a raw Yee field
  or the solver is not divergence-free.
* **Gauss's law** div E = rho, and **charge continuity** d rho/dt + div J = 0,
  from PSC's own checks (``psc::checks::gauss`` every 100 steps, after the
  Marder correction, and ``psc::checks::continuity``). PSC prints
  ``gauss: max_err = X (thres Y)`` and ``continuity: max_err = ...`` in the
  job log, right after the ``**** Step N / M`` line of that step; this
  script parses those lines. The runs abort if a threshold (1e-4) is
  exceeded, so a complete log is itself evidence of the bound; the time
  series shows how close to it the run stayed and whether the Marder
  correction (every 100 steps, diffusion 0.9, 3 passes) was keeping up.

Outputs (``--outdir``, normally ``09_physical_diagnostics/``):
  field_residuals_divB.csv, field_residuals_log.csv, field_residuals_summary.json,
  field_residuals.png

Usage:
    python field_residuals.py --data-dir RUN --outdir out --logs "RUN/*.out"
"""

from __future__ import annotations

import argparse
import csv
import glob
from analysis_contract import strict_dumps
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from data_reader import PICDataReader
from psc_units import B0, DX_DE, step_to_omegaci

ps.apply()

STEP_RE = re.compile(r"\*\*\*\* Step (\d+) / (\d+)")
CHECK_RE = re.compile(r"^(gauss|continuity): max_err = ([0-9eE+\-.naninf]+) \(thres ([0-9eE+\-.]+)\)")


def div_b_yee(bx, by, bz, dy: float, dz: float) -> np.ndarray:
    """Discrete divergence of a face-centred B on a (Nz, Ny) periodic yz grid."""
    del bx  # d/dx = 0 in the 2D runs
    return ((np.roll(bz, -1, axis=0) - bz) / dz + (np.roll(by, -1, axis=1) - by) / dy)


def div_b_rows(field_files: dict[int, str]) -> list[dict]:
    rows = []
    for step, path in sorted(field_files.items()):
        data = PICDataReader.read_multiple_fields_3d(
            path, "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
        bx, by, bz = (PICDataReader.flatten_2d_slice(data[f"h{c}_fc/p0/3d"]).astype(float)
                      for c in "xyz")
        div = div_b_yee(bx, by, bz, DX_DE, DX_DE) * DX_DE / abs(B0)
        rows.append({
            "step": step, "omega_ci_t": step_to_omegaci(step),
            "divB_max_dx_over_B0": float(np.max(np.abs(div))),
            "divB_rms_dx_over_B0": float(np.sqrt(np.mean(div * div))),
        })
    return rows


def parse_logs(paths: list[Path]) -> tuple[list[dict], dict]:
    """Gauss/continuity max_err per step from PSC job logs (restarts merged)."""
    by_step: dict[int, dict] = {}
    thresholds: dict[str, float] = {}
    nmax = None
    for path in paths:
        step = None
        with path.open(errors="replace") as handle:
            for line in handle:
                m = STEP_RE.search(line)
                if m:
                    # The banner prints timestep+1 before the step is taken;
                    # the checks inside that step refer to that same number.
                    step = int(m.group(1))
                    nmax = int(m.group(2))
                    continue
                m = CHECK_RE.match(line.strip())
                if m and step is not None:
                    name, value, thres = m.group(1), float(m.group(2)), float(m.group(3))
                    by_step.setdefault(step, {"step": step})[f"{name}_max_err"] = value
                    thresholds[name] = thres
    rows = []
    for step in sorted(by_step):
        row = by_step[step]
        rows.append({"step": step, "omega_ci_t": step_to_omegaci(step),
                     "gauss_max_err": row.get("gauss_max_err", np.nan),
                     "continuity_max_err": row.get("continuity_max_err", np.nan)})
    return rows, {"thresholds": thresholds, "nmax_from_log": nmax}


def summarize(div_rows: list[dict], log_rows: list[dict], meta: dict,
              log_paths: list[Path]) -> dict:
    def peak(rows, key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals)) if vals.size else None
    last_step = log_rows[-1]["step"] if log_rows else None
    nmax = meta.get("nmax_from_log")
    # PSC runs the checks every `continuity_every` steps, so the last check of a
    # finished run is the last multiple of that cadence, not necessarily nmax.
    steps = np.array([r["step"] for r in log_rows], dtype=float)
    cadence = float(np.median(np.diff(steps))) if steps.size > 1 else None
    return {
        "scientific_status": "UNVERIFIED",
        "reason": "Assess each residual against its declared threshold and complete log coverage; divB tolerance not specified",
        "divB_snapshots": len(div_rows),
        "divB_max_dx_over_B0": peak(div_rows, "divB_max_dx_over_B0"),
        "log_files": [str(p) for p in log_paths],
        "gauss_checks": int(sum(np.isfinite(r["gauss_max_err"]) for r in log_rows)),
        "gauss_max_err": peak(log_rows, "gauss_max_err"),
        "continuity_checks": int(sum(np.isfinite(r["continuity_max_err"]) for r in log_rows)),
        "continuity_max_err": peak(log_rows, "continuity_max_err"),
        "thresholds": meta.get("thresholds", {}),
        "last_logged_step": last_step,
        "nmax_from_log": meta.get("nmax_from_log"),
        "check_cadence_steps": cadence,
        "log_reaches_nmax": bool(last_step is not None and nmax
                                 and (last_step >= nmax
                                      or (cadence is not None and last_step + cadence > nmax))),
    }


def plot(div_rows: list[dict], log_rows: list[dict], meta: dict, outdir: Path):
    panels = int(bool(div_rows)) + int(bool(log_rows))
    if panels == 0:
        return
    fig, axes = plt.subplots(panels, 1, figsize=(8.4, 3.6 * panels + 0.6), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    i = 0
    if div_rows:
        ax = axes[i]
        i += 1
        t = [r["omega_ci_t"] for r in div_rows]
        ax.semilogy(t, [r["divB_max_dx_over_B0"] for r in div_rows], "o-", ms=3,
                    color=ps.c("#58a6ff"), label=r"$\max|\nabla\cdot\mathbf{B}|\,\Delta x/B_0$")
        ax.semilogy(t, [r["divB_rms_dx_over_B0"] for r in div_rows], "s-", ms=3,
                    color=ps.c("#56d364"), label=r"RMS $|\nabla\cdot\mathbf{B}|\,\Delta x/B_0$")
        ax.set_ylabel("div B residual")
        ps.style_axes(ax)
        ps.legend(ax, fontsize=10)
    if log_rows:
        ax = axes[i]
        t = np.array([r["omega_ci_t"] for r in log_rows])
        for key, color, label in (("gauss_max_err", "#ff7b72", r"Gauss: $\max|\nabla\cdot\mathbf{E}-\rho|$"),
                                  ("continuity_max_err", "#d2a8ff",
                                   r"continuity: $\max|\partial_t\rho+\nabla\cdot\mathbf{J}|$")):
            y = np.array([r[key] for r in log_rows], dtype=float)
            ok = np.isfinite(y) & (y > 0)
            if np.any(ok):
                ax.semilogy(t[ok], y[ok], ".", ms=3, color=ps.c(color), label=label)
        for j, (name, thres) in enumerate(meta.get("thresholds", {}).items()):
            ax.axhline(thres, color=ps.MUTED_CLR, ls="--", lw=1.0,
                       label="PSC abort threshold" if j == 0 else None)
        ax.set_ylabel("max error [code]")
        ps.style_axes(ax)
        ps.legend(ax, fontsize=10)
    axes[-1].set_xlabel(r"$t\,\Omega_{ci}$")
    axes[0].set_title("Field-solver constraints", fontsize=13)
    fig.tight_layout()
    ps.save(fig, outdir / "field_residuals.png")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="div B, Gauss and continuity residuals of a PSC run.")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--fields", default=None, help="pfd glob (default: DATA_DIR/pfd.*_p*.h5)")
    p.add_argument("--logs", nargs="*", default=None,
                   help="PSC job logs (globs allowed). Default: DATA_DIR/*.out, *.log, *.txt")
    p.add_argument("--every", type=int, default=1,
                   help="use every N-th field snapshot for div B (default 1)")
    p.add_argument("--outdir", default="physical_diagnostics")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fields = PICDataReader.find_files(args.fields or str(data_dir / "pfd.*_p*.h5"))
    fields = dict(list(sorted(fields.items()))[:: max(args.every, 1)])
    div_rows = div_b_rows(fields) if fields else []

    patterns = args.logs or [str(data_dir / ext) for ext in ("*.out", "*.log", "*.txt")]
    log_paths = sorted({Path(m) for pat in patterns for m in glob.glob(pat)})
    log_rows, meta = parse_logs(log_paths) if log_paths else ([], {})
    if not log_rows:
        print("[MISSING] no 'gauss: max_err' lines found in the job logs; pass --logs with "
              "the SLURM output of the run to include Gauss/continuity residuals.")

    write_csv(outdir / "field_residuals_divB.csv", div_rows)
    write_csv(outdir / "field_residuals_log.csv", log_rows)
    summary = summarize(div_rows, log_rows, meta, log_paths)
    (outdir / "field_residuals_summary.json").write_text(strict_dumps(summary, indent=2))
    plot(div_rows, log_rows, meta, outdir)
    print(strict_dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
