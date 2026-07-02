#!/usr/bin/env python3
"""
compare_physical_cases.py
=========================
Compare integrated diagnostics across PSC cases, e.g. Maxwellian vs Kappa.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
    ax.grid(True, color=GRID_CLR, alpha=0.22, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_case_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(raw).expanduser()
    return path.name, path


def load_case(name: str, path: Path) -> dict:
    rows = {}
    for table_name, table_path in [
        ("anisotropy", path / "anisotropy_table.csv"),
        ("fit", path / "fit_metrics.csv"),
        ("field", path / "field_fluctuation_table.csv"),
        ("energy", path / "energy_table.csv"),
    ]:
        for row in _read_csv(table_path):
            step = int(_to_float(row.get("step"), -1))
            if step < 0:
                continue
            rows.setdefault(step, {"case": name, "step": step})
            rows[step].update({f"{table_name}_{k}": v for k, v in row.items()})

    gamma_rows = _read_csv(path / "growth_rate_summary.csv")
    gamma = _to_float(gamma_rows[0].get("gamma")) if gamma_rows else np.nan
    merged = []
    for step in sorted(rows):
        row = rows[step]
        time = (
            row.get("anisotropy_omega_ci_t")
            or row.get("fit_omega_ci_t")
            or row.get("field_omega_ci_t")
            or row.get("energy_omega_ci_t")
        )
        merged.append({
            "case": name,
            "step": step,
            "omega_ci_t": _to_float(time),
            "A_i": _to_float(row.get("anisotropy_A_i")),
            "R_i": _to_float(row.get("anisotropy_R_i")),
            "beta_parallel_i": _to_float(row.get("anisotropy_beta_parallel_i")),
            "delta_B_rms": _to_float(row.get("field_delta_B_rms")),
            "delta_B_rms_over_B0": _to_float(row.get("field_delta_B_rms_over_B0")),
            "gamma": gamma,
            "kappa_fit": _to_float(row.get("fit_kappa_fit")),
            "F_supra": _to_float(row.get("fit_suprathermal_fraction")),
            "E_B": _to_float(row.get("energy_E_B")),
            "E_total": _to_float(row.get("energy_E_total")),
            "q_parallel": _to_float(row.get("anisotropy_q_parallel_particle")),
            "q_perp": _to_float(row.get("anisotropy_q_perp_particle")),
        })
    return {"name": name, "rows": merged, "gamma": gamma}


def plot_timeseries(cases: list[dict], ykeys: list[str], labels: list[str], path: Path, title: str, yscale=None):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax)
    for case in cases:
        rows = case["rows"]
        if not rows:
            continue
        t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
        for ykey, label in zip(ykeys, labels):
            y = np.array([r[ykey] for r in rows], dtype=float)
            if np.any(np.isfinite(y)):
                ax.plot(t, y, marker="o", lw=1.7, ms=3.5, label=f"{case['name']} {label}")
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(", ".join(labels), color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _save(fig, path)


def plot_growth_bars(cases: list[dict], path: Path):
    names = [case["name"] for case in cases]
    gamma = np.array([case["gamma"] for case in cases], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax)
    ax.bar(names, gamma, color="#58a6ff")
    ax.set_ylabel(r"$\gamma$", color=TEXT_CLR)
    ax.set_title("Growth-rate comparison", color=TEXT_CLR, fontweight="bold")
    _save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Compare physical diagnostics between PSC cases.")
    parser.add_argument("cases", nargs="+", help="Case directories, optionally NAME=/path/to/09_physical_diagnostics")
    parser.add_argument("--outdir", default="comparison_physical", help="Output directory.")
    args = parser.parse_args()

    cases = [load_case(*parse_case_arg(raw)) for raw in args.cases]
    outdir = Path(args.outdir)
    rows = [row for case in cases for row in case["rows"]]
    _write_csv(outdir / "comparison_kappa_vs_maxwellian.csv", rows)

    plot_timeseries(cases, ["A_i", "R_i"], [r"$A_i$", r"$R_i$"],
                    outdir / "comparison_anisotropy.png", "Anisotropy comparison")
    plot_timeseries(cases, ["delta_B_rms_over_B0"], [r"$\delta B_{\rm rms}/B_0$"],
                    outdir / "comparison_deltaB.png", "Magnetic-fluctuation comparison", yscale="log")
    plot_growth_bars(cases, outdir / "comparison_growth_rate.png")
    plot_timeseries(cases, ["E_B", "E_total"], [r"$E_B$", r"$E_{\rm total}$"],
                    outdir / "comparison_energy.png", "Energy comparison")
    plot_timeseries(cases, ["q_parallel", "q_perp"], [r"$q_\parallel$", r"$q_\perp$"],
                    outdir / "comparison_heat_flux.png", "Heat-flux comparison")
    print(f"Comparison written to {outdir}")


if __name__ == "__main__":
    main()
