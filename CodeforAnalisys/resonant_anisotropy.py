#!/usr/bin/env python3
"""
resonant_anisotropy.py — where in parallel velocity the anisotropy sits
=======================================================================
For electromagnetic modes that propagate along B0 the growth rate depends on
the distribution only through two functions of the parallel velocity,

    F(v_par) = int f d^2v_perp,      W(v_par) = int (v_perp^2 / 2) f d^2v_perp,

and a mode resonant with the ions at v_res = (omega_r - Omega_ci)/k_par grows
if (Kennel & Petschek 1966)

    A(v_res) > 1 / (1 - omega_r/Omega_ci),     A(v) = -(dW/dv) / (v F).

A(v) is the anisotropy as the wave sees it. It equals T_perp/T_par at every v
for a bi-Maxwellian and for a bi-kappa, so both start flat at the loaded
value; and the global anisotropy is its average weighted by the parallel
energy, T_perp/T_par = int A v^2 F dv / int v^2 F dv. The tail of a kappa
distribution carries a large share of that weight and is far from the
resonance, so a wave can exhaust the free energy at v_res and leave the global
anisotropy above the marginal value (trajectory_linear_theory.py measures that
excess). This script measures A(v) and says where the excess is.

The derivative is not taken on a histogram. For a smooth window b(|v|),
integration by parts gives

    <A>_b = sum w e_perp (b + |v| b') / sum w v^2 b,     e_perp = v_perp^2 / 2,

the average of A weighted by v^2 F b, with a delta-method error. Gaussian
windows give the curve; four smooth bands that add up to one give a
decomposition whose shares sum exactly to the global anisotropy.

Two steps:

    measure   reads the particle and field snapshots of one run (raw data),
              velocities in the frame of the local field, and writes
              resonant_anisotropy.csv next to the other particle products.
    plot      from those products of one or several runs:
              resonant_anisotropy.png and resonant_anisotropy_bands.csv.

Usage:
    PSC_PROFILE=<case> python resonant_anisotropy.py measure --data-dir RUN --outdir OUT/<case>/03_particles
    python resonant_anisotropy.py plot ROOT_MAXW ROOT_K5 ROOT_K3 --outdir OUT
"""

from __future__ import annotations

import argparse
import csv
import os
from math import erf, sqrt
from pathlib import Path

import numpy as np

PRODUCT = "resonant_anisotropy.csv"
#: Window centres and width of the curve, and band edges, in units of the
#: initial parallel thermal speed sigma_par0 = sqrt(T_par0/m).
CENTRES = np.arange(0.4, 5.001, 0.2)
WINDOW_WIDTH = 0.2
BAND_EDGES = (1.0, 2.0, 3.0)
BAND_SMOOTHING = 0.15
SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
BAND_COLOURS = ("#0072B2", "#56B4E9", "#E69F00", "#D55E00")

_erf = np.vectorize(erf, otypes=[float])


def gaussian_window(centre: float, width: float):
    """b(x) and b'(x) of a Gaussian bump (x = |v_par|)."""
    def b(x):
        return np.exp(-0.5 * ((x - centre) / width) ** 2)

    def db(x):
        return -(x - centre) / width ** 2 * b(x)
    return b, db


def band_windows(edges=BAND_EDGES, smoothing: float = BAND_SMOOTHING):
    """Smooth bands [0, e1], [e1, e2], ..., [e_n, inf) that add up to one for x >= 0."""
    def step(edge):
        return (lambda x: 0.5 * (1.0 + _erf((x - edge) / (sqrt(2.0) * smoothing))),
                lambda x: np.exp(-0.5 * ((x - edge) / smoothing) ** 2) / (smoothing * sqrt(2.0 * np.pi)))
    steps = [step(edge) for edge in edges]
    bands = []
    for j in range(len(edges) + 1):
        lower = steps[j - 1] if j > 0 else None
        upper = steps[j] if j < len(edges) else None

        def b(x, lower=lower, upper=upper):
            return (lower[0](x) if lower else 1.0) - (upper[0](x) if upper else 0.0)

        def db(x, lower=lower, upper=upper):
            return (lower[1](x) if lower else 0.0) - (upper[1](x) if upper else 0.0)
        bands.append((0.0 if j == 0 else edges[j - 1], np.inf if j == len(edges) else edges[j], b, db))
    return bands


def window_anisotropy(u, e_perp, w, b, db) -> dict:
    """<A> over the window b: value, delta-method error, weight shares and effective count.

    ``u``: parallel velocity (centred), ``e_perp``: v_perp^2/2, in the same
    velocity unit squared; ``w``: particle weights.
    """
    x = np.abs(u)
    bx = b(x)
    num = e_perp * (bx + x * db(x))
    den = u ** 2 * bx
    total_den = float(np.sum(w * den))
    if not total_den > 0:
        return {"A": float("nan"), "A_err": float("nan"), "energy_share": 0.0, "number_share": 0.0, "n_eff": 0.0}
    ratio = float(np.sum(w * num)) / total_den
    err = float(np.sqrt(np.sum((w * (num - ratio * den)) ** 2))) / total_den
    wb = w * bx
    return {"A": ratio, "A_err": err,
            "energy_share": total_den / float(np.sum(w * u ** 2)),
            "number_share": float(np.sum(wb)) / float(np.sum(w)),
            "n_eff": float(np.sum(wb)) ** 2 / max(float(np.sum(wb ** 2)), 1e-300)}


def analyse_sample(u, e_perp, w) -> list[dict]:
    """Rows (kind, centre, lo, hi, A, ...) of one snapshot: the curve, the bands and the global value."""
    rows = [{"kind": "global", "centre": float("nan"), "lo": 0.0, "hi": float("inf"),
             **window_anisotropy(u, e_perp, w, lambda x: np.ones_like(x), lambda x: np.zeros_like(x))}]
    for centre in CENTRES:
        rows.append({"kind": "window", "centre": float(centre), "lo": float(centre - WINDOW_WIDTH),
                     "hi": float(centre + WINDOW_WIDTH),
                     **window_anisotropy(u, e_perp, w, *gaussian_window(centre, WINDOW_WIDTH))})
    for lo, hi, b, db in band_windows():
        rows.append({"kind": "band", "centre": float("nan"), "lo": lo, "hi": hi,
                     **window_anisotropy(u, e_perp, w, b, db)})
    return rows


FIELDS = ("step", "omega_ci_t", "species", "frame", "sigma_par0_over_vA", "sigma_par_over_sigma_par0",
          "kind", "centre", "lo", "hi", "A", "A_err", "energy_share", "number_share", "n_eff")


def measure(args) -> int:
    """Resonant anisotropy of every selected particle snapshot of one run (raw data)."""
    import psc_units as pu
    import vdf_spatial as vs
    from data_reader import PICDataReader

    outputs = PICDataReader.discover_outputs(args.data_dir)
    fields, particles = outputs["fields"], outputs["particles"]
    if not fields or not particles:
        print(f"ERROR: missing pfd.* or prt_*.* in {args.data_dir}")
        return 1
    series = next(iter(particles.values()))
    steps = sorted(series)
    if args.max_snapshots and len(steps) > args.max_snapshots:
        pick = np.linspace(0, len(steps) - 1, args.max_snapshots).round().astype(int)
        steps = [steps[i] for i in dict.fromkeys(pick)]
    field_steps = np.array(sorted(fields))
    sigma0 = float(pu.VTH_I_PAR if args.species == "ion" else pu.VTH_E_PAR)   # in c, like v = u/gamma

    rows = []
    for step in steps:
        near = int(field_steps[np.argmin(np.abs(field_steps - step))])
        if abs(near - step) > args.max_step_mismatch:
            print(f"[WARN] step {step}: closest field at {near}; skipped.")
            continue
        part = vs.load_particles(series[step], args.species, args.max_particles)
        vel = vs.local_frame_velocities(part, vs.load_b_field(fields[near]))
        w = part["w"]
        centred = [vel[key] - np.average(vel[key], weights=w) for key in ("v_par", "v_perp1", "v_perp2")]
        u = centred[0] / sigma0
        e_perp = 0.5 * (centred[1] ** 2 + centred[2] ** 2) / sigma0 ** 2
        common = {"step": step, "omega_ci_t": pu.step_to_omegaci(step), "species": args.species,
                  "frame": "local field", "sigma_par0_over_vA": sigma0 / pu.VA,
                  "sigma_par_over_sigma_par0": float(np.sqrt(np.average(u ** 2, weights=w)))}
        sample = analyse_sample(u, e_perp, w)
        rows += [{**common, **row} for row in sample]
        bands = [r for r in sample if r["kind"] == "band"]
        print(f"t Omega_ci = {common['omega_ci_t']:6.1f}: A = {sample[0]['A']:.3f}; bands "
              + ", ".join(f"{r['A']:.2f} ({100 * r['energy_share']:.0f} %)" for r in bands))
    if not rows:
        print("ERROR: no snapshot analysed.")
        return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{args.prefix}{PRODUCT}", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{row[key]:.6g}" if isinstance(row[key], float) else row[key]) for key in FIELDS})
    print(f"wrote {outdir / (args.prefix + PRODUCT)} ({len(steps)} snapshots)")
    return 0


# ── Figure from the products ────────────────────────────────────────────────

def load_product(root: Path) -> dict | None:
    path = next((p for p in (root / "03_particles" / PRODUCT, root / PRODUCT) if p.exists()), None)
    if path is None:
        return None
    with open(path, newline="") as handle:
        rows = [{key: (value if key in ("species", "frame", "kind") else float(value))
                 for key, value in row.items()} for row in csv.DictReader(handle)]
    times = sorted({row["omega_ci_t"] for row in rows})
    return {"root": root, "name": root.name, "rows": rows, "times": times,
            "sigma0": rows[0]["sigma_par0_over_vA"]}


def mode_markers(root: Path) -> dict:
    """Marginal anisotropy and resonant speeds (in v_A) of the dominant mode, when the run products allow."""
    try:
        import paper_figures as pf
        import trajectory_linear_theory as tlt
        run = pf.load_run(root)
        profile = run["profile"]
        state = {"beta_i": profile["beta_i_par"], "a_i": profile["Ti_perp_over_Ti_par"],
                 "beta_e": profile["beta_e_par"], "a_e": profile["Te_perp_over_Te_par"],
                 "mass_ratio": profile["mass_ratio"], "c_over_va": tlt.c_over_va(profile)}
        root0 = tlt.linear_root(state, run["kappa"], run["k_mode"])
        a_marginal, _ = tlt.marginal_anisotropy(state, run["kappa"], run["k_mode"])
        return {"label": run["label"], "a_marginal": a_marginal,
                "v_res": ((1.0 - float(np.real(root0))) / run["k_mode"], 1.0 / (a_marginal * run["k_mode"]))}
    except Exception as exc:                     # products of the field pass not there: curves only
        print(f"[INFO] {root.name}: no mode markers ({type(exc).__name__}: {exc})")
        return {"label": root.name, "a_marginal": float("nan"), "v_res": None}


def band_label(lo: float, hi: float, sigma0: float) -> str:
    if lo == 0:
        return rf"$|v_\parallel| < {hi * sigma0:.1f}\,v_A$"
    if not np.isfinite(hi):
        return rf"$|v_\parallel| > {lo * sigma0:.1f}\,v_A$"
    return rf"${lo * sigma0:.1f}$–${hi * sigma0:.1f}\,v_A$"


def plot(args) -> int:
    os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
    import plot_style as ps
    ps.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    products = [p for p in (load_product(root) for root in args.runs) if p is not None]
    if not products:
        print(f"ERROR: no {PRODUCT} under the given roots (run 'measure' first).")
        return 1
    markers = [mode_markers(p["root"]) for p in products]
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(products), figsize=(max(4.3 * len(products), 8.0), 6.8), sharey="row",
                             sharex="row", squeeze=False, layout="constrained")
    cmap = plt.get_cmap(ps.CMAP_SEQUENTIAL)
    summary, time_handles = [], []
    for i, (prod, mark) in enumerate(zip(products, markers)):
        top, bottom = axes[0, i], axes[1, i]
        shown = [prod["times"][j] for j in dict.fromkeys(
            np.linspace(0, len(prod["times"]) - 1, min(5, len(prod["times"]))).round().astype(int))]
        for j, t in enumerate(shown):
            rows = [r for r in prod["rows"] if r["kind"] == "window" and r["omega_ci_t"] == t
                    and r["n_eff"] >= args.min_count and np.isfinite(r["A_err"]) and r["A_err"] < 0.5]
            x = np.array([r["centre"] for r in rows]) * prod["sigma0"]
            a, err = np.array([r["A"] for r in rows]), np.array([r["A_err"] for r in rows])
            colour = cmap(0.08 + 0.8 * j / max(len(shown) - 1, 1))
            top.fill_between(x, a - err, a + err, color=colour, alpha=0.2, lw=0)
            top.plot(x, a, color=colour, lw=1.6)
            if i == 0:
                time_handles.append(Line2D([], [], color=colour, lw=1.6, label=rf"$t\,\Omega_{{ci}}={t:.0f}$"))
        if mark["v_res"] is not None:
            top.axvspan(min(mark["v_res"]), max(mark["v_res"]), color=ps.MUTED_CLR, alpha=0.12, lw=0)
        for ax in (top, bottom):
            if np.isfinite(mark["a_marginal"]):
                ax.axhline(mark["a_marginal"], color=ps.MUTED_CLR, lw=1.2, ls="--")
        bands = sorted({(r["lo"], r["hi"]) for r in prod["rows"] if r["kind"] == "band"})
        for (lo, hi), colour in zip(bands, BAND_COLOURS):
            rows = sorted((r for r in prod["rows"] if r["kind"] == "band" and (r["lo"], r["hi"]) == (lo, hi)),
                          key=lambda r: r["omega_ci_t"])
            t = np.array([r["omega_ci_t"] for r in rows])
            a, err = np.array([r["A"] for r in rows]), np.array([r["A_err"] for r in rows])
            bottom.fill_between(t, a - err, a + err, color=ps.c(colour), alpha=0.2, lw=0)
            bottom.plot(t, a, color=ps.c(colour), lw=1.6, marker="o", ms=3)
            last = rows[-1]
            summary.append({"run": prod["name"], "band_lo_vA": lo * prod["sigma0"], "band_hi_vA": hi * prod["sigma0"],
                            "A_initial": rows[0]["A"], "A_final": last["A"], "A_final_err": last["A_err"],
                            "energy_share_initial": rows[0]["energy_share"], "energy_share_final": last["energy_share"],
                            "a_marginal": mark["a_marginal"],
                            "excess_contribution": last["energy_share"] * (last["A"] - mark["a_marginal"])})
        top.set_title(mark["label"], fontsize=12)
        top.set_xlabel(r"$|v_\parallel|/v_A$ (local-field frame)")
        bottom.set_xlabel(r"$t\,\Omega_{ci}$")
    axes[0, 0].set_ylabel(r"$A(v_\parallel)$")
    axes[1, 0].set_ylabel(r"$A$, band average")
    band_handles = [Line2D([], [], color=ps.c(colour), lw=1.6, marker="o", ms=3,
                           label=band_label(lo, hi, products[0]["sigma0"]))
                    for (lo, hi), colour in zip(bands, BAND_COLOURS)]
    extra = []
    if any(np.isfinite(m["a_marginal"]) for m in markers):
        extra = [Line2D([], [], color=ps.MUTED_CLR, lw=1.2, ls="--", label="marginal value of the dominant mode"),
                 Line2D([], [], color=ps.MUTED_CLR, lw=6, alpha=0.2,
                        label=r"its resonant $|v_\parallel|$, linear to marginal")]
    fig.legend(handles=time_handles + extra + band_handles, loc="outside lower center", ncol=4, frameon=False,
               fontsize=9.5)
    fig.suptitle("Anisotropy as the wave sees it (flat at the loaded value for a bi-Maxwellian and a bi-kappa)",
                 fontsize=12)
    ps.save(fig, args.outdir / "resonant_anisotropy.png")

    with open(args.outdir / "resonant_anisotropy_bands.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({key: (f"{value:.6g}" if isinstance(value, float) else value) for key, value in row.items()})
    for prod in products:
        mine = [row for row in summary if row["run"] == prod["name"]]
        print(f"{prod['name']}: final A by band " + ", ".join(
            f"{row['A_final']:.2f} ({100 * row['energy_share_final']:.0f} % of the parallel energy)" for row in mine))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="resonant anisotropy of one run, from its particle and field snapshots")
    m.add_argument("--data-dir", required=True)
    m.add_argument("--outdir", required=True)
    m.add_argument("--prefix", default="")
    m.add_argument("--species", choices=["ion", "electron"], default="ion")
    m.add_argument("--max-snapshots", type=int, default=24)
    m.add_argument("--max-particles", type=int, default=4_000_000)
    m.add_argument("--max-step-mismatch", type=int, default=3000)
    m.set_defaults(run=measure)
    p = sub.add_parser("plot", help="figure and band table from the products of one or several runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--min-count", type=float, default=200.0, help="effective particles a window needs to be drawn")
    p.set_defaults(run=plot)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
