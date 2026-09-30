#!/usr/bin/env python3
"""
energy_exchange.py — field-particle energy transfer J_s·E per species
=====================================================================
The instabilities studied here convert free energy of the anisotropic
species into electromagnetic fluctuations and then back into the particles.
``energy_conservation.py`` gives the global budget; this script says *which
species* gives or takes energy and *through which channel*:

    dK_s/dt = ∫ J_s · E dV ,     J_s·E = J_s,par E_par + J_s,perp · E_perp ,

with par/perp relative to the local magnetic field. For a periodic box the
Poynting flux integrates to zero, so dW_EM/dt = -Σ_s ∫ J_s·E dV.

Grid co-location: PSC's moments (``all_1st_cc``) are cell-centred, while E
is edge-centred and B face-centred on the Yee mesh. In the yz plane
(d/dx = 0) E_x sits on the cell corners, E_y on (j+1/2, k) and E_z on
(j, k+1/2); they are averaged to the cell centre before the product. B_x
is already cell-centred in this plane, B_y and B_z are averaged along
their own axis. The species current is J_s = q_s p_s/m_s from PSC's momentum
density p_s = n m <u> (u = gamma v; relativistic corrections are of order
v_th^2/c^2 < 1 % here), or the deposited ``j*_s`` when present.

Consistency check. With the global energies of DiagEnergies (diag.asc),
W_s(t) = ∫_0^t <J_s·E> dt' must match ΔK_s(t)/V. The effective volume in
DiagEnergies' normalisation is taken from the initial magnetic energy,
V = 2 E_B(0)/B0^2 (the fluctuation energy at t = 0 is PIC noise, ~1e-6 of
it), so no cell-volume or weight convention has to be assumed. A mismatch
that grows with time points at the snapshot cadence (J·E of an oscillating
mode is aliased when the output interval is not << 1/omega) or at
numerical heating that does not go through J·E of the resolved fields.

Outputs (``--outdir``, normally ``09_physical_diagnostics/``):
  energy_exchange_table.csv, energy_exchange_summary.json,
  energy_exchange_rate.png, energy_exchange_cumulative.png
"""

from __future__ import annotations

import argparse
import csv
from analysis_contract import strict_dumps, align_time, atomic_json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from data_reader import PICDataReader
from psc_units import B0, DT_CODE, M_ELEC, M_ION, OMEGA_CI, PROFILE_LABEL, step_to_omegaci

ps.apply()

SPECIES = (("i", "ion", +1.0, M_ION), ("e", "electron", -1.0, M_ELEC))
#: Closure of W_s = int <J_s.E> dt against Delta K_s of DiagEnergies, as a
#: fraction of the larger of the two: the thresholds of energy_audit.py.
CLOSURE_PASS, CLOSURE_FAIL = 0.10, 0.50


def aliased(cadence_code: float, mass: float) -> bool:
    """True when the snapshot interval cannot follow the plasma oscillation of a species.

    <J_s.E> of the resolved instability is a slow product, but the field and
    the current also carry the species' plasma oscillation (omega_ps = 1/sqrt(m_s)
    in code units, n = q = 1). Sampled slower than its Nyquist interval
    pi / omega_ps, that fast part aliases into the snapshot mean and the
    integral is not the work: v6b kappa = 5, electrons (cadence * omega_pe = 165)
    gave int <J_e.E> dt = -0.72 against Delta K_e = +0.036 from DiagEnergies.
    """
    return cadence_code * (1.0 / np.sqrt(mass)) > np.pi


def to_cell_centres_yz(ex, ey, ez, bx, by, bz) -> tuple:
    """Average Yee-staggered E (edge) and B (face) to cell centres, (Nz, Ny) maps."""
    r = lambda a, ax: np.roll(a, -1, axis=ax)
    ex_c = 0.25 * (ex + r(ex, 0) + r(ex, 1) + r(r(ex, 0), 1))
    ey_c = 0.5 * (ey + r(ey, 0))          # E_y at (j+1/2, k): average along z
    ez_c = 0.5 * (ez + r(ez, 1))          # E_z at (j, k+1/2): average along y
    by_c = 0.5 * (by + r(by, 1))          # B_y at (j, k+1/2): average along y
    bz_c = 0.5 * (bz + r(bz, 0))          # B_z at (j+1/2, k): average along z
    return ex_c, ey_c, ez_c, bx, by_c, bz_c


def load_fields(path: str) -> dict:
    names = [f"{c}/p0/3d" for c in ("ex_ec", "ey_ec", "ez_ec", "hx_fc", "hy_fc", "hz_fc")]
    data = PICDataReader.read_multiple_fields_3d(path, "jeh", names)
    arr = [PICDataReader.flatten_2d_slice(data[n]).astype(float) for n in names]
    ex, ey, ez, bx, by, bz = to_cell_centres_yz(*arr)
    return {"E": np.stack([ex, ey, ez]), "B": np.stack([bx, by, bz])}


def load_current(path: str, suffix: str, charge: float, mass: float) -> np.ndarray:
    try:
        data = PICDataReader.read_multiple_fields_3d(
            path, "all_1st", [f"j{c}_{suffix}/p0/3d" for c in "xyz"])
        return np.stack([PICDataReader.flatten_2d_slice(data[f"j{c}_{suffix}/p0/3d"]).astype(float)
                         for c in "xyz"])
    except KeyError:
        data = PICDataReader.read_multiple_fields_3d(
            path, "all_1st", [f"p{c}_{suffix}/p0/3d" for c in "xyz"])
        return charge / mass * np.stack(
            [PICDataReader.flatten_2d_slice(data[f"p{c}_{suffix}/p0/3d"]).astype(float) for c in "xyz"])


def exchange_rates(E: np.ndarray, B: np.ndarray, J: np.ndarray) -> dict:
    """Domain means of J·E and its field-aligned split (code units)."""
    bmag = np.sqrt(np.sum(B * B, axis=0))
    b = B / np.where(bmag > 0, bmag, np.nan)
    jpar = np.sum(J * b, axis=0)
    epar = np.sum(E * b, axis=0)
    total = np.sum(J * E, axis=0)
    par = jpar * epar
    return {"JE": float(np.nanmean(total)), "JE_par": float(np.nanmean(par)),
            "JE_perp": float(np.nanmean(total - par))}


def cumulative(t_code: np.ndarray, rate: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rate)
    if rate.size > 1:
        out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t_code))
    return out


def diag_energy_per_volume(paths: list[Path]) -> dict | None:
    """ΔK_s/V and ΔW_EM/V from DiagEnergies, with V = 2 E_B(0)/B0^2."""
    if not paths:
        return None
    from energy_conservation import read_energy_segments
    try:
        rows, _ = read_energy_segments(paths)
    except ValueError as exc:
        print(f"[WARN] DiagEnergies unavailable: {exc}")
        return None
    t = np.array([r["time_code"] for r in rows])
    volume = 2.0 * rows[0]["E_B"] / B0 ** 2
    get = lambda k: np.array([r[k] for r in rows]) / volume
    em = get("E_E") + get("E_B")
    return {"time_code": t, "dK_i": get("E_i") - get("E_i")[0],
            "dK_e": get("E_e") - get("E_e")[0], "dW_em": em - em[0],
            "volume": volume, "baseline_is_t0": bool(t[0] == 0.0)}


def main() -> int:
    p = argparse.ArgumentParser(description="Field-particle energy exchange J_s·E per species.")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--fields", default=None)
    p.add_argument("--moments", default=None)
    p.add_argument("--energy", nargs="*", default=None,
                   help="diag*.asc files for the consistency check (default: DATA_DIR/diag*.asc)")
    p.add_argument("--time-tolerance-code", type=float, default=0.0,
                   help="Verified diagnostic time-rounding tolerance; default 0 (no endpoint snapping)")
    p.add_argument("--outdir", default="physical_diagnostics")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    fields = PICDataReader.find_files(args.fields or str(data_dir / "pfd.*_p*.h5"))
    moments = PICDataReader.find_files(args.moments or str(data_dir / "pfd_moments.*_p*.h5"))
    steps = sorted(set(fields) & set(moments))
    if len(steps) < 2:
        print("[ERROR] need at least two paired field/moment snapshots")
        return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for step in steps:
        fld = load_fields(fields[step])
        row = {"step": step, "omega_ci_t": step_to_omegaci(step), "time_code": step * DT_CODE}
        for suffix, _, charge, mass in SPECIES:
            rates = exchange_rates(fld["E"], fld["B"], load_current(moments[step], suffix, charge, mass))
            row.update({f"{k}_{suffix}": v for k, v in rates.items()})
        rows.append(row)

    t_code = np.array([r["time_code"] for r in rows])
    # Cumulative work W(t) = ∫ <J·E> dt, total and per channel.
    for suffix, *_ in SPECIES:
        for rate_key, work_key in (("JE", "W_total"), ("JE_par", "W_par"), ("JE_perp", "W_perp")):
            w = cumulative(t_code, np.array([r[f"{rate_key}_{suffix}"] for r in rows]))
            for r, value in zip(rows, w):
                r[f"{work_key}_{suffix}"] = float(value)

    energy_paths = ([Path(x) for x in args.energy] if args.energy is not None
                    else sorted(data_dir.glob("diag*.asc")))
    diag = diag_energy_per_volume(energy_paths)
    summary = {"snapshots": len(rows), "cadence_steps": int(np.median(np.diff(steps))),
               "diag_available": diag is not None}
    summary.update({"scientific_status": "UNVERIFIED", "reason": "No closure tolerance or temporal-staggering validation supplied",
                    "current_convention": "deposited current when available; otherwise q<p>/m (u approximation)",
                    "time_source": "step * resolved DT_CODE"})
    cadence_code = float(np.median(np.diff(t_code)))
    for suffix, _, _, mass in SPECIES:
        summary[f"cadence_times_omega_p_{suffix}"] = float(cadence_code / np.sqrt(mass))
        summary[f"aliased_{suffix}"] = bool(aliased(cadence_code, mass))
    if diag is not None:
        for suffix in ("i", "e"):
            dk, coverage = align_time(diag["time_code"], diag[f"dK_{suffix}"], t_code, args.time_tolerance_code)
            work = np.array([r[f"W_total_{suffix}"] for r in rows])
            ok = np.isfinite(dk) & np.isfinite(work)
            # Match baselines at the first jointly supported time, including late-start series.
            if ok.any():
                first, last = np.flatnonzero(ok)[[0, -1]]
                dk = dk - dk[first]
                work = work - work[first]
                residual = work - dk
                scale = max(float(np.nanmax(np.abs(dk[ok]))), float(np.max(np.abs(work[ok]))), 1e-300)
                summary.update({f"W_{suffix}_end": float(work[last]),
                    f"dK_{suffix}_end_diag": float(dk[last]),
                    f"relative_mismatch_{suffix}": float(abs(residual[last]) / scale),
                    f"max_relative_closure_residual_{suffix}": float(np.max(np.abs(residual[ok])) / scale),
                    f"closure_baseline_time_{suffix}": float(t_code[first])})
                for r, value, res, common_work in zip(rows, dk, residual, work):
                    r[f"W_common_{suffix}"] = float(common_work)
                    r[f"dK_diag_{suffix}"] = float(value)
                    r[f"closure_residual_{suffix}"] = float(res)
            summary[f"time_alignment_{suffix}"] = coverage
        summary["volume_from_E_B0"] = diag["volume"]
        # Per-species closure against DiagEnergies decides what the J.E
        # integral can be used for.
        verdicts = {}
        for suffix, name, _, _ in SPECIES:
            mismatch = summary.get(f"relative_mismatch_{suffix}")
            if mismatch is None or not np.isfinite(mismatch):
                continue
            verdicts[name] = ("FAIL" if mismatch >= CLOSURE_FAIL else
                              "PASS" if mismatch <= CLOSURE_PASS else "UNVERIFIED")
            summary[f"closure_status_{suffix}"] = verdicts[name]
        if verdicts:
            order = ("FAIL", "UNVERIFIED", "PASS")
            summary["scientific_status"] = next(v for v in order if v in verdicts.values())
            summary["reason"] = "; ".join(
                f"{name}: int<J.E>dt vs Delta K mismatch {summary[f'relative_mismatch_{name[0]}']:.0%} ({v})"
                + (" -- snapshot cadence aliases the plasma oscillation"
                   if summary.get(f"aliased_{name[0]}") and v != "PASS" else "")
                for name, v in verdicts.items())
    else:
        summary["reason"] = ("Global energy diagnostic missing; "
                             + ", ".join(f"{name} J.E aliased (cadence*omega_p = "
                                         f"{summary[f'cadence_times_omega_p_{sfx}']:.0f})"
                                         for sfx, name, _, _ in SPECIES if summary[f"aliased_{sfx}"])
                             if any(summary[f"aliased_{sfx}"] for sfx, *_ in SPECIES)
                             else "Global energy diagnostic missing or invalid")
    write_csv(outdir / "energy_exchange_table.csv", rows)
    atomic_json(outdir / "energy_exchange_summary.json", summary)
    plot(rows, diag is not None, outdir, summary)
    print(strict_dumps(summary, indent=2))
    return 0


def plot(rows: list[dict], with_diag: bool, outdir: Path, summary: dict | None = None):
    t = np.array([r["omega_ci_t"] for r in rows])
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    colors = {"i": ps.c("#ff7b72"), "e": ps.c("#58a6ff")}

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True, layout="constrained")
    for ax, suffix, name in zip(axes, ("i", "e"), ("(a) ions", "(b) electrons")):
        ax.plot(t, col(f"JE_{suffix}") / (OMEGA_CI * B0 ** 2), "-", color=colors[suffix],
                label=rf"$\langle J_{suffix}\cdot E\rangle$")
        ax.plot(t, col(f"JE_par_{suffix}") / (OMEGA_CI * B0 ** 2), "--", color=ps.c("#56d364"),
                label=rf"$\langle J_{{\parallel {suffix}}}E_\parallel\rangle$")
        ax.plot(t, col(f"JE_perp_{suffix}") / (OMEGA_CI * B0 ** 2), ":", color=ps.c("#d2a8ff"),
                label=rf"$\langle J_{{\perp {suffix}}}\cdot E_\perp\rangle$")
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
        ps.style_axes(ax, name)
        ps.legend(ax, fontsize=10)
    axes[-1].set_xlabel(r"$t\,\Omega_{ci}$")
    fig.supylabel(r"$\langle J_s\cdot E\rangle/(\Omega_{ci}B_0^2/\mu_0)$")
    fig.suptitle(f"Field-particle energy transfer — {PROFILE_LABEL}", fontsize=13, fontweight="bold")
    ps.save(fig, outdir / "energy_exchange_rate.png")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    summary = summary or {}
    for suffix, name in (("i", "ions"), ("e", "electrons")):
        # A species whose snapshot J.E aliases its plasma oscillation, and is
        # not confirmed by DiagEnergies, is drawn thin and dotted and says so.
        unusable = (summary.get(f"aliased_{suffix}") and
                    summary.get(f"closure_status_{suffix}", "UNVERIFIED") != "PASS")
        note = " — aliased, not the work" if unusable else ""
        ax.plot(t, col(f"W_common_{suffix}" if f"W_common_{suffix}" in rows[0] else f"W_total_{suffix}") / B0 ** 2,
                ":" if unusable else "-", lw=1.0 if unusable else 1.8, color=colors[suffix],
                scaley=not unusable,   # an aliased curve must not set the scale
                label=rf"$\int\langle J_{suffix}\cdot E\rangle dt$ ({name}){note}")
        if with_diag:
            ax.plot(t, col(f"dK_diag_{suffix}") / B0 ** 2, "o", ms=3, mfc="none", color=colors[suffix],
                    label=rf"$\Delta K_{suffix}/V$ (DiagEnergies)")
    ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    ax.set_xlabel(r"$t\,\Omega_{ci}$")
    ax.set_ylabel(r"energy density / $(B_0^2/\mu_0)$")
    ax.set_title("Work done by E on each species vs its kinetic-energy change", fontsize=12)
    ps.style_axes(ax)
    ps.legend(ax, fontsize=10)
    ps.save(fig, outdir / "energy_exchange_cumulative.png")


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
