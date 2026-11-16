#!/usr/bin/env python3
"""
article1_figures.py — figures of the ion-anisotropy article, with the numerical controls
=======================================================================================
Ten figures for the article on the ion-cyclotron instability driven by
bi-Maxwellian and bi-kappa ions at equal temperature, from analysis products
only (no snapshots). Every figure that shows a quantity the numerics also
change carries the stable isotropic controls next to the runs, so the reader
sees the size of the artefact beside the effect:

  fig1_conventions        reduced f(v_par) at equal temperature (Kappa-A, these
                          runs) and at equal core (Kappa-B), with the beta each
                          implies and the cyclotron-resonant speed of the mode.
  fig2_mode_growth        (a) amplitude of the box-fundamental mode against the
                          noise level of a single Fourier mode in the stable
                          controls; (b) its local growth rate against linear
                          theory evaluated with the measured state.
  fig3_growth_rate        measured gamma against 1/kappa with linear theory in
                          both conventions.
  fig4_wave_energy        <|dB|^2>/B0^2 minus the scaled noise of the controls
                          (noise_floor.py), log and linear, with the time the
                          mode stops growing.
  fig5_trajectories       (a) global (beta_i_par, A_i) trajectories with the
                          ion-cyclotron marginal curve at the box mode;
                          (b) A_i(t) of runs and controls.
  fig6_band_anisotropy    A(v_par) by parallel-velocity band (resonant_anisotropy.py)
                          relative to t = 0, runs and controls on one axis.
  fig7_numerical          (a) electron temperature, raw and after the scaled
                          control; (b) noise fraction of the fluctuation energy;
                          (c) global budget after the scaled control.
  fig8_kappa_index        1/kappa(t) of the runs and of the stable controls, one
                          estimator for both (the stored f(v_par, v_perp)), with
                          the particle estimator of the runs.
  fig9_kappa_relaxation   ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c int W dt with
                          W the wave energy without the noise; (a) the fit and
                          the background of the control, (b) collapse on the
                          fluence.
  fig10_tail_shape        f(u)/Gaussian, u = dv_par/sigma(t), at three times,
                          and the probability beyond 3 sigma against time for
                          runs and controls.

plus ``article1_numbers.csv`` with the values the text quotes.

Usage:
    python article1_figures.py MAXW_RUN K5_RUN K3_RUN \\
        --control MAXW_CONTROL --control K5_CONTROL --outdir OUT

Runs in the order bi-Maxwellian, kappa 5, kappa 3 (colours are fixed in that
order). Controls are the isotropic runs of the same numerics; a run without a
control of its distribution is drawn without the correction, never with
another distribution's.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np

import plot_style as ps

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import energy_audit  # noqa: E402
import kappa_dynamics as kd  # noqa: E402
import kappa_eff as ke  # noqa: E402
import noise_floor  # noqa: E402
import paper_figures as pf  # noqa: E402
import resonant_anisotropy as ra  # noqa: E402
import trajectory_linear_theory as tlt  # noqa: E402
from linear_theory import ParallelDispersion  # noqa: E402

SERIES = pf.SERIES
CONTROL_STYLE = dict(ls="--", lw=1.4, alpha=0.8)
BAND_NAMES = (r"$|v_\parallel|<1\,\sigma_0$", r"$1$–$2\,\sigma_0$", r"$2$–$3\,\sigma_0$", r"$>3\,\sigma_0$")


# ── Loading ─────────────────────────────────────────────────────────────────

def _table(path: Path) -> dict:
    rows = energy_audit._read_csv(path)
    return {k: energy_audit._col(rows, k) for k in rows[0]} if rows else {}


def kappa_of(name: str) -> float | None:
    for tag, kappa in (("bikappa3", 3.0), ("bikappa5", 5.0)):
        if tag in name:
            return kappa
    return None


def label_of(kappa: float | None) -> str:
    return "bi-Maxwellian" if kappa is None else rf"bi-$\kappa$, $\kappa={kappa:g}$"


def growth_error(t, amp, t0, t1) -> float:
    """Slope error of ln(amp) over [t0, t1] with the effective sample size of correlated residuals."""
    q = (t >= t0) & (t <= t1) & (amp > 0)
    x, y = t[q], np.log(amp[q])
    if x.size < 5:
        return float("nan")
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    n_eff = max(x.size * (1.0 - r1) / (1.0 + r1), 3.0)
    return float(np.sqrt(np.sum(resid ** 2) / (n_eff - 2.0) / np.sum((x - x.mean()) ** 2)))


def same_core_gamma(profile: dict, kappa: float, k: float) -> float:
    """Ion-cyclotron rate at equal core width: T and beta grow by kappa/(kappa - 3/2)."""
    f = kappa / (kappa - 1.5)
    disp = ParallelDispersion(profile["beta_i_par"] * f, profile["Ti_perp_over_Ti_par"],
                              profile["beta_e_par"] * f, profile["Te_perp_over_Te_par"],
                              profile["mass_ratio"], 1.0 / pf.psc_units.VA_OVER_C, kappa=kappa)
    return float(disp.scan([k], "plus")[-1]["gamma_over_Omegai"])


def load(runs: list[Path], controls: list[Path]) -> dict:
    data = {"runs": [], "controls": []}
    for root in runs:
        run = pf.load_run(root)
        run["theory"] = pf.theory(run, run["kappa"])
        run["resonance_v"] = (run["theory"]["omega_r"] - 1.0) / run["k_mode"]
        run["gamma_err_corr"] = growth_error(run["kt_t"], run["kt_amp"], *run["t_lin"])
        run["gamma_err_used"] = float(np.nanmax([run["gamma_err"], run["gamma_err_corr"]]))
        run["fluct"] = _table(root / "09_physical_diagnostics" / "field_fluctuation_table.csv")
        run["bands"] = band_series_particles(root)
        data["runs"].append(run)
    for root in controls:
        kappa = kappa_of(root.name)
        data["controls"].append({
            "name": root.name, "root": root, "kappa": kappa, "label": label_of(kappa) + " control",
            "aniso": _table(root / "09_physical_diagnostics" / "anisotropy_table.csv"),
            "fluct": _table(root / "09_physical_diagnostics" / "field_fluctuation_table.csv"),
            "modes": _table(root / "09_physical_diagnostics" / "mode_amplitude_timeseries.csv"),
            "bands": band_series_histogram(root)})
    audits = [energy_audit.audit_run(r) for r in runs + controls]
    explicit = {}
    for run_root in runs:
        same = [c for c in controls if kappa_of(c.name) == kappa_of(run_root.name)]
        if same:
            explicit[run_root.name] = same[0].name
    energy_audit.pair_controls(audits, explicit)
    for audit in audits:
        name = (audit.get("baseline") or {}).get("control")
        if name and name not in explicit.values():
            audit.pop("baseline")          # an automatic pairing across distributions is not used
    data["audits"] = {a["run"]: a for a in audits}
    data["floors"] = {f["run"]: f for f in noise_floor.audit_paired(audits)}
    return data


def band_series_particles(root: Path) -> dict:
    """{t: [(A, err) per band]} from resonant_anisotropy.csv (particles, local-field frame)."""
    out = {}
    for r in energy_audit._read_csv(root / "03_particles" / ra.PRODUCT):
        if r["kind"] == "band":
            out.setdefault(float(r["omega_ci_t"]), []).append((float(r["A"]), float(r["A_err"])))
    return {t: np.array(v) for t, v in sorted(out.items())}


def band_series_histogram(root: Path) -> dict:
    """The band estimator of resonant_anisotropy.py on the stored f(v_par, v_perp) histograms.

    Bin-averaged moments with the particle counts as weights; it reproduces the
    particle estimator to about 0.02 at t = 0 and is used only for the stable
    controls, which have no particle product.
    """
    out = {}
    for path in (root / "09_physical_diagnostics").glob("vdf_2d_ion_step_*.npz"):
        with np.load(path) as z:
            pe, qe, w, s0 = z["par_edges"], z["perp_edges"], z["counts"], float(z["sigma0"][0])
            t = float(z["omega_ci_t"])
        u = 0.5 * (pe[:-1] + pe[1:]) / s0
        u2 = (pe[:-1] ** 2 + pe[:-1] * pe[1:] + pe[1:] ** 2) / 3.0 / s0 ** 2
        e_perp = 0.25 * (qe[:-1] ** 2 + qe[1:] ** 2) / s0 ** 2
        x = np.abs(u)
        values = []
        for _, _, b, db in ra.band_windows():
            bx = b(x)
            values.append(float(np.sum(w * np.outer(bx + x * db(x), e_perp))
                                / np.sum(w * np.outer(u2 * bx, np.ones_like(e_perp)))))
        out[t] = np.array(values)
    return dict(sorted(out.items()))


# ── Figures ─────────────────────────────────────────────────────────────────

def legend_below(ax, handles=None, ncol=2, fontsize=9.5, offset=-0.25):
    kwargs = dict(loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol, frameon=False, fontsize=fontsize)
    if handles is not None:
        kwargs["handles"] = handles
    ps.legend(ax, **kwargs)


def fig1_conventions(data, out):
    runs = data["runs"]
    p = runs[0]["profile"]
    vth = math.sqrt(p["beta_i_par"] / 2.0)
    v = np.linspace(0.0, 6.0, 500)
    v_res = float(np.mean([abs(r["resonance_v"]) for r in runs]))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True, constrained_layout=True)
    for i, run in enumerate(runs):
        kappa = run["kappa"]
        same_t = pf.reduced_pdf(v / vth, kappa) / vth
        stretch = 1.0 if kappa is None else math.sqrt(kappa / (kappa - 1.5))
        same_core = pf.reduced_pdf(v / (vth * stretch), kappa) / (vth * stretch)
        beta = p["beta_i_par"] * stretch ** 2
        axes[0].plot(v, same_t, color=ps.c(SERIES[i]), lw=1.8,
                     label=rf"{run['label']}, $\beta_{{i\parallel}}={p['beta_i_par']:g}$")
        axes[1].plot(v, same_core, color=ps.c(SERIES[i]), lw=1.8,
                     label=rf"{run['label']}, $\beta_{{i\parallel}}={beta:.3g}$")
    for ax, title in zip(axes, ("(a) equal temperature (Kappa-A, these runs)",
                                "(b) equal core width (Kappa-B)")):
        ax.axvline(v_res, color=ps.MUTED_CLR, ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 0.5)
        ax.set_xlabel(r"$|v_\parallel|/v_A$")
        ps.plain_log_axis(ax, "y")
        ps.style_axes(ax, title)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.2,
                              label=rf"cyclotron resonance $|v_{{\rm res}}|\simeq{v_res:.2f}\,v_A$"))
        legend_below(ax, handles, ncol=1)
    axes[0].set_ylabel(r"$f(v_\parallel)\,v_A$")
    ps.save(fig, out / "fig1_conventions.png")


def fig2_mode_growth(data, out, tracks):
    runs, controls = data["runs"], data["controls"]
    levels = []
    for c in controls:
        m = c["modes"]
        q = m["omega_ci_t"] > 3.0
        levels.append(np.concatenate([m[k][q] for k in m if k.startswith("amp_over_B0")]))
    # A mode of the 20 d_i box averages over 4x the cells of the 10 d_i control: amplitude / 2.
    pooled = np.concatenate(levels) / 2.0 if levels else np.array([])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        keep = ps.measured_fluctuation(run["kt_t"], run["t_settle"]) & (run["kt_amp"] > 0)
        axes[0].plot(run["kt_t"][keep], run["kt_amp"][keep], color=col, lw=1.5, label=run["label"])
        t0, t1 = run["t_lin"]
        axes[0].axvspan(t0, t1, color=col, alpha=0.06, lw=0)
        tr = tracks[run["name"]]
        ok = np.isfinite(tr["gamma_measured"])
        axes[1].errorbar(tr["t"][ok], tr["gamma_measured"][ok], yerr=tr["gamma_measured_err"][ok], fmt="o",
                         ms=3.5, color=col, capsize=0, lw=1.0, label=f"{run['label']}, PIC")
        axes[1].plot(tr["t"], tr["gamma_linear"], "-", color=col, lw=1.6)
    if pooled.size:
        lo, hi = np.quantile(pooled, [0.5, 0.999])
        axes[0].axhspan(lo, hi, color=ps.MUTED_CLR, alpha=0.18, lw=0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$|\delta\hat{\mathbf{B}}(k_\parallel d_i=0.31)|/B_0$")
    ps.plain_log_axis(axes[0], "y")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color=ps.MUTED_CLR, alpha=0.18,
                              label="single-mode noise of the stable controls (median to 99.9 %)"),
                plt.Rectangle((0, 0), 1, 1, color=ps.MUTED_CLR, alpha=0.06, label="fitted linear phase")]
    settle = max(r["t_settle"] for r in runs)
    handles.append(Line2D([], [], ls="none", label=rf"quiet-start build-up ($t\Omega_{{ci}}<{settle:.1f}$) not shown"))
    legend_below(axes[0], handles, ncol=1)
    axes[1].axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    axes[1].set_ylabel(r"$\gamma/\Omega_{ci}$")
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=ps.MUTED_CLR, lw=1.6,
                          label="linear theory with the measured ions and electrons"))
    legend_below(axes[1], handles, ncol=1)
    for ax, title in zip(axes, ("(a) amplitude of the box-fundamental mode",
                                "(b) local growth rate (15 $\\Omega_{ci}^{-1}$ windows)")):
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ps.style_axes(ax, title)
    ps.save(fig, out / "fig2_mode_growth.png")


def fig3_growth_rate(data, out):
    runs = data["runs"]
    p, k = runs[0]["profile"], runs[0]["k_mode"]
    inv = np.linspace(0.0, 0.36, 19)
    equal_t = [pf.theory(runs[0], None if x == 0 else 1.0 / x)["gamma"] for x in inv]
    equal_core = [equal_t[0]] + [same_core_gamma(p, 1.0 / x, k) for x in inv[1:]]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.plot(inv, equal_t, "-", color=ps.MUTED_CLR, lw=1.8, label="linear theory, equal temperature")
    ax.plot(inv, equal_core, "--", color=ps.MUTED_CLR, lw=1.6, label="linear theory, equal core width")
    for i, run in enumerate(runs):
        x = 0.0 if run["kappa"] is None else 1.0 / run["kappa"]
        ax.errorbar(x, run["gamma"], yerr=run["gamma_err_used"], fmt="o", ms=7, capsize=4,
                    color=ps.c(SERIES[i]), label=f"PIC, {run['label']}")
    ax.set_xlabel(r"$1/\kappa$  (0 = Maxwellian)")
    ax.set_ylabel(r"$\gamma/\Omega_{ci}$ at $k_\parallel d_i=%.3f$" % k)
    ax.set_xlim(-0.02, 0.38)
    ps.style_axes(ax, "Ion-cyclotron growth rate")
    legend_below(ax, ncol=1, offset=-0.17)
    ps.save(fig, out / "fig3_growth_rate.png")
    return equal_t, equal_core, inv


def noise_template(controls):
    """<|dB|^2>/B0^2 of the bi-Maxwellian control (the longest), log-linear beyond its end."""
    c = next((c for c in controls if c["kappa"] is None), controls[0])
    t, n = c["fluct"]["omega_ci_t"], c["fluct"]["delta_B_vec_rms_over_B0"] ** 2
    late = t >= t[-1] - 40.0
    slope, icpt = np.polyfit(t[late], np.log(n[late]), 1)
    end = float(t[-1])
    return (lambda x: np.where(np.asarray(x) <= end, np.interp(x, t, n), np.exp(icpt + slope * np.asarray(x)))), end


def wave_energy(run, noise, t_quiet_end):
    f = run["fluct"]
    t, p = f["omega_ci_t"], f["delta_B_vec_rms_over_B0"] ** 2
    quiet = (t > run["t_settle"]) & (t <= t_quiet_end)
    s = float(np.sum(p[quiet] * noise(t[quiet])) / np.sum(noise(t[quiet]) ** 2))
    model = s * noise(t)
    scatter = float(np.std((p[quiet] - model[quiet]) / model[quiet]))
    return t, p, p - model, model, scatter


def fig4_wave_energy(data, out, sats):
    runs = data["runs"]
    noise, t_end = noise_template(data["controls"])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    rows = {}
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        t, p, ex, model, scatter = wave_energy(run, noise, run["t_lin"][0])
        shown = ps.measured_fluctuation(t, run["t_settle"]) & (ex > 3.0 * scatter * model)
        inside, beyond = shown & (t <= t_end), shown & (t > t_end)
        # NaN outside the mask, not dropped samples: an isolated early point
        # would otherwise be joined to the onset by a straight line.
        for ax in axes:
            ax.plot(t, np.where(inside, ex, np.nan), "-", color=col, lw=1.8,
                    label=run["label"] if ax is axes[0] else None)
            ax.plot(t, np.where(beyond, ex, np.nan), ":", color=col, lw=1.8)
        ts = sats[run["name"]]["t_saturation"]
        e_sat = float(np.interp(ts, t, ex))
        axes[1].plot([ts], [e_sat], "o", color=col, ms=6)
        rows[run["name"]] = {"t_saturation": ts, "wave_energy_at_saturation": e_sat,
                             "raw_energy_at_saturation": float(np.interp(ts, t, p))}
    keep = ps.measured_fluctuation(np.linspace(1, t_end, 300), runs[0]["t_settle"])
    tt = np.linspace(1, t_end, 300)[keep]
    axes[0].plot(tt, noise(tt), "--", color=ps.MUTED_CLR, lw=1.4)
    axes[0].set_yscale("log")
    ps.plain_log_axis(axes[0], "y")
    axes[1].set_ylim(bottom=0.0)
    for ax in axes:
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ax.set_ylabel(r"$(\langle|\delta\mathbf{B}|^2\rangle-{\rm noise})/B_0^2$")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, ls="--", lw=1.4, label="noise: stable control"),
                Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.8, label=rf"noise extrapolated ($t\Omega_{{ci}}>{t_end:.0f}$)")]
    legend_below(axes[0], handles, ncol=2)
    legend_below(axes[1], [Line2D([], [], ls="none", marker="o", color=ps.MUTED_CLR,
                                  label="the mode stops growing (local rate crosses zero)")], ncol=1)
    ps.style_axes(axes[0], "(a) wave energy, noise removed")
    ps.style_axes(axes[1], "(b) same, linear scale")
    ps.save(fig, out / "fig4_wave_energy.png")
    return rows


def fig5_trajectories(data, out, tracks):
    runs, controls = data["runs"], data["controls"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    betas = np.concatenate([r["aniso"]["beta_parallel_global"] for r in runs])
    grid = np.linspace(np.nanmin(betas) * 0.97, np.nanmax(betas) * 1.03, 7)
    a_e = float(np.nanmean(runs[0]["aniso"]["anisotropy_e_global"]))
    for beta_e, ls in ((1.0, ":"), (float(runs[0]["aniso"]["beta_e_parallel_global"][-1]), "-")):
        marg = [tlt.marginal_anisotropy({"beta_i": b, "a_i": 2.0, "beta_e": beta_e, "a_e": a_e,
                                         "mass_ratio": runs[0]["profile"]["mass_ratio"],
                                         "c_over_va": 1.0 / pf.psc_units.VA_OVER_C}, None, runs[0]["k_mode"])[0]
                for b in grid]
        axes[0].plot(grid, marg, ls, color=ps.MUTED_CLR, lw=1.4,
                     label=rf"ion-cyclotron marginal at $k_\parallel d_i=0.31$, $\beta_{{e\parallel}}={beta_e:.1f}$")
    for i, run in enumerate(runs):
        a, col = run["aniso"], ps.c(SERIES[i])
        axes[0].plot(a["beta_parallel_global"], a["anisotropy_global"], color=col, lw=1.8, label=run["label"] + " (dots: start, end)")
        axes[0].plot(a["beta_parallel_global"][[0, -1]], a["anisotropy_global"][[0, -1]], "o", color=col, ms=5)
        axes[1].plot(a["omega_ci_t"], a["anisotropy_global"], color=col, lw=1.8, label=run["label"])
        tr = tracks[run["name"]]
        axes[1].plot(tr["t"], tr["a_marginal"], ":", color=col, lw=1.2)
    for c in controls:
        i = 0 if c["kappa"] is None else 1
        axes[1].plot(c["aniso"]["omega_ci_t"], c["aniso"]["A_i"], color=ps.c(SERIES[i]), **CONTROL_STYLE,
                     label=c["label"] + " (isotropic, stable)")
    axes[0].set_xlabel(r"$\beta_{i\parallel}$")
    axes[0].set_ylabel(r"$A_i=T_{i\perp}/T_{i\parallel}$")
    axes[1].set_xlabel(r"$t\Omega_{ci}$")
    axes[1].set_ylabel(r"$A_i$")
    legend_below(axes[0], ncol=1)
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.2, label="marginal anisotropy with the measured state"))
    legend_below(axes[1], handles, ncol=1)
    ps.style_axes(axes[0], "(a) global trajectories")
    ps.style_axes(axes[1], "(b) runs and stable controls")
    ps.save(fig, out / "fig5_trajectories.png")


def fig6_band_anisotropy(data, out):
    runs, controls = data["runs"], data["controls"]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4), sharey=True, constrained_layout=True)
    s0 = math.sqrt(runs[0]["profile"]["beta_i_par"] / 2.0)
    rows = {}
    for i, run in enumerate(runs):
        ts = np.array(list(run["bands"]))
        vals = np.array([run["bands"][t] for t in ts])
        rows[run["name"]] = vals[-1]
        for j, ax in enumerate(axes):
            a0 = vals[0, j, 0]
            ax.errorbar(ts, vals[:, j, 0] / a0, yerr=vals[:, j, 1] / a0, color=ps.c(SERIES[i]), lw=1.6,
                        fmt="-", capsize=0, elinewidth=0.8, label=run["label"])
    for c in controls:
        i = 0 if c["kappa"] is None else 1
        ts = np.array(list(c["bands"]))
        vals = np.array([c["bands"][t] for t in ts])
        for j, ax in enumerate(axes):
            ax.plot(ts, vals[:, j] / vals[0, j], color=ps.c(SERIES[i]), **CONTROL_STYLE,
                    label=c["label"] + " (stable)")
    v_res = float(np.mean([abs(r["resonance_v"]) for r in runs]))
    for j, (ax, name) in enumerate(zip(axes, BAND_NAMES)):
        lo, hi = ra.BAND_EDGES[j - 1] if j else 0.0, ra.BAND_EDGES[j] if j < len(ra.BAND_EDGES) else None
        span = rf"{lo * s0:.1f}–{hi * s0:.1f}$\,v_A$" if hi else rf"$>{lo * s0:.1f}\,v_A$"
        res = r", contains $v_{\rm res}$" if lo * s0 <= v_res < (hi or np.inf) * s0 else ""
        ax.axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ps.style_axes(ax, f"{name} ({span}{res})")
        ax.title.set_fontsize(10.5)
    axes[0].set_ylabel(r"$A(v_\parallel\ {\rm band})/A(t=0)$")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9.5)
    ps.save(fig, out / "fig6_band_anisotropy.png")
    return rows


def fig7_numerical(data, out):
    runs = data["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        audit = data["audits"].get(run["name"], {})
        w = audit.get("window", {}).get("_series")
        if w is not None:
            axes[0].plot(w["omega_ci_t"], w["T_e_over_T_e0"], "-", color=col, lw=1.3, alpha=0.6)
        base = audit.get("baseline") or {}
        bw = (base.get("window") or {}).get("_series")
        if bw is not None:
            axes[0].plot(bw["omega_ci_t"], bw["control"], color=col, **CONTROL_STYLE)
        sw = ((base.get("scaled") or {}).get("window") or {}).get("_series")
        if sw is not None:
            axes[0].plot(sw["omega_ci_t"], sw["corrected"], "-", color=col, lw=2.4, label=run["label"])
        floor = data["floors"].get(run["name"])
        if floor:
            s = floor["components"]["total"]["_series"]
            axes[1].plot(s["omega_ci_t"], s["noise_fraction"], "-", color=col, lw=2.0, label=run["label"])
            if "parallel" in floor["components"]:
                sp = floor["components"]["parallel"]["_series"]
                axes[1].plot(sp["omega_ci_t"], sp["noise_fraction"], ":", color=col, lw=1.6)
        sg = ((base.get("scaled") or {}).get("global") or {}).get("_series")
        if sg is not None:
            gs = audit["global"]["_series"]
            scale = gs["E_total"][0] / gs["E_B"][0]          # residuals in units of E_B(0) = B0^2 V / 2
            for key, style, name in (("E_i", "-", "ions"), ("E_B", "--", "magnetic field"),
                                     ("E_e", "-.", "electrons"), ("E_total", ":", "sum")):
                axes[2].plot(sg["omega_ci_t"], sg[key] * scale, style, color=col, lw=1.8 if key != "E_total" else 2.2,
                             label=f"{name}")
            raw = (gs["E_total"] - gs["E_total"][0]) / gs["E_B"][0]
            axes[2].text(0.02, 0.04, rf"raw total change at $t\Omega_{{ci}}={sg['omega_ci_t'][-1]:.0f}$: "
                         rf"{np.interp(sg['omega_ci_t'][-1], gs['omega_ci_t'], raw):+.1f} $E_B(0)$",
                         transform=axes[2].transAxes, fontsize=9.5, color=ps.MUTED_CLR, va="bottom")
            axes[2].set_title(f"(c) global budget, {run['label']}, after the scaled control", fontsize=11)
    axes[0].axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
    axes[0].set_ylabel(r"$T_e/T_{e0}$ (prt window)")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=1.3, alpha=0.6, label="run, raw"),
                Line2D([], [], color=ps.MUTED_CLR, **CONTROL_STYLE, label="stable control"),
                Line2D([], [], color=ps.MUTED_CLR, lw=2.4, label="run minus scaled control")]
    legend_below(axes[0], handles, ncol=2)
    axes[1].axhline(noise_floor.NOISE_DOMINATED, color=ps.MUTED_CLR, ls="--", lw=1.0)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_ylabel("Noise fraction of the\nmagnetic fluctuation energy")
    handles, _ = axes[1].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=2.0, label=r"$\langle|\delta\mathbf{B}|^2\rangle$"),
                Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.6, label=r"$\langle\delta B_\parallel^2\rangle$ (compressive)")]
    legend_below(axes[1], handles, ncol=2)
    axes[2].axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    axes[2].set_ylabel(r"$\Delta E\,/\,E_B(0)$")
    legend_below(axes[2], ncol=4)
    for ax in axes:
        ax.set_xlabel(r"$t\Omega_{ci}$")
    ps.style_axes(axes[0], "(a) electron temperature")
    ps.style_axes(axes[1], "(b) how much of it is noise")
    axes[2].title.set_fontsize(11)
    ps.style_axes(axes[2])
    ps.save(fig, out / "fig7_numerical.png")


# ── Evolution of kappa ──────────────────────────────────────────────────────

KAPPA_S_MAX = 5.0


def grid_kappa(root: Path) -> np.ndarray:
    """(t, signed 1/kappa) from the stored f(v_par, v_perp): the same estimator for runs and controls."""
    out = []
    for path in (root / "09_physical_diagnostics").glob("vdf_2d_ion_step_*.npz"):
        with np.load(path) as z:
            vp = 0.5 * (z["par_edges"][:-1] + z["par_edges"][1:])
            vq = 0.5 * (z["perp_edges"][:-1] + z["perp_edges"][1:])
            par, perp = np.meshgrid(vp, vq, indexing="ij")
            r = ke.kappa_eff_from_grid(par, perp, z["counts"], s_max=KAPPA_S_MAX)
            out.append((float(z["omega_ci_t"]), ke.signed_inverse_kappa(r["K"], KAPPA_S_MAX)))
    return np.array(sorted(out))


def fit_line(t, y):
    coef, cov = np.polyfit(t, y, 1, cov=True)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def fig8_kappa_index(data, out):
    runs, controls = data["runs"], data["controls"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    rows = {}
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        kr = kd.load_run(run["root"])
        s = kr["pop"]["all"]
        axes[0].plot(s["t"], s["inv"], "-", color=col, lw=1.8, label=f"{run['label']}, particles (local frame)")
        g = grid_kappa(run["root"])
        # Relative to t = 0 only with a tail to lose: 1/kappa of a Maxwellian is ~0.
        if run["kappa"] is not None:
            axes[1].plot(g[:, 0], g[:, 1] / g[0, 1], "o-", color=col, ms=3.5, lw=1.4, label=run["label"])
        rows[run["name"]] = {"inv_kappa_particles": (s["inv"][0], s["inv"][-1])}
    for c in controls:
        i = 0 if c["kappa"] is None else 1
        g = grid_kappa(c["root"])
        c["grid_kappa"] = g
        if g[0, 1] > 0.05:
            nu0, err = fit_line(g[:, 0], -np.log(g[:, 1] / g[0, 1]))
            c["nu0"], c["nu0_err"] = nu0, err
            axes[1].plot(g[:, 0], g[:, 1] / g[0, 1], "s", color=ps.c(SERIES[i]), ms=4, mfc="none",
                         ls="--", lw=1.4, label=c["label"] + rf" (stable), $\nu_0={nu0 * 1e3:.2f}\times10^{{-3}}$")
            tt = np.linspace(0, 160, 50)
            axes[1].plot(tt, np.exp(-nu0 * tt), ":", color=ps.c(SERIES[i]), lw=1.2)
        axes[0].plot(g[:, 0], g[:, 1], "s", color=ps.c(SERIES[i]), ms=4, mfc="none", ls="--", lw=1.2,
                     label=c["label"] + " (stable, histogram estimator)")
    axes[0].axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    axes[0].set_ylabel(r"$1/\kappa$  (0 = Maxwellian)")
    axes[1].set_ylabel(r"$(1/\kappa)/(1/\kappa)_{t=0}$")
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.2, label=r"$e^{-\nu_0 t}$ of the control, extended"))
    legend_below(axes[0], ncol=1)
    legend_below(axes[1], handles, ncol=1)
    for ax in axes:
        ax.set_xlabel(r"$t\Omega_{ci}$")
    ps.style_axes(axes[0], "(a) kappa index of the ions")
    ps.style_axes(axes[1], "(b) relative to t = 0, histogram estimator")
    ps.save(fig, out / "fig8_kappa_index.png")
    return rows


def clean_relaxation(run, noise):
    """kappa_dynamics.relaxation with W = <|dB|^2> minus the scaled noise of the controls."""
    kr = kd.load_run(run["root"])
    f = kr["field"]
    t, w = f["t"], f["W"]
    quiet = (t > run["t_settle"]) & (t <= run["t_lin"][0])
    s = float(np.sum(w[quiet] * noise(t[quiet])) / np.sum(noise(t[quiet]) ** 2))
    wx = w - s * noise(t)
    f["W"] = wx
    f["F"] = np.concatenate([[0.0], np.cumsum(0.5 * (wx[1:] + wx[:-1]) * np.diff(t))])
    return kd.relaxation(kr)


def fig9_kappa_relaxation(data, out):
    runs = [r for r in data["runs"] if r["kappa"] is not None]
    noise, _ = noise_template(data["controls"])
    ctrl = next((c for c in data["controls"] if c.get("nu0") is not None), None)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    rows = {}
    for run in runs:
        col = ps.c(SERIES[data["runs"].index(run)])
        fit = clean_relaxation(run, noise)
        tt = fit["t"] - fit["t0"]
        axes[0].plot(fit["t"], fit["y"], "o", color=col, ms=2.5, alpha=0.6)
        axes[0].plot(fit["t"], -fit["nu0"] * tt - fit["c"] * fit["F"], "-", color=col, lw=1.8,
                     label=rf"{run['label']}: $\nu_0={fit['nu0'] * 1e3:.2f}\times10^{{-3}}$, $c={fit['c']:.3f}$")
        axes[0].plot(fit["t"], -fit["nu0"] * tt, ":", color=col, lw=1.3)
        axes[1].plot(fit["F"], fit["y"] + fit["nu0"] * tt, "o", color=col, ms=2.5, alpha=0.6, label=run["label"])
        ff = np.linspace(0, fit["F"].max(), 50)
        axes[1].plot(ff, -fit["c"] * ff, "-", color=col, lw=1.6)
        rows[run["name"]] = {k: fit[k] for k in ("nu0", "nu0_err", "c", "c_err", "chi2_dof")}
        rows[run["name"]]["background_share_end"] = float((fit["nu0"] * tt[-1]) / (fit["nu0"] * tt[-1] + fit["c"] * fit["F"][-1]))
    if ctrl is not None:
        tt = np.linspace(0, 160, 50)
        axes[0].plot(tt, -ctrl["nu0"] * tt, "--", color=ps.MUTED_CLR, lw=1.4,
                     label=rf"stable control: $-\nu_0 t$, $\nu_0={ctrl['nu0'] * 1e3:.2f}\times10^{{-3}}$")
    axes[0].set_ylabel(r"$\ln[(1/\kappa)/(1/\kappa_0)]$")
    axes[0].set_xlabel(r"$t\Omega_{ci}$")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=1.8, label=r"fit $-\nu_0 t-c\int W\,dt$"),
                Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.3, label=r"$-\nu_0 t$ part of the fit")]
    legend_below(axes[0], handles, ncol=1)
    axes[1].set_xlabel(r"$\int_0^t W\,dt'$, $W=(\langle|\delta\mathbf{B}|^2\rangle-{\rm noise})/B_0^2$", fontsize=11)
    axes[1].set_ylabel(r"$\ln[(1/\kappa)/(1/\kappa_0)]+\nu_0 t$")
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=ps.MUTED_CLR, lw=1.6, label=r"$-c\int W\,dt$"))
    # Lower: the two-line fluence label of this panel is taller.
    legend_below(axes[1], handles, ncol=1, offset=-0.3)
    ps.style_axes(axes[0], "(a) two-term relaxation")
    ps.style_axes(axes[1], "(b) wave part against the wave fluence")
    ps.save(fig, out / "fig9_kappa_relaxation.png")
    return rows


def tail_fraction_histogram(root: Path) -> np.ndarray:
    """(t, P(|u|>3)/Gaussian) from the stored f(v_par, v_perp), u = (v_par - <v_par>)/sigma_par(t)."""
    from scipy.special import erfc
    out = []
    for path in (root / "09_physical_diagnostics").glob("vdf_2d_ion_step_*.npz"):
        with np.load(path) as z:
            pe, w, t = z["par_edges"], z["counts"].sum(axis=1), float(z["omega_ci_t"])
        vc = 0.5 * (pe[:-1] + pe[1:])
        mu = np.sum(w * vc) / w.sum()
        sig = np.sqrt(np.sum(w * (vc - mu) ** 2) / w.sum())
        out.append((t, float(w[np.abs(vc - mu) > 3 * sig].sum() / w.sum() / erfc(3 / np.sqrt(2)))))
    return np.array(sorted(out))


def fig10_tail_shape(data, out):
    from scipy.special import erfc
    runs, controls = data["runs"], data["controls"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    styles = (("-", 1.0), ("--", 0.85), (":", 0.85))
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        shape = kd.load_shape(run["root"])
        edges = shape["edges"]
        centres = 0.5 * (edges[1:] + edges[:-1])
        gauss = kd._gauss_in_bins(edges)
        t = shape["t"]
        picks = (0, int(np.argmin(np.abs(t - run["t_lin"][1]))), len(t) - 1)
        if run["kappa"] is not None:
            for j, (ls, alpha) in zip(picks, styles):
                d = shape["par_density"][j]
                half = centres >= 0
                ratio = 0.5 * (d[half] + d[::-1][half]) / gauss[half]
                good = shape["par_counts"][j][half] + shape["par_counts"][j][::-1][half] >= 20
                axes[0].plot(centres[half][good], ratio[good], ls, color=col, lw=1.7, alpha=alpha)
        a = np.abs(centres)
        width = np.diff(edges)
        tail = np.array([np.sum((shape["par_density"][k] * width)[a > 3.0]) for k in range(len(t))])
        axes[1].plot(t, tail / erfc(3 / np.sqrt(2)), "-", color=col, lw=1.8, label=run["label"])
    for c in controls:
        i = 0 if c["kappa"] is None else 1
        tf = tail_fraction_histogram(c["root"])
        axes[1].plot(tf[:, 0], tf[:, 1], "s", color=ps.c(SERIES[i]), ms=4, mfc="none", ls="--", lw=1.2,
                     label=c["label"] + " (stable)")
    axes[0].axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$u=|v_\parallel-\langle v_\parallel\rangle|/\sigma_\parallel(t)$")
    axes[0].set_ylabel(r"$f(u)/f_{\rm Gauss}(u)$")
    ps.plain_log_axis(axes[0], "y")
    handles = [Line2D([], [], color=ps.c(SERIES[i]), lw=1.7, label=r["label"]) for i, r in enumerate(runs) if r["kappa"] is not None]
    handles += [Line2D([], [], color=ps.MUTED_CLR, ls=ls, lw=1.7, label=name)
                for (ls, _), name in zip(styles, ("initial", "end of linear phase", "final"))]
    legend_below(axes[0], handles, ncol=2)
    axes[1].axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
    axes[1].set_xlabel(r"$t\Omega_{ci}$")
    axes[1].set_ylabel(r"$P(u>3)\,/\,P_{\rm Gauss}(u>3)$")
    legend_below(axes[1], ncol=1)
    ps.style_axes(axes[0], "(a) parallel distribution over the Gaussian")
    ps.style_axes(axes[1], r"(b) probability beyond $3\sigma_\parallel$")
    ps.save(fig, out / "fig10_tail_shape.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs=3, type=Path, help="bi-Maxwellian, kappa 5, kappa 3 result roots")
    p.add_argument("--control", action="append", type=Path, default=[], help="isotropic control result root")
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = load(a.runs, a.control)
    tl_runs = [tlt.load(r) for r in a.runs]
    tracks = {r["name"]: tlt.trajectory(r, 5.0, 15.0) for r in tl_runs}
    sats = {r["name"]: tlt.saturation(r, tracks[r["name"]]) for r in tl_runs}
    fig1_conventions(data, a.outdir)
    fig2_mode_growth(data, a.outdir, tracks)
    equal_t, equal_core, inv = fig3_growth_rate(data, a.outdir)
    energy = fig4_wave_energy(data, a.outdir, sats)
    fig5_trajectories(data, a.outdir, tracks)
    bands = fig6_band_anisotropy(data, a.outdir)
    fig7_numerical(data, a.outdir)
    fig8_kappa_index(data, a.outdir)
    relax = fig9_kappa_relaxation(data, a.outdir)
    fig10_tail_shape(data, a.outdir)
    with (a.outdir / "article1_kappa_numbers.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["run", "nu0", "nu0_err", "c", "c_err", "chi2_dof", "background_share_end"])
        for name, r in relax.items():
            w.writerow([name, f"{r['nu0']:.5f}", f"{r['nu0_err']:.5f}", f"{r['c']:.4f}", f"{r['c_err']:.4f}",
                        f"{r['chi2_dof']:.2f}", f"{r['background_share_end']:.3f}"])
        for c in data["controls"]:
            if c.get("nu0") is not None:
                w.writerow([c["name"], f"{c['nu0']:.5f}", f"{c['nu0_err']:.5f}", "", "", "", "1.000"])

    base = data["runs"][0]
    with (a.outdir / "article1_numbers.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["run", "gamma_pic", "gamma_err_pipeline", "gamma_err_autocorrelated", "gamma_theory_equal_T",
                    "gamma_theory_equal_core", "t_saturation", "wave_energy_at_saturation",
                    "wave_energy_vs_bimaxwellian", "A_i_at_saturation", "A_marginal_at_saturation",
                    "A_tail_band_final", "A_tail_band_final_err", "control"])
        for run in data["runs"]:
            x = 0.0 if run["kappa"] is None else 1.0 / run["kappa"]
            e, s = energy[run["name"]], sats[run["name"]]
            ctrl = (data["audits"][run["name"]].get("baseline") or {}).get("control", "")
            w.writerow([run["name"], f"{run['gamma']:.4f}", f"{run['gamma_err']:.4f}",
                        f"{run['gamma_err_corr']:.4f}", f"{np.interp(x, inv, equal_t):.4f}",
                        f"{np.interp(x, inv, equal_core):.4f}", f"{e['t_saturation']:.1f}",
                        f"{e['wave_energy_at_saturation']:.4f}",
                        f"{e['wave_energy_at_saturation'] / energy[base['name']]['wave_energy_at_saturation'] - 1:+.3f}",
                        f"{s['a_i']:.3f}", f"{s['a_marginal']:.3f}",
                        f"{bands[run['name']][3, 0]:.3f}", f"{bands[run['name']][3, 1]:.3f}", ctrl])
    print(f"Article 1 figures written to {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
