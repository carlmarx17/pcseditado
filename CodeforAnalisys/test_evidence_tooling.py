"""Known-answer tests for the evidence tooling: energy audit, figure content
check, ordered runner, shot-noise-aware initial validation, log coverage and
the growth-map display zoom."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import energy_audit
import plot_style as ps
from quality_report import inspect_run

HERE = Path(__file__).resolve().parent
PHYSICS = {"mass_ratio": 200.0, "B0": 0.08, "beta_i_parallel": 5.0, "A_i": 2.0,
           "beta_e_parallel": 1.0, "A_e": 1.0, "domain_di": 20.0, "grid": [576, 576],
           "nicell_from_profile": 1000, "dx_de": 0.49104635487432063}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_run(root, kappa=None, electron_gain=0.0, ion_loss=0.0, swap=False, global_diag=True,
             growth_series="mode", a_i=2.0, beta_i=5.0, grid=(576, 576), dx_de=None, volume=80000.0):
    """Result tree with DiagEnergies and window tables of a prescribed budget."""
    root.mkdir(parents=True)
    physics = {**PHYSICS, "kappa": kappa, "A_i": a_i, "beta_i_parallel": beta_i, "grid": list(grid),
               "dx_de": dx_de or PHYSICS["dx_de"], "domain_di": PHYSICS["domain_di"] * grid[0] / 576}
    (root / f"{root.name}_analysis_manifest.json").write_text(json.dumps(
        {"case": root.name, "driven_species": "ion", "physics": physics}))
    phys = root / "09_physical_diagnostics"
    b0sq_half = PHYSICS["B0"] ** 2 / 2
    t = np.linspace(0, 100, 21)
    s = t / t[-1]
    if global_diag:
        e_i_ratio = beta_i * (0.5 + a_i)
        e_b, e_i0, e_e0 = b0sq_half * volume, e_i_ratio * b0sq_half * volume, 1.5 * b0sq_half * volume
        rows = []
        for ti, si in zip(t, s):
            e_i, e_e = e_i0 - ion_loss * si, e_e0 + electron_gain * si
            if swap:
                e_i, e_e = e_e, e_i
            rows.append({"time_code": ti * 2500, "omega_ci_t": ti, "E_E": 0.0, "E_B": e_b,
                         "E_e": e_e, "E_i": e_i, "E_total": e_b + e_e + e_i})
        write_csv(phys / "global_energy_table.csv", rows)
    t_e0, t_i0 = b0sq_half, beta_i * b0sq_half
    e_int_i0, e_int_e0 = 1.5 * (t_i0 + 2 * a_i * t_i0) / 3, 1.5 * t_e0
    scale = 1.0 / volume                                   # per unit volume, n = 1
    write_csv(phys / "energy_table.csv", [
        {"step": int(ti * 2500), "omega_ci_t": ti, "E_kin_bulk": 0.0,
         "E_internal_i": e_int_i0 - ion_loss * scale * si,
         "E_internal_e": e_int_e0 + electron_gain * scale * si, "E_B": 0.0, "A_e": 1.0}
        for ti, si in zip(t, s)])
    write_csv(phys / "growth_rate_summary.csv", [
        {"series": growth_series, "gamma": 0.1, "linear_phase_start": 10.0, "linear_phase_end": 30.0,
         "fit_ok": 1, "fit_reject_reason": ""}])
    return root


# ── Energy audit ─────────────────────────────────────────────────────────────

def test_conserving_budget_is_not_failed_and_not_passed(tmp_path):
    a = energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=90.0, ion_loss=100.0))
    assert a["global"]["mapping"]["status"] == "PASS"
    assert a["global"]["status"] == "UNVERIFIED" and a["status"] == "UNVERIFIED"
    assert a["window"]["status"] == "UNVERIFIED"


def test_heating_beyond_the_driver_release_fails(tmp_path):
    a = energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=2000.0, ion_loss=400.0))
    assert a["status"] == "FAIL"
    assert a["global"]["error_over_driver_release"] == pytest.approx(1600 / 400)
    # T_e factor and the Debye resolution follow from the prescribed gain.
    t_e_factor = 1 + 2000.0 / (1.5 * PHYSICS["B0"] ** 2 / 2 * 80000.0)
    assert a["window"]["electron_heating_factor"] == pytest.approx(t_e_factor)
    lde0 = np.sqrt(PHYSICS["B0"] ** 2 / 2)
    assert a["window"]["dx_over_lambda_De_initial"] == pytest.approx(PHYSICS["dx_de"] / lde0)
    assert a["early_time"]["T_e_over_T_e0_at_end"] == pytest.approx(1 + (t_e_factor - 1) * 0.3)
    assert a["window_vs_global"]["domain_wide"]


def test_window_proxy_alone_detects_the_heating(tmp_path):
    a = energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=2000.0, ion_loss=400.0,
                                        global_diag=False))
    assert a["global"] is None and a["status"] == "FAIL" and "prt window" in a["reason"]


def test_swapped_species_columns_fail_the_mapping(tmp_path):
    a = energy_audit.audit_run(fake_run(tmp_path / "run", swap=True))
    assert a["global"]["mapping"]["status"] == "FAIL" and a["status"] == "FAIL"


def test_equal_heating_across_distributions_is_common_mode(tmp_path):
    audits = [energy_audit.audit_run(fake_run(tmp_path / name, kappa=k, electron_gain=2000.0,
                                              ion_loss=400.0 + 50 * i))
              for i, (name, k) in enumerate([("maxw", None), ("k5", 5.0), ("k3", 3.0)])]
    groups = energy_audit.group_runs(audits)
    assert len(groups) == 1 and groups[0]["common_mode"] and groups[0]["relative_spread"] < 1e-12
    different = energy_audit.audit_run(fake_run(tmp_path / "hot", kappa=3.0, electron_gain=4000.0))
    assert not energy_audit.group_runs(audits[:1] + [different])[0]["common_mode"]
    cold = [energy_audit.audit_run(fake_run(tmp_path / f"cold{i}", kappa=k, electron_gain=g))
            for i, (k, g) in enumerate([(None, 1.0), (3.0, 3.0)])]
    assert energy_audit.group_runs(cold)[0]["interpretation"].startswith("No significant electron heating")


def test_quality_report_uses_audit_and_rejects_legacy_estimator(tmp_path):
    run = inspect_run(fake_run(tmp_path / "run", electron_gain=2000.0, ion_loss=400.0,
                               global_diag=False, growth_series="total"))
    checks = {c["check"]: c for c in run["checks"]}
    assert checks["global_energy"]["status"] == "FAIL"
    assert checks["growth_fit"]["status"] == "UNVERIFIED" and "Legacy" in checks["growth_fit"]["reason"]
    assert run["scientific_status"] == "FAIL"


# ── Figure content check ─────────────────────────────────────────────────────

def test_figure_check_flags_what_a_reader_cannot_see(tmp_path):
    t = np.arange(20.0)
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(t, np.where(t % 2 == 0, t, np.nan), label="interleaved")   # the v5 defect
    ax[0].plot(t, t, label="fine")
    ax[0].plot([], [], "o", label="legend proxy")
    ax[1].plot(t, np.where(t % 2 == 0, t, np.nan), "o", label="markers")
    mesh = ax[2].pcolormesh(np.full((3, 3), np.nan))
    fig.colorbar(mesh, ax=ax[2])
    issues = ps.figure_content(fig)["issues"]
    plt.close(fig)
    assert any("'interleaved' draws nothing" in i for i in issues)
    assert not any("fine" in i or "markers" in i or "proxy" in i for i in issues)
    assert any("QuadMesh has no finite values" in i for i in issues)
    assert len(issues) == 3                     # + the empty map panel; colorbar ignored


def test_saving_records_the_content_next_to_the_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ps.save(fig, tmp_path / "ok.png")
    [registry] = tmp_path.glob("figure_qa_*.jsonl")
    entry = json.loads(registry.read_text().splitlines()[-1])
    assert entry["file"] == "ok.png" and entry["issues"] == []


# ── Growth-map display ───────────────────────────────────────────────────────

def test_display_zoom_never_hides_an_accepted_mode(tmp_path):
    from growth_rate_map import plot_growth_rate_map
    kpar, kperp = np.array([-0.6, -0.3, 0.0, 0.3, 0.6]), np.array([0.0, 0.3, 0.6, 0.9, 1.2])
    gamma = np.full((5, 5), np.nan); gamma[3, 4] = 0.2       # only growth at k_perp = 1.2
    power = np.full((5, 5), 1e-12); power[3, 4] = 1.0
    result = {"kpar": kpar, "kperp": kperp, "gamma": gamma, "rvalue": np.where(np.isfinite(gamma), .99, np.nan),
              "final_power": power, "min_rvalue": 0.7}
    plot_growth_rate_map(result, tmp_path / "map.png", "parallel", display_kperp_max=0.9)
    entry = json.loads(next(tmp_path.glob("figure_qa_*.jsonl")).read_text().splitlines()[-1])
    assert entry["issues"] == []


# ── Initial validation and log coverage ──────────────────────────────────────

def particle_file(path, n, rng, t_perp_factor=1.0):
    from validate_moments import M_ION, M_ELEC, TI_PAR, TI_PERP, TE_PAR, TE_PERP
    rows = []
    for q, m, tpar, tperp in ((1, M_ION, TI_PAR, TI_PERP * t_perp_factor), (-1, M_ELEC, TE_PAR, TE_PERP)):
        u = rng.normal(size=(n, 3)) * np.sqrt([tperp / m, tperp / m, tpar / m])
        rows.append(np.rec.fromarrays([np.full(n, q, float), np.full(n, m), u[:, 0], u[:, 1], u[:, 2],
                                       np.full(n, 1.0)], names="q,m,px,py,pz,w"))
    with h5py.File(path, "w") as f:
        g = f.create_group("particles"); g.attrs["lo"] = [0, 0, 0]; g.attrs["hi"] = [1, 2, 2]
        f.create_dataset("particles/p0/1d", data=np.concatenate(rows))


def test_correct_small_sample_passes_and_a_real_offset_fails(tmp_path):
    from validate_moments import measure_particles, validation_rows
    rng = np.random.default_rng(5)
    good, bad = tmp_path / "good.000000.h5", tmp_path / "bad.000000.h5"
    particle_file(good, 3000, rng)
    particle_file(bad, 3000, rng, t_perp_factor=1.25)
    for path, expect in ((good, True), (bad, False)):
        measured, _ = measure_particles(path, cori=4 / 3000)       # n = 1 in 4 cells
        rows = [r for r in validation_rows(measured, 0) if r["quantity"] != "n"]
        assert all(r["status"] == "PASS" for r in rows) is expect
        assert all(r["effective_tolerance_pct"] >= r["tolerance_pct"] for r in rows)


def test_log_reaches_nmax_at_the_last_check_multiple():
    from field_residuals import summarize
    rows = [{"step": s, "gauss_max_err": 1e-6, "continuity_max_err": 1e-8} for s in range(0, 16831, 561)]
    meta = {"thresholds": {"gauss": 1e-4}, "nmax_from_log": 16843}
    assert summarize([], rows, meta, [])["log_reaches_nmax"]
    assert not summarize([], rows[:-5], meta, [])["log_reaches_nmax"]


# ── Ordered runner ───────────────────────────────────────────────────────────

FAKE_MAKE = r'''
import json, os, pathlib, sys, time
args = sys.argv[1:]
stage = args[args.index("-C") + 2]
out = pathlib.Path(next(a.split("=", 1)[1] for a in args if a.startswith("OUTPUT_DIR=")))
spec = json.loads(os.environ["FAKE_STAGES"]).get(stage, {})
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({"stage": stage, "start": time.time(), "args": args}) + "\n")
time.sleep(spec.get("sleep", 0.05))
for name in spec.get("files", [f"{stage}/{stage}.txt"]):
    (out / name).parent.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(stage)
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({"stage": stage, "end": time.time()}) + "\n")
sys.exit(spec.get("rc", 0))
'''


def run_runner(tmp_path, stages, *extra, spec=None):
    data = tmp_path / "data"; data.mkdir(exist_ok=True)
    (tmp_path / "fake_make.py").write_text(FAKE_MAKE)
    env = dict(os.environ, FAKE_STAGES=json.dumps(spec or {}), FAKE_LOG=str(tmp_path / "calls.jsonl"),
               MPLCONFIGDIR=str(tmp_path / "mpl"))
    r = subprocess.run([sys.executable, "run_pipeline.py", "--data-dir", str(data), "--case",
                        "mirror_bimaxwellian_moderate", "--results-root", str(tmp_path / "out"),
                        "--launcher", f"{sys.executable} {tmp_path / 'fake_make.py'}",
                        "--stages", *stages, *extra], cwd=HERE, env=env, capture_output=True, text=True)
    state_path = tmp_path / "out" / "mirror_bimaxwellian_moderate" / "pipeline.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()] \
        if (tmp_path / "calls.jsonl").exists() else []
    return r, state, calls


def test_runner_orders_spectral_after_physics_and_resumes_nothing(tmp_path):
    stages = ["manifest", "physics", "spectral", "fields", "structures"]
    r, state, calls = run_runner(tmp_path, stages, "--jobs", "3", spec={"physics": {"sleep": 0.6}})
    assert r.returncode == 0, r.stderr
    assert all(state["stages"][s]["execution_status"] == "PASS" for s in stages)
    t = {c["stage"] + ("_end" if "end" in c else ""): c.get("start", c.get("end")) for c in calls}
    assert t["manifest_end"] <= min(t[s] for s in stages if s != "manifest")
    assert t["spectral"] >= t["physics_end"]
    assert t["fields"] < t["physics_end"]                   # independent stages overlap
    assert not (tmp_path / "out" / "mirror_bimaxwellian_moderate" / ".staging").exists() or \
        not any((tmp_path / "out" / "mirror_bimaxwellian_moderate" / ".staging").iterdir())
    (tmp_path / "calls.jsonl").unlink()
    r, state, calls = run_runner(tmp_path, stages, "--resume", "--jobs", "3")
    assert r.returncode == 0 and calls == []


def test_runner_refuses_a_stage_overwriting_another(tmp_path):
    spec = {"residuals": {"files": ["09_physical_diagnostics/shared.json"]},
            "physics": {"files": ["09_physical_diagnostics/shared.json"], "sleep": 0.3}}
    r, state, _ = run_runner(tmp_path, ["manifest", "residuals", "physics"], "--keep-going", spec=spec)
    assert r.returncode == 1
    statuses = sorted(state["stages"][s]["execution_status"] for s in ("residuals", "physics"))
    assert statuses == ["FAIL", "PASS"]
    failed = next(s for s in ("residuals", "physics") if state["stages"][s]["execution_status"] == "FAIL")
    assert "would overwrite" in state["stages"][failed]["reason"]


def test_runner_keep_going_and_stop(tmp_path):
    spec = {"physics": {"rc": 1}}
    for sub in "abc":
        (tmp_path / sub).mkdir()
    r, state, _ = run_runner(tmp_path / "a", ["manifest", "physics", "spectral", "fields"], "--keep-going", spec=spec)
    assert r.returncode == 1 and state["stages"]["physics"]["execution_status"] == "FAIL"
    assert state["stages"]["spectral"]["execution_status"] == "PASS"   # ordering, not a hard dependency
    assert state["stages"]["spectral"]["window_source"].startswith("script defaults")
    r, state, _ = run_runner(tmp_path / "b", ["manifest", "physics", "spectral"], spec=spec)
    assert r.returncode == 1 and state["stages"]["spectral"]["execution_status"] == "NOT_RUN"
    r, state, _ = run_runner(tmp_path / "c", ["manifest", "physics"], spec={"manifest": {"rc": 2}})
    assert state["stages"]["physics"]["reason"] == "manifest preflight failed"


# ── Kappa tail against a spatial temperature mixture ─────────────────────────

@pytest.mark.parametrize("kind, verdict", [("mixture", "mixture_explains_tail"),
                                           ("kappa", "intrinsic_at_macrocell_scale"),
                                           ("maxwellian", "no_significant_tail")])
def test_mixture_control_separates_intrinsic_and_spatial_tails(kind, verdict):
    """A Gamma(kappa - 1/2) mixture of Maxwellian blocks IS a kappa = 5 window;
    a kappa in every block is intrinsic; a Maxwellian has no tail."""
    import kappa_eff as ke
    from vdf_validation import mixture_tail_budget
    rng = np.random.default_rng(3)
    beta = rng.gamma(4.5, 1.0, size=256)
    blocks = [rng.standard_normal((3, 4000)) / np.sqrt(b) if kind == "mixture" else
              np.array(ke.sample_bikappa(4000, 5.0, 1.0, 1.0, rng=rng)) if kind == "kappa" else
              rng.standard_normal((3, 4000)) for b in beta]
    v = np.concatenate(blocks, axis=1)
    fit = ke.kappa_eff_from_velocities(*v, n_boot=16, rng=rng)
    upper = fit["kappa"] + 2 * fit["kappa_err"] if np.isfinite(fit["kappa_err"]) else np.inf
    result = mixture_tail_budget([b[0].var() for b in blocks], [.5 * (b[1].var() + b[2].var()) for b in blocks],
                                 [4000] * 256, fit["kappa"], v[0].var(), .5 * (v[1].var() + v[2].var()),
                                 kappa_upper=upper)
    assert result["mixture_verdict"] == verdict
    if kind == "mixture":
        assert result["kappa_mixture"] == pytest.approx(fit["kappa"], rel=0.1)


# ── Isotropic controls: numerical-heating baseline ──────────────────────────

def control(tmp_path, name, kappa=None, gain=2000.0, **kw):
    return fake_run(tmp_path / name, kappa=kappa, electron_gain=gain, a_i=1.0, beta_i=25 / 3, **kw)


@pytest.mark.parametrize("physical_gain, status", [(380.0, "PASS"), (300.0, "UNVERIFIED"), (0.0, "FAIL")])
def test_control_subtraction_closes_the_budget_only_when_it_should(tmp_path, physical_gain, status):
    """Ions release 400; the numerics add 2000 to the electrons of both runs."""
    audits = [energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=2000.0 + physical_gain,
                                              ion_loss=400.0)),
              energy_audit.audit_run(control(tmp_path, "ctrl"))]
    energy_audit.pair_controls(audits)
    run, ctrl = audits
    assert ctrl["role"] == "isotropic_control" and "baseline" not in ctrl
    assert run["status"] == "FAIL"                           # the raw trajectory still heats
    b = run["baseline"]
    assert b["control"] == "ctrl" and b["status"] == status
    assert b["global"]["residual_over_driver_release"] == pytest.approx(abs(physical_gain - 400) / 400)
    e_e0 = 1.5 * PHYSICS["B0"] ** 2 / 2 * 80000.0
    assert b["window"]["T_e_factor_baseline_corrected"] == pytest.approx(1 + physical_gain / e_e0)


def test_control_prefers_the_same_distribution_and_the_same_numerics(tmp_path):
    run = energy_audit.audit_run(fake_run(tmp_path / "k3", kappa=3.0, electron_gain=2300.0, ion_loss=400.0))
    candidates = [energy_audit.audit_run(control(tmp_path, "c_maxw")),
                  energy_audit.audit_run(control(tmp_path, "c_k3", kappa=3.0)),
                  energy_audit.audit_run(control(tmp_path, "c_k3_coarse", kappa=3.0,
                                                 dx_de=2 * PHYSICS["dx_de"]))]
    energy_audit.pair_controls([run, *candidates])
    assert run["baseline"]["control"] == "c_k3" and run["baseline"]["match"]["same_distribution"]
    lone = energy_audit.audit_run(fake_run(tmp_path / "maxw", electron_gain=2300.0, ion_loss=400.0))
    energy_audit.pair_controls([lone, candidates[2]])
    assert "baseline" not in lone                            # different dx: not a control of it
    energy_audit.pair_controls([lone, candidates[1]], {"maxw": "c_k3"})
    assert lone["baseline"]["control"] == "c_k3" and not lone["baseline"]["match"]["same_distribution"]
    assert "another distribution" in lone["baseline"]["reason"]


def test_report_adds_the_baseline_check(tmp_path):
    from quality_report import write_report
    roots = [fake_run(tmp_path / "run", electron_gain=2380.0, ion_loss=400.0), control(tmp_path, "ctrl")]
    runs = write_report(roots, tmp_path / "report")
    checks = {c["check"]: c["status"] for c in runs[0]["checks"]}
    assert checks["global_energy"] == "FAIL" and checks["energy_baseline"] == "PASS"
    assert (tmp_path / "report" / "energy_audit" / "energy_audit_baseline.png").is_file()


# ── Spectral index only over a real decay range ──────────────────────────────

def test_power_law_fit_recovers_a_cascade_and_refuses_noise():
    from spectral_analysis import SpectralAnalyzer
    fit = SpectralAnalyzer._fit_power_law
    analyzer = SpectralAnalyzer.__new__(SpectralAnalyzer)
    k = np.logspace(-1, 1.5, 60)
    cascade = np.where(k < 0.5, k ** 2 / 0.25, (k / 0.5) ** (-5.0 / 3.0)) + 1e-7
    result = fit(analyzer, k, cascade)
    assert result["accepted"] and result["slope"] == pytest.approx(-5.0 / 3.0, abs=0.05)
    rng = np.random.default_rng(0)
    noise = 1e-7 * (1 + 0.3 * rng.random(k.size))
    assert not fit(analyzer, k, noise)["accepted"]
    rising = np.where(k < 0.3, 1.0, 1e-6 * k ** 2.8)          # the noise tail of the old figure
    assert not fit(analyzer, k, rising)["accepted"]


def test_control_with_the_same_ion_loss_leaves_nothing_to_close(tmp_path):
    audits = [energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=2000.0, ion_loss=400.0)),
              energy_audit.audit_run(fake_run(tmp_path / "ctrl", electron_gain=2000.0, ion_loss=400.0,
                                              a_i=1.0, beta_i=25 / 3))]
    energy_audit.pair_controls(audits)
    assert audits[0]["baseline"]["status"] == "UNVERIFIED"
    assert "no net ion energy release" in audits[0]["baseline"]["reason"]


def test_smaller_control_box_is_compared_per_unit_volume(tmp_path):
    """A 10 d_i control (a quarter of the volume, same dx) heats per volume as its twin."""
    run = fake_run(tmp_path / "run", electron_gain=2380.0, ion_loss=400.0)
    small = control(tmp_path, "ctrl", gain=2000.0 / 4, grid=(288, 288), volume=80000.0 / 4)
    audits = [energy_audit.audit_run(run), energy_audit.audit_run(small)]
    energy_audit.pair_controls(audits)
    b = audits[0]["baseline"]
    assert b["volume_ratio_run_over_control"] == pytest.approx(4.0)
    assert b["global"]["residual_over_driver_release"] == pytest.approx(20 / 400)
    assert b["status"] == "PASS"
