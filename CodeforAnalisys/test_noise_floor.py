"""Noise floor and null test from a stable control (noise_floor.py)."""
import csv
import json

import numpy as np
import pytest

import energy_audit
import noise_floor
from test_evidence_tooling import control, fake_run, write_csv

T = np.linspace(0.0, 100.0, 1001)
NOISE = 1e-3 * (1.0 + 0.02 * T)                                   # grows as the electrons heat
WAVE = np.where(T > 10.0, np.minimum(1e-6 * np.exp(0.2 * (T - 10.0)), 0.05), 0.0)


def test_scaled_control_noise_leaves_the_wave():
    """The run carries 0.98 of the control's noise plus a wave that starts at t = 10."""
    part = noise_floor.subtract_noise(T, np.sqrt(0.98 * NOISE + WAVE), T, np.sqrt(NOISE), settle=3.0, quiet_end=10.0)
    assert part["calibrated"] and part["scale"] == pytest.approx(0.98)
    assert part["quiet_scatter"] == pytest.approx(0.0, abs=1e-12)
    s = part["_series"]
    assert s["omega_ci_t"][0] == pytest.approx(3.0)                # quiet-start build-up left out
    assert s["excess"] == pytest.approx(WAVE[T >= 3.0], abs=1e-12)
    # wave = noise: 1e-6 exp(0.2 (t - 10)) = 0.98e-3 (1 + 0.02 t) at t = 47.8
    assert part["signal_dominates_from_omegaci"] == pytest.approx(47.8, abs=0.15)
    assert part["noise_fraction_end"] == pytest.approx(0.98 * NOISE[-1] / (0.98 * NOISE[-1] + 0.05))

    quiet = noise_floor.subtract_noise(T, np.sqrt(0.98 * NOISE), T, np.sqrt(NOISE), settle=3.0, quiet_end=10.0)
    assert quiet["signal_dominates_from_omegaci"] is None and quiet["min_noise_fraction"] == pytest.approx(1.0)
    # Without a quiet interval the control is used as it is.
    plain = noise_floor.subtract_noise(T, np.sqrt(0.98 * NOISE), T, np.sqrt(NOISE), settle=3.0, quiet_end=None)
    assert not plain["calibrated"] and plain["scale"] == 1.0


def noise_modes(seed=0, n=6):
    rng = np.random.default_rng(seed)
    return {"omega_ci_t": T, **{f"amp_over_B0_kpar{0.628 * (i + 1):.3f}_kperp+0.000": 3e-4 * rng.rayleigh(1.0, T.size)
                               for i in range(n)}}


def test_null_gain_grows_as_the_window_shrinks_and_never_reaches_a_real_mode():
    modes = noise_modes()
    amplitudes = [v for k, v in modes.items() if k.startswith("amp_over_B0")]
    long, short = noise_floor.null_gain(T, amplitudes, 30.0), noise_floor.null_gain(T, amplitudes, 1.5)
    assert 1.0 < long["gain"] < 2.0 < short["gain"]                # noise fakes short growth, not long growth
    assert long["windows"] == 6 * (T.size - 300)
    assert noise_floor.null_gain(T, amplitudes, 150.0) is None      # longer than the control
    fits = [{"gamma": 0.1, "linear_phase_start": 10.0, "linear_phase_end": 40.0, "amplitude_gain": float(np.exp(3.0))},
            {"gamma": 0.6, "linear_phase_start": 5.0, "linear_phase_end": 6.5, "amplitude_gain": 2.46},
            {"gamma": 0.1, "linear_phase_start": 0.0, "linear_phase_end": 150.0, "amplitude_gain": 50.0}]
    verdicts = [f["verdict"] for f in noise_floor.null_test(fits, modes, settle=3.0)]
    assert verdicts == ["above_null", "at_null", "untested"]


def with_fluctuations(root, wave, fits):
    """Add the fluctuation, mode and fit tables the noise floor reads to a fake_run tree."""
    phys = root / "09_physical_diagnostics"
    scale = 0.98 if wave is not None else 1.0
    signal = wave if wave is not None else 0.0 * T
    write_csv(phys / "field_fluctuation_table.csv", [
        {"omega_ci_t": t, "delta_B_vec_rms_over_B0": np.sqrt(scale * n + w),
         "delta_B_parallel_rms_over_B0": np.sqrt(scale * 0.25 * n),
         "delta_B_perp_rms_over_B0": np.sqrt(scale * 0.75 * n + w)} for t, n, w in zip(T, NOISE, signal)])
    modes = noise_modes(seed=1 if wave is not None else 0)
    write_csv(phys / "mode_amplitude_timeseries.csv", [dict(zip(modes, row)) for row in zip(*modes.values())])
    write_csv(phys / "mode_growth_table.csv", fits)
    summary = [{"series": "mode", "gamma": 0.1, "linear_phase_start": 10.0, "linear_phase_end": 40.0,
                "amplitude_gain": float(np.exp(3.0)), "fit_ok": 1},
               {"series": "total", "gamma": 0.012, "linear_phase_start": 10.0, "linear_phase_end": 40.0,
                "amplitude_gain": 1.43, "fit_ok": 1},
               {"series": "parallel", "gamma": 0.01, "linear_phase_start": 10.0, "linear_phase_end": 40.0,
                "amplitude_gain": 1.35, "fit_ok": 0}]
    write_csv(phys / "growth_rate_summary.csv", summary)
    return root


FITS = [{"k_parallel_di": 0.314, "k_perp_di": 0.0, "gamma": 0.1, "linear_phase_start": 10.0, "linear_phase_end": 40.0,
         "amplitude_gain": float(np.exp(3.0)), "fit_ok": 1, "physical_branch": "ion-cyclotron"},
        {"k_parallel_di": 0.628, "k_perp_di": 1.571, "gamma": 0.6, "linear_phase_start": 5.0, "linear_phase_end": 6.5,
         "amplitude_gain": 2.46, "fit_ok": 1, "physical_branch": "mirror"},
        {"k_parallel_di": 0.942, "k_perp_di": 0.0, "gamma": 0.9, "linear_phase_start": 7.0, "linear_phase_end": 7.5,
         "amplitude_gain": 1.5, "fit_ok": 0, "physical_branch": ""}]


def test_report_adds_the_noise_floor_and_the_null_test(tmp_path):
    from quality_report import write_report
    run = with_fluctuations(fake_run(tmp_path / "run", electron_gain=2380.0, ion_loss=400.0), WAVE, FITS)
    ctrl = with_fluctuations(control(tmp_path, "ctrl"), None, FITS[2:])
    runs = write_report([run, ctrl], tmp_path / "report")
    checks = {c["check"]: c for c in runs[0]["checks"]}
    floor, null = checks["magnetic_noise_floor"], checks["growth_null_test"]
    assert floor["status"] == "WARN"
    assert "parallel rms never above the noise" in floor["reason"]
    assert "total (gamma 0.012" in floor["reason"]                 # accepted rms fit of the noise
    assert floor["metrics"]["components"]["total"]["signal_dominates_from_omegaci"] == pytest.approx(47.8, abs=0.15)
    assert floor["metrics"]["components"]["parallel"]["signal_dominates_from_omegaci"] is None
    assert null["status"] == "PASS"
    assert null["metrics"]["counts"] == {"above_null": 1, "marginal": 0, "at_null": 1, "untested": 0}
    assert "magnetic_noise_floor" not in {c["check"] for c in runs[1]["checks"]}   # the control has no control

    out = tmp_path / "report" / "noise_floor"
    with (out / "noise_floor_fits.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["kind"], r["verdict"]) for r in rows] == [("rms series", "noise_dominated"),
                                                          ("Fourier mode", "above_null"), ("Fourier mode", "at_null")]
    assert (out / "noise_floor.png").is_file() and (out / "noise_floor_series.csv").is_file()
    # plot_style.save names the registry after the entry script, so look it up by figure.
    records = [json.loads(line) for registry in out.glob("figure_qa_*.jsonl")
               for line in registry.read_text().splitlines() if line.strip()]
    record = [r for r in records if r["file"] == "noise_floor.png"][-1]
    assert record["issues"] == []


def test_runs_without_fluctuation_tables_get_no_noise_floor(tmp_path):
    audits = [energy_audit.audit_run(fake_run(tmp_path / "run", electron_gain=2380.0, ion_loss=400.0)),
              energy_audit.audit_run(control(tmp_path, "ctrl"))]
    energy_audit.pair_controls(audits)
    assert noise_floor.audit_paired(audits) == []
