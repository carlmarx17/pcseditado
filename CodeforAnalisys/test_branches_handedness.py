"""Known-answer tests of the three v6 follow-ups on mode identification.

1. Mirror and ion-cyclotron growth rates are fitted separately
   (physical_diagnostics.classify_mode): a synthetic run with an oblique
   compressive mode (gamma 0.25) and a weaker parallel left-hand wave
   (gamma 0.15) must return both, each in its own series.
2. The handedness of psi_pm is checked physically
   (polarization_dispersion.handedness_check), for both signs of B0, and a
   flipped temporal convention is detected.
3. The mirror threshold of the Brazil plots uses the measured electrons
   (plasma_physics.mirror_threshold_electrons, Hellinger 2007).
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("PSC_PROFILE", "mirror_bimaxwellian_moderate")
HERE = Path(__file__).resolve().parent
CASE = "mirror_bimaxwellian_moderate"


@pytest.fixture(scope="module")
def two_branch_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = {**os.environ, "PSC_PROFILE": CASE, "PSC_ANALYSIS_DATA_DIR": str(root / "run"),
               "MPLCONFIGDIR": str(root / "mpl"), "PSC_FIG_THEME": "paper"}
        env.pop("PSC_ANALYSIS_CONFIG", None)
        subprocess.run([sys.executable, "synthetic_run.py", str(root / "run"), "--case", CASE,
                        "--ngrid", "32", "--snapshots", "31", "--ppc", "20", "--ion-cyclotron"],
                       cwd=HERE, env={**env, "PSC_ANALYSIS_DATA_DIR": ""}, check=True,
                       capture_output=True)
        for script, extra in (
            ("physical_diagnostics.py", ["--data-dir", str(root / "run"), "--outdir", str(root / "phys"),
                                         "--max-map-steps", "1", "--jobs", "1",
                                         "--vdf-cadence-omegaci", "1000"]),
            ("polarization_dispersion.py", ["--fields", str(root / "run" / "pfd.*.h5"),
                                            "--outdir", str(root / "spec")]),
        ):
            result = subprocess.run([sys.executable, script, *extra], cwd=HERE, env=env,
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
        yield root


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_each_branch_gets_its_own_growth_rate(two_branch_run):
    rows = {r["series"]: r for r in _rows(two_branch_run / "phys" / "growth_rate_summary.csv")}
    for series, gamma in (("mode_compressive", 0.25), ("mode_transverse", 0.15)):
        row = rows[series]
        assert row["fit_ok"] == "1", row["fit_reject_reason"]
        assert abs(float(row["gamma"]) / gamma - 1.0) < 0.10, (series, row["gamma"])
    assert "mirror" in rows["mode_compressive"]["amplitude"]
    assert "ion-cyclotron" in rows["mode_transverse"]["amplitude"]
    phase = json.loads((two_branch_run / "phys" / "linear_phase.json").read_text())
    assert phase["classification"] == "compressive_oblique"
    assert phase["physical_branch"] == "mirror"
    ic = phase["branches"]["transverse_parallel"]
    assert ic["mode"]["k_perp_di"] == 0.0 and ic["compressibility"] < 1e-6
    assert phase["branches"]["compressive_oblique"]["compressibility"] > 0.8


def test_weaker_branch_is_followed_even_when_one_dominates(monkeypatch):
    import physical_diagnostics as pd
    n = 32
    z, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    # A strong transverse field at many k hides one weak compressive mode
    # from the total-power ranking.
    rng = np.random.default_rng(3)
    strong = sum(rng.uniform(1, 2) * np.sin(2 * np.pi * (i * z) / n) for i in range(1, 12))
    weak = 1e-3 * np.sin(2 * np.pi * (z + 3 * y) / n)
    fields = {"Bx": strong[None], "By": np.zeros((1, n, n)), "Bz": weak[None]}
    monkeypatch.setattr(pd, "load_fields", lambda step: fields)
    monkeypatch.setattr(pd, "_fluctuation_plane",
                        lambda bx, by, bz: (np.stack([bx[0], by[0], bz[0]]), ("z", "y"), (1., 1.)))
    modes = pd.mode_candidates({0: 0}, 10.0, per_snapshot=8)
    assert any((m["i0"], m["i1"]) == (1, 3) for m in modes)


@pytest.mark.parametrize("theta,compressibility,branch", [
    (72.0, 0.9, "compressive_oblique"), (0.0, 1e-12, "transverse_parallel"),
    (40.0, 0.4, "unclassified"), (72.0, 0.1, "unclassified"), (10.0, 0.6, "unclassified"),
    (np.nan, 0.5, "unclassified")])
def test_classification_rule(theta, compressibility, branch):
    from physical_diagnostics import classify_mode
    assert classify_mode(theta, compressibility) == branch


def test_physical_branch_names_depend_on_the_driver():
    from physical_diagnostics import physical_branch
    assert physical_branch("compressive_oblique", 2.0) == "mirror"
    assert physical_branch("transverse_parallel", 2.0) == "ion-cyclotron"
    assert physical_branch("transverse_parallel", 0.5) == "parallel firehose"
    assert physical_branch("compressive_oblique", 1.0) == ""


def test_split_mode_power_sums_to_total():
    from physical_diagnostics import mode_power
    rng = np.random.default_rng(0)
    comps = rng.standard_normal((3, 16, 16))
    modes = [(1, 3), (2, 0), (8, 8)]
    split = mode_power(comps, modes, split=True)
    np.testing.assert_allclose(split.sum(axis=0), mode_power(comps, modes), rtol=1e-12)


@pytest.mark.parametrize("b0", [0.08, -0.08])
def test_handedness_is_relative_to_b0(b0):
    from polarization_dispersion import gyration_sense, handedness_check
    # A proton gyrates clockwise seen from the tip of B0 (left-hand about B0).
    assert gyration_sense(1.0, b0) == -int(np.sign(b0))
    assert gyration_sense(-1.0, b0) == int(np.sign(b0))
    assert handedness_check(b0)["consistent"]


def test_flipped_time_convention_is_detected(monkeypatch):
    import polarization_dispersion as pol
    original = pol.temporal_dispersion
    monkeypatch.setattr(pol, "temporal_dispersion",
                        lambda A, t, **kw: original(np.conj(A[:, ::-1]), t, **kw))
    assert not pol.handedness_check(0.08)["consistent"]


def test_ion_cyclotron_wave_lands_on_psi_plus_at_positive_omega(two_branch_run):
    params = json.loads((two_branch_run / "spec" / "polarization_fft_parameters.json").read_text())
    assert params["handedness_check"]["consistent"]
    peak = params["channel_peaks"]["plus"]
    # synthetic_run: K_IC = 2 box modes of a 20 d_i box, OMEGA_IC = 0.47 Omega_ci.
    assert np.isclose(peak["k"], 2 * 2 * np.pi / 20.0)
    assert 0 < peak["omega"] and abs(peak["omega"] - 0.47) < 0.1
    # psi_- holds the same wave only as its mirror image at negative omega.
    assert params["channel_peaks"]["minus"]["omega"] < 0


def test_mirror_threshold_with_measured_electrons():
    from plasma_physics import mirror_criterion, mirror_threshold, mirror_threshold_electrons
    beta = np.array([0.5, 1.0, 5.0, 20.0])
    np.testing.assert_allclose(mirror_threshold_electrons(beta, 0.0, 1.0), mirror_threshold(beta))
    for be, ae in ((1.0, 1.0), (8.0, 1.0), (8.0, 1.05), (8.0, 0.95)):
        a = mirror_threshold_electrons(beta, be, ae)
        np.testing.assert_allclose(mirror_criterion(beta, a, be, ae), 0.0, atol=1e-10)
        # Just above the threshold is unstable, just below stable.
        assert np.all(mirror_criterion(beta, a + 1e-3, be, ae) > 0)
        assert np.all(mirror_criterion(beta, a - 1e-3, be, ae) < 0)
    # Hot isotropic electrons stabilise slightly; electron anisotropy dominates.
    at5 = lambda be, ae: float(mirror_threshold_electrons(5.0, be, ae))
    assert at5(0, 1) < at5(1, 1) < at5(8, 1) < 1.01 * at5(0, 1)
    assert at5(8, 1.05) < at5(8, 1) - 0.05 and at5(8, 0.95) > at5(8, 1) + 0.05
