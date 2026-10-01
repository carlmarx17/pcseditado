"""resonant_anisotropy.py: the anisotropy a parallel wave sees, A(v_par) = -(dW/dv)/(v F)."""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import resonant_anisotropy as ra

HERE = Path(__file__).resolve().parent
N = 400_000


def _sample(rng, kappa, anisotropy=2.0):
    """Parallel velocity and v_perp^2/2 of a bi-Maxwellian / bi-kappa of unit parallel variance (loader mixture)."""
    z = rng.standard_normal((3, N))
    scale = 1.0 if kappa is None else np.sqrt((kappa - 1.5) / rng.gamma(kappa - 0.5, 1.0, N))
    return z[0] * scale, 0.5 * anisotropy * (z[1] ** 2 + z[2] ** 2) * scale ** 2


@pytest.mark.parametrize("kappa", [None, 5.0, 3.0])
def test_bi_maxwellian_and_bi_kappa_are_flat_at_the_loaded_anisotropy(kappa):
    u, e_perp = _sample(np.random.default_rng(3), kappa)
    rows = ra.analyse_sample(u, e_perp, np.ones(N))
    for row in rows:
        if row["kind"] == "window" and row["centre"] > 3.0:
            continue                                    # few particles: checked through the bands
        assert abs(row["A"] - 2.0) < 5.0 * row["A_err"] + 0.02, row


def test_band_shares_add_up_to_the_global_anisotropy_exactly():
    u, e_perp = _sample(np.random.default_rng(4), 3.0)
    rows = ra.analyse_sample(u, e_perp, np.ones(N))
    bands = [r for r in rows if r["kind"] == "band"]
    assert sum(r["energy_share"] for r in bands) == pytest.approx(1.0, abs=1e-12)
    assert sum(r["energy_share"] * r["A"] for r in bands) == pytest.approx(rows[0]["A"], abs=1e-10)
    assert rows[0]["A"] == pytest.approx(np.sum(e_perp) / np.sum(u ** 2))


def test_the_tail_of_a_kappa_carries_more_of_the_weight_than_a_maxwellian():
    rng = np.random.default_rng(5)
    tail = {}
    for kappa in (None, 3.0):
        u, e_perp = _sample(rng, kappa)
        tail[kappa] = [r for r in ra.analyse_sample(u, e_perp, np.ones(N)) if r["kind"] == "band"][-1]["energy_share"]
    assert tail[None] == pytest.approx(0.032, abs=0.006)     # beyond 3 sigma, Gaussian
    assert tail[3.0] > 4.0 * tail[None]


def test_a_core_relaxed_to_isotropy_shows_in_the_core_band_only():
    rng = np.random.default_rng(6)
    u, e_perp = _sample(rng, None)
    e_perp = np.where(np.abs(u) < 1.5, 0.5 * e_perp, e_perp)   # T_perp halved for |u| < 1.5
    bands = [r for r in ra.analyse_sample(u, e_perp, np.ones(N)) if r["kind"] == "band"]
    assert bands[0]["A"] == pytest.approx(1.0, abs=0.05)
    assert bands[-1]["A"] == pytest.approx(2.0, abs=6.0 * bands[-1]["A_err"])


def test_measure_and_plot_on_a_synthetic_run(tmp_path):
    case = "mirror_bimaxwellian_moderate"
    run_dir, out = tmp_path / "run", tmp_path / "results" / case
    env = {**os.environ, "PSC_PROFILE": case, "PSC_ANALYSIS_DATA_DIR": str(run_dir),
           "MPLCONFIGDIR": str(tmp_path / "mpl")}
    subprocess.run([sys.executable, "synthetic_run.py", str(run_dir), "--case", case, "--ngrid", "48",
                    "--ppc", "300", "--snapshots", "3"], cwd=HERE, env=env, check=True, capture_output=True)
    subprocess.run([sys.executable, "resonant_anisotropy.py", "measure", "--data-dir", str(run_dir),
                    "--outdir", str(out / "03_particles")], cwd=HERE, env=env, check=True, capture_output=True)
    with open(out / "03_particles" / ra.PRODUCT, newline="") as handle:
        rows = list(csv.DictReader(handle))
    globals_ = [float(r["A"]) for r in rows if r["kind"] == "global"]
    assert len(globals_) == 3 and all(abs(a - 2.0) < 0.1 for a in globals_)
    assert {r["frame"] for r in rows} == {"local field"}
    subprocess.run([sys.executable, "resonant_anisotropy.py", "plot", str(out), "--outdir", str(tmp_path / "fig")],
                   cwd=HERE, env=env, check=True, capture_output=True)
    assert (tmp_path / "fig" / "resonant_anisotropy.png").exists()
    assert (tmp_path / "fig" / "resonant_anisotropy_bands.csv").exists()
