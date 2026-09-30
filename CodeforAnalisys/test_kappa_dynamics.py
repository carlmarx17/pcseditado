"""kappa_dynamics.py on product trees with a known answer, and the signed kappa index.

The trees mimic what vdf_spatial.py and physical_diagnostics.py write: an
index relaxing as 1/kappa = (1/kappa_0) exp(-nu_0 t - c F(t)) under a
prescribed fluctuation energy W(t), b-binned profiles with a prescribed
slope, and an isotropic control with no waves next to the run.
"""

import csv

import numpy as np
import pytest

import kappa_dynamics as kd
import kappa_eff as ke

NU0, C = 1.5e-3, 0.07


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, kappa0, nu0, c, *, wave=True, slope=0.0, n=60, t_end=158.0, err=0.002, seed=0):
    rng = np.random.default_rng(seed)
    t_f = np.linspace(0.0, t_end, 400)
    w = 1e-3 + (0.05 / (1.0 + np.exp(-(t_f - 60.0) / 8.0)) if wave else 0.0 * t_f)
    _write(root / "09_physical_diagnostics" / "field_fluctuation_table.csv",
           [{"step": i, "omega_ci_t": t, "delta_B_vec_rms_over_B0": np.sqrt(v),
             "delta_B_parallel_rms_over_B0": np.sqrt(0.1 * v)} for i, (t, v) in enumerate(zip(t_f, w))])
    fluence = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t_f))])
    t = np.linspace(0.0, t_end, n)
    inv = np.exp(-nu0 * t - c * np.interp(t, t_f, fluence)) / kappa0
    rows, b_rows = [], []
    for k, (tk, ik) in enumerate(zip(t, inv)):
        for population, e in (("all", err), ("hole", 2 * err), ("peak", 2 * err)):
            rows.append({"step": 1000 * k, "omega_ci_t": tk, "population": population,
                         "count": 100000, "b_mean_over_B0": 1.0, "A_local_b": 2.0,
                         "kappa_eff": 1.0 / ik, "inv_kappa_signed": ik + rng.normal(0.0, e),
                         "inv_kappa_signed_error": e,
                         "tail_fraction_par_3sigma": 0.0027 * (1.0 + 10.0 * ik)})
        for b in (0.9, 0.95, 1.0, 1.05):
            b_rows.append({"step": 1000 * k, "omega_ci_t": tk, "b_ref_over_B0": 1.0,
                           "b_center": b, "b_mean": b, "count": 50000, "A": 2.0,
                           "trapped_fraction": 0.1, "kappa_eff": 1.0 / ik,
                           "inv_kappa_signed": ik + slope * np.log(b) + rng.normal(0.0, 2 * err),
                           "inv_kappa_signed_err": 2 * err})
    _write(root / "03_particles" / "vdf_kappa_series.csv", rows)
    _write(root / "03_particles" / "vdf_kappa_b_series.csv", b_rows)
    return root


def test_relaxation_model_separates_background_and_wave_driven_rates(tmp_path):
    run = make_run(tmp_path / "tree" / "mirror_bikappa3_moderate", 3.0, NU0, C)
    out = kd.analyse([run], [], tmp_path / "out")
    fit = out["relaxation"]["mirror_bikappa3_moderate"]
    assert fit["nu0"] == pytest.approx(NU0, abs=3 * fit["nu0_err"])
    assert fit["c"] == pytest.approx(C, abs=3 * fit["c_err"])
    assert fit["r2"] > 0.99
    # Windowed rates follow nu_0 + c W point by point.
    for row in out["rates"]["mirror_bikappa3_moderate"]:
        assert row["rate"] == pytest.approx(NU0 + C * row["W_mean"], abs=4 * row["rate_err"] + 2e-4)
    for name in ("kappa_field_evolution.png", "kappa_relaxation.png", "kappa_vs_local_field.png",
                 "kappa_dynamics_timeseries.csv", "kappa_relaxation_fit.json"):
        assert (tmp_path / "out" / name).exists()
    with (tmp_path / "out" / "kappa_dynamics_timeseries.csv").open() as handle:
        first = next(csv.DictReader(handle))
    assert float(first["tail_fraction_3sigma"]) == pytest.approx(0.0027 * (1 + 10 / 3.0), rel=0.05)


def test_gaussian_tail_reference_and_kappa_values():
    assert kd.GAUSS_TAIL_3SIGMA == pytest.approx(0.0027, rel=1e-3)
    assert kd._gauss_ratio_of_kappa(None) == pytest.approx(1.0)
    # Student t with nu = 2 kappa - 1 and unit variance, beyond 3 sigma
    assert kd._gauss_ratio_of_kappa(3.0) == pytest.approx(4.3, abs=0.1)


def test_isotropic_control_next_to_the_run_is_paired_and_measures_the_background(tmp_path):
    run = make_run(tmp_path / "tree" / "mirror_bikappa3_moderate", 3.0, NU0, C)
    make_run(tmp_path / "tree" / "mirror_bikappa3_isotropic", 3.0, NU0, 0.0, wave=False, seed=1)
    make_run(tmp_path / "tree" / "mirror_bikappa5_isotropic", 5.0, 3 * NU0, 0.0, wave=False, seed=2)
    out = kd.analyse([run], [], tmp_path / "out")
    names = [r["name"] for r in out["runs"]]
    assert names == ["mirror_bikappa3_moderate", "mirror_bikappa3_isotropic"]
    control = out["relaxation"]["mirror_bikappa3_isotropic"]
    assert control["nu0"] == pytest.approx(NU0, abs=3 * control["nu0_err"])
    assert not np.isfinite(control["c"])


def test_local_field_slope_is_null_without_dependence_and_found_with_one(tmp_path):
    flat = make_run(tmp_path / "a" / "mirror_bikappa3_moderate", 3.0, NU0, C, slope=0.0)
    tilted = make_run(tmp_path / "b" / "mirror_bikappa3_moderate", 3.0, NU0, C, slope=0.2, seed=3)
    for root, expected in ((flat, 0.0), (tilted, 0.2)):
        rows = kd.local_field_slopes(kd.load_run(root))
        slopes = np.array([r["slope"] for r in rows])
        errors = np.array([r["slope_err"] for r in rows])
        mean = np.sum(slopes / errors ** 2) / np.sum(1 / errors ** 2)
        assert mean == pytest.approx(expected, abs=4 * np.sqrt(1 / np.sum(1 / errors ** 2)))


def test_older_products_without_the_signed_index_are_read():
    assert kd.index_of({"kappa_eff": "inf", "kappa_eff_error": "nan"}) == (0.0, pytest.approx(np.nan, nan_ok=True))
    inv, err = kd.index_of({"kappa_eff": "4.0", "kappa_eff_error": "0.1"})
    assert inv == pytest.approx(0.25) and err == pytest.approx(0.1 / 16)
    inv, err = kd.index_of({"kappa_eff": "inf", "inv_kappa_signed": "-0.004",
                            "inv_kappa_signed_error": "0.002"})
    assert (inv, err) == (pytest.approx(-0.004), pytest.approx(0.002))


def test_signed_index_is_1_over_kappa_and_continuous_through_the_maxwellian():
    s_max = ke.DEFAULT_S_MAX
    for kappa in (3.0, 5.0, 50.0):
        assert ke.signed_inverse_kappa(ke.truncated_K_kappa(kappa, s_max)) == pytest.approx(1 / kappa, rel=1e-6)
    k_m = ke.truncated_K_maxwellian(s_max)
    assert ke.signed_inverse_kappa(k_m) == pytest.approx(0.0, abs=2e-4)
    grid = np.linspace(k_m - 0.05, ke.truncated_K_kappa(20.0, s_max), 200)
    values = np.array([ke.signed_inverse_kappa(k) for k in grid])
    assert np.all(np.diff(values) > 0)                        # monotonic, no jump at the edge
    assert np.max(np.diff(values)) < 5 * np.median(np.diff(values))


@pytest.mark.parametrize("kappa", [None, 3.0])
def test_influence_function_error_matches_the_bootstrap(kappa):
    rng = np.random.default_rng(4)
    v = (ke.sample_bikappa(60_000, kappa, 1.0, np.sqrt(2.0), rng=rng) if kappa
         else ke.sample_bimaxwellian(60_000, 1.0, np.sqrt(2.0), rng=rng))
    fast = ke.kappa_eff_from_velocities(*v)["inv_kappa_signed_err"]
    boot = ke.kappa_eff_from_velocities(*v, n_boot=40, rng=np.random.default_rng(5))["inv_kappa_signed_err"]
    assert fast == pytest.approx(boot, rel=0.35)


def test_joint_fit_recovers_one_wave_coefficient_for_different_backgrounds(tmp_path):
    a = make_run(tmp_path / "tree" / "mirror_bikappa3_moderate", 3.0, NU0, C, seed=6)
    b = make_run(tmp_path / "tree" / "mirror_bikappa5_moderate", 5.0, 2 * NU0, C, seed=7)
    joint = kd.analyse([a, b], [], tmp_path / "out")["joint"]
    assert joint["c"] == pytest.approx(C, abs=3 * joint["c_err"])
    assert joint["nu0"]["mirror_bikappa5_moderate"] == pytest.approx(
        2 * NU0, abs=3 * joint["nu0_err"]["mirror_bikappa5_moderate"])
    assert joint["chi2_dof"] < 2.0
