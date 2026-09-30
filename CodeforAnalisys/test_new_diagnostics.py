"""Known-answer tests for the diagnostics added in the 2026-09-28 revision:
heat flux, field residuals, structures, energy exchange, convergence, the
shared plasma formulas and the whistler branch of the linear theory."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import energy_exchange as ee
import field_residuals as fr
import heat_flux_analysis as hf
import structures_analysis as sa
from convergence_study import load_run, changed_parameters
from linear_theory import _whistler_marginal_frequency
from plasma_physics import central_uv, diamagnetic_current_x, velocity_from_u
from polarization_dispersion import to_electron_units


class PlasmaFormulaTests(unittest.TestCase):
    def test_diamagnetic_current_sign(self):
        # B = B0 z, P increasing along y: J = B x grad P / B^2 = -(dP/dy)/B0 x.
        ny = nz = 32
        y = np.arange(ny) * 0.5
        p = 1.0 + 0.1 * np.sin(2 * np.pi * y / (ny * 0.5))[None, :].repeat(nz, axis=0)
        b0 = 0.08
        j = diamagnetic_current_x(p, np.zeros_like(p), np.full_like(p, b0), 0.5, 0.5)
        dpdy = (np.roll(p, -1, 1) - np.roll(p, 1, 1)) / (2 * 0.5)
        np.testing.assert_allclose(j, -dpdy / b0, rtol=1e-12)

    def test_uv_temperature_reduces_to_variance(self):
        rng = np.random.default_rng(0)
        u = 1e-3 * rng.standard_normal(10000)
        vx, *_ = velocity_from_u(u, 0 * u, 0 * u)
        w = np.ones_like(u)
        self.assertAlmostEqual(central_uv(u, vx, w) / np.var(u), 1.0, places=5)


class HeatFluxTests(unittest.TestCase):
    def test_symmetric_distribution_has_no_heat_flux(self):
        rng = np.random.default_rng(1)
        v = rng.standard_normal((3, 200000)) * [[1.0], [1.0], [0.5]]
        m = hf.heat_flux_moments(*v, np.ones(v.shape[1]), 1.0, np.array([0, 0, 1.0]))
        self.assertLess(abs(m["q_par_over_q0"]), 0.01)

    def test_known_skewed_distribution_and_frame_invariance(self):
        # Along b: v in {-1, -1, 2} (zero mean) -> q_par = m, T_par = 2m, T = 2m/3,
        # q0 = 1.5 T sqrt(2T/m) = m sqrt(4/3): q_par/q0 = sqrt(3/4).
        base = np.tile([-1.0, -1.0, 2.0], 10)
        zero = np.zeros_like(base)
        m = hf.heat_flux_moments(zero, zero, base, np.ones_like(base), 1.0, np.array([0, 0, 1.0]))
        self.assertAlmostEqual(m["q_par_over_q0"], np.sqrt(0.75), places=10)
        # Same distribution along a tilted field, plus a bulk drift.
        b = np.array([1.0, 2.0, 2.0]) / 3.0
        v = np.outer(b, base) + np.array([[0.3], [-0.1], [0.2]])
        m2 = hf.heat_flux_moments(*v, np.ones_like(base), 1.0, b)
        self.assertAlmostEqual(m2["q_par_over_q0"], np.sqrt(0.75), places=10)
        self.assertAlmostEqual(m2["q_perp_over_q0"], 0.0, places=6)

    def test_abs_heat_flux_of_a_maxwellian_is_the_sampling_floor(self):
        rng = np.random.default_rng(7)
        n, blocks = 400, 400
        values = [abs(hf.heat_flux_moments(*rng.standard_normal((3, n)), np.ones(n), 1.0,
                                           np.array([0, 0, 1.0]))["q_par_over_q0"])
                  for _ in range(blocks)]
        self.assertAlmostEqual(np.mean(values) / (hf.ABS_Q_FLOOR_COEFF / np.sqrt(n)), 1.0, delta=0.1)

    def test_truncation_removes_the_tail(self):
        core = np.tile([-1.0, 1.0], 50)
        vz = np.concatenate([core, [8.0]])       # one fast particle
        zero = np.zeros_like(vz)
        full = hf.heat_flux_moments(zero, zero, vz, np.ones_like(vz), 1.0, np.array([0, 0, 1.0]))
        cut = hf.heat_flux_moments(zero, zero, vz, np.ones_like(vz), 1.0, np.array([0, 0, 1.0]),
                                   s_max=3.0)
        self.assertGreater(full["q_par_over_q0"], 0.1)
        self.assertLess(abs(cut["q_par_over_q0"]), 0.05)


class FieldResidualTests(unittest.TestCase):
    def test_yee_divergence_of_a_curl_is_roundoff(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((40, 40))
        by = (np.roll(a, -1, 0) - a) / 0.5
        bz = 0.08 - (np.roll(a, -1, 1) - a) / 0.5
        div = fr.div_b_yee(None, by, bz, 0.5, 0.5)
        self.assertLess(np.max(np.abs(div)), 1e-12)
        self.assertGreater(np.max(np.abs(fr.div_b_yee(None, by, bz + 0.01 * a, 0.5, 0.5))), 1e-4)

    def test_log_parsing_assigns_checks_to_steps(self):
        text = ("**** Step 100 / 200, Code Time 1, Wall Time 1\n"
                "***** Pushing particles...\n"
                "gauss: max_err = 2.5e-07 (thres 0.0001)\n"
                "**** Step 101 / 200, Code Time 1, Wall Time 1\n"
                "continuity: max_err = 1e-09 (thres 0.0001)\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "job.out"
            path.write_text(text)
            rows, meta = fr.parse_logs([path])
        self.assertEqual([r["step"] for r in rows], [100, 101])
        self.assertAlmostEqual(rows[0]["gauss_max_err"], 2.5e-7)
        self.assertAlmostEqual(rows[1]["continuity_max_err"], 1e-9)
        self.assertEqual(meta["thresholds"]["gauss"], 1e-4)
        self.assertEqual(meta["nmax_from_log"], 200)


class StructureTests(unittest.TestCase):
    def test_periodic_labels_merge_across_the_boundary(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:3, 5:8] = True
        mask[18:20, 5:8] = True          # same structure, wrapped in z
        labels, count = sa.periodic_label(mask)
        self.assertEqual(count, 1)

    def test_elongated_hole_along_b0(self):
        z, y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        db = -0.2 * np.exp(-((z - 32) / 10.0) ** 2 - ((y - 32) / 3.0) ** 2)
        labels, count = sa.periodic_label(db < -0.05)
        (row,) = sa.structure_catalog(db, labels, count, "hole", 0.1)
        self.assertLess(row["angle_to_B0_deg"], 5.0)
        self.assertGreater(row["L_parallel_di"], 2.5 * row["L_perp_di"])
        self.assertAlmostEqual(row["amplitude"], -0.2, places=2)

    def test_pressure_balanced_snapshot(self):
        z, y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        b0, tperp = 0.08, 0.035
        bz = b0 * (1 + 0.05 * np.cos(2 * np.pi * y / 64))
        n = 1.0 - 0.5 * (bz ** 2 - b0 ** 2) / tperp
        snap = {"x": 0 * bz, "y": 0 * bz, "z": bz, "n_i": n,
                "pperp_i": n * tperp, "pperp_e": 0 * n}
        row, _, _ = sa.analyse(snap, 1.0, 1.0, 0.01, 4)
        self.assertLess(row["corr_n_B"], -0.99)
        self.assertLess(row["pressure_balance_ratio"], 0.05)


class EnergyExchangeTests(unittest.TestCase):
    def test_parallel_split(self):
        shape = (8, 8)
        b = np.stack([np.zeros(shape), np.zeros(shape), np.full(shape, 0.08)])
        e = np.stack([np.full(shape, 0.01), np.zeros(shape), np.full(shape, 0.02)])
        j = np.stack([np.full(shape, 3.0), np.zeros(shape), np.full(shape, 5.0)])
        rates = ee.exchange_rates(e, b, j)
        self.assertAlmostEqual(rates["JE"], 0.03 + 0.10)
        self.assertAlmostEqual(rates["JE_par"], 0.10)
        self.assertAlmostEqual(rates["JE_perp"], 0.03)

    def test_uniform_fields_survive_cell_centring_and_work_integrates(self):
        f = [np.full((6, 6), v) for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
        out = ee.to_cell_centres_yz(*f)
        for a, b in zip(out, f):
            np.testing.assert_allclose(a, b)
        t = np.linspace(0, 10, 11)
        np.testing.assert_allclose(ee.cumulative(t, np.full(11, 2.0)), 2.0 * t)


class ConvergenceTests(unittest.TestCase):
    def _run(self, root: Path, tag: str, grid: int, gamma: float):
        (root / "09_physical_diagnostics").mkdir(parents=True)
        physics = {"grid": [grid, grid], "domain_di": 20.0, "nicell_from_profile": 1000,
                   "dt_code_from_profile": 0.3, "mass_ratio": 200.0, "beta_i_parallel": 5.0,
                   "A_i": 2.0, "beta_e_parallel": 1.0, "A_e": 1.0, "kappa": None, "B0": 0.08,
                   "analysis_conventions_version": 5}
        (root / "x_analysis_manifest.json").write_text(json.dumps(
            {"run_tag": tag, "driven_species": "ion", "physics": physics}))
        with (root / "09_physical_diagnostics" / "growth_rate_summary.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["series", "gamma", "gamma_err", "fit_ok"])
            w.writeheader()
            w.writerow({"series": "total", "gamma": gamma, "gamma_err": 0.01, "fit_ok": 1})
        return root

    def test_realizations_and_refinements_are_told_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = load_run("a", self._run(Path(tmp) / "a", "seed1", 576, 0.20))
            b = load_run("b", self._run(Path(tmp) / "b", "seed2", 576, 0.22))
            c = load_run("c", self._run(Path(tmp) / "c", "seed1", 1152, 0.21))
        self.assertEqual(changed_parameters(b, a), "realization")
        self.assertEqual(changed_parameters(c, a), "grid")
        self.assertAlmostEqual(b["gamma"], 0.22)
        self.assertTrue(a["gamma_fit_ok"])


class TheoryUnitTests(unittest.TestCase):
    def test_whistler_marginal_frequency_is_kennel_petschek(self):
        for a_e in (1.5, 2.0, 3.0):
            self.assertAlmostEqual(_whistler_marginal_frequency(a_e), 1 - 1 / a_e, delta=5e-3)

    def test_theory_is_converted_to_electron_units(self):
        theory = {"kdi": np.array([14.142135623730951]), "omega_r": np.array([100.0]),
                  "gamma": np.array([10.0]), "polarization": "minus"}
        out = to_electron_units(theory, 200.0)
        self.assertAlmostEqual(float(out["kdi"][0]), 1.0)
        self.assertAlmostEqual(float(out["omega_r"][0]), 0.5)
        self.assertAlmostEqual(float(out["gamma"][0]), 0.05)


if __name__ == "__main__":
    unittest.main()


def test_heat_flux_map_scale_ignores_a_lone_outlier_and_hatches_noise(tmp_path, monkeypatch):
    # For kappa = 3 the untruncated third moment has infinite variance: one
    # block with a few fast particles used to set the whole colour scale.
    import plot_style
    figures = []
    monkeypatch.setattr(plot_style, "save", lambda fig, *args, **kwargs: figures.append(fig))
    rng = np.random.default_rng(0)
    blocks = [{"jz": jz, "jy": jy, "q_par_over_q0": rng.normal(0.0, 0.01), "q_par_null_se": 0.01}
              for jz in range(8) for jy in range(8)]
    blocks[5]["q_par_over_q0"] = 0.5
    hf.plot_block_map(blocks, (0, 0, 0), (1, 80, 80), 8, "ion", 0, tmp_path, s_max=6.0)
    axis = figures[0].axes[0]
    assert axis.images[0].norm.vmax < 0.1
    hatched = [patch for patch in axis.patches if patch.get_hatch()]
    assert 55 <= len(hatched) <= 63
