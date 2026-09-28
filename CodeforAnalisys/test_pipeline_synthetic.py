"""End-to-end check of the analysis scripts on a synthetic PSC run.

synthetic_run.py writes a run in PSC's on-disk layout with prescribed
physics (growth rate, divergence-free Yee field, pressure-balanced moments,
bi-Maxwellian particles). The scripts run as subprocesses, exactly as the
Makefile calls them, and their outputs are compared with the input.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = "mirror_bimaxwellian_moderate"


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.run_dir = root / "run"
        cls.out = root / "out"
        cls.env = {**os.environ, "PSC_PROFILE": CASE,
                   "PSC_ANALYSIS_DATA_DIR": str(cls.run_dir),
                   "MPLCONFIGDIR": str(root / "mpl"), "PSC_FIG_THEME": "paper"}
        cls.env.pop("PSC_ANALYSIS_CONFIG", None)
        subprocess.run([sys.executable, "synthetic_run.py", str(cls.run_dir), "--case", CASE,
                        "--ngrid", "32", "--snapshots", "31", "--ppc", "80"],
                       cwd=HERE, env={**cls.env, "PSC_ANALYSIS_DATA_DIR": ""}, check=True,
                       capture_output=True)
        for script, extra in (
            ("physical_diagnostics.py", ["--max-map-steps", "1", "--jobs", "1",
                                         "--vdf-cadence-omegaci", "1000"]),
            ("field_residuals.py", []),
            ("structures_analysis.py", []),
            ("energy_exchange.py", []),
            ("estimator_consistency.py", []),
            ("heat_flux_analysis.py", ["--subsamples", "4", "--macrocells", "2"]),
        ):
            result = subprocess.run(
                [sys.executable, script, "--data-dir", str(cls.run_dir), "--outdir", str(cls.out), *extra],
                cwd=HERE, env=cls.env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{script} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_growth_rate_of_the_vector_fluctuation(self):
        total = _csv(self.out / "growth_rate_summary.csv")[0]
        self.assertEqual(total["series"], "total")
        self.assertEqual(total["fit_ok"], "1")
        gamma, err = float(total["gamma"]), float(total["gamma_err"])
        self.assertLess(abs(gamma / 0.25 - 1.0), 0.10)
        self.assertLess(abs(gamma - 0.25), 3 * err + 0.0125)

    def test_div_b_and_log_residuals(self):
        summary = json.loads((self.out / "field_residuals_summary.json").read_text())
        self.assertLess(summary["divB_max_dx_over_B0"], 1e-5)
        self.assertEqual(summary["gauss_checks"], 31)
        self.assertLess(summary["gauss_max_err"], 1e-4)

    def test_global_energy_is_conserved(self):
        summary = json.loads((self.out / "global_energy_summary.json").read_text())
        self.assertLess(summary["max_abs_relative_change"], 1e-9)

    def test_structures_are_pressure_balanced_and_anticorrelated(self):
        summary = json.loads((self.out / "structures_summary.json").read_text())
        self.assertLess(summary["final_corr_n_B"], -0.95)
        self.assertLess(summary["final_pressure_balance_ratio"], 0.2)

    def test_estimators_agree_for_a_uniform_bimaxwellian(self):
        rows = [r for r in _csv(self.out / "estimator_consistency.csv") if r["species"] == "ion"]
        for key in ("A_prt_global", "A_prt_local", "A_mom_window_ratio"):
            # ~2900 ions in the window: sampling sigma_A ~ A sqrt(4/3N) ~ 0.04.
            self.assertAlmostEqual(float(rows[0][key]), 2.0, delta=0.15, msg=key)

    def test_bimaxwellian_heat_flux_is_consistent_with_zero(self):
        for row in _csv(self.out / "heat_flux_table.csv"):
            q, err = float(row["q_par_over_q0_smax6"]), float(row["q_par_over_q0_err"])
            self.assertLess(abs(q), max(4 * err, 0.05), msg=row["species"])
            # <|q|> of a heat-flux-free VDF is the sampling floor, not a flux.
            # Only 4 blocks here, so the check is an order of magnitude; the
            # floor formula itself is tested on a large sample in
            # test_new_diagnostics.HeatFluxTests.
            ratio = float(row["abs_q_par_over_q0"]) / float(row["abs_q_par_noise_floor"])
            self.assertTrue(0.2 < ratio < 2.5, msg=f"{row['species']}: {ratio}")

    def test_particle_temperatures_follow_the_profile(self):
        first = _csv(self.out / "anisotropy_table.csv")[0]
        self.assertAlmostEqual(float(first["A_i"]), 2.0, delta=0.1)
        self.assertAlmostEqual(float(first["beta_parallel_i"]), 5.0, delta=0.4)


if __name__ == "__main__":
    unittest.main()
