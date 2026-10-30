"""Prevent mixing polarization branches or citing rejected growth fits."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from compare_physical_cases import add_energy_changes, load_case, validate_comparison
from polarization_dispersion import interp_theory, load_theory


class TheoryComparisonTests(unittest.TestCase):
    def test_two_channels_and_a_failed_root_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theory.csv"
            path.write_text("kdi,omega_r_over_Omegai,gamma_over_Omegai,polarization,converged,residual\n"
                            "1,0.1,0.2,plus,True,1e-10\n2,0.2,0.3,plus,False,0.1\n3,0.3,0.4,plus,True,1e-10\n"
                            "1,5,6,minus,True,1e-10\n2,6,7,minus,True,1e-10\n")
            with self.assertRaises(ValueError):
                load_theory(str(path))
            plus = load_theory(str(path), "plus")
            minus = load_theory(str(path), "minus")
            self.assertEqual(interp_theory(minus, 1.0), (5.0, 6.0))
            self.assertEqual(interp_theory(plus, 1.0), (0.1, 0.2))
            self.assertTrue(np.isnan(interp_theory(plus, 1.5)[1]))
            self.assertTrue(np.isnan(interp_theory(plus, 2.5)[1]))

    def test_duplicate_k_and_unlabelled_theory_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "theory.csv"
            path.write_text("kdi,omega_r_over_Omegai,gamma_over_Omegai\n1,2,3\n1,4,5\n")
            with self.assertRaises(ValueError):
                load_theory(str(path), "plus")
            with self.assertRaises(ValueError):
                load_theory(str(path), "plus", allow_unverified=True)

    def test_rejected_or_missing_fit_flag_is_not_a_comparison_gamma(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for flag in ["False", "0", "", "True", "1"]:
                (path / "growth_rate_summary.csv").write_text(f"gamma,fit_ok\n0.12,{flag}\n")
                result = load_case("case", path)
                self.assertEqual(np.isfinite(result["gamma"]), flag in ["True", "1"])

    def test_controlled_distribution_and_convergence_comparisons(self):
        parameters = dict(mass_ratio=200, n0=1, B0=0.08, beta_i_parallel=5, A_i=2,
                          beta_e_parallel=1, A_e=1, domain_di=20, grid=[576,576],
                          dt_code_from_profile=0.33, nicell_from_profile=1000, kappa=None)
        cases = [dict(name="M", manifest={"physics": parameters}),
                 dict(name="K", manifest={"physics": parameters | {"kappa": 3}})]
        self.assertEqual(validate_comparison(cases), [])
        self.assertTrue(validate_comparison(cases, "convergence"))
        cases[1]["manifest"]["physics"]["beta_i_parallel"] = 10
        self.assertTrue(validate_comparison(cases))


class EnergyBudgetComparisonTests(unittest.TestCase):
    """Cases compare by their energy changes, over their own initial K."""

    @staticmethod
    def _rows(with_fields: bool = True):
        rows = []
        for step in range(3):
            row = {"omega_ci_t": float(step), "K_i": 2.0 - 0.1 * step, "K_e": 1.0,
                   "E_B": 0.01 * step, "E_B_total": np.nan, "E_E": np.nan,
                   "E_field": np.nan, "E_sum": np.nan}
            if with_fields:
                row.update({"E_B_total": 5.0 + 0.1 * step, "E_E": 0.002 * step,
                            "E_field": 5.0 + 0.102 * step, "E_sum": 8.0 + 0.002 * step})
            rows.append(row)
        return rows

    def test_changes_are_normalised_by_the_initial_kinetic_energy(self):
        rows = self._rows()
        budget = add_energy_changes(rows)
        self.assertEqual(budget["normalisation"], "K_i(t0) + K_e(t0)")
        self.assertAlmostEqual(budget["K0"], 3.0)
        self.assertAlmostEqual(rows[0]["dK_i_over_K0"], 0.0)
        self.assertAlmostEqual(rows[-1]["dK_i_over_K0"], -0.2 / 3.0)
        # The total magnetic energy, not the fluctuation: they are different
        # columns and the budget uses the one carrying B0 <dB_par>.
        self.assertAlmostEqual(rows[-1]["dE_B_over_K0"], 0.2 / 3.0)
        self.assertAlmostEqual(rows[-1]["dE_dB_over_K0"], 0.02 / 3.0)
        self.assertAlmostEqual(rows[-1]["dE_E_over_K0"], 0.004 / 3.0)
        self.assertTrue(budget["has_electric_field"])
        self.assertTrue(budget["has_total_magnetic_energy"])

    def test_a_table_without_the_field_energies_is_flagged_not_zeroed(self):
        rows = self._rows(with_fields=False)
        budget = add_energy_changes(rows)
        self.assertFalse(budget["has_electric_field"])
        self.assertFalse(budget["has_total_magnetic_energy"])
        # Absent is NaN: a zero would draw a flat line that looks measured.
        self.assertTrue(np.isnan(rows[-1]["dE_B_over_K0"]))
        self.assertTrue(np.isnan(rows[-1]["dE_E_over_K0"]))
        self.assertTrue(np.isfinite(rows[-1]["dK_i_over_K0"]))

    def test_a_case_without_kinetic_energy_gets_no_normalisation(self):
        rows = [{"omega_ci_t": 0.0, "K_i": np.nan, "K_e": np.nan, "E_B": 0.01,
                 "E_B_total": np.nan, "E_E": np.nan, "E_field": np.nan, "E_sum": np.nan}]
        budget = add_energy_changes(rows)
        self.assertEqual(budget["normalisation"], "none")
        self.assertTrue(np.isnan(rows[0]["dE_sum_over_K0"]))


if __name__ == "__main__":
    unittest.main()
