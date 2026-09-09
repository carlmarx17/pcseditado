"""Prevent mixing polarization branches or citing rejected growth fits."""

import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np

from compare_physical_cases import load_case, validate_comparison
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


if __name__ == "__main__":
    unittest.main()
