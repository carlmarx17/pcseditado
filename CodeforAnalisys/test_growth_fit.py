"""growth_fit.fit_exponential_growth: the single linear-phase fit of the pipeline."""

import unittest

import numpy as np

from growth_fit import fit_exponential_growth

GAMMA = 0.25


def logistic_series(n: int, noise: float, seed: int, floor: float = 4e-5):
    """Noise floor -> exponential growth -> quasi-linear (logistic) saturation."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 40.0, n)
    g = 3e-4 * np.exp(GAMMA * t)
    a = g / (1.0 + g / 0.08)
    a = np.sqrt(a ** 2 + floor ** 2) * np.exp(noise * rng.standard_normal(n))
    return t, a


class GrowthFitTests(unittest.TestCase):
    def test_recovers_gamma_on_a_saturating_series(self):
        for n, noise in [(41, 0.0), (41, 0.05), (200, 0.1), (2400, 0.1)]:
            for seed in (1, 2):
                t, a = logistic_series(n, noise, seed)
                fit = fit_exponential_growth(t, a)
                with self.subTest(n=n, noise=noise, seed=seed):
                    self.assertTrue(fit["fit_ok"], fit["fit_reject_reason"])
                    # The error bar (+ the documented <= 5 % low bias of a
                    # smooth saturation) always covers the truth ...
                    self.assertLess(abs(fit["gamma"] - GAMMA), 3.0 * fit["gamma_err"] + 0.05 * GAMMA)
                    # ... and a fit that claims 10 % precision delivers it.
                    if fit["gamma_err"] < 0.1 * fit["gamma"]:
                        self.assertLess(abs(fit["gamma"] / GAMMA - 1.0), 0.10)

    def test_window_excludes_the_saturation_roll_over(self):
        t, a = logistic_series(81, 0.0, 0)
        fit = fit_exponential_growth(t, a)
        # The local rate falls to 0.8 gamma (a/a_sat = 0.2) near t ~ 16.
        self.assertLess(fit["linear_phase_end"], 20.0)
        self.assertGreater(fit["linear_phase_end"], fit["linear_phase_start"])

    def test_explicit_window_is_respected(self):
        t, a = logistic_series(81, 0.0, 0)
        fit = fit_exponential_growth(t, a, t_start=5.0, t_end=12.0)
        self.assertEqual(fit["window_source"], "explicit")
        self.assertGreaterEqual(fit["linear_phase_start"], 5.0)
        self.assertLessEqual(fit["linear_phase_end"], 12.0)
        self.assertAlmostEqual(fit["gamma"], GAMMA, delta=0.02)

    def test_decay_is_never_a_valid_growth_rate(self):
        t = np.linspace(0, 10, 30)
        fit = fit_exponential_growth(t, np.exp(-0.3 * t))
        self.assertFalse(fit["fit_ok"])
        self.assertLess(fit["gamma"], 0.0)

    def test_power_versus_amplitude(self):
        # gamma is the growth rate of the amplitude: fitting sqrt(P) returns it.
        t = np.linspace(0, 10, 40)
        power = 1e-8 * np.exp(2 * 0.4 * t)
        fit = fit_exponential_growth(t, np.sqrt(power), t_start=0.0, t_end=10.0)
        self.assertAlmostEqual(fit["gamma"], 0.4, places=6)

    def test_too_short_series(self):
        fit = fit_exponential_growth([0, 1, 2], [1, 2, 4])
        self.assertFalse(fit["fit_ok"])
        self.assertTrue(np.isnan(fit["gamma"]))


if __name__ == "__main__":
    unittest.main()
