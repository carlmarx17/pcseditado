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

    def test_logistic_fit_is_unbiased_where_the_windowed_fit_is_low(self):
        # The band/slope window ends inside the roll-over, so it reads a
        # smooth saturation ~4 % low (documented in the module docstring).
        # Fitting the saturating curve itself has no window to misplace.
        windowed, logistic, covered = [], [], 0
        for n, noise in [(41, 0.0), (41, 0.05), (200, 0.1), (2400, 0.1)]:
            for seed in (1, 2, 3, 4):
                t, a = logistic_series(n, noise, seed)
                fit = fit_exponential_growth(t, a)
                with self.subTest(n=n, noise=noise, seed=seed):
                    self.assertTrue(fit["logistic_fit_ok"], fit["logistic_reject_reason"])
                    # a_sat and the inflection of the generated series.
                    self.assertAlmostEqual(fit["amplitude_saturation"], 0.08, delta=0.004)
                    self.assertAlmostEqual(fit["t_saturation"], 22.4, delta=0.5)
                    self.assertTrue(np.isfinite(fit["gamma_logistic_stderr"]))
                windowed.append(fit["gamma"])
                logistic.append(fit["gamma_logistic"])
                covered += abs(fit["gamma_logistic"] - GAMMA) <= 2 * fit["gamma_logistic_stderr"]
        windowed, logistic = np.array(windowed), np.array(logistic)
        self.assertLess(abs(logistic.mean() / GAMMA - 1.0), 0.01)
        self.assertLess(logistic.std(), windowed.std())
        self.assertLess(abs(logistic.mean() - GAMMA), abs(windowed.mean() - GAMMA))
        self.assertGreaterEqual(covered, int(0.8 * len(logistic)))

    def test_logistic_fit_survives_the_quiet_start_and_a_flat_noise_floor(self):
        # Raw PIC series: quiet-start build-up, noise plateau, growth,
        # saturation. Without the floor as a parameter the fit diverged here.
        gamma = 0.11
        for n in (2400, 121):
            rng = np.random.default_rng(3)
            t = np.linspace(0.066, 158.0, n)
            build_up = 1.0 - 0.96 * np.exp(-t / 1.5)
            g = 8.8e-6 * np.exp(gamma * t)
            mode = g / (1.0 + g / 0.016)
            floor = 1.6e-5 * (1.0 + 0.004 * t)
            a = (build_up * np.sqrt(floor ** 2 + mode ** 2)
                 * np.exp(0.05 * rng.standard_normal(n)))
            fit = fit_exponential_growth(t, a)
            with self.subTest(n=n):
                self.assertTrue(fit["logistic_fit_ok"], fit["logistic_reject_reason"])
                self.assertLess(abs(fit["gamma_logistic"] / gamma - 1.0), 0.10)
                # The inflection is well inside the run, after the window fit.
                self.assertGreater(fit["t_saturation"], fit["linear_phase_end"])
                self.assertLess(fit["t_saturation"], t[-1])

    def test_quiet_start_transient_is_not_the_linear_phase(self):
        # Shape of the dominant mode in the v5 mirror runs (E(k,t) of the
        # k d_i = 0.34 shell): from the quiet start the whole spectrum,
        # the future unstable mode included, builds up x25 in ~2 Omega_ci^-1,
        # sits on a slowly heating noise plateau, then the mode grows ~6
        # e-folds at gamma = 0.11 and saturates. The steepest local slope is
        # the build-up: the fit used to return gamma ~ 0.6 on t < 2.5 and
        # flag it valid.
        gamma = 0.11
        for n in (2400, 121):
            for seed in (1, 2):
                rng = np.random.default_rng(seed)
                t = np.linspace(0.066, 158.0, n)
                build_up = 1.0 - 0.96 * np.exp(-t / 1.5)
                g = 8.8e-6 * np.exp(gamma * t)
                mode = g / (1.0 + g / 0.016)
                floor = 1.6e-5 * (1.0 + 0.004 * t)
                a = (build_up * np.sqrt(floor ** 2 + mode ** 2)
                     * np.exp(0.05 * rng.standard_normal(n)))
                fit = fit_exponential_growth(t, a)
                with self.subTest(n=n, seed=seed):
                    self.assertTrue(fit["fit_ok"], fit["fit_reject_reason"])
                    self.assertGreater(fit["linear_phase_start"], 10.0)
                    self.assertLess(abs(fit["gamma"] / gamma - 1.0), 0.10)

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
