"""kappa_frame_comparison: what pooling over a structured box does to kappa."""

import unittest

import numpy as np

from kappa_eff import DEFAULT_S_MAX, signed_inverse_kappa, truncated_K_maxwellian
from kappa_frame_comparison import inflation_factor, local_K


class InflationFactorTests(unittest.TestCase):
    def test_a_uniform_box_inflates_nothing(self):
        # Every cell the same width: pooling cannot invent a tail.
        for n in (4, 64, 500):
            self.assertAlmostEqual(
                inflation_factor(np.full(n, 0.016), np.full(n, 0.032),
                                 np.full(n, 30000.0)), 1.0, places=12)

    def test_the_factor_does_not_depend_on_the_overall_temperature(self):
        # F is built from ratios to the box mean, so a global rescaling --
        # heating, or a change of units -- must leave it alone.
        rng = np.random.default_rng(0)
        t_par = rng.lognormal(0.0, 0.3, 256)
        t_perp = rng.lognormal(0.0, 0.3, 256)
        counts = rng.integers(1000, 5000, 256).astype(float)
        base = inflation_factor(t_par, t_perp, counts)
        self.assertAlmostEqual(inflation_factor(7.5 * t_par, 7.5 * t_perp, counts),
                               base, places=12)

    def test_inflation_grows_with_the_spread_and_only_then_makes_a_tail(self):
        """The quantitative point: 5 % structure cannot manufacture kappa.

        rms(|B|-B0)/B0 at saturation in the v6 mirror runs is ~0.05. Even
        taking the temperature spread to be as large as that, the apparent
        1/kappa is ~0.002, two orders below the measured 0.11-0.33. Getting a
        kappa out of structure alone needs a spread of order the mean.
        """
        maxwellian = truncated_K_maxwellian(DEFAULT_S_MAX)
        previous = 0.0
        apparent = {}
        for spread in (0.05, 0.2, 0.5, 0.8):
            rng = np.random.default_rng(1)
            widths = np.exp(rng.normal(0.0, np.sqrt(np.log(1 + spread ** 2)), 4096))
            widths /= widths.mean()
            factor = inflation_factor(widths, widths, np.ones(4096))
            self.assertGreater(factor, previous)
            previous = factor
            apparent[spread] = signed_inverse_kappa(factor * maxwellian, DEFAULT_S_MAX)
        self.assertLess(apparent[0.05], 0.01)
        self.assertGreater(apparent[0.8], 0.2)

    def test_local_K_round_trips_through_the_estimator_inversion(self):
        for inv_kappa in (0.3333, 0.2, 0.05, 0.0, -0.004):
            recovered = signed_inverse_kappa(local_K(inv_kappa), DEFAULT_S_MAX)
            self.assertAlmostEqual(recovered, inv_kappa, places=6)

    def test_degenerate_input_is_not_an_answer(self):
        self.assertTrue(np.isnan(inflation_factor([0.016], [0.03], [100.0])))
        self.assertTrue(np.isnan(inflation_factor([np.nan, 1.0], [1.0, 1.0],
                                                  [1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
