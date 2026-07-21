#!/usr/bin/env python3
"""Regression tests for gamma(k_parallel,k_perp) growth-rate maps."""

import unittest

import numpy as np

from growth_rate_map import compute_growth_rate_map


class GrowthRateMapTests(unittest.TestCase):
    def test_recovers_oblique_exponential_growth(self):
        nt, nz, ny = 36, 64, 64
        dz = dy = 1.0
        times = np.linspace(0.0, 10.0, nt)
        gamma = 0.17
        mode_z = 3
        mode_y = 4
        expected_kpar = 2.0 * np.pi * mode_z / (nz * dz)
        expected_kperp = 2.0 * np.pi * mode_y / (ny * dy)

        z = np.arange(nz)[:, None] * dz
        y = np.arange(ny)[None, :] * dy
        phase = expected_kpar * z + expected_kperp * y
        amplitude = np.exp(gamma * times)[:, None, None]
        wave = amplitude * np.cos(phase)[None, :, :]

        result = compute_growth_rate_map(
            wave[None, ...],
            times,
            spacing=(dz, dy),
            axes=("z", "y"),
            parallel_axis="z",
            kpar_max=1.0,
            kperp_max=1.0,
            min_rvalue=0.0,
            fit_frac=(0.0, 1.0),
        )

        candidate = np.where(np.isfinite(result["final_power"]), result["final_power"], -np.inf)
        peak_i, peak_j = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        self.assertAlmostEqual(result["kpar"][peak_i], expected_kpar, places=6)
        self.assertAlmostEqual(result["kperp"][peak_j], expected_kperp, places=6)
        self.assertAlmostEqual(result["gamma"][peak_i, peak_j], gamma, delta=0.02)
        self.assertGreater(result["rvalue"][peak_i, peak_j], 0.99)


if __name__ == "__main__":
    unittest.main()
