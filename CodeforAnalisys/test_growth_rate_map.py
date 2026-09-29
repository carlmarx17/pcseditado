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
            # A single real-valued plane wave is inherently Hermitian-symmetric:
            # equal power sits at (+kpar,+kperp) and (-kpar,-kperp). Fold k_parallel
            # here so the test has one unambiguous peak; the default (unfolded)
            # behavior is exercised separately below.
            fold_negative_k=True,
        )

        candidate = np.where(np.isfinite(result["final_power"]), result["final_power"], -np.inf)
        peak_i, peak_j = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        self.assertAlmostEqual(result["kpar"][peak_i], expected_kpar, places=6)
        self.assertAlmostEqual(result["kperp"][peak_j], expected_kperp, places=6)
        self.assertAlmostEqual(result["gamma"][peak_i, peak_j], gamma, delta=0.02)
        self.assertGreater(result["rvalue"][peak_i, peak_j], 0.99)

    def test_default_keeps_kparallel_signed(self):
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

        self.assertFalse(result["fold_negative_k"])
        self.assertTrue(np.any(result["kpar"] < 0))
        # A real plane wave puts equal power at (+kpar,+kperp) and
        # (-kpar,-kperp); with k_parallel left signed, both should recover
        # the same growth rate at the same |k_perp|.
        pos_i = int(np.argmin(np.abs(result["kpar"] - expected_kpar)))
        neg_i = int(np.argmin(np.abs(result["kpar"] + expected_kpar)))
        j = int(np.argmin(np.abs(result["kperp"] - expected_kperp)))
        self.assertAlmostEqual(result["gamma"][pos_i, j], gamma, delta=0.02)
        self.assertAlmostEqual(result["gamma"][neg_i, j], gamma, delta=0.02)

    def test_default_window_is_the_dominant_mode_linear_phase(self):
        # A mode that grows from the noise, saturates, and is then followed by
        # a long saturated phase: the old default (10-60 % of the run) fitted
        # across the saturation and returned gamma well below the truth.
        nt, nz, ny = 200, 32, 32
        rng = np.random.default_rng(0)
        times = np.linspace(0.0, 160.0, nt)
        gamma = 0.11
        g = 1e-5 * np.exp(gamma * times)
        amplitude = g / (1.0 + g / 0.02)
        kpar = 2.0 * np.pi * 2 / nz
        z = np.arange(nz)[:, None]
        wave = amplitude[:, None, None] * np.cos(kpar * z)[None, :, :] * np.ones((1, 1, ny))
        noise = 2e-5 * rng.standard_normal((nt, nz, ny))

        result = compute_growth_rate_map(
            (wave + noise)[None, ...], times, spacing=(1.0, 1.0), axes=("z", "y"),
            parallel_axis="z", kpar_max=1.0, kperp_max=1.0, fold_negative_k=True,
        )

        self.assertEqual(result["window_source"], "dominant-mode")
        self.assertLess(result["fit_window"][1], 96.0)
        i = int(np.argmin(np.abs(result["kpar"] - kpar)))
        self.assertLess(abs(result["gamma"][i, 0] / gamma - 1.0), 0.10)
        # The fixed fraction still exists as an explicit choice, and is biased low.
        old = compute_growth_rate_map(
            (wave + noise)[None, ...], times, spacing=(1.0, 1.0), axes=("z", "y"),
            parallel_axis="z", kpar_max=1.0, kperp_max=1.0, fold_negative_k=True,
            fit_frac=(0.1, 0.6),
        )
        self.assertLess(old["gamma"][i, 0], 0.8 * gamma)


if __name__ == "__main__":
    unittest.main()
