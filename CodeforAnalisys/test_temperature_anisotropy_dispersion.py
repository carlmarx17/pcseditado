#!/usr/bin/env python3
"""Regression tests for temperature-anisotropy dispersion helpers."""

import unittest

import numpy as np

from temperature_anisotropy_dispersion import compute_folded_kspace_density


class TemperatureAnisotropyDispersionTests(unittest.TestCase):
    def test_folded_kspace_density_recovers_scalar_mode(self):
        n0, n1 = 64, 32
        d0, d1 = 0.5, 0.25
        mode_parallel = 4
        mode_perp = 3
        expected_k_parallel = 2.0 * np.pi * mode_parallel / (n0 * d0)
        expected_k_perp = 2.0 * np.pi * mode_perp / (n1 * d1)

        z = np.arange(n0, dtype=float) * d0
        y = np.arange(n1, dtype=float) * d1
        field = np.sin(expected_k_parallel * z[:, None] + expected_k_perp * y[None, :])

        result = compute_folded_kspace_density(
            field, spacing=(d0, d1), axes=("z", "y"), parallel_axis="z", bins=80
        )
        index = np.unravel_index(int(np.argmax(result["density"])), result["density"].shape)
        k_parallel = 0.5 * (
            result["kpar_edges"][index[0]] + result["kpar_edges"][index[0] + 1]
        )
        k_perp = 0.5 * (
            result["kperp_edges"][index[1]] + result["kperp_edges"][index[1] + 1]
        )

        self.assertAlmostEqual(k_parallel, expected_k_parallel, delta=0.12)
        self.assertAlmostEqual(k_perp, expected_k_perp, delta=0.25)


if __name__ == "__main__":
    unittest.main()
