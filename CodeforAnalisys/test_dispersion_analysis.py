#!/usr/bin/env python3
"""Numerical regression test for frequency--phase-velocity mode recovery."""

import unittest

import numpy as np

from dispersion_analysis import compute_phase_velocity_density, extract_ridges


class DispersionAnalysisTests(unittest.TestCase):
    def test_recovers_plane_wave_phase_velocity(self):
        nt, nz, ny = 32, 64, 16
        dt = 0.25
        dz = 0.5
        mode_t = 3
        mode_z = 4
        omega = 2.0 * np.pi * mode_t / (nt * dt)
        k_parallel = 2.0 * np.pi * mode_z / (nz * dz)
        expected_velocity = omega / k_parallel

        time = np.arange(nt) * dt
        z = np.arange(nz) * dz
        phase = (
            k_parallel * z[None, :, None]
            - omega * time[:, None, None]
        )
        wave = np.sin(phase) * np.ones((1, 1, ny))
        result = compute_phase_velocity_density(
            wave[None, ...],
            time,
            spacing=(dz, 1.0),
            axes=("z", "y"),
            velocity_max=8.0,
            velocity_bins=320,
            frequency_bins=160,
        )
        ridges = extract_ridges(result, ridge_count=1)
        # With ridge_axis="k" (the default) the CSV holds one measured omega per
        # resolved wavenumber, so the wave is the strongest row rather than the
        # row nearest a frequency we would otherwise have to know in advance.
        strongest = max(ridges, key=lambda row: row["spectral_power"])
        self.assertAlmostEqual(
            strongest["k_parallel_d_i"], k_parallel, delta=1e-9
        )
        self.assertAlmostEqual(
            strongest["phase_velocity_over_va"], expected_velocity, delta=0.05
        )

    def test_legacy_omega_axis_still_recovers_the_phase_velocity(self):
        """The pre-existing per-omega walk is kept working behind a flag."""
        nt, nz, ny = 32, 64, 16
        dt, dz = 0.25, 0.5
        omega = 2.0 * np.pi * 3 / (nt * dt)
        k_parallel = 2.0 * np.pi * 4 / (nz * dz)

        time = np.arange(nt) * dt
        z = np.arange(nz) * dz
        wave = np.sin(
            k_parallel * z[None, :, None] - omega * time[:, None, None]
        ) * np.ones((1, 1, ny))
        result = compute_phase_velocity_density(
            wave[None, ...], time, spacing=(dz, 1.0), axes=("z", "y"),
            velocity_max=8.0, velocity_bins=320, frequency_bins=160,
        )
        ridges = extract_ridges(result, ridge_count=1, ridge_axis="omega")
        closest = min(ridges, key=lambda row: abs(row["omega_over_omega_ci"] - omega))
        self.assertAlmostEqual(
            closest["phase_velocity_over_va"], omega / k_parallel, delta=0.05
        )


if __name__ == "__main__":
    unittest.main()
