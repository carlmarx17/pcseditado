"""Physical invariants for normalization, pressure and partial diagnostics."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import physical_diagnostics as diagnostics
import psc_units as units
from plasma_physics import field_aligned_pressures, mirror_threshold


class PhysicalConsistencyTests(unittest.TestCase):
    def test_alfven_speed_matches_ion_scales_and_beta(self):
        self.assertAlmostEqual(units.VA, units.OMEGA_CI * units.DI)
        self.assertAlmostEqual(units.VA_OVER_C, units.VA)
        self.assertAlmostEqual(
            units.TI_PAR / (units.M_ION * units.VA**2), units.BETA_I_PAR / 2
        )
        self.assertEqual(units.B0, units._active["vA_over_c"])

    def test_mirror_reference_satisfies_perpendicular_beta_condition(self):
        beta = np.array([0.1, 1.0, 5.0, 100.0])
        a = mirror_threshold(beta)
        np.testing.assert_allclose(beta * a * (a - 1), 1.0)
        self.assertTrue(np.all(np.isnan(mirror_threshold([0.0, -1.0]))))

    def test_pressure_projection_is_rotation_invariant(self):
        b = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        pressure = 5.0 * np.eye(3) + (2.0 - 5.0) * np.outer(b, b)
        par, perp, b2 = field_aligned_pressures(
            pressure[0, 0], pressure[1, 1], pressure[2, 2],
            pressure[0, 1], pressure[1, 2], pressure[2, 0], *b
        )
        np.testing.assert_allclose([par, perp, b2], [2.0, 5.0, 1.0])
        par, perp, _ = field_aligned_pressures(1, 1, 1, 0, 0, 0, 0, 0, 0)
        self.assertTrue(np.isnan(par) and np.isnan(perp))

    def test_integrated_moments_remove_drift_and_use_local_field(self):
        b = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        pressure = 5.0 * np.eye(3) - 3.0 * np.outer(b, b)
        velocity = np.array([0.3, -0.2, 0.1])
        density = 2.0
        raw = pressure + density * units.M_ION * np.outer(velocity, velocity)
        full = lambda x: np.full((2, 2), x)
        mom = {"rho_i": full(density)}
        for i, axis in enumerate("xyz"):
            mom[f"p{axis}_i"] = full(density * units.M_ION * velocity[i])
            mom[f"t{axis}{axis}_i"] = full(raw[i, i])
        for name, i, j in [("xy", 0, 1), ("yz", 1, 2), ("zx", 2, 0)]:
            mom[f"t{name}_i"] = full(raw[i, j])
        fields = {f"B{axis}": full(b[i]) for i, axis in enumerate("xyz")}
        with patch.object(diagnostics, "load_moments", return_value=mom), \
                patch.object(diagnostics, "load_fields", return_value=fields):
            result = diagnostics.moment_thermal_maps("moments", "fields")
        np.testing.assert_allclose(result["T_parallel"], 1.0)
        np.testing.assert_allclose(result["T_perp"], 2.5)
        np.testing.assert_allclose(result["beta_parallel"], 4.0)

    def test_symmetric_distribution_has_zero_heat_flux_in_all_directions(self):
        velocities = np.array([[1, 2, 3], [-1, -2, -3], [2, -1, 4], [-2, 1, -4]])
        snap = diagnostics.ParticleSnapshot(
            0, 0.0, np.ones(4), np.ones(4), *velocities.T, np.ones(4)
        )
        result = diagnostics.particle_heat_flux(snap)
        for value in result.values():
            self.assertAlmostEqual(value, 0.0)

    def test_energy_proxy_retains_electron_trend_without_claiming_conservation(self):
        particles, fields = [], []
        for step in range(4):
            particles.append({
                "step": step, "E_kin_bulk": 0.0, "E_kin_thermal": 1.5,
                "T_parallel_i": 1.0, "T_perp_i": 1.0,
                "T_parallel_e": 1.0 + step, "T_perp_e": 1.0 + step, "A_e": 1.0,
            })
            fields.append({"step": step, "magnetic_energy_fluct": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            runner = diagnostics.PhysicalDiagnostics.__new__(diagnostics.PhysicalDiagnostics)
            runner.outdir = Path(tmp)
            with patch.object(runner, "plot_energy"):
                runner.run_energy_summary(particles, fields)
            with (Path(tmp) / "energy_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(float(rows[-1]["E_proxy"]), float(rows[0]["E_proxy"]))
            self.assertEqual(rows[-1]["is_conservation_diagnostic"], "False")
            for key in ["E_total", "E_total_corrected", "energy_error", "energy_error_corrected"]:
                self.assertNotIn(key, rows[-1])
            self.assertTrue((Path(tmp) / "electron_energy_trend.csv").exists())
            self.assertFalse((Path(tmp) / "numerical_heating.csv").exists())


if __name__ == "__main__":
    unittest.main()
