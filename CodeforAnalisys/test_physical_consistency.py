"""Physical invariants for normalization, pressure and partial diagnostics."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import physical_diagnostics as diagnostics
import psc_units as units
from data_reader import PICDataReader
from plasma_physics import (
    central_pressure_tensor,
    field_aligned_pressures,
    mirror_threshold,
    moment_fields_by_name,
)

HERE = Path(__file__).resolve().parent


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

    def test_field_energy_densities_carry_the_cross_term_and_the_electric_field(self):
        """0.5<|B|^2> - 0.5<|dB|^2> is B0^2/2 + B0<dB_par>, the first-order term."""
        b0 = 0.1
        db = np.array([[0.02, -0.02], [0.01, 0.03]])      # <dB_par> = 0.01 != 0
        fld = {"Bx": np.zeros_like(db), "By": np.zeros_like(db), "Bz": b0 + db,
               "Ex": np.full_like(db, 0.3), "Ey": np.zeros_like(db), "Ez": np.zeros_like(db)}
        out = diagnostics._field_energy_densities(fld, b0)
        self.assertTrue(out["has_electric_field"])
        self.assertAlmostEqual(out["electric_energy"], 0.5 * 0.3**2)
        self.assertAlmostEqual(out["magnetic_energy_total"], 0.5 * float(np.mean((b0 + db) ** 2)))
        self.assertAlmostEqual(out["magnetic_energy_cross"], b0 * float(np.mean(db)))
        # What the fluctuation energy drops: 0.5|<B>|^2 = B0^2/2 + B0<dB_par>
        # + <dB_par>^2/2, whose cross term is first order in the amplitude.
        mean_db = float(np.mean(db))
        self.assertAlmostEqual(out["magnetic_energy_mean_part"],
                               0.5 * b0**2 + b0 * mean_db + 0.5 * mean_db**2, places=12)
        self.assertGreater(abs(out["magnetic_energy_cross"]), 0.5 * mean_db**2)
        self.assertAlmostEqual(out["field_energy_total"],
                               out["magnetic_energy_total"] + out["electric_energy"])

        del fld["Ex"], fld["Ey"], fld["Ez"]
        without_e = diagnostics._field_energy_densities(fld, b0)
        self.assertFalse(without_e["has_electric_field"])
        # Missing is NaN, never a zero a budget would take for a measurement.
        self.assertTrue(np.isnan(without_e["electric_energy"]))
        self.assertTrue(np.isnan(without_e["field_energy_total"]))

    def test_energy_budget_counts_both_species_and_every_field_energy(self):
        particles, fields = [], []
        for step in range(4):
            particles.append({
                "step": step, "E_kin_bulk": 0.2, "E_kin_thermal": 1.5,
                "E_kin_bulk_i": 0.2, "E_kin_thermal_i": 1.5,
                "E_kin_bulk_e": 0.1, "E_kin_thermal_e": 0.4 + 0.01 * step,
                "T_parallel_i": 1.0, "T_perp_i": 1.0,
                "T_parallel_e": 1.0, "T_perp_e": 1.0, "A_e": 1.0,
            })
            fields.append({"step": step, "magnetic_energy_fluct": 0.1,
                           "magnetic_energy_total": 5.0 + 0.1 * step,
                           "electric_energy": 0.01 * step,
                           "field_energy_total": 5.0 + 0.11 * step})
        with tempfile.TemporaryDirectory() as tmp:
            runner = diagnostics.PhysicalDiagnostics.__new__(diagnostics.PhysicalDiagnostics)
            runner.outdir = Path(tmp)
            with patch.object(runner, "plot_energy"):
                runner.run_energy_summary(particles, fields)
            with (Path(tmp) / "energy_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
        last = rows[-1]
        self.assertAlmostEqual(float(last["K_i"]), 1.7)
        self.assertAlmostEqual(float(last["K_e"]), 0.53)
        # The sum is K_i + K_e + total magnetic + electric, so the electric
        # energy and the electron bulk flow move it; previously neither did.
        self.assertAlmostEqual(float(last["E_field"]), 5.33)
        self.assertAlmostEqual(float(last["E_proxy"]), 1.7 + 0.53 + 5.33)
        self.assertIn("magnetic total + electric", last["budget_scope"])
        # The temperature-based internal energy stays the 3 T_bar / 2 that
        # energy_audit inverts, and is a cross-check rather than a summand.
        self.assertAlmostEqual(float(last["E_internal_i"]), 1.5)
        self.assertAlmostEqual(float(last["thermal_estimator_gap_i"]), 0.0)


class MomentConventionTests(unittest.TestCase):
    """PSC writes the raw second moment, so every consumer has to centre it.

    ``t_ab = n m <u_a v_b>`` is the whole momentum-flux tensor in the box
    frame, not a temperature and not a thermal pressure. Measured on local
    runs (MOMENT_OUTPUT_VALIDATION.md): a density ramp at fixed T swings
    ``t_zz`` by 100 % while ``t_zz / n`` stays at T, and a sheared drift at
    fixed n and T lifts ``t_yy`` by ``n m u_y^2``. These tests pin the
    subtraction in the scripts that read those moments.
    """

    def _drifting_block(self, suffix="i", mass=None):
        """A uniform block: known P (P_par 2, P_perp 5), known drift, raw t_ab."""
        mass = units.M_ION if mass is None else mass
        b = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        pressure = 5.0 * np.eye(3) - 3.0 * np.outer(b, b)
        velocity = np.array([0.3, -0.2, 0.1])
        density = 2.0
        raw = pressure + density * mass * np.outer(velocity, velocity)
        full = lambda x: np.full((4, 4, 1), float(x))
        sign = 1.0 if suffix == "i" else -1.0
        mom = {f"rho_{suffix}/p0/3d": full(sign * density)}
        for i, axis in enumerate("xyz"):
            mom[f"p{axis}_{suffix}/p0/3d"] = full(density * mass * velocity[i])
            mom[f"t{axis}{axis}_{suffix}/p0/3d"] = full(raw[i, i])
        for name, i, j in (("xy", 0, 1), ("yz", 1, 2), ("zx", 2, 0)):
            mom[f"t{name}_{suffix}/p0/3d"] = full(raw[i, j])
        fields = {f"h{axis}_fc/p0/3d": full(b[i]) for i, axis in enumerate("xyz")}
        return mom, fields, raw, b

    def _uncentred_anisotropy(self, raw, b) -> float:
        """What the same block reads as if the drift term were left in."""
        par, perp, _ = field_aligned_pressures(
            raw[0, 0], raw[1, 1], raw[2, 2], raw[0, 1], raw[1, 2], raw[2, 0], *b)
        return float(perp / par)

    def _patched_reader(self, mom, fields):
        def read(path, group, names):
            return mom if "all_1st" in group else fields
        return patch.object(PICDataReader, "read_multiple_fields_3d", side_effect=read)

    def test_anisotropy_trajectory_removes_the_drift(self):
        from anisotropy_analysis import process_snapshot
        mom, fields, raw, b = self._drifting_block()
        with self._patched_reader(mom, fields):
            result = process_snapshot("moments.h5", "fields.h5", "ion")
        self.assertAlmostEqual(result["anisotropy_global"], 2.5, places=6)
        self.assertAlmostEqual(result["beta_global"], 4.0, places=6)
        # Only a test because the uncentred moment lands somewhere else.
        self.assertGreater(abs(self._uncentred_anisotropy(raw, b) - 2.5), 0.3)

    def test_temperature_quantity_maps_remove_the_drift(self):
        from temperature_anisotropy_dispersion import compute_temperature_quantity
        mom, fields, raw, b = self._drifting_block()
        with self._patched_reader(mom, fields):
            aniso = compute_temperature_quantity(
                "moments.h5", "fields.h5", "ion", "anisotropy")
            t_par = compute_temperature_quantity(
                "moments.h5", "fields.h5", "ion", "parallel-temperature")
        self.assertAlmostEqual(aniso["mean"], 2.5, places=6)
        self.assertAlmostEqual(t_par["mean"], 1.0, places=6)
        self.assertEqual(aniso["valid_cells"], 16)
        self.assertGreater(abs(self._uncentred_anisotropy(raw, b) - 2.5), 0.3)

    def test_electron_charge_sign_does_not_flip_the_temperature(self):
        mom, _, _, b = self._drifting_block("e", units.M_ELEC)
        tensor = central_pressure_tensor(
            moment_fields_by_name(mom), "e", units.M_ELEC)
        par, perp, _ = field_aligned_pressures(
            tensor["Pxx"], tensor["Pyy"], tensor["Pzz"],
            tensor["Pxy"], tensor["Pyz"], tensor["Pzx"],
            *(np.full((4, 4, 1), value) for value in b))
        np.testing.assert_allclose(tensor["n"], 2.0)
        np.testing.assert_allclose(par, 2.0, rtol=1e-10)
        np.testing.assert_allclose(perp, 5.0, rtol=1e-10)

    def test_synthetic_moments_hide_a_flow_in_the_raw_second_moment(self):
        """The fixture carries a drift only the per-cell subtraction removes.

        Without it no test would tell a pipeline that centres from one that
        does not: with p = 0 the correction term is identically zero.
        """
        case = "mirror_bimaxwellian_moderate"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {**os.environ, "PSC_PROFILE": case,
                   "PSC_ANALYSIS_DATA_DIR": "", "MPLCONFIGDIR": str(root / "mpl")}
            env.pop("PSC_ANALYSIS_CONFIG", None)
            build = subprocess.run(
                [sys.executable, "synthetic_run.py", str(root / "run"), "--case", case,
                 "--ngrid", "16", "--snapshots", "3", "--ppc", "4"],
                cwd=HERE, env=env, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr[-2000:])
            masses = json.loads(subprocess.run(
                [sys.executable, "-c",
                 "import json, psc_units as u; print(json.dumps([u.M_ION, u.M_ELEC]))"],
                cwd=HERE, env=env, capture_output=True, text=True, check=True).stdout)
            moment_file = sorted((root / "run").glob("pfd_moments.*.h5"))[-1]
            with h5py.File(moment_file, "r") as handle:
                group = handle["all_1st-0"]
                read = lambda name: np.asarray(group[f"{name}/p0/3d"], dtype=float)
                for suffix, mass in zip("ie", masses):
                    n = np.abs(read(f"rho_{suffix}"))
                    py = read(f"py_{suffix}")
                    txx, tyy = read(f"txx_{suffix}"), read(f"tyy_{suffix}")
                    flow = py ** 2 / (n * mass)
                    # t_xx is the same n T_perp with no flow in x, so the
                    # centred t_yy has to come back to it cell by cell.
                    residual = float(np.max(np.abs(tyy - flow - txx)))
                    scale = float(np.mean(txx))
                    self.assertLess(residual / scale, 1e-5, msg=suffix)
                    # The term removed is orders of magnitude above that
                    # residual: dropping it is not a rounding difference.
                    self.assertGreater(float(np.max(flow)), 100.0 * residual, msg=suffix)
                    # A box average takes the flow out of p_y before it can be
                    # subtracted, which is why the centring goes per cell.
                    peak = float(np.max(np.abs(py)))
                    self.assertGreater(peak, 0.0, msg=suffix)
                    self.assertLess(abs(float(np.mean(py))), 0.02 * peak, msg=suffix)
                    if suffix == "i":
                        # The flow has to be worth catching downstream too:
                        # test_pipeline_synthetic reads A_i off these moments
                        # to +-0.1, so the gap between centring and not has to
                        # clear that with room to spare. B is nearly along z
                        # here, so this ratio is A_i up to the wave's tilt.
                        tzz = read(f"tzz_{suffix}")
                        centred = 0.5 * np.mean((txx + tyy - flow) / tzz)
                        uncentred = 0.5 * np.mean((txx + tyy) / tzz)
                        self.assertGreater(uncentred - centred, 0.2)


if __name__ == "__main__":
    unittest.main()
