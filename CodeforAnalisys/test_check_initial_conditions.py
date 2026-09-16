"""Regression tests for the t0 CASE-vs-data cross-check."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from check_initial_conditions import check_initial_conditions


class CheckInitialConditionsTests(unittest.TestCase):
    def particle_file(self, directory, ti_par, ti_perp, te_par, te_perp):
        dtype = [(k, "f8") for k in ["q", "m", "px", "py", "pz", "w"]]
        rows = []

        def species(q, m, tpar, tperp):
            pz = np.sqrt(tpar / m)
            pperp = np.sqrt(tperp / m)
            for spz in (1, -1):
                for spx in (1, -1):
                    for spy in (1, -1):
                        rows.append((q, m, spx * pperp, spy * pperp, spz * pz, 1.0))

        species(1.0, 200.0, ti_par, ti_perp)
        species(-1.0, 1.0, te_par, te_perp)
        data = np.array(rows, dtype=dtype)
        path = Path(directory) / "prt_case.000000.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("particles/p0/1d", data=data)
        return str(Path(directory) / "prt_case.*.h5")

    def field_file(self, directory, b0, n=4):
        path = Path(directory) / "pfd.000000_p000000.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("jeh-0/hz_fc/p0/3d", data=np.full((n, n, 1), b0))
        return str(Path(directory) / "pfd.*_p*.h5")

    def moment_file(self, directory, n0, n=4):
        path = Path(directory) / "pfd_moments.000000_p000000.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("all_1st-0/rho_i/p0/3d", data=np.full((n, n, 1), n0))
            f.create_dataset("all_1st-0/rho_e/p0/3d", data=np.full((n, n, 1), n0))
        return str(Path(directory) / "pfd_moments.*_p*.h5")

    def declared(self, ti_par=0.01, ti_perp=0.02, te_par=1e-4, te_perp=2e-4, b0=0.08, n0=1.0):
        return {
            "ion": {"T_parallel": ti_par, "T_perp": ti_perp},
            "electron": {"T_parallel": te_par, "T_perp": te_perp},
            "B0": b0,
            "N0": n0,
        }

    def test_matching_state_passes(self):
        with tempfile.TemporaryDirectory() as d:
            particles = self.particle_file(d, 0.01, 0.02, 1e-4, 2e-4)
            fields = self.field_file(d, 0.08)
            moments = self.moment_file(d, 1.0)
            failures = check_initial_conditions(
                particles, fields, moments, self.declared(), 10_000, 0.15, 0.05
            )
        self.assertEqual(failures, [])

    def test_wrong_case_temperature_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            # Data was written with double the declared ion T_perp (e.g. wrong CASE).
            particles = self.particle_file(d, 0.01, 0.04, 1e-4, 2e-4)
            fields = self.field_file(d, 0.08)
            moments = self.moment_file(d, 1.0)
            failures = check_initial_conditions(
                particles, fields, moments, self.declared(), 10_000, 0.15, 0.05
            )
        self.assertTrue(any("ion T_perp" in f for f in failures))

    def test_wrong_b0_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            particles = self.particle_file(d, 0.01, 0.02, 1e-4, 2e-4)
            fields = self.field_file(d, 0.05)  # declared B0 is 0.08
            moments = self.moment_file(d, 1.0)
            failures = check_initial_conditions(
                particles, fields, moments, self.declared(), 10_000, 0.15, 0.05
            )
        self.assertTrue(any(f.startswith("B0:") for f in failures))

    def test_missing_particles_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            fields = self.field_file(d, 0.08)
            moments = self.moment_file(d, 1.0)
            failures = check_initial_conditions(
                str(Path(d) / "prt_case.*.h5"), fields, moments,
                self.declared(), 10_000, 0.15, 0.05,
            )
        self.assertTrue(any("No particle snapshots" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
