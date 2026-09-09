"""Global energy normalization and restart provenance checks."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from energy_conservation import read_energy_segments

HEADER = "# time EX2 EY2 EZ2 BX2 BY2 BZ2 E_electron E_ion\n"


class EnergyConservationTests(unittest.TestCase):
    def test_field_factor_and_conserved_particle_field_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag.asc"
            path.write_text(HEADER + "0 2 0 0 0 0 6 3 4\n1 4 0 0 0 0 6 2 4\n")
            rows, summary = read_energy_segments([path])
        self.assertEqual(rows[0]["E_E"], 1)
        self.assertEqual(rows[0]["E_B"], 3)
        self.assertEqual(rows[0]["E_total"], 11)
        self.assertEqual(summary["max_abs_relative_change"], 0)
        self.assertTrue(summary["includes_simulation_t0"])

    def test_overlap_is_deduplicated_but_conflicting_restart_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a.asc", Path(tmp)/"b.asc"
            a.write_text(HEADER + "1 2 0 0 0 0 6 3 4\n2 2 0 0 0 0 6 3 4\n")
            b.write_text(HEADER + "2 2 0 0 0 0 6 3 4\n3 2 0 0 0 0 6 3 4\n")
            rows, summary = read_energy_segments([b, a])
            self.assertEqual(len(rows), 3)
            self.assertFalse(summary["includes_simulation_t0"])
            b.write_text(HEADER + "2 4 0 0 0 0 6 3 4\n")
            with self.assertRaises(ValueError):
                read_energy_segments([a,b])

    def test_increasing_electron_energy_remains_in_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"diag.asc"
            path.write_text(HEADER + "0 0 0 0 0 0 2 1 1\n1 0 0 0 0 0 2 2 1\n")
            rows, summary = read_energy_segments([path])
        self.assertAlmostEqual(rows[-1]["relative_change"], 1/3)
        self.assertFalse(summary["detrended"])

    def test_launcher_preserves_each_previous_segment_without_overwriting(self):
        script = Path(__file__).resolve().parent.parent / "src/preserve_energy_diagnostic.sh"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"diag.asc"
            path.write_text(HEADER + "0 0 0 0 0 0 2 1 1\n")
            for _ in range(2):
                subprocess.run(["bash", str(script), tmp], check=True, capture_output=True)
            archives = list(Path(tmp).glob("diag.archive.*.asc"))
            self.assertEqual(len(archives), 2)
            self.assertTrue(all(p.read_bytes() == path.read_bytes() for p in archives))


if __name__ == "__main__":
    unittest.main()
