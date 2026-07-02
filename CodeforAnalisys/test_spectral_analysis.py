#!/usr/bin/env python3
"""Regression tests for the maintained PSC spectral-analysis path."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_reader import PICDataReader
from spectral_analysis import SpectralAnalyzer


class SpectralAnalysisTests(unittest.TestCase):
    def _write_snapshot(self, root: Path) -> Path:
        path = root / "pfd.000100_p000000.h5"
        nz, ny, nx = 64, 32, 1
        z = np.arange(nz)[:, None, None]
        bx = np.sin(2.0 * np.pi * 3.0 * z / nz) * np.ones((1, ny, nx))
        by = np.zeros_like(bx)
        bz = np.ones_like(bx)
        with h5py.File(path, "w") as handle:
            group = handle.create_group("jeh-uid-test")
            group.create_dataset("hx_fc/p0/3d", data=bx)
            group.create_dataset("hy_fc/p0/3d", data=by)
            group.create_dataset("hz_fc/p0/3d", data=bz)
        return path

    def test_default_pattern_finds_ranked_hdf5_and_auto_selects_yz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._write_snapshot(root)
            found = PICDataReader.find_files(str(root / "pfd.*.h5"))
            self.assertEqual(found, {100: str(snapshot)})

            analyzer = SpectralAnalyzer(dx=1.0, dy=0.5, dz=0.25, outdir=root / "out")
            result = analyzer.process_snapshot(str(snapshot), plane="auto")
            self.assertIsNotNone(result)
            self.assertEqual(result["plane"], "yz")
            self.assertEqual(result["axes"], ("z", "y"))
            self.assertEqual(result["spacing"], (0.25, 0.5))
            self.assertEqual(result["spectra"]["perp"]["psd_2d"].shape, (64, 32))

    def test_transverse_vector_power_recovers_known_parallel_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._write_snapshot(root)
            analyzer = SpectralAnalyzer(dx=1.0, dy=1.0, dz=1.0, outdir=root / "out")
            result = analyzer.process_snapshot(str(snapshot), plane="auto")
            psd = result["spectra"]["perp"]["psd_2d"]
            grids = result["k_grids"]
            peak = analyzer._peak_index(psd, grids)
            self.assertIsNotNone(peak)
            expected = 2.0 * np.pi * 3.0 / 64.0
            self.assertAlmostEqual(abs(grids["k_par"][peak]), expected, places=6)
            self.assertAlmostEqual(grids["k_perp"][peak], 0.0, places=12)
            self.assertLess(np.max(result["spectra"]["parallel"]["psd_2d"]), 1e-20)
            self.assertIsNone(
                analyzer._peak_index(result["spectra"]["parallel"]["psd_2d"], grids)
            )


if __name__ == "__main__":
    unittest.main()
