"""Regression tests for the maintained reconnection analysis.

Covers: rectangular-box geometry read back from synthetic snapshots, region
statistics, kappa recovery through the full particle path, the correlation
helper, and an end-to-end run with a constructed B–kappa relation of known
sign.
"""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import kappa_eff
import reconnection_analysis as ra


MASS_RATIO = 25.0          # keeps the synthetic boxes small (d_i = 5 d_e)
D_I = np.sqrt(MASS_RATIO)
NY, NZ = 64, 128
LY_DI, LZ_DI = 12.8, 25.6  # rectangular, like the production run


def write_field_file(directory, step, bx, by, bz):
    """One pfd snapshot with the (nz, ny, 1) layout and cell-centred crd."""
    path = Path(directory) / f"pfd.{step:06d}_p000000.h5"
    dy = LY_DI * D_I / NY
    dz = LZ_DI * D_I / NZ
    with h5py.File(path, "w") as f:
        for name, data in (("hx_fc", bx), ("hy_fc", by), ("hz_fc", bz)):
            f.create_dataset(f"jeh-0/{name}/p0/3d", data=data[:, :, None])
        # y centred on 0, z starting at 0 — the reconnection convention.
        f.create_dataset("crd[1]/p0/1d",
                         data=-0.5 * LY_DI * D_I + (np.arange(NY) + 0.5) * dy)
        f.create_dataset("crd[2]/p0/1d", data=(np.arange(NZ) + 0.5) * dz)
    return path


def write_prt_file(directory, step, ions, electrons, lo, hi):
    """One prt snapshot; ions/electrons are dicts of y,z,px,py,pz arrays."""
    path = Path(directory) / f"prt_reconnection_comparable.{step:09d}.h5"
    dtype = [(k, "f8") for k in
             ("x", "y", "z", "px", "py", "pz", "q", "m", "w")]
    rows = []
    for spec, q, m in ((ions, 1.0, MASS_RATIO), (electrons, -1.0, 1.0)):
        n = spec["y"].size
        a = np.zeros(n, dtype=dtype)
        a["y"], a["z"] = spec["y"], spec["z"]
        a["px"], a["py"], a["pz"] = spec["px"], spec["py"], spec["pz"]
        a["q"], a["m"], a["w"] = q, m, 1.0
        rows.append(a)
    with h5py.File(path, "w") as f:
        g = f.create_group("particles")
        g.attrs["lo"] = lo
        g.attrs["hi"] = hi
        f.create_dataset("particles/p0/1d", data=np.concatenate(rows))
    return path


def uniform_positions(rng, n, region):
    return (rng.uniform(region.y_lo, region.y_hi, n),
            rng.uniform(region.z_lo, region.z_hi, n))


class GeometryTests(unittest.TestCase):
    def test_rectangular_box_and_dt_from_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            zero = np.zeros((NZ, NY))
            path = write_field_file(d, 0, zero, zero, zero + 0.5)
            geom = ra.resolve_geometry(str(path), "reconnection",
                                       mass_ratio=MASS_RATIO)
            self.assertEqual((geom.ny, geom.nz), (NY, NZ))
            self.assertAlmostEqual(geom.Ly_code / geom.d_i, LY_DI, places=6)
            self.assertAlmostEqual(geom.Lz_code / geom.d_i, LZ_DI, places=6)
            self.assertAlmostEqual(geom.b0, 0.5)
            self.assertAlmostEqual(geom.omega_ci, 1.0 / (MASS_RATIO * 2.0))
            expected_dt = 0.99 / np.sqrt(1 / geom.dy**2 + 1 / geom.dz**2)
            self.assertAlmostEqual(geom.dt_code, expected_dt, places=10)
            # Perturbed sheet at +Ly/4 above the centre (which is y = 0 here).
            self.assertAlmostEqual(geom.sheet_y_code / geom.d_i,
                                   LY_DI / 4.0, places=6)

    def test_region_means_match_direct_average(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(7)
            bz = 0.5 + 0.05 * rng.standard_normal((NZ, NY))
            zero = np.zeros((NZ, NY))
            path = write_field_file(d, 0, zero, zero, bz)
            geom = ra.resolve_geometry(str(path), "reconnection",
                                       mass_ratio=MASS_RATIO)
            b = ra.read_field_snapshot(str(path))
            region = ra.sheet_region(geom, 1.0)
            stats = ra.region_field_stats(b, region, geom.b0)
            direct = np.abs(bz)[region.sl_z, region.sl_y]
            self.assertAlmostEqual(stats["mean_B"], direct.mean() / geom.b0,
                                   places=12)
            self.assertAlmostEqual(stats["min_B"], direct.min() / geom.b0,
                                   places=12)
            # The sheet band must sit around +Ly/4, not the domain centre.
            y_sel = geom.y_centers[region.sl_y] / geom.d_i
            self.assertTrue(np.all(np.abs(y_sel - LY_DI / 4.0) <= 1.0 + 1e-9))


class KappaPathTests(unittest.TestCase):
    def test_kappa_recovered_from_prt_file(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as d:
            zero = np.zeros((NZ, NY))
            fpath = write_field_file(d, 0, zero, zero, zero + 0.5)
            geom = ra.resolve_geometry(str(fpath), "reconnection",
                                       mass_ratio=MASS_RATIO)
            lo, hi = [0, 16, 32], [1, 48, 96]
            n = 200_000
            vpar, vp1, vp2 = kappa_eff.sample_bikappa(
                n, 3.0, 1.0, np.sqrt(2.0), rng=rng)
            region = ra.make_box_region(
                geom, "w",
                geom.y_centers[0] - 0.5 * geom.dy + lo[1] * geom.dy,
                geom.y_centers[0] - 0.5 * geom.dy + hi[1] * geom.dy,
                geom.z_centers[0] - 0.5 * geom.dz + lo[2] * geom.dz,
                geom.z_centers[0] - 0.5 * geom.dz + hi[2] * geom.dz)
            y, z = uniform_positions(rng, n, region)
            ions = {"y": y, "z": z, "px": vp1, "py": vp2, "pz": vpar}
            em = kappa_eff.sample_bimaxwellian(50_000, 1.0, 1.0, rng=rng)
            ye, ze = uniform_positions(rng, 50_000, region)
            electrons = {"y": ye, "z": ze,
                         "px": em[1], "py": em[2], "pz": em[0]}
            ppath = write_prt_file(d, 0, ions, electrons, lo, hi)

            res = ra.kappa_of_snapshot(str(ppath), "ion",
                                       np.array([0.0, 0.0, 1.0]), n_boot=0)
            self.assertLess(abs(res["kappa"] - 3.0), 0.3)
            self.assertEqual(res["n_particles"], n)
            # The electrons must come out Maxwellian-consistent, proving the
            # charge-sign selection actually separates the species.
            res_e = ra.kappa_of_snapshot(str(ppath), "electron",
                                         np.array([0.0, 0.0, 1.0]), n_boot=0)
            self.assertLess(res_e["inv_kappa"], 0.05)

    def test_field_aligned_rotation_is_orthonormal(self):
        rng = np.random.default_rng(3)
        v = rng.standard_normal((3, 1000))
        b_hat = np.array([0.3, -0.5, 0.8])
        b_hat /= np.linalg.norm(b_hat)
        vpar, vp1, vp2 = ra.field_aligned_velocities(*v, b_hat)
        np.testing.assert_allclose(vpar**2 + vp1**2 + vp2**2,
                                   (v**2).sum(axis=0), rtol=1e-12)


class CorrelationTests(unittest.TestCase):
    def test_perfect_and_degenerate_cases(self):
        x = np.linspace(1.0, 2.0, 12)
        up = ra.correlate_series(x, 3.0 * x + 1.0)
        self.assertAlmostEqual(up["pearson_r"], 1.0, places=12)
        self.assertAlmostEqual(up["spearman_rho"], 1.0, places=12)
        down = ra.correlate_series(x, -x)
        self.assertAlmostEqual(down["pearson_r"], -1.0, places=12)
        self.assertTrue(np.isnan(
            ra.correlate_series(x[:2], x[:2])["pearson_r"]))
        self.assertTrue(np.isnan(
            ra.correlate_series(x, np.ones_like(x))["pearson_r"]))

    def test_lag_scan_finds_shift(self):
        rng = np.random.default_rng(5)
        base = np.cumsum(rng.standard_normal(40))
        res = ra.correlate_series(base[:-2], base[2:], max_lag=4)
        self.assertIsNotNone(res["lag_scan"])
        self.assertGreaterEqual(abs(res["lag_scan"]["best_r"]),
                                abs(res["pearson_r"]) - 1e-12)


class EndToEndTests(unittest.TestCase):
    def test_full_run_recovers_constructed_correlation(self):
        rng = np.random.default_rng(11)
        steps = [0, 100, 200, 300, 400]
        kappas = [2.5, 3.0, 4.0, 6.0, 10.0]      # tail relaxes ...
        b_scale = [1.0, 0.95, 0.9, 0.85, 0.8]    # ... while <|B|> drops
        lo, hi = [0, 16, 32], [1, 48, 96]
        with tempfile.TemporaryDirectory() as d:
            zero = np.zeros((NZ, NY))
            for step, scale in zip(steps, b_scale):
                write_field_file(d, step, zero, zero, zero + 0.5 * scale)
            geom = ra.resolve_geometry(
                str(Path(d) / f"pfd.{steps[0]:06d}_p000000.h5"),
                "reconnection", mass_ratio=MASS_RATIO)
            region = ra.make_box_region(
                geom, "w",
                geom.y_centers[0] - 0.5 * geom.dy + lo[1] * geom.dy,
                geom.y_centers[0] - 0.5 * geom.dy + hi[1] * geom.dy,
                geom.z_centers[0] - 0.5 * geom.dz + lo[2] * geom.dz,
                geom.z_centers[0] - 0.5 * geom.dz + hi[2] * geom.dz)
            for step, kap in zip(steps, kappas):
                n = 30_000
                vpar, vp1, vp2 = kappa_eff.sample_bikappa(
                    n, kap, 1.0, 1.0, rng=rng)
                y, z = uniform_positions(rng, n, region)
                ions = {"y": y, "z": z, "px": vp1, "py": vp2, "pz": vpar}
                ye, ze = uniform_positions(rng, 1000, region)
                em = kappa_eff.sample_bimaxwellian(1000, 1.0, 1.0, rng=rng)
                electrons = {"y": ye, "z": ze,
                             "px": em[1], "py": em[2], "pz": em[0]}
                write_prt_file(d, step, ions, electrons, lo, hi)

            outdir = Path(d) / "out"
            args = ra.build_parser().parse_args([
                "--data-dir", d, "--outdir", str(outdir),
                "--profile", "reconnection", "--mass-ratio", str(MASS_RATIO),
                "--kappa-boot", "0", "--panel-times", "2"])
            self.assertEqual(ra.run_analysis(args), 0)

            for name in ("reconnection_field_timeseries.csv",
                         "reconnection_kappa_timeseries.csv",
                         "b_kappa_correlation.json",
                         "reconnection_summary.json",
                         "b_kappa_evolution.png", "b_kappa_scatter.png",
                         "reconnection_overview.png", "reconnected_flux.png"):
                self.assertTrue((outdir / name).exists(), name)

            corr = json.loads((outdir / "b_kappa_correlation.json").read_text())
            # 1/kappa falls with <|B|>: strong positive correlation built in.
            self.assertGreater(corr["window_B_vs_inv_kappa"]["pearson_r"], 0.9)
            self.assertEqual(corr["window_B_vs_inv_kappa"]["n"], len(steps))

            summary = json.loads(
                (outdir / "reconnection_summary.json").read_text())
            self.assertEqual(summary["inputs"]["n_particle_snapshots"],
                             len(steps))
            self.assertAlmostEqual(summary["geometry"]["Ly_di"], LY_DI,
                                   places=5)
            win = summary["regions"]["window"]
            self.assertAlmostEqual(win["y_di"][0],
                                   region.y_lo / geom.d_i, places=6)

    def test_run_without_particles_still_writes_field_series(self):
        with tempfile.TemporaryDirectory() as d:
            zero = np.zeros((NZ, NY))
            for step in (0, 100):
                write_field_file(d, step, zero, zero, zero + 0.5)
            outdir = Path(d) / "out"
            args = ra.build_parser().parse_args([
                "--data-dir", d, "--outdir", str(outdir),
                "--profile", "reconnection", "--mass-ratio", str(MASS_RATIO),
                "--skip-overview"])
            self.assertEqual(ra.run_analysis(args), 0)
            self.assertTrue(
                (outdir / "reconnection_field_timeseries.csv").exists())
            self.assertFalse((outdir / "b_kappa_correlation.json").exists())


if __name__ == "__main__":
    unittest.main()
