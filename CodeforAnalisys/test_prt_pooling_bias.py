"""The prt-window pooling bias: the identity, the estimator and the noise floor.

``plot_prt._compute_particle_energies`` pools the whole prt window under one
mean. These tests pin the three statements ``prt_pooling_bias.py`` rests on:

1. pooling moves exactly D = 0.5 m sum_c w_c |u_c - <u>|^2 from the bulk
   channel into the thermal one, checked against the production function
   itself rather than a reimplementation of it;
2. the per-cell moments recover that same D;
3. with no flow at all, D is shot noise of the cell mean and sits near
   E_th / N_c, so the floor falls as one over particles per cell.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

import prt_pooling_bias as bias

#: 50 cells per axis puts the 0.4..0.6 window on cells 20..29, so the tests
#: cover the slicing too: the plasma outside the window differs. 100 cells is
#: also enough for the variance of the cell mean to be a measurement.
NGRID, WINDOW_LO, WINDOW_HI = 50, 20, 30
MASS = 4.0
T_PAR, T_PERP = 0.02, 0.04
#: 0.5 m <|u - u_c|^2> of the sample below, used for the analytic floor.
E_TH_EXACT = 0.5 * (T_PAR + 2.0 * T_PERP)


def _window_cells() -> list[tuple[int, int]]:
    return [(jz, jy) for jz in range(WINDOW_LO, WINDOW_HI)
            for jy in range(WINDOW_LO, WINDOW_HI)]


def _drift_pattern(amplitude: float) -> dict[tuple[int, int], float]:
    """A sheared drift across the window whose mean very nearly cancels.

    That is the structure one window mean cannot see: it survives in the
    per-cell split and disappears from the pooled one.
    """
    span = WINDOW_HI - WINDOW_LO
    return {(jz, jy): amplitude * np.sin(2.0 * np.pi * (jz - WINDOW_LO) / span)
            for jz, jy in _window_cells()}


def _sample(rng, drifts: dict, per_cell: int):
    """Velocities (3, N) of the window cells with a per-cell drift along y."""
    sigma = np.sqrt(np.array([T_PERP, T_PERP, T_PAR]) / MASS)
    velocity, cell = [], []
    for index, drift in enumerate(drifts.values()):
        local = rng.standard_normal((3, per_cell)) * sigma[:, None]
        local[1] += drift
        velocity.append(local)
        cell.append(np.full(per_cell, index))
    return np.concatenate(velocity, axis=1), np.concatenate(cell)


def _write_prt(path: Path, ions, electrons) -> None:
    """One prt snapshot in PSC's layout, ions and electrons in equal numbers."""
    rows = []
    for velocity, charge, mass in ((ions, 1.0, MASS), (electrons, -1.0, 1.0)):
        count = velocity.shape[1]
        rows.append(np.rec.fromarrays(
            [np.zeros(count), np.zeros(count), np.zeros(count),
             velocity[0], velocity[1], velocity[2],
             np.full(count, charge), np.full(count, mass), np.ones(count)],
            names="x,y,z,px,py,pz,q,m,w"))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("particles")
        group.attrs["lo"] = np.array([0, WINDOW_LO, WINDOW_LO])
        group.attrs["hi"] = np.array([1, WINDOW_HI, WINDOW_HI])
        handle.create_dataset("particles/p0/1d",
                              data=np.concatenate(rows).astype(rows[0].dtype))


def _splits(velocity, cell, mass: float) -> tuple[dict, dict]:
    """The pooled and per-cell splits, per macroparticle, from the sample."""
    mean = velocity.mean(axis=1)
    pooled = {
        "bulk": 0.5 * mass * float((mean ** 2).sum()),
        "thermal": 0.5 * mass * float(((velocity - mean[:, None]) ** 2).sum(axis=0).mean()),
    }
    per_cell = {"bulk": 0.0, "thermal": 0.0}
    for index in np.unique(cell):
        inside = cell == index
        weight = inside.sum() / cell.size
        local = velocity[:, inside]
        local_mean = local.mean(axis=1)
        per_cell["bulk"] += weight * 0.5 * mass * float((local_mean ** 2).sum())
        per_cell["thermal"] += weight * 0.5 * mass * float(
            ((local - local_mean[:, None]) ** 2).sum(axis=0).mean())
    return pooled, per_cell


def _moment_maps(velocity, cell, drifts: dict, per_cell: int, nicell: int,
                 suffix: str = "i") -> dict:
    """Per-cell moments of that sample, in PSC's convention.

    ``t_ab = n m <u_a v_b>`` with v = u, the non-relativistic limit the whole
    analysis works in, so ``central_pressure_tensor`` must return exactly the
    sample covariance of each cell. The density is fixed by the convention
    ``N_c = nicell n_c``, which is what makes the noise floor 1 / N_c.
    """
    maps = {f"{name}_{suffix}": np.zeros((NGRID, NGRID)) for name in bias.MOMENT_NAMES}
    # Outside the window: a different plasma, at rest, so a wrong slice shows.
    maps[f"rho_{suffix}"] += 1.0
    for axis in "xyz":
        maps[f"t{axis}{axis}_{suffix}"] += 0.5 * (T_PAR + T_PERP)

    density = per_cell / nicell
    for index, (jz, jy) in enumerate(drifts):
        local = velocity[:, cell == index]
        maps[f"rho_{suffix}"][jz, jy] = density
        for i, axis in enumerate("xyz"):
            maps[f"p{axis}_{suffix}"][jz, jy] = density * MASS * float(local[i].mean())
            maps[f"t{axis}{axis}_{suffix}"][jz, jy] = (
                density * MASS * float((local[i] * local[i]).mean()))
        for name, i, j in (("xy", 0, 1), ("yz", 1, 2), ("zx", 2, 0)):
            maps[f"t{name}_{suffix}"][jz, jy] = (
                density * MASS * float((local[i] * local[j]).mean()))
    return maps


class PoolingBiasTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261027)
        self.drifts = _drift_pattern(0.08)
        self.per_cell = 300
        self.velocity, self.cell = _sample(self.rng, self.drifts, self.per_cell)

    def test_pooling_moves_the_cell_mean_variance_out_of_the_bulk_channel(self):
        """Checked against plot_prt itself, not against a copy of its formula."""
        from plot_prt import _compute_particle_energies
        electrons = self.rng.standard_normal(self.velocity.shape) * 0.01
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prt_test.000000000.h5"
            _write_prt(path, self.velocity, electrons)
            measured = _compute_particle_energies(str(path))

        pooled, per_cell = _splits(self.velocity, self.cell, MASS)
        self.assertAlmostEqual(measured["ion_kinetic_bulk"], pooled["bulk"], places=12)
        self.assertAlmostEqual(measured["ion_thermal_energy"], pooled["thermal"], places=12)

        # The identity: one D, taken out of bulk and added to thermal.
        d_thermal = pooled["thermal"] - per_cell["thermal"]
        d_bulk = per_cell["bulk"] - pooled["bulk"]
        self.assertAlmostEqual(d_thermal, d_bulk, places=12)
        # And it is worth measuring: this drift pattern cancels in the window
        # mean, so the pooled bulk channel keeps almost none of its energy.
        self.assertGreater(d_bulk, 10.0 * pooled["bulk"])

    def test_per_cell_moments_recover_the_same_bias(self):
        nicell = 512
        maps = _moment_maps(self.velocity, self.cell, self.drifts, self.per_cell, nicell)
        result = bias.pooling_bias(maps, "i", MASS, nicell)
        pooled, per_cell = _splits(self.velocity, self.cell, MASS)

        self.assertEqual(result["cells"], len(self.drifts))
        np.testing.assert_allclose(result["E_th_per_cell"], per_cell["thermal"], rtol=1e-12)
        np.testing.assert_allclose(result["E_bulk_pooled"], pooled["bulk"], rtol=1e-9)
        np.testing.assert_allclose(result["D_total"],
                                   pooled["thermal"] - per_cell["thermal"], rtol=1e-9)
        np.testing.assert_allclose(result["E_th_pooled"], pooled["thermal"], rtol=1e-12)

    def test_noise_floor_falls_as_one_over_particles_per_cell(self):
        """With no flow, D is the shot noise of the cell mean: about E_th / N_c.

        The expectation is (K-1)/K / N_c over K cells, so the analytic bound
        1 / N_c is slightly conservative by construction.
        """
        flat = {cell: 0.0 for cell in _window_cells()}
        cells = len(flat)
        expected = (cells - 1) / cells
        ratios = {}
        for per_cell in (250, 1000, 4000):
            velocity, cell = _sample(self.rng, flat, per_cell)
            maps = _moment_maps(velocity, cell, flat, per_cell, per_cell)
            result = bias.pooling_bias(maps, "i", MASS, per_cell)
            ratios[per_cell] = result["D_over_E_th"] * per_cell
            # n_c = 1 here, so the bound is exactly 1 / N_c.
            np.testing.assert_allclose(result["D_noise_over_E_th"], 1.0 / per_cell,
                                       rtol=1e-10)
            np.testing.assert_allclose(result["E_th_per_cell"], E_TH_EXACT, rtol=0.02)
            # 100 cells give the variance to ~14 %, so 25 % is a loose band.
            self.assertAlmostEqual(ratios[per_cell], expected, delta=0.25 * expected)

        # The floor is a fixed multiple of 1/N_c, not a constant.
        self.assertLess(max(ratios.values()) / min(ratios.values()), 1.5)

    def test_a_window_without_usable_cells_is_an_error_not_a_zero(self):
        maps = _moment_maps(self.velocity, self.cell, self.drifts, self.per_cell, 512)
        maps["rho_i"][:] = 0.0
        with self.assertRaises(RuntimeError):
            bias.pooling_bias(maps, "i", MASS, 512)


if __name__ == "__main__":
    unittest.main()
