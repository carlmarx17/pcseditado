#!/usr/bin/env python3
"""
synthetic_run.py — a small fake PSC run with known answers, to test the pipeline
================================================================================
Writes, in PSC's exact on-disk layout, a run whose physics is prescribed, so
every analysis script can be run end to end (``make thesis``) and its
numbers checked against the input before any COSMA time is spent:

* ``pfd.<step>_p000000.h5``: ``jeh-0/{ex,ey,ez}_ec``, ``{hx,hy,hz}_fc`` on
  the Yee mesh, (Nz, Ny, 1) arrays, plus ``crd[1]``/``crd[2]`` coordinates.
  B = B0 z + curl(A_x x) is built from a vector potential on the staggered
  grid, so its discrete divergence is zero to round-off. The mode is an
  oblique, compressive wave whose amplitude grows as exp(GAMMA t) from a
  noise floor and saturates (logistic), i.e. gamma is known.
  With ``--ion-cyclotron`` a second, weaker wave is added: a parallel
  (k_perp = 0), purely transverse, LEFT-hand polarised wave rotating in the
  ion gyration sense about B0, with its own growth rate GAMMA_IC and real
  frequency OMEGA_IC. It is the known answer for the mirror / ion-cyclotron
  branch separation (physical_diagnostics.classify_mode) and for the
  handedness of psi_pm (polarization_dispersion.handedness_check).
* ``pfd_moments.<step>_p000000.h5``: ``all_1st-0/{rho,px..pz,txx..tzx}_{i,e}``
  with n_i anticorrelated with |B| and P_perp in total-pressure balance.
* ``prt_<case>.<step>.h5``: bi-Maxwellian (or, for kappa profiles, bi-kappa
  with PSC's Gamma-Normal mixture) ions and electrons (A from the profile)
  inside the central 20 % window, with ``lo``/``hi`` attributes.
* ``diag.asc``: DiagEnergies columns with an exactly conserved total.
* ``psc_synthetic_<jobid>.out``: a job log with ``**** Step`` banners and
  ``gauss:``/``continuity:`` check lines.

Usage:
    python synthetic_run.py OUTDIR [--case mirror_bimaxwellian_moderate] [--ngrid 48]
    make thesis DATA_DIR=OUTDIR CASE=mirror_bimaxwellian_moderate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

#: Growth rate of the synthetic mode, in Omega_ci.
GAMMA = 0.25
#: Initial and saturation amplitudes of delta B / B0.
AMP0, AMP_SAT = 3e-4, 0.08
#: Background noise of delta B / B0 (divergence-free).
NOISE = 2e-5
#: Optional ion-cyclotron wave: growth rate and real frequency in Omega_ci,
#: initial and saturation amplitudes of delta B / B0, and k_par in box modes.
GAMMA_IC, OMEGA_IC = 0.15, 0.47
AMP0_IC, AMP_SAT_IC = 1e-4, 0.03
K_IC = 2


def build(outdir: Path, case: str, ngrid: int, t_end_omegaci: float, n_snap: int,
          ppc: int, seed: int = 1, ion_cyclotron: bool = False) -> dict:
    os.environ["PSC_PROFILE"] = case
    os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
    import importlib
    import psc_units
    units = importlib.reload(psc_units)
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    mr, b0, nicell = units.MASS_RATIO, units.B0, units.NICELL
    length_de = units.DOMAIN_DI * np.sqrt(mr)
    dx = length_de / ngrid
    dt = 0.95 * dx / np.sqrt(2.0)
    omega_ci = b0 / mr
    last_step = int(round(t_end_omegaci / (omega_ci * dt)))
    every = max(1, last_step // (n_snap - 1))
    steps = [i * every for i in range(n_snap)]

    z, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid), indexing="ij")
    kz, ky = 1, 3                      # oblique, mostly perpendicular: compressive
    phase = 2 * np.pi * (kz * z + ky * y) / ngrid
    k_code = 2 * np.pi * np.hypot(kz, ky) / length_de
    noise_ax = rng.standard_normal((ngrid, ngrid))

    def amplitude(t):
        g = AMP0 * np.exp(GAMMA * t)
        return g / (1.0 + g / AMP_SAT)

    def amplitude_ic(t):
        g = AMP0_IC * np.exp(GAMMA_IC * t)
        return g / (1.0 + g / AMP_SAT_IC) if ion_cyclotron else 0.0

    ti_par = units.TI_PAR
    ti_perp = units.TI_PERP
    te_par, te_perp = units.TE_PAR, units.TE_PERP
    crd = (np.arange(ngrid) + 0.5) * dx
    lo = np.array([0, ngrid // 2 - int(round(0.1 * ngrid)), ngrid // 2 - int(round(0.1 * ngrid))])
    hi = np.array([1, ngrid // 2 + int(round(0.1 * ngrid)), ngrid // 2 + int(round(0.1 * ngrid))])
    energies = []
    area = length_de ** 2
    e_total0 = None
    log_lines = []
    for step in steps:
        t = step * dt * omega_ci
        a = amplitude(t) * b0
        ax_ = a * np.sin(phase) / k_code + NOISE * b0 * dx * noise_ax
        by = (np.roll(ax_, -1, 0) - ax_) / dx
        bz = b0 - (np.roll(ax_, -1, 1) - ax_) / dx
        # Left-hand wave: dBx + i dBy = a exp[i(k z - omega t)], omega > 0.
        # At fixed z the vector turns as exp(-i omega t), i.e. clockwise seen
        # from +z, the sense in which a positive charge gyrates about B0 z.
        # It depends on z only, so div B stays zero.
        phase_ic = 2 * np.pi * K_IC * z / ngrid - OMEGA_IC * t
        a_ic = amplitude_ic(t) * b0
        bx = a_ic * np.cos(phase_ic)
        by = by + a_ic * np.sin(phase_ic)
        ex, ey, ez = (1e-7 * rng.standard_normal((ngrid, ngrid)) for _ in range(3))
        with h5py.File(outdir / f"pfd.{step:09d}_p000000.h5", "w") as f:
            for name, arr in (("ex_ec", ex), ("ey_ec", ey), ("ez_ec", ez),
                              ("jx_ec", 0 * ex), ("jy_ec", 0 * ex), ("jz_ec", 0 * ex),
                              ("hx_fc", bx), ("hy_fc", by), ("hz_fc", bz)):
                f.create_dataset(f"jeh-0/{name}/p0/3d", data=arr[:, :, None].astype("f4"))
            for axis in (1, 2):
                f.create_dataset(f"crd[{axis}]/p0/1d", data=crd - length_de / 2)

        # Cell-centred |B| and pressure-balanced moments.
        bz_c = 0.5 * (bz + np.roll(bz, -1, 0))
        by_c = 0.5 * (by + np.roll(by, -1, 1))
        bx_c = 0.25 * (bx + np.roll(bx, -1, 0) + np.roll(bx, -1, 1) + np.roll(bx, (-1, -1), (0, 1)))
        bmag = np.sqrt(bz_c ** 2 + by_c ** 2 + bx_c ** 2)
        # Isothermal total-pressure balance: B^2/2 + n (T_perp,i + T_perp,e) = const.
        dpm = 0.5 * (bmag ** 2 - b0 ** 2)
        n_i = 1.0 - dpm / (ti_perp + te_perp)
        with h5py.File(outdir / f"pfd_moments.{step:09d}_p000000.h5", "w") as f:
            for s, sign, tpar, tperp in (("i", 1, ti_par, ti_perp), ("e", -1, te_par, te_perp)):
                pperp = n_i * tperp
                comps = {"rho": sign * n_i, "px": 0 * n_i, "py": 0 * n_i, "pz": 0 * n_i,
                         "txx": pperp, "tyy": pperp, "tzz": n_i * tpar,
                         "txy": 0 * n_i, "tyz": 0 * n_i, "tzx": 0 * n_i}
                for name, arr in comps.items():
                    f.create_dataset(f"all_1st-0/{name}_{s}/p0/3d", data=arr[:, :, None].astype("f4"))

        e_b = 0.5 * float(np.mean(bx ** 2 + by ** 2 + bz ** 2)) * area
        e_e = 0.5 * float(np.mean(ex ** 2 + ey ** 2 + ez ** 2)) * area
        if e_total0 is None:
            k_i0 = 0.5 * (ti_par + 2 * ti_perp) * area
            k_e0 = 0.5 * (te_par + 2 * te_perp) * area
            e_total0 = e_b + e_e + k_i0 + k_e0
            e_b0 = e_b
        k_e = 0.5 * (te_par + 2 * te_perp) * area
        k_i = e_total0 - e_b - e_e - k_e     # the ions pay for the field energy
        energies.append((step * dt, e_e, e_b, k_e, k_i))
        log_lines.append(f"**** Step {step + 1} / {last_step + 1}, Code Time {step * dt:g}, Wall Time 1\n")
        log_lines.append(f"gauss: max_err = {3e-7 * (1 + t / 10):.3e} (thres 0.0001)\n")
        log_lines.append(f"continuity: max_err = {1e-8:.3e} (thres 0.0001)\n")

    # Particles in the prt window, a few snapshots.
    ncell = int(np.prod(hi - lo))
    weight = nicell / ppc
    for step in steps[:: max(1, len(steps) // 5)]:
        rows = []
        for q, m, tpar, tperp in ((1.0, mr, ti_par, ti_perp), (-1.0, 1.0, te_par, te_perp)):
            count = ppc * ncell
            yy = (lo[1] + rng.random(count) * (hi[1] - lo[1])) * dx - length_de / 2
            zz = (lo[2] + rng.random(count) * (hi[2] - lo[2])) * dx - length_de / 2
            # Kappa profiles load the PSC mixture: one S = sqrt((kappa - 3/2)/Y),
            # Y ~ Gamma(kappa - 1/2), shared by the three components.
            scale = (np.sqrt((units.KAPPA - 1.5) / rng.gamma(units.KAPPA - 0.5, 1.0, count))
                     if units.KAPPA else np.ones(count))
            v_perp = np.sqrt(tperp / m) * rng.standard_normal((2, count)) * scale
            v_par = np.sqrt(tpar / m) * rng.standard_normal(count) * scale
            rows.append((np.zeros(count), yy, zz, v_perp[0], v_perp[1], v_par,
                         np.full(count, q), np.full(count, m), np.full(count, weight)))
        data = np.concatenate([np.rec.fromarrays(r, names="x,y,z,px,py,pz,q,m,w") for r in rows])
        with h5py.File(outdir / f"prt_{case}.{step:09d}.h5", "w") as f:
            g = f.create_group("particles")
            g.attrs["lo"], g.attrs["hi"] = lo, hi
            f.create_dataset("particles/p0/1d", data=data)

    with (outdir / "diag.asc").open("w") as f:
        f.write("time EX2 EY2 EZ2 BX2 BY2 BZ2 E_electron E_ion\n")
        for time, e_e, e_b, k_e, k_i in energies:
            f.write(f"{time:.10g} {2*e_e/3:.12g} {2*e_e/3:.12g} {2*e_e/3:.12g} "
                    f"0 {0:.12g} {2*e_b:.12g} {k_e:.12g} {k_i:.12g}\n")
    (outdir / "psc_synthetic_1234.out").write_text("".join(log_lines))
    return {"steps": steps, "dt": dt, "gamma": GAMMA, "ngrid": ngrid, "e_b0": e_b0,
            "gamma_ic": GAMMA_IC if ion_cyclotron else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("outdir", type=Path)
    p.add_argument("--case", default="mirror_bimaxwellian_moderate")
    p.add_argument("--ngrid", type=int, default=48)
    p.add_argument("--t-end", type=float, default=40.0, help="final Omega_ci t")
    p.add_argument("--snapshots", type=int, default=41)
    p.add_argument("--ppc", type=int, default=40, help="particles per cell in the prt window")
    p.add_argument("--ion-cyclotron", action="store_true",
                   help="add the parallel left-hand ion-cyclotron wave (second branch)")
    args = p.parse_args()
    info = build(args.outdir, args.case, args.ngrid, args.t_end, args.snapshots, args.ppc,
                 ion_cyclotron=args.ion_cyclotron)
    print(f"Synthetic {args.case} run in {args.outdir}: {len(info['steps'])} snapshots, "
          f"gamma = {info['gamma']} Omega_ci")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
