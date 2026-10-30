#!/usr/bin/env python3
"""
What does PSC write as ``txx..tzx`` in ``all_1st``: temperature or pressure?

The question is settled by two deliberately trivial runs of
``src/psc_test_pressure_moment.cxx`` and by looking at how the written field
varies in space:

* **Test 1** modulates the density, ``n(z) = n0 (1 + A sin 2 pi z / Lz)``,
  while every particle is drawn from the same uniform isotropic temperature
  ``T0`` with no drift.  A temperature output is flat in z; a pressure output
  follows ``n(z)``, since ``P = n k T``.
* **Test 2** keeps ``n`` and ``T`` uniform and imposes a sheared bulk drift
  ``u_y(z) = u0 sin 2 pi z / Lz`` on both species at once (zero net current).
  The raw second moment ``n m <v v>`` picks up ``n m u_y^2``; the central
  (thermal) one, ``n m <(v - u)(v - u)>``, does not.

Usage::

    python moment_output_test.py --test1 <dir> --test2 <dir> [--step 0]

Each directory is a run directory holding ``pfd_moments.<step>_p000000.h5``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps

ps.apply()

#: Species analysed. The ions carry the drift test: with mi/me = 100 the
#: ram term m u^2 is a hundred times larger for them than for the electrons.
SPECIES = "i"

MOMENTS = ("rho", "px", "py", "pz", "txx", "tyy", "tzz")


def read_profiles(run_dir: Path, step: int, mass: float = 100.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return z and the y-averaged moment profiles of one snapshot."""
    import h5py

    path = run_dir / f"pfd_moments.{step:06d}_p000000.h5"
    if not path.exists():
        raise FileNotFoundError(f"no moment output at {path}")

    with h5py.File(path, "r") as f:
        group = next(k for k in f if k.startswith("all_1st"))
        crd_key = next(k for k in f if k.startswith("crd[2]"))
        z = np.asarray(f[crd_key]["crd[2]"]["p0"]["1d"], dtype=float)
        # Averaging along the homogeneous y direction cuts the particle noise
        # by sqrt(ny) without touching the z structure under test.
        prof = {
            name: np.asarray(
                f[group][f"{name}_{SPECIES}"]["p0"]["3d"], dtype=float
            )[:, :, 0]
            for name in MOMENTS
        }
    # Centre cell by cell: averaging first leaves unresolved flow energy.
    central = thermal_pressure(prof, mass)
    prof.update({f"P{a}{a}": value for a, value in central.items()})
    return z, {name: value.mean(axis=1) for name, value in prof.items()}


def thermal_pressure(prof: dict[str, np.ndarray], mass: float) -> dict[str, np.ndarray]:
    """Central second moment, P_ij = M_ij - p_i p_j / (n m)."""
    if all(f"P{a}{a}" in prof for a in "xyz"):
        return {a: prof[f"P{a}{a}"] for a in "xyz"}
    n = np.where(prof["rho"] > 1e-6, prof["rho"], np.nan)
    return {
        axis: prof[f"t{axis}{axis}"] - prof[f"p{axis}"] ** 2 / (n * mass)
        for axis in ("x", "y", "z")
    }


def compare_models(observed, models, tolerance=0.10):
    """Require an absolute fit and a clear margin; otherwise inconclusive.

    RMS residual is relative to each predicted profile. The default 10%
    tolerance is a diagnostic threshold, not a statistical confidence level.
    """
    observed = np.asarray(observed)
    errors = {}
    for name, expected in models.items():
        expected = np.broadcast_to(expected, observed.shape)
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
            errors[name] = float("inf")
        else:
            scale = np.sqrt(np.mean(expected**2))
            errors[name] = float(np.sqrt(np.mean((observed-expected)**2)) / scale) if scale > 0 else float("inf")
    ranked = sorted(errors, key=errors.get)
    best, second = ranked[:2]
    verdict = best if errors[best] < tolerance and errors[second] > max(2*errors[best], tolerance) else "inconclusive"
    return {"verdict": verdict, "relative_rms_errors": errors}


def report(label: str, z, prof, mass: float, T0: float) -> None:
    """Print the numbers the figure is drawn from."""
    n = prof["rho"]
    P = thermal_pressure(prof, mass)
    uy = prof["py"] / (n * mass)
    span = lambda a: f"{np.nanmin(a):9.5f} .. {np.nanmax(a):9.5f}"
    print(f"\n{label}")
    print(f"  n                  {span(n)}   (swing {100 * np.ptp(n) / n.mean():5.1f} %)")
    print(f"  tzz                {span(prof['tzz'])}   (swing {100 * np.ptp(prof['tzz']) / prof['tzz'].mean():5.1f} %)")
    print(f"  tyy                {span(prof['tyy'])}   (swing {100 * np.ptp(prof['tyy']) / prof['tyy'].mean():5.1f} %)")
    print(f"  tzz / n            {span(prof['tzz'] / n)}   (T0 = {T0})")
    print(f"  (tyy - py^2/nm)/n  {span(P['y'] / n)}   (T0 = {T0})")
    print(f"  u_y = py / (n m)   {span(uy)}")


def make_figure(z1, prof1, z2, prof2, mass: float, T0: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    # ── Panel 1: density gradient at uniform temperature ────────────────────
    ax = axes[0]
    n1 = prof1["rho"]
    x1 = z1 / z1.max()
    ax.plot(x1, n1 / n1.mean(), color=ps.c("#58a6ff"), lw=2.0,
            label=r"density $n/\langle n\rangle$ (imposed)")
    ax.plot(x1, prof1["tzz"] / prof1["tzz"].mean(), color=ps.c("#ff7b72"), lw=2.0,
            ls="--", label=r"code output $t_{zz}/\langle t_{zz}\rangle$")
    ax.plot(x1, prof1["tzz"] / (n1 * T0), color=ps.c("#56d364"), lw=2.0,
            label=r"$t_{zz}/(n\,T_0)$")
    ax.axhline(1.0, color=ps.MUTED_CLR, lw=0.8, ls=":")
    ps.style_axes(ax, "Test 1 — density gradient, uniform $T$")
    ax.set_xlabel(r"$z/L_z$")
    ax.set_ylabel("normalised profile")
    ax.set_ylim(0.0, 1.8)
    ps.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False)

    # ── Panel 2: sheared bulk drift at uniform n and T ──────────────────────
    ax = axes[1]
    n2 = prof2["rho"]
    x2 = z2 / z2.max()
    P2 = thermal_pressure(prof2, mass)
    ram = n2 * mass * (prof2["py"] / (n2 * mass)) ** 2
    ax.plot(x2, prof2["tyy"] / T0, color=ps.c("#ff7b72"), lw=2.0,
            label=r"code output $t_{yy}/T_0$")
    ax.plot(x2, (P2["y"]) / T0, color=ps.c("#56d364"), lw=2.0,
            label=r"$(t_{yy}-p_y^2/nm)/T_0$")
    ax.plot(x2, (n2 * T0 + ram) / T0, color=ps.c("#d2a8ff"), lw=1.4, ls="--",
            label=r"$(n T_0 + n m u_y^2)/T_0$ (predicted)")
    ax.axhline(1.0, color=ps.MUTED_CLR, lw=0.8, ls=":")
    ps.style_axes(ax, "Test 2 — sheared bulk drift, uniform $n$ and $T$")
    ax.set_xlabel(r"$z/L_z$")
    ax.set_ylabel(r"second moment / $T_0$")
    ps.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False)

    ps.save(fig, out)
    print(f"\nfigure -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test1", type=Path, required=True,
                   help="run directory of PSC_MOMENT_TEST=1 (density gradient)")
    p.add_argument("--test2", type=Path, required=True,
                   help="run directory of PSC_MOMENT_TEST=2 (sheared drift)")
    p.add_argument("--step", type=int, default=0, help="output step to read")
    p.add_argument("--mass", type=float, default=100.0, help="ion mass m_i/m_e")
    p.add_argument("--T0", type=float, default=0.01, help="initial temperature")
    p.add_argument("--out", type=Path,
                   default=Path("analysis_results/moment_output_test.png"))
    args = p.parse_args()

    z1, prof1 = read_profiles(args.test1, args.step, args.mass)
    z2, prof2 = read_profiles(args.test2, args.step, args.mass)

    report("TEST 1 — n(z) modulated, T uniform", z1, prof1, args.mass, args.T0)
    report("TEST 2 — n, T uniform, u_y(z) modulated", z2, prof2, args.mass, args.T0)

    n1, n2 = prof1["rho"], prof2["rho"]
    result = {
        "regime": "nonrelativistic; ions q=+1; instantaneous step 0 recommended",
        "density_test": compare_models(prof1["tzz"], {
            "density_weighted_moment": n1 * args.T0,
            "temperature_energy": np.full_like(n1, args.T0),
        }),
        "drift_test": compare_models(prof2["tyy"], {
            "raw_second_moment": n2 * args.T0 + prof2["tyy"] - prof2["Pyy"],
            "central_pressure": n2 * args.T0,
        }),
        "recovered_temperature": {
            label: {a: float(np.mean(thermal_pressure(prof, args.mass)[a] / prof["rho"])) for a in "xyz"}
            for label, prof in (("test1", prof1), ("test2", prof2))
        },
        "expected_T0": args.T0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("\nMODEL COMPARISON\n" + json.dumps(result, indent=2))

    make_figure(z1, prof1, z2, prof2, args.mass, args.T0, args.out)


if __name__ == "__main__":
    main()
