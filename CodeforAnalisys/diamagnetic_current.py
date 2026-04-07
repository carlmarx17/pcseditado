#!/usr/bin/env python3
"""
diamagnetic_current.py
======================
Ion and electron diamagnetic current analysis for PIC simulations (PSC).

Generates:
  - Individual 2D maps for selected simulation snapshots.
  - Animated GIF of the temporal evolution.

Diamagnetic current definition:
  J_d = (nabla P_perp × B_hat) / B

In 2D (YZ plane, B0 || z-hat):
  J_dx =  (dP_perp/dy · Bz  -  dP_perp/dz · By) / B^2
  J_dy, J_dz → 0  in the 2D YZ plane

The dominant out-of-plane component is J_dx.

PSC normalised units:
  - Pressure: P = n m <v^2>  (PIC), with mu_0 = 1
  - Magnetic field: normalised so B0 = B0_ref at t = 0
  - Resulting diamagnetic current in PIC units
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter
import h5py
import glob
import re
import os
import warnings
from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# ── Simulation parameters ────────────────────────────────────────────────────
B0_REF: float = 0.1        # Background field (PIC units)
MU0: float = 1.0           # Permeability (mu_0 = 1 in PSC)
DATA_DIR: str = "../build/src"
OUT_DIR: Path = Path("diamagnetic_plots")
OUT_DIR.mkdir(exist_ok=True)

# Steps to plot individually (representative subset)
PLOT_STEPS: list[int] = [0, 500, 1000, 2000, 3000, 5000, 7000, 9000, 11600]

# Gaussian smoothing width (cells) to suppress PIC shot noise
SMOOTH_SIGMA: float = 4.0

# GIF decimation factor
GIF_STRIDE: int = 5


# ── Dark-theme colour palette ────────────────────────────────────────────────
DARK_BG = "#0c0e14"
PANEL_BG = "#12151f"
TEXT_CLR = "#dde2f0"
GRID_CLR = "#232840"


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _extract_step(filename: str) -> int | None:
    """Extract the integer step number from a PSC HDF5 filename."""
    match = re.search(r"\.(\d+)_", filename)
    return int(match.group(1)) if match else None


def _find_files(pattern: str) -> dict[int, str]:
    """Return {step: filepath} dict for files matching *pattern*."""
    files = sorted(glob.glob(pattern))
    return {
        step: f
        for f in files
        if (step := _extract_step(f)) is not None
    }


def _read_field(filepath: str, group_prefix: str, dataset: str) -> np.ndarray:
    """Read a single 2D field slice from a PSC HDF5 file."""
    with h5py.File(filepath, "r") as f:
        grp = next(k for k in f.keys() if k.startswith(group_prefix))
        data = f[f"{grp}/{dataset}/p0/3d"][()]  # shape (Nz, Ny, 1)
    return data[:, :, 0]  # → (Nz, Ny)


def _read_coords(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D coordinate arrays (y, z) in d_i units."""
    with h5py.File(filepath, "r") as f:
        y_key = next(k for k in f.keys() if k.startswith("crd[1]"))
        z_key = next(k for k in f.keys() if k.startswith("crd[2]"))
        y = f[f"{y_key}/crd[1]/p0/1d"][()]
        z = f[f"{z_key}/crd[2]/p0/1d"][()]
    return y, z


# ── Diamagnetic current computation ─────────────────────────────────────────

def compute_diamagnetic_current(
    mom_file: str,
    fld_file: str,
    sigma: float = SMOOTH_SIGMA,
) -> dict[str, np.ndarray]:
    """
    Compute the diamagnetic current density for ions and electrons.

    Gaussian smoothing (sigma cells) is applied to pressure and B fields
    before computing gradients — essential to suppress PIC shot noise.

    Returns dict with keys:
        Jdia_i, Jdia_e, Jdia_tot  (Nz, Ny)
        Pperp_i, Pperp_e, Bmod
        y, z
    """
    # Perpendicular pressure: P_perp = 0.5 * (Pxx + Pyy)
    Pxx_i = _read_field(mom_file, "all_1st", "txx_i")
    Pyy_i = _read_field(mom_file, "all_1st", "tyy_i")
    Pxx_e = _read_field(mom_file, "all_1st", "txx_e")
    Pyy_e = _read_field(mom_file, "all_1st", "tyy_e")

    Pperp_i_raw = 0.5 * (Pxx_i + Pyy_i)
    Pperp_e_raw = 0.5 * (Pxx_e + Pyy_e)

    # Magnetic field components (no rescaling)
    Bx_raw = _read_field(fld_file, "jeh-", "hx_fc")
    By_raw = _read_field(fld_file, "jeh-", "hy_fc")
    Bz_raw = _read_field(fld_file, "jeh-", "hz_fc")

    # ── Gaussian smoothing ───────────────────────────────────────────────
    smooth = lambda arr: gaussian_filter(arr.astype(float), sigma=sigma)
    Pperp_i = smooth(Pperp_i_raw)
    Pperp_e = smooth(Pperp_e_raw)
    Bx, By, Bz = smooth(Bx_raw), smooth(By_raw), smooth(Bz_raw)

    B2 = Bx**2 + By**2 + Bz**2 + 1e-40

    # Coordinates
    y, z = _read_coords(mom_file)
    dy = np.mean(np.diff(y))
    dz = np.mean(np.diff(z))

    # Pressure gradients (centred differences)
    # Array layout: (Nz, Ny) → axis 0 = z, axis 1 = y
    def _grad(P):
        return np.gradient(P, dy, axis=1), np.gradient(P, dz, axis=0)

    dPi_dy, dPi_dz = _grad(Pperp_i)
    dPe_dy, dPe_dz = _grad(Pperp_e)

    # Diamagnetic current (x-component, out-of-plane):
    #   J_dx = (dP_perp/dy · Bz  -  dP_perp/dz · By) / B^2
    Jdia_i = (dPi_dy * Bz - dPi_dz * By) / B2
    Jdia_e = (dPe_dy * Bz - dPe_dz * By) / B2
    Jdia_tot = Jdia_i + Jdia_e

    return {
        "Jdia_i": Jdia_i,
        "Jdia_e": Jdia_e,
        "Jdia_tot": Jdia_tot,
        "Pperp_i": Pperp_i,
        "Pperp_e": Pperp_e,
        "Bmod": np.sqrt(B2),
        "y": y,
        "z": z,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_frame(
    data: dict,
    step: int,
    vmax_i: float | None = None,
    vmax_e: float | None = None,
    vmax_tot: float | None = None,
    return_img: bool = False,
):
    """
    Render a 3-panel figure (ion / electron / total diamagnetic current).

    If *return_img* is True, return an in-memory PIL Image (for GIF assembly).
    Otherwise save a PNG to disk.
    """
    y = data["y"]
    z = data["z"]
    Ji = data["Jdia_i"]
    Je = data["Jdia_e"]
    Jtot = data["Jdia_tot"]
    Bmod = data["Bmod"]

    # Symmetric colour limits
    if vmax_i is None:
        vmax_i = np.percentile(np.abs(Ji), 99.5)
    if vmax_e is None:
        vmax_e = np.percentile(np.abs(Je), 99.5)
    if vmax_tot is None:
        vmax_tot = np.percentile(np.abs(Jtot), 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)

    configs = [
        (Ji, "RdBu_r", vmax_i, r"$J^{(d)}_x$ ions", r"$J_d^{(\rm i)}$ [a.u.]"),
        (Je, "PuOr_r", vmax_e, r"$J^{(d)}_x$ electrons", r"$J_d^{(\rm e)}$ [a.u.]"),
        (Jtot, "seismic", vmax_tot, r"$J^{(d)}_x$ total", r"$J_d^{(\rm tot)}$ [a.u.]"),
    ]

    for ax, (field, cmap, vm, title, lbl) in zip(axes, configs):
        ax.set_facecolor(PANEL_BG)
        im = ax.pcolormesh(
            y, z, field, cmap=cmap, vmin=-vm, vmax=vm,
            shading="auto", rasterized=True,
        )

        # |B| contour overlay
        lvls = np.linspace(Bmod.min() * 0.6, Bmod.max() * 0.98, 8)
        ax.contour(y, z, Bmod, levels=lvls, colors="white", linewidths=0.5, alpha=0.4)

        cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
        cb.set_label(lbl, fontsize=10, color=TEXT_CLR)
        cb.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

        ax.set_xlabel(r"$y$ [$d_i$]", fontsize=11, color=TEXT_CLR)
        ax.set_ylabel(r"$z$ [$d_i$]", fontsize=11, color=TEXT_CLR)
        ax.set_title(title, fontsize=13, color=TEXT_CLR, pad=8)
        ax.tick_params(
            colors=TEXT_CLR, direction="in", which="both", top=True, right=True,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)

    fig.suptitle(
        rf"Diamagnetic Current  —  $t = {step}\ \Omega_{{ci}}^{{-1}}$"
        "\n"
        r"PSC  ($m_i/m_e = 64$,  $\kappa = 3$,  $B_0 = 0.1$)",
        fontsize=14,
        color=TEXT_CLR,
        y=1.02,
        fontweight="bold",
    )

    if return_img:
        if Image is None:
            raise ImportError("Pillow is required for GIF generation.")
        buf = BytesIO()
        fig.savefig(buf, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        return img
    else:
        out = OUT_DIR / f"jdia_step{step:06d}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mom_files = _find_files(f"{DATA_DIR}/pfd_moments.*.h5")
    fld_files = _find_files(f"{DATA_DIR}/pfd.*.h5")
    common = sorted(set(mom_files) & set(fld_files))

    if not common:
        print("ERROR: No matching HDF5 file pairs found.")
        exit(1)

    print(f"Available snapshots: {len(common)}  ({common[0]} → {common[-1]})")

    # ── 1. Compute global colour range (from a subset) ───────────────────
    print("\n[1/3] Computing global colour range...")
    sample_steps = common[:: max(1, len(common) // 12)]
    vmax_i_all, vmax_e_all, vmax_tot_all = [], [], []

    for s in sample_steps:
        d = compute_diamagnetic_current(mom_files[s], fld_files[s])
        vmax_i_all.append(np.percentile(np.abs(d["Jdia_i"]), 99.5))
        vmax_e_all.append(np.percentile(np.abs(d["Jdia_e"]), 99.5))
        vmax_tot_all.append(np.percentile(np.abs(d["Jdia_tot"]), 99.5))

    vmax_i = float(np.percentile(vmax_i_all, 90))
    vmax_e = float(np.percentile(vmax_e_all, 90))
    vmax_tot = float(np.percentile(vmax_tot_all, 90))
    print(f"   vmax_i={vmax_i:.4f}  vmax_e={vmax_e:.4f}  vmax_tot={vmax_tot:.4f}")

    # ── 2. Individual plots for selected steps ───────────────────────────
    print(f"\n[2/3] Generating individual plots for steps: {PLOT_STEPS}")
    for s in PLOT_STEPS:
        closest = min(common, key=lambda x: abs(x - s))
        print(f"  step {s} → closest available: {closest}")
        data = compute_diamagnetic_current(mom_files[closest], fld_files[closest])
        make_frame(data, closest, vmax_i, vmax_e, vmax_tot)

    # ── 3. Animated GIF ──────────────────────────────────────────────────
    gif_steps = common[::GIF_STRIDE]
    print(f"\n[3/3] Generating GIF with {len(gif_steps)} frames (stride={GIF_STRIDE})...")

    frames = []
    for i, s in enumerate(gif_steps):
        print(f"  frame {i + 1}/{len(gif_steps)}  step={s}", end="\r")
        data = compute_diamagnetic_current(mom_files[s], fld_files[s])
        img = make_frame(data, s, vmax_i, vmax_e, vmax_tot, return_img=True)
        frames.append(img)

    gif_path = OUT_DIR / "jdia_evolution.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,   # ms per frame
        loop=0,         # infinite loop
        optimize=True,
    )
    print(f"\n  GIF saved: {gif_path}  ({len(frames)} frames)")
    print("\nDone.")
