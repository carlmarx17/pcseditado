#!/usr/bin/env python3
"""
magnetic_field_viz.py
=====================
Visualize magnetic field fluctuations from PIC simulation HDF5 files.

Usage
-----
    python magnetic_field_viz.py --plane xy --B0 0.01 --fixed_scale --create_gifs
    python magnetic_field_viz.py --steps 100 200 300 --comps Bx Bz --no_mag
    python magnetic_field_viz.py --smooth 1.5 --out_dir my_images

HDF5 structure expected
-----------------------
    /jeh-<suffix>/
        hx_fc/p0/3d   → Bx  (shape: Nx × Ny × Nz)
        hy_fc/p0/3d   → By
        hz_fc/p0/3d   → Bz
"""

import re
import glob
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from data_reader import PICDataReader

matplotlib.use("Agg")  # Non-interactive backend (safe for headless servers)


# =============================================================================
# Section 1 – Configuration
# =============================================================================

DEFAULTS = dict(
    pattern    = "pfd.*.h5",
    out_dir    = "field_images",
    B0         = 0.01,
    fluct_amp  = 0.1,
    smooth     = 0.8,
    gif_dur    = 0.2,
    plane      = "xy",
    components = ["Bx", "By", "Bz"],
)


# =============================================================================
# Section 2 – Colormaps
# =============================================================================

def _diverging_cmap(name: str, neg: str, pos: str) -> LinearSegmentedColormap:
    """
    Build a high-contrast diverging colormap.

    Saturated colors at the extremes, darker tones toward the center.
    Used for individual field components (positive ↔ negative fluctuations).
    """
    PALETTES = {
        ("blue",  "red")   : ["#00001F","#00008F","#0000FF","#0066FF","#FF0000","#8B0000","#450000"],
        ("green", "purple"): ["#001F00","#004400","#00FF00","#66FF00","#800080","#4B0082","#2A0045"],
        ("cyan",  "orange"): ["#002A2A","#006666","#00FFFF","#00CCCC","#FF8C00","#FF4500","#8B2500"],
    }
    colors = PALETTES.get((neg, pos), PALETTES[("cyan", "orange")])
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def _magnitude_cmap() -> LinearSegmentedColormap:
    """
    Build a sequential colormap for |δB| magnitude.

    Yellow (high) → green → black (near zero).
    """
    colors = ["#FFFF00","#EEEE00","#CCCC00","#AAAA00","#888800",
              "#00FF00","#008800","#004400","#002200","#001100","#000000"]
    return LinearSegmentedColormap.from_list("magnitude", colors, N=256)


COLORMAPS = {
    "Bx"  : _diverging_cmap("Bx",   "blue",  "red"),
    "By"  : _diverging_cmap("By",   "green", "purple"),
    "Bz"  : _diverging_cmap("Bz",   "cyan",  "orange"),
    "Bmag": _magnitude_cmap(),
}


# =============================================================================
# Section 3 – Data Loading
# =============================================================================

def discover_files(pattern: str) -> dict[int, str]:
    """
    Find HDF5 files matching a glob pattern and index them by simulation step.
    Delegates to PICDataReader.

    Returns
    -------
    dict mapping {step_number: file_path}
    """
    return PICDataReader.find_files(pattern)


def load_field_3d(path: str, step: int, B0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and scale the 3D magnetic field arrays from a single HDF5 file.
    Delegates to PICDataReader.

    Returns
    -------
    (Bx, By, Bz) as 3-D numpy arrays
    """
    try:
        fields = PICDataReader.read_multiple_fields_3d(path, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
    except KeyError as e:
        raise KeyError(f"Missing group or dataset in file for step {step}: {e}")
        
    bx = fields["hx_fc/p0/3d"] * B0
    by = fields["hy_fc/p0/3d"] * B0
    bz = fields["hz_fc/p0/3d"] * B0

    if np.all(bx == 0) and np.all(by == 0) and np.all(bz == 0):
        raise ValueError(f"Step {step}: all field components are zero – check your file.")
    return bx, by, bz


def slice_3d(
    bx3d: np.ndarray, by3d: np.ndarray, bz3d: np.ndarray,
    plane: str, idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    Extract a 2-D cross-section from 3-D field arrays.

    Parameters
    ----------
    plane : "xy", "xz", or "yz"
    idx   : slice index along the normal axis (default: midpoint)

    Returns
    -------
    (bx2d, by2d, bz2d, xlabel, ylabel)
    """
    shape = bx3d.shape
    if plane == "xy":
        k = idx if idx is not None else shape[2] // 2
        return bx3d[:, :, k], by3d[:, :, k], bz3d[:, :, k], "x", "y"
    elif plane == "xz":
        k = idx if idx is not None else shape[1] // 2
        return bx3d[:, k, :], by3d[:, k, :], bz3d[:, k, :], "x", "z"
    elif plane == "yz":
        k = idx if idx is not None else shape[0] // 2
        return bx3d[k, :, :], by3d[k, :, :], bz3d[k, :, :], "y", "z"
    else:
        raise ValueError(f"Unknown plane '{plane}'. Choose from: 'xy', 'xz', 'yz'.")


# =============================================================================
# Section 4 – Field Processing
# =============================================================================

def fluctuation(data: np.ndarray, B0: float, sigma: float = 0.0) -> np.ndarray:
    """
    Compute the normalized fluctuation of a single field component.

    Formula:  δB_i = (B_i − ⟨B_i⟩) / B0

    Parameters
    ----------
    data  : 2-D field component array
    B0    : reference field (normalization)
    sigma : Gaussian smoothing width (0 = skip smoothing)
    """
    result = (data - data.mean()) / B0
    return gaussian_filter(result, sigma=sigma) if sigma > 0 else result


def fluctuation_magnitude(
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
    B0: float, sigma: float = 0.0,
) -> np.ndarray:
    """
    Compute the magnitude of the total fluctuation vector.

    Formula:  |δB| = √(δBx² + δBy² + δBz²) / B0

    Parameters
    ----------
    bx, by, bz : 2-D field component arrays
    B0         : reference field (normalization)
    sigma      : Gaussian smoothing applied to the final magnitude
    """
    mag = np.sqrt(
        (bx - bx.mean()) ** 2 +
        (by - by.mean()) ** 2 +
        (bz - bz.mean()) ** 2
    )
    result = gaussian_filter(mag, sigma=sigma) if sigma > 0 else mag
    return result / B0


# =============================================================================
# Section 5 – Plotting
# =============================================================================

def color_limits(
    data: np.ndarray,
    comp: str,
    fixed_vmin: float | None,
    fixed_vmax: float | None,
    fluct_amp: float,
    dynamic: bool,
) -> tuple[float, float]:
    """
    Determine color-scale limits for a field map, in priority order:

    1. Pre-computed global limits (``fixed_vmin`` / ``fixed_vmax``) if provided.
    2. Dynamic limits derived from the current snapshot (99.5th percentile).
    3. Static symmetric range ±fluct_amp (or [0, fluct_amp] for magnitude).
    """
    if fixed_vmin is not None and fixed_vmax is not None:
        return fixed_vmin, fixed_vmax
    if dynamic:
        vmax = max(np.percentile(np.abs(data), 99.5), fluct_amp)
        return (0.0, vmax) if comp == "Bmag" else (-vmax, vmax)
    return (0.0, fluct_amp) if comp == "Bmag" else (-fluct_amp, fluct_amp)


def save_field_image(
    data: np.ndarray,
    comp: str,
    step: int,
    plane: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    out_dir: Path,
) -> Path:
    """
    Render a single 2-D field map and save it as a PNG.

    Parameters
    ----------
    data    : 2-D array (fluctuation or magnitude)
    comp    : "Bx", "By", "Bz", or "Bmag"
    step    : simulation step (used in filename and title)
    plane   : "xy", "xz", or "yz"
    xlabel  : horizontal axis label
    ylabel  : vertical axis label
    vmin    : lower color-scale limit
    vmax    : upper color-scale limit
    out_dir : directory for the output file

    Returns
    -------
    Path of the saved PNG.
    """
    if comp == "Bmag":
        title  = f"|δB| fluctuation – step {step}, plane {plane}"
        cb_lbl = r"$|\delta B| / B_0$"
        fname  = f"Bmag_fluct_step{step}_{plane}.png"
    else:
        title  = f"{comp} fluctuation – step {step}, plane {plane}"
        cb_lbl = rf"$\delta {comp} / B_0$"
        fname  = f"{comp}fluct_step{step}_{plane}.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data.T, origin="lower", cmap=COLORMAPS[comp], vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax).set_label(cb_lbl, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)

    out_path = out_dir / fname
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Section 6 – Animation
# =============================================================================

def create_gif(image_dir: Path, comp: str, plane: str, duration: float) -> Path | None:
    """
    Assemble a GIF from all PNG snapshots of a given component.

    Expects files named like:
        Bmag_fluct_step<N>_<plane>.png   (magnitude)
        <comp>fluct_step<N>_<plane>.png  (components)

    Parameters
    ----------
    image_dir : directory containing the PNG files
    comp      : "Bx", "By", "Bz", or "Bmag"
    plane     : slice plane label (e.g. "xy")
    duration  : seconds per frame

    Returns
    -------
    Path to the created GIF, or None if no images were found.
    """
    pattern = f"Bmag_fluct_step*_{plane}.png" if comp == "Bmag" else f"{comp}fluct_step*_{plane}.png"
    files = sorted(
        image_dir.glob(pattern),
        key=lambda p: int(re.search(r"step(\d+)_", p.name).group(1)),
    )
    if not files:
        print(f"  No images found for {comp} in plane {plane}.")
        return None

    out = image_dir / f"{comp}_fluct_{plane}.gif"
    with imageio.get_writer(out, mode="I", duration=duration, loop=0) as writer:
        for f in files:
            try:
                writer.append_data(imageio.imread(f))
            except Exception as e:
                print(f"  Skipping {f.name}: {e}")

    print(f"  GIF saved: {out}")
    return out


# =============================================================================
# Section 7 – Orchestrator
# =============================================================================

class FieldPlotter:
    """
    End-to-end pipeline: discover files → load data → process → plot → (GIF).

    Parameters
    ----------
    pattern       : glob pattern for HDF5 input files
    B0            : reference magnetic field (normalization denominator, ≠ 0)
    fluct_amp     : fallback amplitude for static color scaling
    out_dir       : directory where images and GIFs are saved
    smooth        : Gaussian smoothing sigma (0 = disabled)
    dynamic_scale : derive color limits from each snapshot's data
    plot_magnitude: include the |δB| magnitude plot
    components    : which components to plot (subset of ["Bx","By","Bz"])
    fixed_scale   : pre-compute global color limits for consistent animations
    """

    def __init__(
        self,
        pattern: str       = DEFAULTS["pattern"],
        B0: float          = DEFAULTS["B0"],
        fluct_amp: float   = DEFAULTS["fluct_amp"],
        out_dir: str       = DEFAULTS["out_dir"],
        smooth: float      = DEFAULTS["smooth"],
        dynamic_scale: bool = True,
        plot_magnitude: bool = True,
        components: list[str] = None,
        fixed_scale: bool  = False,
    ):
        if B0 == 0:
            raise ValueError("B0 cannot be zero (it is used as a normalization factor).")

        self.pattern       = pattern
        self.B0            = B0
        self.fluct_amp     = fluct_amp
        self.out_dir       = Path(out_dir)
        self.smooth        = smooth
        self.dynamic_scale = dynamic_scale
        self.plot_magnitude = plot_magnitude
        self.components    = components or list(DEFAULTS["components"])
        self.fixed_scale   = fixed_scale

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file_map = discover_files(pattern)
        self._vmin: dict[str, float] = {}
        self._vmax: dict[str, float] = {}

        if not self.file_map:
            print(f"Warning: no files found matching '{pattern}'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        plane: str          = "xy",
        steps: list[int]    = None,
        slice_idx: int      = None,
        create_gifs: bool   = False,
        gif_duration: float = DEFAULTS["gif_dur"],
    ) -> list[Path]:
        """
        Execute the full visualization pipeline.

        Parameters
        ----------
        plane        : 2-D plane to visualize ("xy", "xz", or "yz")
        steps        : simulation steps to process (default: all found)
        slice_idx    : index along the normal axis (default: midpoint)
        create_gifs  : assemble animated GIFs after plotting
        gif_duration : seconds per frame

        Returns
        -------
        List of paths to all generated PNG files.
        """
        steps = steps or sorted(self.file_map.keys())

        if self.fixed_scale:
            self._compute_global_scales(steps, plane, slice_idx)

        saved = []
        for step in steps:
            print(f"Processing step {step}...")
            saved.extend(self._process_step(step, plane, slice_idx))

        if create_gifs:
            targets = (["Bmag"] if self.plot_magnitude else []) + self.components
            for comp in targets:
                create_gif(self.out_dir, comp, plane, gif_duration)

        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_global_scales(self, steps: list[int], plane: str, slice_idx: int) -> None:
        """
        Scan every step to determine globally consistent color limits.

        Ensures that the same physical amplitude maps to the same color in all
        snapshots – critical for animations to be visually coherent.
        """
        print("Computing global color limits...")
        all_targets = self.components + (["Bmag"] if self.plot_magnitude else [])
        extrema = {t: {"min": [], "max": []} for t in all_targets}

        for step in steps:
            path = self.file_map.get(step)
            if not path:
                continue
            try:
                bx3d, by3d, bz3d = load_field_3d(path, step, self.B0)
                bx, by, bz, _, _ = slice_3d(bx3d, by3d, bz3d, plane, slice_idx)
            except (KeyError, ValueError) as e:
                print(f"  Skipping step {step}: {e}")
                continue

            if self.plot_magnitude:
                mag = fluctuation_magnitude(bx, by, bz, self.B0, self.smooth)
                extrema["Bmag"]["min"].append(mag.min())
                extrema["Bmag"]["max"].append(mag.max())

            for comp, arr in zip(["Bx", "By", "Bz"], [bx, by, bz]):
                if comp in self.components:
                    f = fluctuation(arr, self.B0, self.smooth)
                    extrema[comp]["min"].append(f.min())
                    extrema[comp]["max"].append(f.max())

        for comp, vals in extrema.items():
            if not vals["min"]:
                continue
            vmin = float(np.percentile(vals["min"], 1))
            vmax = float(np.percentile(vals["max"], 99))
            if comp != "Bmag":                     # enforce symmetry for diverging maps
                bound = max(abs(vmin), abs(vmax))
                vmin, vmax = -bound, bound
            self._vmin[comp] = vmin
            self._vmax[comp] = vmax
            print(f"  {comp}: [{vmin:.4f}, {vmax:.4f}]")

    def _process_step(self, step: int, plane: str, slice_idx: int) -> list[Path]:
        """Load, process, and save all plots for a single simulation step."""
        path = self.file_map.get(step)
        if not path:
            print(f"  No file for step {step}, skipping.")
            return []

        try:
            bx3d, by3d, bz3d = load_field_3d(path, step, self.B0)
            bx, by, bz, xl, yl = slice_3d(bx3d, by3d, bz3d, plane, slice_idx)
        except (KeyError, ValueError) as e:
            print(f"  Error at step {step}: {e}")
            return []

        saved = []

        if self.plot_magnitude:
            data = fluctuation_magnitude(bx, by, bz, self.B0, self.smooth)
            vmin, vmax = color_limits(data, "Bmag", self._vmin.get("Bmag"),
                                      self._vmax.get("Bmag"), self.fluct_amp, self.dynamic_scale)
            out = save_field_image(data, "Bmag", step, plane, xl, yl, vmin, vmax, self.out_dir)
            print(f"  Saved: {out.name}")
            saved.append(out)

        for comp, arr in zip(["Bx", "By", "Bz"], [bx, by, bz]):
            if comp not in self.components:
                continue
            data = fluctuation(arr, self.B0, self.smooth)
            vmin, vmax = color_limits(data, comp, self._vmin.get(comp),
                                      self._vmax.get(comp), self.fluct_amp, self.dynamic_scale)
            out = save_field_image(data, comp, step, plane, xl, yl, vmin, vmax, self.out_dir)
            print(f"  Saved: {out.name}")
            saved.append(out)

        return saved


# =============================================================================
# Section 8 – CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize magnetic field fluctuations from PIC simulation HDF5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input / output
    parser.add_argument("--pattern",  default=DEFAULTS["pattern"],  help="Glob pattern for HDF5 files.")
    parser.add_argument("--out_dir",  default=DEFAULTS["out_dir"],  help="Output directory.")
    # Physics
    parser.add_argument("--B0",       type=float, default=DEFAULTS["B0"],        help="Reference field strength.")
    parser.add_argument("--fluct_amp",type=float, default=DEFAULTS["fluct_amp"], help="Default color-scale amplitude.")
    # Slice
    parser.add_argument("--plane",    default="xy", choices=["xy","xz","yz"],    help="Plane to visualize.")
    parser.add_argument("--slice",    type=int, default=None, dest="slice_idx",  help="Normal-axis slice index.")
    parser.add_argument("--steps",    nargs="*", type=int, default=None,         help="Steps to process (default: all).")
    # Processing
    parser.add_argument("--smooth",   type=float, default=DEFAULTS["smooth"],    help="Gaussian smoothing sigma.")
    parser.add_argument("--comps",    nargs="+", default=list(DEFAULTS["components"]),
                        choices=["Bx","By","Bz"],                                help="Components to plot.")
    parser.add_argument("--no_mag",   action="store_true",                       help="Skip |δB| magnitude plot.")
    # Color scaling
    parser.add_argument("--fixed_scale",      action="store_true", help="Lock color limits globally (recommended for GIFs).")
    parser.add_argument("--no_dynamic_scale", action="store_true", help="Use static ±fluct_amp limits.")
    # Animation
    parser.add_argument("--create_gifs",  action="store_true",                       help="Assemble animated GIFs.")
    parser.add_argument("--gif_duration", type=float, default=DEFAULTS["gif_dur"],   help="Seconds per frame in GIF.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plotter = FieldPlotter(
        pattern        = args.pattern,
        B0             = args.B0,
        fluct_amp      = args.fluct_amp,
        out_dir        = args.out_dir,
        smooth         = args.smooth,
        dynamic_scale  = not args.no_dynamic_scale,
        plot_magnitude = not args.no_mag,
        components     = args.comps,
        fixed_scale    = args.fixed_scale,
    )

    plotter.run(
        plane        = args.plane,
        steps        = args.steps,
        slice_idx    = args.slice_idx,
        create_gifs  = args.create_gifs,
        gif_duration = args.gif_duration,
    )


if __name__ == "__main__":
    main()
