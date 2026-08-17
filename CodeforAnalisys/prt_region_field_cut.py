#!/usr/bin/env python3
"""
prt_region_field_cut.py — Localización de la ventana de partículas (prt)
========================================================================
Responde dos preguntas antes de analizar la VDF dentro de los magnetic holes:

  1. ¿Qué región del dominio se está guardando en los archivos prt?
     PSC sólo escribe partículas de la caja de celdas [lo, hi) fijada en
     `OutputParticlesParams` (ver `src/psc_anisotropy_case.hxx`). La ventana
     real se lee de los atributos `lo`/`hi` del propio archivo prt; si no hay
     archivo prt disponible se usa el fallback de `psc_units`.

  2. ¿Cómo se ve el campo magnético a lo largo de una línea que atraviesa esa
     región? Se genera un corte 1D de |B|, B_par y B_perp sobre la línea, con
     la ventana prt sombreada, junto al mapa 2D de |B|/B0 con la caja marcada.

Convención de ejes (igual que el resto del pipeline): los arrays PSC llegan
como (Nz, Ny); los mapas se dibujan con Z horizontal e Y vertical. Por lo
tanto el corte "horizontal" (--cut-axis z, por defecto) recorre Z a Y fijo,
es decir la dirección paralela a B0.

Uso típico:
    python prt_region_field_cut.py --data-dir ../build/src --outdir salida
    python prt_region_field_cut.py --data-dir /ruta/corrida --steps 300000 600000
    python prt_region_field_cut.py --data-dir /ruta/corrida --average-over-window
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import plot_style as ps

ps.apply()

from data_reader import PICDataReader
from psc_units import (
    B0, DOMAIN_DI_Y, DOMAIN_DI_Z, MASS_RATIO, N_GRID_Y, N_GRID_Z,
    PRT_OUTPUT_LO, PRT_OUTPUT_HI, PROFILE_LABEL, step_to_omegaci,
)

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

DPI = 200
D_I = np.sqrt(MASS_RATIO)   # longitud inercial iónica en unidades de d_e


# ── Ventana prt ──────────────────────────────────────────────────────────────

def read_prt_window(particle_file: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Devuelve (lo, hi, origen) de la ventana de salida de partículas.

    `lo`/`hi` son índices de celda globales, con `hi` exclusivo (ver
    `output_particles_hdf5_impl.hxx`: wdims = hi - lo, bucles `< ihi`).

    La ventana se lee siempre del archivo cuando existe: no es constante entre
    corridas. `psc_anisotropy_case.hxx` usa hoy [0.4, 0.6)*ngrid (20 % por
    eje), pero hay datos escritos por ejecutables previos con [0.3, 0.7)
    (40 % por eje). Confiar en el fallback ubicaría la caja en el lugar
    equivocado sin ningún síntoma visible, así que se avisa del desacuerdo.
    """
    fallback_lo = np.asarray(PRT_OUTPUT_LO, dtype=int)
    fallback_hi = np.asarray(PRT_OUTPUT_HI, dtype=int)

    if particle_file:
        with h5py.File(particle_file, "r") as handle:
            group = handle["particles"]
            lo = np.asarray(group.attrs["lo"], dtype=int)
            hi = np.asarray(group.attrs["hi"], dtype=int)
        if not (np.array_equal(lo, fallback_lo) and np.array_equal(hi, fallback_hi)):
            print(f"[WARN] la ventana prt del archivo (lo={list(lo)}, hi={list(hi)}) "
                  f"no coincide con psc_units (lo={list(fallback_lo)}, "
                  f"hi={list(fallback_hi)}). Se usa la del archivo; revisa que el "
                  f"perfil PSC_PROFILE corresponda a estos datos.")
        return lo, hi, f"atributos de {Path(particle_file).name}"

    return (fallback_lo, fallback_hi,
            "psc_units.PRT_OUTPUT_LO/HI (sin archivo prt)")


def describe_window(lo, hi, ny: int, nz: int, dy_di: float, dz_di: float) -> dict:
    """Resume la ventana prt en celdas, fracción de dominio y d_i."""
    ny_win, nz_win = int(hi[1] - lo[1]), int(hi[2] - lo[2])
    return {
        "cells_y": [int(lo[1]), int(hi[1])],
        "cells_z": [int(lo[2]), int(hi[2])],
        "n_cells_y": ny_win,
        "n_cells_z": nz_win,
        "fraction_of_domain_y": ny_win / ny,
        "fraction_of_domain_z": nz_win / nz,
        "fraction_of_area": (ny_win / ny) * (nz_win / nz),
        "y_di": [lo[1] * dy_di, hi[1] * dy_di],
        "z_di": [lo[2] * dz_di, hi[2] * dz_di],
        "size_di": [ny_win * dy_di, nz_win * dz_di],
        "center_di": [0.5 * (lo[1] + hi[1]) * dy_di, 0.5 * (lo[2] + hi[2]) * dz_di],
    }


# ── Geometría de la grilla ───────────────────────────────────────────────────

def read_grid(field_file: str) -> dict:
    """Extrae la grilla del propio snapshot; cae a psc_units si no está."""
    try:
        crds = PICDataReader.read_multiple_fields_3d(
            field_file, "crd[1]", ["crd[1]/p0/1d"])
        y_code = np.asarray(crds["crd[1]/p0/1d"], dtype=float)
        crds = PICDataReader.read_multiple_fields_3d(
            field_file, "crd[2]", ["crd[2]/p0/1d"])
        z_code = np.asarray(crds["crd[2]/p0/1d"], dtype=float)
    except (KeyError, OSError):
        y_code = z_code = None

    if y_code is None or y_code.size < 2:
        ny, nz = N_GRID_Y, N_GRID_Z
        dy_di, dz_di = DOMAIN_DI_Y / ny, DOMAIN_DI_Z / nz
        return {"ny": ny, "nz": nz, "dy_di": dy_di, "dz_di": dz_di,
                "Ly_di": DOMAIN_DI_Y, "Lz_di": DOMAIN_DI_Z, "source": "psc_units"}

    ny, nz = y_code.size, z_code.size
    dy_di = float(np.diff(y_code).mean()) / D_I
    dz_di = float(np.diff(z_code).mean()) / D_I
    return {"ny": ny, "nz": nz, "dy_di": dy_di, "dz_di": dz_di,
            "Ly_di": ny * dy_di, "Lz_di": nz * dz_di, "source": "snapshot"}


# ── Campos ───────────────────────────────────────────────────────────────────

def read_b_field(field_file: str) -> dict:
    """Lee (Bx, By, Bz) como arrays 2D (Nz, Ny)."""
    fields = PICDataReader.read_multiple_fields_3d(
        field_file, "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
    return {
        "bx": PICDataReader.flatten_2d_slice(fields["hx_fc/p0/3d"]),
        "by": PICDataReader.flatten_2d_slice(fields["hy_fc/p0/3d"]),
        "bz": PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]),
    }


def extract_cut(b: dict, grid: dict, lo, hi, cut_axis: str,
                cut_position_di: float | None, average: bool) -> dict:
    """Perfil 1D a lo largo de la línea de corte.

    cut_axis == "z": línea horizontal (Z variable, Y fijo) → paralela a B0.
    cut_axis == "y": línea vertical   (Y variable, Z fijo) → perpendicular.
    Con `average` se promedia sobre el ancho de la ventana prt en el eje fijo,
    lo que reduce el ruido PIC sin salirse de la región muestreada.
    """
    # Los arrays son (Nz, Ny): eje 0 = Z, eje 1 = Y.
    if cut_axis == "z":
        fixed_axis, fixed_d = 1, grid["dy_di"]        # Y fijo
        fixed_lo, fixed_hi = int(lo[1]), int(hi[1])
        along_d, along_n = grid["dz_di"], grid["nz"]
        win_lo_di, win_hi_di = lo[2] * grid["dz_di"], hi[2] * grid["dz_di"]
    else:
        fixed_axis, fixed_d = 0, grid["dz_di"]        # Z fijo
        fixed_lo, fixed_hi = int(lo[2]), int(hi[2])
        along_d, along_n = grid["dy_di"], grid["ny"]
        win_lo_di, win_hi_di = lo[1] * grid["dy_di"], hi[1] * grid["dy_di"]

    if cut_position_di is None:
        index = (fixed_lo + fixed_hi) // 2             # centro de la ventana prt
    else:
        index = int(np.clip(round(cut_position_di / fixed_d - 0.5), 0,
                            (grid["ny"] if fixed_axis == 1 else grid["nz"]) - 1))

    inside_window = fixed_lo <= index < fixed_hi

    def sample(arr: np.ndarray) -> np.ndarray:
        if average:
            sl = [slice(None), slice(None)]
            sl[fixed_axis] = slice(fixed_lo, fixed_hi)
            return arr[tuple(sl)].mean(axis=fixed_axis)
        return arr.take(index, axis=fixed_axis)

    bx, by, bz = sample(b["bx"]), sample(b["by"]), sample(b["bz"])
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    # B0 está a lo largo de Z, así que B_par = Bz y B_perp = sqrt(Bx^2 + By^2).
    bperp = np.sqrt(bx**2 + by**2)

    return {
        "coord_di": (np.arange(along_n) + 0.5) * along_d,
        "bmag": bmag, "bx": bx, "by": by, "bz": bz, "bperp": bperp,
        "index": index, "position_di": (index + 0.5) * fixed_d,
        "inside_window": inside_window,
        "window_di": (win_lo_di, win_hi_di),
        "averaged": average,
    }


# ── Figura ───────────────────────────────────────────────────────────────────

def plot_step(b: dict, grid: dict, cut: dict, window: dict, lo, hi,
              step: int, cut_axis: str, b0: float, outdir: Path,
              prefix: str) -> Path:
    bmag2d = np.sqrt(b["bx"]**2 + b["by"]**2 + b["bz"]**2) / b0
    # La fluctuación es lo que revela huecos y picos; |B|/B0 crudo los aplana
    # contra el fondo porque todo el mapa vive cerca de 1.
    delta2d = bmag2d - 1.0
    extent = [0.0, grid["Lz_di"], 0.0, grid["Ly_di"]]
    toci = step_to_omegaci(step)

    z0, z1 = window["z_di"]
    y0, y1 = window["y_di"]

    fig = plt.figure(figsize=(15.0, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          hspace=0.26, wspace=0.24)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_cut = fig.add_subplot(gs[1, :])

    # ── Panel (a): dominio completo en fluctuación, con la caja prt ─────────
    lim = float(np.percentile(np.abs(delta2d), 99))
    lim = lim if lim > 0 else 1e-6
    im = ax_map.imshow(delta2d.T, origin="lower", cmap="RdBu_r",
                       vmin=-lim, vmax=lim, aspect="equal", extent=extent)
    cb = fig.colorbar(im, ax=ax_map, pad=0.02, aspect=28)
    cb.set_label(r"$\delta |B| / B_0$", fontsize=15, labelpad=2)

    ax_map.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, fill=False,
                               edgecolor=ps.c("#00c000"), lw=2.4, ls="--",
                               label="prt window"))
    if cut_axis == "z":
        ax_map.plot([0, grid["Lz_di"]], [cut["position_di"]] * 2,
                    color=ps.TEXT_CLR, lw=1.6, ls="-", label="cut line")
    else:
        ax_map.plot([cut["position_di"]] * 2, [0, grid["Ly_di"]],
                    color=ps.TEXT_CLR, lw=1.6, ls="-", label="cut line")

    ax_map.set_xlabel(r"$Z$  [$d_i$]  (parallel to $B_0$)")
    ax_map.set_ylabel(r"$Y$  [$d_i$]  (perpendicular)")
    ax_map.set_title("Full domain", pad=8)
    ax_map.legend(loc="upper right", framealpha=0.85)
    ax_map.tick_params(direction="in", which="both", top=True, right=True)

    # ── Panel (b): sólo lo que se guarda, a escala propia ───────────────────
    iy0, iy1 = window["cells_y"]
    iz0, iz1 = window["cells_z"]
    delta_win = delta2d[iz0:iz1, iy0:iy1]
    lim_w = float(np.percentile(np.abs(delta_win), 99))
    lim_w = lim_w if lim_w > 0 else 1e-6
    im = ax_zoom.imshow(delta_win.T, origin="lower", cmap="RdBu_r",
                        vmin=-lim_w, vmax=lim_w, aspect="equal",
                        extent=[z0, z1, y0, y1])
    cb = fig.colorbar(im, ax=ax_zoom, pad=0.02, aspect=28)
    cb.set_label(r"$\delta |B| / B_0$", fontsize=15, labelpad=2)
    ax_zoom.set_xlabel(r"$Z$  [$d_i$]")
    ax_zoom.set_ylabel(r"$Y$  [$d_i$]")
    ax_zoom.set_title(
        rf"prt window: {window['size_di'][1]:.1f} x {window['size_di'][0]:.1f} $d_i$"
        rf"  ({100 * window['fraction_of_area']:.1f} % of the area)", pad=8)
    ax_zoom.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        rf"{PROFILE_LABEL} — step {step}, $t \approx {toci:.1f}\,\Omega_{{ci}}^{{-1}}$",
        y=0.95, fontsize=20)

    # ── Panel inferior: corte 1D, en fluctuaciones ──────────────────────────
    # |B| y B_par viven cerca de B0 y B_perp cerca de 0: dibujarlos crudos deja
    # tres líneas planas separadas por una década de espacio vacío. Restando la
    # línea base de cada uno, los huecos y picos quedan a la misma escala.
    coord = cut["coord_di"]
    ax_cut.plot(coord, cut["bmag"] / b0 - 1.0, color=ps.c("#111111"), lw=2.0,
                label=r"$\delta |B|/B_0$")
    ax_cut.plot(coord, cut["bz"] / b0 - 1.0, color=ps.c("#1f77b4"), lw=1.5, alpha=0.9,
                label=r"$\delta B_\parallel/B_0$")
    ax_cut.plot(coord, cut["bperp"] / b0, color=ps.c("#d62728"), lw=1.5, alpha=0.9,
                label=r"$B_\perp/B_0$")
    ax_cut.axhline(0.0, color="gray", lw=1.0, ls=":")

    w0, w1 = cut["window_di"]
    ax_cut.axvspan(w0, w1, color="red", alpha=0.12)
    for edge in (w0, w1):
        ax_cut.axvline(edge, color="red", lw=1.6, ls="--")
    ax_cut.text(0.5 * (w0 + w1), ax_cut.get_ylim()[1], " prt region ",
                color="red", ha="center", va="top", fontsize=13)

    along_label = r"$Z$  [$d_i$]" if cut_axis == "z" else r"$Y$  [$d_i$]"
    fixed_label = "Y" if cut_axis == "z" else "Z"
    if cut["averaged"]:
        where = rf"averaged over ${fixed_label}$ inside the prt window"
    else:
        where = rf"${fixed_label} = {cut['position_di']:.2f}\,d_i$"
    ax_cut.set_xlabel(along_label)
    ax_cut.set_ylabel(r"$\delta B / B_0$")
    ax_cut.set_title(
        ("Horizontal cut" if cut_axis == "z" else "Vertical cut") +
        f" — {where}", fontsize=17, pad=8)
    ax_cut.legend(loc="best", framealpha=0.85, ncol=3)
    ax_cut.tick_params(direction="in", which="both", top=True, right=True)
    ax_cut.set_xlim(coord[0], coord[-1])

    outname = outdir / f"{prefix}prt_region_cut_{cut_axis}_step{step:06d}.png"
    ps.save(fig, outname)
    plt.close(fig)
    return outname


def write_cut_csv(cut: dict, b0: float, step: int, cut_axis: str,
                  outdir: Path, prefix: str) -> Path:
    outname = outdir / f"{prefix}prt_region_cut_{cut_axis}_step{step:06d}.csv"
    axis_name = "z_di" if cut_axis == "z" else "y_di"
    w0, w1 = cut["window_di"]
    with open(outname, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([axis_name, "B_over_B0", "Bpar_over_B0",
                         "Bperp_over_B0", "Bx_over_B0", "By_over_B0",
                         "inside_prt_window"])
        for i, coord in enumerate(cut["coord_di"]):
            writer.writerow([
                f"{coord:.6f}",
                f"{cut['bmag'][i] / b0:.6f}",
                f"{cut['bz'][i] / b0:.6f}",
                f"{cut['bperp'][i] / b0:.6f}",
                f"{cut['bx'][i] / b0:.6f}",
                f"{cut['by'][i] / b0:.6f}",
                int(w0 <= coord < w1),
            ])
    return outname


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = PICDataReader.discover_outputs(args.data_dir)
    field_map = outputs["fields"]
    if not field_map:
        print(f"ERROR: no se encontraron archivos pfd.*.h5 en {args.data_dir}")
        return 1

    particle_file = args.particles
    if particle_file is None and outputs["particles"]:
        series = next(iter(outputs["particles"].values()))
        particle_file = series[max(series)]

    lo, hi, source = read_prt_window(particle_file)

    steps = sorted(field_map)
    if args.steps:
        steps = [s for s in args.steps if s in field_map]
        missing = sorted(set(args.steps) - set(field_map))
        if missing:
            print(f"[WARN] steps sin snapshot de campos: {missing}")
    elif args.max_snapshots and len(steps) > args.max_snapshots:
        picks = np.linspace(0, len(steps) - 1, args.max_snapshots).round().astype(int)
        steps = [steps[i] for i in dict.fromkeys(picks)]
    if not steps:
        print("ERROR: ningún step seleccionado.")
        return 1

    grid = read_grid(field_map[steps[0]])
    window = describe_window(lo, hi, grid["ny"], grid["nz"],
                             grid["dy_di"], grid["dz_di"])

    print(f"Perfil:            {PROFILE_LABEL}")
    print(f"Grilla:            {grid['nz']} x {grid['ny']} (Z x Y), "
          f"{grid['Lz_di']:.2f} x {grid['Ly_di']:.2f} d_i  [{grid['source']}]")
    print(f"Ventana prt de:    {source}")
    print(f"  celdas Y:        [{window['cells_y'][0]}, {window['cells_y'][1]})  "
          f"= {window['n_cells_y']} celdas "
          f"({100 * window['fraction_of_domain_y']:.1f} % del dominio)")
    print(f"  celdas Z:        [{window['cells_z'][0]}, {window['cells_z'][1]})  "
          f"= {window['n_cells_z']} celdas "
          f"({100 * window['fraction_of_domain_z']:.1f} % del dominio)")
    print(f"  extensión Y:     [{window['y_di'][0]:.2f}, {window['y_di'][1]:.2f}] d_i")
    print(f"  extensión Z:     [{window['z_di'][0]:.2f}, {window['z_di'][1]:.2f}] d_i")
    print(f"  tamaño:          {window['size_di'][0]:.2f} x {window['size_di'][1]:.2f} d_i "
          f"({100 * window['fraction_of_area']:.2f} % del área total)")
    print(f"Steps a procesar:  {len(steps)}")

    summary = {
        "profile": PROFILE_LABEL,
        "data_dir": str(Path(args.data_dir).resolve()),
        "particle_file": particle_file,
        "window_source": source,
        "grid": grid,
        "prt_window": window,
        "cut_axis": args.cut_axis,
        "averaged_over_window": bool(args.average_over_window),
        "steps": steps,
    }

    for step in steps:
        b = read_b_field(field_map[step])
        cut = extract_cut(b, grid, lo, hi, args.cut_axis,
                          args.cut_position, args.average_over_window)
        if not (cut["averaged"] or cut["inside_window"]):
            print(f"[WARN] step {step}: la línea de corte cae FUERA de la ventana prt.")
        png = plot_step(b, grid, cut, window, lo, hi, step, args.cut_axis,
                        args.B0, outdir, args.prefix)
        csv_path = write_cut_csv(cut, args.B0, step, args.cut_axis,
                                 outdir, args.prefix)
        print(f"  step {step:>8}: {png.name}  |  {csv_path.name}")

    summary_path = outdir / f"{args.prefix}prt_region_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Resumen: {summary_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ubica la ventana de salida prt y grafica B a lo largo de "
                    "una línea que la atraviesa.")
    parser.add_argument("--data-dir", default="../build/src",
                        help="directorio con pfd.*.h5 y prt*.h5")
    parser.add_argument("--particles", default=None,
                        help="archivo prt específico del que leer lo/hi "
                             "(por defecto el último de --data-dir)")
    parser.add_argument("--outdir", default="prt_region_plots")
    parser.add_argument("--prefix", default="",
                        help="prefijo para los nombres de archivo de salida")
    parser.add_argument("--steps", nargs="*", type=int,
                        help="steps concretos; por defecto se muestrean")
    parser.add_argument("--max-snapshots", type=int, default=6,
                        help="número de steps a muestrear si no se dan --steps")
    parser.add_argument("--cut-axis", choices=["z", "y"], default="z",
                        help="z = línea horizontal (paralela a B0), "
                             "y = línea vertical (perpendicular)")
    parser.add_argument("--cut-position", type=float, default=None,
                        help="posición en d_i del eje fijo; por defecto el "
                             "centro de la ventana prt")
    parser.add_argument("--average-over-window", action="store_true",
                        help="promedia el perfil sobre el ancho de la ventana "
                             "prt en el eje fijo (menos ruido PIC)")
    parser.add_argument("--B0", type=float, default=B0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
