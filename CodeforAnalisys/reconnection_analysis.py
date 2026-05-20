"""
reconnection_analysis.py
========================
Análisis completo de la simulación de reconexión magnética local.
Lee archivos HDF5 de psc_reconnection_local y genera:
  1. Panel de reconexión (Bz, Jx, densidad iones) para 4 tiempos
  2. Líneas de campo magnético (By, Bz) con O-point y X-point
  3. Corriente de reconexión Jx en el plano YZ
  4. Evolución temporal del flujo magnético reconectado
"""

import h5py
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os

# ── Parámetros físicos (deben coincidir con psc_reconnection_local.cxx) ──
MASS_RATIO  = 25.0
WPE_WCE     = 2.0
TI_TE       = 5.0
LY_DI       = 10.0
LZ_DI       = 40.0
L_DI        = 0.5
NB_N0       = 0.05
DBY_B0      = 0.03

me, ec, c, eps0 = 1., 1., 1., 1.
TTe  = me*c**2 / (2.*eps0*WPE_WCE**2*(1.+TI_TE))
TTi  = TTe * TI_TE
wci  = 1./(MASS_RATIO*WPE_WCE)
wce  = wci * MASS_RATIO
wpe  = wce * WPE_WCE
wpi  = wpe / np.sqrt(MASS_RATIO)
d_i  = c / wpi
b0   = me*c*wce/ec
L    = L_DI * d_i

# ── Directorio de datos ────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "reconnection_local_output"
OUTPUT_DIR = Path(__file__).parent / "reconnection_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Paletas premium ───────────────────────────────────────────────────
CMAP_B   = "RdBu_r"       # Campo magnético: rojo/azul
CMAP_J   = "seismic"      # Corriente: divergente
CMAP_N   = "inferno"      # Densidad: secuencial cálido
CMAP_PHI = "plasma"       # Flujo
BG       = "#0d1117"
FG       = "#e6edf3"
ACCENT   = "#58a6ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#30363d",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════════
# Helpers de lectura HDF5
# ══════════════════════════════════════════════════════════════════════

def get_group_prefix(f: h5py.File, prefix: str) -> str:
    for k in f.keys():
        if k.startswith(prefix):
            return k
    raise KeyError(f"No group starting with '{prefix}' in {f.filename}")


# Mapeo de nombres lógicos → nombres HDF5 en PSC
FIELD_NAMES = {
    "hx": "hx_fc", "hy": "hy_fc", "hz": "hz_fc",
    "ex": "ex_ec", "ey": "ey_ec", "ez": "ez_ec",
    "jx": "jx_ec", "jy": "jy_ec", "jz": "jz_ec",
}

def read_field(filepath: str, field: str) -> np.ndarray:
    """Lee un campo del archivo pfd.*.h5 → array 2D (nz, ny)."""
    hdf_name = FIELD_NAMES.get(field, field)
    with h5py.File(filepath, "r") as f:
        grp = get_group_prefix(f, "jeh-uid")
        ds  = f[f"{grp}/{hdf_name}/p0/3d"][()]   # shape (nz, ny, 1)
    return ds.squeeze()   # → (nz, ny)


def read_moment(filepath: str, moment: str) -> np.ndarray:
    """Lee un momento del archivo pfd_moments.*.h5 → array 2D (nz, ny)."""
    with h5py.File(filepath, "r") as f:
        grp = get_group_prefix(f, "all_1st_cc-uid")
        ds  = f[f"{grp}/{moment}/p0/3d"][()]
    return ds.squeeze()


def read_coords(filepath: str):
    """Devuelve (y_cc, z_cc) en unidades de d_i (cell-centered)."""
    with h5py.File(filepath, "r") as f:
        ky = get_group_prefix(f, "crd[1]-uid")
        kz = get_group_prefix(f, "crd[2]-uid")
        y  = f[f"{ky}/crd[1]/p0/1d"][()]
        z  = f[f"{kz}/crd[2]/p0/1d"][()]
    return y / d_i, z / d_i   # → en d_i


def get_steps(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    out   = {}
    for fp in files:
        m = re.search(r"\.(\d+)_p", fp)
        if m:
            out[int(m.group(1))] = fp
    return out


# ══════════════════════════════════════════════════════════════════════
# Función auxiliar de plot 2D
# ══════════════════════════════════════════════════════════════════════

def plot2d(ax, data, y, z, cmap, label, vmax=None, symmetric=False):
    """data shape: (nz, ny). Ploteamos z en X e y en Y."""
    vmax = vmax or np.nanpercentile(np.abs(data), 99)
    vmin = -vmax if symmetric else 0.
    # pcolormesh espera (X, Y) = (z, y) para que z quede en eje x
    im = ax.pcolormesh(z, y, data.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(label, color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    ax.set_xlabel(r"$z \; [d_i]$", fontsize=9)
    ax.set_ylabel(r"$y \; [d_i]$", fontsize=9)
    ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)
    return im


def overlay_fieldlines(ax, By, Bz, y, z, nlines=18):
    """Superpone líneas de campo. By,Bz shape (nz,ny); y (ny,), z (nz,)."""
    # Función de flujo A_x: int Bz dz  (integración sobre z)
    dz = z[1] - z[0] if len(z) > 1 else 1.
    dy = y[1] - y[0] if len(y) > 1 else 1.
    # Ax(z,y) = int_0^z Bz dz'  - simplificado
    Ax = np.cumsum(Bz, axis=0) * dz   # shape (nz, ny)
    vmin_ax, vmax_ax = Ax.min(), Ax.max()
    levels = np.linspace(vmin_ax * 0.85, vmax_ax * 0.85, nlines)
    # contour espera (X=z, Y=y, Z=Ax.T) donde Ax.T tiene shape (ny, nz)
    Z2d, Y2d = np.meshgrid(z, y)   # each (ny, nz)
    ax.contour(Z2d, Y2d, Ax.T, levels=levels, colors="white",
               linewidths=0.5, alpha=0.55)


# ══════════════════════════════════════════════════════════════════════
# FIGURA 1 — Panel 4-tiempos: Bz, Jx, densidad total
# ══════════════════════════════════════════════════════════════════════

def plot_reconnection_panel(pfd_steps: dict, mom_steps: dict):
    steps_sorted = sorted(pfd_steps.keys())
    # Elegir 4 tiempos representativos
    idxs  = [0, len(steps_sorted)//3, 2*len(steps_sorted)//3, -1]
    sel   = [steps_sorted[i] for i in idxs]

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("Reconexión Magnética — Hoja de Harris  |  PSC Local",
                 color=FG, fontsize=14, fontweight="bold", y=0.97)

    # Lee coordenadas del primer archivo
    y_di, z_di = read_coords(pfd_steps[sel[0]])

    for col, step in enumerate(sel):
        t_code = step * 0.547  # dt ≈ 0.547 (aproximado del output)
        t_wci  = t_code * wci

        # ── Bz ──────────────────────────────────────────────────────
        Bz = read_field(pfd_steps[step], "hz") / b0
        By = read_field(pfd_steps[step], "hy") / b0

        ax = fig.add_subplot(3, 4, col + 1)
        plot2d(ax, Bz, y_di, z_di, CMAP_B, r"$B_z/b_0$", symmetric=True)
        overlay_fieldlines(ax, By, Bz, y_di, z_di)
        ax.set_title(f"Paso {step}  |  t·Ωᵢ ≈ {t_wci:.1f}", color=ACCENT)
        if col > 0: ax.set_ylabel("")
        if col == 0:
            ax.text(-0.22, 0.5, r"$B_z/b_0$", transform=ax.transAxes,
                    color=FG, fontsize=10, rotation=90, va="center")

        # ── Jx (corriente de reconexión) ────────────────────────────
        if step in mom_steps:
            try:
                jx_e = read_moment(mom_steps[step], "jx_e")
                jx_i = read_moment(mom_steps[step], "jx_i")
                jx_ebg = read_moment(mom_steps[step], "jx_e_bg")
                jx_ibg = read_moment(mom_steps[step], "jx_i_bg")
                Jx = jx_e + jx_i + jx_ebg + jx_ibg
            except KeyError:
                Jx = np.zeros_like(Bz)

            ax2 = fig.add_subplot(3, 4, 4 + col + 1)
            plot2d(ax2, Jx, y_di, z_di, CMAP_J, r"$J_x$", symmetric=True)
            overlay_fieldlines(ax2, By, Bz, y_di, z_di)
            if col > 0: ax2.set_ylabel("")
            if col == 0:
                ax2.text(-0.22, 0.5, r"$J_x$ (reconex.)", transform=ax2.transAxes,
                         color=FG, fontsize=10, rotation=90, va="center")

            # ── Densidad total iones ─────────────────────────────────
            try:
                n_i  = np.abs(read_moment(mom_steps[step], "rho_i"))
                n_ibg= np.abs(read_moment(mom_steps[step], "rho_i_bg"))
                n_tot= n_i + n_ibg
            except KeyError:
                n_tot = np.ones_like(Bz)

            ax3 = fig.add_subplot(3, 4, 8 + col + 1)
            plot2d(ax3, n_tot, y_di, z_di, CMAP_N, r"$n_i / n_0$")
            overlay_fieldlines(ax3, By, Bz, y_di, z_di)
            if col > 0: ax3.set_ylabel("")
            if col == 0:
                ax3.text(-0.22, 0.5, r"$n_i / n_0$", transform=ax3.transAxes,
                         color=FG, fontsize=10, rotation=90, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "reconnection_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Panel reconexión → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURA 2 — Evolución temporal del flujo magnético reconectado
# ══════════════════════════════════════════════════════════════════════

def plot_reconnected_flux(pfd_steps: dict):
    steps = sorted(pfd_steps.keys())
    flux  = []
    times_wci = []

    for step in steps:
        Bz = read_field(pfd_steps[step], "hz") / b0
        By = read_field(pfd_steps[step], "hy") / b0

        # Flujo reconectado ≈ max(By) a lo largo de la hoja (y≈0)
        mid = Bz.shape[1] // 2
        by_sheet = By[:, mid]   # By a lo largo de z en y≈0
        flux.append(np.max(np.abs(by_sheet)))

        t_code = step * 0.547
        times_wci.append(t_code * wci)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)

    ax.plot(times_wci, flux, color=ACCENT, lw=2.2, marker="o",
            ms=4, markerfacecolor="#ff7b72", markeredgewidth=0, alpha=0.9)
    ax.fill_between(times_wci, flux, alpha=0.15, color=ACCENT)

    ax.set_xlabel(r"Tiempo $t \cdot \Omega_i$", fontsize=11)
    ax.set_ylabel(r"Flujo reconectado $\delta B_y / b_0$", fontsize=11)
    ax.set_title("Evolución temporal del flujo magnético reconectado", fontsize=12,
                 color=FG, fontweight="bold")
    ax.grid(color="#30363d", alpha=0.5, lw=0.7)

    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    out = OUTPUT_DIR / "reconnected_flux.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Flujo reconectado → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURA 3 — Snapshot final detallado (6 subplots)
# ══════════════════════════════════════════════════════════════════════

def plot_final_snapshot(pfd_steps: dict, mom_steps: dict):
    step = sorted(pfd_steps.keys())[-1]
    y_di, z_di = read_coords(pfd_steps[step])

    Bz = read_field(pfd_steps[step], "hz") / b0
    By = read_field(pfd_steps[step], "hy") / b0
    Bx = read_field(pfd_steps[step], "hx") / b0
    Ex = read_field(pfd_steps[step], "ex")
    Ey = read_field(pfd_steps[step], "ey")
    Ez = read_field(pfd_steps[step], "ez")
    B2 = Bx**2 + By**2 + Bz**2

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(f"Snapshot final — Paso {step}  |  Reconexión Magnética Harris",
                 color=FG, fontsize=13, fontweight="bold", y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    panels = [
        (gs[0, 0], Bz, CMAP_B,  r"$B_z / b_0$",    True),
        (gs[0, 1], By, CMAP_B,  r"$B_y / b_0$",    True),
        (gs[0, 2], B2, CMAP_N,  r"$|B|^2 / b_0^2$", False),
        (gs[1, 0], Ex, CMAP_J,  r"$E_x$",           True),
        (gs[1, 1], Ey, CMAP_J,  r"$E_y$",           True),
        (gs[1, 2], Ez, CMAP_J,  r"$E_z$",           True),
    ]

    for spec, data, cmap, lbl, sym in panels:
        ax = fig.add_subplot(spec)
        plot2d(ax, data, y_di, z_di, cmap, lbl, symmetric=sym)
        overlay_fieldlines(ax, By, Bz, y_di, z_di)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "final_snapshot_fields.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Snapshot campos → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# FIGURA 4 — Corriente Jx y densidades final
# ══════════════════════════════════════════════════════════════════════

def plot_final_currents(pfd_steps: dict, mom_steps: dict):
    step = sorted(mom_steps.keys())[-1]
    y_di, z_di = read_coords(pfd_steps[max(pfd_steps)])

    By = read_field(pfd_steps[max(pfd_steps)], "hy") / b0
    Bz = read_field(pfd_steps[max(pfd_steps)], "hz") / b0

    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    fig.suptitle("Corrientes y Densidades — Snapshot Final", color=FG,
                 fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fields_m = {
        "jx_e": (r"$J_x^e$ (electrones)", CMAP_J, True),
        "jx_i": (r"$J_x^i$ (iones)", CMAP_J, True),
    }

    try:
        jx_e  = read_moment(mom_steps[step], "jx_e")
        jx_i  = read_moment(mom_steps[step], "jx_i")
        jx_ebg= read_moment(mom_steps[step], "jx_e_bg")
        jx_ibg= read_moment(mom_steps[step], "jx_i_bg")
        n_i   = np.abs(read_moment(mom_steps[step], "rho_i"))
        n_ibg = np.abs(read_moment(mom_steps[step], "rho_i_bg"))
        n_e   = np.abs(read_moment(mom_steps[step], "rho_e"))
        n_ebg = np.abs(read_moment(mom_steps[step], "rho_e_bg"))
        Jx_tot= jx_e + jx_i + jx_ebg + jx_ibg
        n_tot_i = n_i + n_ibg
        n_tot_e = n_e + n_ebg

        datasets = [
            (gs[0, 0], jx_e,    CMAP_J, r"$J_x^e$",      True),
            (gs[0, 1], jx_i,    CMAP_J, r"$J_x^i$",      True),
            (gs[0, 2], Jx_tot,  CMAP_J, r"$J_x^{tot}$",  True),
            (gs[1, 0], n_tot_i, CMAP_N, r"$n_i / n_0$",  False),
            (gs[1, 1], n_tot_e, CMAP_N, r"$n_e / n_0$",  False),
            (gs[1, 2], n_tot_i - n_tot_e, CMAP_J,
             r"$n_i - n_e$ (carga)", True),
        ]
    except KeyError as e:
        print(f"  ⚠ Campo de momento no encontrado: {e}")
        return None

    for spec, data, cmap, lbl, sym in datasets:
        ax = fig.add_subplot(spec)
        plot2d(ax, data, y_di, z_di, cmap, lbl, symmetric=sym)
        overlay_fieldlines(ax, By, Bz, y_di, z_di)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "final_currents_density.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ Corrientes/densidad → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  Análisis de Reconexión Magnética — PSC Local")
    print(f"  Datos: {DATA_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Parámetros físicos
    print(f"  d_i = {d_i:.3f}   b0 = {b0:.4f}   L = {L:.3f}")
    print(f"  TTe = {TTe:.4f}   TTi = {TTi:.4f}   wci = {wci:.5f}\n")

    # Encontrar archivos
    pfd_steps  = get_steps(str(DATA_DIR / "pfd.??????_p000000.h5"))
    mom_steps  = get_steps(str(DATA_DIR / "pfd_moments.??????_p000000.h5"))

    print(f"  Campos encontrados:   {len(pfd_steps)} snapshots")
    print(f"  Momentos encontrados: {len(mom_steps)} snapshots")
    print(f"  Pasos: {sorted(pfd_steps.keys())}\n")

    if not pfd_steps:
        print("  ERROR: No se encontraron archivos pfd.*.h5")
        return

    print("  Generando figuras...\n")

    plot_reconnection_panel(pfd_steps, mom_steps)
    plot_reconnected_flux(pfd_steps)
    plot_final_snapshot(pfd_steps, mom_steps)
    plot_final_currents(pfd_steps, mom_steps)

    print(f"\n{'='*60}")
    print(f"  ¡Análisis completado! Imágenes en:")
    print(f"  {OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    plt.show()


if __name__ == "__main__":
    main()
