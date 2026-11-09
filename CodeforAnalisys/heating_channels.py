"""Periodic-domain energy reservoirs and resolved conversion channels for the thesis."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from energy_exchange import cumulative


def pressure_strain(tensor, dy, dz):
    """Mean -P_ij ∂_j u_i on periodic (z,y) cell-centred arrays, in code units."""
    total = 0.
    for component, py, pz in (("Vx", "Pxy", "Pzx"), ("Vy", "Pyy", "Pyz"),
                              ("Vz", "Pyz", "Pzz")):
        velocity = tensor[component]
        for axis, spacing, pressure in ((1, dy, py), (0, dz, pz)):
            gradient = (np.roll(velocity, -1, axis) - np.roll(velocity, 1, axis)) / (2*spacing)
            total -= np.mean(tensor[pressure]*gradient)
    return float(total)


def heating_figure(run, out):
    """Use one time base, volume and energy-density unit for the whole figure."""
    phys = run / "09_physical_diagnostics"
    exchange = phys / "energy_exchange_table.csv"
    if not exchange.is_file():
        return False
    load = lambda p: np.genfromtxt(p, names=True, delimiter=",", ndmin=1)
    work = load(exchange)
    summary = json.loads((phys / "energy_exchange_summary.json").read_text())
    moment_file = run / "07_structures/snapshot_energy_density.csv"
    moments = load(moment_file) if moment_file.is_file() else None
    if moments is not None:
        _, iw, im = np.intersect1d(work["step"], moments["step"], return_indices=True)
        work, moments = work[iw], moments[im]
    global_file = phys/"global_energy_table.csv"
    glob = load(global_file) if global_file.is_file() else None
    volume = summary.get("volume_from_E_B0")
    if glob is not None and volume and glob["time_code"][0] == 0:
        ok = (work["time_code"] >= glob["time_code"][0]) & (work["time_code"] <= glob["time_code"][-1])
        work = work[ok]
        if moments is not None:
            moments = moments[ok]
    else:
        glob = None
    if len(work) < 2:
        return False
    time, t = work["time_code"], work["omega_ci_t"]
    delta = lambda a: a-a[0]
    data = {"omega_ci_t": t, "time_code": time}
    for s in ("e","i"):
        data["W_"+s] = delta(work["W_total_"+s])
    if glob is not None:
        for k in ("E_E","E_B","E_e","E_i","E_total"):
            data["delta_"+k] = delta(np.interp(time,glob["time_code"],glob[k])) / volume
        data["EM_closure_residual"] = data["delta_E_E"]+data["delta_E_B"]+data["W_e"]+data["W_i"]
    if moments is not None:
        for k in ("E_B","E_E","total_moment_approx"):
            data["moment_delta_"+k] = delta(moments[k])
        for s in ("e","i"):
            data["delta_bulk_"+s] = delta(moments["bulk_"+s])
            data["delta_internal_"+s] = delta(moments["thermal_parallel_"+s]+moments["thermal_perp_"+s])
            key = "pressure_strain_"+s
            if key in moments.dtype.names:
                data["Q_"+s] = cumulative(time,moments[key])
                data["internal_residual_"+s] = data["delta_internal_"+s]-data["Q_"+s]
                data["bulk_residual_"+s] = data["delta_bulk_"+s]-data["W_"+s]+data["Q_"+s]
        if glob is None:
            data["EM_closure_residual"] = data["moment_delta_E_E"]+data["moment_delta_E_B"]+data["W_e"]+data["W_i"]
    out.mkdir(parents=True, exist_ok=True)
    with (out/"heating_channels.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(data); w.writerows(zip(*data.values()))
    fig, axes = plt.subplots(2,2,figsize=(11,7.5),sharex=True,layout="constrained")
    def panel(ax,title,items):
        for key,label in items:
            if key in data:
                ax.plot(t,data[key],label=label)
        ax.set_title(title)
        ax.axhline(0,color=".7",lw=.6)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,"Faltan salidas de dominio",ha="center",transform=ax.transAxes)
    reservoirs = [(f"delta_{k}",label) for k,label in zip(("E_E","E_B","E_i","E_e","E_total"),("Eléctrica","Magnética","Iones","Electrones","Residuo global"))]
    if glob is None:
        reservoirs = [("moment_delta_E_E","Eléctrica"),("moment_delta_E_B","Magnética"),("moment_delta_total_moment_approx","Residuo por momentos")]
        reservoirs += [(f"delta_{k}_{s}",f"{label} {s}") for s in ("i","e") for k,label in (("bulk","Flujo"),("internal","Interna"))]
    panel(axes[0,0],"Cambios de energía del dominio",reservoirs)
    panel(axes[0,1],"Trabajo electromagnético muestreado", [("W_i","∫ Jᵢ·E dt"),("W_e","∫ Jₑ·E dt"),("EM_closure_residual","ΔEM + Σ W")])
    for ax,s,label in zip(axes[1],("e","i"),("Electrones","Iones")):
        panel(ax,label,[(f"delta_E_{s}","ΔK global"),(f"W_{s}","∫ J·E dt"),(f"delta_bulk_{s}","ΔK flujo"),(f"delta_internal_{s}","ΔU interna"),(f"Q_{s}","∫ −P:∇u dt")])
        ax.set_xlabel("t Ωci")
    fig.supylabel("Cambio de energía por volumen (unidades del código)")
    cadence=", ".join(s for s in ("i","e") if summary.get("aliased_"+s))
    fig.suptitle(run.name + (f" · J·E submuestreado ({cadence})" if cadence else ""))
    fig.savefig(out/"heating_channels.png",dpi=160)
    plt.close(fig)
    return True
