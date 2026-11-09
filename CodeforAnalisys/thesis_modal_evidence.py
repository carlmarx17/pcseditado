"""Reduce raw per-object and per-mode measurements without assigning a branch."""
import csv
import json

import numpy as np

from structures_analysis import write_csv


def deepest_holes(path):
    """One sample per overlap track at its largest centre-to-exterior depression."""
    selected = {}
    if not path.is_file():
        return []
    with path.open() as stream:
        for r in csv.DictReader(stream):
            if r["kind"] != "hole" or not r.get("contrast_B_over_B0"):
                continue
            d = float(r["contrast_B_over_B0"])
            if not np.isfinite(d) or d >= 0:
                continue
            ident = r["track_id"]
            if ident not in selected or d < float(selected[ident]["contrast_B_over_B0"]):
                selected[ident] = r
    result = []
    for r in selected.values():
        total = float(r["contrast_B_over_B0"])
        row = {k: r[k] for k in ("track_id", "time", "cells", "exterior_cells")}
        row["depth_B0"] = -total
        for k in ("Bz", "transverse", "fundamental_only"):
            row[k + "_fraction"] = float(r[f"contrast_{k}_over_B0"])/total
        if not np.isclose(row["Bz_fraction"]+row["transverse_fraction"], 1., atol=1e-8):
            raise ValueError("Magnetic contrast decomposition does not close")
        for k in ("P_B", "Pperp_i", "Pperp_e", "P_total", "n_i"):
            row["contrast_"+k] = r.get("contrast_"+k)
        result.append(row)
    return result


def complex_fit(t, z):
    """Exploratory single rotating, exponentially growing coefficient."""
    if len(t) < 6 or np.any(~np.isfinite(z)) or np.any(np.abs(z) == 0):
        return {}
    dt = t - t.mean()
    log_amp = np.log(np.abs(z))
    gamma, intercept = np.polyfit(dt, log_amp, 1)
    phase = np.unwrap(np.angle(z))
    slope, phi = np.polyfit(dt, phase, 1)
    span = t[-1]-t[0]
    ss = np.sum((log_amp-log_amp.mean())**2)
    return {"gamma_Bz": gamma, "r2_log_Bz": 1-np.sum((log_amp-gamma*dt-intercept)**2)/ss if ss else None,
            "omega_plasma_fit": -slope, "phase_coherence": abs(np.mean(np.exp(1j*(phase-slope*dt-phi)))),
            "frequency_bin": 2*np.pi/span, "cycles_in_window": abs(slope)*span/(2*np.pi)}


def snapshot_budget(energy, phase, global_file):
    """Intensive, nonrelativistic moment balance against an independent global sum."""
    t = energy["omega_ci_t"]
    reservoirs = ["E_E", "E_B", *(f"{kind}_{s}" for s in ("i", "e") for kind in
                                  ("bulk", "thermal_parallel", "thermal_perp"))]
    if not set(reservoirs).issubset(energy.dtype.names):
        return []
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Invalid snapshot times")
    points = [("first", t[0]), ("last", t[-1])]
    for name in ("start", "end"):
        q = phase.get(name)
        if q is not None and t[0] <= q <= t[-1]:
            points.append(("linear_" + name, q))
    exact = None
    if global_file.is_file():
        exact = np.genfromtxt(global_file, names=True, delimiter=",", ndmin=1)
    result = []
    for name, q in sorted(points, key=lambda item: item[1]):
        row = {"epoch": name, "omega_ci_t": q, "reference_omega_ci_t": t[0]}
        for key in reservoirs:
            row["change_" + key] = np.interp(q, t, energy[key]) - energy[key][0]
        row["moment_residual_density"] = sum(row["change_"+k] for k in reservoirs)
        row["moment_residual_over_initial"] = row["moment_residual_density"] / energy["total_moment_approx"][0]
        if exact is not None:
            gt = exact["omega_ci_t"]
            # Use strict common coverage without extrapolating rounded endpoint times.
            if gt[0] <= t[0] <= q <= gt[-1]:
                ebase = np.interp(t[0], gt, exact["E_total"])
                row["exact_residual_over_initial"] = (np.interp(q, gt, exact["E_total"]) - ebase) / ebase
        result.append(row)
    return result


def reduce_run(run, out):
    """Use the independently selected leading-mode window across the entire oblique lattice."""
    folder = run / "07_structures"
    raw = folder / "attribution_modes.csv"
    if not raw.exists():
        return []
    out.mkdir(parents=True, exist_ok=True)
    objects = deepest_holes(folder/"structure_tracks.csv")
    write_csv(out/"deepest_holes.csv", objects)
    physics = json.loads(next(run.glob("*_analysis_manifest.json")).read_text())["physics"]
    phase = json.loads((run/"09_physical_diagnostics/linear_phase.json").read_text())
    energy = np.genfromtxt(folder/"snapshot_energy_density.csv", delimiter=",", names=True, ndmin=1)
    data = np.genfromtxt(raw, delimiter=",", names=True, ndmin=1)
    write_csv(out/"moment_budget.csv", snapshot_budget(energy, phase, run/"09_physical_diagnostics/global_energy_table.csv"))
    t_full = energy["omega_ci_t"]
    data_keys = set(data.dtype.names)
    moments_present = {"ion_mean_uy", "ion_mean_uz"} <= set(energy.dtype.names)
    evidence = []
    if phase.get("start") is not None and phase.get("end") is not None and moments_present:
        for iz, iy in sorted(set(zip(data["n_parallel"], data["n_perp_signed"]))):
            if not (iz and iy):
                continue
            subset = data[(data["n_parallel"] == iz) & (data["n_perp_signed"] == iy)]
            t = subset["omega_ci_t"]
            if not np.array_equal(t, t_full):
                raise ValueError("Field and moment time axes differ")
            mask = (t >= phase["start"]) & (t <= phase["end"])
            if np.count_nonzero(mask) < 6:
                continue
            gain = 2*np.pi/(physics["domain_di"]*physics["vA_code"])
            ku = gain*(iz*energy["ion_mean_uz"][mask]+iy*energy["ion_mean_uy"][mask])
            times = t[mask]
            advance = np.r_[0, np.cumsum(.5*(ku[:-1]+ku[1:])*np.diff(times))]
            def field(key):
                return (subset[key+"_re"]+1j*subset[key+"_im"])[mask]
            bz = field("Bz_over_B0")
            row = dict(n_parallel=int(iz), n_perp=int(iy), t_start=times[0], t_end=times[-1],
                       samples=len(times), rms_Bz_over_B0=np.sqrt(np.mean(abs(bz)**2)))
            row.update(complex_fit(times, bz*np.exp(1j*advance)))
            if {"P_B_re", "Pperp_i_re", "Pperp_e_re", "n_i_re"} <= data_keys:
                pb, pt, n = field("P_B"), field("Pperp_i")+field("Pperp_e"), field("n_i")
                norm = np.linalg.norm(n)*np.linalg.norm(bz)
                row["n_Bz_normalized_cospectrum"] = np.vdot(n,bz).real/norm if norm else None
                row["rms_total_over_magnetic_pressure"] = np.linalg.norm(pb+pt)/np.linalg.norm(pb) if np.linalg.norm(pb) else None
            evidence.append(row)
    write_csv(out/"oblique_phase.csv", evidence)
    return [f"{len(objects)} trayectorias de agujeros con contraste medible",
            f"{len(evidence)} armónicos oblicuos en la ventana lineal común"]
