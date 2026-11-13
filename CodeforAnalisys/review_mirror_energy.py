"""Build a review packet from archived moderate runs, without mixing vintages."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import energy_audit
from heating_channels import heating_figure
from thesis_modal_evidence import reduce_run


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, data, columns=None):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns if columns is not None else list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def phase_budget(path, phase_bounds):
    data = np.genfromtxt(path, names=True, delimiter=",")
    keys = ("E_E", "E_B", "E_i", "E_e")
    t = data["omega_ci_t"]
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError(f"Invalid time axis: {path}")
    if not np.allclose(sum(data[k] for k in keys), data["E_total"], rtol=1e-8, atol=0):
        raise ValueError(f"Inconsistent reservoirs: {path}")
    intervals = []
    a, b = phase_bounds.get("start"), phase_bounds.get("end")
    if a is not None and b is not None and t[0] < a < b < t[-1]:
        intervals.extend([("previa", t[0], a), ("lineal", a, b), ("posterior", b, t[-1])])
    intervals.append(("completa", t[0], t[-1]))
    result = []
    for phase, a, b in intervals:
        row = dict(phase=phase, t_start=a, t_end=b, baseline_time=t[0])
        changes = [float(np.diff(np.interp([a, b], t, data[k]))[0]) for k in keys]
        row.update(zip(("delta_" + k for k in keys), changes))
        drift = sum(changes)
        row.update(residual=drift, residual_over_E0=drift/data["E_total"][0],
                   rate_over_E0_per_omegaci=drift/data["E_total"][0]/(b-a),
                   residual_over_electron_gain=drift/changes[3] if changes[3] > 0 else None)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--isotropic-control", type=Path, action="append", default=[],
                        help="Archived result directory of a matched isotropic control")
    args = parser.parse_args()
    archive, out = args.archive.resolve(), args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = [archive / f"mirror_{tag}_moderate" for tag in ("bimaxwellian", "bikappa5", "bikappa3")]
    audits = [energy_audit.audit_run(run) for run in runs]
    controls = [energy_audit.audit_run(control) for control in args.isotropic_control]
    if any(not energy_audit.is_isotropic_control(control) for control in controls):
        parser.error("--isotropic-control requires a manifest with isotropic ions and electrons")
    def time_step(audit):
        manifest = next(Path(audit["root"]).glob("*_analysis_manifest.json"))
        return json.loads(manifest.read_text())["physics"].get("dt_code_from_profile")

    for audit in audits:
        dt = time_step(audit)
        matching = [c for c in controls if c["kappa"] == audit["kappa"]
                    and dt is not None and time_step(c) is not None
                    and np.isclose(dt, time_step(c), rtol=1e-6, atol=0)
                    and (c.get("window") or {}).get("_series", {}).get("omega_ci_t", [None])[0] == 0]
        energy_audit.pair_controls([audit, *matching])
    energy_audit.write_outputs(audits + controls, energy_audit.group_runs(audits), out / "energia")
    baseline_rows = []
    for audit in audits:
        base = audit.get("baseline") or {}
        window = base.get("window") or {}
        stop = window.get("t_end_omegaci")
        stop_text = f"{stop:.2f}" if stop is not None else "—"
        baseline_rows.append(f"| {audit['run']} | {base.get('control', 'Sin pareja compatible')} | "
                             f"{stop_text} | {base.get('status', 'Sin corrección')} |")
    structure_rows, budgets, table, links, candidates = [], [], [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run, label, audit in zip(runs, ("Maxwell", "κ=5", "κ=3"), audits):
        phys, spec, struct = [run / name for name in ("09_physical_diagnostics", "04_spectra", "07_structures")]
        mode_file = spec / "dispersion_modes_yz_total.csv"
        ranked = rows(mode_file) if mode_file.is_file() else []
        mode = next((r for r in ranked if r["power_rank"] == "1" and r["accepted"] == "True"), {})
        phase = audit.get("linear_phase") or {}
        notes = reduce_run(run, out / run.name)
        for candidate in rows(phys / "mode_growth_table.csv"):
            if (float(candidate["theta_kB_deg"]) >= 40 and float(candidate["k_parallel_di"]) != 0
                    and candidate["fit_ok"] in ("1", "True")):
                duration = float(candidate["linear_phase_end"]) - float(candidate["linear_phase_start"])
                candidates.append({"run": run.name, "k_parallel_di": candidate["k_parallel_di"],
                    "k_perp_di": candidate["k_perp_di"], "C_B": candidate["compressibility"],
                    "C_B_window": candidate["compressibility_window"],
                    "fit_start": candidate["linear_phase_start"], "fit_end": candidate["linear_phase_end"],
                    "gamma": candidate["gamma"], "gamma_error": candidate["gamma_err"],
                    "amplitude_gain": candidate["amplitude_gain"], "domega_2pi_over_T": 2*np.pi/duration})
        data = rows(struct / "structures_table.csv")
        for r in data:
            denom = float(r["delta_B_mag_std"])
            structure_rows.append(dict(run=run.name, t=r["omega_ci_t"],
                bz_std_over_b_std=float(r["delta_B_parallel_std_over_B0"])/denom if denom > 0 else None,
                corr_n_B=r["corr_n_B"], pressure_ratio=r["pressure_balance_ratio"],
                hole_area_fraction=r["hole_area_fraction"]))
        w = audit["window"]
        te = w["_series"]
        axes[0, 0].plot(te["omega_ci_t"], te["T_e_over_T_e0"], label=label)
        axes[0, 1].plot([float(r["omega_ci_t"]) for r in data], [float(r["hole_area_fraction"]) for r in data], label=label)
        end = (audit.get("early_time") or {}).get("T_e_over_T_e0_at_end") or np.nan
        table.append(f"| {label} | {float(mode.get('gamma_over_omega_ci', np.nan)):.3f} | {abs(float(mode.get('omega_over_omega_ci', np.nan))):.3f} | {end:.2f} | {w['electron_heating_factor']:.2f} | {structure_rows[-1]['bz_std_over_b_std']:.3f} |")
        global_file = phys / "global_energy_table.csv"
        if global_file.exists():
            for row in phase_budget(global_file, phase):
                budgets.append(dict(run=run.name, **row))
            energy = np.genfromtxt(global_file, delimiter=",", names=True)
            for key, name in zip(("E_E", "E_B", "E_i", "E_e", "E_total"), ("Eléctrica", "Magnética", "Iones", "Electrones", "Residuo")):
                axes[1, 0].plot(energy["omega_ci_t"], (energy[key]-energy[key][0])/energy["E_total"][0], label=f"{label} {name}")
        lifetime_file = struct / "structure_lifetimes.csv"
        lifetime = rows(lifetime_file) if lifetime_file.is_file() else []
        holes = [float(r["observed_duration"]) for r in lifetime if r["kind"] == "hole" and r["left_censored"] == r["right_censored"] == "False"]
        if holes:
            axes[1, 1].step(np.sort(holes), np.arange(1,len(holes)+1)/len(holes), where="post", label=f"{label} (n={len(holes)})")
        rel = os.path.relpath(run, out)
        links.append(f"- {label} · [modo]({rel}/04_spectra/dispersion_modes_yz_total.csv) · [sensibilidad del detector]({rel}/07_structures/structure_sensitivity.csv) · [seguimiento]({rel}/07_structures/structure_lifetimes.csv) · [trabajo por especie]({rel}/09_physical_diagnostics/energy_exchange_summary.json) · [inventario de diagnósticos]({rel}/report/index.html)")
        if heating_figure(run, out / run.name):
            links.append(f"  [Balance y trabajo de los campos]({run.name}/heating_channels.png) · "
                         f"[series por especie]({run.name}/heating_channels.csv)")
        if notes:
            links.append("  " + " · ".join(notes) + ".")
            for filename in ("deepest_holes.csv", "oblique_phase.csv", "moment_budget.csv"):
                if (out / run.name / filename).exists():
                    links.append(f"  [{filename}]({run.name}/{filename})")
            links.append(f"  [Energías por momentos de toda la caja]({rel}/07_structures/snapshot_energy_density.csv)")
    write(out / "estructuras.csv", structure_rows)
    write(out / "candidatos_oblicuos.csv", candidates, ["run", "k_parallel_di", "k_perp_di", "C_B",
          "C_B_window", "fit_start", "fit_end", "gamma", "gamma_error", "amplitude_gain", "domega_2pi_over_T"])
    write(out / "energias_por_fase.csv", budgets, ["run", "phase", "t_start", "t_end", "baseline_time",
          "delta_E_E", "delta_E_B", "delta_E_i", "delta_E_e", "residual", "residual_over_E0",
          "rate_over_E0_per_omegaci", "residual_over_electron_gain"])
    for ax, ylabel in zip(axes.flat, ("Segundo momento central electrónico / inicial", "Fracción de área en agujeros", "Cambio de energía / energía total inicial", "Fracción acumulada de trayectorias")):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Tiempo × Ωci")
        ax.legend(fontsize=8)
    axes[1,1].set_xlabel("Duración observada × Ωci (sin censura)")
    fig.savefig(out / "revision.png", dpi=180)
    plt.close(fig)
    budget_table = [f"| {r['run']} | {r['phase']} | {r['delta_E_E']:+.2f} | {r['delta_E_B']:+.2f} | {r['delta_E_i']:+.2f} | {r['delta_E_e']:+.2f} | {100*r['residual_over_E0']:+.2f} % |" for r in budgets]
    body = """# Revisión de las tres simulaciones moderadas

## Modo dominante y evolución térmica

| Distribución | γ/Ωci | abs(ω)/Ωci | Te/Te₀ al final del ajuste | Te/Te₀ final | σ(Bz)/σ(abs(B)) final |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(table) + """

Las frecuencias corresponden al coeficiente complejo del modo dominante cuando el ajuste ha sido aceptado. La asignación AIC requiere comprobar su dirección, compresibilidad y polarización. Las temperaturas corresponden a la ventana de partículas. El incremento electrónico durante el propio ajuste limita la comparación con una dispersión calculada con parámetros iniciales fijos. La matriz de calidad y los productos históricos conservan la procedencia de cada medición.

![Evolución térmica, ocupación de agujeros, balance y duraciones](revision.png)

## Cierre energético en el dominio periódico

| Corrida | Fase | ΔE eléctrica | ΔE magnética completa | ΔK iones | ΔK electrones | Residuo / E₀ |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(budget_table) + """

Energías integradas en unidades del código. K incluye movimiento colectivo y térmico de cada especie. La suma usa las seis componentes de los campos y las dos energías cinéticas relativistas de DiagEnergies. Las fronteras del ajuste se interpolan en el registro global. Cada fila mide cambios desde el inicio de su intervalo, normalizados al total de la primera muestra global disponible. El CSV registra el tiempo de esa referencia. [Tasas por fase y cociente residuo/ganancia electrónica](energias_por_fase.csv).

Cada fila requiere un diagnóstico global de partículas y campos. Las corridas ausentes de la tabla carecen de ese cierre independiente. La suma de ventana E_proxy permite intercambio por sus fronteras.

El residuo mide la creación neta de energía de la caja cerrada dentro de la precisión del diagnóstico. Su cociente con ΔKₑ compara escalas y no identifica la fracción del calor electrónico de origen numérico. La ficha de trabajo por especie registra si la cadencia permite resolver las escalas de plasma y si la integral de J·E reproduce ΔK. La ventana de partículas ocupa aproximadamente el 4 % del dominio y permite intercambio a través de su frontera.

La auditoría registra por caso la tasa de aumento de ln(Te), la razón entre incrementos en ambas mitades y Δx/λDe. El mecanismo y un coeficiente de corrección por especie requieren controles isotrópicos con la misma distribución, resolución, paso y partículas por celda. Comparar cambios por volumen en tiempos comunes y repetir con resolución y semillas independientes. Una sustracción de control es condicional a la aditividad y debe conservar la curva bruta. [Parámetros y cobertura](energia/energy_audit.csv) · [Series térmicas](energia/energy_audit_electron_heating.csv).

| Corrida | Control emparejado | Último tΩci común térmico | Cierre tras sustracción |
|---|---|---:|---|
""" + "\n".join(baseline_rows) + """

El apareamiento comprueba dt de manifiesto, distribución y compatibilidad numérica. Revisar la procedencia de dt y ppc antes de interpretar la corrección. Un dt estimado por CFL tiene menor respaldo que el valor verificado en el registro de ejecución.

## Estructuras y alcance de la identificación

La [tabla de candidatos oblicuos](candidatos_oblicuos.csv) conserva todos los ajustes aceptados del catálogo con ángulo ≥40° y k paralelo finito, su compresibilidad y la resolución 2π/T de cada ajuste. La selección original de modos es finita. Una frecuencia sin resolver deja abierta la identificación de la rama.

El detector usa la media instantánea de |B|, suavizado periódico de 0.25 di y umbral máximo entre 1σ y 0.02 B₀. El cociente de desviaciones de Bz y |B| limita la contribución longitudinal global. Una anticorrelación densidad–campo y una depresión morfológica por sí solas admiten varios orígenes. El balance de presión de la caja usa σ(Pperp,i + Pperp,e + Pmag)/σ(Pmag). Las duraciones corresponden a asociaciones por solapamiento. La curva excluye censuras y es descriptiva de esa submuestra. [Series por corrida](estructuras.csv).

En los originales, ejecutar structures_analysis.py con --attribution. Los contrastes centro menos borde descomponen exactamente el campo en Bz y |B|−Bz tras el mismo suavizado. El borde se forma con vecinos exteriores por cara, excluye estructuras contiguas y puede infravalorar la profundidad para perfiles graduales. El campo reconstruido con el par transversal fundamental contrasta la hipótesis de modulación. Un balance local se evalúa comparando ΔPtot con las contribuciones térmicas y magnética en la misma estructura. Los coeficientes complejos conservan la fase para contrastar el crecimiento oblicuo con la envolvente transversal. Un residuo compresivo exige comparar frecuencia en el marco del plasma, ruido, coherencia y dispersión oblicua en los armónicos permitidos antes de asignarle un origen mirror. El espaciado 2π/20 = 0.314 di⁻¹ limita los modos oblicuos largos.

## Reproducción

Desde la raíz del repositorio, con el entorno Python de análisis activo:

```sh
python CodeforAnalisys/review_mirror_energy.py analysis_results/v6-reviewed-20261015 --outdir analysis_results/revision_mirror_energia
```

En el sistema que contiene los originales, fijar RUN a la carpeta de una sola corrida indicada en su manifiesto y OUT a una carpeta nueva de resultados para ella:

```sh
PSC_ANALYSIS_DATA_DIR="$RUN" python CodeforAnalisys/structures_analysis.py --data-dir "$RUN" --attribution --outdir "$OUT"
```

La tabla snapshot_energy_density.csv usa todo el dominio. Incluye E y B nativos, energía de flujo y presión térmica paralela y perpendicular de cada especie. La partición por momentos es una aproximación no relativista al balance cinético y debe compararse con DiagEnergies donde ambos existan. Los contrastes y sus identidades temporales están en structure_tracks.csv. Los armónicos de attribution_modes.csv incluyen ambas orientaciones oblicuas y el fundamental paralelo. Las componentes transversales complejas están sin suavizar. El resto de canales lleva el filtro del catálogo. Los momentos de flujo iónico medio permiten estimar el desplazamiento Doppler en el régimen no relativista.

Para evaluar un control isotrópico, añadir --isotropic-control con su directorio de resultados al primer comando, una vez por corrida de control. La corrección requiere coincidencia de distribución y compatibilidad numérica. Los datos brutos, series derivadas y controles deben compartir convención térmica.

## Evidencia por corrida

""" + "\n".join(links) + "\n"
    (out / "REVISION.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
