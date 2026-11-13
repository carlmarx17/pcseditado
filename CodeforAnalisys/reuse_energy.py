"""Inventory existing results, render archived energy, or measure missing pressure work for one run."""
import argparse
import csv
import json
import os
from pathlib import Path
from time import monotonic

PHYS = "09_physical_diagnostics"


def inspect(run):
    def present(relative):
        p = run / relative
        return p.is_file() and p.stat().st_size > 0
    moments = run / "07_structures/snapshot_energy_density.csv"
    columns = []
    if moments.is_file():
        with moments.open() as f:
            columns = next(csv.reader(f), [])
    return {"run": str(run), "global_energy": present(f"{PHYS}/global_energy_table.csv"),
            "work_table": present(f"{PHYS}/energy_exchange_table.csv"),
            "work_summary": present(f"{PHYS}/energy_exchange_summary.json"),
            "pressure_work_both": all(f"pressure_strain_{s}" in columns for s in ("i","e")),
            "modal_results": present(f"{PHYS}/mode_growth_table.csv")}


def pressure_table(run, out, manifest):
    raw = Path(manifest["input_directory"])
    if not raw.is_dir():
        raise ValueError(f"Original run directory unavailable: {raw}")
    os.environ.update(PSC_PROFILE=manifest["case"], PSC_ANALYSIS_DATA_DIR=str(raw))
    os.environ.pop("PSC_ANALYSIS_CONFIG", None)
    config = raw/"analysis_config.json"
    if config.is_file():
        os.environ["PSC_ANALYSIS_CONFIG"] = str(config)
    from data_reader import PICDataReader
    from structures_analysis import load_snapshot
    from psc_units import step_to_omegaci
    import numpy as np

    files = PICDataReader.discover_outputs(str(raw))
    with (run/PHYS/"energy_exchange_table.csv").open() as f:
        history = [(int(r["step"]), float(r["omega_ci_t"])) for r in csv.DictReader(f)]
    steps = [step for step, t in history]
    if len(steps) < 2 or np.any(np.diff(steps) <= 0):
        raise ValueError("Missing or unordered archived work series")
    for step, t in history:
        if step not in files["fields"] or step not in files["moments"]:
            raise ValueError(f"Missing raw paired snapshot at {step}")
        if not np.isclose(step_to_omegaci(step), t, atol=1e-10, rtol=1e-6):
            raise ValueError(f"Archived time and original configuration disagree at {step}")
    parts = [f"{k}_{s}" for s in ("e","i") for k in ("bulk","thermal_parallel","thermal_perp")]
    target = out/"snapshot_energy_density.csv"
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w", dir=out, delete=False, newline="") as f:
        tmp = Path(f.name)
        try:
            for i, (step, t) in enumerate(history):
                e = load_snapshot(files["fields"][step], files["moments"][step], budget=True)["energy"]
                row = {"step": step, "omega_ci_t": t, "total_moment_approx":
                       sum(e[k] for k in ("E_B","E_E",*parts)), **e}
                if i == 0:
                    writer = csv.DictWriter(f, row)
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    scan = sub.add_parser("inventory", help="List archived run products using only metadata and CSV headers")
    scan.add_argument("root", type=Path)
    render = sub.add_parser("figure", help="Use existing tables for one periodic anisotropy run")
    render.add_argument("run", type=Path)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--measure-pressure", action="store_true",
                        help="Read raw paired fields and moments only if archived pressure work is missing")
    args = p.parse_args()
    if args.action == "inventory":
        if not args.root.is_dir():
            p.error(f"Results directory unavailable: {args.root}")
        for folder in sorted(args.root.rglob(PHYS)):
            if folder.is_dir() and ".staging" not in folder.parts:
                print(json.dumps(inspect(folder.parent), ensure_ascii=False))
        return
    run = args.run.resolve()
    m = list(run.glob("*_analysis_manifest.json"))
    if len(m) != 1:
        p.error("Select one run with a unique analysis manifest")
    manifest = json.loads(m[0].read_text())
    if manifest.get("instability") not in ("mirror","firehose","whistler"):
        p.error("This energy budget requires a supported periodic anisotropy run")
    status = inspect(run)
    if not status["work_table"] or not status["work_summary"]:
        p.error("Missing archived work table/summary. Run the existing energy-exchange stage first.")
    out = args.out.resolve()
    if out == run or (run in out.parents and not out.is_relative_to(run/".staging")):
        p.error("Use a separate output directory to preserve the archive")
    out.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    measured = None
    read_raw = False
    if args.measure_pressure and not status["pressure_work_both"]:
        from analysis_contract import inventory_identity, atomic_json, provenance
        files = [m[0], run/PHYS/"energy_exchange_table.csv"]
        from data_reader import PICDataReader
        raw = PICDataReader.discover_outputs(manifest["input_directory"])
        sources = [*raw["fields"].values(), *raw["moments"].values()]
        config = Path(manifest["input_directory"])/"analysis_config.json"
        if config.is_file():
            files.append(config)
        signature = {"manifest": str(m[0]), "tables": inventory_identity(files),
                     "inputs": inventory_identity(sources), "source": provenance()["analysis_source_sha256"]}
        marker = out/"pressure_source.json"
        cached = json.loads(marker.read_text()) if marker.exists() else {}
        measured = out/"snapshot_energy_density.csv"
        if (cached.get("signature") != signature or not measured.is_file()
                or cached.get("output") != inventory_identity([measured])):
            measured = pressure_table(run, out, manifest)
            read_raw = True
            atomic_json(marker, {"signature": signature, "output": inventory_identity([measured])})
    from heating_channels import heating_figure
    if not heating_figure(run, out, moment_file=measured):
        p.error("Fewer than two common snapshots for this balance")
    from analysis_contract import atomic_json
    from energy_audit import audit_run, strip_private
    from review_mirror_energy import phase_budget
    audit = audit_run(run)
    global_table = run / PHYS / "global_energy_table.csv"
    phases = phase_budget(global_table, audit.get("linear_phase") or {}) if global_table.exists() else []
    atomic_json(out/"budget_summary.json", {"source": str(run), "audit": strip_private(audit),
                "global_phases": phases, "work_sampling": json.loads((run/PHYS/"energy_exchange_summary.json").read_text())})
    if manifest["instability"] == "mirror":
        from thesis_modal_evidence import reduce_run
        reduce_run(run, out)
    print(json.dumps({"case": manifest["case"], "source": str(run), "out": str(out),
                      "read_raw": read_raw, "elapsed_seconds": round(monotonic()-start,2)}))


if __name__ == "__main__":
    main()
