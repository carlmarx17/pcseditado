#!/bin/bash -l
#
# =====================================================================
#  analisis_resolution_mirror.sh — analysis of the resolution test of
#  psc_mirror_bimaxwellian_moderate (cosma_jobs/simulacion/
#  sim_mirror_bimaxwellian_moderate_res_{ppc4000,ngrid1152}.sh) against the
#  production run already analysed with reanalysis_v6_all.sh.
#
#  Each variant goes through run_pipeline.py (it reads the variant's
#  analysis_config.json, so dt, ppc and cadence are the ones that ran).
#  Then:
#    energy_audit.py   electron heating vs time of the three runs and at
#                      their last common time (energy_audit_common_time.csv):
#                      heating ~1/ppc -> particle noise; heating falling with
#                      dx/lambda_De -> finite-grid heating
#    convergence_study.py  gamma and the other observables side by side
#                      (compare only quantities defined before t Omega_ci ~ 41:
#                      the variants stop there)
#
#  Submit after both variants have finished (folder names from their jobs):
#    sbatch --export=ALL,PPC_RUN=psc_mirror_bimaxwellian_moderate_res_ppc4000_<jobid>,GRID_RUN=psc_mirror_bimaxwellian_moderate_res_ngrid1152_<jobid> \
#      cosma_jobs/analisis/analisis_resolution_mirror.sh
# =====================================================================

#SBATCH --job-name=resolution_mirror
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/resolution_mirror.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/resolution_mirror.%J.err
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
# shellcheck source=../../src/cosma_adios2_env.sh
source "$REPO/src/cosma_adios2_env.sh"
cd "$REPO/CodeforAnalisys" || exit 1

RUN_ROOT=/cosma7/data/dp433/dc-mart18/anisotropy_adios2
PY="${PYTHON:-$REPO/.venv/bin/python}"
OUT="${OUT:-$REPO/analysis_results/v6_resolution}"
BASE_RESULTS="${BASE_RESULTS:-$REPO/analysis_results/v6/mirror_bimaxwellian_moderate}"
CASE=mirror_bimaxwellian_moderate
STEP=(srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task="${SLURM_CPUS_PER_TASK:-28}")
STAGES=(manifest residuals physics spectral structures energy-exchange estimators validate)
export MPLCONFIGDIR="/tmp/psc-mpl-${SLURM_JOB_ID:-manual}" PSC_FIG_THEME=paper
say() { echo "[$(date '+%F %T')] $*"; }

: "${PPC_RUN:?set PPC_RUN to the ppc4000 run folder}"
: "${GRID_RUN:?set GRID_RUN to the ngrid1152 run folder}"
if [ ! -f "$BASE_RESULTS/pipeline.json" ]; then
    echo "ERROR: $BASE_RESULTS has no v6 analysis; run reanalysis_v6_all.sh first." >&2
    exit 2
fi
for folder in "$PPC_RUN" "$GRID_RUN"; do
    [ -f "$RUN_ROOT/$folder/analysis_config.json" ] || {
        echo "ERROR: $RUN_ROOT/$folder/analysis_config.json missing (not a resolution-test run?)" >&2; exit 2; }
done

PIDS=()
for pair in "ppc4000:$PPC_RUN" "ngrid1152:$GRID_RUN"; do
    tag="${pair%%:*}"; folder="${pair#*:}"
    mkdir -p "$OUT/$tag"
    say "analyse $tag: $RUN_ROOT/$folder"
    ( "$PY" run_pipeline.py --data-dir "$RUN_ROOT/$folder" --case "$CASE" --results-root "$OUT/$tag" \
          --jobs 2 --launcher "${STEP[*]}" --keep-going --resume --stages "${STAGES[@]}" \
          --make-option "RUN_TAG=res_$tag" > "$OUT/$tag/runner.log" 2>&1
      echo $? > "$OUT/$tag/runner.rc" ) &
    PIDS+=($!)
done
wait "${PIDS[@]}"

RUNS=("base=$BASE_RESULTS" "ppc4000=$OUT/ppc4000/$CASE" "ngrid1152=$OUT/ngrid1152/$CASE")
"$PY" energy_audit.py "${RUNS[@]}" --outdir "$OUT/energy_audit" > "$OUT/energy_audit.log" 2>&1
"$PY" convergence_study.py "${RUNS[@]}" --reference base --outdir "$OUT/convergence" > "$OUT/convergence.log" 2>&1

say "runners: ppc4000 rc=$(cat "$OUT/ppc4000/runner.rc") ngrid1152 rc=$(cat "$OUT/ngrid1152/runner.rc")"
say "electron heating at the common time:"
cat "$OUT/energy_audit/energy_audit_common_time.csv"
