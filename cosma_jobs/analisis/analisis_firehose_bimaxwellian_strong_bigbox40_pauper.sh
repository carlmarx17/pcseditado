#!/bin/bash -l
# Full analysis for the completed strong bi-Maxwellian firehose 40 d_i run.
# One independent Makefile stage is assigned to each of eight nodes.
# Override DATA_DIR, SNAPSHOT_EVERY or RESULTS_ROOT through --export=ALL.
#
# Example:
# sbatch cosma_jobs/analisis/analisis_firehose_bimaxwellian_strong_bigbox40_pauper.sh

#SBATCH --job-name=an_fh_bimax_strong_40di
#SBATCH --output=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bimax_strong_bigbox40.%J.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/logs/analysis_firehose_bimax_strong_bigbox40.%J.err
#SBATCH --partition=cosma7-rp-pauper
#SBATCH --account=dp433
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/CodeforAnalisys
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

set -uo pipefail
REPO=/cosma7/data/dp433/dc-mart18/pcseditado
source "$REPO/src/cosma_adios2_env.sh"
cd "$REPO/CodeforAnalisys"

RUN_ROOT=/cosma7/data/dp433/dc-mart18/anisotropy_adios2
DATA_DIR="${DATA_DIR:-$RUN_ROOT/psc_firehose_bimaxwellian_strong_bigbox40_11952824}"
CASE=firehose_bimaxwellian_strong_bigbox40
RESULTS_ROOT="${RESULTS_ROOT:-../analysis_results/run_aware_v4}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
GIF_EVERY="${GIF_EVERY:-10000}"
FIELDS_GIF="${FIELDS_GIF-1}"
DISPERSION_THETA_MAX="${DISPERSION_THETA_MAX-}"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi
mkdir -p /cosma7/data/dp433/dc-mart18/logs
MAKE_VARS=("SNAPSHOT_EVERY=$SNAPSHOT_EVERY" "GIF_EVERY=$GIF_EVERY" "FIELDS_GIF=$FIELDS_GIF" "DISPERSION_THETA_MAX=$DISPERSION_THETA_MAX")
if ! compgen -G "$DATA_DIR/pfd.*_p*.h5" > /dev/null && compgen -G "$DATA_DIR/pfd.*.bp" > /dev/null; then
    MAKE_VARS+=("FIELD_PATTERN=$DATA_DIR/pfd.*.bp" "MOMENT_PATTERN=$DATA_DIR/pfd_moments.*.bp" "PARTICLE_PATTERN=$DATA_DIR/prt_${CASE}.*.bp")
fi

srun --nodes=1 --ntasks=1 --exclusive make manifest DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${MAKE_VARS[@]}" || exit $?
STAGES=(brazil fields particles spectral diamagnetic heatflux validate physics energy-if-present)
if [[ "$CASE" == mirror_* || "$CASE" == M_* ]]; then STAGES+=(mirror); fi
declare -A PIDS
for stage in "${STAGES[@]}"; do
    log="/cosma7/data/dp433/dc-mart18/logs/analysis_${CASE}_${stage}.${SLURM_JOB_ID}.log"
    srun --nodes=1 --ntasks=1 --exclusive --job-name="$stage" make "$stage" DATA_DIR="$DATA_DIR" CASE="$CASE" RESULTS_ROOT="$RESULTS_ROOT" "${MAKE_VARS[@]}" > "$log" 2>&1 &
    PIDS[$stage]=$!
done
fail=0
for stage in "${STAGES[@]}"; do
    wait "${PIDS[$stage]}" || fail=1
done
exit "$fail"
