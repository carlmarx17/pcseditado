#!/bin/bash -l
#SBATCH --job-name=tesis_reconexion
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
set -eo pipefail
repo=${SLURM_SUBMIT_DIR:-$PWD}
source "$repo/src/cosma_adios2_env.sh"
set -u
unset PSC_PROFILE PSC_ANALYSIS_CONFIG PSC_ANALYSIS_DATA_DIR
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data=${1:?Pass the original double-Harris run directory}
profile=${2:?Pass reconnection or reconnection_comparable}
exec "${PYTHON:-$repo/.venv/bin/python}" "$repo/CodeforAnalisys/reconnection_job.py" "$data" "$profile" \
    "${OUT:-$repo/analysis_results/tesis_pendientes/reconexion/$(basename "$data")}"
