#!/bin/bash -l
#SBATCH --job-name=tesis_pendientes
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

set -eo pipefail
repo=${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ -f "$PWD/CodeforAnalisys/submit_missing.py" ]]; then repo=$PWD; fi
source "$repo/src/cosma_adios2_env.sh"
set -u
unset PSC_ANALYSIS_CONFIG PSC_ANALYSIS_DATA_DIR PSC_ANALYSIS_RUN_ID PSC_PROFILE
export MPLBACKEND=Agg PSC_FIG_THEME=paper OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/psc-mpl-${SLURM_JOB_ID:-$UID}"
py=${PYTHON:-$repo/.venv/bin/python}
if [[ ${1:-} == *.json ]]; then
    exec "$py" "$repo/CodeforAnalisys/submit_missing.py" --worker "$1"
fi
exec "$py" "$repo/CodeforAnalisys/submit_missing.py" "$@"
