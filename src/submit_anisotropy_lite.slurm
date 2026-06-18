#!/bin/bash -l
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=64
#SBATCH --time=08:00:00
#SBATCH --job-name=psc_lite
#SBATCH --output=/cosma7/data/dp433/dc-mart18/lite_runs/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/lite_runs/%x_%j.err

set -euo pipefail

: "${LITE_CASE:?Set LITE_CASE to M_lite, F_lite, or W_lite}"

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
RUN_DIR="$BASE/lite_runs/${LITE_CASE}_${SLURM_JOB_ID}"
EXECUTABLE="$REPO/build/src/psc_${LITE_CASE}"

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5

export ADIOS2_DIR="$HOME/adios2"
export PATH="$ADIOS2_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

mkdir -p "$RUN_DIR"
cp "$EXECUTABLE" "$RUN_DIR/"
cp "$REPO/CodeforAnalisys/lite_instability_analysis.py" "$RUN_DIR/"
cp "$REPO/CodeforAnalisys/data_reader.py" "$RUN_DIR/"
cp "$REPO/CodeforAnalisys/psc_units.py" "$RUN_DIR/"
cd "$RUN_DIR"

echo "case=$LITE_CASE"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "start=$(date --iso-8601=seconds)"

mpirun -np "$SLURM_NTASKS" "./psc_${LITE_CASE}"

echo "simulation_end=$(date --iso-8601=seconds)"
PSC_PROFILE="$LITE_CASE" python lite_instability_analysis.py \
  --data-dir "$RUN_DIR" --outdir "$RUN_DIR/analysis"
echo "analysis_end=$(date --iso-8601=seconds)"
