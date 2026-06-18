#!/bin/bash -l
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1024
#SBATCH --time=48:00:00
#SBATCH --job-name=psc_aniso
#SBATCH --output=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/%x_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="$REPO/build-adios2-nohdf5"
RUN_ROOT="$BASE/anisotropy_adios2"
PSC_TARGET="${PSC_TARGET:-psc_mirror_kappa3}"
RUN_DIR="$RUN_ROOT/${PSC_TARGET}_${SLURM_JOB_ID}"
ADIOS2_DIR="${ADIOS2_DIR:-$HOME/adios2-nohdf5}"
HDF5_ROOT="${HDF5_ROOT:-/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4}"

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

export PATH="$ADIOS2_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$ADIOS2_DIR/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

mkdir -p "$RUN_DIR"
cp "$BUILD_DIR/src/$PSC_TARGET" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

echo "target=$PSC_TARGET"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true

export PSC_NICELL="${PSC_NICELL:-2000}"
export PSC_CHECKPOINT_EVERY="${PSC_CHECKPOINT_EVERY:-7500}"
export PSC_FIELDS_EVERY="${PSC_FIELDS_EVERY:-1000}"
export PSC_PARTICLES_EVERY="${PSC_PARTICLES_EVERY:-10000}"

mpirun -np "$SLURM_NTASKS" "./$PSC_TARGET"

echo "end=$(date --iso-8601=seconds)"
