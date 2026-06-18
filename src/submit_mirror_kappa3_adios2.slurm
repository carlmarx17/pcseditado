#!/bin/bash -l
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1024
#SBATCH --time=48:00:00
#SBATCH --job-name=mirror_kappa3
#SBATCH --output=/cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/%x_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/%x_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="$REPO/build-adios2-nohdf5"
RUN_DIR="$BASE/mirror_kappa3_adios2/run_${SLURM_JOB_ID}"
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
cp "$BUILD_DIR/src/psc_mirror_kappa3" "$RUN_DIR/"
cp "$REPO/adios2cfg.xml" "$RUN_DIR/"
cd "$RUN_DIR"

echo "case=mirror_kappa3"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "start=$(date --iso-8601=seconds)"
adios2-config --version || true

export PSC_NGRID=1536
export PSC_NP_Y=64
export PSC_NP_Z=16
export PSC_NICELL=1000
export PSC_NMAX=1800000
export PSC_CHECKPOINT_EVERY=7500
export PSC_FIELDS_EVERY=750
export PSC_PARTICLES_EVERY=1000

mpirun -np "$SLURM_NTASKS" ./psc_mirror_kappa3

echo "end=$(date --iso-8601=seconds)"
