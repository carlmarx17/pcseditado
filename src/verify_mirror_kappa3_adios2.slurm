#!/bin/bash
#SBATCH --partition=cosma7-rp
#SBATCH --account=dp433
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --job-name=verify_mk3_adios2
#SBATCH --output=/cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/verify_%j.out
#SBATCH --error=/cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/verify_%j.err

set -euo pipefail

BASE=/cosma7/data/dp433/dc-mart18
REPO="$BASE/pcseditado"
BUILD_DIR="$REPO/build-adios2-nohdf5"
RUN_DIR="$BASE/mirror_kappa3_adios2/verify_${SLURM_JOB_ID}"
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

echo "verify_start=$(date --iso-8601=seconds)"
echo "adios2_config=$(command -v adios2-config || true)"
adios2-config --version || true

export PSC_NGRID=32
export PSC_NP_Y=1
export PSC_NP_Z=1
export PSC_NICELL=4
export PSC_NMAX=4
export PSC_CHECKPOINT_EVERY=2
export PSC_FIELDS_EVERY=2
export PSC_PARTICLES_EVERY=2

mpirun -np 1 ./psc_mirror_kappa3
test -d checkpoint_2.bp
test -d checkpoint_4.bp

mkdir restart
cp ./psc_mirror_kappa3 ./adios2cfg.xml restart/
cd restart
export PSC_RESTART="$RUN_DIR/checkpoint_2.bp"
export PSC_NMAX=3
mpirun -np 1 ./psc_mirror_kappa3
test -d checkpoint_3.bp

echo "verify_ok=$(date --iso-8601=seconds)"
