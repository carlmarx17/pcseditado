#!/bin/bash -l
#SBATCH --ntasks=1024
#SBATCH -J psc_kappa3
#SBATCH -o /cosma7/data/dp433/dc-mart18/kappa3.%J.out
#SBATCH -e /cosma7/data/dp433/dc-mart18/kappa3.%J.err
#SBATCH -p cosma7-rp
#SBATCH -A dp433
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --nodes=37
#SBATCH --ntasks-per-node=28
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

# Cargar módulos
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5

# ADIOS2 instalado en $HOME/adios2
export ADIOS2_DIR=$HOME/adios2
export PATH=$ADIOS2_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ADIOS2_DIR/lib:$ADIOS2_DIR/lib64:$LD_LIBRARY_PATH

# Directorio de run (se crea si no existe)
RUN_DIR=/cosma7/data/dp433/dc-mart18/run_kappa3_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Copiar ejecutable y config ADIOS2
cp /cosma7/data/dp433/dc-mart18/pcseditado/build/src/psc_mirror_kappa3 "$RUN_DIR/"
cp /cosma7/data/dp433/dc-mart18/pcseditado/adios2cfg.xml               "$RUN_DIR/"

cd "$RUN_DIR"

# Ejecutar
mpirun -np $SLURM_NTASKS ./psc_mirror_kappa3
