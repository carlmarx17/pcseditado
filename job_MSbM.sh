#!/bin/bash -l

#SBATCH --ntasks=1024

#SBATCH -J psc_M_S_bM

#SBATCH -o output_MSbM.%J.out

#SBATCH -e error_MSbM.%J.err

#SBATCH -p cosma7

#SBATCH -A dp433

#SBATCH --exclusive

#SBATCH -t 72:00:00

#SBATCH --chdir=/cosma7/data/dp433/dc-mart18/pcseditado/build/src

#SBATCH --mail-type=END

#SBATCH --mail-user=dc-mart18@cosma.dur.ac.uk

module load gnu_comp/14.1.0

module load openmpi/5.0.3

module load parallel_hdf5

export ADIOS2_DIR=$HOME/adios2

export PATH=$ADIOS2_DIR/bin:$PATH

export LD_LIBRARY_PATH=$ADIOS2_DIR/lib:$LD_LIBRARY_PATH

mpirun -np $SLURM_NTASKS ./psc_M_S_bM
