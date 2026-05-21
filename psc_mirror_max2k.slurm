#!/bin/bash -l
# ============================================================
#  PSC Mirror Maxwellian 2k — Multi-nodo híbrido
#  Nodos: pauli (246 GB) + feynman-00 (126 GB)
#  128 ranks MPI = 64 por nodo → ~125 GB por nodo
#  1024 parches / 128 ranks = 8 parches/rank  ✓
# ============================================================
#SBATCH --job-name=psc_mirror_max2k
#SBATCH --clusters=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=2
#SBATCH --nodelist=pauli,feynman-00
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/homes/observatorio/cmartinezsi/pcs_run
#SBATCH --output=/homes/observatorio/cmartinezsi/pcs_run/psc_mirror_max2k_%j.out
#SBATCH --error=/homes/observatorio/cmartinezsi/pcs_run/psc_mirror_max2k_%j.err

module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

EXECUTABLE=./psc_mirror_maxwellian_2k

if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: No se encuentra $EXECUTABLE"
    exit 1
fi

echo "=========================================="
echo "Inicio : $(date)"
echo "Nodos  : $SLURM_JOB_NODELIST"
echo "Ranks  : $SLURM_NTASKS  ($SLURM_NTASKS_PER_NODE por nodo)"
echo "=========================================="

mpirun -np 128 $EXECUTABLE

echo "=========================================="
echo "Fin    : $(date)"
echo "=========================================="
