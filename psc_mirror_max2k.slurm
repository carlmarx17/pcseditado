#!/bin/bash -l
# ============================================================
#  PSC Mirror Maxwellian 2k — Nodo único pauli
#  Nodo: pauli (246 GB RAM, 64 CPUs)
#  64 ranks MPI → ~120 GB estimados
#  1024 parches / 64 ranks = 16 parches/rank  ✓
# ============================================================
#SBATCH --job-name=psc_mirror_max2k
#SBATCH --clusters=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=pauli
#SBATCH --ntasks=64
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
echo "Ranks  : $SLURM_NTASKS"
echo "=========================================="

mpirun -np 64 $EXECUTABLE

echo "=========================================="
echo "Fin    : $(date)"
echo "=========================================="
