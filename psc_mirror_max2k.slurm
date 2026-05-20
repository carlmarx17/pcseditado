#!/bin/bash -l
#SBATCH --job-name=psc_mirror_max2k
#SBATCH --cluster=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodelist=nodo-00
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_mirror_max2k_%j.out
#SBATCH --error=psc_mirror_max2k_%j.err

# ============================================================
# psc_mirror_maxwellian_2k  —  Mirror Maxwellian 2048×2048
# ============================================================
# RAM estimada:
#   Partículas: 2 esp × 2048² × 1000 ppc × 28 B ≈ 235 GB
#   Campos + buffers MPI + overhead OS:            ≈  15 GB
#   TOTAL:                                         ≈ 250 GB
#   Límite SLURM (--mem):                            250 GB  ← ~50% de 502 GB
#   RAM libre en nodo-00:                           ~252 GB  ← no peta
#
# MPI layout:
#   np = {1, 32, 32} → 1024 parches
#   1024 parches / 128 ranks = 8 parches/rank  ✓ balance perfecto
# ============================================================

module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

RUNDIR=/scratchsan/observatorio/Cmartinezsi/mirror_maxwellian_2k
mkdir -p ${RUNDIR}
cd ${RUNDIR}

echo "========================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodo:        $(hostname)"
echo "Inicio:      $(date)"
echo "RAM total:   $(free -h | awk '/Mem:/{print $2}')"
echo "Límite job:  250 GB"
echo "========================================"

mpirun -np 128 \
    --bind-to core \
    --map-by core \
    /scratchsan/observatorio/Cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian_2k

EXIT_CODE=$?
echo "Fin: $(date)  |  Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
