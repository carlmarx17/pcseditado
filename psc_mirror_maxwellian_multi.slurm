#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian — Mirror Instability (Maxwelliana)
# Resolución: 1536×1536 | Dominio: 20 d_i | 1000 ppc
# Nodo: nodo-00 (128 CPUs, ~514 GB RAM)
# Clúster: cecc | Partición: cpu.cecc
# Tareas MPI: 128 | Patches: 256 (16×16) | mass_ratio: 200
# RAM estimada: ~150 GB (cabe holgado en los 514 GB de nodo-00)
# Tiempo máximo en cecc: 4 días
#============================================================
#SBATCH --job-name=psc_mirror_mxw_cecc
#SBATCH --clusters=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=nodo-00
#SBATCH --ntasks=128
#SBATCH --time=4-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_mirror_mxw_cecc_%j.out
#SBATCH --error=psc_mirror_mxw_cecc_%j.err

# ------------------------------------------------------------------
# Entorno
# ------------------------------------------------------------------
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# Directorio de trabajo
WORKDIR=/scratchsan/observatorio/Cmartinezsi/pcseditado/mirror_maxwellian_1536
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian — Clúster CECC (nodo-00)"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodo        : nodo-00"
echo "Tareas MPI  : 128"
echo "RAM Node    : 514 GB"
echo "Inicio      : $(date)"
echo "=========================================================="

# Ejecutar usando el nodo-00 completo (128 cores)
mpirun -np 128 \
       --bind-to core \
       /scratchsan/observatorio/Cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
