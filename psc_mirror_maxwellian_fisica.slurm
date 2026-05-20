#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian — Mirror (Física Cluster)
# Resolución: 1536×1536 | Dominio: 20 d_i | 1000 ppc
# Nodos: feynman-00 (72 CPUs, ~126 GB) + boltzmann (56 CPUs, ~126 GB)
# Clúster: fisica | Partición: cpu.cecc
# Tareas MPI: 128 | Patches: 256 (16×16)
# RAM total: ~252 GB (cabe holgado para los ~150 GB de la simulación)
#============================================================
#SBATCH --job-name=psc_mirror_mxw_fis
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --nodelist=feynman-00,boltzmann
#SBATCH --ntasks=128
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_mirror_mxw_fis_%j.out
#SBATCH --error=psc_mirror_mxw_fis_%j.err

# ------------------------------------------------------------------
# Entorno
# ------------------------------------------------------------------
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# Directorio de trabajo
WORKDIR=/scratchsan/observatorio/cmartinezsi/pcseditado/mirror_maxwellian_1536
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian — Clúster FISICA (feynman + boltzmann)"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodos       : feynman-00, boltzmann"
echo "Tareas MPI  : 128"
echo "RAM Total   : 252 GB"
echo "Inicio      : $(date)"
echo "=========================================================="

# Ejecutar distribuyendo explícitamente por nodo
mpirun -H feynman-00:72,boltzmann:56 \
       -np 128 \
       --map-by slot \
       --bind-to core \
       /scratchsan/observatorio/cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
