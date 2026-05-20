#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian — Mirror Instability (Maxwelliana)
# Resolución: 1536×1536 | Dominio: 20 d_i | 1000 ppc
# Nodos: feynman-00 (72 CPUs, ~123 GB) + nodo-00 (128 CPUs, ~502 GB)
# Total: 200 CPUs | 256 patches MPI | RAM estimada: ~150 GB
# Tiempo físico: ~657 Ω_i^{-1} | nmax=1,800,000 | 7 días
#
# Mejora respecto a malla 1024:
#   dx/λ_De: 1.53 → 1.02  (casi sub-Debye)
#   dx/d_e:  0.195 → 0.130 (mejor resolución electrónica)
#============================================================
#SBATCH --job-name=psc_mirror_mxw_1536
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --nodelist=feynman-00,nodo-00
#SBATCH --ntasks=200
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_mirror_mxw_1536_%j.out
#SBATCH --error=psc_mirror_mxw_1536_%j.err

# ------------------------------------------------------------------
# Entorno
# ------------------------------------------------------------------
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# Directorio de trabajo (checkpoints y salidas de diagnóstico)
WORKDIR=/scratchsan/observatorio/cmartinezsi/pcseditado/mirror_maxwellian_1536
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian — Malla 1536×1536"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodos       : ${SLURM_JOB_NODELIST}"
echo "Tareas MPI  : ${SLURM_NTASKS}"
echo "Patches     : 256  (16×16)"
echo "RAM estimada: ~150 GB"
echo "Directorio  : ${WORKDIR}"
echo "Inicio      : $(date)"
echo "=========================================================="

# ------------------------------------------------------------------
# Ejecutar Mirror Maxwellian con resolución 1536×1536
# Distribución explícita por nodo (heterogéneos: 72 + 128 CPUs)
# feynman-00: 72 slots | nodo-00: 128 slots
# ------------------------------------------------------------------
mpirun -H feynman-00:72,nodo-00:128 \
       -np 200 \
       --map-by slot \
       --bind-to core \
       /scratchsan/observatorio/cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
