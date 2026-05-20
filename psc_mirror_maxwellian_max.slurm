#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian — Mirror Instability (Maxwelliana)
# Resolución: 2048×2048 | Dominio: 20 d_i | 1000 ppc
# Nodos: feynman-00 (72) + nodo-00 (128) + boltzmann (56) = 256 CPUs
# Total: 256 CPUs | 1024 patches MPI (32×32) | RAM estimada: ~265 GB
# Tiempo físico: ~657 Ω_i^{-1} | nmax=2,400,000 | 7 días
#
# Mejora respecto a malla 1024/1536:
#   dx/λ_De: 0.77  (sub-Debye completa)
#   dx/d_e:  0.098 (máxima resolución de escala electrónica)
#============================================================
#SBATCH --job-name=psc_mirror_mxw_2048
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --qos=low
#SBATCH --nodes=3
#SBATCH --nodelist=feynman-00,nodo-00,boltzmann
#SBATCH --ntasks=256
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_mirror_mxw_2048_%j.out
#SBATCH --error=psc_mirror_mxw_2048_%j.err

# ------------------------------------------------------------------
# Entorno
# ------------------------------------------------------------------
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# Directorio de trabajo (checkpoints y salidas de diagnóstico)
WORKDIR=/scratchsan/observatorio/cmartinezsi/pcseditado/mirror_maxwellian_2048
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian — Malla 2048×2048 (Sub-Debye)"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodos       : ${SLURM_JOB_NODELIST}"
echo "Tareas MPI  : ${SLURM_NTASKS}"
echo "Patches     : 1024 (32×32)"
echo "RAM estimada: ~265 GB"
echo "Directorio  : ${WORKDIR}"
echo "Inicio      : $(date)"
echo "=========================================================="

# ------------------------------------------------------------------
# Ejecutar Mirror Maxwellian con resolución 2048×2048
# Distribución explícita por nodo (72 + 128 + 56 CPUs)
# ------------------------------------------------------------------
mpirun -H feynman-00:72,nodo-00:128,boltzmann:56 \
       -np 256 \
       --map-by slot \
       --bind-to core \
       /scratchsan/observatorio/cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian_2k

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
