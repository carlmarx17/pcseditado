#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian — Mirror Instability (Maxwelliana)
# Resolución: 1536×1536 | Dominio: 20 d_i | 1000 ppc | mass_ratio: 200
# Nodo: nodo-00 (128 CPUs, ~514 GB RAM)
# Clúster: cecc | Partición: cpu.cecc
# Filesystem compartido: /homes/observatorio/cmartinezsi (NFS desde perseus)
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

# Rutas (nodo-00 ve /homes/observatorio/cmartinezsi vía NFS)
BASEDIR=/homes/observatorio/cmartinezsi/pcs_run
WORKDIR=${BASEDIR}/mirror_maxwellian_1536
EXECUTABLE=${BASEDIR}/psc_mirror_maxwellian

mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian 1536 — Clúster CECC (nodo-00)"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodo        : ${SLURM_JOB_NODELIST}"
echo "Tareas MPI  : ${SLURM_NTASKS}"
echo "Ejecutable  : ${EXECUTABLE}"
echo "Directorio  : ${WORKDIR}"
echo "Inicio      : $(date)"
echo "=========================================================="

# Verificar que el ejecutable existe
if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Ejecutable no encontrado: ${EXECUTABLE}"
    echo "Copia el ejecutable con:"
    echo "  cp /scratchsan/observatorio/Cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian ~/pcs_run/"
    exit 1
fi

mpirun -np 128 \
       --bind-to core \
       ${EXECUTABLE}

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
