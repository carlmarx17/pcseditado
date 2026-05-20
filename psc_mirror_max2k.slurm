#!/bin/bash -l
#============================================================
# Job: psc_mirror_maxwellian_2k — Mirror (Maxwelliana 2k)
# Resolución: 2048×2048 | Dominio: 20 d_i | 1000 ppc | mass_ratio: 200
# Nodo: nodo-00 (128 CPUs, ~514 GB RAM)
# Clúster: cecc | Partición: cpu.cecc
# Filesystem: /homes/observatorio/cmartinezsi (NFS desde perseus)
#============================================================
#SBATCH --job-name=psc_mirror_max2k
#SBATCH --clusters=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=nodo-00
#SBATCH --ntasks=128
#SBATCH --time=4-00:00:00
#SBATCH --exclusive
#SBATCH --chdir=/homes/observatorio/cmartinezsi/pcs_run
#SBATCH --output=/homes/observatorio/cmartinezsi/pcs_run/psc_mirror_max2k_%j.out
#SBATCH --error=/homes/observatorio/cmartinezsi/pcs_run/psc_mirror_max2k_%j.err

# ------------------------------------------------------------------
# Entorno
# ------------------------------------------------------------------
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

# Rutas
BASEDIR=/homes/observatorio/cmartinezsi/pcs_run
WORKDIR=${BASEDIR}/mirror_maxwellian_2048
EXECUTABLE=${BASEDIR}/psc_mirror_maxwellian_2k

mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Mirror Maxwellian 2k — Clúster CECC (nodo-00)"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Nodo        : ${SLURM_JOB_NODELIST}"
echo "Tareas MPI  : ${SLURM_NTASKS}"
echo "Ejecutable  : ${EXECUTABLE}"
echo "Directorio  : ${WORKDIR}"
echo "Inicio      : $(date)"
echo "RAM total   : $(free -h | awk '/Mem:/{print $2}')"
echo "=========================================================="

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Ejecutable no encontrado: ${EXECUTABLE}"
    exit 1
fi

mpirun -np 128 \
       --bind-to hwthread \
       ${EXECUTABLE}

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
