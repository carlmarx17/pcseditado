#!/bin/bash -l
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

module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

EXECUTABLE=./psc_mirror_maxwellian_2k

if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: No se encuentra $EXECUTABLE"
    exit 1
fi

echo "Inicio: $(date)"
mpirun -np 128 $EXECUTABLE
echo "Fin:    $(date)"
