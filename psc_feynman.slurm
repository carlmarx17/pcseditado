#!/bin/bash -l
#SBATCH --job-name=psc_feynman
#SBATCH --clusters=fisica
#SBATCH --partition=cpu.cecc
#SBATCH --nodelist=feynman-00
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --output=psc_feynman_%j.out
#SBATCH --error=psc_feynman_%j.err

# Limpiar y cargar módulos
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2  # O el gcc compatible que uses

# Ejecutar simulación con mpirun (evita problemas de PMI con srun)
# Corriendo la versión psc_temp_aniso
mpirun -np 32 /scratchsan/observatorio/cmartinezsi/pcseditado/build/src/psc_temp_aniso

# Si deseas correr la versión maxwelliana, descomenta la siguiente línea y comenta la anterior:
# mpirun -np 32 /scratchsan/observatorio/cmartinezsi/pcseditado/build/src/psc_maxwellian
