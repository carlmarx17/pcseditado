#!/bin/bash
#SBATCH --cluster=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodelist=nodo-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=diag_nodo00_%j.out
#SBATCH --error=diag_nodo00_%j.err

echo "=== DIAGNOSTIC RUN ON NODO-00 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Groups: $(groups)"
echo "Current Directory: $(pwd)"
echo ""

echo "=== TESTING HOME DIRECTORY ==="
ls -ld ~
touch ~/test_write_from_nodo00 && echo "Write to home: OK" && rm ~/test_write_from_nodo00
echo ""

echo "=== TESTING SCRATCH DIRECTORY ==="
echo "Checking /scratchsan mount status:"
df -h /scratchsan
echo "Checking directories:"
ls -ld /scratchsan/observatorio/Cmartinezsi
ls -ld /scratchsan/observatorio/Cmartinezsi/pcseditado
ls -l /scratchsan/observatorio/Cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian_2k
echo ""

echo "=== TESTING EXECUTABLE LIBRARIES ==="
ldd /scratchsan/observatorio/Cmartinezsi/pcseditado/build/src/psc_mirror_maxwellian_2k
echo ""

echo "=== TESTING ENVIRONMENT ==="
module list
echo "GCC path: $(which gcc)"
echo "MPI path: $(which mpirun)"
