#!/bin/bash -l

# Cargar los módulos necesarios en feynman (ajusta según los disponibles)
module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2
# Asegúrate de cargar los módulos de CMake y ADIOS2 si es necesario:
# module load cmake
# module load adios2

# Crear directorio de compilación
mkdir -p build
cd build

# Ejecutar CMake activando solo ADIOS2
cmake .. -DPSC_USE_ADIOS2=ON

# Compilar
make -j $(nproc)
