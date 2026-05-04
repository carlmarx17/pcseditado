#!/bin/bash
# cmake.sh — Configura el build de PSC con soporte ADIOS2
# Uso: mkdir -p build && cd build && bash ../cmake.sh
# Requiere: MPI (OpenMPI), ADIOS2 (/usr/lib/cmake/adios2), HDF5-MPI

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DUSE_VPIC=OFF \
    -DPSC_USE_ADIOS2=ON \
    -DADIOS2_DIR=/usr/lib/cmake/adios2 \
    -DCMAKE_CXX_STANDARD=17 \
    ..
