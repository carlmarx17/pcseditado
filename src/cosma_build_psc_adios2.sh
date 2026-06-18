#!/bin/bash -l
set -euo pipefail

BASE="${BASE:-/cosma7/data/dp433/dc-mart18}"
REPO="${REPO:-$BASE/pcseditado}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-adios2-nohdf5}"
ADIOS2_DIR="${ADIOS2_DIR:-$HOME/adios2-nohdf5}"
HDF5_ROOT="${HDF5_ROOT:-/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4}"

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

export PATH="$ADIOS2_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$ADIOS2_DIR/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"

if [ -d "$ADIOS2_DIR/lib64/cmake/adios2" ]; then
  ADIOS2_CMAKE_DIR="$ADIOS2_DIR/lib64/cmake/adios2"
else
  ADIOS2_CMAKE_DIR="$ADIOS2_DIR/lib/cmake/adios2"
fi

cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DUSE_VPIC=OFF \
  -DPSC_USE_ADIOS2=ON \
  -DADIOS2_DIR="$ADIOS2_CMAKE_DIR" \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DHDF5_C_COMPILER_EXECUTABLE="$(command -v h5pcc)" \
  -DBUILD_TESTING=OFF

cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-16}" --target psc_mirror_kappa3

grep -n "PSC_HAVE_ADIOS2" "$BUILD_DIR/src/include/PscConfig.h"
ldd "$BUILD_DIR/src/psc_mirror_kappa3" | grep -i adios || true
