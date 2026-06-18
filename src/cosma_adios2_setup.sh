#!/bin/bash -l
set -euo pipefail

PREFIX="${ADIOS2_DIR:-$HOME/adios2-nohdf5}"
ADIOS2_VERSION="${ADIOS2_VERSION:-2.12.0}"
BUILD_ROOT="${BUILD_ROOT:-$HOME/build_adios2_nohdf5}"

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

if command -v adios2-config >/dev/null 2>&1; then
  echo "adios2-config already available: $(command -v adios2-config)"
  adios2-config --version || true
  exit 0
fi

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

if [ ! -d "ADIOS2-$ADIOS2_VERSION" ]; then
  curl -L "https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VERSION}.tar.gz" \
    -o "adios2-${ADIOS2_VERSION}.tar.gz"
  tar -xzf "adios2-${ADIOS2_VERSION}.tar.gz"
fi

cmake -S "ADIOS2-$ADIOS2_VERSION" -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_BUILD_EXAMPLES=OFF \
  -DADIOS2_BUILD_TESTING=OFF

cmake --build build -j "${BUILD_JOBS:-8}"
cmake --install build

echo "ADIOS2 installed in $PREFIX"
"$PREFIX/bin/adios2-config" --version || true
