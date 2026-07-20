#!/bin/bash
set -euo pipefail

BASE="${BASE:-/cosma7/data/dp433/dc-mart18}"
REPO="${REPO:-$BASE/pcseditado}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-adios2-nohdf5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=src/cosma_adios2_env.sh
source "$SCRIPT_DIR/cosma_adios2_env.sh"

CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
if [ -z "$CMAKE_BIN" ] && [ -x "$HOME/bin/cmake" ]; then
  CMAKE_BIN="$HOME/bin/cmake"
fi
if [ -z "$CMAKE_BIN" ] && [ -x /usr/bin/cmake ]; then
  CMAKE_BIN=/usr/bin/cmake
fi
test -x "$CMAKE_BIN" || {
  echo "ERROR: cmake executable not found" >&2
  exit 1
}

if [ -d "$ADIOS2_DIR/lib64/cmake/adios2" ]; then
  ADIOS2_CMAKE_DIR="$ADIOS2_DIR/lib64/cmake/adios2"
else
  ADIOS2_CMAKE_DIR="$ADIOS2_DIR/lib/cmake/adios2"
fi

test -d "$ADIOS2_CMAKE_DIR" || {
  echo "ERROR: ADIOS2 CMake config not found in $ADIOS2_CMAKE_DIR" >&2
  echo "       ADIOS2_DIR=$ADIOS2_DIR" >&2
  exit 1
}

echo "Using ADIOS2_DIR=$ADIOS2_DIR"
echo "Using adios2-config=$(command -v adios2-config)"
adios2-config --version || true
echo "Using h5pcc=$(command -v h5pcc)"

"$CMAKE_BIN" -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DUSE_VPIC=OFF \
  -DPSC_USE_ADIOS2=ON \
  -DADIOS2_DIR="$ADIOS2_CMAKE_DIR" \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DHDF5_C_COMPILER_EXECUTABLE="$(command -v h5pcc)" \
  -DBUILD_TESTING=OFF

TARGETS=(
  psc_turbulence_3D_kappa_3
  psc_turbulence_3D_maxwellian_3
  psc_mirror_bimaxwellian_strong
  psc_mirror_bimaxwellian_moderate
  psc_mirror_bimaxwellian_weak
  psc_firehose_bimaxwellian_strong
  psc_firehose_bimaxwellian_moderate
  psc_firehose_bimaxwellian_weak
  psc_whistler_bimaxwellian_strong
  psc_whistler_bimaxwellian_moderate
  psc_whistler_bimaxwellian_weak
  psc_mirror_bikappa3 psc_mirror_bikappa3_moderate psc_mirror_bikappa5
  psc_firehose_bikappa3 psc_firehose_bikappa5
)

if [ -n "${PSC_TARGETS:-}" ]; then
  read -r -a TARGETS <<< "$PSC_TARGETS"
fi

"$CMAKE_BIN" --build "$BUILD_DIR" -j "${BUILD_JOBS:-16}" --target "${TARGETS[@]}"

grep -n "PSC_HAVE_ADIOS2" "$BUILD_DIR/src/include/PscConfig.h"
for target in "${TARGETS[@]}"; do
  test -x "$BUILD_DIR/src/$target"
done
ldd "$BUILD_DIR/src/psc_mirror_bikappa3" | grep -i adios || true
