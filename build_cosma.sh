#!/bin/bash -l
# =====================================================================
#  build_cosma.sh
#  Compila PSC con ADIOS2 en el build/ normal del repo.
#  Ejecutar en el login node de COSMA:
#    chmod +x build_cosma.sh
#    ./build_cosma.sh
# =====================================================================

set -euo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
BUILD_DIR=$REPO/build
ADIOS2_DIR=$HOME/adios2

echo "=== Cargando módulos ==="
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5

echo "=== Comprobando ADIOS2 en $ADIOS2_DIR ==="
$ADIOS2_DIR/bin/adios2-config --version || {
  echo "ERROR: adios2-config no encontrado en $ADIOS2_DIR/bin/"
  exit 1
}

echo "=== Borrando y recreando build/ ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "=== Configurando CMake con ADIOS2 ==="
cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPSC_USE_ADIOS2=ON \
  -DCMAKE_PREFIX_PATH="$ADIOS2_DIR" \
  -DUSE_CUDA=OFF \
  -DUSE_VPIC=OFF \
  -DBUILD_TESTING=OFF

echo "=== Compilando psc_mirror_kappa3 ==="
cmake --build "$BUILD_DIR" -j 8 --target psc_mirror_kappa3

echo ""
echo "=== Verificación ==="
grep "PSC_HAVE_ADIOS2" "$BUILD_DIR/src/include/PscConfig.h"
ldd "$BUILD_DIR/src/psc_mirror_kappa3" | grep -i adios
echo ""
echo "LISTO: $BUILD_DIR/src/psc_mirror_kappa3"
