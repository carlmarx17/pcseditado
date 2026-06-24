#!/bin/bash
# =====================================================================
#  build_cosma.sh
#  Compila PSC con ADIOS2 + parallel HDF5 en build/ del repo.
#  Ejecutar en el login node de COSMA:
#    chmod +x build_cosma.sh
#    ./build_cosma.sh
# =====================================================================

# --- Inicializar módulos manualmente (sin -l para no disparar .bashrc) ---
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

# --- Limpiar conda del PATH para evitar conflictos ---
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_SHLVL
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E "conda|anaconda|miniconda" | tr '\n' ':' | sed 's/:$//')
export PATH

set -euo pipefail

REPO=/cosma7/data/dp433/dc-mart18/pcseditado
BUILD_DIR=$REPO/build
ADIOS2_DIR=$HOME/adios2
HDF5_ROOT=/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4

echo "=== Cargando módulos ==="
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

echo "=== Verificando herramientas ==="
$ADIOS2_DIR/bin/adios2-config --version
h5pcc --version
echo "HDF5_ROOT=$HDF5_ROOT"

echo "=== Borrando build/ anterior y recreando ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "=== Configurando CMake ==="
cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPSC_USE_ADIOS2=ON \
  -DCMAKE_PREFIX_PATH="$ADIOS2_DIR;$HDF5_ROOT" \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DHDF5_C_COMPILER_EXECUTABLE="$(command -v h5pcc)" \
  -DUSE_CUDA=OFF \
  -DUSE_VPIC=OFF \
  -DBUILD_TESTING=OFF

echo "=== Compilando psc_mirror_kappa3 ==="
cmake --build "$BUILD_DIR" -j 8 --target psc_mirror_kappa3

echo ""
echo "=== Verificacion final ==="
grep "PSC_HAVE_ADIOS2" "$BUILD_DIR/src/include/PscConfig.h"
ldd "$BUILD_DIR/src/psc_mirror_kappa3" | grep -E "hdf5|adios"
echo ""
echo "LISTO: $BUILD_DIR/src/psc_mirror_kappa3"
