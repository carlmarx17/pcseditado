# Guia simple: activar ADIOS2 en COSMA para PSC

Objetivo: compilar PSC con soporte ADIOS2 para que los checkpoints se guarden
como `checkpoint_<step>.bp/`, verificar que se pueden leer con `bpls`, probar
restart, y dejar listo `psc_mirror_kappa3`.

## Versiones usadas y verificadas

```text
Cluster: COSMA7
Compiler: gnu_comp/14.1.0
MPI: openmpi/5.0.3
HDF5 paralelo PSC: parallel_hdf5/1.14.4
ADIOS2: 2.12.0
ADIOS2 install: $HOME/adios2-nohdf5
PSC build: /cosma7/data/dp433/dc-mart18/pcseditado/build-adios2-nohdf5
```

Importante: ADIOS2 se instalo con `ADIOS2_USE_HDF5=OFF`. Esto evita conflictos
con HDF5 en COSMA. PSC sigue usando `parallel_hdf5/1.14.4` para sus salidas
HDF5/MRC normales.

## 1. Entrar al cluster

```bash
ssh -X dc-mart18@login7.cosma.dur.ac.uk
cd /cosma7/data/dp433/dc-mart18/pcseditado
```

## 2. Cargar modulos

Usar siempre estos modulos para instalar ADIOS2, compilar PSC y correr:

```bash
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4
```

Comprobar HDF5 paralelo:

```bash
which h5pcc
h5pcc -show
```

Debe apuntar a:

```text
/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4/bin/h5pcc
```

Si aparece `h5cc` de Conda, no usarlo para compilar PSC.

## 3. Instalar ADIOS2

Ruta recomendada: usar el script ya preparado.

```bash
chmod +x src/cosma_adios2_setup.sh
src/cosma_adios2_setup.sh
```

Si hay que crearlo manualmente:

```bash
nano src/cosma_adios2_setup.sh
```

Pegar esto:

```bash
#!/bin/bash -l
set -euo pipefail

PREFIX="${ADIOS2_DIR:-$HOME/adios2-nohdf5}"
ADIOS2_VERSION="${ADIOS2_VERSION:-2.12.0}"
BUILD_ROOT="${BUILD_ROOT:-$HOME/build_adios2_nohdf5}"

module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

if [ -x "$PREFIX/bin/adios2-config" ]; then
  "$PREFIX/bin/adios2-config" --version
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

cmake --build build -j "${BUILD_JOBS:-4}"
cmake --install build

"$PREFIX/bin/adios2-config" --version
```

Guardar en `nano`: `Ctrl+O`, `Enter`, `Ctrl+X`.

Ejecutar:

```bash
chmod +x src/cosma_adios2_setup.sh
BUILD_JOBS=4 src/cosma_adios2_setup.sh
```

Comprobar:

```bash
$HOME/adios2-nohdf5/bin/adios2-config --version
```

Resultado esperado:

```text
ADIOS 2.12.0
```

## 4. Cambios necesarios en PSC

### CMake

No hay que editar `Makefile`. El cambio relevante esta en `src/CMakeLists.txt`.
Debe existir este target:

```cmake
add_psc_kappa_executable(psc_mirror_kappa3 psc_mirror_kappa.cxx 3.0 kappa3)
```

Comprobar:

```bash
grep -n "psc_mirror_kappa3" src/CMakeLists.txt
```

### Codigo de Mirror kappa

`src/psc_mirror_kappa.cxx` debe aceptar variables de entorno para poder hacer
pruebas pequenas y restart sin editar el codigo cada vez.

Comprobar que existen estas referencias:

```bash
grep -n "PSC_CHECKPOINT_EVERY\\|PSC_RESTART\\|PSC_NGRID\\|PSC_NICELL" src/psc_mirror_kappa.cxx
```

Debe incluir:

```cpp
psc_params.write_checkpoint_every_step =
  envInt("PSC_CHECKPOINT_EVERY", 7500);
```

Y en `main()`:

```cpp
if (const char* restart = std::getenv("PSC_RESTART")) {
  read_checkpoint_filename = restart;
}
```

Si no esta, editar:

```bash
nano src/psc_mirror_kappa.cxx
```

Variables utiles ya soportadas:

```text
PSC_NMAX
PSC_NGRID
PSC_NICELL
PSC_NP_Y
PSC_NP_Z
PSC_CHECKPOINT_EVERY
PSC_FIELDS_EVERY
PSC_PARTICLES_EVERY
PSC_RESTART
```

## 5. Compilar PSC con ADIOS2

Ruta recomendada: usar el script ya preparado.

```bash
chmod +x src/cosma_build_psc_adios2.sh
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

Si hay que crearlo manualmente:

```bash
nano src/cosma_build_psc_adios2.sh
```

Pegar esto:

```bash
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

cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-4}" --target psc_mirror_kappa3

grep -n "PSC_HAVE_ADIOS2" "$BUILD_DIR/src/include/PscConfig.h"
ldd "$BUILD_DIR/src/psc_mirror_kappa3" | egrep -i "adios|hdf5" || true
```

Guardar y ejecutar:

```bash
chmod +x src/cosma_build_psc_adios2.sh
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

Confirmaciones obligatorias:

```bash
grep -n "PSC_HAVE_ADIOS2" build-adios2-nohdf5/src/include/PscConfig.h
ldd build-adios2-nohdf5/src/psc_mirror_kappa3 | egrep -i "adios|hdf5"
```

Debe verse:

```text
#define PSC_HAVE_ADIOS2
libadios2_... => /cosma/home/dp433/dc-mart18/adios2-nohdf5/...
libhdf5... => /cosma/local/parallel-hdf5/.../1.14.4/...
```

## 6. Prueba pequena en login node

Esta prueba tarda segundos y verifica que se generan checkpoints `.bp`.

```bash
module purge
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

BASE=/cosma7/data/dp433/dc-mart18/pcseditado
ADIOS2_DIR=$HOME/adios2-nohdf5
HDF5_ROOT=/cosma/local/parallel-hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4

export PATH="$ADIOS2_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"

RUN=/cosma7/data/dp433/dc-mart18/adios2_smoke_mirror_kappa3_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
cp "$BASE/build-adios2-nohdf5/src/psc_mirror_kappa3" "$RUN/"
cp "$BASE/adios2cfg.xml" "$RUN/"
cd "$RUN"

export PSC_NGRID=16
export PSC_NP_Y=1
export PSC_NP_Z=1
export PSC_NICELL=1
export PSC_NMAX=4
export PSC_CHECKPOINT_EVERY=2
export PSC_FIELDS_EVERY=9999
export PSC_PARTICLES_EVERY=9999
export OMP_NUM_THREADS=1

mpirun -np 1 ./psc_mirror_kappa3 > smoke.out 2> smoke.err
ls -ld checkpoint_*.bp
$ADIOS2_DIR/bin/bpls checkpoint_2.bp | head -50
```

Resultado esperado:

```text
checkpoint_2.bp/
checkpoint_4.bp/
grid::...
mflds::...
mprts::...
```

## 7. Probar restart

Desde el mismo directorio `RUN` de la prueba:

```bash
mkdir restart
cp psc_mirror_kappa3 adios2cfg.xml restart/
cd restart

export PSC_RESTART=../checkpoint_2.bp
export PSC_NGRID=16
export PSC_NP_Y=1
export PSC_NP_Z=1
export PSC_NICELL=1
export PSC_NMAX=3
export PSC_CHECKPOINT_EVERY=2
export PSC_FIELDS_EVERY=9999
export PSC_PARTICLES_EVERY=9999
export OMP_NUM_THREADS=1

mpirun -np 1 ./psc_mirror_kappa3 > restart.out 2> restart.err
ls -ld checkpoint_3.bp
grep -E "Reading checkpoint|Writing checkpoint|Finished" restart.out
$ADIOS2_DIR/bin/bpls checkpoint_3.bp | head -30
```

Resultado esperado:

```text
**** Reading checkpoint...
**** Writing checkpoint...
checkpoint_3.bp/
```

## 8. Job Slurm de verificacion

Opcional si ya se hizo la prueba en login node, pero recomendado antes del job
grande.

```bash
mkdir -p /cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2
sbatch src/verify_mirror_kappa3_adios2.slurm
squeue -u dc-mart18
```

Revisar:

```bash
tail -100 /cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/verify_*.out
```

Debe aparecer:

```text
verify_ok=...
```

## 9. Enviar Mirror kappa=3 grande

Solo hacerlo despues de verificar ADIOS2 y restart.

```bash
sbatch src/submit_mirror_kappa3_adios2.slurm
```

Parametros del job grande:

```text
PSC_NGRID=1536
PSC_NP_Y=64
PSC_NP_Z=16
PSC_NICELL=1000
PSC_NMAX=1800000
PSC_CHECKPOINT_EVERY=7500
PSC_FIELDS_EVERY=750
PSC_PARTICLES_EVERY=1000
```

Recursos Slurm actuales:

```text
partition: cosma7-rp
account: dp433
ntasks: 1024
time: 48:00:00
```

Salidas esperadas:

```text
checkpoint_7500.bp/
checkpoint_15000.bp/
...
pfd.* / pfd_moments.*
prt_mirror_kappa3.*
```

## 10. Reiniciar job grande desde checkpoint

Editar una copia del script Slurm:

```bash
cp src/submit_mirror_kappa3_adios2.slurm src/restart_mirror_kappa3_adios2.slurm
nano src/restart_mirror_kappa3_adios2.slurm
```

Antes de `mpirun`, agregar:

```bash
export PSC_RESTART=/cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/run_JOBID/checkpoint_7500.bp
```

Luego enviar:

```bash
sbatch src/restart_mirror_kappa3_adios2.slurm
```

## 11. Errores comunes

Si aparece:

```text
write_checkpoint not available without adios2
```

El binario no fue compilado con `PSC_HAVE_ADIOS2`. Repetir la compilacion y
confirmar:

```bash
grep -n "PSC_HAVE_ADIOS2" build-adios2-nohdf5/src/include/PscConfig.h
```

Si aparecen errores de link con:

```text
H5Pset_fapl_mpio
H5Pset_dxpl_mpio
H5Pset_fapl_subfiling
```

Hay mezcla de HDF5 serial/Conda o ADIOS2 con HDF5 incompatible. Usar:

```text
ADIOS2_USE_HDF5=OFF
parallel_hdf5/1.14.4
HDF5_C_COMPILER_EXECUTABLE=$(which h5pcc)
```

Si `bpls checkpoint_2.bp` no muestra `grid`, `mflds`, `mprts`, el checkpoint no
es valido o la corrida fallo antes de cerrar ADIOS2.

## Verificacion ya realizada

En COSMA se verifico:

```text
ADIOS2: $HOME/adios2-nohdf5, version 2.12.0, MPI ON, HDF5 OFF
PSC: build-adios2-nohdf5/src/psc_mirror_kappa3
PSC_HAVE_ADIOS2: definido
HDF5: parallel_hdf5/1.14.4
```

Prueba realizada:

```text
RUN=/cosma7/data/dp433/dc-mart18/adios2_smoke_mirror_kappa3_20260617_133628
checkpoint_2.bp generado
checkpoint_4.bp generado
restart/checkpoint_3.bp generado desde PSC_RESTART=checkpoint_2.bp
```
