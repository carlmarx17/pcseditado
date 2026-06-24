# PSC con ADIOS2 en COSMA7

Procedimiento operativo para compilar, validar, ejecutar y reiniciar los casos
de anisotropía de PSC con checkpoints ADIOS2.

## Estado validado

Configuración comprobada el 24 de junio de 2026:

```text
Repositorio: /cosma7/data/dp433/dc-mart18/pcseditado
Build único: /cosma7/data/dp433/dc-mart18/pcseditado/build
ADIOS2:      $HOME/adios2 (versión 2.12.0.182)
Partición:   cosma7-rp
Cuenta:      dp433
Compilador:  gnu_comp/14.1.0
MPI:         openmpi/5.0.3
HDF5:        parallel_hdf5/1.14.4
Launcher:    mpirun
```

ADIOS2 está realmente activado porque:

```bash
grep -n PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_kappa3 | grep adios2
```

La salida esperada contiene:

```text
#define PSC_HAVE_ADIOS2
libadios2_cxx_mpi.so
libadios2_core_mpi.so
```

También se validaron:

- escritura de checkpoints BP5;
- lectura de un checkpoint y continuación del cálculo;
- ejecución MPI con 28 procesos en un nodo;
- ejecución MPI con 56 procesos en dos nodos;
- ejecución de producción con 1024 procesos en 37 nodos.

## Archivos importantes

```text
src/cosma_adios2_env.sh
src/cosma_adios2_setup.sh
src/cosma_build_psc_adios2.sh
src/submit_anisotropy_adios2.slurm
adios2cfg.xml
build/
```

Solo debe existir una carpeta de compilación llamada `build`. No usar
`build-adios2` ni `build-adios2-nohdf5`.

## 1. Entrar y cargar el entorno

```bash
ssh dc-mart18@login7.cosma.dur.ac.uk
cd /cosma7/data/dp433/dc-mart18/pcseditado
source src/cosma_adios2_env.sh
```

Comprobar el entorno:

```bash
adios2-config --version
command -v mpirun
command -v h5pcc
module list
```

El script limpia Conda, carga los módulos compatibles y configura
`ADIOS2_DIR`, `PATH` y `LD_LIBRARY_PATH`.

## 2. Instalar ADIOS2 si falta

Este paso no es necesario mientras exista:

```text
$HOME/adios2/bin/adios2-config
```

Comprobar:

```bash
test -x "$HOME/adios2/bin/adios2-config"
$HOME/adios2/bin/adios2-config --version
```

Si no existe:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_adios2_setup.sh
```

No mezclar esta instalación con módulos antiguos de ADIOS2, OpenMPI o HDF5.

## 3. Crear el build único

Para una reconstrucción limpia:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
rm -rf build
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

El script configura CMake con:

```text
PSC_USE_ADIOS2=ON
USE_CUDA=OFF
USE_VPIC=OFF
BUILD_TESTING=OFF
```

Nota de COSMA7: actualmente `cmake` está disponible en el login, pero no
necesariamente en los nodos de cómputo. Un job de compilación puede fallar con:

```text
cmake: command not found
```

No confundir ese problema de entorno con un fallo de ADIOS2. Si se quiere
compilar completamente en un nodo, primero debe instalarse o exponerse una
versión de CMake accesible desde ese nodo.

## 4. Verificar la compilación

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
source src/cosma_adios2_env.sh

grep -n PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_kappa3 | grep -i adios
```

Comprobar los ejecutables:

```bash
ls -l \
  build/src/psc_mirror_kappa3 \
  build/src/psc_mirror_kappa5 \
  build/src/psc_firehose_kappa3 \
  build/src/psc_firehose_kappa5 \
  build/src/psc_M_S_bM \
  build/src/psc_F_S_bM \
  build/src/psc_W_S_bM
```

No enviar producción si falta `PSC_HAVE_ADIOS2` o si `ldd` muestra
`not found`.

## 5. Por qué se usa mpirun

El launcher validado es:

```text
mpirun -np $SLURM_NTASKS
```

No usar `srun --mpi=pmi2` con OpenMPI 5. En una prueba anterior produjo:

```text
No PMIx server was reachable, but a PMI1/2 was detected.
1024 singletons will be started.
```

Eso inicia rangos MPI independientes, consume memoria masivamente y termina
en OOM. `src/cosma_adios2_env.sh` usa `mpirun` por defecto.

## 6. Enviar un caso

El target por defecto es `psc_mirror_kappa3`:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
sbatch src/submit_anisotropy_adios2.slurm
```

Otros casos:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_kappa5 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_firehose_kappa3 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_firehose_kappa5 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_M_S_bM \
  src/submit_anisotropy_adios2.slurm
```

El script solicita:

```text
37 nodos
28 procesos por nodo
1024 procesos MPI
48 horas
partición cosma7-rp
cuenta dp433
```

No fija un `--nodelist`: Slurm selecciona nodos libres.

## 7. Fallos de prólogo de Slurm

Si un job termina inmediatamente con:

```text
State=CANCELLED
Reason=Prolog
ExitCode=0:0
```

y no genera `.out` ni `.err`, el script PSC nunca llegó a ejecutarse. Es un
fallo del prólogo del nodo, no de ADIOS2.

Consultar:

```bash
scontrol show job -dd JOBID
sacct -j JOBID \
  --format=JobID,State,ExitCode,Elapsed,NodeList,Reason -X
```

Si COSMA todavía no ha reparado los nodos afectados, se pueden excluir
temporalmente al enviar:

```bash
sbatch --exclude='m[7031-7043]' src/submit_anisotropy_adios2.slurm
```

La exclusión debe ser temporal y basarse en fallos de prólogo observados; no
se debe convertir en una lista fija permanente.

## 8. Confirmar que el job usa ADIOS2

```bash
JOBID=12345678
LOG=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_aniso_${JOBID}.out

grep -E '^(target|job|nodes|ntasks|adios2_dir|adios2_config|launcher)=' "$LOG"
```

Debe mostrar:

```text
adios2_dir=/cosma/home/dp433/dc-mart18/adios2
adios2_config=/cosma/home/dp433/dc-mart18/adios2/bin/adios2-config
launcher=mpirun
launcher=mpirun -np 1024 ./psc_mirror_kappa3
```

Verificar además el binario copiado al directorio de ejecución:

```bash
TARGET=psc_mirror_kappa3
RUN=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/${TARGET}_${JOBID}

ldd "$RUN/$TARGET" | grep -i adios
```

## 9. Confirmar la escritura de checkpoints

Para `psc_mirror_kappa3` y `psc_mirror_kappa5`:

```text
campos y momentos: cada 750 pasos
partículas:        cada 1000 pasos
checkpoint:        cada 7500 pasos
nmax:              1800000
```

Para `psc_firehose_kappa3` y `psc_firehose_kappa5`:

```text
campos y momentos: cada 500 pasos
partículas:        cada 1000 pasos
checkpoint:        cada 5000 pasos
nmax:              1200000
```

Antes del primer intervalo no habrá una carpeta `checkpoint_*.bp`. Eso no
significa que ADIOS2 esté desactivado.

Cuando se alcance el intervalo:

```bash
find "$RUN" -maxdepth 1 -type d -name 'checkpoint_*.bp' -print
find "$RUN" -maxdepth 2 -type f -path '*.bp/*' -ls | head
```

Una carpeta BP5 válida contiene archivos como:

```text
data.0
md.0
md.idx
profiling.json
```

## 10. Monitorizar

```bash
squeue -j "$JOBID" -o '%.18i %.16P %.24j %.2t %.10M %.4D %R'
tail -f "$LOG"
```

Diagnóstico y resultados:

```bash
tail -f "$RUN/diag.asc"
ls -ltr "$RUN" | tail
du -sh "$RUN"
```

No ejecutar la simulación directamente en el login.

## 11. Reiniciar desde un checkpoint

Ejemplo:

```bash
export PSC_RESTART=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_kappa3_JOBID/checkpoint_7500.bp

sbatch \
  --export=ALL,PSC_TARGET=psc_mirror_kappa3,PSC_RESTART="$PSC_RESTART" \
  src/submit_anisotropy_adios2.slurm
```

El ejecutable debe imprimir:

```text
**** Reading checkpoint...
```

y continuar desde el paso almacenado.

## 12. Prueba corta opcional

Las variables de entorno permiten reducir el problema:

```bash
sbatch \
  --export=ALL,PSC_TARGET=psc_mirror_kappa3,PSC_NMAX=4,PSC_NGRID=16,PSC_NP_Y=1,PSC_NP_Z=1,PSC_NICELL=16,PSC_CHECKPOINT_EVERY=2,PSC_FIELDS_EVERY=2,PSC_PARTICLES_EVERY=2 \
  src/submit_anisotropy_adios2.slurm
```

Para una prueba hay que ajustar también los recursos Slurm. No enviar esa
configuración con 37 nodos.

La prueba correcta debe:

1. terminar con código cero;
2. crear `checkpoint_2.bp`;
3. contener `data.0`, `md.0` y `md.idx`;
4. permitir un restart con `PSC_RESTART`.

## Problemas frecuentes

### `write_checkpoint not available without adios2`

El ejecutable fue compilado sin ADIOS2 o se tomó de otro build:

```bash
grep PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_kappa3 | grep -i adios
```

### `libadios2_*.so => not found`

```bash
source src/cosma_adios2_env.sh
echo "$ADIOS2_DIR"
echo "$LD_LIBRARY_PATH"
```

### OOM inmediato con 1024 procesos

Buscar en el error:

```text
1024 singletons will be started
```

Si aparece, se usó `srun --mpi=pmi2`. Volver a `mpirun`.

### Job cancelado sin logs

Consultar `Reason=Prolog`. Si aparece, revisar o excluir temporalmente los
nodos afectados.

### No aparece todavía `checkpoint_*.bp`

Comprobar el paso actual y el intervalo de checkpoint. Por ejemplo,
`psc_mirror_kappa3` no escribe el primer checkpoint hasta el paso 7500.
