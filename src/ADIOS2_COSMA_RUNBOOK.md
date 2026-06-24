# ADIOS2 en COSMA para anisotropía PSC

Esta es la ruta operativa para compilar y correr los casos de anisotropía con
checkpoints ADIOS2 en COSMA. Mantiene separado el build ADIOS2 del build normal
del repositorio.

## Archivos relevantes

```text
src/cosma_adios2_env.sh              # carga módulos COSMA actuales y localiza ADIOS2
src/cosma_adios2_setup.sh            # instala ADIOS2 si no existe en $HOME/adios2
src/cosma_build_psc_adios2.sh        # compila los targets de anisotropía listos
src/submit_anisotropy_adios2.slurm    # job de producción, selecciona ejecutable con PSC_TARGET
adios2cfg.xml                        # config ADIOS2 copiada al directorio de run
```

Los scripts asumen:

```text
repo:       /cosma7/data/dp433/dc-mart18/pcseditado
ADIOS2:     $HOME/adios2 si existe; si no, $HOME/adios2-nohdf5
build:      /cosma7/data/dp433/dc-mart18/pcseditado/build-adios2-nohdf5
runs:       /cosma7/data/dp433/dc-mart18/anisotropy_adios2
modules:    gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.14.4
```

## Preparar ADIOS2

Solo hace falta si no existe `adios2-config`. En tu COSMA actual ya aparece
como `$HOME/adios2/bin/adios2-config`, así que normalmente puedes saltar este
paso.

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
chmod +x src/cosma_adios2_setup.sh
BUILD_JOBS=4 src/cosma_adios2_setup.sh
```

Comprobar:

```bash
$HOME/adios2/bin/adios2-config --version
```

## Compilar PSC con ADIOS2

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
chmod +x src/cosma_build_psc_adios2.sh
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

Confirmaciones esperadas:

```bash
grep -n "PSC_HAVE_ADIOS2" build-adios2-nohdf5/src/include/PscConfig.h
ldd build-adios2-nohdf5/src/psc_mirror_kappa3 | grep -i adios
```

Debe aparecer `PSC_HAVE_ADIOS2` y una librería `libadios2_*` desde el
`ADIOS2_DIR` detectado. El script también comprueba que existan todos los
ejecutables de anisotropía listos.

## Flujo Kappa en COSMA

Los casos Kappa de producción son archivos completos e independientes:

```text
src/psc_mirror_kappa3.cxx      # kappa=3, mirror, nmax=1800000
src/psc_mirror_kappa5.cxx      # kappa=5, mirror, nmax=1800000
src/psc_firehose_kappa3.cxx    # kappa=3, firehose, nmax=1200000
src/psc_firehose_kappa5.cxx    # kappa=5, firehose, nmax=1200000
```

No se debe usar un script de verificación corto para producción. La ruta
correcta es compilar con ADIOS2 y enviar `src/submit_anisotropy_adios2.slurm`.

Compilar:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

Confirmar que los ejecutables Kappa existen y quedaron enlazados con ADIOS2:

```bash
ls -l build-adios2-nohdf5/src/psc_mirror_kappa3 \
      build-adios2-nohdf5/src/psc_mirror_kappa5 \
      build-adios2-nohdf5/src/psc_firehose_kappa3 \
      build-adios2-nohdf5/src/psc_firehose_kappa5

grep -n "PSC_HAVE_ADIOS2" build-adios2-nohdf5/src/include/PscConfig.h
ldd build-adios2-nohdf5/src/psc_mirror_kappa3 | grep -i adios
```

Enviar `mirror kappa=3`, que es el target por defecto:

```bash
sbatch src/submit_anisotropy_adios2.slurm
```

Enviar los otros Kappa:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_kappa5 src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_firehose_kappa5 src/submit_anisotropy_adios2.slurm
```

Comprobar que Slurm asignó la corrida grande:

```bash
squeue --me
scontrol show job JOBID | egrep 'NumNodes=|NumTasks=|Command=|SubmitLine='
```

Debe verse `ST R`, `NODES 37` en `squeue`, y `NumTasks=1024` en
`scontrol`. Si aparece muy poco tiempo en `CG` o termina enseguida, revisar el
`.out` y confirmar que no se sobrescribió `PSC_NMAX` con una prueba corta.

Monitorear una corrida ya creada:

```bash
JOBID=11463779
TARGET=psc_mirror_kappa3
RUN=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/${TARGET}_${JOBID}

tail -f "$RUN/diag.asc"
tail -50 /cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_aniso_${JOBID}.out
```

Ver si ya empezó a escribir salidas:

```bash
watch -n 30 "ls -ltr $RUN | tail"
watch -n 60 "ls -d $RUN/checkpoint_*.bp 2>/dev/null | tail"
```

Para `psc_mirror_kappa3` y `psc_mirror_kappa5`, las salidas por defecto son:

```text
campos/momentos: cada 750 pasos
partículas:      cada 1000 pasos
checkpoints:     cada 7500 pasos
nmax:            1800000
```

Para `psc_firehose_kappa3` y `psc_firehose_kappa5`:

```text
campos/momentos: cada 500 pasos
partículas:      cada 1000 pasos
checkpoints:     cada 5000 pasos
nmax:            1200000
```

## Enviar jobs

```bash
sbatch src/submit_anisotropy_adios2.slurm
```

También se puede usar cualquier ejecutable de anisotropía listo:

```bash
sbatch --export=ALL,PSC_TARGET=psc_M_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_F_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_W_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_mirror_kappa5 src/submit_anisotropy_adios2.slurm
sbatch --export=ALL,PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
```

Los scripts limpian Conda y lanzan MPI con `srun` por defecto para evitar
fallos de `prted` al arrancar OpenMPI desde Slurm. El wrapper solo selecciona
plugins PMIx para `srun`; no usa `pmi2` automáticamente porque con OpenMPI 5
puede aparecer:

```text
No PMIx server was reachable ... 1024 singletons will be started
```

Ese síntoma significa que los ranks no formaron un `MPI_COMM_WORLD` normal y
puede acabar en OOM masivo. Si COSMA cambia el plugin PMIx, se puede probar:

```bash
sbatch --export=ALL,PSC_SRUN_MPI_TYPE=pmix_v4 src/submit_anisotropy_adios2.slurm
```

Solo si hace falta volver a OpenMPI directo:

```bash
sbatch --export=ALL,PSC_LAUNCHER=mpirun src/submit_anisotropy_adios2.slurm
```

Parámetros actuales por defecto:

```text
partition:              cosma7-rp
account:                dp433
nodes:                  37
ntasks-per-node:        28
ntasks:                 1024
time:                   48:00:00
```

La grilla, `PSC_NMAX`, partículas por celda y frecuencias de salida salen del
ejecutable seleccionado. Se pueden sobrescribir con variables de entorno si
hace falta hacer una prueba corta, por ejemplo:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_kappa3,PSC_NMAX=1000 src/submit_anisotropy_adios2.slurm
```

No usar `--ntasks=616` con estos ejecutables sin cambiar también
`PSC_NP_Y/PSC_NP_Z` y asegurar que la grilla se divide exactamente. Los casos
actuales usan `64 x 16 = 1024` patches.

Cada job escribe en:

```text
/cosma7/data/dp433/dc-mart18/anisotropy_adios2/PSC_TARGET_JOBID/
```

Salidas esperadas:

```text
checkpoint_7500.bp/
checkpoint_15000.bp/
pfd.*
pfd_moments.*
prt_mirror_kappa3.*
```

## Restart desde checkpoint

Crear una copia del script grande:

```bash
cp src/submit_anisotropy_adios2.slurm src/restart_anisotropy_adios2.slurm
```

En la copia, antes de `psc_mpi_run`, añadir:

```bash
export PSC_RESTART=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/PSC_TARGET_JOBID/checkpoint_7500.bp
```

Luego enviar:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_kappa3 src/restart_anisotropy_adios2.slurm
```

## Problemas comunes

Para revisar jobs, el usuario va con `-u`; no va dentro de `--format`:

```bash
sacct -u dc-mart18 --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

Si `sacct --format=...` termina con `Invalid field requested: "dc-mart18"`,
el comando mezcló el usuario con las columnas de salida.

Si aparece `write_checkpoint not available without adios2`, se está usando el
build equivocado. Recompilar con `src/cosma_build_psc_adios2.sh` y correr
`build-adios2-nohdf5/src/psc_mirror_kappa3`.

Si `adios2-config` no aparece, revisar:

```bash
export PATH="$HOME/adios2-nohdf5/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/adios2-nohdf5/lib64:$HOME/adios2-nohdf5/lib:${LD_LIBRARY_PATH:-}"
```

Si tu instalación es la actual de COSMA, usa:

```bash
export ADIOS2_DIR="$HOME/adios2"
export PATH="$ADIOS2_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ADIOS2_DIR/lib64:$ADIOS2_DIR/lib:${LD_LIBRARY_PATH:-}"
```

También se detecta automáticamente `ADIOS2_DIR=$HOME/build_adios2_nohdf5/build`
si ese build existe.

No cargues `cosma/2018`, `hdf5/1.10.3` ni `adios2/2.7.1`: esos módulos no
aparecen en el árbol actual de COSMA. Los scripts cargan
`gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.14.4`.

Si el job falla al copiar el binario, confirmar que existe:

```bash
ls -l build-adios2-nohdf5/src/psc_mirror_kappa3
```
