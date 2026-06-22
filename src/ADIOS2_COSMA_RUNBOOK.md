# ADIOS2 en COSMA para anisotropía PSC

Esta es la ruta operativa para compilar y correr los casos de anisotropía con
checkpoints ADIOS2 en COSMA. Mantiene separado el build ADIOS2 del build normal
del repositorio.

## Archivos relevantes

```text
src/cosma_adios2_env.sh              # carga módulos COSMA actuales y localiza ADIOS2
src/cosma_adios2_setup.sh            # instala ADIOS2 si no existe en $HOME/adios2
src/cosma_build_psc_adios2.sh        # compila los targets de anisotropía listos
src/verify_mirror_kappa3_adios2.slurm # prueba checkpoint + restart
src/submit_anisotropy_adios2.slurm    # job grande, selecciona ejecutable con PSC_TARGET
src/submit_anisotropy_adios2_big.slurm # job grande 72h, exclusive, notificación por email
adios2cfg.xml                        # config ADIOS2 copiada al directorio de run
```

Los scripts asumen:

```text
repo:       /cosma7/data/dp433/dc-mart18/pcseditado
ADIOS2:     $HOME/adios2 si existe; si no, $HOME/adios2-nohdf5
build:      /cosma7/data/dp433/dc-mart18/pcseditado/build-adios2-nohdf5
verify:     /cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2
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

## Verificar antes del job grande

```bash
mkdir -p /cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2
sbatch src/verify_mirror_kappa3_adios2.slurm
squeue -u dc-mart18
```

Revisar:

```bash
tail -100 /cosma7/data/dp433/dc-mart18/mirror_kappa3_adios2/verify_*.out
```

El test es correcto si aparece `verify_ok=...`. Genera:

```text
checkpoint_2.bp/
checkpoint_4.bp/
restart/checkpoint_3.bp/
```

## Enviar jobs grandes

```bash
sbatch src/submit_anisotropy_adios2.slurm
```

Para producción larga estilo plantilla COSMA, con `--exclusive`, 72 horas y
correo al terminar/fallar:

```bash
sbatch src/submit_anisotropy_adios2_big.slurm
```

También se puede usar cualquier ejecutable de anisotropía listo:

```bash
sbatch --export=PSC_TARGET=psc_M_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=PSC_TARGET=psc_F_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=PSC_TARGET=psc_W_S_bM src/submit_anisotropy_adios2.slurm
sbatch --export=PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
```

Para el Slurm grande se usa la misma forma:

```bash
sbatch --export=PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2_big.slurm
```

Los scripts limpian Conda y lanzan MPI con `srun` por defecto para evitar
fallos de `prted` al arrancar OpenMPI desde Slurm. Si COSMA cambia el plugin
PMIx, se puede probar:

```bash
sbatch --export=PSC_SRUN_MPI_TYPE=pmix_v4 src/submit_anisotropy_adios2.slurm
```

Solo si hace falta volver a OpenMPI directo:

```bash
sbatch --export=PSC_LAUNCHER=mpirun src/submit_anisotropy_adios2.slurm
```

Parámetros actuales por defecto:

```text
partition:              cosma7-rp
account:                dp433
ntasks:                 1024
time:                   48:00:00
PSC_NICELL:             2000
PSC_CHECKPOINT_EVERY:   7500
PSC_FIELDS_EVERY:       1000
PSC_PARTICLES_EVERY:    10000
```

La grilla y `PSC_NMAX` salen del ejecutable seleccionado, aunque se pueden
sobrescribir con variables de entorno si hace falta.

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
sbatch --export=PSC_TARGET=psc_mirror_kappa3 src/restart_anisotropy_adios2.slurm
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
