# Simulaciones de anisotropía PSC

Este archivo lista los códigos listos para compilar en `src/CMakeLists.txt`.
La guía de ejecución con checkpoints ADIOS2 en COSMA está en
`src/ADIOS2_COSMA_RUNBOOK.md`.

## Estructura del código

Los casos de anisotropía comparten la implementación común:

```text
src/psc_anisotropy_case.hxx
```

Cada `.cxx` define solo la configuración de su caso:

```text
PSC_CASE_LABEL
PSC_DISTRIBUTION_LABEL
PSC_OUTPUT_BASENAME
PSC_NMAX_DEFAULT
PSC_CHECKPOINT_EVERY_DEFAULT
PSC_MASS_RATIO
PSC_BETA_E_PAR
PSC_BETA_I_PAR
PSC_TI_PERP_OVER_TI_PAR
PSC_TE_PERP_OVER_TE_PAR
PSC_DOMAIN_DI
PSC_NGRID_DEFAULT
PSC_NICELL_DEFAULT
```

Los casos Kappa activan además:

```cpp
#define PSC_USE_KAPPA 1
#define PSC_KAPPA 3.0 // o 5.0 desde CMake
```

La inicialización Kappa sigue usando `SetupParticles::createKappaMultivariate`.
Los casos bi-Maxwellian usan la ruta Maxwellian normal de `SetupParticles`.

## Ejecutables listos

### Mirror bi-Maxwellian

| Ejecutable | Archivo | beta_i_parallel | Ti_perp/Ti_parallel | beta_e_parallel | Te_perp/Te_parallel |
|---|---|---:|---:|---:|---:|
| `psc_M_S_bM` | `psc_M_S_bM.cxx` | 5.0 | 3.0 | 1.0 | 1.0 |
| `psc_M_M_bM` | `psc_M_M_bM.cxx` | 5.0 | 2.0 | 1.0 | 1.0 |
| `psc_M_W_bM` | `psc_M_W_bM.cxx` | 6.0 | 1.5 | 1.0 | 1.0 |

### Firehose bi-Maxwellian

| Ejecutable | Archivo | beta_i_parallel | Ti_perp/Ti_parallel | beta_e_parallel | Te_perp/Te_parallel |
|---|---|---:|---:|---:|---:|
| `psc_F_S_bM` | `psc_F_S_bM.cxx` | 10.0 | 0.1 | 1.0 | 1.0 |
| `psc_F_M_bM` | `psc_F_M_bM.cxx` | 6.0 | 0.3 | 1.0 | 1.0 |
| `psc_F_W_bM` | `psc_F_W_bM.cxx` | 3.0 | 0.6 | 1.0 | 1.0 |

### Whistler bi-Maxwellian

| Ejecutable | Archivo | beta_i_parallel | Ti_perp/Ti_parallel | beta_e_parallel | Te_perp/Te_parallel |
|---|---|---:|---:|---:|---:|
| `psc_W_S_bM` | `psc_W_S_bM.cxx` | 1.0 | 1.0 | 0.5 | 3.0 |
| `psc_W_M_bM` | `psc_W_M_bM.cxx` | 1.0 | 1.0 | 0.5 | 2.0 |
| `psc_W_W_bM` | `psc_W_W_bM.cxx` | 1.0 | 1.0 | 0.5 | 1.5 |

### Kappa

| Ejecutable | Archivo base | kappa | beta_i_parallel | Ti_perp/Ti_parallel | grilla |
|---|---|---:|---:|---:|---:|
| `psc_mirror_kappa3` | `psc_mirror_kappa.cxx` | 3 | 5.0 | 3.0 | 1536 x 1536 |
| `psc_mirror_kappa5` | `psc_mirror_kappa.cxx` | 5 | 5.0 | 3.0 | 1536 x 1536 |
| `psc_firehose_kappa3` | `psc_firehose_kappa.cxx` | 3 | 10.0 | 0.1 | 1024 x 1024 |
| `psc_firehose_kappa5` | `psc_firehose_kappa.cxx` | 5 | 10.0 | 0.1 | 1024 x 1024 |

Todos los casos de producción usan `PSC_NICELL_DEFAULT=2000` y checkpoints
ADIOS2 activos por defecto cuando se compilan con `PSC_USE_ADIOS2=ON`.

## Compilación local

Ejemplo con el build existente:

```bash
cmake --build build --target psc_M_S_bM
cmake --build build --target psc_mirror_kappa3
```

Compilar todos los casos de anisotropía:

```bash
cmake --build build --target \
  psc_M_S_bM psc_M_M_bM psc_M_W_bM \
  psc_F_S_bM psc_F_M_bM psc_F_W_bM \
  psc_W_S_bM psc_W_M_bM psc_W_W_bM \
  psc_mirror_kappa3 psc_mirror_kappa5 \
  psc_firehose_kappa3 psc_firehose_kappa5
```

## Compilación ADIOS2 en COSMA

Para checkpoints ADIOS2, usar el build separado:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

Luego verificar y enviar:

```bash
sbatch src/verify_mirror_kappa3_adios2.slurm
sbatch src/submit_anisotropy_adios2.slurm
```

Para producción larga en COSMA:

```bash
sbatch src/submit_anisotropy_adios2_big.slurm
```

El submit usa `psc_mirror_kappa3` por defecto. Para otro target:

```bash
sbatch --export=PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
```
