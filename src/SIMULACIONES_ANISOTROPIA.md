# Simulaciones de anisotropía PSC

Catálogo de casos listos para compilar y sus parámetros físicos.
La guía de ejecución con ADIOS2 en COSMA está en `ADIOS2_COSMA_RUNBOOK.md`.

## Estructura del código

Todos los casos comparten `psc_anisotropy_case.hxx`. Cada `.cxx` define solo
su configuración mediante macros:

```
PSC_CASE_LABEL / PSC_DISTRIBUTION_LABEL / PSC_OUTPUT_BASENAME
PSC_NMAX_DEFAULT / PSC_CHECKPOINT_EVERY_DEFAULT
PSC_MASS_RATIO / PSC_LAMBDA0 / PSC_VA_OVER_C
PSC_BETA_E_PAR / PSC_BETA_I_PAR
PSC_TI_PERP_OVER_TI_PAR / PSC_TE_PERP_OVER_TE_PAR
PSC_DOMAIN_DI / PSC_NGRID_DEFAULT / PSC_NICELL_DEFAULT
```

Los casos Kappa activan además:
```cpp
#define PSC_USE_KAPPA 1
#define PSC_KAPPA 3.0  // o 5.0 desde CMake
```

## Configuración común

| Parámetro | Valor |
|---|---:|
| Configuración PSC | `PscConfig1vbecSingle<dim_yz>` |
| Campo de fondo | `B0 = 0.05` |
| `vA/c` | 0.05 |
| `mi/me` | 200 |
| `lambda0` | 20 |
| Densidad inicial | 1.0 |
| Partículas por celda | 2000 (defecto) |
| Fronteras | Periódicas |
| Checkpoint ADIOS2 | cada 7500 pasos |

El campo paralelo es `z`:
```
T_parallel = T_z
T_perp = (T_x + T_y) / 2
A = T_perp / T_parallel
```

## Mirror

Iones con exceso de temperatura perpendicular. Electrones isotrópicos.
Criterio: `beta_i_parallel * (A_i - 1) > 1`

| Ejecutable | Archivo | Régimen | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_M_S_bM` | `psc_M_S_bM.cxx` | Strong | 5.0 | 3.0 | 1.0 | 1.0 | 1408×1408 |
| `psc_M_M_bM` | `psc_M_M_bM.cxx` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 | 1408×1408 |
| `psc_M_W_bM` | `psc_M_W_bM.cxx` | Weak | 6.0 | 1.5 | 1.0 | 1.0 | 1408×1408 |

## Firehose

Iones con exceso de temperatura paralela. Electrones isotrópicos.
Criterio: `beta_i_parallel * (1 - A_i) > 2`

| Ejecutable | Archivo | Régimen | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_F_S_bM` | `psc_F_S_bM.cxx` | Strong | 10.0 | 0.1 | 1.0 | 1.0 | 1408×1408 |
| `psc_F_M_bM` | `psc_F_M_bM.cxx` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 | 1408×1408 |
| `psc_F_W_bM` | `psc_F_W_bM.cxx` | Weak | 3.0 | 0.6 | 1.0 | 1.0 | 1408×1408 |

## Whistler

Electrones con exceso de temperatura perpendicular. Iones isotrópicos.
Criterio: `A_e > 1 + 0.21 / beta_e_parallel^0.6`

| Ejecutable | Archivo | Régimen | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_W_S_bM` | `psc_W_S_bM.cxx` | Strong | 1.0 | 1.0 | 0.5 | 3.0 | 1408×1408 |
| `psc_W_M_bM` | `psc_W_M_bM.cxx` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 | 1408×1408 |
| `psc_W_W_bM` | `psc_W_W_bM.cxx` | Weak | 1.0 | 1.0 | 0.5 | 1.5 | 1408×1408 |

## Kappa

| Ejecutable | Archivo base | κ | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_mirror_kappa3` | `psc_mirror_kappa.cxx` | 3 | 5.0 | 3.0 | 1.0 | 1.0 | 1536×1536 |
| `psc_mirror_kappa5` | `psc_mirror_kappa.cxx` | 5 | 5.0 | 3.0 | 1.0 | 1.0 | 1536×1536 |
| `psc_firehose_kappa3` | `psc_firehose_kappa.cxx` | 3 | 10.0 | 0.1 | 1.0 | 1.0 | 1024×1024 |
| `psc_firehose_kappa5` | `psc_firehose_kappa.cxx` | 5 | 10.0 | 0.1 | 1.0 | 1.0 | 1024×1024 |

## Salidas

Los casos escriben campos, momentos y partículas:
```
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_<basename>.<step>.h5        # región central ~20% de cada dirección
checkpoint_<step>.bp/            # solo con ADIOS2
```

## Compilación local

```bash
cmake --build build --target psc_M_S_bM
cmake --build build --target psc_mirror_kappa3
```

Todos los targets:
```bash
cmake --build build --target \
  psc_M_S_bM psc_M_M_bM psc_M_W_bM \
  psc_F_S_bM psc_F_M_bM psc_F_W_bM \
  psc_W_S_bM psc_W_M_bM psc_W_W_bM \
  psc_mirror_kappa3 psc_mirror_kappa5 \
  psc_firehose_kappa3 psc_firehose_kappa5
```

## Compilación y ejecución con ADIOS2 en COSMA

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
sbatch src/verify_mirror_kappa3_adios2.slurm
sbatch src/submit_anisotropy_adios2.slurm       # 48h
sbatch src/submit_anisotropy_adios2_big.slurm    # 72h, exclusive
```

Para otro target:
```bash
sbatch --export=PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
```

Los scripts limpian Conda y usan `srun` por defecto para evitar fallos de
`prted` al arrancar OpenMPI desde Slurm.

Detalles operativos en `ADIOS2_COSMA_RUNBOOK.md`.
