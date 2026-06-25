# Simulaciones de anisotropía PSC

Catálogo de casos listos para compilar y sus parámetros físicos.
La guía de ejecución con ADIOS2 en COSMA está en `ADIOS2_COSMA_RUNBOOK.md`.

## Estructura del código

Todos los casos comparten `psc_anisotropy_case.hxx`. Cada archivo de caso solo
define la distribución, el régimen físico y sus parámetros:

```text
psc_mirror_bikappa3.cxx
psc_mirror_bikappa5.cxx
psc_firehose_bikappa3.cxx
psc_firehose_bikappa5.cxx
```

Los casos bi-Kappa activan `PSC_USE_KAPPA=1` y especifican `PSC_KAPPA`.
Los bi-Maxwellianos usan el valor por defecto `PSC_USE_KAPPA=0`.

```
PSC_CASE_LABEL / PSC_DISTRIBUTION_LABEL / PSC_OUTPUT_BASENAME
PSC_NMAX_DEFAULT / PSC_CHECKPOINT_EVERY_DEFAULT
PSC_MASS_RATIO / PSC_LAMBDA0 / PSC_VA_OVER_C
PSC_BETA_E_PAR / PSC_BETA_I_PAR
PSC_TI_PERP_OVER_TI_PAR / PSC_TE_PERP_OVER_TE_PAR
PSC_DOMAIN_DI / PSC_NGRID_DEFAULT / PSC_NICELL_DEFAULT
PSC_KAPPA
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
| Dominio | `20 d_i × 20 d_i` |
| Grilla | `1024×1024` |
| Resolución | `51.2 celdas/d_i` |
| Partículas por celda | 1500 (defecto) |
| Pasos máximos | 1,200,000 |
| Fronteras | Periódicas |
| Campos/momentos | cada 500 pasos |
| Partículas | cada 10,000 pasos |
| Checkpoint ADIOS2 | cada 5000 pasos |
| Continuidad de carga | cada 5000 pasos |
| Diagnóstico de energía | cada 5000 pasos (`diag.asc`) |
| Balanceo de carga | cada 2500 pasos |

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
| `psc_mirror_bimaxwellian_strong` | `psc_mirror_bimaxwellian_strong.cxx` | Strong | 5.0 | 3.0 | 1.0 | 1.0 | 1024×1024 |
| `psc_mirror_bimaxwellian_moderate` | `psc_mirror_bimaxwellian_moderate.cxx` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 | 1024×1024 |
| `psc_mirror_bimaxwellian_weak` | `psc_mirror_bimaxwellian_weak.cxx` | Weak | 6.0 | 1.5 | 1.0 | 1.0 | 1024×1024 |

## Firehose

Iones con exceso de temperatura paralela. Electrones isotrópicos.
Criterio: `beta_i_parallel * (1 - A_i) > 2`

| Ejecutable | Archivo | Régimen | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_firehose_bimaxwellian_strong` | `psc_firehose_bimaxwellian_strong.cxx` | Strong | 10.0 | 0.1 | 1.0 | 1.0 | 1024×1024 |
| `psc_firehose_bimaxwellian_moderate` | `psc_firehose_bimaxwellian_moderate.cxx` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 | 1024×1024 |
| `psc_firehose_bimaxwellian_weak` | `psc_firehose_bimaxwellian_weak.cxx` | Weak | 3.0 | 0.6 | 1.0 | 1.0 | 1024×1024 |

## Whistler

Electrones con exceso de temperatura perpendicular. Iones isotrópicos.
Criterio: `A_e > 1 + 0.21 / beta_e_parallel^0.6`

| Ejecutable | Archivo | Régimen | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_whistler_bimaxwellian_strong` | `psc_whistler_bimaxwellian_strong.cxx` | Strong | 1.0 | 1.0 | 0.5 | 3.0 | 1024×1024 |
| `psc_whistler_bimaxwellian_moderate` | `psc_whistler_bimaxwellian_moderate.cxx` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 | 1024×1024 |
| `psc_whistler_bimaxwellian_weak` | `psc_whistler_bimaxwellian_weak.cxx` | Weak | 1.0 | 1.0 | 0.5 | 1.5 | 1024×1024 |

## Bi-Kappa

| Ejecutable | Archivo | κ | beta_i_par | A_i | beta_e_par | A_e | Grilla |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_mirror_bikappa3` | `psc_mirror_bikappa3.cxx` | 3 | 5.0 | 3.0 | 1.0 | 1.0 | 1024×1024 |
| `psc_mirror_bikappa5` | `psc_mirror_bikappa5.cxx` | 5 | 5.0 | 3.0 | 1.0 | 1.0 | 1024×1024 |
| `psc_firehose_bikappa3` | `psc_firehose_bikappa3.cxx` | 3 | 10.0 | 0.1 | 1.0 | 1.0 | 1024×1024 |
| `psc_firehose_bikappa5` | `psc_firehose_bikappa5.cxx` | 5 | 10.0 | 0.1 | 1.0 | 1.0 | 1024×1024 |

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
cmake --build build --target psc_mirror_bimaxwellian_strong
cmake --build build --target psc_mirror_bikappa3
```

Todos los targets:
```bash
cmake --build build --target \
  psc_mirror_bimaxwellian_strong psc_mirror_bimaxwellian_moderate psc_mirror_bimaxwellian_weak \
  psc_firehose_bimaxwellian_strong psc_firehose_bimaxwellian_moderate psc_firehose_bimaxwellian_weak \
  psc_whistler_bimaxwellian_strong psc_whistler_bimaxwellian_moderate psc_whistler_bimaxwellian_weak \
  psc_mirror_bikappa3 psc_mirror_bikappa5 \
  psc_firehose_bikappa3 psc_firehose_bikappa5
```

## Compilación y ejecución con ADIOS2 en COSMA

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
sbatch src/submit_anisotropy_adios2.slurm
```

Para otro target:
```bash
sbatch --export=ALL,PSC_TARGET=psc_firehose_bikappa3 src/submit_anisotropy_adios2.slurm
```

Los intervalos de control pueden modificarse sin recompilar:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_bikappa3,PSC_BALANCE_INTERVAL=2500,PSC_CONTINUITY_EVERY=5000,PSC_ENERGIES_EVERY=5000 \
  src/submit_anisotropy_adios2.slurm
```

Los scripts limpian Conda y usan `srun` por defecto para evitar fallos de
`prted` al arrancar OpenMPI desde Slurm.

Detalles operativos en `ADIOS2_COSMA_RUNBOOK.md`.
