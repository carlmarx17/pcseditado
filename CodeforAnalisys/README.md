# Análisis de salidas PSC

Esta carpeta contiene la pipeline mantenida para analizar las corridas de
anisotropía `M_*_bM`, `F_*_bM`, `W_*_bM` y los casos Kappa/Maxwellian.

## Entrada esperada

Cada directorio de datos debe contener una sola corrida PSC:

```text
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_<caso>.<step>.h5
```

Los checkpoints ADIOS2 (`checkpoint_<step>.bp/`) son para restart de la
simulación. La pipeline de análisis trabaja con las salidas HDF5 de campos,
momentos y partículas.

## Uso rápido

Desde `CodeforAnalisys`:

```bash
make show-inputs DATA_DIR=/ruta/a/run CASE=M_S_bM
make analysis DATA_DIR=/ruta/a/run CASE=M_S_bM
```

También se puede ejecutar por caso:

```bash
make F_M_bM DATA_DIR=/ruta/a/F_M_bM
```

## Casos soportados

| `CASE` | Inestabilidad | Especie | Parámetros iniciales |
|---|---|---|---|
| `M_S_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=3.0` |
| `M_M_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=2.0` |
| `M_W_bM` | Mirror | ion | `beta_i_parallel=6`, `A_i=1.5` |
| `F_S_bM` | Firehose | ion | `beta_i_parallel=10`, `A_i=0.1` |
| `F_M_bM` | Firehose | ion | `beta_i_parallel=6`, `A_i=0.3` |
| `F_W_bM` | Firehose | ion | `beta_i_parallel=3`, `A_i=0.6` |
| `W_S_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=3.0` |
| `W_M_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=2.0` |
| `W_W_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=1.5` |

`psc_units.py` define los perfiles físicos y nombres de salida. No usar un
perfil de producción para analizar otro caso: `F_M_bM` no es equivalente a
`firehose_maxwellian`.

## Salidas

La salida queda bajo:

```text
analysis_results/<CASE>/
```

Subcarpetas principales:

```text
01_anisotropy/     evolución de A, beta y trayectoria Brazil
02_fields/         mapas de campos y fluctuaciones
03_particles/      VDF y momentos de partículas
04_spectra/        espectros y modos dominantes
05_diamagnetic/    corrientes diamagnéticas
06_heat_flux/      flujo de calor y regiones espaciales
```

## Definiciones usadas

Los scripts proyectan la presión térmica central respecto al campo local:

```text
P_parallel = b_hat . P . b_hat
P_perp = [Tr(P) - P_parallel] / 2
A = P_perp / P_parallel
beta_parallel = 2 P_parallel / |B|^2
```

Para Firehose se reportan dos convenciones:

```text
A_i = T_i_perp / T_i_parallel       # aumenta hacia 1
R_i = T_i_parallel / T_i_perp=1/A_i # disminuye hacia 1
```

## Documentación técnica

Para estructura interna de archivos, datasets HDF5 y responsabilidades de cada
script, ver:

```text
CodeforAnalisys/ANALISIS_ESTRUCTURA.md
```
