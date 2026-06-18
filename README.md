# PSC anisotropy workspace

Este repositorio contiene PSC más un conjunto de casos de anisotropía de
temperatura preparados para correr en COSMA con checkpoints ADIOS2.

La base del código sigue siendo PSC. La documentación operativa de este
workspace está organizada así:

| Documento | Uso |
|---|---|
| `src/SIMULACIONES.md` | Catálogo de ejecutables listos, parámetros físicos y compilación local. |
| `src/ADIOS2_COSMA_RUNBOOK.md` | Compilar con ADIOS2, verificar checkpoints/restart y enviar jobs en COSMA. |
| `src/SIMULACIONES_ANISOTROPIA_BM.md` | Detalle físico de los nueve casos bi-Maxwellian. |
| `src/SIMULACIONES_INESTABILIDADES_TEMPERATURA_ANALISIS.md` | Criterios físicos y plan de análisis para Mirror, Firehose y Whistler. |
| `CodeforAnalisys/README.md` | Cómo ejecutar la pipeline de análisis. |
| `CodeforAnalisys/ANALISIS_ESTRUCTURA.md` | Contrato técnico de archivos, lectores y salidas. |
| `src/SIMULACIONES_RECONNECTION.md` | Referencia separada para reconexión magnética. |

## Uso rápido en COSMA

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
sbatch src/verify_mirror_kappa3_adios2.slurm
sbatch src/submit_anisotropy_adios2.slurm
```

Por defecto se ejecuta `psc_mirror_kappa3`. `PSC_TARGET` puede ser cualquiera
de los ejecutables listados en
`src/SIMULACIONES.md`, por ejemplo `psc_M_S_bM`, `psc_F_S_bM`,
`psc_W_S_bM`, `psc_mirror_kappa3` o `psc_firehose_kappa3`.

```bash
sbatch --export=PSC_TARGET=psc_firehose_kappa3 src/submit_anisotropy_adios2.slurm
```

## Compilación local

```bash
cmake --build build --target psc_M_S_bM
cmake --build build --target psc_mirror_kappa3
```

## Análisis

```bash
cd CodeforAnalisys
make analysis DATA_DIR=/ruta/a/la/corrida CASE=M_S_bM
```

Los casos de producción usan 2000 partículas por celda por defecto y escriben
checkpoints cuando se compilan con `PSC_USE_ADIOS2=ON`.
