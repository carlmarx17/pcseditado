# cosma_jobs — scripts SLURM para COSMA7 (cuenta dp433)

Todos los scripts `.sh` de envío a SLURM, organizados por tipo y con
nombres que dicen qué corren. Los paths internos son **absolutos**
(`/cosma7/data/dp433/dc-mart18/...`), así que se envían igual desde la
raíz del repo sin importar dónde estén.

Raíz del repo en COSMA: `/cosma7/data/dp433/dc-mart18/pcseditado`

---

## Qué hay aquí

### `simulacion/` — corridas PSC (PIC)

| Script | Qué corre | Caja | Malla |
|---|---|---|---|
| `sim_mirror_bimaxwellian_strong_MSbM.sh` | Mirror fuerte Bi-Maxwelliana (legacy `psc_M_S_bM`) | 20 d_i | — |
| `sim_mirror_kappa3.sh` | Mirror Kappa-3 (`psc_mirror_kappa3`) | 20 d_i | — |
| `sim_mirror_bikappa3_moderate.sh` | Mirror Bi-Kappa-3 moderate (compara vs bimaxwellian moderate) | 20 d_i | ngrid 576 |
| `sim_firehose_bimaxwellian_moderate_40di.sh` | Firehose moderate Bi-Maxwelliana, caja grande | **40 d_i** | ngrid 1152 |
| `sim_firehose_bikappa3_40di.sh` | Firehose Bi-Kappa-3, caja grande | **40 d_i** | ngrid 1152 |

### `analisis/` — pipeline Python sobre corridas ya terminadas

| Script | Analiza | Partición / límite |
|---|---|---|
| `analisis_mirror_bimaxwellian_moderate_pauper.sh` | Mirror bimaxwellian moderate | cosma7-rp-pauper / 24h |
| `analisis_mirror_bimaxwellian_moderate_rp.sh` | Mirror bimaxwellian moderate (más prioridad) | cosma7-rp / 72h |
| `analisis_mirror_bikappa3_moderate_pauper.sh` | Mirror **bikappa3** moderate | cosma7-rp-pauper / 24h |
| `analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh` | **NUEVO** — Firehose bimaxwellian moderate, caja 40 d_i | cosma7-rp-pauper / 24h |
| `analisis_firehose_bikappa3_bigbox40_pauper.sh` | **NUEVO** — Firehose bikappa3, caja 40 d_i | cosma7-rp-pauper / 24h |

> Cada job de análisis reparte las 8 etapas independientes del target
> `common` del Makefile (`brazil fields particles spectral diamagnetic
> heatflux validate physics`) en 8 nodos, una etapa por nodo.

#### Cuántas figuras genera cada etapa

`fields` y `diamagnetic` filtran los snapshots con `paso % SNAPSHOT_EVERY == 0`
(más siempre el primero y el último). Con la cadencia de salida de PSC
(`PSC_FIELDS_EVERY_DEFAULT = 500`) y `nmax = 1 200 000`, una corrida deja
~2400 snapshots de campos:

| `SNAPSHOT_EVERY` | PNG guardados por panel |
|---|---|
| 500 (= toda la salida) | ~2400 |
| 10 000 | 121 |
| **100 000** (default) | **13** |
| 200 000 | 7 |

`GIF_EVERY` (default 10 000) es independiente: esos frames se renderizan
en memoria y solo quedan dentro del `.gif`, no como PNG sueltos. Ambos se
pueden pasar al job sin editarlo:

```bash
sbatch --export=ALL,SNAPSHOT_EVERY=200000,GIF_EVERY=20000 \
  cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
```

> Si un análisis viejo dejó miles de PNG, es que el checkout de COSMA es
> anterior al commit que introdujo este filtro (`c18f8dd6e` /
> `2ec0e39a0`). Hacer `git pull` en `/cosma7/data/dp433/dc-mart18/pcseditado`
> antes de volver a enviar.

---

## Renombrado (mapa viejo → nuevo)

Los scripts que estaban sueltos en la raíz se movieron aquí:

| Antes (raíz) | Ahora |
|---|---|
| `job_MSbM.sh` | `simulacion/sim_mirror_bimaxwellian_strong_MSbM.sh` |
| `job_kappa.sh` | `simulacion/sim_mirror_kappa3.sh` |
| `job_mirror_bikappa3_moderate.sh` | `simulacion/sim_mirror_bikappa3_moderate.sh` |
| `job_analysis_mirror_bimaxwellian_moderate.sh` | `analisis/analisis_mirror_bimaxwellian_moderate_pauper.sh` |
| `job_analysis_mirror_bimaxwellian_moderate_cosma7.sh` | `analisis/analisis_mirror_bimaxwellian_moderate_rp.sh` |

---

## Paso a paso

Todo se ejecuta desde la raíz del repo en COSMA:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
```

### A) Análisis del caso Bi-Kappa moderate (lo pedido)

Ya apunta a `DATA_DIR=.../psc_mirror_bikappa3_moderate_11618877` y
`CASE=mirror_bikappa3_moderate`. No hay que compilar nada.

```bash
sbatch cosma_jobs/analisis/analisis_mirror_bikappa3_moderate_pauper.sh
```

Salidas: `/cosma7/data/dp433/dc-mart18/logs/analysis_bikappa3_moderate.<JOBID>.{out,err}`
y un log por etapa `analysis_mirror_bikappa3_moderate_<etapa>.<JOBID>.log`.
Resultados: `CodeforAnalisys/../analysis_results/mirror_bikappa3_moderate/`.

> Si cambia el JOBID de la corrida bikappa (el `_11618877`), edita la
> línea `DATA_DIR=` del script antes de enviar.

### B) Firehose caja 40 d_i (Bi-Maxwelliana y Bi-Kappa)

El tamaño de caja **es compile-time** (`#define PSC_DOMAIN_DI` en
`src/psc_anisotropy_case.hxx`, default 20 d_i) — **no** es variable de
entorno. Por eso cada caso de 40 d_i es un ejecutable propio y hay que
**compilarlo una vez** antes de enviar.

**1) Compilar los dos ejecutables (una sola vez):**

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
  PSC_TARGETS="psc_firehose_bimaxwellian_moderate_bigbox40 psc_firehose_bikappa3_bigbox40" \
  src/cosma_build_psc_adios2.sh
```

Comprobar que quedaron:

```bash
ls -l build/src/psc_firehose_bimaxwellian_moderate_bigbox40 \
      build/src/psc_firehose_bikappa3_bigbox40
```

**2) Enviar las corridas:**

```bash
sbatch cosma_jobs/simulacion/sim_firehose_bimaxwellian_moderate_40di.sh
sbatch cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh
```

Cada job crea su carpeta en
`/cosma7/data/dp433/dc-mart18/anisotropy_adios2/<target>_<JOBID>/`.

> **Nota de resolución (a propósito):** con 40 d_i y `ngrid=576` la
> resolución baja de ~28.8 a ~14.4 celdas/d_i. Elegido así por costo.
> El costo de cómputo por paso es ~igual al de una corrida de 20 d_i
> (mismas 576² celdas y 1000 ppc), solo cambia dx físico. Si algún día
> se quiere la misma resolución que los otros casos, hay que usar
> `PSC_NGRID=1152` (≈4× más caro) — se puede pasar por entorno:
> `sbatch --export=ALL,PSC_NGRID=1152 cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh`.

### C) Análisis de las corridas firehose 40 d_i

Los dos requisitos ya están hechos: los perfiles
`firehose_bimaxwellian_moderate_bigbox40` y `firehose_bikappa3_bigbox40`
existen en `CodeforAnalisys/psc_units.py` (domain 40 d_i, ngrid 1152), y
los scripts de análisis están en `analisis/`.

```bash
sbatch cosma_jobs/analisis/analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh
sbatch cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
```

> **bikappa3 tiene tres carpetas de corrida** (`_11654252`, `_11657054`,
> `_11657093`) porque `sim_firehose_bikappa3_40di.sh` crea un `RUN_DIR`
> nuevo por `SLURM_JOB_ID` en cada envío; los snapshots quedan repartidos
> y ninguna carpeta tiene la corrida completa. Sin `DATA_DIR` el job elige
> la que tenga más snapshots y deja las tres listadas en el log. Revisar
> los `.out`/`.err` de cada job id antes de dar el resultado por bueno, y
> si hace falta forzar la carpeta:
>
> ```bash
> sbatch --export=ALL,DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_firehose_bikappa3_bigbox40_11657093 \
>   cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
> ```

---

## Seguimiento de jobs

```bash
squeue -u dc-mart18            # cola
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
scancel <JOBID>                # cancelar
```
