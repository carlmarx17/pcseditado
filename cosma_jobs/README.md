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
| `sim_firehose_bimaxwellian_moderate_40di.sh` | **NUEVO** — Firehose moderate Bi-Maxwelliana, caja grande | **40 d_i** | ngrid 576 |
| `sim_firehose_bikappa3_40di.sh` | **NUEVO** — Firehose Bi-Kappa-3, caja grande | **40 d_i** | ngrid 576 |

### `analisis/` — pipeline Python sobre corridas ya terminadas

| Script | Analiza | Partición / límite |
|---|---|---|
| `analisis_mirror_bimaxwellian_moderate_pauper.sh` | Mirror bimaxwellian moderate | cosma7-rp-pauper / 24h |
| `analisis_mirror_bimaxwellian_moderate_rp.sh` | Mirror bimaxwellian moderate (más prioridad) | cosma7-rp / 72h |
| `analisis_mirror_bikappa3_moderate_pauper.sh` | **NUEVO** — Mirror **bikappa3** moderate | cosma7-rp-pauper / 24h |

> Cada job de análisis reparte las 8 etapas independientes del target
> `common` del Makefile (`brazil fields particles spectral diamagnetic
> heatflux validate physics`) en 8 nodos, una etapa por nodo.

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

### C) Análisis de las corridas firehose 40 d_i (más adelante)

Cuando terminen, para analizarlas hace falta:
1. Un perfil en `CodeforAnalisys/psc_units.py` para
   `firehose_bimaxwellian_moderate_bigbox40` / `firehose_bikappa3_bigbox40`
   con **domain = 40 d_i** y **ngrid = 576** (si no, `check_resolution`
   del Makefile falla).
2. Un script de análisis copiado de los de `analisis/` cambiando
   `DATA_DIR` y `CASE`.

---

## Seguimiento de jobs

```bash
squeue -u dc-mart18            # cola
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
scancel <JOBID>                # cancelar
```
