# cosma_jobs — SLURM scripts for COSMA7 (account dp433)

All SLURM submission scripts (`.sh`), organized by type and named after what they
run. Internal paths are **absolute** (`/cosma7/data/dp433/dc-mart18/...`), so they
submit the same way from the repository root regardless of where they live.

Repository root on COSMA: `/cosma7/data/dp433/dc-mart18/pcseditado`

---

## What is here

### `simulacion/` — PSC (PIC) runs

| Script | What it runs | Box | Grid |
|---|---|---|---|
| `sim_mirror_bimaxwellian_strong_MSbM.sh` | Strong bi-Maxwellian mirror (legacy `psc_M_S_bM`) | 20 d_i | — |
| `sim_mirror_kappa3.sh` | Mirror Kappa-3 (`psc_mirror_kappa3`) | 20 d_i | — |
| `sim_mirror_bikappa3_moderate.sh` | Moderate bi-Kappa-3 mirror (compares against bimaxwellian moderate) | 20 d_i | ngrid 576 |
| `sim_firehose_bimaxwellian_moderate_40di.sh` | Moderate bi-Maxwellian firehose, big box | **40 d_i** | ngrid 1152 |
| `sim_firehose_bikappa3_40di.sh` | Bi-Kappa-3 firehose, big box | **40 d_i** | ngrid 1152 |

### `analisis/` — Python pipeline over finished runs

| Script | Analyses | Partition / limit |
|---|---|---|
| `analisis_mirror_bimaxwellian_moderate_pauper.sh` | Mirror bimaxwellian moderate | cosma7-rp-pauper / 24h |
| `analisis_mirror_bimaxwellian_moderate_rp.sh` | Mirror bimaxwellian moderate (higher priority) | cosma7-rp / 72h |
| `analisis_mirror_bikappa3_moderate_pauper.sh` | Mirror **bikappa3** moderate | cosma7-rp-pauper / 24h |
| `analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh` | Firehose bimaxwellian moderate, 40 d_i box | cosma7-rp-pauper / 24h |
| `analisis_firehose_bikappa3_bigbox40_pauper.sh` | Firehose bikappa3, 40 d_i box | cosma7-rp-pauper / 24h |

> Each analysis job spreads the 8 independent stages of the Makefile `common`
> target (`brazil fields particles spectral diamagnetic heatflux validate
> physics`) across 8 nodes, one stage per node.

#### How many figures each stage generates

`fields` and `diamagnetic` filter snapshots with `step % SNAPSHOT_EVERY == 0`
(plus always the first and the last). With the PSC output cadence
(`PSC_FIELDS_EVERY_DEFAULT = 500`) and `nmax = 1 200 000`, a run leaves ~2400
field snapshots:

| `SNAPSHOT_EVERY` | PNGs saved per panel |
|---|---|
| 500 (= all output) | ~2400 |
| 10 000 | 121 |
| **100 000** (default) | **13** |
| 200 000 | 7 |

`GIF_EVERY` (default 10 000) is independent: those frames are rendered in memory
and only end up inside the `.gif`, not as separate PNGs. Both can be passed to
the job without editing it:

```bash
sbatch --export=ALL,SNAPSHOT_EVERY=200000,GIF_EVERY=20000 cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
```

> If an old analysis left thousands of PNGs, the COSMA checkout predates the
> commit that introduced this filter (`c18f8dd6e` / `2ec0e39a0`). Run `git pull`
> in `/cosma7/data/dp433/dc-mart18/pcseditado` before resubmitting.

---

## Renaming (old → new)

Scripts that used to sit loose in the repository root were moved here:

| Before (root) | Now |
|---|---|
| `job_MSbM.sh` | `simulacion/sim_mirror_bimaxwellian_strong_MSbM.sh` |
| `job_kappa.sh` | `simulacion/sim_mirror_kappa3.sh` |
| `job_mirror_bikappa3_moderate.sh` | `simulacion/sim_mirror_bikappa3_moderate.sh` |
| `job_analysis_mirror_bimaxwellian_moderate.sh` | `analisis/analisis_mirror_bimaxwellian_moderate_pauper.sh` |
| `job_analysis_mirror_bimaxwellian_moderate_cosma7.sh` | `analisis/analisis_mirror_bimaxwellian_moderate_rp.sh` |

---

## Step by step

Everything is run from the repository root on COSMA:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
```

### A) Analysis of the moderate bi-Kappa case

It already points at `DATA_DIR=.../psc_mirror_bikappa3_moderate_11618877` and
`CASE=mirror_bikappa3_moderate`. Nothing needs to be compiled.

```bash
sbatch cosma_jobs/analisis/analisis_mirror_bikappa3_moderate_pauper.sh
```

Outputs: `/cosma7/data/dp433/dc-mart18/logs/analysis_bikappa3_moderate.<JOBID>.{out,err}`
and one log per stage, `analysis_mirror_bikappa3_moderate_<stage>.<JOBID>.log`.
Results: `CodeforAnalisys/../analysis_results/mirror_bikappa3_moderate/`.

> If the JOBID of the bikappa run changes (the `_11618877` part), edit the
> `DATA_DIR=` line of the script before submitting.

### B) Firehose in a 40 d_i box (bi-Maxwellian and bi-Kappa)

The box size is **compile-time** (`#define PSC_DOMAIN_DI` in
`src/psc_anisotropy_case.hxx`, default 20 d_i) — it is **not** an environment
variable. That is why each 40 d_i case is its own executable and must be
**compiled once** before submitting.

**1) Build the two executables (only once):**

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado && BUILD_DIR="$PWD/build" BUILD_JOBS=4 PSC_TARGETS="psc_firehose_bimaxwellian_moderate_bigbox40 psc_firehose_bikappa3_bigbox40" src/cosma_build_psc_adios2.sh
```

Check that they were produced:

```bash
ls -l build/src/psc_firehose_bimaxwellian_moderate_bigbox40 build/src/psc_firehose_bikappa3_bigbox40
```

**2) Submit the runs:**

```bash
sbatch cosma_jobs/simulacion/sim_firehose_bimaxwellian_moderate_40di.sh
```

```bash
sbatch cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh
```

Each job creates its own folder in
`/cosma7/data/dp433/dc-mart18/anisotropy_adios2/<target>_<JOBID>/`.

> **Resolution note (deliberate):** with 40 d_i and `ngrid=576` the resolution
> drops from ~28.8 to ~14.4 cells/d_i. Chosen this way for cost. The compute cost
> per step is about the same as a 20 d_i run (same 576² cells and 1000 ppc); only
> the physical dx changes. If the same resolution as the other cases is ever
> wanted, use `PSC_NGRID=1152` (≈4× more expensive) — it can be passed through
> the environment:
> `sbatch --export=ALL,PSC_NGRID=1152 cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh`.

### C) Analysis of the 40 d_i firehose runs

Both prerequisites are already met: the profiles
`firehose_bimaxwellian_moderate_bigbox40` and `firehose_bikappa3_bigbox40` exist
in `CodeforAnalisys/psc_units.py` (domain 40 d_i, ngrid 1152), and the analysis
scripts are in `analisis/`.

```bash
sbatch cosma_jobs/analisis/analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh
```

```bash
sbatch cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
```

> **bikappa3 has three run folders** (`_11654252`, `_11657054`, `_11657093`)
> because `sim_firehose_bikappa3_40di.sh` creates a new `RUN_DIR` per
> `SLURM_JOB_ID` on every submission; the snapshots are split across them and no
> single folder holds the complete run. Without `DATA_DIR` the job picks the one
> with the most snapshots and lists all three in the log. Check the `.out`/`.err`
> of each job id before trusting the result, and force the folder if needed:
>
> ```bash
> sbatch --export=ALL,DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_firehose_bikappa3_bigbox40_11657093 cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
> ```

---

## Job monitoring

```bash
squeue -u dc-mart18
```

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

```bash
scancel <JOBID>
```
