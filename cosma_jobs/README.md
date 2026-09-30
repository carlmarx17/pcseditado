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
| `sim_firehose_bimaxwellian_strong_40di.sh` | Strong bi-Maxwellian firehose, big box — the **controlled twin** of the bi-Kappa-3 run | **40 d_i** | ngrid 1152 |
| `sim_firehose_bikappa5_40di.sh` | Bi-Kappa-5 firehose, big box — third member of the strong firehose series | **40 d_i** | ngrid 1152 |
| `sim_mirror_bikappa5_moderate.sh` | Moderate bi-Kappa-5 mirror — third member of the moderate mirror series | 20 d_i | ngrid 576 |
| `sim_mirror_{bimaxwellian,bikappa5,bikappa3}_isotropic.sh` | **Isotropic controls** of the moderate mirror series (A_i = 1, same ion thermal energy, beta_i_par = 25/3, same dx/dt/ppc): their electron heating is the numerical baseline subtracted per unit volume by `energy_audit.py`; 10 nodes | **10 d_i** | ngrid 288 |
| `sim_mirror_bimaxwellian_moderate_res_{ppc4000,ngrid1152}.sh` | **Resolution test** of the moderate bi-Maxwellian mirror to t Omega_ci ~ 41: 4x ppc, or dx/2 (dt/2); 83 nodes, ~22 h and ~43 h (estimated from the 1152² whistler jobs) | 20 d_i | ngrid 576 / 1152 |
| `sim_whistler_bimaxwellian_{strong,moderate,weak}.sh` | Bi-Maxwellian whistler (Ae = 3.0 / 2.0 / 1.5); electron-scale cadence, each only until relaxation (~6 / ~20 / ~43 h on 83 nodes) | 20 d_i | **ngrid 1152**, 2000 ppc |
| `sim_whistler_bikappa{3,5}_strong.sh` | Bi-Kappa whistler strong (κ = 3 / 5) — distribution twins of the bi-Maxwellian strong | 20 d_i | **ngrid 1152**, 2000 ppc |
| `sim_whistler_{bimaxwellian,bikappa3,bikappa5}_strong_mr800.sh` | Whistler strong series with **mi/me = 800** (the batch to run); ~23 h on 83 nodes each | 20 d_i = 566 d_e | **ngrid 2304**, 2000 ppc |
| `sim_whistler_bikappa3_moderate.sh` | Moderate bi-Kappa-3 whistler — **still at the old 576², 1000 ppc, 80 000-step settings**: it is not the distribution twin of the production moderate run (1152², 2000 ppc, 140 000 steps) until its settings are aligned | 20 d_i | ngrid 576 |

The ion-scale simulation scripts set `PSC_CHECKPOINT_EVERY=150000` (whistler scripts use a per-regime value, see `src/WHISTLER_PARAMETROS.md`). Checkpoints are only used
to restart a run; running a case without one of these scripts falls back to the
header default (`PSC_CHECKPOINT_EVERY_DEFAULT = 5000`), which writes 240 checkpoints
per run.

### Batch that closes the kappa comparison

With these three runs each series has the full bi-Maxwellian / κ=5 / κ=3
set, and the members differ **only** in the distribution:

| Series | bi-Maxwellian | κ=5 | κ=3 |
|---|---|---|---|
| Mirror moderate, 20 d_i (β_i∥=5, A_i=2) | done | **`sim_mirror_bikappa5_moderate.sh`** | done |
| Firehose strong, 40 d_i (β_i∥=10, A_i=0.1) | **`sim_firehose_bimaxwellian_strong_40di.sh`** | **`sim_firehose_bikappa5_40di.sh`** | done |

Build the three executables once, then submit (from the repository root on COSMA):

```bash
BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
  PSC_TARGETS="psc_mirror_bikappa5_moderate psc_firehose_bikappa5_bigbox40 psc_firehose_bimaxwellian_strong_bigbox40" \
  src/cosma_build_psc_adios2.sh

sbatch cosma_jobs/simulacion/sim_mirror_bikappa5_moderate.sh
sbatch cosma_jobs/simulacion/sim_firehose_bimaxwellian_strong_40di.sh
sbatch cosma_jobs/simulacion/sim_firehose_bikappa5_40di.sh
```

Note the job id each submission prints: it becomes the `RUN_TAG` needed to
resume from a checkpoint if 48 h are not enough.

### Isotropic controls of the moderate mirror series (numerical heating)

The three moderate mirror runs heat their electrons x8.3–8.6, isotropically
and to within 3 % of each other, while their total energy is not conserved
(`CodeforAnalisys/IMPLEMENTATION_V6.md`). Each control repeats its twin with
`A_i = 1` and the same ion thermal energy (`beta_i_par = 25/3`): same dx, dt,
ppc, cadence and electrons in a 10 d_i box (288², a quarter of the cells, 10
nodes), but mirror stable. What its
electrons gain is the numerical baseline that `energy_audit.py` (and the v6
evidence report) subtract from the anisotropic run. `PSC_ENERGIES_EVERY=500`
is mandatory: the corrected closure uses `diag.asc`.

```bash
BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
  PSC_TARGETS="psc_mirror_bimaxwellian_isotropic psc_mirror_bikappa5_isotropic psc_mirror_bikappa3_isotropic" \
  src/cosma_build_psc_adios2.sh
```

```bash
sbatch cosma_jobs/simulacion/sim_mirror_bimaxwellian_isotropic.sh
```

```bash
sbatch cosma_jobs/simulacion/sim_mirror_bikappa5_isotropic.sh
```

```bash
sbatch cosma_jobs/simulacion/sim_mirror_bikappa3_isotropic.sh
```

When they finish, analyse them together with their twins (they are paired
automatically by numerics, electrons and distribution):

```bash
sbatch --export=ALL,EXTRA_RUNS="mirror_bimaxwellian_isotropic:psc_mirror_bimaxwellian_isotropic_<jobid> mirror_bikappa5_isotropic:psc_mirror_bikappa5_isotropic_<jobid> mirror_bikappa3_isotropic:psc_mirror_bikappa3_isotropic_<jobid>" cosma_jobs/analisis/reanalysis_v6_all.sh
```

### Resolution test of the moderate mirror (what drives the heating)

```bash
sbatch cosma_jobs/simulacion/sim_mirror_bimaxwellian_moderate_res_ppc4000.sh
```

```bash
sbatch cosma_jobs/simulacion/sim_mirror_bimaxwellian_moderate_res_ngrid1152.sh
```

Both use the production executable (no new build) and stop at t Omega_ci ~ 41.
After `reanalysis_v6_all.sh` has analysed the production run:

```bash
sbatch --export=ALL,PPC_RUN=psc_mirror_bimaxwellian_moderate_res_ppc4000_<jobid>,GRID_RUN=psc_mirror_bimaxwellian_moderate_res_ngrid1152_<jobid> cosma_jobs/analisis/analisis_resolution_mirror.sh
```

`analysis_results/v6c_resolution/energy_audit/energy_audit_common_time.csv`:
heating ~4x lower with 4x ppc -> particle noise; much lower with dx/2 ->
finite-grid heating.

### Whistler strong series with mi/me = 800 (batch to run)
| `sim_whistler_bikappa3_moderate.sh` | Moderate bi-Kappa-3 whistler — **still at the old 576², 1000 ppc, 80 000-step settings**: it is not the distribution twin of the production moderate run (1152², 2000 ppc, 140 000 steps) until its settings are aligned | 20 d_i | ngrid 576 |

The whistler strong series runs with **mi/me = 800** (decision and full
numbers: `src/WHISTLER_PARAMETROS.md` §6). The three runs are
distribution twins (β_e∥ = 0.5, A_e = 3.0; bi-Maxwellian, κ=3, κ=5),
each identical to its mi/me = 200 case except `PSC_MASS_RATIO`.

Per run: 83 nodes × 28 (2 304 ranks), ~23 h expected inside a 48 h limit,
~8–16 GB RAM per node, ~0.58 TB durable output plus a ~0.64 TB final
checkpoint. The group quota does not fit all three at once, so they go
**two at a time**.

1. Free space first: delete the intermediate checkpoints of finished
   runs (keep the last one of each), then check
   `lfs quota -hg dp433 /cosma7`. It needs ≥ ~2.5 TB free.

2. Update and build (from the repository root on COSMA):

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
git pull origin main
BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
  PSC_TARGETS="psc_whistler_bimaxwellian_strong_mr800 psc_whistler_bikappa3_strong_mr800 psc_whistler_bikappa5_strong_mr800" \
  src/cosma_build_psc_adios2.sh
ls -l build/src/psc_whistler_*_strong_mr800
mkdir -p /cosma7/data/dp433/dc-mart18/anisotropy_adios2
```

3. Submit the first two, one per partition:

```bash
sbatch cosma_jobs/simulacion/sim_whistler_bimaxwellian_strong_mr800.sh
sbatch --partition=cosma7 cosma_jobs/simulacion/sim_whistler_bikappa3_strong_mr800.sh
```

   In each `.out` the header must print
   `run = nmax 40000, ngrid 2304, nicell 2000, np 1x48x48`; if it shows
   576/1000/1200000, `scancel` it: the overrides were not applied.

4. When both have finished: if the domain-averaged A_e(t) changed by
   less than ~2% over the last 100 Ω_ce⁻¹, the relaxation is complete —
   delete their `checkpoint_40000.bp` (~0.64 TB each) and submit κ=5:

```bash
sbatch cosma_jobs/simulacion/sim_whistler_bikappa5_strong_mr800.sh
```

   If A_e(t) was still falling, extend instead (`PSC_NMAX` is the
   absolute final step):

```bash
sbatch --export=ALL,RUN_TAG=<tag>,PSC_NMAX=60000,PSC_RESTART=<run_dir>/checkpoint_40000.bp \
  cosma_jobs/simulacion/sim_whistler_<dist>_strong_mr800.sh
```

### Whistler strong series with mi/me = 200 (kept as reference)

Superseded as the first batch by the mi/me = 800 series above; kept
ready in case a mass-ratio comparison is wanted. The three **strong**
whistler runs — bi-Maxwellian, κ=5 and κ=3, differing **only** in the
distribution (β_e∥ = 0.5, A_e = 3.0 in all three). Numerics and budget:
`src/WHISTLER_PARAMETROS.md`. Each strong run goes **only until the
relaxation**: 40 000 steps = 528 Ω_ce⁻¹, 83 nodes × 28, ~6 h expected
inside a 12 h limit, ~4 GB RAM/node, ~0.2 TB durable output (~47 000
core-h for the three). The strong runs write **only the final**
checkpoint (`checkpoint_40000.bp`, ~160 GB) to spare the nearly full
dp433 group quota; if a job dies mid-run it is simply resubmitted from
t=0 (~6 h). If A_e(t) is still falling at the end, extend from the final
checkpoint (below) with a larger `PSC_NMAX`.

The three strong runs fit side by side across the two identical COSMA7
partitions (`cosma7-rp`, `cosma7`; 83 nodes each). `--partition` on the
command line overrides the script:

```bash
sbatch --partition=cosma7 cosma_jobs/simulacion/sim_whistler_bikappa5_strong.sh
```

Build the three executables once, then submit (from the repository root on
COSMA):

```bash
BUILD_DIR="$PWD/build" BUILD_JOBS=4 \
  PSC_TARGETS="psc_whistler_bimaxwellian_strong psc_whistler_bikappa3_strong psc_whistler_bikappa5_strong" \
  src/cosma_build_psc_adios2.sh

sbatch cosma_jobs/simulacion/sim_whistler_bimaxwellian_strong.sh
sbatch cosma_jobs/simulacion/sim_whistler_bikappa3_strong.sh
sbatch cosma_jobs/simulacion/sim_whistler_bikappa5_strong.sh
```

To extend a finished strong run whose A_e(t) had not yet relaxed (e.g.
another 20 000 steps), restart from its final checkpoint with a larger
`PSC_NMAX` — `nmax` counts absolute steps, not additional ones:

```bash
sbatch --export=ALL,RUN_TAG=<tag>,PSC_NMAX=60000,PSC_RESTART=<run_dir>/checkpoint_40000.bp \
  cosma_jobs/simulacion/sim_whistler_bimaxwellian_strong.sh
```

After each run finishes, delete its `checkpoint_*.bp` (~160 GB each) once
the analysis manifests are written.

### `analisis/` — Python pipeline over finished runs

| Script | Analyses | Partition / limit |
|---|---|---|
| `analisis_mirror_bimaxwellian_moderate_pauper.sh` | Mirror bimaxwellian moderate | cosma7-rp-pauper / 24h |
| `analisis_mirror_bimaxwellian_moderate_rp.sh` | Mirror bimaxwellian moderate (higher priority) | cosma7-rp / 72h |
| `analisis_mirror_bikappa3_moderate_pauper.sh` | Mirror **bikappa3** moderate | cosma7-rp-pauper / 24h |
| `analisis_firehose_bimaxwellian_moderate_bigbox40_pauper.sh` | Firehose bimaxwellian moderate, 40 d_i box | cosma7-rp-pauper / 24h |
| `analisis_firehose_bikappa3_bigbox40_pauper.sh` | Firehose bikappa3, 40 d_i box | cosma7-rp-pauper / 24h |
| `reanalysis_v5_all.sh` | Historical: the v5 pipeline (section D). Refuses a v6 checkout on purpose | cosma7-rp / 48h, 20 nodes |
| **`reanalysis_v6_all.sh`** | **Runs the v6 pipeline (`run_pipeline.py`) on every finished run into `analysis_results/v6c`; deletes nothing** (see section E) | cosma7-rp / 48h, 20 nodes |

> The per-case scripts above predate the v5 analysis revision (2026-09-28)
> and write to `run_aware_v4`; use `reanalysis_v6_all.sh` for new products.

### `utils/` — helpers that do not submit anything

| Script | What it does |
|---|---|
| `merge_run_folders.sh` | Reports the step range of every run folder of a case and, **only if the ranges are disjoint**, hardlinks them into a single `<target>_merged` folder. Refuses to merge overlapping folders, because those are independent runs from t = 0 rather than segments of one run. Sources are never modified. |

```bash
cosma_jobs/utils/merge_run_folders.sh psc_firehose_bikappa3_bigbox40
```

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

> **bikappa3 is split across several run folders** (`_11657054`, `_11657093`; an
> earlier `_11654252` is no longer on disk). The cause: `RUN_DIR` used to be keyed
> on `SLURM_JOB_ID`, and the job never exported `PSC_RESTART` — so every
> resubmission created a new folder **and restarted from t = 0**. The folders are
> therefore independent partial runs, not consecutive segments of one run, and
> concatenating them would splice two different trajectories into one time series.
>
> Both scripts now key `RUN_DIR` on `RUN_TAG` (defaults to the first job id, which
> is printed in the log) and refuse to start on top of an existing run without
> `PSC_RESTART`. To continue a run instead of starting a new one:
>
> ```bash
> sbatch --export=ALL,RUN_TAG=11657093,PSC_RESTART=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_firehose_bikappa3_bigbox40_11657093/checkpoint_<step>.bp cosma_jobs/simulacion/sim_firehose_bikappa3_40di.sh
> ```
>
> To find out what the existing folders actually contain, run
> `cosma_jobs/utils/merge_run_folders.sh` (see below). Without `DATA_DIR` the
> analysis job picks the folder with the most snapshots and lists them all in the
> log. Check the `.out`/`.err` of each job id before trusting the result, and
> force the folder if needed:
>
> ```bash
> sbatch --export=ALL,DATA_DIR=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_firehose_bikappa3_bigbox40_11657093 cosma_jobs/analisis/analisis_firehose_bikappa3_bigbox40_pauper.sh
> ```

### D) Full re-analysis with the v5 pipeline (clean + all stages)

The v5 revision of `CodeforAnalisys` (2026-09-28) changes numbers that go
into the thesis (gamma of transverse modes was 2x, gamma(k) biased low,
J_dia sign, the heat flux was a proxy, whistler k range). Every old product
must be replaced. One job does it for every finished run:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado && git pull
```

```bash
sbatch cosma_jobs/analisis/reanalysis_v5_all.sh
```

What it does, in this order (nothing is deleted until the new code has
proven itself on the node):

1. refuses to start if the checkout is older than v5;
2. preflight on a compute node: unit tests and the synthetic end-to-end run;
3. skips any run whose last field snapshot is short of `nmax` (unfinished);
4. per run, new manifest + initial-condition check; only if it passes are
   the run's old results deleted (`analysis_results/run_aware_v4/<case>`,
   `analysis_results/<case>`, old comparisons). Nothing else is touched;
5. every analysis stage of every run as its own job step (83 steps for the
   five runs), longest first, one step per node, on 20 nodes;
6. comparisons of the controlled series (only the distribution changes;
   `compare-physics`, `kappa_evolution`): mirror moderate 20 d_i
   bi-Maxwellian / kappa 5 / kappa 3, and firehose strong 40 d_i
   bi-Maxwellian / kappa 3 (both job scripts use identical settings);
7. `analysis_results/v5/REANALYSIS_SUMMARY_<jobid>.txt` with the status of
   every step; per-step logs in `logs/reanalysis_v5_<jobid>/`.

Runs included: `firehose_bimaxwellian_strong_bigbox40` (_12062436; the
other folder, _12031167, is not used), `firehose_bikappa3_bigbox40`
(_11657093), `mirror_bimaxwellian_moderate` (_11596993),
`mirror_bikappa3_moderate` (_11618877), `mirror_bikappa5_moderate` (_12063822).
Left out until they finish, with their old results kept:
`firehose_bimaxwellian_moderate_bigbox40` (_11643619) and
`whistler_bimaxwellian_strong_mr800` (_12068623; it also needs a mi/me = 800
profile in `psc_units.py`). To add a run, append `CASE:folder` to `RUNS=(...)`.

Overrides without editing:

```bash
sbatch --export=ALL,ONLY="mirror_bikappa5_moderate" cosma_jobs/analisis/reanalysis_v5_all.sh
```

```bash
sbatch --export=ALL,DRY_RUN=1 cosma_jobs/analisis/reanalysis_v5_all.sh
```

`CLEAN=0` keeps the old results; `GROWTH_T_START=.. GROWTH_T_END=..` fixes the
linear-phase window of every gamma fit; `--nodes=N --ntasks=N` changes the
parallelism without changing the results. The job was tested end to end on
synthetic runs with the same folder names (87/87 steps OK, cleanup limited to
the analysed cases); the ADIOS2 (`.bp`) reading path is the one already used
by the per-case jobs.

### E) Re-analysis with the v6 pipeline (nothing deleted)

Revision 6 (analysis conventions version 6) replaces the growth reference by
the modal fit, fixes VDF coordinates and structure definitions, and adds the
evidence report and the energy audit. The v5 products stay as the historical
baseline; this job writes a new tree, `analysis_results/v6c` by default (the
final one: the earlier `v6` and `v6b` trees are left as they are):

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado && git pull
```

```bash
.venv/bin/python -m pip install -r CodeforAnalisys/requirements.txt
```

```bash
sbatch cosma_jobs/analisis/reanalysis_v6_all.sh
```

What it does:

1. refuses a checkout older than v6 (and `reanalysis_v5_all.sh` refuses a v6
   one, so the two trees never mix);
2. preflight on a compute node: the whole test suite (`pytest`, which includes
   the synthetic end-to-end run of every script);
3. skips unfinished runs, and runs that already have a v6 directory unless
   `RESUME=1`;
4. one `run_pipeline.py` per run, all at once; each launches its stages as
   `srun` steps (one node each, up to nodes/runs per run, longest first):
   manifest preflight, then `physics` -> `spectral` with the accepted linear
   phase, every other stage alongside; mirror runs add `theory-liouville` and
   `theory` (the parallel ion-cyclotron branch that competes with the mirror
   and grows in these runs, never a mirror prediction), firehose runs
   `theory`; isotropic controls (`*_isotropic`) run neither;
5. comparisons of the controlled series, only between runs whose `physics`
   stage passed (`compare-physics`, `kappa_evolution`), and for the mirror
   series the publication figures (`paper_figures.py`) in
   `analysis_results/v6c/paper_figures/`;
6. `analysis_results/v6c/quality_report/index.html`: evidence matrix of every
   run and the energy audit across runs (an isotropic control present in the
   tree is paired with its twin and its heating subtracted);
7. `analysis_results/v6c/REANALYSIS_SUMMARY_<jobid>.txt`: execution status and
   time of every stage, and the scientific status of every run.

Isotropic controls are added with `EXTRA_RUNS` (in the same job or later with
`RESUME=1` and the same `NEW_ROOT`):

```bash
sbatch --export=ALL,EXTRA_RUNS="mirror_bimaxwellian_isotropic:<folder> mirror_bikappa5_isotropic:<folder> mirror_bikappa3_isotropic:<folder>" cosma_jobs/analisis/reanalysis_v6_all.sh
```

Per-stage logs are in `analysis_results/v6c/<case>/logs/`, the runner and
comparison logs in `logs/reanalysis_v6_<jobid>/`. An interrupted job continues
with `sbatch --export=ALL,RESUME=1 ...`: completed stages whose products are
unchanged are skipped, provided code, inputs and options are identical. The
other overrides of section D (`ONLY`, `DRY_RUN`, `GROWTH_T_START/END`,
`--nodes`) work the same way. The script was run end to end on a laptop
against synthetic runs with the production folder names, through a stand-in
`srun`.

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
