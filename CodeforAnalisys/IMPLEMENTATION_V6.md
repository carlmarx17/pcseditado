# Analysis v6 implementation record

Date: 2026-09-28. Historical products in `results/v5` and simulation parameters
were preserved. Pre-existing local modal-growth and comparison edits were
extended, not discarded.

## Delivered

- Shared version/provenance, strict JSON, sampling, geometry and density utilities.
- Source/input fingerprints, reproducibility metadata and explicit scientific
  evidence statuses; HTML/CSV/JSON audit for the three delivered mirror cases.
- Correct velocity-space VDF displays and explicitly labelled u-space fits;
  cylindrical normalization, comparable solver objectives and held-out scores.
- Earlier modal candidate discovery, separate fastest-growth candidate,
  correlated-residual uncertainty and estimator-version comparison guards.
- Common-time energy closure with explicit bounded rounding tolerance and full
  residual coverage; the energy anomaly is not detrended or concealed.
- Mean-centred structures, co-located pressure balance, sensitivity sweeps and
  conservative split/merge tracking.
- Explicit heat-flux averaging, effective counts and a matched asymptotic null
  noise estimate, with its limitations recorded.
- Ordered isolated stage execution, completion records, resumability, exclusive
  output locking and timing logs. See `README.md` for the commands.

## Verification

- Known-answer and invariance tests cover annular normalization, Fourier
  translation/scaling/Parseval/Nyquist, early-mode discovery, weighted and rotated
  heat-flux moments, correlated errors, missing evidence, mixed estimators,
  held-out scores and structure splits.
- The synthetic diagnostic suite exercises the actual scripts as subprocesses.
- An additional runner smoke test completed manifest, energy, residuals,
  structures and energy exchange. A separate manifest resume check skipped the
  completed stage successfully using matching source/input/output fingerprints.
- The synthetic structure map was visually inspected. `git diff --check` passed.
- The knowledge graph was rebuilt and enriched; graphify reported existing
  extraction warnings and a report-refresh KeyError. The enriched graph JSON was
  saved, but that report refresh was not successful.

## Limits and outstanding work

This is a substantial processing revision, not completion of every research item
in the plan. Original COSMA data/logs are required to regenerate production
figures and resolve the energy anomaly. New convergence runs, oblique kinetic
solver integration, rigorous mixture/model-selection validation, full nonlinear
uncertainty calibration, production I/O/cache benchmarking and final publication
QA remain outstanding. The detailed item-by-item status is in section 9 of
`RESULTS_V5_IMPROVEMENT_PLAN.md`.

The local v5 evidence report deliberately leaves all three runs UNVERIFIED:
passing synthetic tests does not establish physical validity of those runs.

Final full-suite result on the completed code: **153 tests passed, 14 subtests
passed in 60.83 seconds**, using `.venv/bin/python -m pytest -q CodeforAnalisys`.

## Follow-up (2026-09-29): production path, energy audit and QA

### Defects found and corrected

- **Production regeneration was blocked.** `reanalysis_v5_all.sh` refuses any
  checkout whose conventions version is not 5, and there was no v6 job. Added
  `cosma_jobs/analisis/reanalysis_v6_all.sh` (see `cosma_jobs/README.md`, E).
  It was run end to end locally on synthetic runs with the production folder
  names through a stand-in `srun`, under bash 3.2; that run caught an
  empty-array expansion that aborts `set -u` on bash < 4.4.
- **Runner.** Only 5 of 15 stages had been exercised. The full run showed
  `energy-if-present` and `physics` writing the same four files, so every
  `--resume` re-ran both (including the most expensive stage). A stage can now
  never overwrite another stage's products; the pair is rejected up front.
  Added `--jobs`, `--launcher` (srun per stage), `--keep-going`, per-process
  peak memory (was cumulative over children), early case validation, `theory`
  refused for mirror cases, a lock warning on filesystems without `flock`, and
  re-running of stages ordered after a re-run stage.
- **gamma(k) map hid the growing mode.** The default 0.9 d_i^-1 k_perp display
  zoom cropped the only accepted cell of the synthetic run (k_perp d_i = 0.94),
  leaving an empty map although the CSV held gamma = 0.236 (input 0.25). The
  zoom now widens to every accepted cell and says so; axes follow the bins.
- **Evidence report.** An accepted fit of the legacy `total` series (domain rms,
  biased low) counted as PASS; the energy check ignored the prt-window moments
  when `diag.asc` was missing; the PASS reason of the initial state read
  "validation required". All corrected.
- **Initial validation** used fixed 2 % tolerances whatever the sample size, so
  the known-answer synthetic run FAILED; tolerances are now max(fixed, 4
  sampling standard errors from the measured fourth moment). **Log coverage**
  required the last check at exactly nmax instead of within one check cadence.
- Figure-level: an empty `vdf_spatial` macro-cell panel now says why; a legend
  proxy in `growth_rate_vs_k` used a marker the data do not use; Spanish verdict
  strings in `electron_energy_trend.csv` are now English.
- The 16 stale grid/ppc header comments of the case files now state the values
  actually run (576², 1000 ppc; job scripts confirm, manifests detect 576²). No
  define changed. The parity checker is down from 18 to 2 findings.

### Added

- `energy_audit.py`, used by the report: diagnostic mapping, closure against
  the driver's release, window proxy, dx/lambda_De(t), beta_e(t), early-time
  T_e error, cross-run common-mode test.
- Figure content records at every `plot_style.save` and a `figures` check.
- Spatial-mixture control of the kappa tail in `vdf_spatial.py`
  (`vdf_validation.mixture_tail_budget`), validated on an exact Gamma mixture of
  Maxwellians (kappa_mix 5.12 vs kappa_eff 5.09), a true kappa and a Maxwellian.
- `test_evidence_tooling.py`: 17 known-answer tests (audit, figure check,
  runner ordering/collision/resume/keep-going through a stand-in make, shot-noise
  tolerance, log cadence, map zoom, mixture control). `pytest` joined
  `requirements.txt`, since the COSMA preflight runs the suite.

### Energy audit of the v5 deliveries (most important result)

`analysis_results/v6_audit/energy_audit/` (regenerable with the command in the
README). All three mirror-moderate runs **FAIL** the energy budget:

| Run | T_e end / T_e0 | e gain / (ion + field release), window | dx/lambda_De start → end | A_e end | Global DiagEnergies |
|---|---|---|---|---|---|
| bi-Maxwellian | 8.32 | 6.4 | 8.71 → 3.02 | 0.98 | not delivered |
| bi-kappa 5 | 8.43 | 6.5 | 8.72 → 3.00 | 0.98 | +64.6 % total, 5.5x the ion release; mapping PASS |
| bi-kappa 3 | 8.55 | 6.2 | 8.75 → 2.99 | 0.98 | not delivered |

- The DiagEnergies columns reproduce the initial partition (E_i/E_B = 12.498,
  E_e/E_B = 1.492, expected 12.5 / 1.5 with a 0.4 % relativistic correction):
  the anomaly is **not** a column, factor or volume error.
- The window and global electron heating agree within 2.3 %: it is domain-wide.
- The heating differs by only 3.1 % between the three distributions: it belongs
  to the shared numerical setup, not to the distribution being compared.
- It is isotropic, starts at t = 0 and accelerates (second half of the run heats
  2.2x more than the first). A classical finite-grid instability slows as
  lambda_De approaches dx; this one does not, so the mechanism is **not**
  assigned. Candidates remain dx/lambda_De = 8.7 with first-order shapes and
  particle noise.
- Early-time error: T_e/T_e0 = 1.23 at t Omega_ci = 10, 1.45–1.48 at 20,
  ~2.0 at 40, and 2.7–2.9 at the end of the legacy fitted linear phase
  (t Omega_ci ≈ 63–68). beta_e ends near 8.4 instead of 1. Late-time electron
  pressure, pressure balance and saturation are affected most; a linear growth
  rate is only defensible with this T_e drift stated alongside it.

Discriminating reruns (a proposal; not submitted, parameters unchanged): short
runs of `mirror_bimaxwellian_moderate` to t Omega_ci ≈ 40 (`PSC_NMAX=300000`)
with (a) the baseline, (b) `PSC_NICELL=4000`, (c) `PSC_NGRID=1152` (dx/lambda_De
4.3, dt halves by CFL). Noise heating scales with 1/ppc, grid heating with dx;
both are environment overrides of the existing executable. A second-order shape
(`PscConfig2ndSingle`) would need a new executable and a case-integrity review.

### Still outstanding

- Production regeneration: submit `reanalysis_v6_all.sh` on COSMA.
- The energy anomaly needs the reruns above; post-processing cannot repair it.
- Oblique kinetic theory for the mirror (solver choice and validation), full
  nonlinear confidence intervals, heat-flux floor calibration on production
  tails, advection-aware tracking, I/O benchmarking and the final thesis figure
  selection remain research or production work.
- Migration of the two monolithic reconnection cases (structural; needs a
  decision per the case-integrity rules).

## Follow-up 2 (2026-09-29): isotropic controls and publication review

### Isotropic controls of the numerical heating

- Cases `psc_mirror_{bimaxwellian,bikappa5,bikappa3}_isotropic`: the moderate
  twins with A_i = 1 and the same ion thermal energy (beta_i,par = 25/3), same
  numerics and electrons; CMake targets, case table
  (`src/SIMULACIONES_ANISOTROPIA.md`) and job scripts
  (`cosma_jobs/simulacion/sim_mirror_*_isotropic.sh`, `PSC_ENERGIES_EVERY=500`).
  The parity checker accepts them as distribution twins (still only the two
  reconnection findings).
- `energy_audit.py` pairs each run with its control and subtracts the
  control's energy changes: baseline-corrected electron heating and closure
  (PASS <= 0.1, FAIL >= 0.5 of the ion release; the raw status is kept). The
  evidence report adds an `energy_baseline` check and figure; the v6 job takes
  the controls through `EXTRA_RUNS` and reports over the whole v6 tree.
- Known-answer tests: exact closure recovered, wrong control rejected,
  same-distribution preference, explicit pairing.

- Analysis profiles `mirror_*_isotropic` added to `psc_units.py`. All 28
  profiles of maintained cases were checked against their case files and job
  scripts (beta, A, kappa, box, mi/me, grid, ppc, nmax, cadences) and match;
  `test_profiles.py` now enforces it. A synthetic control run goes through the
  pipeline and is paired automatically; a control that loses as much ion energy
  as its run leaves nothing to close and is reported UNVERIFIED, not FAIL.

### Publication review of the figures

- Automatic layout checks at every save: overlapping text of any kind, crowded
  ticks, legends or annotations covering data. On the synthetic run the
  pipeline went from 111 findings in 129 figures to 0 in 147 (all figures now
  pass through the check; three scripts used to bypass the shared style).
- `fluctuationofmagneticfiel.py`, `plot_prt.py` and `validate_moments.py` moved
  to `plot_style` (they had a dark theme, 150–220 dpi, no PDF, no check); the
  Okabe-Ito palette replaced a red/green pair that colour-blind readers cannot
  separate. ~300 lines of unreachable or never-called code removed.
- One convention for every map (z along B0 horizontal, equal aspect, integer
  d_i ticks); `structures_analysis` and `heat_flux_analysis` drew the
  transpose. k_par horizontal in all k-space maps (the mirror dB_z spectrum
  was transposed). Time as t Omega_ci everywhere, velocities as v/v_A.
- Axis offsets folded into labels, log axes with 1-2-5 labels, autocrop
  clamped to the computed grid, headroom above extreme points, legends moved
  off data; every one of the 88 figure types was inspected visually.

### Physics corrections found in the review

- `plot_prt.py` and `vdf_spatial.py` binned u = gamma v while labelling v; both
  now convert, weight and normalise by the whole population.
- The |v_perp| reference was a 1-D Gaussian; it is the Rayleigh distribution of
  a 2-D Maxwellian (verified on the synthetic bi-Maxwellian).
- `mirror_area_fraction` (cells below B0 - std|B|) is ~16–25 % for any amplitude:
  renamed and no longer shown as a hole area.
- The spectral index was fitted over the middle half of k regardless of the
  physics and reported a noise slope k^+2.8; it is now fitted only over a
  decaying range above the noise floor (known-answer test added).
- kappa fits at their bound (80) are shown as Maxwellian-consistent in 1/kappa
  with conditional errors; the A-vs-field scatter files were overwritten at
  every step (only the last survived) and plotted raw |B| in code units.
- A factor-2 labelling error in the magnetic-energy panel of `plot_prt.py`.

`METHODOLOGY_REVIEW.md` evaluates every method physically. Its main findings:
the moderate "mirror" runs are equally unstable to the ion-cyclotron branch,
and the strongest v5 mode is parallel and transverse (IC-like), so mirror
growth must be measured on the oblique compressive component; electrons carry
up to half of the perpendicular pressure by the end because of the numerical
heating; the 20 d_i box samples only k rho_i = 0.70, 1.40, 2.11; a single
realisation per case cannot yet establish a distribution effect.

Verification: 205 tests and 14 subtests pass without warnings; the full
pipeline on the synthetic run writes 147 figures with 0 content or layout
findings; no unused imports remain; the COSMA job was rerun locally end to end.

## Follow-up 3 (2026-09-29): smaller control box and resolution test

- The isotropic controls now run in a 10 d_i box at 288² (`PSC_DOMAIN_DI 10.0`,
  `PSC_NGRID=288`, 256 ranks on 10 nodes): the twins' dx, dt, ppc and cadence
  with a quarter of the cells. The energy audit pairs controls by dx, ppc and
  electrons rather than by box, and subtracts their energy changes per unit
  volume (ratio of E_B(0)); known-answer test with a quarter-volume control.
  The case-integrity rule for `PSC_DOMAIN_DI` records the exception.
- Resolution test of `psc_mirror_bimaxwellian_moderate` to t Omega_ci ~ 41:
  `PSC_NICELL=4000` and `PSC_NGRID=1152` (dx/lambda_De 4.3, dt/2), same
  executable, production cadence in physical time. Each job writes
  `analysis_config.json`, which `run_pipeline.py` now uses automatically, so
  the manifests carry the real ppc and dt (`runtime_verified`).
  `analisis_resolution_mirror.sh` runs both variants, the energy audit with
  labelled runs and a comparison at the last common time, and the convergence
  table; the whole chain was run locally on synthetic variants.
