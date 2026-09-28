# Whistler runs: parameter study and run configuration

Decision record for the whistler anisotropy-instability campaign
(`psc_whistler_bimaxwellian_{strong,moderate,weak}`). The whistler is the
only **electron-scale** instability in the case matrix: it grows on `d_e`
and `Omega_ce^-1` while the shared setup (`psc_anisotropy_case.hxx`) was
sized for ion-scale mirror/firehose runs. This document checks every
numerical parameter against (a) the kinetic linear theory of our exact
cases, computed with `CodeforAnalisys/linear_theory.py`, and (b) published
PIC studies of the same instability, and records what is changed for the
whistler jobs and what is deliberately kept identical to the rest of the
matrix.

Companion reading: `ESCALADO_INESTABILIDADES.md` (general scaling of the
matrix, ion-scale reasoning, numerical-heating mitigations).

## 1. Linear theory of our three regimes

`linear_theory.py` (parallel propagation, both species kinetic, validated
by `make theory-self-test`) with the exact case parameters
`beta_i_par = 1.0, A_i = 1.0, beta_e_par = 0.5, mi/me = 200, B0 = 0.08`:

| Case | A_e | gamma_max/Ω_ce | k_peak d_e | omega_r/Ω_ce at peak | unstable band k d_e | 10 e-folds [Ω_ce⁻¹] |
|---|---:|---:|---:|---:|---|---:|
| strong (bi-Max) | 3.0 | 0.175 | 0.82 | 0.50 | 0.38 – 1.41 | 57 |
| moderate (bi-Max) | 2.0 | 0.054 | 0.71 | 0.39 | 0.40 – ~1.0 | 186 |
| weak (bi-Max) | 1.5 | 0.0075 | 0.59 | 0.28 | 0.39 – 0.70 | 1 330 |
| strong (bi-kappa κ=3) | 3.0 | 0.144 | 0.83 | 0.49 | — | 69 |
| strong (bi-kappa κ=5) | 3.0 | 0.160 | 0.83 | 0.49 | — | 62 |
| moderate (bi-kappa κ=3) | 2.0 | 0.044 | 0.70 | 0.37 | — | 228 |
| weak (bi-kappa κ=3) | 1.5 | 0.0084 | 0.55 | 0.25 | — | 1 190 |

Two consequences drive everything below:

1. All the physics lives at `k d_e ≈ 0.4–1.4` and `omega_r ≈ 0.25–0.5 Ω_ce`
   (plus harmonic content above during saturation).
2. Growth is fast: even the weak case is done growing within
   ~1 300 Ω_ce⁻¹ ≈ 6.7 Ω_ci⁻¹. The bi-kappa κ=3 twins have rates within
   ~20% of the bi-Maxwellian ones (slightly lower for strong/moderate,
   slightly *higher* for weak — the marginal-condition stimulation of
   Lazar et al. 2019), so one run configuration serves both distributions.

## 2. What comparable published PIC studies use

| Study | Dim | Ions | ω_pe/Ω_ce | Box [d_e] | Δx [d_e] | ppc | Duration |
|---|---|---|---:|---:|---:|---:|---|
| An et al. 2017, JGR 122 (whistler parameter scan) | 2D | immobile | — | 10.24 (512²) | 0.02 | 81 | per-case |
| Lazar et al. 2019, Ap&SS 364:171 (bi-kappa whistler, PIC+QL) | 1D | mp/me=1836 | 20 | 512 (2048 cells) | 0.25 | 10 000 | 150 Ω_ce⁻¹ (Δt=0.01 ω_pe⁻¹) |
| Abdul, Matthews & Mace 2021, PoP 28 062104 (2D bi-kappa whistler, κ=2,3,∞) | 2D | kinetic | — | (paywalled; note: they pick a *large* anisotropy "for reasonable run times") | | | |
| Cui et al. 2022, Front. ASS 9:941241 (whistler anisotropy vs turbulence) | 3D | mi/me=400 | 2.24 | 51.2 (512³) | 0.10 | 48/species | >447 Ω_ce⁻¹ (Δt=0.05 ω_pe⁻¹) |
| Gary, Liu & Winske 2011, PoP 18 082902 (low-β_e whistler) | 2D | kinetic | >1 | (abstract only) | | | β_e∥ = 0.01–0.1 |

Reading of the table: boxes span 10–512 d_e (what matters is having many
discrete modes inside the unstable band, not absolute size), grids resolve
Δx = 0.02–0.25 d_e, durations are a few hundred Ω_ce⁻¹, ppc anywhere from
48 to 10⁴, and ω_pe/Ω_ce = 2–20 brackets our 12.5.

## 3. Our setup measured against that

Fixed by the shared header (`mi/me = 200`, `B0 = 0.08` ⇒ `Ω_ce = 0.08 ω_pe`,
`ω_pe/Ω_ce = 12.5`, `d_i = 14.14 d_e`, box `20 d_i`, grid 576², cfl 0.95 ⇒
`Δt = 0.33 ω_pe⁻¹`, i.e. **37.9 steps per Ω_ce⁻¹** and 7 580 per Ω_ci⁻¹):

| Quantity | Value | Verdict for whistler |
|---|---|---|
| Box | 283 d_e; Δk d_e = 0.0222 | **Good.** ~27 discrete modes inside k d_e = 0.4–1.0; larger than most published boxes. |
| Δx | 0.491 d_e ⇒ 15.5 cells per λ_peak; 9 cells at the strong band edge (k d_e = 1.4) | **Marginal but usable** for the unstable band. Coarser than every study above. |
| Δx/λ_De | 12.3 (β_e∥ = 0.5 ⇒ λ_De = 0.04 d_e) | **Weakest point** of the setup (mirror/firehose runs have 8.7). Grid heating grows with run length — the strongest argument for not running longer than needed. |
| Δx/ρ_e | ρ_e = 0.61–0.87 d_e ⇒ 1.2–1.8 cells per ρ_e | Marginal for perpendicular electron kinetics. |
| Δt | 0.33 ω_pe⁻¹ (19 steps per plasma period) | Stable; accuracy marginal but identical to the whole matrix. |
| ppc | 1000 | Comfortable (literature: 48–10⁴). |

Verdict: **the box and ppc are fine; the grid is marginal-but-workable;
the defaults for duration and output cadence are wrong for electron
scales** and must be overridden per job (they are env-overridable by
design):

- `nmax = 1.2M` steps = 31 600 Ω_ce⁻¹ is ~24× past the saturation of the
  *slowest* case; the extra 1.05M steps only accumulate grid heating at
  Δx/λ_De = 12.3, at ~40 h × 1024 ranks per run.
- `fields_every = 500` ⇒ Δt_out = 13.2 Ω_ce⁻¹ ⇒ Nyquist 0.24 Ω_ce, **below
  every whistler ω_r in §1**: with the default cadence the ω–k diagram of
  the whistler branch is pure aliasing (this is the case §7.6 of
  `CodeforAnalisys/README.md` warns about).

## 4. Run configuration adopted for the whistler jobs

**Decision (2026-09-28): the whistler family runs on a refined grid.**
Halving Δx (576² → 1152²) and doubling ppc (1000 → 2000) was chosen to
fix the two marginal entries of §3 (Δx/λ_De 12.3 → 6.1; 31 cells per
λ_peak, 2.5–3.5 cells per ρ_e). This deliberately departs from the
mirror/firehose numerics — it is a per-family refinement, uniform across
the three whistler cases and their future bi-kappa twins, applied by
**environment overrides only** in the job scripts (the `.cxx` files and
the shared header keep the matrix defaults; the `psc_units` whistler
profiles describe the production values). Whistler↔whistler comparisons
stay exactly parallel; whistler↔mirror/firehose comparisons are now
cross-resolution and must be made in physical units only.

With ngrid 1152 the CFL timestep halves automatically:
`Δt = 0.165 ω_pe⁻¹` ⇒ **75.8 steps per Ω_ce⁻¹**, 15 160 per Ω_ci⁻¹, so
all step-denominated cadences double to keep the same *physical* cadence.
Grid, ppc and output cadence are identical in all `cosma_jobs/simulacion/sim_whistler_*.sh`;
the run length is set per regime (next subsection):

| Variable | Value | In physical units | Why |
|---|---:|---|---|
| `PSC_NGRID` | 1 152 | Δx = 0.245 d_e; Δx/λ_De = 6.1 | Resolves the unstable band with 18–31 cells/λ and ~3 cells per ρ_e; matches the finer end of the 2D literature (§2). |
| `PSC_NICELL` | 2 000 | 5.3×10⁹ particles | Lower noise floor for the weak case (γ = 0.0075 Ω_ce) and cleaner κ_eff estimates. |
| `PSC_NP_Y/Z` | 48×48 | 2 304 ranks, 24×24-cell patches | Same decomposition as the bigbox40 jobs. |
| `PSC_FIELDS_EVERY` | 100 | Δt_out = 1.32 Ω_ce⁻¹ ⇒ ω_Ny = 2.38 Ω_ce | Minimum that resolves the whole branch (ω_r ≤ 0.5 Ω_ce) with ≥ 4× margin *and* keeps ≥ 4 samples per e-fold for the strong-case γ(k) fits. |
| `PSC_ENERGIES_EVERY` | 20 | 0.26 Ω_ce⁻¹ | Global γ from the δB² series (one reduce; ~15 000 ASCII lines). |

### Run length: only until relaxation

Each regime runs **only until the anisotropy has relaxed**, no further.
Rule for the family: `t_end ≈ 8 × t_10`, with `t_10 = 10/γ_max` of the
*slowest* distribution twin of that regime (§1). That covers growth from
the particle noise, saturation, and the relaxation of A_e towards the
marginal state (A_e ≈ 1.32 at β_e∥ = 0.5), and is at or above the
150–450 Ω_ce⁻¹ in which the literature resolves the whole evolution (§2).
PSC always writes a checkpoint when the run ends (`checkpointing_.final`
in `include/psc.hxx`), so a run whose A_e(t) is still falling at the end
is **extended from that checkpoint** instead of every run being paid
long in advance.

| Regime | slowest twin t_10 [Ω_ce⁻¹] | `PSC_NMAX` | t_end [Ω_ce⁻¹] | t_end [Ω_ci⁻¹] | `PSC_PARTICLES_EVERY` (VDF dumps) | `PSC_CHECKPOINT_EVERY` | walltime |
|---|---:|---:|---:|---:|---|---:|---:|
| strong (bi-Max, κ5, κ3) | 69 (κ=3) | **40 000** | 528 | 2.6 | 2 500 = 33 Ω_ce⁻¹ (~17) | 20 000 + final | 12 h |
| moderate | 228 (κ=3) | **140 000** | 1 847 | 9.2 | 10 000 = 132 Ω_ce⁻¹ (15) | 70 000 + final | 36 h |
| weak | 1 330 (bi-Max) | **300 000** | 3 958 | 19.8 | 20 000 = 264 Ω_ce⁻¹ (15) | 150 000 + final | 72 h |

The weak case is the one exception to the 8× rule (it would need ~800 000
steps): it sits close to threshold (A_e = 1.5 vs ~1.32 marginal) and has
little anisotropy to relax, so it runs ~3 × t_10 and is extended only if
A_e(t) is still falling. Distribution twins of a regime share every
value in the table; regimes differ in length **on purpose**.

Stopping criterion to decide an extension: the domain-averaged A_e(t)
from the moments changes by less than ~2% over the last 100 Ω_ce⁻¹.

Time conversions for the analysis of these runs: 1 Ω_ce⁻¹ = 75.8 steps;
1 Ω_ci⁻¹ = 200 Ω_ce⁻¹ = 15 160 steps; ω/Ω_ci = (ω/Ω_ce)·200 and
k d_i = k d_e · 14.14 (the electron presets of `dispersion_analysis.py`
apply these rescalings; use `DISPERSION_MODE=whistler`).

**Campaign order.** The first batch is the **strong series** — the
distribution twins `psc_whistler_bimaxwellian_strong`,
`psc_whistler_bikappa5_strong`, `psc_whistler_bikappa3_strong` (identical
β/A defines; only `PSC_USE_KAPPA`/`PSC_KAPPA` differ), all with the §4
configuration and the strong run length (40 000 steps). Their growth
rates are within 20% of each other (§1), so one configuration serves the
three. Moderate and weak wait for these
results. Build/submit commands: `cosma_jobs/README.md`.

## 5. Resource and storage budget

Calibrated against the observed ~48 h of a standard ionic run
(576², 1000 ppc, 1.2M steps on 1024 ranks ⇒ R ≈ 4.5×10⁶
particle-pushes/s/core, matching §4 of `ESCALADO_INESTABILIDADES.md`):

- **RAM**: 5.31×10⁹ particles ⇒ 170–340 GB aggregate ⇒ ~3–4 GB per node
  on 83 nodes (COSMA7 nodes have 512 GB) — never the constraint. The node
  count is set by wallclock, not memory.
- **Work per run** scales with nmax: 5.31×10⁹ particles × nmax steps;
  300 000 steps = 2.0× one ionic run.

| Regime | Wallclock (2 304 ranks) | core-h per run | `pfd` + `pfd_moments` | `prt_*` | Durable per run |
|---|---:|---:|---|---|---:|
| strong | ~5.7 h | ~16 000 | 400 + 400 files, ~74 GB | ~17 × 8 GB ≈ 130 GB | **~0.2 TB** |
| moderate | ~20 h | ~55 000 | 1 400 + 1 400, ~260 GB | 15 × 8 GB ≈ 115 GB | ~0.38 TB |
| weak | ~43 h | ~118 000 | 3 000 + 3 000, ~560 GB | 15 × 8 GB ≈ 120 GB | ~0.7 TB |

The first batch (strong series: bi-Max, κ5, κ3) is **~47 000 core-h and
~0.6 TB durable** in total, instead of the ~354 000 core-h and ~2.1 TB
it would have cost at a uniform 300 000 steps. Each run also writes two
transient checkpoints of ~340 GB (mid-run + final): delete the mid-run
one when the run ends and keep the final one until the relaxation is
confirmed. The moments are ~74% of the field-side volume because the
header ties their cadence to `fields_every`.

## 6. Remaining open decision

**Isotropic control run** (A_e = 1, everything else identical to §4) to
measure the pure grid-heating + noise baseline (recommendation §6.2 of
`ESCALADO_INESTABILIDADES.md`). This is a new case file, i.e. a
case-matrix addition, and is still pending. At Δx/λ_De = 6.1 and at
most 19.8 Ω_ci⁻¹ of run time the expected heating is far smaller than at the old
12.3, which lowers the urgency but does not replace the measurement.

## References

- An, X., et al. (2017), JGR Space Physics 122, 2001, doi:10.1002/2017JA023895.
- Abdul, R. F., Matthews, A. P., Mace, R. L. (2021), Phys. Plasmas 28, 062104, doi:10.1063/5.0047638.
- Cui, et al. (2022), Front. Astron. Space Sci. 9:941241, doi:10.3389/fspas.2022.941241.
- Gary, S. P., Liu, K., Winske, D. (2011), Phys. Plasmas 18, 082902, doi:10.1063/1.3610378.
- Kim, H. P., Hwang, J., Seough, J. J., Yoon, P. H. (2017), JGR Space Physics 122, 4410, doi:10.1002/2016JA023558.
- Lazar, M., López, R. A., Shaaban, S. M., Poedts, S., Fichtner, H. (2019), Astrophys. Space Sci. 364:171 (arXiv:1910.01506).
