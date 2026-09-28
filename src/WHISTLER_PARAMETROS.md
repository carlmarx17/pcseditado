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

Set identically in the three `cosma_jobs/simulacion/sim_whistler_*.sh`
(and to be reused verbatim by future bi-kappa whistler twins). These are
**environment overrides only**: `src/` is untouched, grid/box/ppc/Δt stay
identical to the whole anisotropy matrix, so cross-family comparability is
preserved — only the sampling and the stopping time are adapted to the
electron scales.

| Variable | Value | In physical units | Why |
|---|---:|---|---|
| `PSC_NMAX` | 150 000 | 3 958 Ω_ce⁻¹ = 19.8 Ω_ci⁻¹ | ≥ 3× the weak case's growth-to-saturation (1 330 Ω_ce⁻¹), leaving a ≥ 2 600 Ω_ce⁻¹ relaxation tail; 8× cheaper (~5 h on 1024 ranks) and 8× less accumulated grid heating than the 1.2M default. Literature runs are 150–450 Ω_ce⁻¹. |
| `PSC_FIELDS_EVERY` | 50 | Δt_out = 1.32 Ω_ce⁻¹ ⇒ ω_Ny = 2.38 Ω_ce | Resolves the whole whistler branch (ω_r ≤ 0.5 Ω_ce) with margin ≥ 4×. Per-mode γ(k) sampling: 4.3 / 14 / 100 samples per e-fold (strong/moderate/weak); ~3 000 snapshots (~130 GB fields+moments). Halve to 25 only if the strong-case γ(k) fits come out noisy. |
| `PSC_ENERGIES_EVERY` | 10 | 0.26 Ω_ce⁻¹ | Global γ comes from the δB² series essentially for free (one reduce); 22 samples per e-fold even for the strong case. |
| `PSC_PARTICLES_EVERY` | 5 000 | 132 Ω_ce⁻¹ | 30 VDF snapshots (~1 GB each, central 20% window): resolves the A_e(t) and κ_eff(t) relaxation instead of the 15 sparse dumps of the default. |
| `PSC_CHECKPOINT_EVERY` | 50 000 | ⅓ of the run | The old 150 000 exceeds the new nmax (zero mid-run checkpoints). |

Everything else (`PSC_NGRID = 576`, `PSC_NICELL = 1000`, `np = 32×32`)
stays at the matrix values.

Time conversions for the analysis of these runs: 1 Ω_ce⁻¹ = 37.9 steps;
1 Ω_ci⁻¹ = 200 Ω_ce⁻¹ = 7 580 steps; ω/Ω_ci = (ω/Ω_ce)·200 and
k d_i = k d_e · 14.14 (the electron presets of `dispersion_analysis.py`
apply these rescalings; use `DISPERSION_MODE=whistler`).

## 5. Open decisions (not applied — they break matrix parity)

Both options below change the numerics of the whistler family away from
the mirror/firehose matrix and therefore need an explicit decision (see
`.claude/skills/psc-case-integrity/SKILL.md`):

1. **Finer grid.** `PSC_NGRID = 1152` in the whistler jobs would give
   Δx = 0.245 d_e (Δx/λ_De = 6.1, 31 cells per λ_peak) at ~8× the cost
   (4× cells, 2× steps for the same physical time — Δt halves with Δx).
   It also makes the whistler runs numerically different from their own
   `psc_units` profiles and from the rest of the matrix.
2. **Isotropic control run** (A_e = 1, everything else identical) to
   measure the pure grid-heating baseline at Δx/λ_De = 12.3
   (recommendation §6.2 of `ESCALADO_INESTABILIDADES.md`). This is a new
   case file, i.e. a case-matrix addition.

Until decided, the adopted position is: run the matrix grid (576²), keep
runs short (§4), and treat the isotropic-control question as pending.

## References

- An, X., et al. (2017), JGR Space Physics 122, 2001, doi:10.1002/2017JA023895.
- Abdul, R. F., Matthews, A. P., Mace, R. L. (2021), Phys. Plasmas 28, 062104, doi:10.1063/5.0047638.
- Cui, et al. (2022), Front. Astron. Space Sci. 9:941241, doi:10.3389/fspas.2022.941241.
- Gary, S. P., Liu, K., Winske, D. (2011), Phys. Plasmas 18, 082902, doi:10.1063/1.3610378.
- Kim, H. P., Hwang, J., Seough, J. J., Yoon, P. H. (2017), JGR Space Physics 122, 4410, doi:10.1002/2016JA023558.
- Lazar, M., López, R. A., Shaaban, S. M., Poedts, S., Fichtner, H. (2019), Astrophys. Space Sci. 364:171 (arXiv:1910.01506).
