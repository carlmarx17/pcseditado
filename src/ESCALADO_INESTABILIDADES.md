# From reconnection to large runs of anisotropy instabilities

Analysis based on (a) direct reading of this repository (`src/psc_reconnection.cxx`,
`src/psc_anisotropy_case.hxx`, `src/include/setup_particles.hxx`, COSMA jobs) and
(b) verified literature (links at the end). What comes from reading the code is marked
**[code]**; what is verified in papers, **[lit.]**; general practice not
verified against a specific source, **[general]**. Where I could not verify something, I say so.

---

## 1. Starting point: what's already in the repo **[code]**

Contrary to the premise of "adapting the reconnection deck," the repo **already contains** an
instability framework (`psc_anisotropy_case.hxx` + bi-Maxwellian and bi-kappa firehose/mirror
cases). The right question is no longer "how to convert the Harris sheet into a
homogeneous plasma" (done), but **whether the current setup scales well and what needs fixing before
burning hours on COSMA**. Comparison of the two decks:

| Parameter | `psc_reconnection.cxx` | `psc_anisotropy_case.hxx` |
|---|---|---|
| Geometry | 2D `dim_yz`, double Harris | 2D `dim_yz`, homogeneous, B₀ = ẑ |
| Box | 25.6 × 51.2 d_i | 20 × 20 d_i (bigbox: 40 × 40) |
| Grid | 256 × 512 → Δx = 0.1 d_i | 576² → Δx = 0.0347 d_i (bigbox 1152²) |
| m_i/m_e | 25 | 200 |
| ω_pe/Ω_ce | 2 | 12.5 (see §5, naming of `vA_over_c`) |
| ppc (`nicell`) | 100 × 4 species | 1000 × 2 species |
| CFL | 0.99 | 0.95 |
| Distribution | κ = 3 multivariate | bi-Maxwellian or bi-kappa (T⊥ ≠ T∥ via `npt.T[]`) |
| Boundaries | Periodic | Periodic |
| Balance | every 500 | every 2500 |
| Default duration | nmax 10⁷ (manual cap) | nmax 1.2 × 10⁶ ≈ 158 Ω_ci⁻¹ (computed below) |

The anisotropic initialization is already structurally correct: `npt.T[0]=T[1]=T⊥`,
`npt.T[2]=T∥` with B₀ along z, grid axes aligned with B₀ — which is the only case in
which `T[3]` by grid axes is equivalent to a gyrotropic bi-Maxwellian. If you ever
tilt B₀, the temperature matrix has to be rotated by hand (the sampler doesn't know about B).

The `createKappaMultivariate` sampler **[code]** uses a Gaussian-Gamma scale
mixture: `Y ~ Gamma(κ−0.5)`, `S = √((κ−1.5)/Y)`, `p_i = Z_i·S·√(T_i/m)`.
That generates the standard bi-kappa f ∝ [1 + Q/(κ−3/2)]^−(κ+1) in the
"temperature-preserving" convention: the variance is exactly T_i (E[S²] = 1). This is the
correct convention for comparing with NHDS/LEOPARD/ALPS **if** you pass them the same physical
T; check which θ vs T convention each solver uses (some parametrize with
θ² = (1−3/(2κ))·2T/m).

## 2. What the literature does in large runs **[lit.]**

Parameters extracted from the papers (read directly, not from memory):

**Micera et al. 2020, ApJ 893:130** (parallel proton firehose, full PIC
semi-implicit ECsim, 1D): box L = 60 d_i chosen explicitly to fit
**>20 wavelengths of the most unstable mode**; Δx ≈ 0.074 d_i;
Δt = 0.5 ω_pe⁻¹; **10⁴ ppc per species**; periodic; realistic mass;
ω_pe/Ω_ce = 63.24. They report that runs with different resolution and ppc give
similar results (explicit convergence test). Note: ECsim is
semi-implicit energy-conserving and is **not** required to resolve λ_De — explicit
PSC is (§3).

**Hellinger et al. 2019, ApJ 883:178** (firehose vs turbulence, hybrid expanding
box, 3D): grid 512 × 512 × 256; Δx = Δy = 0.25 d_i, Δz = 0.5 d_i (box
128 × 128 × 128 d_i); **400 ppc** (protons); Δt = 0.05 Ω_ci⁻¹ with subcycling of the
B field at Δt/10; resistivity η = 10⁻³ μ₀v_A²/ω_ci to avoid energy
accumulation at the grid scale; expansion t_exp = 10⁴ Ω_ci⁻¹; periodic. Hybrid:
no electron scale to resolve, which is why they can use Δx of 0.25–0.5 d_i.

**Riquelme, Quataert & Verscharen 2015, ApJ 800:27** (mirror + shear-driven IC,
full PIC TRISTAN-MP, β ~ 1–100): the nonlinear state is dominated by mirror; the
anisotropy saturates near the mirror linear threshold; δB ~ 0.3⟨B⟩ in the
secular phase; μ stops being conserved when δB ≳ 0.1⟨B⟩. I did not extract their
numerical-resolution table — if you need their Δx/ppc, their section 2 has to be read in detail.

**Relevant to your kappa case** (existence verified, setups not extracted):
López et al. 2019, ApJL 873:L20 (electron bi-kappa firehose, PIC); López et al.
2022, ApJ 930:158 (2D PIC firehose coupling p⁺/e⁻ scales); "Hybrid Simulation and
Quasi-linear Theory of Bi-Kappa Proton Instabilities" (ApJ 2023); and a
rejection-sampling method for kappa in PIC (arXiv:2512.04272) against which you can
benchmark your sampler.

**Recurring design patterns** **[general, consistent with the above]**:

- 1D is enough for modes with k ∥ B (parallel firehose, parallel EMIC); 2D is the minimum
  for mirror and oblique firehose; 3D only when you're pitting parallel against
  oblique modes simultaneously or adding turbulence. Your `dim_yz` with B₀ = ẑ captures
  k∥ (z) and k⊥ (y) in one plane: correct for mirror and oblique firehose, with the
  2D limitation of a single k-plane.
- Box: the operative rule is L ≳ 10–20 λ_peak of the dominant mode, i.e.
  k_min = 2π/L ≲ k_peak/10–20. Full ion PIC: boxes of 20–100 d_i. Your 20 d_i gives
  k_min·d_i ≈ 0.31 — for firehose/mirror with k_peak·d_i ~ 0.3–0.8 that leaves the peak
  barely at the 1st–3rd harmonic: **thin**. The bigbox40 (k_min·d_i ≈ 0.157) is the
  minimum defensible choice; for the post-saturation inverse cascade and low-k
  oblique modes, even larger is better.
- ppc in explicit full PIC: 100–1000 typical, 10⁴ in luxury 1D setups. Your 1000 are well
  positioned; noise in energy scales ∝ 1/ppc and in amplitude ∝ 1/√ppc.
- Duration: growth with γ/Ω_ci ~ 10⁻³–10⁻¹ depending on proximity to the threshold →
  saturation in tens–hundreds of Ω_ci⁻¹; the quasilinear relaxation toward the
  marginal threshold (what gets compared against the β∥–T⊥/T∥ plane) requires hundreds to ~10³ Ω_ci⁻¹.

## 3. Critical numerics when moving from reconnection to instabilities

**Grid heating / λ_De** **[lit. + code]**. Explicit PIC with linear interpolation
heats numerically if Δx ≳ 3–3.5 λ_De (Birdsall & Langdon; see also
arXiv:2606.25528 on numerical thermalization and arXiv:2503.05123 on smoothing).
Numbers from your decks:

- Reconnection: T_e = 1/48 → λ_De = 0.144 d_e; Δx = 0.5 d_e → **Δx/λ_De ≈ 3.5**. At the limit but defensible.
- Anisotropy: T_e∥ = β_e∥·B₀²/2 = 0.0032 (β_e∥ = 1) → λ_De = 0.057 d_e;
  Δx = 0.491 d_e → **Δx/λ_De ≈ 8.7**. The comment in `psc_anisotropy_case.hxx`
  says "dx/lambda_De ~ 3.78", but that number only comes out when using an
  ionic temperature (√T_i∥ with β_i = 5 gives 3.9); with the **electron** λ_De — which is the one
  that governs grid heating — you're ~2.5× above the classical criterion. This is the
  first thing I'd check (checklist §6): it may be injecting spurious heating
  into the electrons over the course of 10⁶ steps, and a T_e(t) that only rises
  directly contaminates your β∥–T⊥/T∥ plane.

Mitigations if the control run confirms heating: raise β_e∥ (hotter electrons
→ larger λ_De), refine the grid (expensive: cost ∝ N²·steps in 2D), or current
smoothing. **I did not find a configurable binomial filter / current smoothing in
public PSC** — if your edited version didn't add one, don't count on it.

**Particle noise and mode seeding** **[general + code]**. In a homogeneous
plasma the instability grows from the thermal noise of the macroparticles. With
more ppc the noise floor drops (∝ 1/ppc in energy), the linear phase lasts longer and
the γ fit is cleaner; with few ppc the modes start from already-nonlinear
amplitudes or the noise buries small γ. Detail of your code: PSC initializes
all particles at the **cell center** (`x_cc`), not uniformly — the
initial density-noise spectrum is not that of a thermal plasma and takes ~ one
plasma period to thermalize. It's not a problem for γ (you measure after the
first few Ω_ci⁻¹) but it explains initial transients. Also, `createKappaMultivariate`
uses `std::random_device` per thread **[code]**: runs are not bit-for-bit
reproducible; to compare γ between identical runs, consider a fixed seed.

**Discrete k and comparison with linear theory** **[general]**. The periodic box
only admits k_n = 2πn/L. Your measured γ for "the dominant mode" is the γ(k_n) of the
harmonic closest to the theoretical peak, not the continuum γ_max. Compare against
NHDS/LEOPARD/ALPS **evaluated exactly at the k_n of your box** (and in your k
direction within the y-z plane), not against the maximum of the curve. This is the
physical reason why a small box ⇒ a smaller apparent γ and a shifted apparent
threshold — it matters directly for your contours in the β∥ vs T⊥/T∥ plane.

**CFL and dispersion** **[general + code]**. `cfl = 0.99` (reconnection) leaves
almost no margin; the anisotropy case's 0.95 is the usual choice. Near the Courant
limit the EM dispersion error of the Yee scheme is largest precisely at high k;
for 10⁶-step runs with whistlers/EMIC in play, 0.75–0.95 is more prudent.
Micera et al. use a different (semi-implicit) scheme, so their Δt is not comparable.

**Boundaries and initialization** — already settled in your case: periodic +
homogeneous with no drifts (the reconnection deck needed a double Harris sheet
precisely to be periodic; the homogeneous case has no such restriction). No initial
field perturbation is needed either: it seeds itself from the noise.

**Conservation and correctors** **[code]**. Marder every 100 + Gauss check
every 100 are active in the anisotropy case (in the reconnection case the Gauss
check is disabled, negative interval). `DiagEnergies` defaults to 0 in the header
(`PSC_ENERGIES_EVERY_DEFAULT 0`) even though your runbook says 5000 via the
environment: for instabilities that time series is your primary diagnostic (γ of
the δB² growth comes out of it for free) — turn it on ALWAYS and at high cadence
(50–100 steps; it's cheap, a single global reduce).

## 4. Timescales and cost (numbers from your setup) **[code, arithmetic]**

With m_i/m_e = 200, B₀ = 0.08 (ω_pe units): Ω_ci⁻¹ = m_i/B₀ = 2500 ω_pe⁻¹.
Δt = 0.95 · (0.491/√2) ≈ 0.33 ω_pe⁻¹ → **~7600 steps per Ω_ci⁻¹**.

- nmax 1.2 × 10⁶ ≈ 158 Ω_ci⁻¹: enough for growth and saturation of
  moderate/strong drives (γ/Ω_ci ≳ 10⁻²); **short** for weak drives near the threshold
  and for the long quasilinear relaxation. Compute nmax per case: t_fin ≈ 10/γ + 200–500 Ω_ci⁻¹.
- Outputs: fields every 500 steps = 0.066 Ω_ci⁻¹ (≈15 samples per Ω_ci⁻¹ — plenty
  for γ; the criterion is ≥10 samples per e-folding, i.e. interval ≤ 1/(10γ)).
- Checkpoint every 5000 steps = 0.66 Ω_ci⁻¹ → 240 checkpoints in one run. Each
  checkpoint serializes ~6.6 × 10⁸ particles (≥20 GB): that's a lot of I/O. With the
  48 h limit of cosma7-rp, checkpointing every ~2–4 h of wallclock is enough
  (equivalent to every 5–10 × 10⁴ steps).

**Memory**: N_prt = n_cells × nicell × n_species. Standard: 576² × 1000 × 2 =
6.6 × 10⁸ particles; at ~32–64 B/particle (single precision + sorting overhead)
→ 25–45 GB aggregate + fields (negligible in comparison). Bigbox40: ×4.

**Core-hours** (formula, not a promise): cost ≈ N_prt × n_steps / R, with
R ≈ 3–10 × 10⁶ particle-pushes/s/core on CPU **[general]**. Standard:
6.6 × 10⁸ × 1.2 × 10⁶ / (5 × 10⁶) ≈ 4 × 10⁴ core-h (~40 h on 1024 ranks). Bigbox40:
~1.6 × 10⁵ core-h — consistent with your job of 83 nodes × 28 × 48 h, which already
plans to resume from checkpoint. The 48 × 48 decomposition with 24²-cell patches is
fine; note that in a **homogeneous** plasma the load balancer barely does anything
(unlike the Harris sheet, which concentrates particles): you can raise
`PSC_BALANCE_INTERVAL` or disable it and save that overhead.

**Diagnostics — what to save**: prioritize (1) a dense energy time series (cheap),
(2) B fields at fixed cadence for δB(k,t) spectra and per-mode γ fitting,
(3) moments (n, v, P⊥, P∥ per species) at the same cadence — your trajectory in the
β∥–T⊥/T∥ plane comes out of P⊥/P∥, (4) raw particles only in a subregion and rarely
— your deck already restricts to the 0.4–0.6 window of the box **[code]**, good. The f(v)
for comparison with kappa theory are reconstructed from those sparse dumps.

## 5. Specific checks on your code (direct reading) **[code]**

1. **`vA_over_c` is not v_A/c.** The code does `g.B0 = g.vA_over_c` with B in
   units where Ω_ce = B. That fixes ω_ce/ω_pe = 0.08 (ω_pe/Ω_ce = 12.5); the resulting
   physical v_A/c is B₀/√(m_i n) = 0.08/√200 ≈ 5.7 × 10⁻³. The betas are fine
   (they're defined directly from B₀²), but any interpretation of
   velocities in units of "input v_A" is off by a factor of √m_i.
2. **Δx/λ_De ≈ 8.7 with the electron λ_De** (§3) vs. the 3.78 commented in the
   header. Verify with a control run.
3. **`DiagEnergies` defaults to 0** — make sure all jobs export
   `PSC_ENERGIES_EVERY` (and set it down to 50–100).
4. **Non-reproducible RNG seed** (`std::random_device` in the kappa sampler).
5. **Checkpoint every 5000 steps** = excessive I/O (§4).
6. **CFL**: 0.95 OK; don't inherit the 0.99 from the reconnection deck.
7. The reconnection deck initializes an **isotropic** kappa (T[0]=T[1]=T[2]); your
   anisotropy case already does the bi-kappa correctly via T[] — nothing to port back.

## 6. Decision checklist before the next large campaign

1. For each point (β∥, T⊥/T∥, κ): run NHDS/LEOPARD/ALPS first → k_peak,
   γ_max, mode direction. From there: L ≥ 10–20·(2π/k_peak) and nmax ≥
   (10/γ_max + 300 Ω_ci⁻¹)/Δt. The box is decided from the solver, not the other way around.
2. Isotropic control run (T⊥/T∥ = 1, everything else the same): measure pure
   grid heating (secular T_e(t), T_i(t)) and the δB²(k) noise floor. If T_e rises
   appreciably over ~100 Ω_ci⁻¹ → apply §3 mitigations before producing.
3. Convergence in ppc (250/500/1000) and in Δx (×2) in a small box, a single
   physical point, comparing γ of the dominant mode. Micera et al. do exactly this.
4. γ per mode: exponential fit of log|δB_k|² in the linear phase, for each
   harmonic k_n; compare against the solver evaluated at those same k_n.
5. Total energy conserved to <1% over the whole run (with Marder active, ΔE is
   diagnostic, not an energy correction).
6. Budget: cost ≈ N_prt·n_steps/R (§4); add 20% for diagnostics and I/O and
   plan resumptions for >48 h.
7. Decide 2D vs 3D based on the physics, not by default: mirror vs EMIC in
   competition (your β∥–T⊥/T∥ plane with T⊥ > T∥) is sensitive to dimensionality;
   Riquelme et al. 2015 (2D/3D PIC) and Hellinger et al. 2019 (3D hybrid) are the
   reference points for comparison.

## Unverified / pending

- Resolution details from Riquelme et al. 2015 and from the López et al.
  2019/2022 setups (the papers exist and are the relevant ones; I did not extract their tables).
- Whether public PSC (or your fork) has configurable current smoothing: I did not
  find it in the documentation or in the sources I reviewed.
- Exact conservation properties of PSC's `1vbec` pusher (it is
  charge-conserving by Villasenor–Buneman construction; regarding its grid-heating
  behavior, I found no published characterization).
- The exact "3.4–5" factor cited in your header as the safe limit for Δx/λ_De: the
  classical Birdsall & Langdon criterion is of order 3; the precise value depends on
  the interpolation order and the scheme. Treat it as an order of magnitude.

## Appendix: `psc_reconnection_comparable` case **[code]**

Created for the reconnection ↔ instabilities comparison with parameters matched to
the anisotropy cases (bigbox40 convention):

| Parameter | Value | Status |
|---|---|---|
| m_i/m_e | 200 | matched |
| Box | 40 × 40 d_i (sheets at ±10 d_i) | matched |
| Grid | 1152² → 28.8 cells/d_i, Δx = 0.491 d_e | matched |
| `nicell` | 1000 | matched |
| CFL | 0.95 | matched |
| κ | 3.0 | matched |
| np | 48 × 48 (2304 ranks, 24² patches) | matched |
| Outputs/checks/balance/energies | same intervals and same env-overrides | matched |
| ω_pe/Ω_ce | **2.0** (anisotropy: 12.5) | **deliberate difference** |

The reason for the single difference: in Harris, pressure balance fixes
T_e = 1/(2(ω_pe/Ω_ce)²(1+T_i/T_e)); with 12.5 you'd get λ_De = 0.023 d_e →
Δx/λ_De ≈ 21 (guaranteed grid heating). With 2.0: λ_De = 0.144 d_e →
Δx/λ_De = 3.4, just as sound as the corrected anisotropy cases. Consequence:
compare in ionic units (d_i, Ω_ci, v_A, B₀), not in ω_pe or c.

Case scales: Ω_ci⁻¹ = 400 ω_pe⁻¹ ≈ 1212 steps (Δt ≈ 0.33 ω_pe⁻¹);
default nmax 250,000 ≈ 206 Ω_ci⁻¹ (`PSC_NMAX` to change it). Particles
≈ 1152² × 1000 × Σn ≈ 6–7 × 10⁸ (similar to the standard anisotropy case).
Cost ≈ 250k steps: ~6× cheaper per Ω_ci⁻¹ than anisotropy in step count, total on the
order of 3–5 × 10⁴ core-h with R ~ 5 × 10⁶ pushes/s/core.

Warning found while reviewing the bigbox cases **[code]**: `psc_firehose_*_bigbox40`
has `PSC_DOMAIN_DI=40` at compile time but `ngrid` defaults to **576** (inherited
from the header); getting the correct resolution depends on exporting `PSC_NGRID=1152`
in the job. If launched without that variable, it silently runs at half resolution
(14.4 cells/d_i, Δx/λ_De ×2 worse). The comparable reconnection case already ships with 1152
as the default to avoid that failure mode.

## References consulted

- Micera et al. 2020, ApJ 893:130 — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ab7faa) · [arXiv:1907.08502](https://arxiv.org/abs/1907.08502)
- Hellinger et al. 2019, ApJ 883:178 — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ab3e01) · [arXiv:1908.07760](https://arxiv.org/abs/1908.07760)
- Riquelme, Quataert & Verscharen 2015, ApJ 800:27 — [IOPscience](https://iopscience.iop.org/article/10.1088/0004-637X/800/1/27) · [arXiv:1402.0014](https://arxiv.org/abs/1402.0014)
- López et al. 2019, ApJL 873:L20; López et al. 2022, ApJ 930:158 (setups no extraídos)
- Hybrid simulation & QL theory of bi-kappa proton instabilities — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/aceb5b)
- Kappa sampling en PIC — [arXiv:2512.04272](https://arxiv.org/abs/2512.04272)
- Grid heating / termalización numérica — [arXiv:2606.25528](https://arxiv.org/abs/2606.25528) · smoothing: [arXiv:2503.05123](https://arxiv.org/abs/2503.05123) · [Finite spatial-grid effects (CPC)](https://www.sciencedirect.com/science/article/abs/pii/S001046552030268X)
- PSC: Germaschewski et al. 2016, JCP 318:305 — [arXiv:1310.7866](https://arxiv.org/abs/1310.7866) · [repo](https://github.com/psc-code/psc) · [docs](https://psc.readthedocs.io/en/latest/)
- Hellinger et al. 2006 (umbral marginal solar wind, contexto del plano β∥–T⊥/T∥); Agudelo Rueda et al. 2024, ApJ 971:109 (referencia base de tu deck de reconexión)
