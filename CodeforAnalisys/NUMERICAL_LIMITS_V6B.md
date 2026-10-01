# What limits the v6b mirror-moderate runs, with numbers

Checks made on the analysis products of the three runs (bi-Maxwellian, kappa 5,
kappa 3; beta_i_par = 5, A_i = 2, m_i/m_e = 200, 576^2 cells, 20 d_i box).
Each item says what was measured, what it affects, and what would remove it.
Changing a run parameter is a decision for the case matrix, not made here.

## 1. The Debye length is not resolved: grid heating of the electrons

| Quantity | Value |
|---|---|
| B0 (profile key `vA_over_c`, historical name) | 0.08 = Omega_ce/omega_pe |
| omega_pe/Omega_ce | 12.5 |
| v_A/c = B0/sqrt(m_i/m_e) | 0.0057 |
| Cell size | 0.49 d_e |
| Electron Debye length at t = 0 | 0.057 d_e |
| dx / lambda_De at t = 0 | 8.7 |
| dx / lambda_De at the end | 3.0 |
| Electron thermal energy, end / start | 8.3 to 8.6 |
| Particle + field energy, relative change | +66 % |

The electrons heat until the Debye length reaches about dx/3, the usual end
point of finite-grid heating, the same in the three runs. The energy is not
conserved: the gain of the electrons (0.035 per particle in code units) is six
times the loss of the ions.

Affects: the electron beta (1 to 8), hence the mirror threshold and, weakly,
the ion-cyclotron rate (both now evaluated with the measured electrons); every
energy budget.

Removes it: dx <= about 3 lambda_De. With the same grid that means a larger B0
(omega_pe/Omega_ce of about 4 instead of 12.5), which also shortens the run in
steps; or a 3 times finer grid; a higher-order particle shape and current
smoothing lower the rate without removing it.

## 2. The ions lose energy from t = 0

Before any wave grows (t Omega_ci < 30) the ion thermal energy falls at
1.7e-3 to 1.8e-3 Omega_ci (T_par by 6 %, T_perp by 4 %), and by 15 % over the
run. The wave magnetic energy at saturation is 2e-4 per particle against an ion
loss of 6e-3: the numerical sink is thirty times the energy of the wave.

The background relaxation rate of the kappa index measured independently,
nu_0 = (1.5 to 2.2)e-3 Omega_ci (kappa_dynamics.py), is the same number. This
points to one numerical process (thermalization by the particle noise, Jubin et
al. 2024) behind both; the isotropic control runs decide.

Affects: beta_i_par at the onset (4.7, not 5), the kappa index in time, any
statement on how the released anisotropy energy is shared.

## 3. One unstable mode in the box

The box allows k_par d_i = 0.314 n. Linear theory at the initial state gives
0.124 Omega_ci for n = 1 and 0.05 for n = 2, which becomes stable as soon as the
anisotropy drops. The wave is one mode and its mirror image: a standing wave
(the stripes of |B| perpendicular to B0 in 07_structures).

With dB/B0 = 0.2 the trapping frequency sqrt(k v_perp Omega dB/B0) is 0.38
Omega_ci, three times the growth rate, and the trapping half-width in parallel
velocity is 2.4 v_A = 1.5 sigma_par0. The resonance at |v_par| = 1.8 to 2.4 v_A
therefore covers the bulk up to about 3 sigma_par0 and not the tail.

Affects: saturation is by trapping in a coherent wave, not by quasi-linear
diffusion in a spectrum; published simulations use boxes of hundreds of d_i.
The anisotropy left above marginal in the kappa runs
(trajectory_linear_theory.py) may be specific to this.

Removes it: the 40 d_i box profiles (twice the modes) or longer in the parallel
direction only.

## 4. The mirror mode

The plasma is mirror-unstable throughout (Gamma from 8.6 to about 1). The box
holds oblique modes at k rho_i of 1 and above only ((0.31, 0.31) and
(0.31, 0.63) in k d_i), not the long parallel wavelengths; the plane is 2D; and
the ion-cyclotron mode removes the anisotropy in about 60 Omega_ci^-1. The
solver of the project has no oblique root, so the mirror growth rate at these
parameters is not known here.

## 5. The run ends while the kappa runs still relax

The kappa runs saturate about 10 Omega_ci^-1 later than the bi-Maxwellian one
and their anisotropy is still falling at t Omega_ci = 158 (about 0.003 per
Omega_ci^-1). Final-state comparisons between the three are not at the same
stage.

## 6. Estimators

- The anisotropy from the box moments and from the particles of the prt window
  in the local-field frame differ by up to 0.05 once dB/B0 is 0.2.
- v6b has six particle snapshots; the amplitude series is that of the k-shell
  of the mode (v6c writes the single mode).
- `linear_theory.py` gives both species the same kappa and cannot evaluate a
  damped bi-kappa root.

## What is not a defect

The growth rate falls with a stronger tail because the three runs have the same
temperature; published work that reports a rise holds the core fixed and so
compares a hotter plasma (FIGURE_GUIDE.md, section 2).
