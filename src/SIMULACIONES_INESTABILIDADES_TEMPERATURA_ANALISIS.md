# Analysis plan: temperature-anisotropy instabilities

This document summarizes what the analysis must check for each family of
simulations. The concrete commands are in `CodeforAnalisys/README.md`.

## Central variables

For each species:

```text
A = T_perp / T_parallel
beta_parallel = 2 P_parallel / |B|^2
```

In the analysis scripts, `P_parallel` is projected onto the local direction of
`B`, not just onto the `z` axis. This avoids interpreting a local rotation of the
field as thermal relaxation.

## Mirror

Physical condition:

```text
A_i > 1
beta_i_perp * (A_i - 1) = beta_i_parallel * A_i * (A_i - 1) > 1
```

This is a cold-electron, bi-Maxwellian reference, not the full hot-electron
or Kappa threshold. On a parallel-beta plot the reference curve is
`A = (1 + sqrt(1 + 4/beta_i_parallel))/2`.
See `CodeforAnalisys/AUDITORIA_FISICA_PAPER.md` for publication limitations.

Expected signatures:

- growth of compressive fluctuations in `|B|`;
- hole-like or magnetic-mirror structures;
- anticorrelation between density and field magnitude;
- global trajectory approaching the marginal threshold.

Cases:

```text
psc_mirror_bimaxwellian_strong, psc_mirror_bimaxwellian_moderate, psc_mirror_bimaxwellian_weak, psc_mirror_bikappa3, psc_mirror_bikappa5
```

## Firehose

Physical condition:

```text
A_i < 1
beta_i_parallel * (1 - A_i) > 2
```

Expected signatures:

- growth of transverse fluctuations;
- reduction of the parallel pressure excess;
- `A_i` increases towards 1 if `T_perp/T_parallel` is used;
- the inverse `T_parallel/T_perp` decreases towards 1.

Cases:

```text
psc_firehose_bimaxwellian_strong, psc_firehose_bimaxwellian_moderate, psc_firehose_bimaxwellian_weak, psc_firehose_bikappa3, psc_firehose_bikappa5
```

## Whistler

Practical condition:

```text
A_e > 1 + 0.21 / beta_e_parallel^0.6
```

Expected signatures:

- growth at electron scales;
- electron anisotropy decreasing towards the threshold;
- spectrum dominated by modes compatible with parallel or oblique propagation.

Cases:

```text
psc_whistler_bimaxwellian_strong, psc_whistler_bimaxwellian_moderate, psc_whistler_bimaxwellian_weak
```

## Minimum diagnostics

A run must not be judged from a single figure. The analysis package must produce
at least:

```text
anisotropy_evolution.csv
evolucion_anisotropia.png
brazil_trayectoria.png
dominant_mode_evolution_xy.png
spectrum_2d_final_*.png
particle_anisotropy_evolution.csv
```

To compare cases, always keep:

- the same definition of `A`;
- the same driving species;
- the same time interval normalized to `Omega_ci`;
- the same snapshot selection criterion.
