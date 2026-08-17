# PSC anisotropy simulations

Catalogue of build-ready cases and their physical parameters.
The ADIOS2 execution guide on COSMA is in `ADIOS2_COSMA_RUNBOOK.md`.

## Code structure

All cases share `psc_anisotropy_case.hxx`. Each case file only
defines the label, the distribution, `PSC_KAPPA` when applicable, and the
physical parameters of the regime:

```text
psc_mirror_bikappa3.cxx
psc_mirror_bikappa5.cxx
psc_firehose_bikappa3.cxx
psc_firehose_bikappa5.cxx
```

Bi-Kappa cases enable `PSC_USE_KAPPA=1` and specify `PSC_KAPPA`.
Bi-Maxwellian cases use the default value `PSC_USE_KAPPA=0`.

```
PSC_BETA_E_PAR / PSC_BETA_I_PAR
PSC_TI_PERP_OVER_TI_PAR / PSC_TE_PERP_OVER_TE_PAR
PSC_KAPPA
```

The common defaults are centralized so that all executables use the
same grid, outputs and intervals, unless overridden via the environment.

## Common configuration

| Parameter | Value |
|---|---:|
| PSC configuration | `PscConfig1vbecSingle<dim_yz>` |
| Background field | `B0 = 0.08` |
| `vA/c` | 0.08 |
| `mi/me` | 200 |
| `lambda0` | 20 |
| Initial density | 1.0 |
| Domain | `20 d_i × 20 d_i` |
| Grid | `576×576` |
| Resolution | `28.8 cells/d_i` (`dx = dz = 0.034722 d_i` ≈ `0.491 d_e`, `dx/λ_De ≈ 3.78`) |
| Particles per cell | 1000 (default) |
| Maximum steps | Depend on saturation (`PSC_NMAX`, default 1,200,000) |
| Boundaries | Periodic |
| Fields/moments | every 500 steps |
| Particles | every 10,000 steps |
| ADIOS2 checkpoint | every 5000 steps |
| Continuity check | every 5000 steps |
| Energy diagnostic | every 5000 steps (`diag.asc`) |
| Load balancing | every 2500 steps |

The parallel field is `z`:
```
T_parallel = T_z
T_perp = (T_x + T_y) / 2
A = T_perp / T_parallel
```

## Mirror

Ions with excess perpendicular temperature. Isotropic electrons.
Criterion: `beta_i_parallel * (A_i - 1) > 1`

| Executable | File | Regime | beta_i_par | A_i | beta_e_par | A_e | Grid |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_mirror_bimaxwellian_strong` | `psc_mirror_bimaxwellian_strong.cxx` | Strong | 5.0 | 3.0 | 1.0 | 1.0 | 576×576 |
| `psc_mirror_bimaxwellian_moderate` | `psc_mirror_bimaxwellian_moderate.cxx` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 | 576×576 |
| `psc_mirror_bimaxwellian_weak` | `psc_mirror_bimaxwellian_weak.cxx` | Weak | 6.0 | 1.5 | 1.0 | 1.0 | 576×576 |

## Firehose

Ions with excess parallel temperature. Isotropic electrons.
Criterion: `beta_i_parallel * (1 - A_i) > 2`

| Executable | File | Regime | beta_i_par | A_i | beta_e_par | A_e | Grid |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_firehose_bimaxwellian_strong` | `psc_firehose_bimaxwellian_strong.cxx` | Strong | 10.0 | 0.1 | 1.0 | 1.0 | 576×576 |
| `psc_firehose_bimaxwellian_moderate` | `psc_firehose_bimaxwellian_moderate.cxx` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 | 576×576 |
| `psc_firehose_bimaxwellian_weak` | `psc_firehose_bimaxwellian_weak.cxx` | Weak | 3.0 | 0.6 | 1.0 | 1.0 | 576×576 |

## Whistler

Electrons with excess perpendicular temperature. Isotropic ions.
Criterion: `A_e > 1 + 0.21 / beta_e_parallel^0.6`

| Executable | File | Regime | beta_i_par | A_i | beta_e_par | A_e | Grid |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_whistler_bimaxwellian_strong` | `psc_whistler_bimaxwellian_strong.cxx` | Strong | 1.0 | 1.0 | 0.5 | 3.0 | 576×576 |
| `psc_whistler_bimaxwellian_moderate` | `psc_whistler_bimaxwellian_moderate.cxx` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 | 576×576 |
| `psc_whistler_bimaxwellian_weak` | `psc_whistler_bimaxwellian_weak.cxx` | Weak | 1.0 | 1.0 | 0.5 | 1.5 | 576×576 |

## Bi-Kappa

| Executable | File | κ | beta_i_par | A_i | beta_e_par | A_e | Grid |
|---|---|---|---:|---:|---:|---:|---:|
| `psc_mirror_bikappa3` | `psc_mirror_bikappa3.cxx` | 3 | 5.0 | 3.0 | 1.0 | 1.0 | 576×576 |
| `psc_mirror_bikappa5` | `psc_mirror_bikappa5.cxx` | 5 | 5.0 | 3.0 | 1.0 | 1.0 | 576×576 |
| `psc_firehose_bikappa3` | `psc_firehose_bikappa3.cxx` | 3 | 10.0 | 0.1 | 1.0 | 1.0 | 576×576 |
| `psc_firehose_bikappa5` | `psc_firehose_bikappa5.cxx` | 5 | 10.0 | 0.1 | 1.0 | 1.0 | 576×576 |

## Outputs

The cases write fields, moments and particles:
```
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_<basename>.<step>.h5        # central region ~20% of each direction
checkpoint_<step>.bp/            # ADIOS2 only
```

## Local build

```bash
cmake --build build --target psc_mirror_bimaxwellian_strong
cmake --build build --target psc_mirror_bikappa3
```

All targets:
```bash
cmake --build build --target \
  psc_mirror_bimaxwellian_strong psc_mirror_bimaxwellian_moderate psc_mirror_bimaxwellian_weak \
  psc_firehose_bimaxwellian_strong psc_firehose_bimaxwellian_moderate psc_firehose_bimaxwellian_weak \
  psc_whistler_bimaxwellian_strong psc_whistler_bimaxwellian_moderate psc_whistler_bimaxwellian_weak \
  psc_mirror_bikappa3 psc_mirror_bikappa5 \
  psc_firehose_bikappa3 psc_firehose_bikappa5
```

## Build and run with ADIOS2 on COSMA

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
sbatch src/submit_anisotropy_adios2.slurm
```

For another target:
```bash
sbatch --export=ALL,PSC_TARGET=psc_firehose_bikappa3 src/submit_anisotropy_adios2.slurm
```

The simulation's editable parameters can be changed without recompiling:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_bikappa3,PSC_NMAX=1200000,PSC_BALANCE_INTERVAL=2500,PSC_CONTINUITY_EVERY=5000,PSC_ENERGIES_EVERY=5000 \
  src/submit_anisotropy_adios2.slurm
```

`PSC_NMAX` is not used to standardize the physics: it is the maximum step
limit, adjusted per case according to when saturation is observed.

Available overrides: `PSC_NMAX`, `PSC_NGRID`, `PSC_NP_Y`, `PSC_NP_Z`,
`PSC_NICELL`, `PSC_CHECKPOINT_EVERY`, `PSC_FIELDS_EVERY`,
`PSC_PARTICLES_EVERY`, `PSC_BALANCE_INTERVAL`, `PSC_CONTINUITY_EVERY`,
`PSC_ENERGIES_EVERY` and `PSC_RESTART`.

The scripts clean up Conda and use `srun` by default to avoid `prted`
failures when starting OpenMPI from Slurm.

Operational details are in `ADIOS2_COSMA_RUNBOOK.md`.
