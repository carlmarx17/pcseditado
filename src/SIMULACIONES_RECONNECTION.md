# PSC Simulation Documentation — Magnetic Reconnection

## Overview

PIC simulations of **collisionless magnetic reconnection** with a **double Harris
current sheet**. The reconnection plane is YZ, with X as the out-of-plane
direction. All of them use `PscConfig1vbecSingle` in 2D (dim_yz), full PIC,
**fully periodic** boundary conditions.

Base reference for plasmoid and fluctuation ideas: Agudelo Rueda et al.,
ApJ 971, 109 (2024), doi:10.3847/1538-4357/ad5e73.

---

## Catalog of reconnection codes

| File | Distribution | Grid | ppc | nmax | MPI ranks | RAM | Use |
|---|---|---|---|---|---|---|---|
| `psc_reconnection` | Kappa (κ=3) | 256×512 | 100 | 10M | 8 | ~30 GB | Production |
| `psc_reconnection_local` | Maxwellian | 64×256 | 10 | 5,000 | 1 | ~1 GB | Local PC |
| `psc_reconnection_mini` | Maxwellian | 32×128 | 4 | 200 | 1 | ~50 MB | Quick test |
| `psc_reconnection_comparable` | Kappa (κ=3) | 1152×1152 | 1000 | 250k | 2304 | ~40 GB | Comparison with anisotropy |

> The first three use mᵢ/mₑ = 25, ωpe/ωce = 2, Ti/Te = 5, Ly = 25.6 dᵢ, Lz = 51.2 dᵢ.
> `psc_reconnection_comparable`: mᵢ/mₑ = 200, box 40×40 dᵢ, 28.8 cells/dᵢ,
> matched to the bigbox40 anisotropy cases (the only difference: ωpe/ωce = 2;
> see `ESCALADO_INESTABILIDADES.md`, appendix).

---

## What is magnetic reconnection?

Magnetic reconnection is the process by which antiparallel magnetic field
lines "break" and reconnect, converting magnetic energy into kinetic and
thermal energy of the plasma. It is fundamental in:
- Solar flares
- Geomagnetic storms
- Coronal heating
- Turbulence dissipation in the solar wind

### Harris current sheet

The Harris equilibrium is an analytical solution where a current sheet
separates regions with antiparallel magnetic field:

```
B_z(y) = B₀ · tanh(y / L)          ← magnetic field
n(y)   = n₀ · sech²(y / L) + n_bg  ← plasma density
J_x(y) = (B₀/μ₀L) · sech²(y / L)  ← out-of-plane current
```

The current is sustained by particles drifting in the x direction
(out-of-plane): ions in +x, electrons in −x.

---

## Why a double current sheet?

Jefferson noted: *"if you want it to be periodic you need two current sheets."*

With a **single sheet** + periodic BCs in Y, the magnetic field has a
discontinuity at the boundaries (it goes from +B₀ at one edge to +B₀ at the
other, with no reversal). With **two antiparallel sheets**, the field closes
continuously across the periodic boundaries:

```
      ┌────────────────────────────────────────────┐
      │  B→    sheet 1    B←    sheet 2    B→      │
      │ +B₀  ──── 0 ──── −B₀  ──── 0 ──── +B₀     │
      │      y=−Ly/4            y=+Ly/4            │
      └──────────────── periodic ──────────────────┘
```

The double-sheet field is:

```
B_z(y) = B₀ · [tanh((y + Ly/4)/L) − tanh((y − Ly/4)/L) − 1]
```

And the density:

```
n(y) = n₀ · [sech²((y + Ly/4)/L) + sech²((y − Ly/4)/L)] + n_bg
```

---

## Total pressure balance

Jefferson emphasized: *"make sure P_total = Pᵢ + Pₑ + P_m = constant."*

On a vertical cut (y direction), the total pressure must be constant
**without including the perturbation**:

```
P_total(y) = n(y)·(Tᵢ + Tₑ) + B(y)²/(2μ₀) = constant
```

### Numerical verification

The code prints `P_total(y)` at 6 points along the vertical cut at start-up:

| Position | n_Harris | \|B\| | P_plasma | P_mag | P_total |
|---|---|---|---|---|---|
| y = −Ly/4 (center of sheet 1) | ~2.0 | ~0 | high | ~0 | P₀ |
| y = 0 (between sheets) | ~0 | ~B₀ | low | high | P₀ |
| y = +Ly/4 (center of sheet 2) | ~2.0 | ~0 | high | ~0 | P₀ |

The error must be < 1% at every point. If not, there is a bug.

### How is the balance achieved?

The Harris relation `n₀(Tᵢ + Tₑ) = B₀²/(2μ₀)` is automatically satisfied
by the temperature formula:

```cpp
TTe = me·c² / (2·ε₀·(ωpe/ωce)²·(1 + Ti/Te))
TTi = TTe · Ti/Te
```

The background temperatures equal the Harris temperatures (`Tib_Ti = 1, Teb_Te = 1`)
so that the background pressure `n_bg·(Tᵢ + Tₑ)` is an additive constant
that does not break the equilibrium.

---

## Initial perturbation

Jefferson noted: *"a perturbation is usually placed in the center so
reconnection starts. The pressure balance is computed without the
perturbation applied."*

The perturbation is applied **only to the sheet at y = +Ly/4** using a
vector potential that guarantees ∇·B = 0 by construction:

```
δA_x = ε · cos(k_z·(z − Lz/2)) / cosh((y − Ly/4) / σ)

δB_y = ∂(δA_x)/∂z   → perturbation in B_y
δB_z = −∂(δA_x)/∂y  → perturbation in B_z  (∇·B = 0 guaranteed)
```

With `σ = L` (= sheet thickness) and `ε = 0.03·B₀·σ` (3% of B₀).
The `1/cosh` envelope makes the perturbation decay exponentially away from
the perturbed sheet, leaving the other sheet unperturbed.

---

## Ideas taken from Agudelo Rueda et al. 2024

The paper studies how induced magnetic fluctuations, similar to turbulence,
modify a Harris current sheet and plasmoid formation. The central idea for
my simulation is this: it is not enough to start reconnection; I must
distinguish whether magnetic islands grow via normal tearing or whether the
system ends up dominated by small fluctuations that break the sheet without
forming coherent plasmoids.

### Differences with our current case

| Point | 2024 paper | Our `psc_reconnection` |
|---|---|---|
| Plasma | Electron-positron pair, `m_i/m_e = 1` | Artificial ion-electron, `m_i/m_e = 25` |
| Domain | Single Harris sheet with conducting boundaries in `y` | Double Harris sheet for periodic boundaries |
| Resolution | `dy = dz = 0.11 d_e` | `dy = dz = 0.50 d_e` in production |
| ppc | 400 | 100 |
| Sheet thickness | `Delta = 4 d_e` | `L = 0.5 d_i = 1.25 d_e` |
| Forcing | Langevin antenna and `k, omega` modes | Localized seed-type perturbation on one sheet |
| Scientific question | When tearing/plasmoids are suppressed | First validate reconnection and plasmoids; then test fluctuations |

For this reason I should not copy the parameters literally. The paper serves
as physical and diagnostic guidance, but my case has a different mass ratio,
a different sheet thickness and a different boundary condition.

### What I can use directly as a physical criterion

1. **Control case:** a run without strong turbulent fluctuations must form
   X-points, O-points, elongated current sheets and plasmoids.
2. **Large, long-scale perturbation:** a central pinch-type perturbation
   can accelerate the arrival at the reconnection state, but does not
   necessarily represent turbulence.
3. **Small-scale, large fluctuations:** the paper's most important result
   is that fluctuations with a wavelength comparable to or smaller than the
   sheet thickness (`lambda <= Delta`) and amplitude above a critical
   threshold can suppress the growth of magnetic islands.
4. **Scale-dependent threshold:** the critical amplitude `delta B_c` is not
   universal; it depends on the wavelength. Smaller scales can more directly
   affect particle orbits.
5. **Energy and velocity distribution:** when plasmoids are suppressed, the
   injected energy tends to heat the plasma and modify the velocity
   distribution instead of feeding large magnetic islands.

### Translation into our code

The current code implements a localized seed:

```
delta A_x = epsilon * cos(k_z*(z - Lz/2)) / cosh((y - Ly/4)/sigma)
```

This is useful for starting reconnection in a specific sheet. It is not yet
a Langevin antenna. A future extension inspired by the paper would be to add
out-of-plane vector potential modes:

```
delta A_x(y,z,t) = Re[ sum_j b_j(t) / k_j * exp(i k_j · r) ]
delta B_ext = curl(delta A_x xhat)
delta J_ext = curl(delta B_ext) / mu0
```

and update `b_j(t)` with a random phase/force, an excitation frequency
`omega_0` and a decorrelation rate `gamma_0`. That external current should
be added to the field advance, not only to the initial condition.

### Recommended sweep for the thesis

To avoid spending resources before validating the base case, the reasonable
order is:

1. **Control:** the current `psc_reconnection` with 3% perturbation, no
   turbulent forcing. Confirm pressure, X-point, outflows and islands.
2. **Weak seed:** lower `dby_b0` to see whether tearing appears
   spontaneously or depends too heavily on the seed.
3. **Central pinch:** try a long-scale perturbation like the paper's to
   accelerate the steady state without introducing kinetic-scale noise.
4. **Multi-mode fluctuations:** implement an external antenna afterward
   with large modes and small amplitude. This case should resemble the
   control if the paper's result applies.
5. **Suppression test:** use modes with `lambda <= L` and larger amplitude.
   Check whether the growth of `J_x,max` decreases and whether O-points stop
   forming.

For our production case `L = 0.5 d_i = 1.25 d_e`. So a suppression test
inspired by the paper should use wavelengths on the order of
`lambda <= 1.25 d_e`, not just large modes the size of the domain.

---

## Out-of-plane velocity

Jefferson noted: *"in the direction out of the plane, a velocity must be
given to the particles."*

This is the **drift velocity** that sustains the J_x current of each Harris
sheet. In the double sheet, the drifts reverse between the two sheets:

```
Sheet 1 (y = −Ly/4):  J_x > 0  →  ions drift in +x, electrons in −x
Sheet 2 (y = +Ly/4):  J_x < 0  →  ions drift in −x, electrons in +x
```

Implementation: the drift is weighted by the local density contribution:

```cpp
drift_weight = (n_sheet1 − n_sheet2) / (n_sheet1 + n_sheet2)
// +1 near sheet 1, −1 near sheet 2, 0 between sheets

v_drift_ion = +2·Tᵢ/(B₀·L) × drift_weight
v_drift_ele = −2·Tₑ/(B₀·L) × drift_weight
```

---

## Physical parameters

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Mass ratio | mᵢ/mₑ | 25 | Artificial (real ≈ 1836) |
| Frequency ratio | ωpe/ωce | 2.0 | |
| Temperature ratio | Tᵢ/Tₑ | 5.0 | |
| Sheet thickness | L/dᵢ | 0.5 | Half-thickness |
| Domain Y | Ly/dᵢ | 25.6 | Separation between sheets: 12.8 dᵢ |
| Domain Z | Lz/dᵢ | 51.2 | Space for outflows |
| Background density | nb/n₀ | 0.20 | 20% of the Harris peak |
| Guide field | bg | 0.0 | Anti-parallel (no guide) |
| Perturbation | δBy/B₀ | 0.03 | 3% on only one sheet |

### Derived quantities (in code units)

```
dᵢ = √(mᵢ/mₑ) / (ωpe/ωce) = √25 / 2 = 2.5 dₑ
B₀ = ωce·mₑ/e = 0.5  (in PSC units)
Tₑ = 1 / (2·(ωpe/ωce)²·(1+Ti/Te)) = 1/(2·4·6) = 0.02083
Tᵢ = 5·Tₑ = 0.1042
λ_De = √Tₑ ≈ 0.144
```

---

## Resolution and scales

| Code | Grid | dy/dₑ | dz/dₑ | dy/dᵢ | ppc | Total particles |
|---|---|---|---|---|---|---|
| `psc_reconnection` | 256×512 | 0.50 | 0.50 | 0.20 | 100 | ~26M |
| `psc_reconnection_local` | 64×256 | 2.0 | 1.0 | 0.80 | 10 | ~330K |
| `psc_reconnection_mini` | 32×128 | 4.0 | 1.0 | 1.60 | 4 | ~33K |

> **Note from Jefferson**: if the resolution or particle count is too low,
> the tearing instability appears too fast (numerically, not physically).
> `psc_reconnection` (256×512, 100 ppc) is designed to avoid this.

### Resolution comparison with the literature and anisotropy simulations

In Full PIC simulations, adequately resolving the inertial scales (dₑ, dᵢ)
and the Debye length (λ_De) is a computational challenge. A technical
comparison follows:

| Simulation / Study | mᵢ/mₑ | dx/dₑ | dx/dᵢ | dx/λ_De | ppc | Distribution |
|---|---|---|---|---|---|---|
| **Reconnection (Production)** | 25 | 0.25 | 0.10 | 1.73 | 100 | Kappa / Max |
| **Reconnection (Local)** | 25 | 1.00 | 0.40 | 6.94 | 10 | Maxwellian |
| **Anisotropies (Mirror/Firehose)** | 100-200 | 0.20 | 0.014-0.02 | 1.53 | 1000-2000 | Kappa / Max |
| *Daughton et al. 2011 (Reconnection)* | 25-100 | 0.1-0.2 | ~0.02-0.04 | ~1.0 | ~100 | Maxwellian |
| *Agudelo Rueda et al. 2024* | Pair (1) | 0.1-0.5 | 0.1-0.5 | ~1.0 | 500-1000 | Maxwellian |

**Scientific analysis:**
1. **Skin-depth resolution:** in our production reconnection case
   (`psc_reconnection`), `dx = 0.25 dₑ`, which is fine enough to capture
   the electron physics in the diffusion layer, fully consistent with the
   literature standard (e.g., Daughton et al., who typically use
   `dx ~ 0.1-0.2 dₑ`).
2. **Debye under-resolution:** for our reconnection plasma, λ_De ≈ 0.144 dₑ,
   which implies `dx/λ_De ≈ 1.73`. As in our anisotropy simulations (where
   `dx/λ_De ≈ 1.53`), the Debye length is slightly under-resolved. This
   trade-off is common and accepted in PIC simulations with artificial mass,
   in order to cover a physical domain (Lz = 51.2 dᵢ) large enough to allow
   the reconnection outflows to develop. The resulting grid noise (aliasing)
   is mitigated by using higher-order particle shapes and a reasonable
   number of particles per cell (100 ppc).
3. **Ion scale:** the scale separation `dᵢ = 5 dₑ` in reconnection (due to
   mᵢ/mₑ=25) yields `dx = 0.1 dᵢ`. In anisotropy runs (which use larger
   domains and larger mass ratios, mᵢ/mₑ=100-200), `dx ≈ 0.02 dᵢ` is
   reached. Both approaches guarantee excellent resolution of the ion
   physics.

---

## Cluster resources and best options

The memory consumption of these reconnection simulations is substantially
lower than that of the anisotropy simulations, since the 2D domain is much
smaller.

| Executable | Recommended environment | Required CPUs and RAM | Steps |
|---|---|---|---|
| `psc_reconnection` | **feynman-00**, **pauli**, or **planck** | 8 CPUs, ~30 GB RAM | 10M |
| `psc_reconnection_local` | Local PC or test node (`maxwell`) | 1 CPU, ~1 GB RAM | 5,000 |
| `psc_reconnection_mini` | Local PC for debugging | 1 CPU, <100 MB RAM | 200 |

**Execution recommendation:**
To observe the *steady-state* and plasmoid formation, run `psc_reconnection`
on the **feynman-00** node. Since it requires only 8 ranks and 30 GB of
memory, it can run easily even if the node is partially occupied, unlike the
anisotropy simulations (Mirror/Firehose) which demand blocks of 64-128 CPUs
and >150 GB of RAM.

---

## Boundary conditions

```
Fields:     {PERIODIC, PERIODIC, PERIODIC}  in {x, y, z}
Particles:  {PERIODIC, PERIODIC, PERIODIC}  in {x, y, z}
```

Fully periodic. The double Harris sheet allows this without discontinuities.

---

## What to observe (steady state)

Jefferson noted: *"it is important for reconnection to reach steady state,
where you have an elongated current sheet and the outflows are horizontal."*

When analyzing the results, look for:

1. **Elongated sheet**: the perturbed sheet (y = +Ly/4) thins at the center
2. **X-point**: point where B_z ≈ 0 and B_y ≈ 0 (the reconnection point)
3. **Horizontal outflows**: plasma jets exiting in ±z from the X-point
4. **Vertical inflows**: plasma entering from ±y toward the X-point
5. **Magnetic islands**: possible formation of plasmoids/islands
6. **Stable sheet 2**: the unperturbed sheet (y = −Ly/4) must remain stable

### Diagnostics inspired by the paper

To connect the analysis with Agudelo Rueda et al. 2024, it is useful to
measure:

1. **`B_y(z,t)` at the center of the perturbed sheet:** build a stack plot.
   Alternating blue-red patterns indicate X-points and island growth.
2. **Normalized `max(|J_x|)`:** if plasmoids grow, a strong growth phase
   should appear; if tearing is suppressed, `J_x,max` should not grow the
   same way.
3. **`max(|B_y|)` and rate `gamma_By`:** use it as a proxy for the tearing
   mode, taking care to separate real island growth from a direct response
   to the forcing.
4. **Electromagnetic, kinetic and thermal energy:** in forced runs energy
   need not be conserved; what matters is whether the energy feeds islands
   or heats the plasma.
5. **O-point/X-point count:** to distinguish reconnection with plasmoids
   from a fragmented sheet without coherent islands.
6. **Velocity distributions near the X-point and outflow:** if the forcing
   suppresses islands, look for thermal increase and anisotropy instead of
   coherent flow.

---

## Execution

### Quick test
```bash
mpirun -n 1 ../build/src/psc_reconnection_mini
# Should print "Pressure Balance" with err < 1% and finish with "Test completed!"
```

### Local development
```bash
mpirun -n 1 ../build/src/psc_reconnection_local
# Generates pfd.*.h5 and pfd_moments.*.h5 every 50 steps
```

### Production (server, 8 MPI ranks, ~30 GB):
```bash
mpirun -n 8 ./psc_reconnection
# 256×512, 100 ppc, Kappa κ=3, 10M steps
```

---

## SLURM script for production

```bash
#!/bin/bash -l
#SBATCH --job-name=psc_reconnection
#SBATCH --clusters=cecc
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=feynman-00
#SBATCH --ntasks=8
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --chdir=/homes/observatorio/cmartinezsi/pcs_run
#SBATCH --output=/homes/observatorio/cmartinezsi/pcs_run/psc_reconnection_%j.out
#SBATCH --error=/homes/observatorio/cmartinezsi/pcs_run/psc_reconnection_%j.err

module purge
module load MPI/openmpi/4.1.1
module load lang/gcc/9.2

BASEDIR=/homes/observatorio/cmartinezsi/pcs_run
WORKDIR=${BASEDIR}/reconnection_kappa
EXECUTABLE=${BASEDIR}/psc_reconnection

mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "=========================================================="
echo "  PSC Reconnection — Double Harris + Kappa κ=3"
echo "=========================================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURM_JOB_NODELIST}"
echo "MPI tasks   : ${SLURM_NTASKS}"
echo "Executable  : ${EXECUTABLE}"
echo "Directory   : ${WORKDIR}"
echo "Start       : $(date)"
echo "=========================================================="

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Executable not found: ${EXECUTABLE}"
    exit 1
fi

mpirun -np 8 \
       --bind-to core \
       ${EXECUTABLE}

EXIT_CODE=$?
echo "=========================================================="
echo "End         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
```

---

## References

- Agudelo Rueda, Liu, Germaschewski, Hesse & Bessho 2024 (ApJ 971, 109): "On the Effect of Inducing Turbulence-like Fluctuations in a Harris Current Sheet Configuration and Plasmoid Formation"
- Harris 1962: Original Harris current sheet equilibrium
- Birn et al. 2001 (JGR 106): GEM reconnection challenge (double Harris setup standard)
- Daughton et al. 2011 (Nature Phys 7): PIC reconnection with plasmoids
