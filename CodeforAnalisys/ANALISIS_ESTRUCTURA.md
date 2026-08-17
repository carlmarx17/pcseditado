# Structure of the PSC analysis ecosystem

> Technical documentation of the post-processing pipeline for the standard
> bi-Maxwellian runs `F_*_bM`, `M_*_bM`, `W_*_bM` and the legacy
> Maxwellian/Kappa cases.

For day-to-day commands see `CodeforAnalisys/README.md`. This document describes
the file contract, the datasets and the internal responsibilities.

## 0. Contract of a run

Each data directory must contain a single simulation. For `CASE=F_M_bM`, an HDF5
series is accepted:

```text
pfd.<step>_p000000.h5
pfd_moments.<step>_p000000.h5
prt_F_M_bM.<step>.h5
```

or the equivalent ADIOS2 series:

```text
pfd.<step>.bp/
pfd_moments.<step>.bp/
prt_F_M_bM.<step>.bp/
```

The production command is:

```bash
cd CodeforAnalisys && make analysis DATA_DIR=../path/F_M_bM CASE=F_M_bM
```

`CASE` selects the physical parameters, driving species, time normalization,
particle name and output folder. The manifest
`analysis_results/F_M_bM/F_M_bM_analysis_manifest.json` records these decisions
and the detected steps.

For Firehose both conventions are reported:

```text
A_i = T_i_perp / T_i_parallel       # increases towards 1
R_i = T_i_parallel / T_i_perp=1/A_i # decreases towards 1
```

Saying only that "the anisotropy should go down" is ambiguous without stating
which of these two ratios is meant.

---

## 1. Output file formats

The pipeline accepts HDF5 (`.h5`) and ADIOS2 BP (`.bp`) snapshots:

| File pattern | Contents | Read by |
|--------------------------|------------------------------------------------|--------------------------------------|
| `prt_<CASE>.<step>.h5` or `.bp/` | Particle data (q, m, px, py, pz, w) | `physical_diagnostics.py`, particle scripts |
| `pfd.<step>_pN.h5` or `pfd.<step>.bp/` | EM fields on the grid | `physical_diagnostics.py`, field scripts |
| `pfd_moments.<step>_pN.h5` or `.bp/` | Particle moments | `physical_diagnostics.py`, moment scripts |

> **Important:** `checkpoint_<step>.bp/` is a restart checkpoint and does not
> automatically replace `pfd`, `pfd_moments` or `prt_*`. Field, moment and
> particle analysis requires those series, in HDF5 or BP. A simulation may write
> ADIOS2 checkpoints and keep its regular diagnostics in HDF5 if the executable
> uses `WriterDefault`.

---

## 2. Internal structure of the HDF5 files

### 2.1 Particle files — `prt.*.h5`

```
prt.000001200.h5
└── particles/
    └── p0/
        └── 1d/          ← dataset holding a structured array
            ├── q[N]     ← charge: +Zi (ions) / -1 (electrons)
            ├── m[N]     ← mass: 200.0 (ions) / 1.0 (electrons)
            ├── w[N]     ← statistical weight (= 1.0 with fractional_n=true)
            ├── px[N]    ← x momentum  [m * v_x in PSC units]
            ├── py[N]    ← y momentum
            └── pz[N]    ← z momentum  (parallel direction = z ∥ B₀)
```

**How it is read in Python:**

```python
import h5py
import numpy as np

with h5py.File("prt.000001200.h5", "r") as f:
    dset = f["particles"]["p0"]["1d"]

    q  = dset["q"][:]   # +1 ions, -1 electrons
    m  = dset["m"][:]   # 200.0 or 1.0
    px = dset["px"][:]  # perpendicular x momentum
    py = dset["py"][:]  # perpendicular y momentum
    pz = dset["pz"][:]  # parallel momentum

# Separate species
ions  = np.where(q > 0)
elecs = np.where(q < 0)

# PSC stores u = gamma*v; in these non-relativistic runs u ~= v.
T_par_ions = 200.0 * np.var(pz[ions])
```

### 2.2 Field files — `pfd.*.h5`

```
pfd.001200_p0.h5
└── jeh-<UID>/              ← group with a dynamic prefix (run UID)
    ├── hx_fc/p0/3d[Nx,Ny,Nz]   ← Bx, face-centered
    ├── hy_fc/p0/3d[Nx,Ny,Nz]   ← By
    ├── hz_fc/p0/3d[Nx,Ny,Nz]   ← Bz  (∥ to the background field B₀)
    ├── ex_ec/p0/3d              ← Ex, edge-centered
    ├── ey_ec/p0/3d              ← Ey
    └── ez_ec/p0/3d              ← Ez
```

**How it is read (with the `PICDataReader` helper):**

```python
from data_reader import PICDataReader

fields = PICDataReader.read_multiple_fields_3d(
    "pfd.001200_p0.h5",
    "jeh-",                         # group prefix (ignores the UID)
    ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
)

Bx = fields["hx_fc/p0/3d"]   # 3D array (Nx, Ny, Nz)
By = fields["hy_fc/p0/3d"]
Bz = fields["hz_fc/p0/3d"]
B2 = Bx**2 + By**2 + Bz**2
```

### 2.3 Moment files — `pfd_moments.*.h5`

```
pfd_moments.001200_p0.h5
└── all_1st-<UID>/
    ├── rho_i/p0/3d    ← ion density  n_i(y,z)
    ├── txx_i/p0/3d    ← component Pxx = n m <vx vx>
    ├── tyy_i/p0/3d    ← component Pyy
    ├── tzz_i/p0/3d    ← component Pzz  (parallel pressure)
    ├── jx_i/p0/3d     ← ion current x
    ├── rho_e/p0/3d    ← electron density
    └── ...
```

**Temperature from moments (no particles needed):**

```python
moments = PICDataReader.read_multiple_fields_3d(
    "pfd_moments.001200_p0.h5",
    "all_1st",
    ["txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d", "rho_i/p0/3d"],
)

n    = moments["rho_i/p0/3d"].ravel()
Pxx  = moments["txx_i/p0/3d"].ravel()
Pzz  = moments["tzz_i/p0/3d"].ravel()

T_par  = Pzz / n          # parallel temperature per cell
T_perp = 0.5 * Pxx / n    # (Pxx + Pyy average) / 2n
A_i    = T_perp / T_par   # anisotropy
```

---

## 2b. ADIOS2 format: `.bp` files

If PSC is built with `PSC_HAVE_ADIOS2` and `WriterAdios2` is selected in the
`.cxx`:

```cpp
// In the simulation .cxx:
using Writer = WriterADIOS2;   // instead of WriterDefault (HDF5/MRC)
```

then **all** output files (fields, moments, particles, checkpoints) change from
`.h5` to `.bp`.

### File names: `.h5` → `.bp`

| HDF5 (WriterDefault) | ADIOS2 (WriterADIOS2) |
|---------------------------------|------------------------------------|
| `pfd.000001200_p0.h5`           | `pfd.000001200.bp/`                |
| `pfd_moments.000001200_p0.h5`   | `pfd_moments.000001200.bp/`        |
| `prt.000001200.h5`              | `prt.000001200.bp/`                |
| —                               | `checkpoint_5000.bp/`              |

> **Note:** a `.bp` is not a single file but a **directory** containing `md.idx`,
> `md.0`, `data.0`, etc. For the user it behaves as a single file.

### Internal structure: what changes?

The logical hierarchy of the data **is the same** as with HDF5. What changes is
the container and the reading API:

```
pfd.000001200.bp/
├── step         (int)       ← simulation step
├── time         (double)    ← time in code units
├── length       (Real3)     ← domain extent [Lx, Ly, Lz]
├── corner       (Real3)     ← lower corner of the domain
├── ib           (Int3)      ← ghost boundary offset
├── im           (Int3)      ← dimensions including ghosts
└── jeh-<UID>/
    ├── hx_fc/p0/3d[Nx,Ny,Nz]   ← Bx (face-centered) — same path as HDF5
    ├── hy_fc/p0/3d[Nx,Ny,Nz]   ← By
    ├── hz_fc/p0/3d[Nx,Ny,Nz]   ← Bz
    ├── ex_ec/p0/3d              ← Ex (edge-centered)
    └── ...
```

The particle datasets (`prt.*.bp`) have the same structure
`particles/p0/1d/{q, m, px, py, pz, w}`.

### Extra metadata in `.bp`

ADIOS2 automatically adds:
- `step` and `time` as scalar variables in every file.
- Domain `length` and `corner` (which did not exist in the native `.h5`).
- `ib` / `im` — ghost boundary offsets and dimensions (useful to reconstruct the
  global domain from MPI patches).

This metadata is written by `WriterADIOS2::begin_step()`:
```cpp
file_.put("step", step);
file_.put("time", time);
file_.put("length", grid.domain.length);
file_.put("corner", grid.domain.corner);
```

### How to read `.bp` in Python

```python
import adios2
import numpy as np

# Open a field .bp file
with adios2.open("pfd.000001200.bp", "r") as f:
    for step in f:
        # Read metadata
        sim_step = step.read("step")
        sim_time = step.read("time")

        # Read fields — same logical path as in HDF5
        Bz = step.read("jeh/hz_fc/p0/3d")
        Bx = step.read("jeh/hx_fc/p0/3d")
        By = step.read("jeh/hy_fc/p0/3d")

# Open a particle .bp file
with adios2.open("prt.000001200.bp", "r") as f:
    for step in f:
        q  = step.read("particles/p0/1d/q")
        m  = step.read("particles/p0/1d/m")
        px = step.read("particles/p0/1d/px")
        py = step.read("particles/p0/1d/py")
        pz = step.read("particles/p0/1d/pz")
```

> **Key difference from `h5py`:** in ADIOS2 the API is `step.read("path")`
> instead of `f["path"][:]`. Also, group navigation uses a flat `/` rather than
> the HDF5 object hierarchy (`f["particles"]["p0"]["1d"]`).

### Difference in the group UID prefix

| HDF5 | ADIOS2 |
|-------------------------------|-------------------------------|
| `jeh-abc123/hx_fc/p0/3d`     | `jeh/hx_fc/p0/3d`            |
| `all_1st-xyz789/txx_i/p0/3d` | `all_1st/txx_i/p0/3d`        |

In HDF5, PSC appends a UUID hash to the group name (`jeh-<uid>`) to avoid MPI
collisions. In ADIOS2 this hash is **not** added — the prefix is clean (`jeh/`,
`all_1st/`). This means `PICDataReader.get_uid_group()` is not needed with `.bp`.

### Implemented `.bp` support

| Component | Status |
|---|---|
| `data_reader.py` | Unified HDF5/BP reader, UID group resolution and compatibility with `FileReader`, `Stream` and `adios2.open`. |
| `physical_diagnostics.py` | Particles, fields and moments are read through `PICDataReader`; there is no parallel HDF5 path. |
| Discovery | Automatically looks for `pfd`, `pfd_moments` and `prt_*` in both formats. |
| Spectra | The master diagnostic computes $E_{B_\perp}(k)$ directly. |
| Checkpoints | Reserved for restart; not interpreted as physical snapshots. |

### Recommended dual strategy

To support both formats without duplicating code:

```python
import os

def open_data_file(filepath):
    """Return a reader according to the file extension."""
    if filepath.endswith(".bp") or os.path.isdir(filepath):
        import adios2
        return adios2.open(filepath, "r")
    else:
        import h5py
        return h5py.File(filepath, "r")
```

Alternatively, use the environment variable:
```bash
export PSC_IO_BACKEND=adios2   # or "hdf5" (default)
```

---

## 3. Script tree and responsibilities

```
CodeforAnalisys/
│
├── psc_units.py              ← CENTRAL MODULE: constants and unit conversions
│   │                            B0, OMEGA_CI, DI, TI_PAR, TI_PERP, KAPPA ...
│   └── (imported by every other script)
│
├── data_reader.py            ← UNIFIED HDF5/ADIOS2 READER
│   │                            PICDataReader: discovery, opening and path resolution
│   └── (imported by anisotropy_analysis, diamagnetic_current, mirror_physics)
│
├── plot_prt.py               ← PARTICLE ANALYSIS (reads prt.*.h5)
│   ├── Plot 1: 2D VDF        f(v_⊥, v_∥) — log heat map
│   ├── Plot 2: Kappa vs Max  theoretical distribution vs data
│   ├── Plot 3: KS + AD       goodness-of-fit tests
│   ├── Plot 4: VDF snapshots multi-time panels
│   ├── Plot 5: 1D evolution  f(v_∥,t) and f(v_⊥,t) as a heatmap
│   ├── Plot 6: Anisotropy    T_⊥/T_∥ vs time
│   ├── Plot 7: Brazil plot   T_⊥/T_∥ vs β_∥ with thresholds
│   ├── Plot 9: 1D VDF evol.  overlaid lines + suprathermal tail
│   ├── Plot 10: E partition  E_mag / E_kin / E_thermal vs time
│   └── Plot 11: Heat flux    q_∥ and q_⊥ in localized regions
│
├── anisotropy_analysis.py    ← ANISOTROPY ANALYSIS (reads pfd_moments + pfd)
│   └── Brazil plot from grid moments (full spatial resolution)
│
├── mirror_physics.py         ← MIRROR HOLES (reads pfd)
│   └── 2D maps of |B|, out-of-plane current, fluctuation contours
│
├── diamagnetic_current.py    ← DIAMAGNETIC CURRENT (reads pfd_moments + pfd)
│   └── Ion, electron and total J_dia maps
│
├── fluctuationofmagneticfiel.py  ← δB FLUCTUATIONS (reads pfd)
│   └── δB and δB/B₀ maps, GIF animations
│
├── spectral_analysis.py      ← SPECTRAL ENGINE (reads pfd HDF5/BP)
│   └── 1D and 2D PSD of the magnetic components
│
├── physical_diagnostics.py  ← MASTER HDF5/BP DIAGNOSTIC
│   ├── particles, temperatures, VDFs and fits
│   ├── moments, maps, currents and correlations
│   ├── fluctuations, growth and energy
│   └── transverse spectrum E_Bperp(k)
│
├── validate_moments.py       ← VALIDATION (reads prt.*.h5)
│   └── Checks that measured moments = initialization parameters
│
├── plot_vdf_3d.py            ← 3D VDF (reads prt.*.h5)
│   └── 3D surface f(vx, vy, vz)
│
├── plot_moments_scatter_3d.py ← 3D SCATTER (reads prt.*.h5)
│   └── Momentum scatter + 3D histograms
│
└── Makefile                  ← ORCHESTRATOR
    ├── make brazil     → anisotropy_analysis.py
    ├── make mirror     → mirror_physics.py
    ├── make diamagnetic → diamagnetic_current.py
    ├── make fields     → fluctuationofmagneticfiel.py
    ├── make spectral   → spectral_analysis.py
    ├── make validate   → validate_moments.py
    ├── make particles  → plot_prt.py + plot_vdf_3d.py + plot_moments_scatter_3d.py
    └── make all        → everything except spectral and report
```

---

## 4. Complete data flow

```
psc_mirror_kappa.cxx              psc_firehose_kappa.cxx
psc_mirror_maxwellian.cxx         psc_firehose_maxwellian.cxx
         │
         │  (PIC simulation)
         ▼
   ../build/src/
   ├── prt.000000000.h5     ← t=0
   ├── prt.000001200.h5     ← t=1200
   ├── ...
   ├── pfd.001200_p0.h5
   ├── pfd_moments.001200_p0.h5
   └── ...
         │
         │  (Makefile → Python scripts)
         │
   ┌─────┴──────────────────────────────────┐
   │                                        │
   ▼                                        ▼
prt.*.h5                             pfd.*.h5 + pfd_moments.*.h5
   │                                        │
   ├── plot_prt.py                          ├── anisotropy_analysis.py
   │     └── prt_plots/                     │     └── anisotropy_plots/
   │         ├── vdf_2d_ions.png            │         └── brazil_plot_anisotropy.png
   │         ├── vdf_2d_electrons.png       │
   │         ├── kappa_comparison_*.png     ├── mirror_physics.py
   │         ├── goodness_of_fit_*_cdf.png  │     └── mirror_plots/
   │         ├── vdf_2d_*_step*.png         │
   │         ├── distribution_evolution_*.png │
   │         ├── brazil_plot.png            ├── diamagnetic_current.py
   │         ├── vdf_1d_parallel_evolution.png │     └── diamagnetic_plots/
   │         ├── vdf_1d_perp_evolution.png  │
   │         ├── particle_energy_partition.png │
   │         ├── magnetic_energy_fluctuation.png │
   │         ├── heat_flux_regions.png      ├── fluctuationofmagneticfiel.py
   │         └── heat_flux_timeseries.png   │     └── field_images/
   │                                        │
   ├── validate_moments.py                  └── spectral_analysis.py
   │     └── validation_plots/                    └── (under development)
   ├── plot_vdf_3d.py
   └── plot_moments_scatter_3d.py
```

---

## 5. Central module: `psc_units.py`

All physical constants derived from the `.cxx` file live here:

| Variable | Value (Mirror) | Value (Firehose) | Meaning |
|----------------|---------------|------------------|--------------------------------------|
| `MASS_RATIO`   | 200.0         | 200.0            | artificial mᵢ/mₑ                     |
| `B0`           | 0.05          | 0.05             | Background field [= vA/c]            |
| `VA`           | 0.05          | 0.05             | Alfvén speed [c=1]                   |
| `OMEGA_CI`     | 0.000250      | 0.000250         | Ion cyclotron frequency              |
| `DI`           | ≈14.142       | ≈14.142          | Ion inertial length [cells]          |
| `NICELL`       | Profile-dependent | Profile-dependent | Particles per cell and species    |
| `BETA_I_PAR`   | 5.0           | 10.0             | Ion parallel beta                    |
| `TI_PAR`       | 0.00625       | 0.0125           | Ion parallel temperature             |
| `TI_PERP`      | 0.01875       | 0.00125          | Ion perpendicular temperature        |
| `Ti_⊥/Ti_∥`    | 3.0           | 0.1              | Ion anisotropy                       |
| `KAPPA`        | `3.0`/`None`  | `3.0`/`None`     | Kappa or Maxwellian                  |

> **Grid resolution and physical scales:**
>
> The grid parameters depend on the executable:
>
> | Profile | Domain | Grid | Δx [d_i] | Δx [d_e] | ppc | nmax |
> |---|---|---|---|---|---|---|
> | `M_S_bM` | 30 × 30 d_i | 1408² | 0.0213 | **0.301** | 1000 | 1,650,000 |
> | Legacy mirror | 32 × 32 d_i | 1536² | 0.0208 | **0.295** | 1000 | 1,800,000 |
> | Legacy firehose | 32 × 32 d_i | 1024² | 0.0312 | **0.442** | 1000 | 1,200,000 |
>
> - `d_e = c/ω_pe = 1` code cell (PSC: c=1, n₀=1, mₑ=1)
> - `d_i = √(mᵢ/mₑ) × d_e = √200 ≈ 14.14 d_e`
> - The listed configurations resolve the electron skin depth (`Δx < 1 d_e`).
> - The final physical time must be computed with the `dt` and `nmax` of the
>   active profile.
>
> `PscConfig1vbecSingle` = **full PIC** (1st order Villasenor-Buneman
> edge-centered). Both species (ions and electrons) are **kinetic particles**.

**Profile selection (environment variable):**

```bash
export PSC_PROFILE=mirror_kappa
```

```bash
export PSC_PROFILE=mirror_maxwellian
```

```bash
export PSC_PROFILE=firehose_kappa
```

```bash
export PSC_PROFILE=firehose_maxwellian
```

**Common unit conversions:**

```python
from psc_units import OMEGA_CI, DI, VA

# Simulation step → physical time
t_physical = step * dt_code * OMEGA_CI   # in units of Ωci⁻¹

# Cell → ion inertial length
x_di = x_cells / DI

# Momentum → velocity in units of vA
v_va = p / VA
```

---

## 6. `PICDataReader` helper — how it finds the files

```python
# 1. Find every file matching a glob pattern
files = PICDataReader.find_files("../build/src/pfd_moments.*.h5")
# Returns: {1200: "pfd_moments.001200_p0.h5", 2400: "...", ...}

# 2. Find the dynamic group inside the HDF5 file
# PSC appends a UID to the group name: "jeh-abc123" or "all_1st-xyz"
group_name = PICDataReader.get_uid_group(f, "jeh-")  # finds "jeh-<any_uid>"

# 3. Read several datasets from the same file in a single open
fields = PICDataReader.read_multiple_fields_3d(
    filename, group_prefix, list_of_dataset_paths
)
```

> **Why the dynamic prefix?**
> PSC adds a hash or UID to each HDF5 group to avoid collisions when writing in
> parallel from multiple MPI ranks. `get_uid_group()` resolves that name at
> runtime without needing to know it in advance.

---

## 7. Additional diagnostics (added to `plot_prt.py`)

### Plot 9 — Evolution of the 1D distribution function

**What it does:** overlays `f(v_∥)` and `f(v_⊥)` at several times (blue→red
colormap = early→late), with a reference Maxwellian drawn as a dashed black line.
It quantifies the **suprathermal tail** as the fraction of particles with
`|v| > 3 v_th`.

```
Outputs: `prt_plots/vdf_1d_parallel_evolution.png` and
`prt_plots/vdf_1d_perp_evolution.png`.
```

**Physics:** shows directly whether the kappa distribution keeps its power-law
tail during the evolution or whether the instability modifies it.

---

### Plot 10 — Energy partition

**What it does:** plots the time evolution of:
- `E_kin_bulk = ½ mᵢ ⟨v⟩²` (mean-flow energy)
- `E_kin_therm = ½ mᵢ ⟨δv²⟩` (random kinetic energy)
- `E_int_ion  = (3/2) Nᵢ Tᵢ` (ion internal energy)
- `E_int_elec = (3/2) Nₑ Tₑ` (electron internal energy)
- `E_B = (δB_rms)²/2` (if field files are available)

All normalized to `E₀` (initial total energy).

```
Outputs: `prt_plots/particle_energy_partition.png` and
`prt_plots/magnetic_energy_fluctuation.png`.
```

**Physics:** reproduces the methodology of PIC studies of anisotropy
instabilities (Hellinger & Trávníček 2008; Kunz et al. 2014) to track how the
energy budget is redistributed between Maxwellian and kappa distributions.

---

### Plot 11 — Heat flux diagnostic

**What it does:** computes the components of the heat flux tensor:

```
q_∥ = (m/2) ⟨δv² · δv_z⟩
q_⊥ = (m/2) ⟨δv² · δv_⊥⟩
```

Particles are split into **4 regions** by `v_z` quartiles (a proxy for spatial
position when `x,y,z` coordinates are not available). It produces a bar panel per
region and a time series.

```
Outputs: prt_plots/heat_flux_regions.png
         prt_plots/heat_flux_timeseries.png
```

**Physics:** characterizes the non-thermal energy transport associated with the
instability dynamics; essential to distinguish the behaviour of kappa
distributions (larger heat flux) from Maxwellian ones.

---

## 8. Quick run

```bash
make all
```

```bash
make particles
```

```bash
make brazil
```

```bash
make validate
```

```bash
make clean
```

(Run from `CodeforAnalisys/`. `make all` uses `DATA_DIR=../build/src`;
`make particles` runs plots 1–11 of `plot_prt.py`; `make brazil` produces the
Brazil plot from grid moments; `make validate` checks moments against the
initialization parameters; `make clean` removes all output directories.)

**Direct run on a specific file:**
```bash
python plot_prt.py ../build/src/prt.000001200.h5
```

```bash
python plot_prt.py "../build/src/prt.*.h5"
```

(The second form takes all snapshots, i.e. the time evolution.)

---

## 9. Unified pipeline: 17 physical diagnostics

The integrated entry point is:

```bash
PSC_PROFILE=F_S_bM python physical_diagnostics.py --data-dir ../corridas_locales/mi_prueba --outdir ../analysis_results/F_S_bM/09_physical_diagnostics
```

If `--particles`, `--fields` or `--moments` are not given,
`PICDataReader.discover_outputs()` automatically selects the `.h5` or `.bp`
series. Two formats must not be mixed for the same step.

### 1. Data reading

Loads particle, field and moment snapshots from HDF5 or ADIOS2. HDF5 groups with
a UID and clean ADIOS2 names are resolved through the same API.

### 2. Species separation

Separates ions and electrons by the sign of the charge:

```text
q > 0: ions
q < 0: electrons
```

### 3. Temperatures and anisotropy

Computes:

```text
T_parallel = m <(v_parallel - <v_parallel>)²>
T_perp     = m/2 [<(vx-<vx>)²> + <(vy-<vy>)²>]
A          = T_perp / T_parallel
R          = T_parallel / T_perp
```

Main outputs: `validation_table.csv` and `anisotropy_table.csv`.

### 4. Time series

Generates `anisotropy_vs_time.png` and
`temperature_parallel_perp_vs_time.png`.

### 5. Two-dimensional VDF

Builds logarithmic $f(v_\perp,v_\parallel)$ maps for the selected particle
snapshots: `vdf_2d_step_<step>.png`.

### 6. Maxwellian and Kappa fits

Fits both distributions, compares global and tail errors, estimates $\kappa$ and
computes the suprathermal fraction. Outputs: `fit_metrics.csv`,
`kappa_fit_vs_time.png`, `suprathermal_fraction_vs_time.png` and
`kappa_vs_maxwellian_step_<step>.png`.

### 7. Brazil plots

Plots $\langle\beta_{\parallel i}\rangle$ against $\langle A_i\rangle$ and
overlays the mirror/firehose thresholds. Output: `brazil_plot_global.png`.
(`brazil_plot_spatial.png` was an alias of the same file in earlier versions; it
is no longer generated.)

### 8. Spatial anisotropy maps

Reconstructs $T_{\parallel i}$, $T_{\perp i}$ and $A_i(y,z)$ from grid moments.
Produces time statistics in `anisotropy_spatial_stats.csv` and
`A_i_map_step_<step>.png` maps.

### 9. Magnetic fluctuations

Computes $|B|$, $\delta B/B_0$, minima, the depth of mirror structures and
$\delta B_{\mathrm{rms}}(t)$. Tabular output: `field_fluctuation_table.csv`.

### 10. Growth rate

Fits the approximately linear phase of $\ln(\delta B_{\mathrm{rms}})$ to estimate
$\gamma$. Outputs: `growth_rate_summary.csv` and `growth_rate_fit.png`.

### 11. Magnetic components

Separates parallel and transverse fluctuations relative to $B_0\parallel z$.
Output: `deltaB_components_comparison.png`, which already contains both
components. (`deltaB_parallel_vs_time.png` and `deltaB_perp_vs_time.png` were
byte-identical aliases of the same file, not separate plots; they are no longer
generated.)

### 12. Transverse magnetic spectrum

The master diagnostic reuses the numerical core of `SpectralAnalyzer` and
computes:

```text
PSD_Bperp(k_y,k_z) = PSD_deltaBx + PSD_deltaBy
E_Bperp(k)         = radial sum of PSD_Bperp
```

It applies a Hann window, obtains the dominant mode and fits a power law over the
available central interval. Outputs:

```text
magnetic_spectrum_step_<step>.png
magnetic_spectrum_table.csv
```

The table records the plane, the spacing, `peak_k`, the peak power, the slope and
the correlation coefficient of the fit.

### 13. Diamagnetic currents

Computes ion, electron and total maps from $\nabla P_\perp\times B/B^2$. Outputs:
`J_dia_i_map_step_<step>.png`, `J_dia_e_map_step_<step>.png` and
`J_dia_total_map_step_<step>.png`.

### 14. Spatial correlations

Evaluates the correlations of $A_i$ with $\delta B$, $|B|$, $J_{dia}$ and
density. Produces `spatial_correlations.csv` and the corresponding scatter plots.

### 15. Energy partition and conservation

Combines bulk, thermal and magnetic energy proxies; computes the relative
variation with respect to the first available state. Outputs: `energy_table.csv`,
`energy_partition.png` and `energy_conservation_error.png`.

### 16. Heat fluxes

Computes the global third particle moments $q_\parallel$ and $q_\perp$ for each
snapshot. Outputs: `heat_flux_parallel_vs_time.png` and
`heat_flux_perp_vs_time.png`.

It also computes spatial proxies from the pressure tensor and the bulk velocity:

```text
q_parallel ≈ P_parallel v_parallel
q_perp     ≈ P_perp |v_perp|
```

The domain is split into four fixed quadrants to compare localized transport.
Outputs:

```text
localized_heat_flux_table.csv
q_parallel_map_step_<step>.png
q_perp_map_step_<step>.png
```

`heat_flux_analysis.py` remains available for a specialized run, but the basic
localized diagnostic is already part of the master pipeline and uses the same
HDF5/BP reader.

### 17. Maxwellian vs Kappa comparison

`compare_physical_cases.py` combines the tables of two or more cases:

```bash
python compare_physical_cases.py maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics --outdir ../analysis_results/comparison_physical
```

It compares anisotropy, growth, fluctuations, energy, heat flux and the
suprathermal fraction without re-reading the original snapshots.

## 10. Format verification

Install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` includes the `adios2` Python bindings; a C++ ADIOS2
installation on its own does not guarantee that `import adios2` works.

Local HDF5 test:

```bash
PSC_PROFILE=F_S_bM ../.venv/bin/python physical_diagnostics.py --data-dir ../corridas_locales/mi_prueba --outdir /tmp/psc_physical_validation --max-particle-steps 2 --max-map-steps 2
```

ADIOS2 check on COSMA:

```bash
source ../src/cosma_adios2_env.sh && python -c "import adios2; print(adios2.__file__)"
```

```bash
python physical_diagnostics.py --data-dir /path/to/snapshots_bp --outdir /path/to/results
```

If the directory only contains `checkpoint_<step>.bp`, the physical series `pfd`,
`pfd_moments` and `prt_*` are missing; the checkpoint is not an equivalent input
for these 17 diagnostics.

## 11. Implementation and validation performed

Changes incorporated into the pipeline:

1. `PICDataReader.open_data_file()` unifies HDF5 and ADIOS2.
2. Discovery recognizes `.h5` files and `.bp` directories.
3. Path resolution accepts both clean groups and groups with a UID.
4. BP particles are read as separate variables `q`, `m`, `px`, `py`, `pz` and
   `w`.
5. `physical_diagnostics.py` no longer opens HDF5 directly.
6. The transverse spectrum $E_{B_\perp}(k)$ is part of the master diagnostic.
7. Regional heat-flux maps and statistics were added.
8. `spectral_analysis.py` has a NumPy fallback when SciPy is not installed.

Validation run on `corridas_locales/mi_prueba`:

```text
25 HDF5 field snapshots
25 HDF5 moment snapshots
2 selected particle snapshots
2 selected spectral snapshots
4 spatial regions × 25 steps = 100 localized-flux rows
```

Verified results:

```text
physical_diagnostics.py: exits with code 0
compare_physical_cases.py: exits with code 0
magnetic_spectrum_table.csv: generated
localized_heat_flux_table.csv: generated
q_parallel_map_step_<step>.png: generated
q_perp_map_step_<step>.png: generated
```

The ADIOS2 path was also checked with a simulated backend reproducing
`FileReader` and with separate field and particle variables. A real cluster test
requires the ADIOS2 Python bindings to be installed in the environment that runs
the analysis.
