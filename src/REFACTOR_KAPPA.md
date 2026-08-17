# Refactor: Kappa distribution implementation

## Motivation

The original code only supported Maxwellian particle initialization.
To simulate plasmas with non-thermal tails (solar wind, magnetospheres, etc.)
support for Kappa distributions was needed.

## Original version (before the refactor)

```cpp
struct psc_particle_npt
{
  int kind;
  double n;
  double p[3];   // plain double[3]
  double T[3];
  psc::particle::Tag tag;
};

template <typename MP>
struct SetupParticles
{
  // No kappa member
  // No createKappaMultivariate
  // No InitNpFunc / InitNpFunc

  int get_n_in_cell(const psc_particle_npt& npt)
  {
    // Box-Muller inline in setupParticle:
    float ran1..ran6; do { ... } while(...);
    double pxi = npt.p[0]
      + sqrtf(-2.f * npt.T[0] / m * sqr(beta) * logf(1.0 - ran1))
      * cosf(2.f * M_PI * ran2);
    // ... setupParticle returns Inject with {pxi, pyi, pzi}
  }

  // A single template <typename FUNC> for everything
  // operator() wraps a simple lambda → filled in manually
  // partition duplicates the triple loop
};
```

## Changes in `src/include/setup_particles.hxx`

### 1. `psc_particle_npt` — `Double3` instead of `double[3]`

```cpp
// Before:
double p[3];
double T[3];

// Now:
Double3 p;
Double3 T;
```

`Double3` is a vector alias, easier to copy and use in lambdas.

### 2. New `psc_particle_np` structure

```cpp
struct psc_particle_np
{
  int kind;
  double n;
  std::function<Double3()> p;  // function that generates a random momentum
  psc::particle::Tag tag;
};
```

Before, the momentum was sampled directly in `setupParticle` with Box-Muller.
Now the **momentum generation** (a lambda in `np.p`) is separated from
particle **injection**. This allows using Maxwellian or Kappa
without changing the injection loop.

### 3. `InitNptFunc` and `InitNpFunc` — type erasure

Before, everything was `template <typename FUNC>`. Now there are two wrappers that
normalize the callback signature to `(kind, pos, patch, idx, npt/np)`:

- **`InitNptFunc`** — for lambdas that fill `psc_particle_npt` (Maxwellian).  
  Accepts either the simple signature `(kind, pos, npt)` or the full one `(kind, pos, patch, idx, npt)`.

- **`InitNpFunc`** — for lambdas that fill `psc_particle_np` (Kappa).  
  Only accepts the full signature `(kind, pos, patch, idx, np)`.

### 4. New `createMaxwellian` function (refactor)

The Box-Muller code that was inline in `setupParticle` is extracted into a
reusable function that returns `std::function<Double3()>`:

```cpp
std::function<Double3()> createMaxwellian(const psc_particle_npt& npt)
{
  assert(npt.kind >= 0 && npt.kind < kinds_.size());
  double beta = norm_.beta;
  double m = kinds_[npt.kind].m;

  return [=]() {
    static rng::Normal<double> dist;
    Double3 p;
    for (int i = 0; i < 3; i++)
      p[i] = dist.get(npt.p[i], beta * std::sqrt(npt.T[i] / m));
    // optional gamma correction
    return p;
  };
}
```

Uses `rng::Normal` instead of `random()` + `logf` + `cosf` + rejection.

### 5. New `createKappaMultivariate` function (main addition)

```cpp
std::function<Double3()> createKappaMultivariate(const psc_particle_npt& npt)
{
  assert(npt.kind >= 0 && npt.kind < kinds_.size());
  assert(kappa > 1.5);

  double beta = norm_.beta;
  double m = kinds_[npt.kind].m;
  double k = kappa;

  return [=]() {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::gamma_distribution<double> dist_gamma(k - 0.5, 1.0);
    static rng::Normal<double> dist_norm;

    double Y = dist_gamma(gen);
    double S = std::sqrt((k - 1.5) / (Y + 1e-12));

    Double3 p;
    for (int i = 0; i < 3; i++) {
      double Z = dist_norm.get(0.0, 1.0);
      p[i] = npt.p[i] + Z * S * beta * std::sqrt(npt.T[i] / m);
    }
    // optional gamma correction
    return p;
  };
}
```

Sampling of the anisotropic multivariate Kappa distribution:
- `Y ~ Gamma(kappa - 0.5, 1)`
- `S = sqrt((kappa - 1.5) / Y)`
- `p[i] = drift[i] + Z_i * S * beta * sqrt(T[i] / m)` with `Z_i ~ N(0,1)`

Requires `kappa > 1.5` (assert). Each direction can have an independent
temperature and drift (anisotropy).

### 6. `initNpt_to_initNp` — backward compatibility

Converts an `InitNptFunc` → `InitNpFunc` by calling `createMaxwellian`
automatically:

```cpp
InitNpFunc initNpt_to_initNp(InitNptFunc& init_npt)
{
  return InitNpFunc(
    [&](int kind, Double3 pos, int patch, Int3 idx, psc_particle_np& np) {
      psc_particle_npt npt{};
      npt.kind = np.kind;
      init_npt(kind, pos, patch, idx, npt);
      np.n = npt.n;
      np.p = createMaxwellian(npt);
      np.tag = npt.tag;
    });
}
```

Lambdas that fill `npt` (the old style) keep working without changes:
they get converted to Maxwellian under the hood.

### 7. `op_cellwise` + `centerer` — remove duplication

The `jx, jy, jz` triple loop is extracted into `op_cellwise`, used both by
`setupParticles` and `partition`:

```cpp
template <typename OpFunc>
void op_cellwise(const Grid_t& grid, int patch, InitNpFunc init_np, OpFunc&& op)
{
  // ... loop over jx, jy, jz ...
  // pos using centerer.get_pos() instead of a manual calculation
}
```

### 8. New `kappa` member

```cpp
double kappa = 3.0; // κ parameter of the Kappa distribution
```

### Summary of structural changes

| Aspect | Before | Now |
|---|---|---|
| `psc_particle_npt.p/T` | `double[3]` | `Double3` |
| Momentum generation | Box-Muller inline in `setupParticle` | `createMaxwellian` / `createKappaMultivariate` return `std::function<Double3()>` |
| Callback type | `template <typename FUNC>` | `InitNptFunc` / `InitNpFunc` with type erasure |
| Simple callback signature | Wrapped manually in `operator()` | `InitNptFunc` wraps it automatically |
| Cell loop | Duplicated in `setupParticles` and `partition` | Single `op_cellwise` |
| Position calculation | Manual `x_cc(jx)`, `y_cc(jy)`, `z_cc(jz)` | `centerer.get_pos(patch, index)` |
| Distributions | Maxwellian only | Maxwellian (`createMaxwellian`) + Kappa (`createKappaMultivariate`) |

## Initialization flows

```
Maxwellian (PSC_USE_KAPPA=0):
  init_npt(kind, pos, npt)             ← fills npt.n, npt.T
    → InitNptFunc wraps it
    → initNpt_to_initNp converts it:
        np.p = createMaxwellian(npt)   ← Box-Muller
    → setupParticle calls np.p()       ← samples momentum

Kappa (PSC_USE_KAPPA=1):
  init_np(kind, pos, patch, idx, np)   ← fills npt by hand
    → np.p = createKappaMultivariate(npt) ← Gamma + Normal
    → setupParticle calls np.p()       ← samples momentum
```

## Changes in `src/psc_anisotropy_case.hxx`

`#if PSC_USE_KAPPA` was added to select between two branches of
`initializeParticles`:

- **`PSC_USE_KAPPA=0` (Maxwellian):** Uses the simple lambda
  `(kind, crd, npt)` → `InitNptFunc` → `createMaxwellian(npt)`
  (automatic conversion in `initNpt_to_initNp`).

- **`PSC_USE_KAPPA=1` (Kappa):** Uses the full lambda
  `(kind, pos, patch, idx, np)` → requires filling `npt` by hand and calling
  `createKappaMultivariate(npt)`.

The κ parameter is set with `setup_particles.kappa = PSC_KAPPA;`.

### New `#define`s in the `.hxx`

```cpp
#ifndef PSC_USE_KAPPA
#define PSC_USE_KAPPA 0     // 0 = Maxwellian, 1 = Kappa
#endif

#ifndef PSC_KAPPA
#define PSC_KAPPA 3.0       // default value of κ
#endif
```

## Use cases

| File | `PSC_USE_KAPPA` | Distribution |
|---|---|---|
| `psc_mirror_bimaxwellian_strong.cxx` | (undefined, default 0) | Maxwellian |
| `psc_firehose_bimaxwellian_strong.cxx` | (undefined, default 0) | Maxwellian |
| `psc_mirror_bimaxwellian_strong.cxx` | (undefined, default 0) | Bi-Maxwellian |
| `psc_mirror_bikappa3.cxx` | 1 | Kappa (κ=3) |
| `psc_mirror_bikappa5.cxx` | 1 | Kappa (κ=5) |
| `psc_firehose_bikappa3.cxx` | 1 | Kappa (κ=3) |
| `psc_firehose_bikappa5.cxx` | 1 | Kappa (κ=5) |

---

# Required changes in the turbulence scripts

## `psc_turbulence_2D_kappa_3.cxx`

### 1. BUG: missing `npt.kind = kind;` (critical)

```cpp
// Current (L. 292) — does NOT assign kind:
psc_particle_npt npt{};

// Should be:
psc_particle_npt npt{};
npt.kind = kind;            // ← add this line
```

`createKappaMultivariate` uses `npt.kind` to get the species mass
(`kinds_[npt.kind].m`). Without this, `npt.kind` stays at 0 (MY_ION) for both
species, and the electrons get initialized with the ion mass.

Reference: `psc_mirror_bikappa3.cxx`.

### 2. Missing `PSC_RESTART` in `main()`

```cpp
if (const char* restart = std::getenv("PSC_RESTART")) {
    read_checkpoint_filename = restart;
}
```

### 3. `#include <random>` misplaced

Move it to the top of the file with the other includes.

---

## `psc_turbulence_2D_maxwellian_3.cxx`

### 1. Missing `PSC_RESTART` in `main()`
### 2. `#include <random>` misplaced

---

## Summary of changes

| File | Change | Severity |
|---|---|---|
| `psc_turbulence_2D_kappa_3.cxx` | `npt.kind = kind;` | **BUG** — incorrect results |
| `psc_turbulence_2D_kappa_3.cxx` | `PSC_RESTART` in `main()` | Lost functionality |
| `psc_turbulence_2D_kappa_3.cxx` | `#include <random>` at the top | Style |
| `psc_turbulence_2D_maxwellian_3.cxx` | `PSC_RESTART` in `main()` | Lost functionality |
| `psc_turbulence_2D_maxwellian_3.cxx` | `#include <random>` at the top | Style |
