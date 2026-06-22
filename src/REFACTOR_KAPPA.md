# Refactor: Implementación de la distribución Kappa

## Motivación

El código original solo soportaba inicialización Maxwelliana de partículas.
Para simular plasmas con colas no térmicas (viento solar, magnetosferas, etc.)
se necesitaba soporte para distribuciones Kappa.

## Versión original (antes del refactor)

```cpp
struct psc_particle_npt
{
  int kind;
  double n;
  double p[3];   // double[3] plano
  double T[3];
  psc::particle::Tag tag;
};

template <typename MP>
struct SetupParticles
{
  // Sin miembro kappa
  // Sin createKappaMultivariate
  // Sin InitNpFunc / InitNpFunc

  int get_n_in_cell(const psc_particle_npt& npt)
  {
    // Box-Muller inline en setupParticle:
    float ran1..ran6; do { ... } while(...);
    double pxi = npt.p[0]
      + sqrtf(-2.f * npt.T[0] / m * sqr(beta) * logf(1.0 - ran1))
      * cosf(2.f * M_PI * ran2);
    // ... setupParticle devuelve Inject con {pxi, pyi, pzi}
  }

  // Un solo template <typename FUNC> para todo
  // operator() envuelve lambda simple → completa manualmente
  // partition duplica el triple loop
};
```

## Cambios en `src/include/setup_particles.hxx`

### 1. `psc_particle_npt` — `Double3` en vez de `double[3]`

```cpp
// Antes:
double p[3];
double T[3];

// Ahora:
Double3 p;
Double3 T;
```

`Double3` es un alias vectorial, más fácil de copiar y usar en lambdas.

### 2. Nueva estructura `psc_particle_np`

```cpp
struct psc_particle_np
{
  int kind;
  double n;
  std::function<Double3()> p;  // función que genera momento aleatorio
  psc::particle::Tag tag;
};
```

Antes el momento se sampleaba directo en `setupParticle` con Box-Muller.
Ahora se separa la **generación del momento** (una lambda en `np.p`) de
la **inyección** de la partícula. Esto permite usar Maxwellianas o Kappa
sin cambiar el loop de inyección.

### 3. `InitNptFunc` e `InitNpFunc` — type erasure

Antes todo era `template <typename FUNC>`. Ahora hay dos wrappers que
normalizan la firma del callback a `(kind, pos, patch, idx, npt/np)`:

- **`InitNptFunc`** — para lambdas que llenan `psc_particle_npt` (Maxwellian).  
  Acepta firma simple `(kind, pos, npt)` o completa `(kind, pos, patch, idx, npt)`.

- **`InitNpFunc`** — para lambdas que llenan `psc_particle_np` (Kappa).  
  Solo acepta firma completa `(kind, pos, patch, idx, np)`.

### 4. Nueva función `createMaxwellian` (refactor)

El Box-Muller que estaba inline en `setupParticle` se extrae a una función
reutilizable que devuelve `std::function<Double3()>`:

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
    // gamma correction opcional
    return p;
  };
}
```

Usa `rng::Normal` en vez de `random()` + `logf` + `cosf` + rechazo.

### 5. Nueva función `createKappaMultivariate` (principal adición)

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
    // gamma correction opcional
    return p;
  };
}
```

Muestreo de distribución Kappa multivariante anisotrópica:
- `Y ~ Gamma(kappa - 0.5, 1)`
- `S = sqrt((kappa - 1.5) / Y)`
- `p[i] = drift[i] + Z_i * S * beta * sqrt(T[i] / m)` con `Z_i ~ N(0,1)`

Requiere `kappa > 1.5` (assert). Cada dirección puede tener temperatura
y drift independientes (anisotropía).

### 6. `initNpt_to_initNp` — compatibilidad hacia atrás

Convierte un `InitNptFunc` → `InitNpFunc` llamando `createMaxwellian`
automáticamente:

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

Las lambdas que llenan `npt` (estilo antiguo) siguen funcionando sin cambios:
se convierten a Maxwellian bajo el capó.

### 7. `op_cellwise` + `centerer` — eliminan duplicación

El triple loop `jx, jy, jz` se extrae a `op_cellwise`, usado tanto por
`setupParticles` como por `partition`:

```cpp
template <typename OpFunc>
void op_cellwise(const Grid_t& grid, int patch, InitNpFunc init_np, OpFunc&& op)
{
  // ... loop sobre jx, jy, jz ...
  // pos con centerer.get_pos() en vez de cálculo manual
}
```

### 8. Nuevo miembro `kappa`

```cpp
double kappa = 3.0; // Parámetro κ de la distribución Kappa
```

### Resumen de cambios estructurales

| Aspecto | Antes | Ahora |
|---|---|---|
| `psc_particle_npt.p/T` | `double[3]` | `Double3` |
| Generación de momento | Box-Muller inline en `setupParticle` | `createMaxwellian` / `createKappaMultivariate` devuelven `std::function<Double3()>` |
| Tipo de callback | `template <typename FUNC>` | `InitNptFunc` / `InitNpFunc` con type erasure |
| Firma de callback simple | Envuelta manual en `operator()` | `InitNptFunc` la envuelve automáticamente |
| Loop de celdas | Duplicado en `setupParticles` y `partition` | `op_cellwise` único |
| Cálculo de posición | `x_cc(jx)`, `y_cc(jy)`, `z_cc(jz)` manual | `centerer.get_pos(patch, index)` |
| Distribuciones | Solo Maxwelliana | Maxwelliana (`createMaxwellian`) + Kappa (`createKappaMultivariate`) |

## Flujos de inicialización

```
Maxwelliano (PSC_USE_KAPPA=0):
  init_npt(kind, pos, npt)             ← llena npt.n, npt.T
    → InitNptFunc lo envuelve
    → initNpt_to_initNp lo convierte:
        np.p = createMaxwellian(npt)   ← Box-Muller
    → setupParticle llama np.p()       ← samplea momento

Kappa (PSC_USE_KAPPA=1):
  init_np(kind, pos, patch, idx, np)   ← llena npt a mano
    → np.p = createKappaMultivariate(npt) ← Gamma + Normal
    → setupParticle llama np.p()       ← samplea momento
```

## Cambios en `src/psc_anisotropy_case.hxx`

Se agregó `#if PSC_USE_KAPPA` para seleccionar entre dos ramas de
`initializeParticles`:

- **`PSC_USE_KAPPA=0` (Maxwelliano):** Usa lambda simple
  `(kind, crd, npt)` → `InitNptFunc` → `createMaxwellian(npt)`
  (conversión automática en `initNpt_to_initNp`).

- **`PSC_USE_KAPPA=1` (Kappa):** Usa lambda completa
  `(kind, pos, patch, idx, np)` → requiere llenar `npt` a mano y llamar
  `createKappaMultivariate(npt)`.

El parámetro κ se configura con `setup_particles.kappa = PSC_KAPPA;`.

### Nuevos `#define` en el `.hxx`

```cpp
#ifndef PSC_USE_KAPPA
#define PSC_USE_KAPPA 0     // 0 = Maxwelliano, 1 = Kappa
#endif

#ifndef PSC_KAPPA
#define PSC_KAPPA 3.0       // valor por defecto de κ
#endif
```

## Casos de uso

| Archivo | `PSC_USE_KAPPA` | Distribución |
|---|---|---|
| `psc_mirror_maxwellian.cxx` | (no definido, default 0) | Maxwelliana |
| `psc_firehose_maxwellian.cxx` | (no definido, default 0) | Maxwelliana |
| `psc_M_S_bM.cxx` | (no definido, default 0) | Bi-Maxwelliana |
| `psc_mirror_kappa3.cxx` | 1 | Kappa (κ=3) |
| `psc_mirror_kappa5.cxx` | 1 | Kappa (κ=5) |
| `psc_firehose_kappa3.cxx` | 1 | Kappa (κ=3) |
| `psc_firehose_kappa5.cxx` | 1 | Kappa (κ=5) |

---

# Cambios necesarios en los scripts de turbulencia

## `psc_turbulence_2D_kappa_3.cxx`

### 1. BUG: falta `npt.kind = kind;` (crítico)

```cpp
// Actual (L. 292) — NO asigna kind:
psc_particle_npt npt{};

// Debería ser:
psc_particle_npt npt{};
npt.kind = kind;            // ← añadir esta línea
```

`createKappaMultivariate` usa `npt.kind` para obtener la masa de la especie
(`kinds_[npt.kind].m`). Sin esto, `npt.kind` queda en 0 (MY_ION) para ambas
especies, y los electrones se inicializan con masa de ión.

Referencia: `psc_mirror_kappa3.cxx`.

### 2. Falta `PSC_RESTART` en `main()`

```cpp
if (const char* restart = std::getenv("PSC_RESTART")) {
    read_checkpoint_filename = restart;
}
```

### 3. `#include <random>` mal ubicado

Moverlo al principio del archivo con los otros includes.

---

## `psc_turbulence_2D_maxwellian_3.cxx`

### 1. Falta `PSC_RESTART` en `main()`
### 2. `#include <random>` mal ubicado

---

## Resumen de cambios

| Archivo | Cambio | Gravedad |
|---|---|---|
| `psc_turbulence_2D_kappa_3.cxx` | `npt.kind = kind;` | **BUG** — resultados incorrectos |
| `psc_turbulence_2D_kappa_3.cxx` | `PSC_RESTART` en `main()` | Funcionalidad perdida |
| `psc_turbulence_2D_kappa_3.cxx` | `#include <random>` al inicio | Estilo |
| `psc_turbulence_2D_maxwellian_3.cxx` | `PSC_RESTART` en `main()` | Funcionalidad perdida |
| `psc_turbulence_2D_maxwellian_3.cxx` | `#include <random>` al inicio | Estilo |
