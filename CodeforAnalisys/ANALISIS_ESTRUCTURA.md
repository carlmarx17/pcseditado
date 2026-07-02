# Estructura del Ecosistema de Análisis PSC

> Documentación técnica de la pipeline de post-procesamiento para las corridas
> normales bi-Maxwellianas `F_*_bM`, `M_*_bM`, `W_*_bM` y los casos
> Maxwellian/Kappa heredados.

Para comandos de uso diario, ver `CodeforAnalisys/README.md`. Este documento
describe el contrato de archivos, datasets y responsabilidades internas.

## 0. Contrato de una corrida

Cada directorio de datos debe contener una sola simulación. Para
`CASE=F_M_bM`, se admite una serie HDF5:

```text
pfd.<step>_p000000.h5
pfd_moments.<step>_p000000.h5
prt_F_M_bM.<step>.h5
```

o la serie ADIOS2 equivalente:

```text
pfd.<step>.bp/
pfd_moments.<step>.bp/
prt_F_M_bM.<step>.bp/
```

El comando de producción es:

```bash
cd CodeforAnalisys
make analysis DATA_DIR=../ruta/F_M_bM CASE=F_M_bM
```

`CASE` selecciona parámetros físicos, especie impulsora, normalización temporal,
nombre de partículas y carpeta de salida. El manifiesto
`analysis_results/F_M_bM/F_M_bM_analysis_manifest.json` registra estas
decisiones y los pasos detectados.

Para Firehose se reportan ambas convenciones:

```text
A_i = T_i_perp / T_i_parallel       # aumenta hacia 1
R_i = T_i_parallel / T_i_perp=1/A_i # disminuye hacia 1
```

Decir solamente que "la anisotropía debe bajar" es ambiguo sin indicar cuál
de estas dos razones se está usando.

---

## 1. Formatos de Archivos de Salida

La pipeline acepta snapshots HDF5 (`.h5`) y ADIOS2 BP (`.bp`):

| Patrón de archivo        | Contenido                                      | Leído por                            |
|--------------------------|------------------------------------------------|--------------------------------------|
| `prt_<CASE>.<step>.h5` o `.bp/` | Datos de partículas (q, m, px, py, pz, w) | `physical_diagnostics.py`, scripts de partículas |
| `pfd.<step>_pN.h5` o `pfd.<step>.bp/` | Campos EM en grilla | `physical_diagnostics.py`, scripts de campos |
| `pfd_moments.<step>_pN.h5` o `.bp/` | Momentos de partículas | `physical_diagnostics.py`, scripts de momentos |

> **Importante:** `checkpoint_<step>.bp/` es un checkpoint de restart y no
> sustituye automáticamente a `pfd`, `pfd_moments` o `prt_*`. El análisis de
> campos, momentos y partículas requiere esas series, en HDF5 o BP. Una
> simulación puede escribir checkpoints ADIOS2 y mantener sus diagnósticos
> regulares en HDF5 si el ejecutable usa `WriterDefault`.

---

## 2. Estructura Interna de los Archivos HDF5

### 2.1 Archivos de Partículas — `prt.*.h5`

```
prt.000001200.h5
└── particles/
    └── p0/
        └── 1d/          ← dataset con structured array
            ├── q[N]     ← carga: +Zi (iones) / -1 (electrones)
            ├── m[N]     ← masa: 200.0 (iones) / 1.0 (electrones)
            ├── w[N]     ← peso estadístico (= 1.0 con fractional_n=true)
            ├── px[N]    ← momento x  [m * v_x en unidades PSC]
            ├── py[N]    ← momento y
            └── pz[N]    ← momento z  (dirección paralela = z ∥ B₀)
```

**Cómo se lee en Python:**

```python
import h5py
import numpy as np

with h5py.File("prt.000001200.h5", "r") as f:
    dset = f["particles"]["p0"]["1d"]

    q  = dset["q"][:]   # +1 iones, -1 electrones
    m  = dset["m"][:]   # 200.0 ó 1.0
    px = dset["px"][:]  # momento perpendicular x
    py = dset["py"][:]  # momento perpendicular y
    pz = dset["pz"][:]  # momento paralelo

# Separar especies
ions  = np.where(q > 0)
elecs = np.where(q < 0)

# PSC guarda u = gamma*v; en estas corridas no relativistas u ~= v.
T_par_ions = 200.0 * np.var(pz[ions])
```

### 2.2 Archivos de Campos — `pfd.*.h5`

```
pfd.001200_p0.h5
└── jeh-<UID>/              ← grupo con prefijo dinámico (UID de la corrida)
    ├── hx_fc/p0/3d[Nx,Ny,Nz]   ← Bx en caras (face-centered)
    ├── hy_fc/p0/3d[Nx,Ny,Nz]   ← By
    ├── hz_fc/p0/3d[Nx,Ny,Nz]   ← Bz  (∥ al campo de fondo B₀)
    ├── ex_ec/p0/3d              ← Ex (edge-centered)
    ├── ey_ec/p0/3d              ← Ey
    └── ez_ec/p0/3d              ← Ez
```

**Cómo se lee (con el helper `PICDataReader`):**

```python
from data_reader import PICDataReader

fields = PICDataReader.read_multiple_fields_3d(
    "pfd.001200_p0.h5",
    "jeh-",                         # prefijo del grupo (ignora el UID)
    ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
)

Bx = fields["hx_fc/p0/3d"]   # array 3D (Nx, Ny, Nz)
By = fields["hy_fc/p0/3d"]
Bz = fields["hz_fc/p0/3d"]
B2 = Bx**2 + By**2 + Bz**2
```

### 2.3 Archivos de Momentos — `pfd_moments.*.h5`

```
pfd_moments.001200_p0.h5
└── all_1st-<UID>/
    ├── rho_i/p0/3d    ← densidad iónica  n_i(y,z)
    ├── txx_i/p0/3d    ← componente Pxx = n m <vx vx>
    ├── tyy_i/p0/3d    ← componente Pyy
    ├── tzz_i/p0/3d    ← componente Pzz  (presión paralela)
    ├── jx_i/p0/3d     ← corriente iónica x
    ├── rho_e/p0/3d    ← densidad electrónica
    └── ...
```

**Temperatura desde momentos (sin partículas):**

```python
moments = PICDataReader.read_multiple_fields_3d(
    "pfd_moments.001200_p0.h5",
    "all_1st",
    ["txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d", "rho_i/p0/3d"],
)

n    = moments["rho_i/p0/3d"].ravel()
Pxx  = moments["txx_i/p0/3d"].ravel()
Pzz  = moments["tzz_i/p0/3d"].ravel()

T_par  = Pzz / n          # temperatura paralela por celda
T_perp = 0.5 * Pxx / n    # (promedio Pxx + Pyy) / 2n
A_i    = T_perp / T_par   # anisotropía
```

---

## 2b. Formato ADIOS2: Archivos `.bp`

Si PSC se compila con `PSC_HAVE_ADIOS2` y se selecciona `WriterAdios2` en el `.cxx`:

```cpp
// En el .cxx de simulación:
using Writer = WriterADIOS2;   // en vez de WriterDefault (HDF5/MRC)
```

entonces **todos** los archivos de salida (campos, momentos, partículas, checkpoints)
cambian de `.h5` a `.bp`.

### Nombres de archivo: `.h5` → `.bp`

| HDF5 (WriterDefault)            | ADIOS2 (WriterADIOS2)              |
|---------------------------------|------------------------------------|
| `pfd.000001200_p0.h5`           | `pfd.000001200.bp/`                |
| `pfd_moments.000001200_p0.h5`   | `pfd_moments.000001200.bp/`        |
| `prt.000001200.h5`              | `prt.000001200.bp/`                |
| —                               | `checkpoint_5000.bp/`              |

> **Nota:** Un `.bp` no es un archivo único sino un **directorio** que contiene
> `md.idx`, `md.0`, `data.0`, etc. Para el usuario se maneja como si fuera
> un solo archivo.

### Estructura interna: ¿Qué cambia?

La jerarquía lógica de los datos **es la misma** que con HDF5.  Lo que cambia
es el contenedor y la API de lectura:

```
pfd.000001200.bp/
├── step         (int)       ← paso de simulación
├── time         (double)    ← tiempo en unidades de código
├── length       (Real3)     ← extensión del dominio [Lx, Ly, Lz]
├── corner       (Real3)     ← esquina inferior del dominio
├── ib           (Int3)      ← offset del ghost boundary
├── im           (Int3)      ← dimensiones incluyendo ghost
└── jeh-<UID>/
    ├── hx_fc/p0/3d[Nx,Ny,Nz]   ← Bx (face-centered) — misma ruta que HDF5
    ├── hy_fc/p0/3d[Nx,Ny,Nz]   ← By
    ├── hz_fc/p0/3d[Nx,Ny,Nz]   ← Bz
    ├── ex_ec/p0/3d              ← Ex (edge-centered)
    └── ...
```

Los datasets de partículas (`prt.*.bp`) tienen la misma estructura
`particles/p0/1d/{q, m, px, py, pz, w}`.

### Metadatos extra en `.bp`

ADIOS2 agrega automáticamente:
- `step` y `time` como variables escalares en cada archivo.
- `length` y `corner` del dominio (no existían en el `.h5` nativo).
- `ib` / `im` — offsets y dimensiones del ghost boundary (útiles para
  reconstruir el dominio global a partir de parches MPI).

Estos metadatos son escritos por `WriterADIOS2::begin_step()`:
```cpp
file_.put("step", step);
file_.put("time", time);
file_.put("length", grid.domain.length);
file_.put("corner", grid.domain.corner);
```

### Cómo leer `.bp` en Python

```python
import adios2
import numpy as np

# Abrir un archivo .bp de campos
with adios2.open("pfd.000001200.bp", "r") as f:
    for step in f:
        # Leer metadatos
        sim_step = step.read("step")
        sim_time = step.read("time")

        # Leer campos — misma ruta lógica que en HDF5
        Bz = step.read("jeh/hz_fc/p0/3d")
        Bx = step.read("jeh/hx_fc/p0/3d")
        By = step.read("jeh/hy_fc/p0/3d")

# Abrir un archivo .bp de partículas
with adios2.open("prt.000001200.bp", "r") as f:
    for step in f:
        q  = step.read("particles/p0/1d/q")
        m  = step.read("particles/p0/1d/m")
        px = step.read("particles/p0/1d/px")
        py = step.read("particles/p0/1d/py")
        pz = step.read("particles/p0/1d/pz")
```

> **Diferencia clave con `h5py`:**  En ADIOS2 la API es `step.read("ruta")`
> en lugar de `f["ruta"][:]`.  Además, la navegación por grupos usa `/` plano
> en vez de la jerarquía de objetos de HDF5 (`f["particles"]["p0"]["1d"]`).

### Diferencia en el prefijo UID del grupo

| HDF5                          | ADIOS2                        |
|-------------------------------|-------------------------------|
| `jeh-abc123/hx_fc/p0/3d`     | `jeh/hx_fc/p0/3d`            |
| `all_1st-xyz789/txx_i/p0/3d` | `all_1st/txx_i/p0/3d`        |

En HDF5, PSC agrega un hash UUID al nombre del grupo (`jeh-<uid>`) para evitar
colisiones MPI.  En ADIOS2 este hash **no se agrega** — el prefijo es limpio
(`jeh/`, `all_1st/`).  Esto significa que `PICDataReader.get_uid_group()`
no es necesario con `.bp`.

### Soporte implementado para `.bp`

| Componente | Estado |
|---|---|
| `data_reader.py` | Lector unificado HDF5/BP, resolución de grupos UID y compatibilidad con `FileReader`, `Stream` y `adios2.open`. |
| `physical_diagnostics.py` | Partículas, campos y momentos se leen mediante `PICDataReader`; no contiene una ruta HDF5 paralela. |
| Descubrimiento | Busca automáticamente `pfd`, `pfd_moments` y `prt_*` en ambos formatos. |
| Espectros | El diagnóstico maestro calcula directamente $E_{B_\perp}(k)$. |
| Checkpoints | Se reservan para restart; no se interpretan como snapshots físicos. |

### Estrategia dual recomendada

Para soportar ambos formatos sin duplicar código:

```python
import os

def open_data_file(filepath):
    """Retorna un lector según la extensión del archivo."""
    if filepath.endswith(".bp") or os.path.isdir(filepath):
        import adios2
        return adios2.open(filepath, "r")
    else:
        import h5py
        return h5py.File(filepath, "r")
```

O bien, usar la variable de entorno:
```bash
export PSC_IO_BACKEND=adios2   # ó "hdf5" (default)
```

---

## 3. Árbol de Scripts y sus Responsabilidades

```
CodeforAnalisys/
│
├── psc_units.py              ← MÓDULO CENTRAL: constantes y conversiones de unidades
│   │                            B0, OMEGA_CI, DI, TI_PAR, TI_PERP, KAPPA ...
│   └── (importado por todos los demás scripts)
│
├── data_reader.py            ← LECTOR HDF5/ADIOS2 unificado
│   │                            PICDataReader: descubrimiento, apertura y resolución de rutas
│   └── (importado por anisotropy_analysis, diamagnetic_current, mirror_physics)
│
├── plot_prt.py               ← ANÁLISIS DE PARTÍCULAS (lee prt.*.h5)
│   ├── Plot 1: VDF 2D        f(v_⊥, v_∥) — mapa de calor log
│   ├── Plot 2: Kappa vs Max  comparación distribución teórica vs datos
│   ├── Plot 3: KS + AD       tests de bondad de ajuste
│   ├── Plot 4: Snapshots VDF paneles multi-tiempo
│   ├── Plot 5: Evolución 1D  f(v_∥,t) y f(v_⊥,t) como heatmap
│   ├── Plot 6: Anisotropía   T_⊥/T_∥ vs tiempo
│   ├── Plot 7: Brazil plot   T_⊥/T_∥ vs β_∥ con umbrales
│   ├── Plot 9: VDF 1D evol.  ← NUEVO: líneas superpuestas + cola supratermal
│   ├── Plot 10: Partición E  ← NUEVO: E_mag / E_kin / E_térmica vs tiempo
│   └── Plot 11: Flujo de calor ← NUEVO: q_∥ y q_⊥ en regiones localizadas
│
├── anisotropy_analysis.py    ← ANÁLISIS DE ANISOTROPÍA (lee pfd_moments + pfd)
│   └── Brazil plot desde momentos de grilla (resolución espacial completa)
│
├── mirror_physics.py         ← HOYOS DE ESPEJO (lee pfd)
│   └── Mapas 2D de |B|, corriente out-of-plane, contornos de fluctuación
│
├── diamagnetic_current.py    ← CORRIENTE DIAMAGNÉTICA (lee pfd_moments + pfd)
│   └── Mapas J_dia iónica, electrónica y total
│
├── fluctuationofmagneticfiel.py  ← FLUCTUACIONES δB (lee pfd)
│   └── Mapas de δB, δB/B₀, animaciones GIF
│
├── spectral_analysis.py      ← MOTOR ESPECTRAL (lee pfd HDF5/BP)
│   └── PSD 1D y 2D de componentes magnéticas
│
├── physical_diagnostics.py  ← DIAGNÓSTICO MAESTRO HDF5/BP
│   ├── partículas, temperaturas, VDF y ajustes
│   ├── momentos, mapas, corrientes y correlaciones
│   ├── fluctuaciones, crecimiento y energía
│   └── espectro transversal E_Bperp(k)
│
├── validate_moments.py       ← VALIDACIÓN (lee prt.*.h5)
│   └── Verifica que momentos medidos = parámetros de inicialización
│
├── plot_vdf_3d.py            ← VDF 3D (lee prt.*.h5)
│   └── Superficie 3D f(vx, vy, vz)
│
├── plot_moments_scatter_3d.py ← SCATTER 3D (lee prt.*.h5)
│   └── Scatter de momentos + histogramas en 3D
│
└── Makefile                  ← ORQUESTADOR
    ├── make brazil     → anisotropy_analysis.py
    ├── make mirror     → mirror_physics.py
    ├── make diamagnetic → diamagnetic_current.py
    ├── make fields     → fluctuationofmagneticfiel.py
    ├── make spectral   → spectral_analysis.py
    ├── make validate   → validate_moments.py
    ├── make particles  → plot_prt.py + plot_vdf_3d.py + plot_moments_scatter_3d.py
    └── make all        → todo excepto spectral y report
```

---

## 4. Flujo de Datos Completo

```
psc_mirror_kappa.cxx              psc_firehose_kappa.cxx
psc_mirror_maxwellian.cxx         psc_firehose_maxwellian.cxx
         │
         │  (simulación PIC)
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
   │     └── validation_plots/                    └── (en desarrollo)
   ├── plot_vdf_3d.py
   └── plot_moments_scatter_3d.py
```

---

## 5. Módulo Central: `psc_units.py`

Todas las constantes físicas derivadas del archivo `.cxx` están aquí:

| Variable       | Valor (Mirror) | Valor (Firehose) | Significado                          |
|----------------|---------------|------------------|--------------------------------------|
| `MASS_RATIO`   | 200.0         | 200.0            | mᵢ/mₑ artificial                    |
| `B0`           | 0.05          | 0.05             | Campo de fondo [= vA/c]              |
| `VA`           | 0.05          | 0.05             | Velocidad de Alfvén [c=1]            |
| `OMEGA_CI`     | 0.000250      | 0.000250         | Frecuencia ciclotrón iónica          |
| `DI`           | ≈14.142       | ≈14.142          | Longitud inercial iónica [celdas]    |
| `NICELL`       | Según perfil  | Según perfil     | Partículas por celda y especie       |
| `BETA_I_PAR`   | 5.0           | 10.0             | Beta paralelo iónico                 |
| `TI_PAR`       | 0.00625       | 0.0125           | Temperatura iónica paralela          |
| `TI_PERP`      | 0.01875       | 0.00125          | Temperatura iónica perpendicular     |
| `Ti_⊥/Ti_∥`    | 3.0           | 0.1              | Anisotropía iónica                   |
| `KAPPA`        | `3.0`/`None`  | `3.0`/`None`     | Kappa ó Maxwellian                   |

> **Resolución de grilla y escalas físicas:**
>
> Los parámetros de grilla dependen del ejecutable:
>
> | Perfil | Dominio | Grid | Δx [d_i] | Δx [d_e] | ppc | nmax |
> |---|---|---|---|---|---|---|
> | `M_S_bM` | 30 × 30 d_i | 1408² | 0.0213 | **0.301** | 1000 | 1,650,000 |
> | Mirror heredado | 32 × 32 d_i | 1536² | 0.0208 | **0.295** | 1000 | 1,800,000 |
> | Firehose heredado | 32 × 32 d_i | 1024² | 0.0312 | **0.442** | 1000 | 1,200,000 |
>
> - `d_e = c/ω_pe = 1` celda de código (PSC: c=1, n₀=1, mₑ=1)
> - `d_i = √(mᵢ/mₑ) × d_e = √200 ≈ 14.14 d_e`
> - Las configuraciones listadas resuelven la skin depth electrónica (`Δx < 1 d_e`).
> - El tiempo físico final debe calcularse con el `dt` y `nmax` del perfil activo.
>
> `PscConfig1vbecSingle` = **full PIC** (1st order Villasenor-Buneman Edge-Centered).
> Ambas especies (iones y electrones) son **partículas cinéticas**.

**Selección de perfil (variable de entorno):**

```bash
# Mirror con kappa (por defecto)
export PSC_PROFILE=mirror_kappa

# Mirror con Maxwellian
export PSC_PROFILE=mirror_maxwellian

# Firehose con kappa
export PSC_PROFILE=firehose_kappa

# Firehose con Maxwellian
export PSC_PROFILE=firehose_maxwellian
```

**Conversión de unidades frecuente:**

```python
from psc_units import OMEGA_CI, DI, VA

# Paso de simulación → tiempo físico
t_physical = step * dt_code * OMEGA_CI   # en unidades Ωci⁻¹

# Celda → longitud inercial iónica
x_di = x_cells / DI

# Momento → velocidad en unidades de vA
v_va = p / VA
```

---

## 6. Helper `PICDataReader` — Cómo Encuentra los Archivos

```python
# 1. Buscar todos los archivos que coincidan con un patrón glob
files = PICDataReader.find_files("../build/src/pfd_moments.*.h5")
# Retorna: {1200: "pfd_moments.001200_p0.h5", 2400: "...", ...}

# 2. Encontrar el grupo dinámico dentro del HDF5
# PSC agrega un UID al nombre del grupo: "jeh-abc123" o "all_1st-xyz"
group_name = PICDataReader.get_uid_group(f, "jeh-")  # encuentra "jeh-<cualquier_uid>"

# 3. Leer múltiples datasets de un mismo archivo en una sola apertura
fields = PICDataReader.read_multiple_fields_3d(
    filename, group_prefix, list_of_dataset_paths
)
```

> **¿Por qué el prefijo dinámico?**
> PSC añade un hash o UID a cada grupo HDF5 para evitar colisiones cuando se
> escribe en paralelo desde múltiples MPI ranks. `get_uid_group()` resuelve
> ese nombre en tiempo de ejecución sin necesidad de conocerlo de antemano.

---

## 7. Nuevos Diagnósticos (añadidos a `plot_prt.py`)

### Plot 9 — Evolución de la Función de Distribución 1D

**Qué hace:** Superpone `f(v_∥)` y `f(v_⊥)` en múltiples tiempos (colormap
azul→rojo = temprano→tardío), con una Maxwelliana de referencia trazada en
línea negra discontinua. Cuantifica la **cola supratermal** como fracción de
partículas con `|v| > 3 v_th`.

```
Salidas: `prt_plots/vdf_1d_parallel_evolution.png` y
`prt_plots/vdf_1d_perp_evolution.png`.
```

**Física:** Permite ver directamente si la distribución kappa mantiene su cola
de ley de potencia durante la evolución o si la inestabilidad la modifica.

---

### Plot 10 — Partición de Energía

**Qué hace:** Traza la evolución temporal de:
- `E_cin_bulk = ½ mᵢ ⟨v⟩²` (energía del flujo medio)
- `E_cin_term = ½ mᵢ ⟨δv²⟩` (energía cinética aleatoria)
- `E_int_ion  = (3/2) Nᵢ Tᵢ` (energía interna iónica)
- `E_int_elec = (3/2) Nₑ Tₑ` (energía interna electrónica)
- `E_B = (δB_rms)²/2` (si hay archivos de campo disponibles)

Todas normalizadas a `E₀` (energía total inicial).

```
Salidas: `prt_plots/particle_energy_partition.png` y
`prt_plots/magnetic_energy_fluctuation.png`.
```

**Física:** Reproduce la metodología de estudios PIC de inestabilidades
de anisotropía (Hellinger & Trávníček 2008; Kunz et al. 2014) para rastrear
cómo se redistribuye el presupuesto de energía entre distribuciones Maxwelliana
y kappa.

---

### Plot 11 — Diagnóstico de Flujo de Calor

**Qué hace:** Calcula los componentes del tensor de flujo de calor:

```
q_∥ = (m/2) ⟨δv² · δv_z⟩
q_⊥ = (m/2) ⟨δv² · δv_⊥⟩
```

Las partículas se dividen en **4 regiones** según cuartiles de `v_z`
(proxy de posición espacial cuando no se dispone de coordenadas `x,y,z`).
Genera un panel de barras por región y una serie temporal.

```
Salidas: prt_plots/heat_flux_regions.png
         prt_plots/heat_flux_timeseries.png
```

**Física:** Caracteriza el transporte de energía no térmica asociado a la
dinámica de la inestabilidad; esencial para distinguir el comportamiento de
distribuciones kappa (mayor flujo de calor) vs Maxwelliana.

---

## 8. Ejecución Rápida

```bash
# Desde CodeforAnalisys/

# Todos los análisis (usa DATA_DIR=../build/src)
make all

# Solo partículas (plots 1–11 de plot_prt.py)
make particles

# Solo Brazil plot desde momentos de grilla
make brazil

# Validación de momentos contra parámetros de inicialización
make validate

# Limpiar todos los directorios de salida
make clean
```

**Ejecución directa con un archivo específico:**
```bash
python plot_prt.py ../build/src/prt.000001200.h5
python plot_prt.py "../build/src/prt.*.h5"    # todos los snapshots (evolución temporal)
```

---

## 9. Pipeline unificada: 17 diagnósticos físicos

El punto de entrada integrado es:

```bash
PSC_PROFILE=F_S_bM python physical_diagnostics.py \
  --data-dir ../corridas_locales/mi_prueba \
  --outdir ../analysis_results/F_S_bM/09_physical_diagnostics
```

Si no se pasan `--particles`, `--fields` o `--moments`,
`PICDataReader.discover_outputs()` selecciona automáticamente las series
`.h5` o `.bp`. No se deben mezclar dos formatos para el mismo paso.

### 1. Lectura de datos

Carga snapshots de partículas, campos y momentos desde HDF5 o ADIOS2. Los
grupos HDF5 con UID y los nombres limpios de ADIOS2 se resuelven con la misma
API.

### 2. Separación de especies

Separa iones y electrones por el signo de la carga:

```text
q > 0: iones
q < 0: electrones
```

### 3. Temperaturas y anisotropía

Calcula:

```text
T_parallel = m <(v_parallel - <v_parallel>)²>
T_perp     = m/2 [<(vx-<vx>)²> + <(vy-<vy>)²>]
A          = T_perp / T_parallel
R          = T_parallel / T_perp
```

Salidas principales: `validation_table.csv` y `anisotropy_table.csv`.

### 4. Series temporales

Genera `anisotropy_vs_time.png` y
`temperature_parallel_perp_vs_time.png`.

### 5. VDF bidimensional

Construye mapas logarítmicos
$f(v_\perp,v_\parallel)$ para los snapshots de partículas seleccionados:
`vdf_2d_step_<step>.png`.

### 6. Ajustes Maxwelliano y Kappa

Ajusta ambas distribuciones, compara errores globales y de cola, estima
$\kappa$ y calcula la fracción supratermal. Salidas:
`fit_metrics.csv`, `kappa_fit_vs_time.png`,
`suprathermal_fraction_vs_time.png` y
`kappa_vs_maxwellian_step_<step>.png`.

### 7. Brazil plots

Representa $\langle\beta_{\parallel i}\rangle$ frente a
$\langle A_i\rangle$ y superpone umbrales mirror/firehose. Salidas:
`brazil_plot_global.png` y `brazil_plot_spatial.png`.

### 8. Mapas espaciales de anisotropía

Reconstruye $T_{\parallel i}$, $T_{\perp i}$ y
$A_i(y,z)$ desde momentos de grilla. Produce estadísticas temporales en
`anisotropy_spatial_stats.csv` y mapas `A_i_map_step_<step>.png`.

### 9. Fluctuaciones magnéticas

Calcula $|B|$, $\delta B/B_0$, mínimos, profundidad de estructuras mirror y
$\delta B_{\mathrm{rms}}(t)$. Salida tabular:
`field_fluctuation_table.csv`.

### 10. Tasa de crecimiento

Ajusta la fase aproximadamente lineal de
$\ln(\delta B_{\mathrm{rms}})$ para estimar $\gamma$. Salidas:
`growth_rate_summary.csv` y `growth_rate_fit.png`.

### 11. Componentes magnéticas

Separa fluctuaciones paralelas y transversales respecto a
$B_0\parallel z$. Salidas:
`deltaB_parallel_vs_time.png`, `deltaB_perp_vs_time.png` y
`deltaB_components_comparison.png`.

### 12. Espectro magnético transversal

El diagnóstico maestro reutiliza el núcleo numérico de `SpectralAnalyzer` y
calcula:

```text
PSD_Bperp(k_y,k_z) = PSD_deltaBx + PSD_deltaBy
E_Bperp(k)         = suma radial de PSD_Bperp
```

Aplica ventana de Hann, obtiene el modo dominante y ajusta una ley de potencia
en el intervalo central disponible. Salidas:

```text
magnetic_spectrum_step_<step>.png
magnetic_spectrum_table.csv
```

La tabla registra plano, espaciado, `peak_k`, potencia máxima, pendiente y
coeficiente de correlación del ajuste.

### 13. Corrientes diamagnéticas

Calcula mapas iónicos, electrónicos y totales a partir de
$\nabla P_\perp\times B/B^2$. Salidas:
`J_dia_i_map_step_<step>.png`,
`J_dia_e_map_step_<step>.png` y
`J_dia_total_map_step_<step>.png`.

### 14. Correlaciones espaciales

Evalúa correlaciones de $A_i$ con $\delta B$, $|B|$, $J_{dia}$ y densidad.
Produce `spatial_correlations.csv` y los scatter plots correspondientes.

### 15. Partición y conservación de energía

Combina proxies de energía bulk, térmica y magnética; calcula la variación
relativa respecto al primer estado disponible. Salidas:
`energy_table.csv`, `energy_partition.png` y
`energy_conservation_error.png`.

### 16. Flujos de calor

Calcula los terceros momentos globales de partículas
$q_\parallel$ y $q_\perp$ para cada snapshot. Salidas:
`heat_flux_parallel_vs_time.png` y `heat_flux_perp_vs_time.png`.

Además calcula proxies espaciales desde el tensor de presión y la velocidad
bulk:

```text
q_parallel ≈ P_parallel v_parallel
q_perp     ≈ P_perp |v_perp|
```

El dominio se divide en cuatro cuadrantes fijos para comparar transporte
localizado. Salidas:

```text
localized_heat_flux_table.csv
q_parallel_map_step_<step>.png
q_perp_map_step_<step>.png
```

`heat_flux_analysis.py` permanece disponible para una ejecución especializada,
pero el diagnóstico localizado básico ya forma parte del pipeline maestro y
usa el mismo lector HDF5/BP.

### 17. Comparación Maxwelliana frente a Kappa

`compare_physical_cases.py` combina las tablas de dos o más casos:

```bash
python compare_physical_cases.py \
  maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics \
  kappa=../analysis_results/mirror_kappa/09_physical_diagnostics \
  --outdir ../analysis_results/comparison_physical
```

Compara anisotropía, crecimiento, fluctuaciones, energía, flujo de calor y
fracción supratermal sin volver a leer los snapshots originales.

## 10. Verificación de formatos

Instalar las dependencias del análisis:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` incluye las bindings Python `adios2`; la instalación C++
de ADIOS2 por sí sola no garantiza que `import adios2` esté disponible.

Prueba HDF5 local:

```bash
PSC_PROFILE=F_S_bM ../.venv/bin/python physical_diagnostics.py \
  --data-dir ../corridas_locales/mi_prueba \
  --outdir /tmp/psc_physical_validation \
  --max-particle-steps 2 --max-map-steps 2
```

Comprobación ADIOS2 en COSMA:

```bash
source ../src/cosma_adios2_env.sh
python -c "import adios2; print(adios2.__file__)"

python physical_diagnostics.py \
  --data-dir /ruta/a/snapshots_bp \
  --outdir /ruta/a/resultados
```

Si el directorio solo contiene `checkpoint_<step>.bp`, faltan las series
físicas `pfd`, `pfd_moments` y `prt_*`; el checkpoint no es una entrada
equivalente para estos 17 diagnósticos.

## 11. Implementación y validación realizadas

Cambios incorporados a la pipeline:

1. `PICDataReader.open_data_file()` unifica HDF5 y ADIOS2.
2. El descubrimiento reconoce archivos `.h5` y directorios `.bp`.
3. La resolución de rutas acepta grupos limpios y grupos con UID.
4. Las partículas BP se leen como variables separadas `q`, `m`, `px`, `py`,
   `pz` y `w`.
5. `physical_diagnostics.py` dejó de abrir HDF5 directamente.
6. El espectro transversal $E_{B_\perp}(k)$ forma parte del diagnóstico
   maestro.
7. Se añadieron mapas y estadísticas regionales de flujo de calor.
8. `spectral_analysis.py` dispone de fallback NumPy cuando SciPy no está
   instalado.

Validación ejecutada sobre `corridas_locales/mi_prueba`:

```text
25 snapshots de campos HDF5
25 snapshots de momentos HDF5
2 snapshots de partículas seleccionados
2 snapshots espectrales seleccionados
4 regiones espaciales × 25 pasos = 100 filas de flujo localizado
```

Resultados comprobados:

```text
physical_diagnostics.py: finaliza con código 0
compare_physical_cases.py: finaliza con código 0
magnetic_spectrum_table.csv: generado
localized_heat_flux_table.csv: generado
q_parallel_map_step_<step>.png: generado
q_perp_map_step_<step>.png: generado
```

La ruta ADIOS2 también se comprobó con un backend simulado que reproduce
`FileReader` y con variables separadas de campos y partículas. La prueba real
en un clúster requiere que las bindings Python ADIOS2 estén instaladas en el
entorno que ejecuta el análisis.
