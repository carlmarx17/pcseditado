# Estructura del Ecosistema de Análisis PSC

> Documentación técnica de la pipeline de post-procesamiento para las simulaciones
> `psc_mirror_kappa`, `psc_mirror_maxwellian`, `psc_firehose_kappa`, `psc_firehose_maxwellian`.

---

## 1. Formatos de Archivos de Salida

PSC genera dos tipos de archivos HDF5 (`.h5`) durante la simulación:

| Patrón de archivo        | Contenido                                      | Leído por                            |
|--------------------------|------------------------------------------------|--------------------------------------|
| `prt.NNNNNNNN.h5`        | Datos de partículas (q, m, px, py, pz, w)      | `plot_prt.py`, `validate_moments.py` |
| `pfd.NNNNNN_pN.h5`       | Campos EM en grilla (Bx, By, Bz, Ex, Ey, Ez)  | `anisotropy_analysis.py`, `mirror_physics.py` |
| `pfd_moments.NNNNNN_pN.h5` | Momentos de partículas (P_ij, rho, J)        | `anisotropy_analysis.py`, `diamagnetic_current.py` |

> **Nota sobre `.bp`:** Si la simulación se compiló con soporte ADIOS2 activado,
> los datos se escriben en formato `.bp` (Binary Pack de ADIOS2) en lugar de `.h5`.
> En ese caso el lector debe usar la librería `adios2` en Python en lugar de `h5py`.
> Los scripts actuales asumen `.h5` (salida HDF5 nativa de PSC).

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

# Temperatura paralela iónica (T = m * Var(v) = Var(p) / m)
T_par_ions = np.var(pz[ions]) / 200.0
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

### Qué cambiar en los scripts de análisis para `.bp`

| Componente | Cambio necesario |
|---|---|
| `data_reader.py` | Agregar un `PICDataReaderBP` que use `adios2.open()` en vez de `h5py.File()`. No necesita `get_uid_group()` (sin UUID hash). |
| `plot_prt.py` | Cambiar `h5py.File → adios2.open` en `load_particles()` y `load_particle_phase_space()`. La ruta interna pasa de `f["particles"]["p0"]["1d"]["q"]` a `step.read("particles/p0/1d/q")`. |
| `validate_moments.py` | Mismo cambio en `load_particles()`. |
| `anisotropy_analysis.py` | Usar el nuevo reader BP en `process_snapshot()`. |
| `psc_units.py` | Cambiar los patrones: `FIELD_FILE_PATTERN = "pfd.*.bp"`, `PARTICLE_FILE_PATTERN = "prt.*.bp"`. |
| `Makefile` | Actualizar `DATA_DIR` y los globs de `--fields` / `--moments` para buscar `.bp` en vez de `.h5`. |

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
├── data_reader.py            ← LECTOR HDF5 genérico
│   │                            PICDataReader: find_files(), read_multiple_fields_3d()
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
├── spectral_analysis.py      ← ESPECTROS (lee pfd) [en desarrollo]
│   └── PSD 1D y 2D de componentes magnéticas
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
   │         ├── vdf_2d.png                 │         └── brazil_plot_anisotropy.png
   │         ├── kappa_vs_maxwellian.png    │
   │         ├── goodness_of_fit.png        ├── mirror_physics.py
   │         ├── vdf_time_snapshots.png     │     └── mirror_plots/
   │         ├── distribution_evolution.png │
   │         ├── brazil_plot.png            ├── diamagnetic_current.py
   │         ├── vdf_1d_evolution.png       │     └── diamagnetic_plots/
   │         ├── energy_partition.png       │
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
| `NICELL`       | 2000          | 2000             | Partículas por celda                 |
| `BETA_I_PAR`   | 5.0           | 10.0             | Beta paralelo iónico                 |
| `TI_PAR`       | 0.00625       | 0.0125           | Temperatura iónica paralela          |
| `TI_PERP`      | 0.01875       | 0.00125          | Temperatura iónica perpendicular     |
| `Ti_⊥/Ti_∥`    | 3.0           | 0.1              | Anisotropía iónica                   |
| `KAPPA`        | `3.0`/`None`  | `3.0`/`None`     | Kappa ó Maxwellian                   |

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
v_va = (p / m) / VA
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
Salida: prt_plots/vdf_1d_evolution.png
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
Salida: prt_plots/energy_partition.png
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
