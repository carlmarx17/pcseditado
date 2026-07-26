# PSC — Inestabilidades por anisotropía de temperatura

Fork de [PSC](https://github.com/psc-code/psc) (Particle-in-Cell, Kai Germaschewski
et al.) usado como base numérica de una tesis de Maestría en Física de la
Universidad Nacional de Colombia.

## Objetivo

Determinar cómo cambia el desarrollo de las inestabilidades por anisotropía de
temperatura en un plasma sin colisiones cuando la distribución de velocidades
no es Maxwelliana.

Concretamente, se simulan con PIC cinético las tres familias de inestabilidad
que regulan la anisotropía en plasmas espaciales —**mirror**, **firehose** y
**whistler**— partiendo de dos distribuciones iniciales con los *mismos*
parámetros macroscópicos (β, A = T⊥/T∥, mi/me):

- **bi-Maxwelliana**, el caso de referencia;
- **bi-Kappa** (κ = 3, 5), con colas supratérmicas.

La pregunta es qué se mueve al cambiar la distribución: la tasa de crecimiento
lineal γ, el espectro de modos excitados k(γ_max), el nivel de saturación de
δB, y el estado final al que el plasma relaja en el plano (β∥, A) —el
"diagrama de Brasil". El repositorio contiene tanto los casos de simulación
como la pipeline de análisis que produce esos diagnósticos.

## Qué se incorporó sobre PSC

El núcleo de PSC (solver de campos, pusher de partículas, I/O, balanceo) es
**upstream sin modificar**. Lo propio de este repositorio es:

| Componente | Dónde | Qué aporta |
|---|---|---|
| Caso base parametrizado | `src/psc_anisotropy_case.hxx` | Plantilla común a todos los casos de anisotropía: geometría, salidas, intervalos y overrides por entorno. |
| Distribución bi-Kappa | `src/psc_anisotropy_case.hxx` (`PSC_USE_KAPPA`, `PSC_KAPPA`) | Muestreo de una bi-Kappa anisotrópica como condición inicial, alternativa a la bi-Maxwelliana de PSC. |
| 15 casos de inestabilidad | `src/psc_{mirror,firehose,whistler}_*.cxx` | Un ejecutable por régimen físico; cada archivo solo fija etiqueta, distribución y parámetros. |
| Checkpoint/restart ADIOS2 | `src/cosma_*.sh`, `adios2cfg.xml` | Corridas largas resumibles (BP5), necesarias para llegar a saturación en HPC. |
| Pipeline de análisis | `CodeforAnalisys/` | Post-procesamiento completo: anisotropía, espectros, dispersión, tasas de crecimiento, diagnósticos físicos. |
| Scripts de cluster | `cosma_jobs/` | Jobs SLURM de simulación y de análisis, con nombres que dicen qué corren. |
| Resultados curados | `analysis_results/` | Salidas versionadas de las corridas terminadas (cadencia de 100k pasos). |

## Estructura del repositorio

```text
src/                    Código PSC + casos de anisotropía (.cxx) y docs de física
CodeforAnalisys/        Pipeline de análisis (Python + Makefile)
CodeforAnalisysLocal/   Variante para corridas locales pequeñas
cosma_jobs/             Jobs SLURM: simulacion/ y analisis/
analysis_results/       Salidas curadas por caso
presentation/           Material de presentación
docs/, doxygen/         Documentación del PSC upstream
python/, matlab/, ...   Utilidades upstream
```

## Casos de simulación

Configuración común: `PscConfig1vbecSingle<dim_yz>`, dominio 20 d_i × 20 d_i,
grilla 576×576 (≈28.8 celdas/d_i), mi/me = 200, vA/c = 0.08, fronteras
periódicas, campo de fondo a lo largo de **z** (T∥ = T_z, T⊥ = (T_x+T_y)/2).

### Mirror — iones con exceso perpendicular
Criterio: `β_i∥ (A_i − 1) > 1`

| Ejecutable | Régimen | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_mirror_bimaxwellian_strong` | Strong | 5.0 | 3.0 | 1.0 | 1.0 |
| `psc_mirror_bimaxwellian_moderate` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 |
| `psc_mirror_bimaxwellian_weak` | Weak | 6.0 | 1.5 | 1.0 | 1.0 |

### Firehose — iones con exceso paralelo
Criterio: `β_i∥ (1 − A_i) > 2`

| Ejecutable | Régimen | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_firehose_bimaxwellian_strong` | Strong | 10.0 | 0.1 | 1.0 | 1.0 |
| `psc_firehose_bimaxwellian_moderate` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 |
| `psc_firehose_bimaxwellian_weak` | Weak | 3.0 | 0.6 | 1.0 | 1.0 |

### Whistler — electrones con exceso perpendicular
Criterio: `A_e > 1 + 0.21 / β_e∥^0.6`

| Ejecutable | Régimen | β_i∥ | A_i | β_e∥ | A_e |
|---|---|---:|---:|---:|---:|
| `psc_whistler_bimaxwellian_strong` | Strong | 1.0 | 1.0 | 0.5 | 3.0 |
| `psc_whistler_bimaxwellian_moderate` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 |
| `psc_whistler_bimaxwellian_weak` | Weak | 1.0 | 1.0 | 0.5 | 1.5 |

### Bi-Kappa — contrapartes no-Maxwellianas

| Ejecutable | κ | β_i∥ | A_i | Compara contra |
|---|---:|---:|---:|---|
| `psc_mirror_bikappa3` | 3 | 5.0 | 3.0 | `psc_mirror_bimaxwellian_strong` |
| `psc_mirror_bikappa5` | 5 | 5.0 | 3.0 | `psc_mirror_bimaxwellian_strong` |
| `psc_mirror_bikappa3_moderate` | 3 | 5.0 | 2.0 | `psc_mirror_bimaxwellian_moderate` |
| `psc_firehose_bikappa3` | 3 | 10.0 | 0.1 | `psc_firehose_bimaxwellian_strong` |
| `psc_firehose_bikappa5` | 5 | 10.0 | 0.1 | `psc_firehose_bimaxwellian_strong` |

### Caja grande (40 d_i)

`psc_firehose_bimaxwellian_moderate_bigbox40` y `psc_firehose_bikappa3_bigbox40`
duplican el dominio para dar cabida a los modos firehose de mayor longitud de
onda. El tamaño de caja es **compile-time** (`PSC_DOMAIN_DI` en
`psc_anisotropy_case.hxx`), por eso son ejecutables aparte.

Además de la línea de anisotropía, el repositorio mantiene casos de
**reconexión magnética** (`psc_reconnection*.cxx`), documentados en
`src/SIMULACIONES_RECONNECTION.md`.

## Pipeline de análisis (`CodeforAnalisys/`)

Toma los snapshots de una corrida (HDF5 `.h5` o ADIOS2 `.bp`) y produce los
diagnósticos de la tesis. Orquestado por `Makefile`; el `PSC_PROFILE` fija los
parámetros físicos y la normalización de cada caso (`psc_units.py`).

```bash
cd CodeforAnalisys
make show-inputs DATA_DIR=/ruta/a/la/corrida CASE=mirror_bikappa3_moderate
make analysis    DATA_DIR=/ruta/a/la/corrida CASE=mirror_bikappa3_moderate
```

`make analysis` = `manifest` + las 8 etapas de `common`, que también pueden
correrse por separado:

| Etapa | Script | Produce |
|---|---|---|
| `brazil` | `anisotropy_analysis.py` | Evolución de A(t), β∥(t) y trayectoria en el diagrama de Brasil |
| `fields` | `fluctuationofmagneticfiel.py` | Mapas de δB por snapshot y GIFs de la evolución |
| `spectral` | `spectral_analysis.py` | E(k, Ω_ci t), γ(k) log-lineal por capa, helicidad σ_m(k), compresibilidad |
| `dispersion` | `dispersion_analysis.py` | Relación de dispersión ω(k) y mapa de densidad modal ω/Ω_ci vs \|v_ph\|/v_A |
| `growth-map` | `growth_rate_map.py` | γ(k∥, k⊥) sin binning radial — distingue modos paralelos de oblicuos |
| `polarization` | `polarization_dispersion.py` | Ramas de polarización ± y tasas con gating por R² |
| `diamagnetic` | `diamagnetic_current.py` | Corriente diamagnética (firma de mirror) |
| `heatflux` | `heat_flux_analysis.py` | Flujo de calor paralelo y perpendicular |
| `particles` | `plot_prt.py` | Distribuciones de velocidad de la salida de partículas |
| `validate` | `validate_moments.py` | Consistencia momentos de grilla vs partículas |
| `physics` | `physical_diagnostics.py` | Tablas resumen: energía, anisotropía, ajustes, correlaciones |

Extras: `compare_physical_cases.py` (comparación bi-Maxwelliana vs bi-Kappa),
`check_resolution.py` (verifica que la resolución del perfil coincida con los
datos), y tests con `pytest`.

Cada corrida escribe `analysis_results/<CASE>/` con la estructura
`01_anisotropy/`, `02_fields/`, `04_spectra/`, `05_diamagnetic/`,
`06_heat_flux/`, `09_physical_diagnostics/` y un
`<CASE>_analysis_manifest.json` que registra los parámetros y los pasos
detectados.

## Compilación local

```bash
cmake --build build --target psc_mirror_bimaxwellian_strong
cmake --build build --target psc_mirror_bikappa3
```

## Documentación

| Documento | Contenido |
|---|---|
| `src/SIMULACIONES_ANISOTROPIA.md` | Catálogo completo de casos y sus parámetros físicos |
| `src/SIMULACIONES_INESTABILIDADES_TEMPERATURA_ANALISIS.md` | Criterios físicos y plan de análisis de mirror / firehose / whistler |
| `src/ESCALADO_INESTABILIDADES.md` | Escalado y dimensionamiento de las corridas |
| `src/REFACTOR_KAPPA.md` | Notas del refactor de la distribución bi-Kappa |
| `src/SIMULACIONES_RECONNECTION.md` | Casos de reconexión magnética |
| `src/ADIOS2_COSMA_RUNBOOK.md` | Procedimiento ADIOS2/COSMA (referencia técnica) |
| `CodeforAnalisys/README.md` | Uso diario de la pipeline |
| `CodeforAnalisys/ANALISIS_ESTRUCTURA.md` | Contrato de archivos, lectores y salidas |
| `cosma_jobs/README.md` | Qué hace cada job SLURM |

> La operativa del cluster (acceso, envío de jobs, monitorización, checkpoints,
> problemas frecuentes) se mantiene fuera del repositorio, en Notion:
> **COSMA7 — Runbook operativo**.

## Licencia y créditos

PSC es obra de Kai Germaschewski y colaboradores (UNH, LMU); ver `LICENSE` y el
historial de git. Este fork añade únicamente los casos de anisotropía y la
pipeline de análisis descritos arriba.
