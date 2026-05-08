# Documentación de Simulaciones PSC

## Visión general

Este directorio contiene cuatro simulaciones PIC (Particle-In-Cell) para estudiar
inestabilidades cinéticas en plasmas magnetizados. Todas usan el integrador
`PscConfig1vbecSingle` (1st-order Villasenor-Buneman Edge-Centered), que es
**full PIC**: tanto iones como electrones son partículas cinéticas.

| Archivo | Inestabilidad | Distribución |
|---|---|---|
| `psc_mirror_kappa.cxx` | Mirror | Kappa (κ=3) |
| `psc_mirror_maxwellian.cxx` | Mirror | Maxwellian |
| `psc_firehose_kappa.cxx` | Firehose | Kappa (κ=3) |
| `psc_firehose_maxwellian.cxx` | Firehose | Maxwellian |

---

## Parámetros físicos

### Parámetros comunes a las 4 simulaciones

| Parámetro | Símbolo | Valor | Descripción |
|---|---|---|---|
| Razón de masas | `mᵢ/mₑ` | 200 | Masa artificial (real ≈ 1836) |
| Velocidad de Alfvén | `vA/c` | 0.05 | Campo magnético de fondo |
| Campo magnético | `B₀` | 0.05 | `B₀ = vA/c` en unidades PSC |
| Densidad | `n` | 1.0 | Densidad de referencia |
| Carga iónica | `Zᵢ` | 1 | Singly ionized |
| Longitud de Coulomb | `λ₀` | 20 | Parámetro de colisiones |
| ppc | `nicell` | 2000 | Partículas por celda por especie |

### Parámetros de anisotropía

| Parámetro | Mirror | Firehose | Descripción |
|---|---|---|---|
| `βᵢ‖` | **5.0** | **10.0** | Beta iónico paralelo |
| `Tᵢ⊥/Tᵢ‖` | **3.0** | **0.1** | Anisotropía iónica |
| `βₑ‖` | 1.0 | 1.0 | Beta electrónico paralelo |
| `Tₑ⊥/Tₑ‖` | 1.0 | 1.0 | Electrones isotrópicos |

La **inestabilidad mirror** requiere `Tᵢ⊥ > Tᵢ‖` (exceso de energía perpendicular):

```
Condición mirror: βᵢ⊥ - βᵢ‖ > 1  →  βᵢ‖(Tᵢ⊥/Tᵢ‖ - 1) > 1
                  5.0 × (3.0 - 1) = 10.0  ✓ inestable
```

La **inestabilidad firehose** requiere `Tᵢ‖ > Tᵢ⊥` (exceso de energía paralela):

```
Condición firehose: βᵢ‖ - βᵢ⊥ > 2  →  βᵢ‖(1 - Tᵢ⊥/Tᵢ‖) > 2
                    10.0 × (1 - 0.1) = 9.0  ✓ inestable
```

---

## Parámetros de grilla y resolución

### Jerarquía de escalas de longitud

En PSC, las unidades se definen como `c = 1`, `ωₚₑ = 1`, `mₑ = 1`, lo que implica:

| Escala | Símbolo | Valor en unidades de código | Valor en d_e |
|---|---|---|---|
| Skin depth electrónica | `dₑ = c/ωₚₑ` | **1.0** (por definición) | 1.0 |
| Skin depth iónica | `dᵢ = c/ωₚᵢ` | `√(mᵢ/mₑ) = √200` | **≈ 14.14** |
| Longitud de Debye | `λ_De = vₜₑ/ωₚₑ` | `√(Tₑ‖) = √(βₑ B₀²/2)` | **≈ 0.035** |

La jerarquía completa:
```
λ_De ≈ 0.035 d_e  ←  sub-resuelta (inevitable con masa artificial)
d_e  = 1.000 d_e  ←  ✅ RESUELTA (Δx = 0.59 d_e)
d_i  = 14.14 d_e  ←  ✅ bien resuelta (Δx = 0.042 d_i)
```

### Configuración de grilla (igual para los 4 perfiles)

| Parámetro | Valor | Descripción |
|---|---|---|
| Dominio | 32 × 32 dᵢ | Extensión física en unidades iónicas |
| Dominio en d_e | 452.5 × 452.5 d_e | Equivalente electrónico |
| Grid | **768 × 768** | Número de celdas por dimensión |
| **Δx** | **0.589 d_e = 0.042 dᵢ** | Resolución espacial |
| Δx < dₑ | ✅ Sí | Condición exigida por el director |
| Δx/λ_De | ≈ 16.7 | λ_De sub-resuelta (esperado) |
| Parches MPI | `np = {1, 8, 4}` | 32 parches para 32 cores |
| Celdas/parche | 96 × 192 | Balance de carga uniforme |
| RAM estimada | **~75.5 GB** | 24.5 GB libres en servidor 100 GB |

> **¿Por qué 768?** 768 = 32 × 24, divisible exactamente por 32 parches MPI.
> Es el mayor grid que cabe en 100 GB con Δx < d_e y sin comprometer el OS.

### Comparación de resoluciones históricas

| Configuración | Grid | Δx [d_e] | RAM | Δx < dₑ |
|---|---|---|---|---|
| Original (v1) | 128² | 3.54 | 2 GB | ❌ |
| Intermedia (v2) | 448² | 1.01 | 26 GB | ❌ |
| **Actual (v3)** | **768²** | **0.59** | **75 GB** | **✅** |
| Con 0.25 d_e (inviable) | 1810² | 0.25 | ~1.5 TB | ✅ |

---

## Parámetros temporales

| Parámetro | Valor | Descripción |
|---|---|---|
| `CFL` | 0.95 | Condición de Courant-Friedrichs-Lewy |
| `dt` | ≈ 0.396 | `CFL × Δx / √2` (en unidades código) |
| `nmax` | **600,000** | Pasos de integración totales |
| `t_max` | **59.4 Ω_cᵢ⁻¹** | Tiempo físico máximo |
| `Ωᵢ` | `ZᵢB₀/mᵢ = 0.05/200` | Frecuencia ciclotrónica iónica |
| Checkpoints | cada 5,000 pasos | Para reanudación de simulaciones |

> **¿Por qué nmax = 600,000?** El timestep es ~6× más pequeño que la configuración
> original (128², dt ≈ 2.375). Para alcanzar el mismo tiempo físico (59.4 Ω_ci⁻¹)
> se necesitan 6× más pasos: 100,000 × 6 ≈ 600,000.

---

## Salida de datos

### Campos y momentos (`outf`)

| Parámetro | Valor |
|---|---|
| `pfield.out_interval` | 500 pasos |
| `tfield.out_interval` | 500 pasos |
| `tfield.average_every` | 100 pasos |
| Formato | HDF5 (WriterDefault) |

Con `nmax = 600,000` y salida cada 500 pasos → **1,200 snapshots** de campos.

### Partículas (`outp`)

Las partículas se guardan solo de la región central `8 × 8 dᵢ`:

| Grid | Región central | Celdas guardadas | Fracción |
|---|---|---|---|
| 768² | [288, 480] × [288, 480] | 192 × 192 | 6.25% |

Esto reduce el storage de partículas de ~TB a manejable (~62 GB total).

### Energías (`oute`)

Diagnóstico de energías cada 100 pasos (ligero, siempre activo).

---

## Distribuciones de velocidad

### Maxwellian

Distribución gaussiana estándar. Función de distribución:
```
f(v) ∝ exp(-v²/2T)
```

### Kappa (κ = 3)

Distribución con colas en ley de potencia, típica de plasmas espaciales:
```
f(v) ∝ (1 + v²/(κ·Tₑff))^(-(κ+1))
```
con κ = 3 (colas supra-térmicas pronunciadas, característica del viento solar).

Las temperaturas efectivas se ajustan para que `⟨v²⟩` sea el mismo que en la Maxwellian:
```
Tₑff = T × (κ - 3/2) / κ   para κ > 3/2
```

---

## Colisiones Coulomb

El operador de colisiones se activa con `collision_interval = -10`
(negativo = fracción de pasos, no periódico fijo).

```cpp
collision_nu = 3.76 × Tₑ‖² / (Zᵢ × λ₀)
```

---

## Corrección de Marder

Para mantener la condición `∇·E = ρ/ε₀` numéricamente:

| Parámetro | Valor |
|---|---|
| `marder_diffusion` | 0.9 |
| `marder_loop` | 3 |
| `marder_interval` | 100 pasos |
| `gauss.check_interval` | 100 pasos |

---

## Diferencias entre los 4 archivos

| Archivo | `beta_i_par` | `Ti_perp/Ti_par` | Distribución | Inestabilidad activa |
|---|---|---|---|---|
| `psc_mirror_kappa.cxx` | 5.0 | 3.0 | Kappa (κ=3) | Mirror |
| `psc_mirror_maxwellian.cxx` | 5.0 | 3.0 | Maxwellian | Mirror |
| `psc_firehose_kappa.cxx` | 10.0 | 0.1 | Kappa (κ=3) | Firehose |
| `psc_firehose_maxwellian.cxx` | 10.0 | 0.1 | Maxwellian | Firehose |

Las 4 simulaciones tienen **exactamente la misma grilla, dominio, nmax y condiciones
de contorno** — solo difieren en la anisotropía inicial y la forma de la distribución.
Esto permite comparar directamente el efecto de la distribución (Kappa vs Maxwellian)
y el tipo de inestabilidad (Mirror vs Firehose).

---

## Compilación y ejecución

```bash
# Desde el directorio build:
cmake .. -DCMAKE_BUILD_TYPE=Release
make psc_mirror_kappa psc_mirror_maxwellian psc_firehose_kappa psc_firehose_maxwellian -j32

# Ejecución con 32 MPI ranks:
mpirun -n 32 ./psc_mirror_kappa
mpirun -n 32 ./psc_mirror_maxwellian
mpirun -n 32 ./psc_firehose_kappa
mpirun -n 32 ./psc_firehose_maxwellian
```

> **Nota de memoria**: cada simulación requiere ~75 GB de RAM. No correr más de una
> simultáneamente en el servidor de 100 GB.

---

## Referencias

- Hellinger & Trávníček 2008 (JGR 113): Full PIC mirror, mᵢ/mₑ=64, Δ≈4 dₑ
- Riquelme et al. 2015 (ApJ 800): Full PIC, mᵢ/mₑ=100, Δ=1 dₑ
- Kunz et al. 2014 (ApJL 814): Hybrid PIC mirror, Δ=0.5 dᵢ
- Birdsall & Langdon 1991: *Plasma Physics via Computer Simulation* (PIC fundamentos)
