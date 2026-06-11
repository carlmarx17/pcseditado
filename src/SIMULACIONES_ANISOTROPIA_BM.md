# Scan Bi-Maxwelliano — Inestabilidades por Anisotropía de Temperatura

Nueve simulaciones PIC 2D de plasma homogéneo con campo magnético uniforme
**B₀ ẑ**. Cada inestabilidad crece espontáneamente del ruido térmico de las
partículas — no se impone perturbación externa.

---

## ⚙️ Configuración común

```
┌─────────────────────────────────────────────────────────┐
│  Razón de masas          mᵢ/mₑ = 200                   │
│  Velocidad de Alfvén     vA/c  = 0.05    →  B₀ = 0.05  │
│  Partículas por celda    ppc   = 2000                   │
│  Densidad                n     = 1.0                    │
│  Carga iónica            Zᵢ    = 1                      │
│  CFL                           = 0.95                   │
│  Pasos totales           nmax  = 1,650,000              │
│  Fronteras                     = Periódicas (todo)      │
│  Distribución                  = Bi-Maxwelliana         │
│  Código PIC              PscConfig1vbecSingle<dim_yz>   │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 Dominio y resolución

```
┌────────────────────────────────────────────────────────────────────┐
│  Dominio físico     20 dᵢ × 20 dᵢ  =  282.8 × 282.8  (código)   │
│  Grilla             1408 × 1408     (dim_yz, x invariante)        │
│  Patches MPI        8 × 8 = 64                                    │
│                                                                    │
│  dᵢ = √(mᵢ/mₑ) = √200 ≈ 14.14   (inercia iónica)                │
│  dₑ = 1.0                         (inercia electrónica)           │
│                                                                    │
│  dx = 282.8 / 1408 ≈ 0.201                                        │
│  dx / dₑ    ≈ 0.20   ← resuelve escala electrónica ✓              │
│  dx / λ_De  ≈ 5.68   ← Debye NO resuelta (aceptable con 2k ppc)  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📸 Diagnósticos de salida

```
┌──────────────────────────────────────────────────────────────────┐
│  Campos (pfield)       cada 690 pasos                            │
│  Momentos (pfield)     cada 690 pasos                            │
│  tfield                desactivado                               │
│  Partículas            cada 1000 pasos                           │
│  Energías              cada 100 pasos                            │
│                                                                  │
│  Región de partículas: central 20% del dominio por eje           │
│  [0.4×1408, 0.6×1408] = [563, 844] en cada eje                  │
│  → Solo guarda ~4% del total de partículas por snapshot          │
└──────────────────────────────────────────────────────────────────┘
```

> Las partículas se guardan **solo de la región central** para no saturar
> el disco. Los campos y momentos instantáneos cubren todo el dominio mediante
> `pfield`; la salida promediada `tfield` está desactivada.

---

## 🪞 Mirror — Iones anisotrópicos (Tᵢ⊥ > Tᵢ∥)

Electrones isotrópicos en todos los casos: βₑ∥ = 1.0, Aₑ = 1.0

```
Condición de inestabilidad:  βᵢ⊥ − βᵢ∥ > 1
```

| | **M-S-bM** | **M-M-bM** | **M-W-bM** |
|---|:---:|:---:|:---:|
| Régimen | Strong | Moderate | Weak |
| Archivo | `psc_M_S_bM.cxx` | `psc_M_M_bM.cxx` | `psc_M_W_bM.cxx` |
| βᵢ∥ | 5.0 | 5.0 | 6.0 |
| Aᵢ = Tᵢ⊥/Tᵢ∥ | **3.0** | **2.0** | **1.5** |
| Tᵢ,par | 6.25×10⁻³ | 6.25×10⁻³ | 7.50×10⁻³ |
| Tᵢ,perp | 1.875×10⁻² | 1.25×10⁻² | 1.125×10⁻² |
| βᵢ⊥ − βᵢ∥ | **10.0** ✓ | **5.0** ✓ | **3.0** ✓ |

---

## 🔥 Firehose — Iones anisotrópicos (Tᵢ∥ > Tᵢ⊥)

Electrones isotrópicos en todos los casos: βₑ∥ = 1.0, Aₑ = 1.0

```
Condición de inestabilidad:  βᵢ∥ − βᵢ⊥ > 2
```

| | **F-S-bM** | **F-M-bM** | **F-W-bM** |
|---|:---:|:---:|:---:|
| Régimen | Strong | Moderate | Weak |
| Archivo | `psc_F_S_bM.cxx` | `psc_F_M_bM.cxx` | `psc_F_W_bM.cxx` |
| βᵢ∥ | 10.0 | 6.0 | 3.0 |
| Aᵢ = Tᵢ⊥/Tᵢ∥ | **0.1** | **0.3** | **0.6** |
| Tᵢ,par | 1.25×10⁻² | 7.50×10⁻³ | 3.75×10⁻³ |
| Tᵢ,perp | 1.25×10⁻³ | 2.25×10⁻³ | 2.25×10⁻³ |
| βᵢ∥ − βᵢ⊥ | **9.0** ✓ | **4.2** ✓ | **1.2** ~ |

> F-W-bM es marginal (1.2 < 2). Puede tardar mucho en desarrollar la
> inestabilidad o mostrar solo crecimiento muy lento.

---

## 🌊 Whistler — Electrones anisotrópicos (Tₑ⊥ > Tₑ∥)

Iones isotrópicos en todos los casos: βᵢ∥ = 1.0, Aᵢ = 1.0

```
Anisotropía electrónica → opera a escala dₑ
La resolución dx ≈ 0.2 dₑ es crítica para estos casos
```

| | **W-S-bM** | **W-M-bM** | **W-W-bM** |
|---|:---:|:---:|:---:|
| Régimen | Strong | Moderate | Weak |
| Archivo | `psc_W_S_bM.cxx` | `psc_W_M_bM.cxx` | `psc_W_W_bM.cxx` |
| βₑ∥ | 0.5 | 0.5 | 0.5 |
| Aₑ = Tₑ⊥/Tₑ∥ | **3.0** | **2.0** | **1.5** |
| Tₑ,par | 6.25×10⁻⁴ | 6.25×10⁻⁴ | 6.25×10⁻⁴ |
| Tₑ,perp | 1.875×10⁻³ | 1.25×10⁻³ | 9.375×10⁻⁴ |
| Tᵢ (isotrópico) | 1.25×10⁻³ | 1.25×10⁻³ | 1.25×10⁻³ |

---

## 💾 Memoria RAM

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Partículas:  2 × 1408² × 2000 × 28 B  =  207 GB           │
│  Campos EM:   1408² × 12 × 4 B         =    0.1 GB          │
│  Buffers MPI + sort + overhead          ≈   15 GB            │
│  ────────────────────────────────────────────────             │
│  TOTAL POR CASO                         ≈  222 GB            │
│                                                              │
│  (28 B/partícula = 7 floats: x,y,z, px,py,pz, w)            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### ¿Dónde correr?

| Nodo | RAM | ¿Cabe? | Uso recomendado |
|---|---|---|---|
| **pauli** | 247 GB | ✅ (25 GB margen) | Mirror |
| **planck** | 248 GB | ✅ (26 GB margen) | Firehose |
| **nodo-00** | 514 GB | ✅ (caben 2) | Whistler |
| feynman-00 | 126 GB | ❌ | — |
| maxwell | 64 GB | ❌ | — |

---

## 🔢 Fórmula de temperaturas

Todas las temperaturas se derivan de β y B₀:

```
T_par  = β_par × B₀² / 2
T_perp = A × T_par

Ejemplo M-S-bM:
  Ti_par  = 5.0 × 0.05²/2  = 6.25×10⁻³
  Ti_perp = 3.0 × 6.25×10⁻³ = 1.875×10⁻²
  Te_par  = 1.0 × 0.05²/2  = 1.25×10⁻³
  Te_perp = 1.0 × 1.25×10⁻³ = 1.25×10⁻³
```

En el código: `T[0]`, `T[1]` = perpendicular (x, y) · `T[2]` = paralelo (z, a lo largo de B₀)

---

## 🚀 Script SLURM genérico

```bash
#!/bin/bash
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=pauli
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --job-name=M_S_bM
#SBATCH --output=%x_%j.out

mpirun -np 64 --bind-to core ./psc_M_S_bM
```

> Cambiar `--nodelist`, `--job-name` y el ejecutable para cada caso.
