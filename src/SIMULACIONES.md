# Documentación de Simulaciones PSC — Anisotropías

## Visión general

Simulaciones PIC (Particle-In-Cell) de inestabilidades cinéticas **Mirror** y **Firehose**
en plasmas magnetizados. Todas usan `PscConfig1vbecSingle` (Villasenor-Buneman Edge-Centered
1er orden) en 2D (dim_yz), full PIC (iones + electrones cinéticos).

---

## Catálogo de códigos de anisotropía

| Archivo | Inestabilidad | Distribución | mr | Grilla | ppc | vA/c | βᵢ‖ | Tᵢ⊥/Tᵢ‖ | RAM |
|---|---|---|---|---|---|---|---|---|---|
| `psc_mirror_kappa` | Mirror | Kappa (κ=3) | 100 | 1024² | 1000 | 0.05 | 5.0 | 3.0 | ~67 GB |
| `psc_mirror_maxwellian` | Mirror | Maxwellian | 200 | 1536² | 1000 | 0.05 | 5.0 | 3.0 | ~150 GB |
| `psc_mirror_maxwellian_2k` | Mirror | Maxwellian | 200 | 2048² | 1000 | 0.05 | 5.0 | 3.0 | ~250 GB |
| `psc_mirror_maxwellian_pauli` | Mirror | Maxwellian | 200 | 1408² | **2000** | 0.05 | 5.0 | 3.0 | ~222 GB |
| `psc_firehose_kappa` | Firehose | Kappa (κ=3) | 100 | 1024² | 1000 | 0.05 | 10.0 | 0.1 | ~67 GB |
| `psc_firehose_maxwellian` | Firehose | Maxwellian | 100 | 1024² | 1000 | 0.05 | 10.0 | 0.1 | ~67 GB |

> Todos con βₑ‖ = 1.0, Tₑ⊥/Tₑ‖ = 1.0 (electrones isotrópicos), λ₀ = 20, n = 1.

---

## Parámetros físicos comunes

| Parámetro | Símbolo | Valor |
|---|---|---|
| Razón de masas | mᵢ/mₑ | 100–200 |
| Velocidad de Alfvén | vA/c | 0.05 |
| Campo magnético | B₀ | 0.05 |
| Densidad | n | 1.0 |
| Carga iónica | Zᵢ | 1 |
| Longitud de Coulomb | λ₀ | 20 |
| CFL | — | 0.95 |

### Cálculo de λ_De

```
Te_par = β_e‖ × B0²/2 = 1.0 × 0.05²/2 = 0.00125
λ_De   = √Te_par = √0.00125 ≈ 0.0354
```

Este valor es más realista para un plasma no relativista porque `vA/c = 0.05`
mantiene la velocidad de Alfvén claramente por debajo de la velocidad de la luz.
La consecuencia numérica es que `λ_De` queda más pequeña y por eso no se
resuelve con la grilla actual. Ese compromiso es aceptable aquí porque el
objetivo principal es la dinámica cinética de Mirror/Firehose a escala de
`d_e` y `d_i`, no resolver oscilaciones electrostáticas Debye.

### Condiciones de inestabilidad

**Mirror** requiere Tᵢ⊥ > Tᵢ‖:
```
βᵢ⊥ - βᵢ‖ > 1  →  5.0×(3.0 - 1) = 10.0  ✓
```

**Firehose** requiere Tᵢ‖ > Tᵢ⊥:
```
βᵢ‖ - βᵢ⊥ > 2  →  10.0×(1 - 0.1) = 9.0  ✓
```

---

## Resolución de grilla y Debye

| Código | mr | dᵢ | Dominio | dx | dx/dₑ | dx/λ_De | ¿Resuelve Debye? |
|---|---|---|---|---|---|---|---|
| `mirror_kappa` | 100 | 10.0 | 200 | 0.195 | 0.20 ✅ | **5.52** | No |
| `mirror_maxwellian` | 200 | 14.14 | 282.8 | 0.184 | 0.184 ✅ | **5.21** | No |
| `mirror_maxwellian_2k` | 200 | 14.14 | 282.8 | 0.138 | 0.14 ✅ | **3.90** | No |
| `mirror_maxwellian_pauli` | 200 | 14.14 | 282.8 | 0.201 | 0.20 ✅ | **5.68** | No |
| `firehose_kappa` | 100 | 10.0 | 200 | 0.195 | 0.20 ✅ | **5.52** | No |
| `firehose_maxwellian` | 100 | 10.0 | 200 | 0.195 | 0.20 ✅ | **5.52** | No |

> Nota: con `vA/c = 0.05`, la longitud de Debye queda más sub-resuelta
> (`dx/λ_De ~ 4-6`). Por eso estas corridas deben interpretarse como
> simulaciones orientadas a escalas inerciales y a crecimiento Mirror/Firehose,
> no como estudios de física Debye fina. El ruido numérico se reduce con alto
> ppc (1000-2000), chequeos de Gauss/Marder y diagnósticos de energía.

---

## Capacidad actual del cluster

Esta lectura viene de `squeue`, `sinfo` y `scontrol show nodes` tomada el
27 de mayo de 2026. La dejo en la documentación porque para mi tesis no basta
con tener una simulación interesante: también necesito que el caso pueda
terminar sin que SLURM lo mate por memoria o por pedir más núcleos de los
necesarios.

### Nodos IDLE útiles para estas simulaciones

| Nodo | CPUs | RAM | Ideal para |
|---|---|---|---|
| **pauli** | 64 | 247 GB | 1408² (222 GB), 1536² (150 GB), 1024² (67 GB) |
| **planck** | 64 | 248 GB | Ídem pauli |
| **feynman-00** | 72 | 126 GB | 1024² (67 GB) |
| **maxwell** | 64 | 64 GB | Solo pruebas controladas; 1024² queda demasiado justo |
| **hercules7** | 32 | 62 GB | No recomendado para producción 1024² |
| **egeo-016** | 64 | 63 GB | No recomendado para producción 1024² |
| **hercules3** | 16 | 52 GB | Solo versiones pequeñas o debug |

### Nodos MIX

| Nodo | Estado observado | Ideal para |
|---|---|---|
| **nodo-00** | mixed, 128 CPUs y 514 GB totales, con muchos jobs de otro usuario | 1536² o 2048² si SLURM da recursos |
| **nodo-01** | mixed, 256 CPUs y 514 GB totales, algunos jobs activos y GPU MI210 | 1536² o 2048² si SLURM da recursos |

### Nodos que no debo usar para producción ahora

| Nodo | Motivo |
|---|---|
| **boltzmann**, **hercules6**, **egeo-020..024**, **logcecc**, **hubble** | `down*` / no responden |
| **hercules1** | `inval` / mantenimiento |
| **hercules2**, **hercules4**, **nas2** | `drain` |
| **hercules5** | `allocated`, casi sin memoria libre en la lectura |

> Nota importante: aunque el nodo se llame **maxwell**, no es el mejor para las
> simulaciones Maxwellian de producción. Tiene ~64 GB y los casos 1024² están
> estimados en ~67 GB antes del overhead de PSC, sort, buffers y salida. Para no
> arriesgar OOM, lo dejo como nodo de prueba pequeña, no como nodo de tesis.

---

## Recomendación para el primer objetivo de tesis

Mi primer objetivo debe priorizar una simulación **Maxwellian** porque funciona
como caso base: si luego comparo contra Kappa, necesito primero una referencia
Maxwellian bien documentada, estable y reproducible. Con la capacidad actual,
el mejor equilibrio entre ciencia, tiempo y seguridad de memoria es:

### Ganador: `psc_firehose_maxwellian` en `feynman-00`

| Aspecto | Evaluación |
|---|---|
| **Por qué este código** | Es Maxwellian, entonces sirve como línea base directa para la tesis |
| **RAM** | ~67 GB; en `feynman-00` puedo pedir 110-120 GB y dejar margen |
| **CPUs** | 32 parches; correr con 32 MPI ranks es suficiente y evita pedir 64 sin necesidad |
| **Costo** | 1024², 1000 ppc, 1.2M pasos; es el Maxwellian más razonable para iniciar |
| **Física** | Firehose crece más rápido que Mirror, así que permite validar antes la evolución |
| **Riesgo** | Mucho menor que `mirror_maxwellian`, `mirror_maxwellian_pauli` o `mirror_maxwellian_2k` |

### Recomendación ordenada (mejor a peor):

1. **`psc_firehose_maxwellian`** → mejor primer caso Maxwellian. Usar `feynman-00`, 32 ranks, 110-120 GB. Es el más seguro para no morir por RAM.
2. **`psc_mirror_maxwellian`** → segundo objetivo Maxwellian. Usar `pauli` o `planck`, 64 ranks, 180-220 GB. Es más pesado y más lento, pero más cercano al caso Mirror principal.
3. **`psc_mirror_maxwellian_pauli`** → solo si `pauli` o `planck` están completamente libres. Pide ~222 GB; dejar margen con `--mem=235G` o más.
4. **`psc_mirror_maxwellian_2k`** → no lo usaría primero. Necesita nodos grandes (`nodo-00`/`nodo-01`) y puede tardar demasiado para una primera validación.
5. **Kappa (`psc_firehose_kappa`, `psc_mirror_kappa`)** → importante como comparación posterior, pero no debe reemplazar la línea base Maxwellian del primer objetivo.

### Script recomendado para empezar ahora

```bash
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=feynman-00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=115G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=firehose_max

mpirun -np 32 --bind-to core ./psc_firehose_maxwellian
```

### Alternativa si quiero Mirror Maxwellian como primer resultado

```bash
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --nodelist=pauli
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=220G
#SBATCH --time=14-00:00:00
#SBATCH --job-name=mirror_max

mpirun -np 64 --bind-to core ./psc_mirror_maxwellian
```

> Regla práctica: para no matar el job por memoria, pedir siempre más memoria
> que la estimación. Para ~67 GB pido 110-120 GB; para ~150 GB pido 200-220 GB;
> para ~222 GB pido 235-240 GB. No usar `--exclusive` si solo necesito 32 ranks
> y el scheduler permite compartir nodo, porque puede aumentar la espera.

> Recomendación concreta: ejecutar primero `psc_firehose_maxwellian` en
> `feynman-00` con 32 ranks y 115 GB. Después, si ese caso valida bien energía,
> campo y crecimiento de la inestabilidad, correr `psc_mirror_maxwellian` en
> `pauli` o `planck` con 64 ranks y 220 GB.
