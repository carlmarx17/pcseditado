# Documentación de Simulaciones PSC — Reconexión Magnética

## Visión general

Simulaciones PIC de **reconexión magnética colisionless** con **doble hoja de corriente
Harris** (double Harris current sheet). El plano de reconexión es YZ, con X como dirección
out-of-plane. Todas usan `PscConfig1vbecSingle` en 2D (dim_yz), full PIC,
condiciones de frontera **totalmente periódicas**.

Referencia base para ideas de plasmoides y fluctuaciones: Agudelo Rueda et al.,
ApJ 971, 109 (2024), doi:10.3847/1538-4357/ad5e73.

---

## Catálogo de códigos de reconexión

| Archivo | Distribución | Grilla | ppc | nmax | MPI ranks | RAM | Uso |
|---|---|---|---|---|---|---|---|
| `psc_reconnection` | Kappa (κ=3) | 256×512 | 100 | 10M | 8 | ~30 GB | Producción |
| `psc_reconnection_local` | Maxwellian | 64×256 | 10 | 5,000 | 1 | ~1 GB | PC local |
| `psc_reconnection_mini` | Maxwellian | 32×128 | 4 | 200 | 1 | ~50 MB | Test rápido |

> Todos con mᵢ/mₑ = 25, ωpe/ωce = 2, Ti/Te = 5, Ly = 25.6 dᵢ, Lz = 51.2 dᵢ.

---

## ¿Qué es la reconexión magnética?

La reconexión magnética es el proceso por el cual líneas de campo magnético
antiparalelas se "rompen" y se reconectan, convirtiendo energía magnética en
energía cinética y térmica del plasma. Es fundamental en:
- Erupciones solares
- Tormentas geomagnéticas
- Calentamiento de la corona solar
- Disipación de turbulencia en el viento solar

### Harris current sheet

El equilibrio de Harris es una solución analítica donde una hoja de corriente
separa regiones con campo magnético antiparalelo:

```
B_z(y) = B₀ · tanh(y / L)          ← campo magnético
n(y)   = n₀ · sech²(y / L) + n_bg  ← densidad de plasma
J_x(y) = (B₀/μ₀L) · sech²(y / L)  ← corriente out-of-plane
```

La corriente es sostenida por partículas que driftan en la dirección x
(out-of-plane): iones en +x, electrones en −x.

---

## ¿Por qué doble hoja de corriente?

Jefferson indicó: *"si quieres que sea periódico necesitas dos hojas de corriente"*.

Con **una sola hoja** + BCs periódicas en Y, el campo magnético tiene
una discontinuidad en los bordes (pasa de +B₀ en un borde a +B₀ en el otro,
sin reversión). Con **dos hojas antiparalelas**, el campo se cierra de forma
continua a través de los bordes periódicos:

```
      ┌────────────────────────────────────────────┐
      │  B→    hoja 1     B←     hoja 2    B→      │
      │ +B₀  ──── 0 ──── −B₀  ──── 0 ──── +B₀     │
      │      y=−Ly/4            y=+Ly/4            │
      └──────────────── periódico ─────────────────┘
```

El campo de la doble hoja es:

```
B_z(y) = B₀ · [tanh((y + Ly/4)/L) − tanh((y − Ly/4)/L) − 1]
```

Y la densidad:

```
n(y) = n₀ · [sech²((y + Ly/4)/L) + sech²((y − Ly/4)/L)] + n_bg
```

---

## Balance de presión total

Jefferson enfatizó: *"asegurarse que P_total = Pᵢ + Pₑ + P_m = constante"*.

En un corte vertical (dirección y), la presión total debe ser constante
**sin incluir la perturbación**:

```
P_total(y) = n(y)·(Tᵢ + Tₑ) + B(y)²/(2μ₀) = constante
```

### Verificación numérica

El código imprime `P_total(y)` en 6 puntos del corte vertical al inicio:

| Posición | n_Harris | |B| | P_plasma | P_mag | P_total |
|---|---|---|---|---|---|
| y = −Ly/4 (centro hoja 1) | ~2.0 | ~0 | alto | ~0 | P₀ |
| y = 0 (entre hojas) | ~0 | ~B₀ | bajo | alto | P₀ |
| y = +Ly/4 (centro hoja 2) | ~2.0 | ~0 | alto | ~0 | P₀ |

El error debe ser < 1% en todos los puntos. Si no, hay un bug.

### ¿Cómo se logra el balance?

La relación Harris `n₀(Tᵢ + Tₑ) = B₀²/(2μ₀)` se satisface automáticamente
por la fórmula de temperatura:

```cpp
TTe = me·c² / (2·ε₀·(ωpe/ωce)²·(1 + Ti/Te))
TTi = TTe · Ti/Te
```

Las temperaturas del fondo son iguales a las de Harris (`Tib_Ti = 1, Teb_Te = 1`)
para que la presión del fondo `n_bg·(Tᵢ + Tₑ)` sea una constante aditiva
que no rompe el equilibrio.

---

## Perturbación inicial

Jefferson indicó: *"usualmente se pone una perturbación en el centro para que
empiece la reconexión. El balance de presión se calcula sin poner la perturbación."*

La perturbación se aplica **solo en la hoja de y = +Ly/4** usando un potencial
vectorial que garantiza ∇·B = 0 por construcción:

```
δA_x = ε · cos(k_z·(z − Lz/2)) / cosh((y − Ly/4) / σ)

δB_y = ∂(δA_x)/∂z   → perturbación en B_y
δB_z = −∂(δA_x)/∂y  → perturbación en B_z  (∇·B = 0 garantizado)
```

Con `σ = L` (= grosor de la hoja) y `ε = 0.03·B₀·σ` (3% de B₀).
La envolvente `1/cosh` hace que la perturbación decaiga exponencialmente
lejos de la hoja perturbada, dejando la otra hoja sin perturbar.

---

## Ideas tomadas de Agudelo Rueda et al. 2024

El artículo estudia cómo fluctuaciones magnéticas inducidas, parecidas a
turbulencia, modifican una hoja de corriente Harris y la formación de
plasmoides. La idea central para mi simulación es esta: no basta con iniciar
la reconexión; debo distinguir si las islas magnéticas crecen por tearing normal
o si el sistema queda dominado por fluctuaciones pequeñas que rompen la hoja sin
formar plasmoides coherentes.

### Diferencias con nuestro caso actual

| Punto | Artículo 2024 | Nuestro `psc_reconnection` |
|---|---|---|
| Plasma | Par electrón-positrón, `m_i/m_e = 1` | Ión-electrón artificial, `m_i/m_e = 25` |
| Dominio | Una hoja Harris con fronteras conductoras en `y` | Doble hoja Harris para fronteras periódicas |
| Resolución | `dy = dz = 0.11 d_e` | `dy = dz = 0.50 d_e` en producción |
| ppc | 400 | 100 |
| Grosor de hoja | `Delta = 4 d_e` | `L = 0.5 d_i = 1.25 d_e` |
| Forzamiento | Antena de Langevin y modos `k, omega` | Perturbación localizada tipo semilla en una hoja |
| Pregunta científica | Cuándo se suprime tearing/plasmoides | Primero validar reconexión y plasmoides; luego probar fluctuaciones |

Por eso no debo copiar los parámetros literalmente. El artículo sirve como guía
física y de diagnóstico, pero mi caso tiene otra separación de masas, otro
grosor de hoja y otra condición de frontera.

### Qué puedo usar directamente como criterio físico

1. **Caso de control:** una corrida sin fluctuaciones turbulentas fuertes debe
   formar puntos-X, puntos-O, hojas de corriente elongadas y plasmoides.
2. **Perturbación grande de escala larga:** una perturbación tipo pinch central
   puede acelerar la llegada al estado de reconexión, pero no necesariamente
   representa turbulencia.
3. **Fluctuaciones grandes de escala pequeña:** el resultado más importante del
   artículo es que fluctuaciones con longitud de onda comparable o menor que el
   grosor de la hoja (`lambda <= Delta`) y amplitud sobre un umbral crítico
   pueden suprimir el crecimiento de islas magnéticas.
4. **Umbral dependiente de escala:** la amplitud crítica `delta B_c` no es
   universal; depende de la longitud de onda. Escalas más pequeñas pueden afectar
   más directamente las órbitas de partículas.
5. **Energía y distribución de velocidades:** cuando se suprimen plasmoides, la
   energía inyectada tiende a calentar el plasma y a modificar la distribución
   de velocidades en lugar de alimentar islas magnéticas grandes.

### Traducción a nuestro código

El código actual implementa una semilla localizada:

```
delta A_x = epsilon * cos(k_z*(z - Lz/2)) / cosh((y - Ly/4)/sigma)
```

Esto es útil para iniciar reconexión en una hoja concreta. No es todavía una
antena de Langevin. Una extensión futura inspirada en el artículo sería sumar
modos de potencial vectorial fuera del plano:

```
delta A_x(y,z,t) = Re[ sum_j b_j(t) / k_j * exp(i k_j · r) ]
delta B_ext = curl(delta A_x xhat)
delta J_ext = curl(delta B_ext) / mu0
```

y actualizar `b_j(t)` con una fase/fuerza aleatoria, una frecuencia de excitación
`omega_0` y una tasa de decorrelación `gamma_0`. Esa corriente externa debería
sumarse al avance de campos, no a la condición inicial solamente.

### Barrido recomendado para tesis

Para no gastar recursos antes de validar el caso base, el orden razonable es:

1. **Control:** `psc_reconnection` actual con perturbación 3%, sin forzamiento
   turbulento. Confirmar presión, X-point, outflows e islas.
2. **Semilla débil:** bajar `dby_b0` para ver si el tearing aparece
   espontáneamente o si depende demasiado de la semilla.
3. **Pinch central:** probar una perturbación de escala larga tipo artículo para
   acelerar el estado estacionario sin introducir ruido de escala cinética.
4. **Fluctuaciones multi-modo:** implementar después una antena externa con
   modos grandes y amplitud pequeña. Este caso debería parecerse al control si
   el artículo aplica.
5. **Prueba de supresión:** usar modos con `lambda <= L` y amplitud mayor.
   Buscar si baja el crecimiento de `J_x,max` y si dejan de formarse puntos-O.

Para nuestro caso de producción `L = 0.5 d_i = 1.25 d_e`. Entonces una prueba de
supresión inspirada en el artículo debería usar longitudes de onda del orden de
`lambda <= 1.25 d_e`, no solo modos grandes del tamaño del dominio.

---

## Velocidad out-of-plane

Jefferson indicó: *"en la dirección que sale del plano, se debe poner una
velocidad en las partículas"*.

Esta es la **velocidad drift** que sostiene la corriente J_x de cada hoja Harris.
En la doble hoja, los drifts se invierten entre las dos hojas:

```
Hoja 1 (y = −Ly/4):  J_x > 0  →  iones driftan en +x, electrones en −x
Hoja 2 (y = +Ly/4):  J_x < 0  →  iones driftan en −x, electrones en +x
```

Implementación: el drift se pondera por la contribución de densidad local:

```cpp
drift_weight = (n_sheet1 − n_sheet2) / (n_sheet1 + n_sheet2)
// +1 cerca de hoja 1, −1 cerca de hoja 2, 0 entre hojas

v_drift_ion = +2·Tᵢ/(B₀·L) × drift_weight
v_drift_ele = −2·Tₑ/(B₀·L) × drift_weight
```

---

## Parámetros físicos

| Parámetro | Símbolo | Valor | Descripción |
|---|---|---|---|
| Razón de masas | mᵢ/mₑ | 25 | Artificial (real ≈ 1836) |
| Ratio de frecuencias | ωpe/ωce | 2.0 | |
| Razón de temperaturas | Tᵢ/Tₑ | 5.0 | |
| Grosor de hoja | L/dᵢ | 0.5 | Half-thickness |
| Dominio Y | Ly/dᵢ | 25.6 | Separación entre hojas: 12.8 dᵢ |
| Dominio Z | Lz/dᵢ | 51.2 | Espacio para outflows |
| Densidad de fondo | nb/n₀ | 0.20 | 20% del pico Harris |
| Guide field | bg | 0.0 | Anti-parallel (sin guide) |
| Perturbación | δBy/B₀ | 0.03 | 3% solo en una hoja |

### Cantidades derivadas (en unidades de código)

```
dᵢ = √(mᵢ/mₑ) / (ωpe/ωce) = √25 / 2 = 2.5 dₑ
B₀ = ωce·mₑ/e = 0.5  (en unidades PSC)
Tₑ = 1 / (2·(ωpe/ωce)²·(1+Ti/Te)) = 1/(2·4·6) = 0.02083
Tᵢ = 5·Tₑ = 0.1042
λ_De = √Tₑ ≈ 0.144
```

---

## Resolución y escalas

| Código | Grid | dy/dₑ | dz/dₑ | dy/dᵢ | ppc | Partículas totales |
|---|---|---|---|---|---|---|
| `psc_reconnection` | 256×512 | 0.50 | 0.50 | 0.20 | 100 | ~26M |
| `psc_reconnection_local` | 64×256 | 2.0 | 1.0 | 0.80 | 10 | ~330K |
| `psc_reconnection_mini` | 32×128 | 4.0 | 1.0 | 1.60 | 4 | ~33K |

> **Nota de Jefferson**: si la resolución o el número de partículas es muy bajo,
> la inestabilidad tearing aparece muy rápido (numéricamente, no físicamente).
> `psc_reconnection` (256×512, 100 ppc) está diseñado para evitar esto.

### Comparativa de Resolución con la Literatura y Simulaciones de Anisotropía

En las simulaciones Full PIC, resolver adecuadamente las escalas inerciales (dₑ, dᵢ) y la longitud de Debye (λ_De) es un desafío computacional. A continuación se presenta una comparativa técnica:

| Simulación / Estudio | mᵢ/mₑ | dx/dₑ | dx/dᵢ | dx/λ_De | ppc | Distribución |
|---|---|---|---|---|---|---|
| **Reconexión (Producción)** | 25 | 0.25 | 0.10 | 1.73 | 100 | Kappa / Max |
| **Reconexión (Local)** | 25 | 1.00 | 0.40 | 6.94 | 10 | Maxwellian |
| **Anisotropías (Mirror/Firehose)** | 100-200 | 0.20 | 0.014-0.02 | 1.53 | 1000-2000 | Kappa / Max |
| *Daughton et al. 2011 (Reconexión)* | 25-100 | 0.1-0.2 | ~0.02-0.04 | ~1.0 | ~100 | Maxwellian |
| *Agudelo Rueda et al. 2024* | Pair (1) | 0.1-0.5 | 0.1-0.5 | ~1.0 | 500-1000 | Maxwellian |

**Análisis Científico:**
1. **Resolución de la piel inercial:** En nuestro caso de reconexión de producción (`psc_reconnection`), `dx = 0.25 dₑ`, lo cual es lo suficientemente fino para capturar la física electrónica en la capa de difusión, estando en total concordancia con el estándar de la literatura (ej. Daughton et al., que usa típicamente `dx ~ 0.1-0.2 dₑ`).
2. **Sub-resolución de Debye:** Para nuestro plasma de reconexión, λ_De ≈ 0.144 dₑ, lo que implica `dx/λ_De ≈ 1.73`. Al igual que en nuestras simulaciones de anisotropías (donde `dx/λ_De ≈ 1.53`), la longitud de Debye está ligeramente sub-resuelta. Este compromiso es común y aceptado en simulaciones PIC con masa artificial para poder abarcar un dominio físico (Lz = 51.2 dᵢ) suficientemente grande que permita el desarrollo de los flujos de reconexión (outflows). El ruido de grilla (aliasing) derivado de esto se mitiga mediante el uso de partículas con formas de orden superior y un número de partículas por celda razonable (100 ppc).
3. **Escala Iónica:** La separación de escalas `dᵢ = 5 dₑ` en reconexión (debido a mᵢ/mₑ=25) permite tener `dx = 0.1 dᵢ`. En anisotropías (donde se usan dominios más grandes y masas mayores, mᵢ/mₑ=100-200), se alcanza `dx ≈ 0.02 dᵢ`. Ambos enfoques garantizan una excelente resolución de la física iónica.

---

## Recursos del Cluster y Mejores Opciones

El consumo de memoria de estas simulaciones de reconexión es sustancialmente menor que las de anisotropía, ya que el dominio 2D es mucho más pequeño.

| Ejecutable | Entorno recomendado | CPUs y RAM requeridos | Pasos |
|---|---|---|---|
| `psc_reconnection` | **feynman-00**, **pauli**, o **planck** | 8 CPUs, ~30 GB RAM | 10M |
| `psc_reconnection_local` | PC local o nodo de test (`maxwell`) | 1 CPU, ~1 GB RAM | 5,000 |
| `psc_reconnection_mini` | PC local para debug | 1 CPU, <100 MB RAM | 200 |

**Recomendación de ejecución:**
Para observar el *steady-state* y la formación de plasmoides, ejecutar `psc_reconnection` en el nodo **feynman-00**. Dado que requiere solo 8 ranks y 30 GB de memoria, puede ejecutarse fácilmente incluso si el nodo está parcialmente ocupado, a diferencia de las simulaciones de anisotropía (Mirror/Firehose) que demandan bloques de 64-128 CPUs y >150 GB de RAM.

---

## Condiciones de frontera

```
Campos:     {PERIODIC, PERIODIC, PERIODIC}  en {x, y, z}
Partículas: {PERIODIC, PERIODIC, PERIODIC}  en {x, y, z}
```

Totalmente periódicas. La doble hoja Harris permite esto sin discontinuidades.

---

## Qué observar (steady state)

Jefferson indicó: *"importante que la reconexión alcance el steady state, donde
tienes una hoja de corriente elongada y los outflows son horizontales."*

Al analizar los resultados, buscar:

1. **Hoja elongada**: La hoja perturbada (y = +Ly/4) se adelgaza en el centro
2. **X-point**: Punto donde B_z ≈ 0 y B_y ≈ 0 (el punto de reconexión)
3. **Outflows horizontales**: Jets de plasma saliendo en ±z desde el X-point
4. **Inflows verticales**: Plasma entrando desde ±y hacia el X-point
5. **Islas magnéticas**: Posible formación de plasmoides/islas
6. **Hoja 2 estable**: La hoja no perturbada (y = −Ly/4) debe permanecer estable

### Diagnósticos inspirados en el artículo

Para conectar el análisis con Agudelo Rueda et al. 2024, conviene medir:

1. **`B_y(z,t)` en el centro de la hoja perturbada:** construir un stack plot.
   Patrones alternados azul-rojo indican puntos-X y crecimiento de islas.
2. **`max(|J_x|)` normalizado:** si crecen plasmoides, debe aparecer una fase de
   crecimiento fuerte; si se suprime tearing, `J_x,max` no debe crecer igual.
3. **`max(|B_y|)` y tasa `gamma_By`:** usarlo como proxy del modo de tearing, con
   cuidado de separar crecimiento real de islas y respuesta directa al forcing.
4. **Energía electromagnética, cinética y térmica:** en corridas forzadas la
   energía no tiene por qué conservarse; lo importante es ver si la energía
   alimenta islas o calienta el plasma.
5. **Conteo de puntos-O/puntos-X:** para distinguir reconexión con plasmoides
   de una hoja fragmentada sin islas coherentes.
6. **Distribuciones de velocidad cerca del X-point y outflow:** si el forcing
   suprime islas, buscar aumento térmico y anisotropía en vez de flujo coherente.

---

## Ejecución

### Test rápido
```bash
mpirun -n 1 ../build/src/psc_reconnection_mini
# Debe imprimir "Pressure Balance" con err < 1% y terminar con "Test completed!"
```

### Desarrollo local
```bash
mpirun -n 1 ../build/src/psc_reconnection_local
# Genera pfd.*.h5 y pfd_moments.*.h5 cada 50 pasos
```

### Producción (servidor, 8 MPI ranks, ~30 GB):
```bash
mpirun -n 8 ./psc_reconnection
# 256×512, 100 ppc, Kappa κ=3, 10M pasos
```

---

## Script SLURM para producción

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
echo "Nodo        : ${SLURM_JOB_NODELIST}"
echo "Tareas MPI  : ${SLURM_NTASKS}"
echo "Ejecutable  : ${EXECUTABLE}"
echo "Directorio  : ${WORKDIR}"
echo "Inicio      : $(date)"
echo "=========================================================="

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Ejecutable no encontrado: ${EXECUTABLE}"
    exit 1
fi

mpirun -np 8 \
       --bind-to core \
       ${EXECUTABLE}

EXIT_CODE=$?
echo "=========================================================="
echo "Fin         : $(date)"
echo "mpirun exit : ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
```

---

## Referencias

- Agudelo Rueda, Liu, Germaschewski, Hesse & Bessho 2024 (ApJ 971, 109): "On the Effect of Inducing Turbulence-like Fluctuations in a Harris Current Sheet Configuration and Plasmoid Formation"
- Harris 1962: Original Harris current sheet equilibrium
- Birn et al. 2001 (JGR 106): GEM reconnection challenge (double Harris setup standard)
- Daughton et al. 2011 (Nature Phys 7): PIC reconnection with plasmoids
