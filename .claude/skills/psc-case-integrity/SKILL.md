---
name: psc-case-integrity
description: Reglas de integridad para la matriz de casos de simulación PSC — inestabilidades de anisotropía (mirror/firehose/whistler, bi-Maxwellianas y bi-Kappa) y reconexión magnética en hoja de Harris parametrizada por esa misma anisotropía. Úsala siempre que se vaya a crear, editar, refactorizar, comparar o revisar cualquier archivo psc_*.cxx de src/, psc_anisotropy_case.hxx, psc_reconnection*, los targets de CMakeLists.txt asociados, o cuando se hable de cambiar kappa, beta, anisotropía, grilla, ppc, mass ratio, tamaño de caja, espesor de hoja, densidad de fondo, amplitud de perturbación o cualquier parámetro de una corrida — incluso si el usuario solo dice "cambiá esto en el caso moderate" o "agregá un caso nuevo". El punto de la skill es proteger la comparabilidad entre casos, porque los cambios estructurales rompen la matriz experimental y deben frenarse y consultarse antes de aplicarse.
---

# Integridad de la matriz de casos PSC

## Por qué esto importa

Este repositorio no es una colección de simulaciones sueltas: es **un
experimento controlado**. La tesis parametriza el efecto de la anisotropía de
temperatura, primero en plasma uniforme y después sobre la reconexión
magnética. Esa comparación solo tiene sentido si **todo lo demás es idéntico**:
misma grilla, misma resolución, mismo `mi/me`, mismas partículas por celda,
mismos intervalos de salida, mismas fronteras, mismo `B0`, y —en reconexión— la
misma hoja de Harris.

Si un caso corre en 576² y su hermano en 1024², la diferencia observada en la
tasa de crecimiento puede venir de la física o de la resolución, y ya no hay
forma de saberlo. Cuando eso pasa, no se pierde un archivo: se pierden semanas
de cómputo en COSMA y, peor, la conclusión física deja de ser defendible.

Por eso la regla central es simple:

> Entre casos de una misma familia, **lo único que puede cambiar es la física
> del régimen**: `PSC_KAPPA`, los betas paralelos y las razones de anisotropía.
> Cualquier otra diferencia es un defecto, no una decisión de diseño.

## Las dos familias de setup

La matriz tiene dos setups, y la anisotropía es el eje que los atraviesa a los
dos. Ese cruce **es** el argumento de la tesis: primero se caracteriza la
inestabilidad en plasma uniforme, después se pregunta qué le hace a la
reconexión.

**A. Anisotropía pura** — plasma uniforme, campo de fondo uniforme, sobre
`psc_anisotropy_case.hxx`:

```
psc_<inestabilidad>_<distribucion>_<regimen>.cxx
   inestabilidad: mirror | firehose | whistler
   distribucion:  bimaxwellian | bikappa3 | bikappa5
   regimen:       strong | moderate | weak
```

**B. Reconexión** — doble hoja de Harris, sobre `psc_reconnection_case.hxx`,
con **la misma anisotropía** que los casos de A:

```
psc_reconnection_<distribucion>_<inestabilidad>_<regimen>.cxx
psc_reconnection_<distribucion>_isotropic.cxx        # controles
```

La matriz de reconexión son ocho casos: dos controles isotrópicos
(bi-Maxwelliano y bi-Kappa) más las tres inestabilidades en régimen `moderate`
por cada distribución. El control bi-Kappa isotrópico no es opcional: sin él,
cualquier diferencia en un caso `bikappa3_mirror_moderate` mezcla el efecto de
las colas supratérmicas con el de la anisotropía y no se puede separar.

**Los casos de reconexión no son una familia aparte.** Están dentro de la misma
matriz y se rigen por las mismas reglas. Lo único que cambia es que tienen un
conjunto extra de invariantes propios (los de la hoja de Harris) y un ancla
cruzada contra la familia A.

## Anatomía de un caso

Cada caso es un archivo deliberadamente **delgado**. Toda la física, la malla y
las salidas viven en el header compartido de su setup. El `.cxx` solo declara
qué régimen es:

```cpp
// ======================================================================
// psc_mirror_bimaxwellian_moderate - Mirror Moderate Bi-Maxwellian
//
// beta_i_parallel=5.0, Ai=Ti_perp/Ti_parallel=2.0
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// ======================================================================

#define PSC_CASE_LABEL "mirror_bimaxwellian_moderate"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_mirror_bimaxwellian_moderate"

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 5.0
#define PSC_TI_PERP_OVER_TI_PAR 2.0
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
```

Esa delgadez **es** el mecanismo de integridad. Mientras ningún caso tenga
código propio, es estructuralmente imposible que dos casos difieran en algo que
no sea su régimen. Defender esa delgadez es la tarea principal de esta skill.

### Lista blanca de `#define` por caso

Identidad (obligatorios, los tres):

| Define | Regla |
|---|---|
| `PSC_CASE_LABEL` | igual al nombre del archivo sin `psc_` ni `.cxx` |
| `PSC_DISTRIBUTION_LABEL` | `"Bi-Maxwellian"` o `"Bi-Kappa"`, coherente con `PSC_USE_KAPPA` |
| `PSC_OUTPUT_BASENAME` | `"prt_" + PSC_CASE_LABEL` |

Física del régimen (obligatorios, los cuatro, en ambos setups):

| Define | Qué es |
|---|---|
| `PSC_BETA_I_PAR` | beta paralelo de iones |
| `PSC_BETA_E_PAR` | beta paralelo de electrones |
| `PSC_TI_PERP_OVER_TI_PAR` | anisotropía iónica `A_i` |
| `PSC_TE_PERP_OVER_TE_PAR` | anisotropía electrónica `A_e` |

Distribución (opcionales, van juntos o no van):

| Define | Qué es |
|---|---|
| `PSC_USE_KAPPA` | `1` en casos bi-Kappa; ausente (default `0`) en bi-Maxwellianos |
| `PSC_KAPPA` | índice κ; solo tiene sentido con `PSC_USE_KAPPA 1` |

Excepción documentada, solo en setup A:

| Define | Cuándo |
|---|---|
| `PSC_DOMAIN_DI` | **solo** en variantes `*_bigbox40`, con valor `40.0` |

`PSC_DOMAIN_DI` es la única excepción porque el tamaño de caja no es
overrideable por variable de entorno (ver el `#ifndef` en el header), así que un
estudio de convergencia de caja exige un ejecutable aparte. Es un experimento
deliberado sobre el tamaño del dominio, no una divergencia accidental — y por
eso la variante bigbox debe ser *idéntica en todo lo demás* a su caso base.

**Cualquier otro `#define` en un `.cxx` de caso es una alerta.** En particular
`PSC_NGRID_DEFAULT`, `PSC_NICELL_DEFAULT`, `PSC_MASS_RATIO`, `PSC_VA_OVER_C`,
`PSC_LAMBDA0`, `PSC_NMAX_DEFAULT`, `PSC_FIELDS_EVERY_DEFAULT`,
`PSC_PARTICLES_EVERY_DEFAULT`, `PSC_CHECKPOINT_EVERY_DEFAULT`,
`PSC_BALANCE_INTERVAL`, y en reconexión `PSC_L_DI`, `PSC_TI_TE`, `PSC_NB_N0`,
`PSC_DBY_B0`, `PSC_BG`, `PSC_WPE_WCE`: todos viven en el header de su setup y
ponerlos en un caso individual rompe exactamente lo que hay que proteger.

Tampoco debe haber en un `.cxx` de caso: `#include` que no sea el del header de
su setup, funciones, `struct`, `main`, `setupParticles`, ni lógica de ningún
tipo.

## Grupos de paridad

Estas son las comparaciones que el diseño experimental sostiene. Al tocar
cualquier miembro de un grupo, revisá a sus hermanos.

**1. Hermanos de régimen** (setup A) — dentro de una inestabilidad y
distribución, los `strong` / `moderate` / `weak` difieren *solo* en betas y
anisotropías.

**2. Gemelos de distribución** — mismo régimen físico, distinta distribución
inicial. Deben compartir *exactamente* los cuatro valores de beta/anisotropía;
lo único que cambia es `PSC_USE_KAPPA`/`PSC_KAPPA`/`PSC_DISTRIBUTION_LABEL`:

```
psc_mirror_bimaxwellian_moderate      ↔  psc_mirror_bikappa3_moderate
psc_reconnection_bimaxwellian_mirror_moderate
                                      ↔  psc_reconnection_bikappa3_mirror_moderate
```

Este es el corazón de la tesis (aislar el efecto de las colas supratérmicas).
Si los betas se desincronizan entre gemelos, la comparación se invalida por
completo y sin ruido visible: los dos casos corren perfecto y dan resultados
que no significan nada.

**3. Gemelos de caja** (setup A) — idénticos salvo `PSC_DOMAIN_DI`.

**4. Ancla cruzada A↔B** — cada caso de reconexión anisótropo debe llevar
**exactamente** los cuatro valores de beta/anisotropía de su caso de anisotropía
pura correspondiente, **referidos al plasma de fondo**:

```
psc_reconnection_bimaxwellian_mirror_moderate    ancla en  psc_mirror_bimaxwellian_moderate
psc_reconnection_bikappa3_firehose_moderate      ancla en  psc_firehose_bikappa3_moderate
                                                 (o su gemelo bi-Maxwelliano si aún no existe)
```

Sin este anclaje no se puede decir "el mismo plasma que era mirror-inestable en
uniforme, ahora sobre una hoja de corriente". Se estarían comparando dos
plasmas distintos y la pregunta de la tesis queda sin responder.

El anclaje es sobre el **fondo**, no sobre la población de Harris, y eso no es
un detalle: el equilibrio de Harris fija `β⊥ = 1` en la hoja por construcción,
así que los β de la matriz uniforme (5.0, 6.0, 10.0) son imposibles ahí. La
anisotropía `A_i`, `A_e` sí se aplica a las dos poblaciones; los β solo tienen
sentido anclados upstream. Si alguna vez ves un caso de reconexión con
`PSC_BETA_I_PAR` "aplicado a la hoja", está mal planteado.

**5. Invariantes de Harris** (setup B) — los ocho casos comparten espesor de
hoja `L_di`, `Ti/Te`, campo guía, amplitud y longitud de onda de perturbación,
`wpe/wce`, `mi/me`, caja, grilla, ppc e intervalos de salida.

**Ojo con una excepción que no es opcional:** `nb_n0`, `Tib_Ti` y `Teb_Te`
**no** son invariantes, son cantidades *derivadas del ancla*. El β paralelo del
fondo sale de

```
β_i_b∥ = (n_b/n_0) · (T_ib/T_i) · T_i∥/(T_i∥ + T_e∥)
```

que con `Ti/Te = 5` da `β_i_b∥ = (n_b/n_0)·(T_ib/T_i)·(5/6)`. Para llegar al
`β_i∥ = 5.0` de mirror moderate hace falta `(n_b/n_0)·(T_ib/T_i) = 6`, y cada
inestabilidad pide un producto distinto (mirror 6.0, firehose 7.2, whistler
1.2). Esos tres defines los calcula el header a partir de los β anclados —
nadie los ajusta a mano, y si aparecen escritos a mano en un `.cxx` es un
defecto. Las consecuencias físicas de correr con un fondo así de caliente están
en `references/harris_anisotropo.md`.

**6. Controles isotrópicos** — en `psc_reconnection_*_isotropic` tiene que
valer `A_i = A_e = 1.0`. Un control con anisotropía no es un control.

## Qué hacer ante un pedido que rompe la paridad

Cuando lo que te piden cambiaría algo que no sea kappa/beta/anisotropía en un
solo caso — o cambiaría un header, que afecta a todo su setup — **no lo
apliques todavía**. No es obstruccionismo: el usuario casi siempre quiere el
cambio, pero quiere aplicarlo a la familia entera, y desde afuera no se ve
cuántos casos ya corridos quedarían huérfanos.

Frená y respondé con esta estructura:

```
Ese cambio rompe la paridad de <grupo>.

Qué cambiaría:   <parámetro>: <valor actual> → <valor nuevo>
Casos afectados: <lista de los hermanos que quedarían distintos>
Consecuencia:    <qué comparación de la tesis deja de ser válida>

Opciones:
  a) Aplicarlo a los N casos del grupo (mantiene la comparabilidad;
     invalida las corridas previas de esos casos)
  b) Aplicarlo solo aquí como variante nueva, con archivo y target propios
  c) Cancelar

¿Cuál preferís?
```

Los cambios que **sí** podés aplicar directamente, porque no tocan la
estructura: ajustar un beta o una anisotropía de un caso (es su definición),
corregir un `PSC_OUTPUT_BASENAME` o `PSC_CASE_LABEL` desalineado del nombre del
archivo, y actualizar comentarios para que reflejen los valores reales.

## Al crear un caso nuevo

Copiá el hermano más cercano y cambiá únicamente los defines de régimen. Nunca
lo escribas desde cero: escribirlo de memoria es justamente cómo se cuela un
define de más. Después, tres pasos que se olvidan seguido y dejan el caso
inconsistente:

1. `add_psc_executable(psc_<nombre>)` en `src/CMakeLists.txt`
2. una fila en la tabla correspondiente de `src/SIMULACIONES_ANISOTROPIA.md`
3. el encabezado de comentario del `.cxx` con los valores **reales** (los
   comentarios que mienten son peores que no tener comentario: alguien va a
   citar esos números en la tesis)

## Reconexión: lo que hay que tener en cuenta

La anisotropía va **tanto en la población de Harris como en el fondo**. Eso es
más realista que anisotropizar solo el fondo, pero rompe supuestos del
equilibrio de Harris clásico, y esos supuestos están cableados en el código
actual. Antes de tocar o crear cualquier caso de reconexión, leé
`references/harris_anisotropo.md`: cubre el balance de presión con `P_perp`, por
qué la línea neutra es firehose-inestable por construcción, la normalización de
temperatura de las Kappa (un factor 2 a κ=3 que rompe el equilibrio en silencio)
y los valores de referencia de la literatura para la hoja de Harris.

Ese archivo importa especialmente porque tres de esos problemas **no producen
ningún error visible**: la simulación corre, produce salidas plausibles, y el
resultado es incorrecto. Si estás por decir "listo, ya está el caso de
reconexión anisótropo", verificalo contra esa lista primero.

Los dos archivos monolíticos originales (`psc_reconnection.cxx` y
`psc_reconnection_comparable.cxx`, ~470 líneas cada uno) son el punto de partida
del setup B. Cuando se migren al esquema delgado, el header
`psc_reconnection_case.hxx` debe ser **propio de reconexión**, no una rama
dentro de `psc_anisotropy_case.hxx`: meter condicionales de hoja de Harris en el
header de anisotropía obligaría a revalidar los 16 casos uniformes ya corridos a
cambio de nada. Comparten el esquema y la parametrización de anisotropía, no el
código.

## Verificar antes de dar por cerrado

Después de crear o editar cualquier caso, corré el verificador:

```bash
python3 scripts/check_case_parity.py <ruta-al-repo>/src
```

Chequea la lista blanca de defines, los seis grupos de paridad, la coherencia
entre `PSC_USE_KAPPA` y `PSC_DISTRIBUTION_LABEL`, que los labels coincidan con
el nombre del archivo, que los números del comentario de encabezado coincidan
con los defines reales, y que cada caso tenga su target en `CMakeLists.txt`.
Sale con código 1 si encuentra algo.

Leé el reporte completo antes de responder: si hay hallazgos preexistentes que
no vienen de tu cambio, mencionalos pero no los arregles en silencio dentro de
otra tarea — mezclarlos hace imposible revisar cualquiera de los dos.

## Nota sobre los scripts de corrida

`PSC_ALLOW_ENV_OVERRIDES` está en `1`, así que los `.sh` de `cosma_jobs/`
pueden pisar por entorno casi todo lo que el header define (`PSC_NGRID`,
`PSC_NICELL`, `PSC_NMAX`, intervalos de salida). Esto significa que **la
paridad también se puede romper fuera del código**: dos casos idénticos en
`src/` pueden correr con grillas distintas si sus job scripts difieren.

Si estás tocando job scripts de casos que se van a comparar entre sí, aplicá el
mismo criterio: las variables de entorno de simulación deben coincidir entre
hermanos, y si una tiene que diferir, que sea explícito y comentado en el
script.
