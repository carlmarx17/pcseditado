# Hoja de Harris con anisotropía y colas Kappa

Leé este archivo antes de crear o modificar cualquier caso de reconexión. Los
primeros tres puntos son fallas **silenciosas**: la simulación corre, produce
salidas plausibles, y el resultado está mal.

## 1. El balance de presión ya no usa la temperatura total

El equilibrio de Harris clásico se sostiene porque la presión de plasma en el
centro de la hoja compensa la presión magnética del exterior:

```
n0 (T_i + T_e) = B0² / (2 μ0)
```

En `psc_reconnection.cxx` esto está cableado en la línea que calcula `TTe`:

```cpp
g.TTe = me*c² / (2 ε0 (wpe/wce)² (1 + Ti/Te));
```

Con anisotropía en la población de Harris esa relación deja de valer. La
presión que sostiene el gradiente perpendicular al campo es la **perpendicular**,
no la total:

```
n0 (T_i_perp + T_e_perp) = B0² / (2 μ0)
```

Si el código sigue derivando `TTe` de la temperatura isotrópica y después
aplica `A_i`, `A_e` encima, la hoja arranca con presión perpendicular
incorrecta por un factor de orden `A`. Para `A_i = 2.0` (mirror moderate) eso
es un desbalance del 100%: la hoja se expande o colapsa en los primeros
`ω_ci⁻¹` y el transitorio se parece bastante a un inicio de reconexión. Es fácil
confundirlo con física.

**Qué hacer:** el header de reconexión debe derivar `T_perp` del balance y
después obtener `T_par = T_perp / A`, no al revés. Y el bloque de verificación
de balance de presión que ya existe en el archivo (imprime `P_total` en varias
posiciones `y`) tiene que recalcularse con `P_perp`; si no, va a reportar
"balance OK" mientras el equilibrio real está roto.

También conviene revisar la relación de la velocidad de deriva que sostiene la
corriente: con `T_par ≠ T_perp` cambia el espesor efectivo de la hoja para el
mismo `B0`.

## 1b. El equilibrio fija β⊥ = 1 en la hoja: el ancla va en el fondo

Reescribiendo el balance de presión como beta, la condición
`n0(T_i⊥ + T_e⊥) = B0²/2μ0` **es** `β_i⊥ + β_e⊥ = 1`. No es una elección: la
población de Harris tiene β⊥ total igual a 1 por construcción.

Corolario incómodo: de ahí sale `β_i∥ (A_i - 1) < 1` siempre, o sea que la
población de la hoja **nunca** puede cruzar el umbral mirror en el sentido de la
familia A. Los β de la matriz uniforme (5.0, 6.0, 10.0) simplemente no existen
dentro de una hoja de Harris.

Por eso el ancla cruzada se define sobre el **plasma de fondo**, que es además
donde la literatura pone la anisotropía relevante para el tearing. La
anisotropía `A_i`, `A_e` sí se aplica a las dos poblaciones; los β solo se
anclan upstream.

El β paralelo del fondo, en función de los parámetros del header:

```
β_i_b∥ = (n_b/n_0) · (T_ib/T_i) · T_i∥/(T_i∥ + T_e∥)
```

Con `Ti/Te = 5` el último factor es `5/6`, así que
`β_i_b∥ = (n_b/n_0)·(T_ib/T_i)·0.833`. Los valores GEM actuales
(`n_b/n_0 = 0.2`, `T_ib/T_i = 1`) dan `β_i_b∥ = 0.167`. Para los anclajes de la
matriz:

| Caso | β_i∥ objetivo | `(n_b/n_0)·(T_ib/T_i)` requerido |
|---|---:|---:|
| mirror moderate | 5.0 | 6.0 |
| firehose moderate | 6.0 | 7.2 |
| whistler moderate | 1.0 | 1.2 |

**Dos consecuencias que hay que aceptar a ojos abiertos:**

1. Con el producto en 6–7, el fondo domina la presión total y la presión
   magnética pasa a ser una fracción chica. Esto es **reconexión de alto β**, un
   régimen distinto del GEM clásico (β ~ 0.2): la tasa de reconexión
   normalizada, la velocidad de Alfvén relevante y la estructura de la región de
   difusión cambian. Hay literatura específica de reconexión a alto β con
   anisotropía en el heliosheath que sirve de comparación
   (https://arxiv.org/pdf/1107.5558). No es un problema, pero sí algo que la
   tesis tiene que decir explícitamente en vez de presentarlo como "el caso
   GEM con anisotropía".
2. β varía entre inestabilidades (5.0 / 6.0 / 1.0), así que **los dos controles
   isotrópicos no pueden estar al mismo β que los tres casos anisótropos**. Con
   dos controles, β queda como confusor entre control y caso. Documentá a qué β
   corren los controles; si en algún momento hay presupuesto de cómputo, un
   control isotrópico por inestabilidad (al β de esa inestabilidad) elimina el
   confusor por completo.

**Y el corolario que decide si la matriz vale la pena:** con los valores GEM
actuales el fondo tiene `β_i = 0.167`, `β_e = 0.033`. Contra los criterios de
umbral del propio repo, las anisotropías del régimen moderate quedan **todas por
debajo del umbral** a ese β:

| Inestabilidad | Criterio | A del régimen moderate | A necesaria a β actual |
|---|---|---:|---:|
| mirror | `β_i∥(A_i - 1) > 1` | 2.0 | > 7.0 |
| firehose | `β_i∥(1 - A_i) > 2` | 0.3 | imposible a ese β |
| whistler | `A_e > 1 + 0.21/β_e∥^0.6` | 2.0 | > 2.62 |

O sea que si se corre la matriz sin tocar `n_b/n_0` ni `T_ib/T_i`, seis de los
ocho casos son linealmente estables por una razón trivial y no por física
interesante. Subir el producto a 6.0–7.2 es lo que hace que la pregunta tenga
sentido — y es exactamente lo que empuja el setup al régimen de alto β del punto
anterior. Los dos hechos son la misma decisión vista de dos lados, y conviene
presentarla así en la tesis en vez de justificarlos por separado.

## 2. La línea neutra es firehose-inestable por construcción

El criterio de firehose es

```
1 + μ0 (P_par - P_perp) / B²  <  0
```

En el centro de la hoja `B → 0`, así que **cualquier** exceso paralelo
(`A_i < 1`, o sea todos los casos firehose) viola el criterio ahí, sin importar
cuán chica sea la anisotropía. No es un error de setup: es geometría.

Consecuencia práctica: los casos `psc_reconnection_*_firehose_moderate` van a
mostrar fluctuaciones dentro de la hoja desde el paso cero, y **no son el modo
tearing**. Hay que separarlas antes de interpretar cualquier tasa de
crecimiento. Un control útil es comparar contra el caso isotrópico de la misma
distribución: lo que aparece en ambos es tearing, lo que aparece solo en el
firehose es el modo de anisotropía.

Matteini et al. (2013) encontraron justamente que las fluctuaciones firehose
excitadas por el exceso paralelo **no** desestabilizan eficientemente la hoja —
la hoja sigue estable mucho después de que el firehose satura. El caso opuesto
(exceso perpendicular, rama ion-ciclotrón/mirror) sí acelera el tearing. Ese
contraste es un resultado esperable de la matriz y sirve como validación: si tus
casos mirror y firehose dan lo mismo, algo está mal en el setup.

Para el caso whistler (anisotropía electrónica, `A_e > 1` con `A_i = 1`) hay
literatura específica sobre hojas de corriente sostenidas por anisotropía de
presión electrónica y su estabilidad 3D — relevante si en algún momento se
extiende a 3D, porque los modos dominantes ahí no son los mismos que en 2D.

## 3. La normalización de temperatura de las Kappa (verificado, está bien)

Es un error clásico pasarle a un generador Kappa el mismo `T[]` que a la
Maxwelliana cuando el generador lo interpreta como `θ²`: la temperatura real
queda inflada por `κ/(κ - 3/2)`, que es exactamente **2.0 a κ=3**. En
anisotropía uniforme eso desplaza β por un factor común; en reconexión sería
peor, porque el balance de presión es una condición absoluta y un factor 2
destruiría el equilibrio de la hoja.

**En este repo no pasa.** `createKappaMultivariate` en
`src/include/setup_particles.hxx` construye

```cpp
Y = Gamma(κ - 0.5, 1);   S = sqrt((κ - 1.5) / Y);
p[i] = Z · S · beta · sqrt(T[i] / m);
```

y como `E[1/Y] = 1/(κ - 1.5)`, sale `E[S²] = 1` exacto. La varianza del momento
es `β²T/m`, igual que en la Maxwelliana: `T[]` ya es temperatura cinética. No
hay corrección pendiente ni retroactiva.

Vale la pena reverificarlo si alguien toca ese generador — el error es
silencioso y el equilibrio de Harris no perdona.

Nota adicional: las SKD tienen singularidades para κ ≤ 3/2 (el `assert(kappa >
1.5)` del código lo refleja) y colas formalmente superlumínicas. κ=3 y κ=5 están
bien; si en algún momento se quiere bajar de κ=2, hace falta una Kappa
regularizada (RKD), para la que existe un equilibrio de Harris generalizado
publicado.

## 3b. Los monolitos de reconexión NO son bi-Kappa, aunque lo parezcan

`psc_reconnection.cxx:454` y `psc_reconnection_comparable.cxx:440` hacen

```cpp
setup_p.kappa = 3.0;   // "The parameter kappa for the distribution!"
```

y `psc_reconnection_comparable.cxx:34` incluso dice en un comentario "kappa
multivariada isótropa (kappa=3)". **Las dos corridas son Maxwellianas.**

El motivo: `setup_p.kappa` es solo un campo de configuración. Quien elige el
muestreador es el lambda de inicialización de partículas. La ruta genérica
`SetupParticles::setupParticles` llama incondicionalmente a
`createMaxwellian` (`setup_particles.hxx:281`); `createKappaMultivariate` se
invoca a mano, y el único lugar del repo que lo hace es
`psc_anisotropy_case.hxx:321`, dentro de su propio lambda.

Así que asignar `kappa` sin llamar al muestreador es código muerto que *parece*
configuración activa. Consecuencias concretas:

- Los resultados de reconexión obtenidos hasta ahora son bi-Maxwellianos,
  independientemente de lo que digan los comentarios.
- Al construir `psc_reconnection_case.hxx` hay que **portar la llamada
  explícita** a `createKappaMultivariate` bajo `PSC_USE_KAPPA`, copiando el
  patrón del header de anisotropía. Si solo se setea `setup_p.kappa`, los cuatro
  casos bi-Kappa de la matriz B van a correr Maxwellianos y a compararse contra
  los controles bi-Maxwellianos como si fueran distintos. Es la falla más cara
  de esta lista: cuatro corridas de COSMA que producen exactamente el mismo
  resultado que su control y una conclusión de "las colas kappa no afectan la
  reconexión" que sería un artefacto puro.
- Los comentarios de los dos monolitos que hablan de kappa hay que corregirlos o
  borrarlos.

## 4. Valores de referencia de la hoja de Harris

El benchmark canónico es el GEM Reconnection Challenge (Birn et al. 2001). Sus
parámetros son el punto de comparación que cualquier lector de la tesis va a
tener en la cabeza:

| Parámetro | GEM | En `psc_reconnection.cxx` |
|---|---|---|
| Espesor de hoja `L` | `0.5 d_i` | `0.5 d_i` |
| `T_i / T_e` | 5 | 5.0 |
| Densidad de fondo `n_b/n_0` | 0.2 | 0.2 |
| Perturbación `ψ0` | `0.1 B0 d_i` | `dby/B0 = 0.03` |
| `m_i/m_e` | 25 | 25 |
| Caja | `25.6 × 12.8 d_i` | `25.6 × 51.2 d_i`, doble hoja periódica |

Dos diferencias deliberadas del repo que conviene poder justificar cuando
alguien pregunte:

- **Perturbación 3% en vez de 10%.** Una perturbación chica deja ver la fase
  lineal del tearing antes de que domine la dinámica no lineal. Con 10% el
  sistema arranca prácticamente en régimen no lineal. Para medir tasas de
  crecimiento —que es lo que la matriz quiere comparar— 3% es la elección
  correcta, pero el tiempo hasta la reconexión rápida es más largo y hay que
  presupuestar pasos en consecuencia.
- **Doble hoja con fronteras totalmente periódicas** en vez de hoja única con
  fronteras conductoras. Evita artefactos de frontera, a costa de que las dos
  hojas puedan interactuar si `Ly` es chico. Con `Ly = 25.6 d_i` las hojas están
  a `12.8 d_i` una de otra, suficiente para esta escala.

`psc_reconnection_comparable.cxx` usa `wpe/wce = 2.0` en vez de 12.5, con la
justificación de `dx/λ_De` documentada en el propio archivo. Ese valor controla
cuán magnetizados están los electrones y no es un parámetro cosmético: dos casos
de reconexión con `wpe/wce` distinto **no son comparables entre sí**.

## 5. Qué debe ser idéntico entre los ocho casos de reconexión

Todo esto vive en el header de reconexión y ningún caso individual debería
redefinirlo:

```
L_di, Ti_Te, Tib_Ti, Teb_Te, nb_n0, dby_b0, bg, Lpert_Lz,
wpe_wce, mass_ratio, Lx_di, Ly_di, Lz_di, ngrid, nicell,
intervalos de salida, condiciones de frontera
```

Lo único que varía entre los ocho: `PSC_USE_KAPPA`, `PSC_KAPPA`,
`PSC_BETA_I_PAR`, `PSC_BETA_E_PAR`, `PSC_TI_PERP_OVER_TI_PAR`,
`PSC_TE_PERP_OVER_TE_PAR`.

## Referencias

- Birn et al. (2001), *Geospace Environmental Modeling (GEM) Magnetic
  Reconnection Challenge*, JGR 106, 3715 — parámetros estándar de la hoja.
  https://ui.adsabs.harvard.edu/abs/2001JGR...106.3715B/abstract
- Matteini et al. (2013), *Proton temperature anisotropy and magnetic
  reconnection in the solar wind: effects of kinetic instabilities on current
  sheet stability* — anisotropía del plasma circundante y estabilidad del
  tearing. https://arxiv.org/pdf/1212.2101
- *Generalized Harris Sheet Equilibrium in Regularized Kappa Distributed
  Plasmas*, ApJ (2023) — equilibrio de Harris con Kappa regularizadas.
  https://iopscience.iop.org/article/10.3847/1538-4357/acf851
- *Three-Dimensional Stability of Current Sheets Supported by Electron Pressure
  Anisotropy* — relevante para el caso whistler.
  https://arxiv.org/pdf/1910.02848
- Hietala et al. (2015), *Ion temperature anisotropy across a magnetotail
  reconnection jet*, GRL — anisotropía observada en chorros de reconexión.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GL065168
