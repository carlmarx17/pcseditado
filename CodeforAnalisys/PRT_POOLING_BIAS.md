# Sesgo del pooling de la ventana prt

Fecha: 2026-10-27. Mide cuánta energía de flujo resuelto `plot_prt._compute_particle_energies`
anota como térmica, usando los momentos por celda. Script: `prt_pooling_bias.py`.
Tests: `test_prt_pooling_bias.py`.

## Qué se mide

`_compute_particle_energies` parte la energía de la ventana prt con **una sola media
global**:

```
E_bulk = 0.5 m |<u>|^2        E_th = 0.5 m <|u - <u>|^2>
```

Escribiendo la ventana como celdas `c` con `N_c` macropartículas, la ley de la varianza
total es exacta:

```
E_th(pooled)   = E_th(por celda)   + D
E_bulk(pooled) = E_bulk(por celda) - D
D = 0.5 m sum_c w_c |u_c - <u>|^2,     w_c = N_c / N
```

El pooling mueve **exactamente** `D` del canal bulk al térmico. No es una aproximación:
es una identidad sobre la muestra. `D` es la varianza (pesada por densidad) de la
velocidad media de celda, y los momentos la dan directo con `u_c = p_c / (n_c m)`.

`D` no es todo física. La media de celda `u_c` promedia `N_c` partículas, así que carga
ruido de muestreo de varianza `(2 T_perp + T_par) / (m N_c)` aunque el plasma sea
uniforme. Como `E_th = (2 T_perp + T_par) / 2`, el piso es

```
D_ruido / E_th = 1 / N_c
```

o sea **uno sobre las partículas por celda**. Por eso `t = 0` no se descarta: ahí no hay
flujo todavía, y el primer punto es el piso medido.

## Validación

`test_prt_pooling_bias.py`, cuatro pruebas:

- La identidad se verifica **contra `plot_prt._compute_particle_energies` misma**, no
  contra una reimplementación: se escribe un snapshot prt con deriva conocida por celda,
  se llama a la función de producción y se compara con la partición por celda. Coincide a
  12 decimales.
- El estimador basado en momentos recupera la misma `D` a 1e-9.
- Sin flujo, `D/E_th` cae como `1/N_c` sobre 100 celdas y tres valores de ppc, dentro del
  25 % de la esperanza `(K-1)/K / N_c`.

## Corridas

Dos corridas locales de `psc_mirror_bimaxwellian_moderate`, **idénticas salvo las
partículas por celda**, con los overrides de entorno que ya soporta
`psc_anisotropy_case.hxx`:

| | valor |
| --- | --- |
| ngrid | 96 (producción: 576) |
| dominio | 20 d_i, sin cambios |
| nicell | 50 y 200 (producción: 1000) |
| pasos | 19000, hasta t Ω_ci = 15 |
| ventana prt | celdas 38..56 por eje, 361 celdas |

La salida de partículas quedó desactivada: el HDF5 local es serial y PSC pide HDF5
paralelo para escribirla. No hace falta — la ventana es un bloque fijo de celdas, así que
los momentos alcanzan.

**Resolución reducida.** `dx/λ_De = 52` contra 8.68 en producción, y 20× menos partículas
por celda. Los números de abajo son de estas corridas, no de las de producción.

## Resultado

### El piso de ruido se comporta como dice la fórmula

En `t = 0` no hay flujo, así que `D` es puro ruido de muestreo:

| nicell | `D/E_th` medido en t=0 | `1/N_c` |
| --- | --- | --- |
| 50 | 1.91 % (iones), 1.99 % (electrones) | 2.00 % |
| 200 | 0.51 % (iones), 0.52 % (electrones) | 0.50 % |

Cuadruplicar las partículas por celda divide el piso por 3.8. La fórmula se sostiene sobre
datos reales, no solo en el test sintético.

### Iones: ~0.3 % de `E_th`, separado del ruido por diferencia entre corridas

`D/E_th` arranca en el piso, sube a 3.7 % en t Ω_ci ≈ 0.8 (el asentamiento del ruido del
quiet start) y después baja hasta ~1 % a 50 ppc, **por debajo** del piso nominal de 2 %: el
depósito de primer orden promedia celdas vecinas, así que el ruido real está por debajo de
`σ²/N_c` y el `1/N_c` es cota superior, como se dijo arriba.

Las dos corridas lo separan. Con `D = D_ruido + D_fis` y `D_ruido(50) = 4 D_ruido(200)`:

| t Ω_ci | 50 ppc | 200 ppc | razón | `D_fis` implícito |
| --- | --- | --- | --- | --- |
| 0.0 | 1.91 % | 0.51 % | 3.77 | — (no hay flujo) |
| 2.4 | 1.75 % | 0.89 % | 1.96 | 0.60 % |
| 4.8 | 1.33 % | 0.58 % | 2.28 | 0.33 % |
| 6.3 | 1.23 % | 0.59 % | 2.09 | 0.38 % |
| 7.9 | 1.04 % | 0.47 % | 2.20 | 0.28 % |

En `t = 0` la razón es 3.77 ≈ 4: puro ruido. Después se estabiliza cerca de 2, que no es ni
ruido puro (4) ni estructura pura (1). La parte física sale **0.3–0.6 % de `E_th`**, y el
ruido real a 200 ppc queda en ~0.2 %, un 40 % del `1/N_c` nominal — otra vez el suavizado
del depósito.

Ese `D_fis` iónico es aproximadamente plano en el tiempo, a diferencia del electrónico que
sigue al modo. Es un argumento a favor de la resta: si el flujo iónico entre celdas fuera
del modo mirror, crecería con él.

**La resta supone que `D_fis` es igual en las dos corridas, y no lo es del todo.** La
inestabilidad crece de la semilla de ruido, que va como `1/sqrt(ppc)`, así que a 200 ppc el
modo está más chico a un `t` dado. Si `D_fis(200) < D_fis(50)`, la resta sobreestima el
ruido: los 0.3–0.6 % son cota inferior del sesgo físico iónico.

### Electrones: estructura real, pero **no convergida** en estas corridas

| t Ω_ci | 50 ppc | 200 ppc | razón |
| --- | --- | --- | --- |
| 0.0 | 1.99 % | 0.52 % | 3.86 |
| 4.8 | 19.9 % | 2.52 % | 7.90 |
| 9.5 | 31.4 % | 3.10 % | 10.12 |
| 15.0 | 47.3 % | 4.46 % | 10.62 |

El piso se queda plano mientras `D` sube 24× en una corrida y 9× en la otra: no es ruido
instantáneo, es estructura que crece con el tiempo.

Pero la razón entre corridas **no baja hacia 1**: arranca en 3.86 (ruido puro, como debe) y
se estabiliza cerca de 10-12. Si esto fuera la corriente diamagnética del modo mirror —una
estructura física, independiente de ppc— las dos corridas tendrían que acercarse a medida
que el modo satura. No se acercan: quedan separadas por una década.

En escala, `D_e ∝ N_c^-1.7` (10.6 = 4^1.7). El ruido puro daría exponente 1 y una
estructura física convergida daría 0. Un exponente de 1.7 dice que el flujo electrónico
entre celdas está **sembrado por ruido de partículas y amplificado**, no convergido. Con
`dx/λ_De = 52` eso es esperable: la dinámica electrónica de escala de malla no está
resuelta.

**Conclusión honesta para electrones: el mecanismo es real y el diagnóstico lo mide, pero
la magnitud de estas corridas no se traslada a producción.** Extrapolar el exponente a
1000 ppc daría ~0.3 %, y producción además resuelve Debye 6× mejor — pero extrapolar una
cantidad no convergida no es una medición.

### El canal que de verdad queda mal es el bulk

`D` es chico frente a `E_th` (para iones) pero enorme frente a `E_bulk`: la media única de
la ventana conserva entre el 1 % y el 10 % de la energía de flujo resuelta. En la última
instantánea a 50 ppc el bulk pooled se queda corto por un factor 9 (iones) y 130
(electrones). Para leer energía térmica el pooling es defendible en iones; para leer
energía de flujo no lo es en ninguna de las dos especies.

## Lo que esto no dice

- **No es el número de producción.** A 1000 ppc el piso nominal baja a 0.1 % (y el real,
  con el suavizado, a ~0.04 %), así que el canal iónico se mide ahí directo, sin necesidad
  de restar dos corridas. El electrónico depende fuerte de ppc y de la resolución de Debye,
  así que el suyo solo sale de producción. Correr el script sobre los momentos de COSMA da
  los dos números de la tesis.
- **Las dos corridas no se comparan a tiempo fijo.** La inestabilidad crece de la semilla
  de ruido de partículas, que va como `1/sqrt(ppc)`: a 200 ppc el modo arranca 2× más chico
  y a un `t` dado está en otra amplitud. El `t = 0` sí se compara — ahí no hay modo.
- **El depósito suaviza.** `p_c/n_c` ya mezcla celdas vecinas, así que `D` medida así es
  cota inferior de la estructura real y el piso `1/N_c` es cota superior del ruido. La
  estructura de flujo por debajo de ~2 celdas no la ve ni este estimador ni ninguna
  corrección por celda.
- **No se tocó `plot_prt`.** Cambiar la partición movería números ya publicados; esta
  medición dice cuánto, no decide qué hacer. Solo se agregó un puntero en su docstring.

## Reproducción

```sh
cmake --build build --target psc_mirror_bimaxwellian_moderate -j 6
mkdir -p analysis_results/prt_pooling_bias/runs/nicell50
(cd analysis_results/prt_pooling_bias/runs/nicell50 && \
 PSC_NGRID=96 PSC_NICELL=50 PSC_NMAX=19000 PSC_FIELDS_EVERY=1000 \
 PSC_PARTICLES_EVERY=-1 PSC_CHECKPOINT_EVERY=-1 PSC_NP_Y=2 PSC_NP_Z=2 \
 mpirun -n 4 ../../../../build/src/psc_mirror_bimaxwellian_moderate > run.log 2>&1)
PSC_PROFILE=mirror_bimaxwellian_moderate MPLBACKEND=Agg .venv/bin/python \
  CodeforAnalisys/prt_pooling_bias.py \
  --data-dir analysis_results/prt_pooling_bias/runs/nicell50 --nicell 50 \
  --outdir analysis_results/prt_pooling_bias/nicell50
```

Sobre una corrida de producción es la última orden sola, con `--nicell 1000` y el
`--data-dir` de los momentos. Figura, CSV y datos crudos quedan en
`analysis_results/prt_pooling_bias/` (ignorado por Git).
