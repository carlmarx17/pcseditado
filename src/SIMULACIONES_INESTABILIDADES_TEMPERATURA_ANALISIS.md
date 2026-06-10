# Analisis propuesto de inestabilidades por anisotropia de temperatura

## Configuracion comun de las simulaciones

Todas las corridas del barrido usan una configuracion 2D PIC completa
`PscConfig1vbecSingle<dim_yz>`, con iones y electrones cineticos. La direccion x
es invariante, mientras que el dominio numerico resuelve las direcciones y-z.

Los parametros comunes son:

| Parametro | Valor |
|---|---:|
| Razon de masas, `mi/me` | 200 |
| Velocidad de Alfven, `vA/c` | 0.05 |
| Campo magnetico, `B0` | 0.05 |
| Densidad inicial, `n` | 1.0 |
| Particulas por celda, `ppc` | 2000 |
| Grilla | 1408 x 1408 |
| Patches MPI | 8 x 8 = 64 |
| Dominio fisico | 20 di x 20 di |
| Fronteras | Periodicas |
| Pasos totales, `nmax` | 1,650,000 |
| Distribucion inicial | Bi-Maxwelliana |

La temperatura paralela de cada especie se obtiene desde beta:

```text
T_parallel = beta_parallel * B0^2 / 2
T_perp     = A * T_parallel
```

En el codigo, `T[0]` y `T[1]` corresponden a las temperaturas perpendiculares, y
`T[2]` a la temperatura paralela al campo magnetico de fondo.

La distribucion inicial no es isotropica cuando `A != 1`. Esa anisotropia es la
energia libre que alimenta el crecimiento de Mirror, Firehose o Whistler. No se
impone una perturbacion externa: cada modo debe crecer desde el ruido termico de
las particulas en un plasma inicialmente homogeneo y periodico.

## Condiciones de inestabilidad que se van a evaluar

### 1. Mirror: exceso de temperatura perpendicular ionica

La inestabilidad Mirror aparece cuando los iones tienen mas presion
perpendicular que paralela:

```text
Ti_perp > Ti_parallel
A_i = Ti_perp / Ti_parallel > 1
```

El criterio simplificado que se usara para clasificar los casos es:

```text
beta_i_perp - beta_i_parallel > 1
```

Como `beta_i_perp = A_i * beta_i_parallel`, la condicion se puede escribir como:

```text
beta_i_parallel * (A_i - 1) > 1
```

Fisicamente, esta condicion significa que la presion perpendicular ionica es
suficientemente grande para amplificar compresiones del campo magnetico. En un
modo Mirror se espera que aparezcan estructuras no propagantes o de propagacion
muy lenta, con anticorrelacion entre densidad y magnitud del campo magnetico:
regiones con `|B|` mas alto tienden a tener menor densidad, y regiones con
`|B|` mas bajo tienden a acumular plasma.

El analisis buscara:

- crecimiento de fluctuaciones en `|B|`;
- anticorrelacion espacial entre `n_i` y `|B|`;
- redistribucion de energia desde la anisotropia ionica hacia campos y calor;
- reduccion progresiva de `A_i` hacia valores mas cercanos al umbral marginal.

### 2. Firehose: exceso de temperatura paralela ionica

La inestabilidad Firehose aparece cuando los iones tienen mas presion paralela
que perpendicular:

```text
Ti_parallel > Ti_perp
A_i = Ti_perp / Ti_parallel < 1
```

El criterio simplificado que se usara es:

```text
beta_i_parallel - beta_i_perp > 2
```

Como `beta_i_perp = A_i * beta_i_parallel`, la condicion equivalente es:

```text
beta_i_parallel * (1 - A_i) > 2
```

Fisicamente, el exceso de presion paralela reduce la tension magnetica efectiva.
Cuando esa tension no alcanza para estabilizar las perturbaciones, el campo puede
ondularse y crecer una fluctuacion Firehose. En comparacion con Mirror, Firehose
puede mostrar una respuesta mas ondulatoria y una transferencia rapida de energia
desde el movimiento paralelo ionico hacia fluctuaciones magneticas transversales.

El analisis buscara:

- crecimiento de componentes transversales del campo magnetico;
- evolucion temporal de la anisotropia ionica `A_i`;
- fase lineal de crecimiento y posterior saturacion;
- comparacion entre caso fuerte, moderado y debil;
- verificacion especial del caso marginal, donde el crecimiento puede ser lento
  o no aparecer claramente dentro del tiempo simulado.

### 3. Whistler: exceso de temperatura perpendicular electronica

La familia Whistler se activa cuando los electrones, no los iones, tienen
anisotropia perpendicular:

```text
Te_perp > Te_parallel
A_e = Te_perp / Te_parallel > 1
```

Estos modos operan a escala electronica. Por eso, en estos casos es
especialmente importante que la grilla resuelva bien `d_e`. Con `dx ~= 0.2 d_e`,
el barrido esta disenado para capturar la dinamica electronica relevante, aunque
la longitud de Debye no quede completamente resuelta.

El analisis buscara:

- crecimiento temprano de fluctuaciones a escala electronica;
- transferencia de energia desde la anisotropia electronica;
- reduccion de `A_e` durante la saturacion;
- diferencias temporales respecto a Mirror y Firehose, ya que Whistler deberia
  evolucionar en escalas electronicas mas rapidas.

## Simulaciones que se van a correr

Se correra una matriz de nueve simulaciones bi-Maxwellianas. El diseno separa
tres familias fisicas y tres niveles de distancia al umbral: Strong, Moderate y
Weak. Esta organizacion permite estudiar no solo si una inestabilidad aparece,
sino como cambia su tasa de crecimiento, saturacion y firma espacial cuando la
anisotropia inicial se acerca al limite marginal.

### Casos Mirror

| Caso | Archivo | Regimen | beta_i_parallel | A_i | Criterio |
|---|---|---|---:|---:|---:|
| M-S-bM | `psc_M_S_bM.cxx` | Strong | 5.0 | 3.0 | 10.0 |
| M-M-bM | `psc_M_M_bM.cxx` | Moderate | 5.0 | 2.0 | 5.0 |
| M-W-bM | `psc_M_W_bM.cxx` | Weak | 6.0 | 1.5 | 3.0 |

Los tres casos satisfacen `beta_i_parallel * (A_i - 1) > 1`, por lo que todos
deberian ser inestables. Se espera que M-S-bM tenga el crecimiento mas claro y
rapido, M-M-bM sirva como caso intermedio, y M-W-bM permita observar un
crecimiento mas cercano al umbral.

### Casos Firehose

| Caso | Archivo | Regimen | beta_i_parallel | A_i | Criterio |
|---|---|---|---:|---:|---:|
| F-S-bM | `psc_F_S_bM.cxx` | Strong | 10.0 | 0.1 | 9.0 |
| F-M-bM | `psc_F_M_bM.cxx` | Moderate | 6.0 | 0.3 | 4.2 |
| F-W-bM | `psc_F_W_bM.cxx` | Weak/marginal | 3.0 | 0.6 | 1.2 |

F-S-bM y F-M-bM cumplen claramente el criterio `beta_i_parallel * (1 - A_i) >
2`. F-W-bM queda por debajo del umbral simplificado. Ese caso se conserva porque
es cientificamente util como control marginal: si no crece, ayuda a separar
crecimiento fisico de ruido numerico; si crece lentamente, permite evaluar la
sensibilidad del criterio fluidizado frente al comportamiento cinetico real.

### Casos Whistler

| Caso | Archivo | Regimen | beta_e_parallel | A_e |
|---|---|---|---:|---:|
| W-S-bM | `psc_W_S_bM.cxx` | Strong | 0.5 | 3.0 |
| W-M-bM | `psc_W_M_bM.cxx` | Moderate | 0.5 | 2.0 |
| W-W-bM | `psc_W_W_bM.cxx` | Weak | 0.5 | 1.5 |

En estos tres casos los iones permanecen isotropicos. La comparacion se centra
en como la anisotropia electronica modifica los campos a escala `d_e`, y en si
la saturacion ocurre mas temprano que en las inestabilidades ionicas.

## Analisis que se va a realizar

### Evolucion temporal de energia

Primero se revisara la conservacion y transferencia de energia. Para cada
simulacion se calculara la evolucion temporal de:

- energia magnetica;
- energia electrica;
- energia cinetica ionica;
- energia cinetica electronica;
- energia total.

El objetivo es verificar que el crecimiento de campos durante la fase lineal
proviene de la energia libre de la anisotropia y no de una deriva numerica
dominante. En la saturacion se espera una disminucion de la anisotropia y una
redistribucion entre energia de particulas y campos.

### Medicion de tasas de crecimiento

Para cada caso se identificara una ventana temporal donde la amplitud del modo
crece aproximadamente de forma exponencial. En esa ventana se ajustara:

```text
delta B(t) ~ delta B0 * exp(gamma * t)
```

La tasa `gamma` se comparara entre Strong, Moderate y Weak. La expectativa es:

```text
gamma_Strong > gamma_Moderate > gamma_Weak
```

En F-W-bM no se asumira crecimiento claro de antemano. Ese caso se analizara
como marginal y se revisara si la senal supera de manera robusta el nivel de
ruido.

### Mapas espaciales de campos y densidad

Se analizaran snapshots 2D de:

- `B_y`, `B_z` y `|B|`;
- densidad ionica y electronica;
- momentos de velocidad;
- presiones paralela y perpendicular, si estan disponibles en los diagnosticos.

Para Mirror se dara prioridad a la relacion espacial entre densidad y `|B|`.
Para Firehose se revisara la ondulacion del campo y el crecimiento de
fluctuaciones transversales. Para Whistler se inspeccionaran estructuras mas
finas, asociadas a escalas electronicas.

### Evolucion de anisotropias

El diagnostico central sera la evolucion de:

```text
A_i(t) = Ti_perp(t) / Ti_parallel(t)
A_e(t) = Te_perp(t) / Te_parallel(t)
```

Cada inestabilidad debe consumir parte de la anisotropia que la alimenta. Por
eso, una senal fisica consistente seria observar que el sistema se mueve hacia
el umbral marginal despues de la fase de crecimiento.

### Comparacion entre familias

El analisis final comparara las tres familias:

- Mirror: respuesta ionica asociada a exceso de presion perpendicular.
- Firehose: respuesta ionica asociada a exceso de presion paralela.
- Whistler: respuesta electronica asociada a exceso de presion perpendicular.

La comparacion se hara usando las mismas metricas en los nueve casos:
crecimiento de campo, tasa `gamma`, saturacion, cambio de anisotropia y firma
espacial dominante.

## Variables en los outputs

Los codigos del barrido escriben tres tipos principales de salida: campos,
momentos y energias. Las salidas de campos y momentos se escriben cada 690 pasos;
los momentos temporales se acumulan para promedio cada 138 pasos; las particulas
se guardan cada 400 pasos en la region central del dominio; y las energias se
registran cada 100 pasos.

| Salida | Frecuencia | Uso en el analisis |
|---|---:|---|
| `pfd*` | cada 690 pasos | Campos instantaneos para mapas 2D y amplitudes de modo |
| `tfd*` | cada 690 pasos, promedio cada 138 | Momentos promediados para densidad, corriente y tensor de presion |
| `prt_*` | cada 400 pasos | Particulas de la region central para distribuciones de velocidad |
| energias | cada 100 pasos | Evolucion global de energia de campos y particulas |

### Campos electromagneticos

El item de campos se escribe con nombre `jeh`. Sus componentes son:

```text
jx_ec, jy_ec, jz_ec
ex_ec, ey_ec, ez_ec
hx_fc, hy_fc, hz_fc
```

Para el analisis, `hx_fc`, `hy_fc` y `hz_fc` se interpretan como las componentes
del campo magnetico numerico. A partir de ellas se calcula:

```text
|B| = sqrt(hx_fc^2 + hy_fc^2 + hz_fc^2)
delta B = |B| - <|B|>
```

En Mirror se usara sobre todo `|B|` y su anticorrelacion con densidad. En
Firehose se revisara el crecimiento de componentes transversales del campo. En
Whistler se inspeccionaran fluctuaciones de campo a escala electronica.

### Momentos de particulas

El item de momentos se escribe con nombre `all`. Sus componentes incluyen, por
especie:

```text
rho
jx, jy, jz
px, py, pz
txx, tyy, tzz, txy, tyz, tzx
```

Como los archivos agregan sufijos por especie, estas variables aparecen
separadas para electrones e iones. En el analisis se usaran asi:

| Magnitud | Variable de salida | Uso |
|---|---|---|
| Densidad | `rho` o momento de carga por especie | Mapas de densidad y correlacion con `|B|` |
| Corriente | `jx`, `jy`, `jz` | Identificar respuesta electromagnetica y estructuras coherentes |
| Momento | `px`, `py`, `pz` | Calcular flujos medios si se normaliza por densidad |
| Tensor cinetico | `txx`, `tyy`, `tzz`, `txy`, `tyz`, `tzx` | Estimar presiones y anisotropias |

Para obtener temperaturas o presiones fisicas no basta con leer `txx`, `tyy` y
`tzz` directamente: hay que restar la contribucion del flujo medio. En forma
esquematica:

```text
P_aa = T_aa - (p_a^2 / rho_m)
```

donde `a` representa `x`, `y` o `z`, y `rho_m` es la densidad de masa de la
especie. Como el campo de fondo esta en `z`, se tomara:

```text
P_parallel = P_zz
P_perp     = (P_xx + P_yy) / 2
A          = P_perp / P_parallel
```

Esto permite seguir `A_i(t)` para Mirror/Firehose y `A_e(t)` para Whistler.

### Energias globales

El diagnostico de energias imprime las contribuciones:

```text
EX2, EY2, EZ2
BX2, BY2, BZ2
E_electron, E_ion
```

Con estas columnas se construyen:

```text
E_electrica = EX2 + EY2 + EZ2
E_magnetica = BX2 + BY2 + BZ2
E_particulas = E_electron + E_ion
E_total = E_electrica + E_magnetica + E_particulas
```

El crecimiento de la inestabilidad debe verse como aumento de energia de campo
alimentado por la energia libre de la anisotropia, no como una deriva monotona
artificial de la energia total.

## Gasto de memoria y volumen de datos

Cada corrida de la matriz bi-Maxwelliana usa `1408 x 1408` celdas, `2000`
particulas por celda y dos especies. El numero total de macroparticulas es:

```text
N_part = 2 especies * 1408^2 celdas * 2000 ppc
       = 7.93e9 macroparticulas
```

Con el almacenamiento usado por PSC para particulas, la estimacion practica es:

```text
Particulas: 2 * 1408^2 * 2000 * 28 B  ~= 207 GB
Campos EM:  1408^2 * 12 * 4 B         ~= 0.1 GB
Overhead, buffers MPI, sort y salida   ~= 15 GB
Total por corrida                      ~= 222 GB
```

Por eso estas corridas deben tratarse como simulaciones de produccion de memoria
alta. El pedido de memoria recomendado es de al menos `235-240 GB` por caso,
para dejar margen frente a buffers, ordenamiento de particulas y escritura de
diagnosticos.

| Recurso | Estimacion por corrida |
|---|---:|
| RAM minima fisica estimada | ~222 GB |
| RAM recomendada en SLURM | 235-240 GB |
| MPI ranks | 64 |
| Particulas totales | ~7.93e9 |
| Particulas guardadas por snapshot | ~16% del total |

La salida de particulas se limita a la region central:

```text
[0.3 * 1408, 0.7 * 1408] en y
[0.3 * 1408, 0.7 * 1408] en z
```

Esto guarda aproximadamente el 40% del dominio en cada direccion resuelta, es
decir cerca del 16% de las particulas por snapshot. Esta decision reduce de
forma importante el gasto de disco sin eliminar la informacion necesaria para
analizar distribuciones de velocidad en la zona central.

Los campos y momentos si se escriben sobre todo el dominio. Por eso el volumen
de datos crecera principalmente con el numero de snapshots `pfd*` y `tfd*`, pero
el costo dominante en RAM durante la ejecucion seguira siendo el arreglo de
particulas.

## Criterios de interpretacion

Una simulacion se considerara fisicamente util si cumple estas condiciones:

1. La energia total no muestra una deriva numerica que domine el resultado.
2. La amplitud de campo supera claramente el nivel inicial de ruido.
3. La familia de la inestabilidad coincide con la anisotropia impuesta.
4. La anisotropia que alimenta el modo disminuye durante la saturacion.
5. La estructura espacial observada es compatible con la fisica esperada.

Si una corrida no muestra crecimiento claro, no se descartara automaticamente.
En particular, los casos Weak sirven para evaluar el limite de estabilidad. Un
resultado sin crecimiento puede ser valioso si ayuda a confirmar que el umbral
elegido separa correctamente casos inestables y marginales.
