# Auditoria fisica y trabajo pendiente para publicacion

Fecha: 2026-09-09. Revision del codigo, configuraciones, informe LaTeX y fuentes
primarias. No se ha realizado una revision bibliografica exhaustiva.

## Seguimiento (2026-09-16)

Re-verificacion linea por linea del codigo (C++ y Python) y de los tests
frente a esta auditoria. Lo que sigue es el estado real, no una promesa:

**Cerrado y verificado en codigo** (existe, tiene test, el test pasa):
`DiagEnergies` encendido (`PSC_ENERGIES_EVERY=500` por defecto) y su consumidor
`energy_conservation.py` (4 tests); escala Kappa `sqrt((kappa-1.5)/kappa)` en
`linear_theory.py`; `solve()` ya no devuelve la ultima iteracion sin residuo
chico; `load_theory()` separa por polarizacion y rechaza k duplicado;
`compare_physical_cases.py` filtra por `fit_ok`; `b_crit`/`a_max` distinguidos
y autoconsistentes en `liouville_kappa.py` (21 self-checks); error relativista
u-vs-v cuantificado por `particle_kinematics_validity()`; geometria por
corrida resuelta y validada contra el snapshot real (`run_geometry.py`, 10
tests) en vez de asumida de un solo perfil global; resolucion grid-vs-perfil
verificada (`check_resolution.py`); identificacion modal sin inferir el
nombre de la inestabilidad, con manejo explicito de Nyquist, sidelobes de
ventana y degrowth (`dispersion_modes.py`, 25 tests); evolucion de kappa
cruzada con la fase lineal medida (`kappa_evolution.py`); verificacion de que
el estado a t=0 medido coincide con lo que declara `CASE`
(`check_initial_conditions.py`, nuevo en esta revision, 4 tests, wireado en
`make manifest`).

**Seguia sin hacerse y se corrigio en esta revision:** no existia ningun
chequeo automatico de que los datos en `DATA_DIR` correspondan al `CASE`
declarado mas alla de la geometria (P0 "Reproducir inicializacion"). Se anadio
`check_initial_conditions.py`: lee el primer snapshot de particulas, campos y
momentos, mide T_par/T_perp por especie, B0 y n, y falla si no coinciden con
`psc_units.py` (tolerancia 15% en temperaturas por ruido de muestreo finito
a t=0, 5% en B0 por ser condicion inicial determinista). Wireado como
prerrequisito de `manifest`.

**Sigue abierto, sin cambios:** absolutamente ninguna de las correcciones de
arriba se ha probado contra una corrida real. Los dos `diag.asc` que existen
en esta copia (`./diag.asc` y `corridas_locales/mi_prueba/diag.asc`) estan
vacios y con fecha anterior al fix de energia (junio y julio, el fix es del
9 de septiembre) — ni siquiera una corrida local de prueba ha pasado por el
diagnostico nuevo. Tampoco hay estudio de convergencia (dx/dt/ppc por
separado), barrido de tamano de caja, ni ramas oblicuas de teoria lineal
(ALPS sigue sin instalarse). El estado fisico de las simulaciones — ¿conserva
energia?, ¿que modo crecio?, ¿converge? — sigue sin respuesta porque nada de
esto corrio aun sobre datos reales. Ver el detalle marcado `[CERRADO
2026-09-16]` en cada seccion de abajo para lo que ya no bloquea, y lo que
sigue bloqueando.

## Dictamen y alcance

La infraestructura cubre anisotropia, campos, espectros, crecimiento modal,
polarizacion, VDF y estructuras. Sin embargo, **todavia no permite certificar
las conclusiones fisicas ni dar el trabajo por listo para un paper**.
La prioridad es cerrar unidades, validacion numerica, identificacion modal y
comparaciones controladas. Generar mas mapas no resuelve esas carencias.

En esta copia no estan las tablas de produccion citadas en
`reporte_analisis_anisotropias.tex`. `analysis_results/` no aporta esas series;
hay dos CSV de pruebas bajo `test_results/F_S_bM_local/04_spectra/` y salidas de
una prueba local en `corridas_locales/mi_prueba`. No se ha supuesto que esa
prueba corresponda al caso mirror moderate. Los numeros historicos del informe
no se han vuelto a medir y sus PDFs no se han regenerado.

## Errores corregidos en esta revision

| Problema | Correccion | Productos que deben regenerarse |
|---|---|---|
| `psc_units.py` igualaba `VA=B0` con `mi=200`, `n0=1` | `VA=B0/sqrt(n0*mi)=Omega_ci*di`; se conserva el B0 simulado | Velocidades normalizadas, VDF, conversion omega/omega_pi y teoria que utiliza c/vA |
| Mirror usaba `1+1/beta_parallel` | Referencia `A=(1+sqrt(1+4/beta_parallel))/2`, identificada como limite bi-Maxwelliano con electrones frios | Brazil plots, umbrales y distancias a las curvas |
| `moment_thermal_maps` usaba Pzz aunque leyera B local | Tensor central completo y proyeccion local, compartida con el analisis de anisotropia | Mapas y estadisticas termicas integradas |
| Se restaba una tendencia electronica del presupuesto energetico | Se conserva la tendencia observada y se exporta `E_proxy`, sin total corregido ni afirmacion de conservacion | `energy_table.csv`, comparaciones y figuras de energia |
| Flujo perpendicular usaba la media de una cantidad siempre positiva | Media de las componentes del tercer momento antes de calcular su modulo | Columnas `q_*_particle` y comparaciones |
| Documentacion indicaba dx/lambda_De=3.78 | Valor corregido a 8.68 para el perfil 576, beta_e_parallel=1 | Descripcion de metodos y tablas de resolucion |

El cambio de VA multiplica las velocidades expresadas en vA por sqrt(200)
respecto a la normalizacion anterior, y los cocientes T/(mi*vA^2) por 200.
No cambia B0, las temperaturas en unidades de codigo ni Omega_ci.
La clave historica `vA_over_c` dentro de los perfiles sigue representando la
entrada B0 del C++; las constantes exportadas `VA` y `VA_OVER_C` son fisicas.

La referencia mirror resulta de beta_perp*(A-1)=1 y beta_perp=A*beta_parallel.
No es el umbral mirror CGL ni incluye electrones calientes o distribuciones
Kappa. La condicion multiespecie contiene terminos adicionales.
[Hellinger (2007), ecuaciones 1 y 16](https://space.asu.cas.cz/~helinger/hell07.pdf).

Si se reutilizan SOLO los numeros impresos del informe, A=1.9996 -> 1.2975 y
beta_parallel=4.994 -> 5.589, la referencia corregida es 1.1710 -> 1.1549 y
la distancia es 0.8286 -> 0.1426. Esto indica acercamiento a esa referencia;
no demuestra estabilidad marginal ni identifica por si solo el modo mirror.

## Problemas que siguen abiertos

### 1. Conservacion y convergencia: bloqueantes

`physical_diagnostics.py` utiliza particulas de la ventana prt y campos del
dominio completo, supone n=1 para convertir temperaturas en densidad de energia,
y omite energia electrica y movimiento colectivo electronico. Su suma es un
proxy parcial. Su variacion NO es el error energetico de la simulacion.
Una tendencia electronica lineal e isotropa no permite decidir si el origen es
fisico o numerico; sustraerla tampoco valida la corrida.

**[CERRADO 2026-09-16, sin validar con datos]** `DiagEnergies` ya esta
encendido (`PSC_ENERGIES_EVERY=500` por defecto en `psc_anisotropy_case.hxx`
y en los scripts de `cosma_jobs/simulacion/`), `preserve_energy_diagnostic.sh`
conserva `diag.asc` entre restarts, y `energy_conservation.py` lo lee sin
detrending y reporta E_E, E_B, E_i, E_e, E_total y el cambio relativo (4 tests
en `test_energy_conservation.py`, todos pasan). El viejo proxy de
`physical_diagnostics.py` quedo reetiquetado honestamente como `E_proxy` /
`is_conservation_diagnostic: False`, ya no se presenta como conservacion.
Pendiente: revisar el efecto del corrector Marder y los residuos de Gauss,
continuidad y div B — eso no esta cubierto por `DiagEnergies`. Y sobre todo:
**ningun `diag.asc` real existe todavia.** Los dos que hay en esta copia
(`./diag.asc`, `corridas_locales/mi_prueba/diag.asc`) estan vacios y son de
antes del fix. No hay un solo numero real de conservacion global medido con
el diagnostico correcto — ni de una corrida local de prueba, ni de COSMA.

Para 20 di / 576, mi=200, B0=0.08, beta_e_parallel=1:

```text
di = 14.1421 d_e
dx = 0.491046 d_e
Te_parallel = 0.0032
lambda_De = sqrt(Te_parallel) = 0.0565685 d_e
dx/lambda_De = 8.68056
dt*omega_pe = 0.329861
dt*Omega_ce = 0.0263889
```

Esto documenta subresolucion de Debye; no prueba por si solo que toda la
evolucion sea espuria. Exige medir convergencia de calentamiento, gamma,
amplitud saturada y anisotropia final. No hay un limite universal de dx/lambda_D
que sustituya esa comprobacion. Los perfiles whistler tienen beta_e diferente.

### 2. Teoria lineal: aun no validada para la comparacion completa

`linear_theory.py` solo resuelve propagacion paralela. No proporciona mirror
ni firehose oblicuo — **esto sigue abierto**, ver ALPS mas abajo.

**[CERRADO 2026-09-16]** `ParallelDispersion` ya aplica
`sqrt((kappa-1.5)/kappa)` a `theta_parallel` para la parte Kappa (linea 145 de
`linear_theory.py`), en vez de reusar `sqrt(beta_parallel)` sin ese factor. La
distincion bi-Kappa vs product-bi-Kappa
[Lazar et al. (2011)](https://academic.oup.com/mnras/article/410/1/663/1038700)
sigue sin verificarse contra una referencia independiente.

**[CERRADO 2026-09-16]** `solve()` ahora devuelve NaN si el residuo no baja de
`residual_tol`, no la ultima iteracion del bucle agotado (`self.last_solve`
registra `converged`, `residual`, `iterations`). `load_theory()` en
`polarization_dispersion.py` ya separa por columna `polarization` y rechaza k
duplicado dentro de una misma rama en vez de interpolar dos ramas como una
curva unica.

Para los modos oblicuos, una opcion publicada es ALPS, que acepta VDF
girotropicas arbitrarias y angulos de propagacion generales. **Sigue sin
instalarse ni ejecutarse.** Evaluar su uso con los parametros y VDF realmente
inicializados antes de afirmar nada sobre mirror/firehose oblicuo en el paper.
[Documentacion de ALPS](https://danielver02.github.io/ALPS/).

### 3. Identificacion de modos y crecimiento

El nombre del ejecutable no identifica el modo que ha crecido. Medir juntos
gamma(k_parallel,k_perp), omega_r, angulo, compresibilidad, polarizacion y
correlacion densidad-|B|. Separar mirror/EMIC y firehose paralelo/oblicuo.
La distincion entre ramas tambien es central en
[Hellinger et al. (2006)](https://space.asu.cas.cz/~helinger/hellal06.pdf).

Los analisis de gamma por modo y pruebas sinteticas ya existen. Falta demostrar
su validez en produccion: ventana lineal previa a la relajacion apreciable,
potencia sobre el ruido, incertidumbres y sensibilidad al intervalo de ajuste.
Para potencia P, gamma=(1/2)*d(log P)/dt. El ajuste de una RMS global o de un
anillo mezcla modos.

**[CERRADO 2026-09-16]** `compare_physical_cases.py` ya filtra por `fit_ok`
antes de usar gamma (`gamma_raw`/`gamma_fit_ok` quedan separados en la tabla).
`growth_rate_map.py` ahora acepta `--component parallel|perp` y
`--t-start/--t-end` para acotar la ventana lineal; el Makefile corre ambas
componentes. `dispersion_modes.py` (25 tests) maneja explicitamente Nyquist,
sidelobes de ventana, degrowth y no infiere el nombre de la inestabilidad —
son exactamente los puntos que senala el parrafo siguiente. Todo esto sigue
**sin correr contra una corrida de produccion real** (ver Seguimiento arriba).

En L=20 di el primer k es 2*pi/L=0.31416/di. Un pico en esa celda puede estar
limitado por la caja. En L=40 di es 0.15708/di. Comparar a dx constante.
La ventana espacial Hann mezcla celdas vecinas de Fourier: comprobar los modos
del dominio periodico completo sin esa ventana. El zero-padding temporal no
aumenta la resolucion fisica. Para mirror aperiódico, omega_r cercano a cero
es una comprobacion de compatibilidad, no una rama propagante.

Con salidas cada 500 pasos en el perfil 576, Delta_t_out*Omega_ce=13.1944 y
omega_Nyquist/Omega_ce=0.2381. Un whistler por encima de ese limite no puede
recuperarse de esas salidas. Elegir cadencia usando la frecuencia predicha.

### 4. Particulas, presion y transporte

El escritor HDF5 guarda `px,py,pz = prt.u()`, es decir, momento por unidad
de masa, gamma*v; no velocidad exacta ni momento m*v. El deposito de PSC usa
M_ab=<m*u_a*v_b>, con v=u/sqrt(1+u^2). Varios analisis, incluido el integrado,
tratan u como v y restan p_a*p_b/(n*m), que es una aproximacion no relativista.

**[CERRADO 2026-09-16, parcial]** `particle_kinematics_validity()` en
`physical_diagnostics.py` ya cuantifica ese error por especie: compara la
energia no relativista aproximada contra `u^2/(gamma+1)` exacto, la presion
diagonal `m*<u*v>` central contra `m*<u*u>`, y la fraccion de particulas con
`|u|>c`. Esta enganchada en el pipeline principal. Lo que sigue faltando es
usar ese numero: si el error resulta significativo para alguna especie o cola,
reconstruir esas magnitudes con la convencion relativista correcta en vez de
solo reportar el error — no se ha hecho, y no se sabra si hace falta hasta
correr esto contra datos reales.

Las temperaturas de particulas integradas siguen referidas a B0 y a una deriva
media de la ventana; los mapas de momentos corregidos usan B local y deriva
por celda. No compararlos como estimadores identicos. Comparar primero misma
region, proyeccion, pesos y sustraccion de flujo. Medir tambien la diferencia
entre media de A y cociente de presiones medias, y la fraccion de celdas que
el filtrado de `anisotropy_analysis.py` descarta.

El flujo de calor no relativista es q=(m/2)*integral |v-U|^2*(v-U)*f*d^3v.
P_parallel*U_parallel es un proxy convectivo, no ese tercer momento. El flujo
de particulas corregido es por particula, respecto a B0, dentro de la ventana;
todavia requiere densidad y definiciones locales para medir transporte espacial.
Para Kappa=3, los errores gaussianos del tercer momento requieren especial
cuidado: el sexto momento de la Kappa ideal no existe. Declarar truncamiento,
sensibilidad al rango de velocidades e incertidumbre entre realizaciones.

### 5. Liouville y kappa local: hipotesis, no resultado demostrado

El modelo supone energia y momento magnetico conservados y una conexion de
particulas pasantes con una VDF de referencia. Deben medirse estacionariedad,
E_parallel/potencial, rho/L_B y evolucion de mu en trayectorias para aplicar
esa interpretacion a estructuras concretas. **Esto sigue sin hacerse:** PSC no
esta guardando trayectorias de particulas individuales entre pasos (solo
snapshots), asi que "evolucion de mu en trayectorias" no se puede medir con
la salida actual sin instrumentar ese tracking primero.

La conservacion del exponente en la expresion de la VDF pasante no garantiza
que el estimador kappa_eff de una VDF recortada, mezclada o parcialmente atrapada
permanezca constante. Comparar datos y modelo con el mismo muestreo y estimador.
Los cierres empty/own/flat son posibilidades del modelo; su orden temporal
necesita evidencia independiente.

**[CERRADO 2026-09-16]** `b_crit=1-1/A0` y `a_max=1/A0` ya estan diferenciados
de forma consistente en todo `liouville_kappa.py` (antes habia docstrings que
los confundian). El self-test (`--self-test`, 21 comprobaciones) verifica
explicitamente `a_max` consistente con `b_crit`, que `theta_perp_eff^2` es
finito por encima de `b_crit` e indefinido por debajo, y que la extension
`own` no esta definida para `a > a_max`. Pasa completo. Esto valida la
consistencia interna del modelo, no la hipotesis fisica de arriba — esa
segunda parte sigue siendo una hipotesis sin datos reales que la prueben.

## Analisis que faltan, en orden de prioridad

| Prioridad | Trabajo | Evidencia minima que debe entregar | Datos | Estado 2026-09-16 |
|---|---|---|---|---|
| P0 | Reproducir inicializacion | Tabla medida de n, B0, Tpar/Tperp, beta y VDF de ambas especies; dt real y region prt | Campos, momentos, particulas iniciales y log | Chequeo automatico construido (`check_initial_conditions.py`); **no corrido contra ninguna corrida real** |
| P0 | Conservacion global | E_E+E_B+E_i+E_e y deriva sin restas; residuos de campo | diag.asc completo o diagnostico de todas las particulas | Diagnostico e instrumentacion listos y probados con datos sinteticos; **cero `diag.asc` reales existen** |
| P0 | Convergencia | Cambios de gamma, saturacion, A final y calentamiento al variar dx, dt y PPC separadamente | Corridas de control | Sin empezar, requiere corridas dedicadas en COSMA |
| P0 | Dominio y estadistica | Caja mayor a dx fijo; dispersion entre semillas | Corridas adicionales o existentes equivalentes | Sin empezar, requiere corridas dedicadas en COSMA |
| P0 | Identificacion y teoria | Misma rama PIC/teoria en k, omega, gamma, polarizacion; error del ajuste | Series de campo y solver validado | Herramientas construidas y probadas (`dispersion_modes.py`, `linear_theory.py`); **no corridas contra produccion**; rama oblicua (ALPS) sin instalar |
| P0 | Maxwelliana vs Kappa | Misma beta fisica, A, B0, masas, malla, caja y electrones; diferencias con incertidumbre | Pares equivalentes | Pares definidos abajo; no confirmado que se hayan re-corrido con las correcciones de unidades/umbral de esta auditoria |
| P1 | Saturacion y estructuras | Balance de presiones, n-|B|, profundidad, escala y duracion de estructuras | Campos y momentos sincronizados | Sin empezar |
| P1 | VDF local y atrapamiento | Pasantes/atrapadas, A local, kappa_eff y sensibilidad a seleccion/rango | Particulas con posiciones; trayectorias si se afirma mecanismo | Modelo internamente consistente (`liouville_kappa.py`, 21/21 self-tests); tracking de trayectorias no instrumentado, hipotesis fisica sin probar |
| P1 | Intercambio de energia | J_s dot E, cambios de energia por especie y efectos numericos | Campos, corrientes y energia global | Sin empezar |
| P2 | Transporte y generalizacion | Tercer momento local, otro mi/me o 3D segun la afirmacion | Nuevos diagnosticos/corridas si procede | Sin empezar |

P0 bloquea las afirmaciones principales. P1 es necesario si el mecanismo
propuesto es atrapamiento o modificacion local de la VDF. P2 depende del alcance:
no hace falta convertir el primer paper en un estudio de todas las familias.
Como primer diseno de convergencia, usar base mas dos refinamientos de malla,
un control temporal y uno de PPC, una caja mayor, y al menos tres realizaciones
del par central. Ajustar el numero de semillas a la incertidumbre observada.
La tolerancia numerica debe ser menor que el efecto fisico que se intenta medir;
estos numeros son una propuesta de trabajo, no reglas de una revista.

Pares definidos en esta copia: mirror moderate A_i=2 con
`mirror_bimaxwellian_moderate` y `mirror_bikappa3_moderate`; mirror strong A_i=3
con su Maxwelliana y `mirror_bikappa3/5`; firehose strong beta=10, A_i=0.1 con
su Maxwelliana y `firehose_bikappa3/5`. No comparar Kappa firehose strong con
Maxwelliana moderate beta=6, A_i=0.3 para atribuir diferencias solo a Kappa.
Confirmar tambien si cambia la distribucion electronica: la seleccion Kappa
compartida del cargador no constituye por si sola un barrido exclusivo de iones.

## Contribucion y figuras del paper

Ya existe una comparacion de simulaciones hibridas 2D y teoria cuasilineal para
mirror/EMIC con protones bi-Kappa. Por tanto, mostrar relajacion de anisotropia
al variar Kappa no establece por si solo novedad.
[Lopez et al. (2023), ApJ 954, 191](https://doi.org/10.3847/1538-4357/aceb5b).

Una pregunta concreta, por comprobar, es: a presion y anisotropia iniciales
iguales, como cambia la distribucion local de particulas en estructuras
compresivas y que parte de ese cambio explica un modelo adiabatico con
poblacion atrapada. Los electrones cineticos pueden ser otra contribucion si
su efecto se separa del calentamiento numerico. No afirmar novedad hasta
completar una busqueda bibliografica centrada en la pregunta elegida.

Propuesta de seis figuras, con tablas numericas y barras de error:

1. VDF iniciales de ambas especies y parametros medidos del par comparado.
2. Validacion energetica y convergencia de los observables principales.
3. gamma(kpar,kperp), identificacion modal y comparacion cuantitativa con teoria.
4. A(t), beta(t), delta B(t) y saturacion con incertidumbres entre realizaciones.
5. Estructuras: |B|, densidad, presion y region real de muestreo de particulas.
6. VDF/kappa_eff local y predicciones de cierres con su dominio de validez.

Un resultado negativo tambien puede sostener una contribucion: por ejemplo,
que la variacion aparente de kappa_eff se explique por muestreo/mezcla y no
requiera cambiar el exponente intrinseco. Esa conclusion necesita controles.

## Verificacion y regeneracion

**Actualizado 2026-09-16.** `python3 -m unittest discover -p "test_*.py"` desde
`CodeforAnalisys` corre 37 tests (incluye energia, geometria por corrida,
condiciones iniciales, consistencia fisica, teoria lineal y comparacion de
teorias) y los 37 pasan. Ademas, `test_dispersion_modes.py` tiene 25 pruebas
en estilo pytest que `unittest discover` no recoge (mezcla de convenciones en
la suite, pendiente de unificar); correrlas con
`python3 -m pytest test_dispersion_modes.py` — tambien pasan las 25. El
self-test de `liouville_kappa.py --self-test` pasa sus 21 comprobaciones.
Esto valida consistencia interna del codigo con datos sinteticos. **No valida
ninguna corrida real** — ver Seguimiento al inicio de este documento.

Desde `CodeforAnalisys`:

```bash
MPLCONFIGDIR=/tmp/psc-matplotlib python3 -m unittest discover -p "test_*.py" -v
MPLCONFIGDIR=/tmp/psc-matplotlib python3 -m pytest test_dispersion_modes.py -q
```

**Antes de gastar tiempo de computo de COSMA:** validar el camino completo con
una corrida local corta (unos cientos de pasos alcanza). Ahora mismo los dos
`diag.asc` que existen en el repo (`./diag.asc`,
`corridas_locales/mi_prueba/diag.asc`) estan vacios — ni siquiera eso se ha
probado. El orden razonable es: (1) corrida local corta con
`PSC_ENERGIES_EVERY` bajo, (2) `make manifest` (ya corre
`check_initial_conditions.py`) y `make energy` sobre esa corrida para
confirmar que el camino entero produce numeros sensatos, (3) recien entonces
lanzar produccion en COSMA con confianza en la instrumentacion.

Cuando esten accesibles los datos de produccion, regenerar en un directorio
nuevo con CASE exacto y los targets `manifest`, `brazil`, `physics`, `spectral`
y `growth-map` pertinentes. Conservar los CSV originales y marcar sus figuras
como anteriores a esta auditoria. El manifiesto nuevo registra version de
convenciones 2, VA, Omega_ci, di y dt del perfil; el dt aun debe cotejarse con
el log real. No mezclar columnas antiguas `E_total`/`energy_error` con `E_proxy`.
