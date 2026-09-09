# Auditoria fisica y trabajo pendiente para publicacion

Fecha: 2026-09-09. Revision del codigo, configuraciones, informe LaTeX y fuentes
primarias. No se ha realizado una revision bibliografica exhaustiva.

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

Usar el diagnostico global existente `DiagEnergies`: integra los campos y
suma m*(gamma-1) con pesos de ambas especies y reduccion MPI. Su activacion
se controla con `PSC_ENERGIES_EVERY`; el valor por defecto en
`psc_anisotropy_case.hxx` es 0. Configurarlo con cadencia suficiente en las
nuevas corridas y conservar `diag.asc` por segmento de restart: el constructor
abre el archivo en modo escritura y puede reemplazar el segmento anterior.
El `diag.asc` de la raiz de esta copia esta vacio.

Reportar E_E, E_B completo, E_i, E_e, E_total y su cambio sin detrending.
Revisar tambien el efecto del corrector Marder, activo cada 100 pasos, y los
residuos de Gauss, continuidad y div B. En dominio periodico no hay una entrada
externa de energia que pueda suponerse para justificar una deriva.

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
ni firehose oblicuo. Su self-test pasa tres comprobaciones basicas; la del
firehose da beta_parallel-beta_perp=1.963 frente al limite 2, con tolerancia
finita. Esto no valida todas las ramas ni la parte Kappa.

El cargador Kappa usa una mezcla de gaussianas con varianza T/m fijada. Con
la definicion de Z_kappa del solver, la escala de su argumento debe derivarse
de esa misma distribucion: theta_parallel^2=(2*kappa-3)*T_parallel/(kappa*m).
Actualmente `ParallelDispersion` usa sqrt(beta_parallel) tambien para Kappa,
sin el factor sqrt((kappa-1.5)/kappa). La susceptibilidad completa debe
contrastarse con una referencia independiente para la misma convencion de T.
La distincion entre distribuciones bi-Kappa y product-bi-Kappa tambien importa.
[Lazar et al. (2011)](https://academic.oup.com/mnras/article/410/1/663/1038700).

Ademas, `solve()` puede devolver la ultima iteracion al agotar el bucle sin
comprobar que el residuo sea pequeno. `load_theory()` en
`polarization_dispersion.py` ignora la columna de polarizacion; el generador
puede escribir dos ramas para el mismo k. No interpolar ese CSV como una curva
unica. Guardar rama, residuo, convergencia y continuidad modal antes de comparar.

Para los modos oblicuos, una opcion publicada es ALPS, que acepta VDF
girotropicas arbitrarias y angulos de propagacion generales. Evaluar su uso con
los parametros y VDF realmente inicializados; no se ha instalado ni ejecutado
en esta revision. [Documentacion de ALPS](https://danielver02.github.io/ALPS/).

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
anillo mezcla modos. `compare_physical_cases.py` ademas lee el gamma global sin
comprobar `fit_ok`; excluir ajustes rechazados antes de usar esa comparacion.

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
Cuantificar su error para cada especie y para las colas; si importa, reconstruir
las magnitudes con la misma convencion relativista y marco de referencia.

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

`liouville_kappa.py` y `vdf_spatial.py` tenian cambios del usuario y no se han
modificado. El modelo supone energia y momento magnetico conservados y una
conexion de particulas pasantes con una VDF de referencia. Deben medirse
estacionariedad, E_parallel/potencial, rho/L_B y evolucion de mu en trayectorias
para aplicar esa interpretacion a estructuras concretas.

La conservacion del exponente en la expresion de la VDF pasante no garantiza
que el estimador kappa_eff de una VDF recortada, mezclada o parcialmente atrapada
permanezca constante. Comparar datos y modelo con el mismo muestreo y estimador.
Los cierres empty/own/flat son posibilidades del modelo; su orden temporal
necesita evidencia independiente. La positividad de theta_perp_eff al extender
la expresion a todo el espacio de velocidades no es automaticamente una cota
universal de profundidad de estructuras PIC. Ademas, algunas docstrings
confunden b_crit=1-1/A0 con a_max=1/A0, donde a=1-b.

## Analisis que faltan, en orden de prioridad

| Prioridad | Trabajo | Evidencia minima que debe entregar | Datos |
|---|---|---|---|
| P0 | Reproducir inicializacion | Tabla medida de n, B0, Tpar/Tperp, beta y VDF de ambas especies; dt real y region prt | Campos, momentos, particulas iniciales y log |
| P0 | Conservacion global | E_E+E_B+E_i+E_e y deriva sin restas; residuos de campo | diag.asc completo o diagnostico de todas las particulas |
| P0 | Convergencia | Cambios de gamma, saturacion, A final y calentamiento al variar dx, dt y PPC separadamente | Corridas de control |
| P0 | Dominio y estadistica | Caja mayor a dx fijo; dispersion entre semillas | Corridas adicionales o existentes equivalentes |
| P0 | Identificacion y teoria | Misma rama PIC/teoria en k, omega, gamma, polarizacion; error del ajuste | Series de campo y solver validado |
| P0 | Maxwelliana vs Kappa | Misma beta fisica, A, B0, masas, malla, caja y electrones; diferencias con incertidumbre | Pares equivalentes |
| P1 | Saturacion y estructuras | Balance de presiones, n-|B|, profundidad, escala y duracion de estructuras | Campos y momentos sincronizados |
| P1 | VDF local y atrapamiento | Pasantes/atrapadas, A local, kappa_eff y sensibilidad a seleccion/rango | Particulas con posiciones; trayectorias si se afirma mecanismo |
| P1 | Intercambio de energia | J_s dot E, cambios de energia por especie y efectos numericos | Campos, corrientes y energia global |
| P2 | Transporte y generalizacion | Tercer momento local, otro mi/me o 3D segun la afirmacion | Nuevos diagnosticos/corridas si procede |

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

Pasaron 13 pruebas de normalizacion, simetria del flujo, proyeccion rotada,
presupuesto parcial, FFT, velocidad de fase y crecimiento oblicuo sintetico.
Tambien pasaron las tres comprobaciones internas de `linear_theory.py`, con
las limitaciones indicadas arriba. Se uso `python3` del sistema porque la
`.venv` local no tiene scipy. Esto valida cambios de codigo, no las corridas.

Desde `CodeforAnalisys`:

```bash
MPLCONFIGDIR=/tmp/psc-matplotlib python3 -m unittest -v test_physical_consistency.py test_spectral_analysis.py test_dispersion_analysis.py test_temperature_anisotropy_dispersion.py test_growth_rate_map.py
```

Cuando esten accesibles los datos de produccion, regenerar en un directorio
nuevo con CASE exacto y los targets `manifest`, `brazil`, `physics`, `spectral`
y `growth-map` pertinentes. Conservar los CSV originales y marcar sus figuras
como anteriores a esta auditoria. El manifiesto nuevo registra version de
convenciones 2, VA, Omega_ci, di y dt del perfil; el dt aun debe cotejarse con
el log real. No mezclar columnas antiguas `E_total`/`energy_error` con `E_proxy`.
