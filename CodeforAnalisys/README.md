# Guía del código de análisis de simulaciones PIC

## 1. Introducción: por qué se realizan estos análisis

Las simulaciones PIC producen campos electromagnéticos, momentos de fluido y
datos de partículas. Sin post-procesamiento, estos archivos solamente muestran
el estado numérico del plasma; no demuestran por sí solos qué inestabilidad
creció, qué energía la alimentó ni cómo alcanzó la saturación.

Esta carpeta convierte las salidas de PSC en diagnósticos físicos para responder
cinco preguntas:

1. **¿La simulación comenzó con los parámetros correctos?**
   Se validan densidad, velocidades medias, temperaturas y anisotropía.
2. **¿El plasma está en una región teóricamente inestable?**
   Se comparan `A = T_perp/T_parallel` y `beta_parallel` con los umbrales
   Mirror, Firehose e ion-cyclotron.
3. **¿La perturbación realmente crece y qué estructura forma?**
   Se estudian `delta B`, hoyos magnéticos, corrientes y mapas espaciales.
4. **¿Qué escalas espaciales dominan?**
   Se calculan espectros de potencia y números de onda dominantes.
5. **¿Cómo se redistribuye la energía de las partículas?**
   Se analizan VDF, temperaturas, energía y diagnósticos aproximados de
   transporte de calor.

El flujo general es:

```text
simulación PSC
    |
    +-- pfd.*.h5          -> campos E, B y J
    +-- pfd_moments.*.h5  -> densidad, momento y tensor de presión
    +-- prt*.h5           -> partículas individuales
    |
    v
lectura y selección del snapshot
    |
    v
cálculo de variables físicas
    |
    v
normalización, filtrado y selección de rango
    |
    v
gráficas PNG, animaciones GIF y tablas CSV
```

No se debe interpretar una sola gráfica como prueba concluyente. Una
identificación robusta combina evolución temporal, firma espacial, espectro,
anisotropía y conservación o transferencia de energía.

Los análisis implementados siguen diagnósticos usados en teoría cinética,
observaciones espaciales y simulaciones PIC o híbridas. Las citas `[R1]`,
`[R2]`, etc. remiten a la bibliografía de la sección 9. Una cita indica el
antecedente físico o metodológico del diagnóstico; no significa que este código
reproduzca exactamente la geometría, normalización o modelo numérico del paper.

## 2. Archivos de entrada

| Archivo | Variables principales | Uso |
|---|---|---|
| `pfd.<step>_p0.h5` | `Bx`, `By`, `Bz`, `Ex`, `Ey`, `Ez`, `Jx`, `Jy`, `Jz` | Fluctuaciones, estructuras, energía de campo y espectros |
| `pfd_moments.<step>_p0.h5` | `rho_s`, `p_i,s`, `t_ij,s` | Densidad, velocidad, presión, temperatura y anisotropía |
| `prt*.h5` | `q`, `m`, `w`, `px`, `py`, `pz` | VDF, momentos directos, energía y colas supratérmicas |

El campo de fondo se orienta en `z`; por tanto:

```text
dirección paralela      = z
direcciones perpendiculares = x, y
```

`PICDataReader` busca los archivos, extrae el paso de simulación, localiza los
grupos HDF5 con UID dinámico y devuelve los datasets como arreglos NumPy. Los
scripts emparejan campos y momentos por el mismo `step`.

## 3. Variables y cálculos comunes

### 3.1 Momento, velocidad y presión

Para una especie `s`:

```text
u_i = p_i = gamma v_i
v_i ~= u_i                    # régimen no relativista de estas corridas
delta u_i = u_i - <u_i>
```

Cuando se dispone de densidad y primer momento, la presión térmica central se
obtiene removiendo el flujo medio:

```text
P_ii = t_ii - p_i^2 / (n m_s)
P_parallel = P_zz
P_perp = (P_xx + P_yy) / 2
```

Algunos scripts usan directamente `t_ii` como presión. Esa aproximación es
válida cuando el flujo medio es despreciable, pero puede sobreestimar la
temperatura si aparece una deriva macroscópica.

### 3.2 Temperatura y anisotropía

Desde partículas, el cálculo usado por `plot_prt.py` es:

```text
T_parallel = m_s Var(p_z)
T_perp = m_s [Var(p_x) + Var(p_y)] / 2
A_s = T_perp / T_parallel
```

Desde momentos de grilla:

```text
T_parallel = P_parallel / n
T_perp = P_perp / n
A_s = P_perp / P_parallel
```

La anisotropía es la fuente de energía libre:

- `A_i > 1`: favorece Mirror o ion-cyclotron.
- `A_i < 1`: favorece Firehose.
- `A_e > 1`: puede favorecer Whistler.
- `A = 1`: distribución isotrópica.

### 3.3 Beta de plasma

Con `mu0 = 1` en las unidades normalizadas:

```text
P_mag = |B|^2 / 2
beta_parallel = P_parallel / P_mag
              = 2 P_parallel / |B|^2
```

`beta_parallel` mide la importancia de la presión paralela frente a la presión
magnética. Es necesario usar el campo local cuando se quiere un mapa espacial y
`B0` cuando se quiere caracterizar la condición inicial.

### 3.4 Fluctuaciones magnéticas

Para cada componente:

```text
delta B_i = B_i - <B_i>
delta B_i / B0 = (B_i - <B_i>) / B0
|delta B| / B0 =
    sqrt(delta Bx^2 + delta By^2 + delta Bz^2) / B0
```

Restar la media elimina el campo uniforme y deja la perturbación generada por
la inestabilidad.

### 3.5 Tiempo y espacio normalizados

Las conversiones están centralizadas en `psc_units.py`:

```text
Omega_ci = Zi B0 / mi
omega_pi = sqrt(n0 Zi^2 / mi)
di = 1 / omega_pi
t Omega_ci = step * dt_code * Omega_ci
x / di = x_code / di
```

Antes de ejecutar un análisis se debe seleccionar el perfil correcto:

```bash
export PSC_PROFILE=mirror_maxwellian
# mirror_kappa, firehose_maxwellian o firehose_kappa
```

## 4. Análisis y gráficas generadas

### 4.1 Brazil plot y evolución de anisotropía

**Código:** `anisotropy_analysis.py`
**Entradas:** `pfd_moments.*.h5` y `pfd.*.h5`

Variables por celda:

```text
P_perp = (Pxx + Pyy) / 2
A_i = P_perp / Pzz
beta_i_parallel = 2 Pzz / |B|^2
```

Se eliminan celdas con densidad o presión no física y valores extremos. Después
se comparan los datos con:

```text
Mirror:          A = 1 + 1/beta_parallel
Firehose:        A = 1 - 1/beta_parallel
Ion-cyclotron:   A = 1 + 0.43/beta_parallel^0.42
```

Gráficas:

- `brazil_acumulado.png`: histograma 2D de todos los puntos en
  `(beta_parallel, A_i)`, con bins logarítmicos y umbrales teóricos.
- `evolucion_temporal.png`: mediana y rango intercuartílico de `A_i` y
  `beta_parallel` contra `t Omega_ci`.
- `brazil_grid_snapshots.png`: estado del plasma en snapshots seleccionados.

**Importancia:** muestra si el plasma comienza en una región inestable y si la
evolución lo lleva hacia la estabilidad marginal. El uso de mediana y cuartiles
evita que unas pocas celdas ruidosas controlen la conclusión.

**Limitación:** este script usa `t_ii` directamente; no descuenta el flujo medio.

**Antecedentes:** Hellinger et al. comparan observaciones de `beta_parallel` y
`T_perp/T_parallel` con umbrales cinéticos Mirror, ion-cyclotron y Firehose
[R1]. Bale et al. añaden la potencia de fluctuaciones magnéticas sobre el mismo
plano anisotropía-beta [R2]. Por eso este tipo de gráfica permite relacionar la
posición del plasma en el espacio de parámetros con la actividad magnética.

### 4.2 Estructuras Mirror

**Código:** `mirror_physics.py`
**Entrada:** `pfd.*.h5`

Se extraen:

```text
Bz_normalizado = Bz / B0
Jx = corriente fuera del plano Y-Z
```

Ambos campos se suavizan con un filtro gaussiano. Los límites de color se basan
en percentiles para evitar que un valor aislado o ruido PIC oculte la estructura
principal.

**Salida:** `mirror_physics_stepXXXXXX.png`, con mapas de `Bz/B0` y `Jx`.

**Importancia:** los modos Mirror producen depresiones o realces compresivos del
campo y corrientes alrededor de sus bordes. La coincidencia espacial entre la
estructura magnética y `Jx` fortalece la identificación física.

**Antecedentes:** las simulaciones cinéticas de Porazik y Johnson estudian la
evolución espacial y saturación de estructuras Mirror [R3]. Hoilijoki et al.
usan mapas de campo y plasma para identificar ondas Mirror mediante su carácter
compresivo y la anticorrelación entre densidad y campo [R4].

### 4.3 Corriente diamagnética

**Código:** `diamagnetic_current.py`
**Entradas:** campos y momentos

Se calcula:

```text
P_perp,s = (Pxx,s + Pyy,s) / 2
J_d,s = (grad P_perp,s x B) / |B|^2
```

En el plano `Y-Z`, la componente dominante implementada es:

```text
Jdia_x,s =
    [(dP_perp,s/dy) Bz - (dP_perp,s/dz) By] / |B|^2

Jdia_total = Jdia_i + Jdia_e
```

Presión y campo se suavizan antes de calcular gradientes. Se generan tres mapas:
contribución iónica, electrónica y total, con contornos de `|B|`.

**Salidas:** `diamagnetic_plots/jdia_stepXXXXXX.png` y
`diamagnetic_plots/jdia_evolution.gif`.

**Importancia:** comprueba si los gradientes de presión alrededor de las
estructuras magnéticas producen las paredes de corriente esperadas.

**Limitación:** `np.gradient` usa separación de una celda; el resultado está en
unidades de grilla mientras no se divida explícitamente por `dy` y `dz`.

**Antecedente:** Yao et al. calculan por separado los gradientes de presión de
iones y electrones y sus corrientes diamagnéticas, y comparan la suma con la
corriente medida [R5]. El panel ión-electrón-total de este código sigue esa
lógica de separación de contribuciones.

### 4.4 Fluctuaciones del campo magnético

**Código:** `fluctuationofmagneticfiel.py`
**Entrada:** `pfd.*.h5`

Genera mapas de `delta Bx/B0`, `delta By/B0`, `delta Bz/B0` y
`|delta B|/B0`. Se puede elegir el plano `xy`, `xz` o `yz`, el índice de corte y
el ancho del filtro gaussiano.

Para comparar tiempos, el código puede calcular una escala de color global:
percentiles 1 y 99 de todos los snapshots, simetrizados alrededor de cero para
las componentes con signo.

**Salidas:** PNG por componente y snapshot, más GIF temporales.

**Importancia:** permite localizar la fase lineal, la formación de estructuras y
la saturación sin confundir el campo de fondo con la perturbación.

**Antecedentes:** Bale et al. muestran que la potencia magnética y la
compresibilidad aumentan cerca de los umbrales de anisotropía [R2]. Kunz et al.
siguen la amplitud de las fluctuaciones Firehose y Mirror y su saturación
mientras la anisotropía permanece cerca del umbral marginal [R6].

### 4.5 Análisis espectral

**Código:** `spectral_analysis.py`
**Entrada:** `pfd.*.h5`

Proceso:

1. Extrae un plano 2D.
2. Resta la media de cada componente.
3. Aplica una ventana de Hann 2D.
4. Calcula la transformada rápida de Fourier.
5. Construye la densidad espectral:

```text
PSD(k0,k1) = |FFT2(delta B)|^2 / (N0 N1)^2
k_i = 2 pi fftfreq(N_i, d_i)
```

6. Suma la PSD en anillos de `|k|` para obtener `E(k)`.
7. Ajusta una ley de potencia en la región central disponible:

```text
log10 E(k) = alpha log10 k + C
```

También guarda los modos con mayor potencia y sus valores de `k_parallel`,
`k_perp` y `|k|`.

**Salidas:** espectros 1D, PSD 2D, PSD anisotrópica, evolución del modo dominante,
`spectral_summary_*.csv` y `dominant_modes_*.csv`.

**Importancia:** identifica la longitud de onda dominante y permite distinguir
modos paralelos, oblicuos y perpendiculares.

**Limitaciones:** el rango de ajuste se selecciona automáticamente como la mitad
central del espectro; no garantiza un rango inercial físico. Además, los
espectros `perp` y `total` se calculan sobre la magnitud de la fluctuación, no
como suma independiente de las PSD de cada componente.

**Antecedentes:** Kunz et al. analizan las escalas de las fluctuaciones Mirror y
Firehose y la aparición de potencia sub-Larmor [R6]. Riquelme et al. usan
simulaciones PIC para comparar la evolución, amplitud y escalas espaciales de
Mirror e ion-cyclotron [R7]. Estos trabajos motivan mostrar PSD 2D, `E(k)` y
modos dominantes, no solamente mapas espaciales.

### 4.6 Validación de la condición inicial

**Código:** `validate_moments.py`
**Entrada:** un archivo de partículas

Se separan especies mediante el signo de la carga:

```text
iones: q > 0
electrones: q < 0
```

Se validan:

```text
n = N_particulas * cori / N_celdas
<v_i> ~= <p_i>
T_i = m Var(p_i) / beta_norm^2
A = [(Tx + Ty)/2] / Tz
kurtosis = <delta p^4> / <delta p^2>^2
```

La deriva se compara con la velocidad térmica `sqrt(T/m)`. La temperatura y
densidad se comparan con la inicialización definida por el perfil activo.

**Salida:** resumen en terminal y `validation_summary.png`.

**Importancia:** evita atribuir a una inestabilidad errores que ya estaban
presentes en el muestreo inicial, la masa, el peso de partículas o la
normalización.

**Antecedente:** Riquelme et al. verifican que la evolución de temperaturas y
anisotropías de las partículas sea consistente con la energía transferida a las
fluctuaciones y con la relajación al umbral [R7]. En este proyecto la validación
se separa como paso previo para comprobar la condición inicial antes de estudiar
esa evolución.

### 4.7 Funciones de distribución de velocidad

**Códigos:** `plot_prt.py`, `plot_vdf_3d.py`,
`plot_moments_scatter_3d.py`
**Entrada:** `prt*.h5`

Variables:

```text
v_parallel ~= pz
v_perp ~= sqrt(px^2 + py^2)
```

Las VDF se construyen con histogramas 1D o 2D normalizados. La escala logarítmica
permite observar simultáneamente el núcleo y las colas.

Distribuciones teóricas usadas:

```text
Maxwelliana:
f(v) = exp[-v^2/(2 vth^2)] / [sqrt(2 pi) vth]

Kappa:
f(v) = C(kappa,vth)
       [1 + v^2/((2 kappa - 3) vth^2)]^(-kappa)
```

`plot_prt.py` también calcula pruebas de bondad de ajuste, evolución temporal de
VDF, anisotropía y Brazil plot desde partículas. `plot_vdf_3d.py` representa el
histograma `f(v_parallel,v_perp)` como superficie suavizada.

**Importancia:** muestra cómo la inestabilidad deforma la distribución, reduce
la anisotropía o modifica las colas supratérmicas. Es el diagnóstico cinético
que no puede obtenerse solamente de promedios de fluido.

**Antecedentes:** Riquelme et al. examinan distribuciones de partículas y
dispersión en ángulo de pitch durante la saturación de inestabilidades iónicas
[R7]. Gary y Karimabadi comparan umbrales de Whistler, mirror electrónico y
Weibel para distribuciones electrónicas bi-Maxwellianas [R8]. Kim et al.
presentan evolución temporal de anisotropía, beta y energía de onda y la
contrastan con una simulación PIC de la inestabilidad Whistler [R9].

### 4.8 Partición de energía

**Código:** `plot_prt.py`

Para cada especie:

```text
E_bulk = (1/2) m |<v>|^2
E_random = (1/2) m <|v - <v>|^2>
T_promedio = (2 T_perp + T_parallel) / 3
E_internal = N (2 T_perp + T_parallel) / 2
E_deltaB = deltaB_rms^2 / 2
```

Las componentes de partículas se normalizan con la suma inicial `E0`.

**Salida:** `prt_plots/energy_partition.png`.

**Importancia:** una inestabilidad física debe transferir energía desde la
anisotropía de partículas hacia otras componentes térmicas, flujo o campos.

**Limitación:** la gráfica no es un balance global exacto: usa una muestra de
partículas y la energía magnética corresponde solamente a la fluctuación. No
incluye necesariamente toda la energía eléctrica ni todos los términos de PSC.

**Antecedentes:** Seough y Yoon representan conjuntamente la evolución temporal
de anisotropía, beta y densidad de energía de onda para estudiar crecimiento y
saturación [R10]. Kunz et al. y Riquelme et al. siguen la energía magnética de
las fluctuaciones y la relajación de la presión anisótropa en simulaciones
cinéticas [R6, R7].

### 4.9 Flujo de calor

Existen dos diagnósticos diferentes.

#### A. Tercer momento desde partículas (experimental)

**Código:** sección de flujo de calor en `plot_prt.py`

```text
q_parallel = (m/2) <delta v^2 delta vz>
q_perp = (m/2) <delta v^2 delta vperp>
```

Los grupos `R1...R4` se forman con cuantiles de `pz`, no con coordenadas
espaciales. Por eso no son regiones físicas reales. Este gráfico quedó fuera
del flujo estándar `make particles`; no debe usarse para afirmar transporte
espacial.

#### B. Proxy desde presión y velocidad de flujo

**Código:** `heat_flux_analysis.py`

Primero se calcula:

```text
v = p / (n m)
b = B / |B|
v_parallel = v . b
P_parallel = b . P . b
P_perp = [tr(P) - P_parallel] / 2
```

Después:

```text
q_parallel_proxy = P_parallel v_parallel
q_perp_proxy = P_perp |v_perp|
```

**Salidas:** mapas de anisotropía, beta y proxies de flujo, además de
`heatflux_anisotropy_evolution.png`.

**Importancia:** permite estudiar dónde el flujo macroscópico transporta energía
térmica.

**Limitación esencial:** `P v` no es el tercer momento central completo del
flujo de calor. Debe reportarse como **proxy de transporte de energía**, no como
medición exacta de `q`.

**Antecedentes:** la definición cinética rigurosa del flujo de calor como tercer
momento central de la distribución se discute explícitamente en Guo y Tang
[R11]. El efecto de inestabilidades Whistler sobre la regulación del flujo de
calor y la evolución de la VDF electrónica se estudia mediante PIC en Micera et
al. [R12]. Estas referencias respaldan mantener separados el tercer momento
calculado desde partículas y el proxy `P v` calculado desde momentos de grilla.

## 5. Cómo se generan las gráficas

Todos los scripts siguen el mismo patrón:

1. `PICDataReader.find_files()` crea un diccionario `{step: archivo}`.
2. Se conservan únicamente pasos comunes cuando se necesitan campos y momentos.
3. Se leen los datasets HDF5 y se reduce la dimensión invariante.
4. Se calculan las variables derivadas descritas arriba.
5. Se filtran densidades no físicas, divisiones por cero y valores no finitos.
6. Cuando corresponde, se aplica `gaussian_filter` para reducir ruido de
   partículas.
7. Las coordenadas se convierten a `d_i` y el tiempo a `Omega_ci^-1`.
8. Matplotlib crea mapas, series temporales, histogramas o espectros.
9. Se guardan PNG con resolución entre 150 y 300 dpi; algunos análisis ensamblan
   los PNG en GIF y otros exportan tablas CSV.

El filtrado gaussiano mejora la lectura visual y estabiliza gradientes, pero
también elimina potencia a números de onda altos. El valor de `sigma` debe
registrarse en cualquier reporte científico.

## 6. Ejecución

El `Makefile` usa por defecto:

```text
DATA_DIR = ../build/src
PARTICLE_BASENAME = prt_M_S_bM
B0_REF = 0.05
```

Comandos:

```bash
cd CodeforAnalisys

make brazil        # anisotropía y beta
make mirror        # estructuras Mirror
make diamagnetic   # corriente diamagnética
make fields        # fluctuaciones magnéticas
make spectral      # espectros y modos dominantes
make validate      # condición inicial
make particles     # VDF y diagnósticos cuantitativos de partículas
make particles-3d  # visualizaciones 3D cualitativas (opcionales)
make heatflux      # proxy de flujo de calor
make all           # todo excepto spectral y report
```

Para analizar otra corrida, puede sobrescribir la ruta sin editar el archivo:

```bash
make all DATA_DIR=/ruta/a/los/h5
```

Para otra simulacion, indique tambien el prefijo de sus particulas:

```bash
make all PARTICLE_BASENAME=prt_mirror_maxwellian
```

Seleccione también `PSC_PROFILE` y verifique `B0_REF`, tamaño de grilla,
dominio y frecuencia de salida. El perfil por defecto es `M_S_bM`, que
corresponde a `psc_M_S_bM.cxx`: malla `1408x1408`, dominio `30 d_i`,
`1000` partículas por celda y especie, campos/momentos cada `1000` pasos
y partículas cada `10000` pasos.

## 7. Criterios mínimos para interpretar resultados

Una conclusión debe comprobar, como mínimo:

- que la condición inicial medida coincide con la configurada;
- que `delta B/B0` supera de forma sostenida el nivel de ruido;
- que existe una fase de crecimiento antes de la saturación;
- que la anisotropía evoluciona hacia el umbral marginal;
- que la estructura espacial coincide con la inestabilidad propuesta;
- que el modo dominante es consistente con la escala física esperada;
- que los resultados no dependen únicamente del filtro, rango de color o
  selección de snapshots;
- que los diagnósticos aproximados se etiquetan como proxies.

`ANALISIS_ESTRUCTURA.md` contiene información adicional sobre la estructura
interna de los archivos HDF5 y ADIOS2.

## 8. Relación entre papers, análisis y gráficas

| Referencia | Qué analiza el paper | Diagnóstico relacionado en este código | Gráfica o salida |
|---|---|---|---|
| `[R1]` Hellinger et al. (2006) | Umbrales cinéticos y distribución de observaciones en `(beta_parallel, T_perp/T_parallel)` | Clasificación Mirror, Firehose e ion-cyclotron | Brazil plot |
| `[R2]` Bale et al. (2009) | Potencia y compresibilidad magnética cerca de los umbrales | Relación entre anisotropía y crecimiento de `delta B` | Brazil plot y mapas de fluctuaciones |
| `[R3]` Porazik & Johnson (2013) | Evolución no lineal y saturación de estructuras Mirror | Mapas espaciales de la componente compresiva | `mirror_physics_step*.png` |
| `[R4]` Hoilijoki et al. (2016) | Identificación de Mirror mediante campo, densidad y condición de inestabilidad | Comparación espacial de `|B|`, densidad y anisotropía | Mapas Mirror y de momentos |
| `[R5]` Yao et al. (2017) | Gradientes de presión y corriente diamagnética iónica/electrónica | `Jdia_i`, `Jdia_e`, `Jdia_total` | Panel de corriente diamagnética |
| `[R6]` Kunz et al. (2014) | Crecimiento, saturación, marginalidad y escalas de Mirror/Firehose | Evolución de `delta B`, anisotropía y espectros | Series temporales, mapas y PSD |
| `[R7]` Riquelme et al. (2015) | PIC de Mirror e ion-cyclotron: anisotropía, energía, escalas y partículas | VDF, energía, anisotropía y espectros | VDF, partición de energía y PSD |
| `[R8]` Gary & Karimabadi (2006) | Umbrales Whistler, mirror electrónico y Weibel | Interpretación de casos con `A_e > 1` | Brazil plot electrónico y espectro |
| `[R9]` Kim et al. (2017) | Regulación de anisotropía electrónica por Whistler y comparación PIC | Evolución de `A_e`, beta y fluctuaciones | Series temporales y energía de onda |
| `[R10]` Seough & Yoon (2012) | Evolución cuasilineal de anisotropía, beta y energía de onda | Crecimiento y saturación temporal | Anisotropía-beta y energía |
| `[R11]` Guo & Tang (2012) | Definición del flujo de calor como tercer momento central | `q_parallel` y `q_perp` desde partículas | Flujo de calor por población |
| `[R12]` Micera et al. (2020) | PIC de inestabilidad Whistler de flujo de calor y evolución de VDF | Transporte cinético y deformación de VDF | VDF y series de flujo de calor |

## 9. Bibliografía

- **[R1]** Hellinger, P., Trávníček, P., Kasper, J. C., & Lazarus, A. J.
  (2006). *Solar wind proton temperature anisotropy: Linear theory and
  WIND/SWE observations*. Geophysical Research Letters, 33, L09101.
  <https://doi.org/10.1029/2006GL025925>
- **[R2]** Bale, S. D., Kasper, J. C., Howes, G. G., Quataert, E., Salem, C.,
  & Sundkvist, D. (2009). *Magnetic fluctuation power near proton temperature
  anisotropy instability thresholds in the solar wind*. Physical Review
  Letters, 103, 211101. <https://doi.org/10.1103/PhysRevLett.103.211101>
- **[R3]** Porazik, P., & Johnson, J. R. (2013). *Gyrokinetic particle
  simulation of nonlinear evolution of mirror instability*. Journal of
  Geophysical Research: Space Physics, 118. <https://doi.org/10.1002/2013JA019308>
- **[R4]** Hoilijoki, S., Palmroth, M., Walsh, B. M., et al. (2016). *Mirror
  modes in the Earth's magnetosheath: Results from a global hybrid-Vlasov
  simulation*. Journal of Geophysical Research: Space Physics, 121, 4191-4204.
  <https://doi.org/10.1002/2015JA022026>
- **[R5]** Yao, S. T., Shi, Q. Q., Yao, Z. H., et al. (2017). *A direct
  examination of the dynamics of dipolarization fronts using MMS*. Journal of
  Geophysical Research: Space Physics, 122.
  <https://doi.org/10.1002/2016JA023401>
- **[R6]** Kunz, M. W., Schekochihin, A. A., & Stone, J. M. (2014).
  *Firehose and mirror instabilities in a collisionless shearing plasma*.
  Physical Review Letters, 112, 205003.
  <https://doi.org/10.1103/PhysRevLett.112.205003>
- **[R7]** Riquelme, M. A., Quataert, E., & Verscharen, D. (2015).
  *Particle-in-cell simulations of continuously driven mirror and ion
  cyclotron instabilities in high beta astrophysical and heliospheric plasmas*.
  The Astrophysical Journal, 800, 27.
  <https://doi.org/10.1088/0004-637X/800/1/27>
- **[R8]** Gary, S. P., & Karimabadi, H. (2006). *Linear theory of electron
  temperature anisotropy instabilities: Whistler, mirror, and Weibel*. Journal
  of Geophysical Research: Space Physics, 111.
  <https://doi.org/10.1029/2006JA011764>
- **[R9]** Kim, S., Yoon, P. H., Choe, G. S., & Wang, L. (2017). *Electron
  temperature anisotropy regulation by whistler instability*. Journal of
  Geophysical Research: Space Physics, 122, 4410-4419.
  <https://doi.org/10.1002/2016JA023558>
- **[R10]** Seough, J., & Yoon, P. H. (2012). *Quasilinear theory of
  anisotropy-beta relations for proton cyclotron and parallel firehose
  instabilities*. Journal of Geophysical Research: Space Physics, 117.
  <https://doi.org/10.1029/2012JA017645>
- **[R11]** Guo, Z., & Tang, X.-Z. (2012). *Parallel heat flux from low to high
  parallel temperature along a magnetic field line*. Physical Review Letters,
  108, 165005. <https://doi.org/10.1103/PhysRevLett.108.165005>
- **[R12]** Micera, A., Zhukov, A. N., López, R. A., et al. (2020).
  *Particle-in-cell simulation of whistler heat-flux instabilities in the solar
  wind: Heat-flux regulation and electron halo formation*. The Astrophysical
  Journal Letters, 903, L23. <https://doi.org/10.3847/2041-8213/abc0e8>
