# Análisis de salidas PSC

Esta carpeta contiene la pipeline mantenida para analizar las corridas de
anisotropía `M_*_bM`, `F_*_bM`, `W_*_bM` y los casos Kappa/Maxwellian.

## Entrada esperada

Cada directorio de datos debe contener una sola corrida PSC:

```text
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_<caso>.<step>.h5
```

Los checkpoints ADIOS2 (`checkpoint_<step>.bp/`) son para restart de la
simulación. La pipeline de análisis trabaja con las salidas HDF5 de campos,
momentos y partículas.

## Uso rápido

Desde `CodeforAnalisys`:

```bash
make show-inputs DATA_DIR=/ruta/a/run CASE=M_S_bM
make analysis DATA_DIR=/ruta/a/run CASE=M_S_bM
```

También se puede ejecutar por caso:

```bash
make F_M_bM DATA_DIR=/ruta/a/F_M_bM
```

Para ejecutar únicamente el análisis espectral:

```bash
make spectral DATA_DIR=/ruta/a/run CASE=F_S_bM_local
```

El script detecta automáticamente el plano físico no degenerado (`xy`, `xz`
o `yz`) y toma el espaciado en unidades de \(d_i\) del perfil seleccionado
mediante `PSC_PROFILE`. El mismo target genera
`dispersion_density_<plano>_perp_absolute.png`, un mapa de densidad modal
\(\omega/\Omega_{ci}\) frente a \(|v_{\rm ph}|/v_A\), y superpone con puntos
negros las crestas de mayor potencia.

La FFT temporal se rellena con ceros para dibujar las crestas con continuidad;
esto interpola el espectro, pero no aumenta el número de frecuencias físicamente
independientes determinado por la cantidad y cadencia de snapshots.

Para generar solamente ese diagrama:

```bash
make dispersion DATA_DIR=/ruta/a/run CASE=F_S_bM_local
```

Las pruebas de regresión se ejecutan con:

```bash
../.venv/bin/python -m unittest -v \
  test_spectral_analysis.py test_dispersion_analysis.py
```

## Casos soportados

| `CASE` | Inestabilidad | Especie | Parámetros iniciales |
|---|---|---|---|
| `M_S_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=3.0` |
| `M_M_bM` | Mirror | ion | `beta_i_parallel=5`, `A_i=2.0` |
| `M_W_bM` | Mirror | ion | `beta_i_parallel=6`, `A_i=1.5` |
| `F_S_bM` | Firehose | ion | `beta_i_parallel=10`, `A_i=0.1` |
| `F_M_bM` | Firehose | ion | `beta_i_parallel=6`, `A_i=0.3` |
| `F_W_bM` | Firehose | ion | `beta_i_parallel=3`, `A_i=0.6` |
| `W_S_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=3.0` |
| `W_M_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=2.0` |
| `W_W_bM` | Whistler | electrón | `beta_e_parallel=0.5`, `A_e=1.5` |

`psc_units.py` define los perfiles físicos y nombres de salida. No usar un
perfil de producción para analizar otro caso: `F_M_bM` no es equivalente a
`firehose_maxwellian`.

## Salidas

La salida queda bajo:

```text
analysis_results/<CASE>/
```

Subcarpetas principales:

```text
01_anisotropy/     evolución de A, beta y trayectoria Brazil
02_fields/         mapas de campos y fluctuaciones
03_particles/      VDF y momentos de partículas
04_spectra/        espectros y modos dominantes
05_diamagnetic/    corrientes diamagnéticas
06_heat_flux/      flujo de calor y regiones espaciales
07_mirror_structures/ depresiones locales de |B| para mirror
08_validation/     validación puntual contra partículas
09_physical_diagnostics/ diagnóstico integrado con las salidas estándar
```

El target integrado:

```bash
make physics DATA_DIR=/ruta/a/run CASE=M_M_bM
```

genera en `09_physical_diagnostics/` las tablas y figuras de la lista física:
`validation_table.csv`, `validation_summary.txt`, `anisotropy_table.csv`,
`fit_metrics.csv`, `field_fluctuation_table.csv`, `growth_rate_summary.csv`,
`anisotropy_spatial_stats.csv`, `spatial_correlations.csv`, `energy_table.csv`,
mapas `T_parallel/T_perp/A_i`, `deltaB`, `mirror_holes`, `J_dia`, VDF 2D,
ajuste Maxwellian/Kappa, tasa de crecimiento, correlaciones y energía.

Para comparar casos ya analizados, por ejemplo Maxwelliano vs Kappa:

```bash
make compare-physics \
  COMPARE_CASES="maxwellian=../analysis_results/mirror_maxwellian/09_physical_diagnostics kappa=../analysis_results/mirror_kappa/09_physical_diagnostics"
```

Esto produce `comparison_kappa_vs_maxwellian.csv`,
`comparison_anisotropy.png`, `comparison_deltaB.png`,
`comparison_growth_rate.png`, `comparison_energy.png` y
`comparison_heat_flux.png`.

## Fundamento físico, fórmulas y variables

Esta sección explica qué pregunta física responde cada análisis. Las fórmulas
están escritas en las unidades normalizadas de PSC, donde
\(\mu_0=1\), \(c=1\), \(n_0=1\), \(m_e=1\) y \(m_i/m_e=200\).

### 1. Construcción de la presión térmica

#### 1.1. Razón física

Los archivos de momentos contienen segundos momentos completos, que mezclan
movimiento térmico y movimiento colectivo del plasma. Para medir temperatura,
anisotropía o beta se debe eliminar primero la contribución de la velocidad
macroscópica. De lo contrario, un flujo del plasma podría interpretarse
incorrectamente como calentamiento.

#### 1.2. Fórmulas

Para cada especie \(s\):

$$
n_s = |\rho_s|,
\qquad
u_{i,s} = \frac{p_{i,s}}{n_s m_s},
$$

$$
P_{ij,s}
= M_{ij,s}
- \frac{p_{i,s}p_{j,s}}{n_s m_s}
= M_{ij,s}-n_s m_s u_{i,s}u_{j,s}.
$$

#### 1.3. Variables

1. \(s\): especie, ion \(i\) o electrón \(e\).
2. \(n_s\): densidad numérica de la especie.
3. \(\rho_s\): densidad almacenada por PSC; para electrones se usa su
   magnitud.
4. \(m_s\): masa de la especie.
5. \(p_{i,s}\): primer momento de momento lineal en la dirección \(i\).
6. \(u_{i,s}\): velocidad macroscópica o de deriva.
7. \(M_{ij,s}\): segundo momento bruto guardado como `txx`, `txy`, etc.
8. \(P_{ij,s}\): tensor de presión térmica central.

### 2. Presión y temperatura respecto al campo magnético

#### 2.1. Razón física

Las inestabilidades Mirror, Firehose y Whistler dependen de la diferencia entre
la presión paralela y perpendicular al campo magnético. La dirección física
relevante es el campo local, no necesariamente un eje fijo de la malla.

#### 2.2. Fórmulas

$$
\mathbf{B}=(B_x,B_y,B_z),
\qquad
B=|\mathbf{B}|=\sqrt{B_x^2+B_y^2+B_z^2},
\qquad
\hat{\mathbf b}=\frac{\mathbf B}{B}.
$$

$$
P_{\parallel,s}
=\hat{\mathbf b}\cdot\mathsf P_s\cdot\hat{\mathbf b},
$$

$$
P_{\perp,s}
=\frac{\operatorname{Tr}(\mathsf P_s)-P_{\parallel,s}}{2},
$$

$$
T_{\parallel,s}=\frac{P_{\parallel,s}}{n_s},
\qquad
T_{\perp,s}=\frac{P_{\perp,s}}{n_s}.
$$

La expansión usada en el código es:

$$
\begin{aligned}
P_\parallel={}&P_{xx}b_x^2+P_{yy}b_y^2+P_{zz}b_z^2\\
&+2P_{xy}b_xb_y+2P_{yz}b_yb_z+2P_{zx}b_zb_x.
\end{aligned}
$$

`anisotropy_analysis.py` y `heat_flux_analysis.py` usan esta proyección local.
Algunos mapas auxiliares de `physical_diagnostics.py` aproximan
\(P_\parallel=P_{zz}\) y
\(P_\perp=(P_{xx}+P_{yy})/2\), suponiendo que el campo guía permanece
principalmente en \(z\).

#### 2.3. Variables

1. \(B_x,B_y,B_z\): componentes del campo magnético.
2. \(B\): magnitud local del campo.
3. \(\hat{\mathbf b}\): vector unitario paralelo al campo.
4. \(\mathsf P_s\): tensor de presión térmica de la especie.
5. \(P_{\parallel,s}\): presión en la dirección del campo.
6. \(P_{\perp,s}\): promedio de las dos presiones perpendiculares.
7. \(T_{\parallel,s}\), \(T_{\perp,s}\): temperaturas paralela y
   perpendicular.

### 3. Anisotropía y beta paralela

#### 3.1. Razón física

La anisotropía mide qué dirección contiene más energía térmica. La beta compara
la presión térmica con la presión magnética y determina cuánto puede el campo
magnético resistir la deformación producida por el plasma.

#### 3.2. Fórmulas

$$
A_s=\frac{T_{\perp,s}}{T_{\parallel,s}}
    =\frac{P_{\perp,s}}{P_{\parallel,s}},
\qquad
R_s=\frac{T_{\parallel,s}}{T_{\perp,s}}=\frac{1}{A_s},
$$

$$
P_B=\frac{B^2}{2\mu_0},
\qquad
\beta_{\parallel,s}
=\frac{P_{\parallel,s}}{P_B}
=\frac{2\mu_0P_{\parallel,s}}{B^2}.
$$

Como PSC usa \(\mu_0=1\):

$$
\beta_{\parallel,s}=\frac{2P_{\parallel,s}}{B^2}.
$$

Para Firehose se muestran las dos convenciones: \(A_i\) aumenta hacia uno
durante la relajación, mientras \(R_i=1/A_i\) disminuye hacia uno.

#### 3.3. Variables

1. \(A_s\): anisotropía perpendicular/paralela.
2. \(R_s\): anisotropía inversa.
3. \(P_B\): presión magnética.
4. \(\mu_0\): permeabilidad magnética, igual a uno en unidades de código.
5. \(\beta_{\parallel,s}\): beta paralela de la especie.

### 4. Umbrales de inestabilidad y Brazil plot

#### 4.1. Razón física

El Brazil plot coloca cada estado del plasma en el plano
\((\beta_\parallel,A)\). Su objetivo es comprobar si la condición inicial está
en la región inestable y si la evolución se acerca al umbral de estabilidad
marginal debido a la dispersión de partículas por las ondas generadas.

#### 4.2. Fórmulas utilizadas

1. Mirror iónico:

   \[
   A_i>1+\frac{1}{\beta_{\parallel i}},
   \qquad
   \beta_{\parallel i}(A_i-1)>1.
   \]

2. Firehose fluido:

   \[
   A_i<1-\frac{2}{\beta_{\parallel i}},
   \qquad
   \beta_{\parallel i}(1-A_i)>2.
   \]

3. Firehose oblicuo, aproximación cinética mostrada en la figura:

   \[
   A_i=1-\frac{1.4}{(\beta_{\parallel i}-0.11)^{0.55}}.
   \]

4. Ion-cyclotron, curva de referencia:

   \[
   A_i=1+\frac{0.43}{\beta_{\parallel i}^{0.42}}.
   \]

5. Whistler electrónico:

   \[
   A_e>1+\frac{0.21}{\beta_{\parallel e}^{0.6}}.
   \]

#### 4.3. Interpretación

1. Por encima del umbral Mirror se esperan fluctuaciones compresivas de
   \(B\) y estructuras tipo espejo.
2. Por debajo del umbral Firehose se esperan fluctuaciones principalmente
   transversales y reducción del exceso de presión paralela.
3. Por encima del umbral Whistler se espera crecimiento de ondas en escalas
   electrónicas y disminución de \(A_e\).
4. La trayectoria global usa el cociente entre presiones promediadas en
   volumen, no el promedio simple de cocientes celda a celda.

### 5. Fluctuaciones magnéticas y estructuras Mirror

#### 5.1. Razón física

Las inestabilidades convierten energía libre de la anisotropía en
fluctuaciones electromagnéticas. Separar las componentes paralela y
perpendicular permite distinguir una respuesta compresiva, típica de Mirror,
de una respuesta transversal, importante en Firehose y Whistler.

#### 5.2. Fórmulas

$$
\delta B=B-B_0,
\qquad
\frac{\delta B_{\rm rms}}{B_0}
=\frac{\sqrt{\langle(B-B_0)^2\rangle}}{B_0},
$$

$$
\frac{\delta B_{\parallel,\rm rms}}{B_0}
=\frac{\sqrt{\langle(B_z-B_0)^2\rangle}}{B_0},
$$

$$
\frac{\delta B_{\perp,\rm rms}}{B_0}
=\frac{\sqrt{\langle(B_x-\langle B_x\rangle)^2
+ (B_y-\langle B_y\rangle)^2\rangle}}{B_0}.
$$

Para cuantificar hoyos magnéticos:

$$
D_{\rm mirror}=1-\frac{\min(B)}{B_0},
$$

$$
f_{\rm area}
=\frac{N[B<B_0-\sigma_B]}{N_{\rm celdas}},
\qquad
\sigma_B=\operatorname{std}(B).
$$

#### 5.3. Variables

1. \(B_0\): campo guía inicial.
2. \(\delta B\): perturbación de la magnitud del campo.
3. \(\langle\cdot\rangle\): promedio espacial.
4. \(\sigma_B\): desviación estándar espacial de \(B\).
5. \(D_{\rm mirror}\): profundidad relativa del hoyo magnético.
6. \(f_{\rm area}\): fracción del dominio ocupada por campos bajos.

### 6. Tasa de crecimiento lineal

#### 6.1. Razón física

Durante la fase lineal de una inestabilidad, la amplitud de la perturbación
crece exponencialmente. La pendiente de su logaritmo permite medir la tasa de
crecimiento y comparar corridas fuertes, medias, débiles, Maxwellianas y
Kappa.

#### 6.2. Fórmulas

$$
\delta B_{\rm rms}(t)=\delta B_0 e^{\gamma t},
$$

$$
\ln\delta B_{\rm rms}(t)=\ln\delta B_0+\gamma t,
\qquad
\gamma=\frac{d}{dt}\ln\delta B_{\rm rms}.
$$

El tiempo se presenta como:

$$
\tau=\Omega_{ci}t,
\qquad
\Omega_{ci}=\frac{|q_i|B_0}{m_i}.
$$

#### 6.3. Variables

1. \(\delta B_0\): amplitud inicial de la perturbación.
2. \(\gamma\): tasa de crecimiento lineal.
3. \(t\): tiempo en unidades internas de PSC.
4. \(\tau=\Omega_{ci}t\): tiempo normalizado al girociclo iónico.
5. \(q_i,m_i\): carga y masa del ion.

### 7. Análisis espectral

#### 7.1. Razón física

El espectro identifica las longitudes de onda que contienen más energía y
permite comprobar si el modo dominante tiene la escala y orientación esperadas
para la inestabilidad. También separa propagación paralela y perpendicular al
campo guía.

#### 7.2. Fórmulas

Para una fluctuación bidimensional \(f(\mathbf x)\):

$$
\widetilde f(\mathbf k)=\mathcal F\{W(\mathbf x)f(\mathbf x)\},
\qquad
\operatorname{PSD}(\mathbf k)
=\frac{|\widetilde f(\mathbf k)|^2}{(N_1N_2)^2},
$$

$$
k_j=\frac{2\pi n_j}{N_j\Delta x_j},
\qquad
k=\sqrt{k_1^2+k_2^2}.
$$

El espectro radial suma la potencia de los modos que pertenecen al mismo
intervalo de \(k\):

$$
E(k)=\sum_{\mathbf k\ {\rm en\ el\ anillo}\ k}
\operatorname{PSD}(\mathbf k).
$$

El ajuste de ley de potencia usa:

$$
E(k)=Ck^\alpha,
\qquad
\log_{10}E=\log_{10}C+\alpha\log_{10}k.
$$

Para el espectro magnético transversal integrado:

$$
\operatorname{PSD}_{\perp}
=\operatorname{PSD}(\delta B_x)+\operatorname{PSD}(\delta B_y).
$$

#### 7.3. Variables

1. \(W\): ventana de Hann bidimensional usada para reducir fuga espectral.
2. \(\mathbf k\): vector de onda.
3. \(N_j\): número de celdas en la dirección \(j\).
4. \(\Delta x_j\): separación física entre celdas.
5. \(E(k)\): potencia espectral radial.
6. \(\alpha\): pendiente espectral.
7. \(k_\parallel,k_\perp\): componentes respecto al campo guía, tomado en
   \(z\).

### 8. Distribuciones de velocidad y ajuste Maxwelliano/Kappa

#### 8.1. Razón física

Las VDF muestran cómo se redistribuyen las partículas. Una distribución Kappa
posee colas supratérmicas más pobladas que una Maxwelliana; comparar ambos
ajustes permite saber si las partículas energéticas modifican el crecimiento,
la relajación o el transporte.

#### 8.2. Fórmulas

$$
v_\parallel=v_z,
\qquad
v_\perp=\sqrt{v_x^2+v_y^2},
$$

$$
T_\parallel=m\,\operatorname{Var}(v_z),
\qquad
T_\perp=\frac{m}{2}
\left[\operatorname{Var}(v_x)+\operatorname{Var}(v_y)\right].
$$

Forma Maxwelliana unidimensional ajustada:

$$
f_M(v)=C\exp\left(-\frac{v^2}{2\sigma^2}\right).
$$

Forma Kappa utilizada por el ajuste:

$$
f_\kappa(v)=C\left[
1+\frac{v^2}{(2\kappa-3)\sigma^2}
\right]^{-\kappa},
\qquad \kappa>1.5.
$$

La fracción supratérmica se estima como:

$$
f_{\rm supra}
=\frac{\sum w_p\,[|\mathbf v_p|>3v_{\rm th}]}
{\sum w_p}.
$$

#### 8.3. Variables

1. \(v_x,v_y,v_z\): componentes de velocidad de las partículas; en el
   régimen no relativista se aproximan por los momentos normalizados de PSC.
2. \(w_p\): peso estadístico de la partícula.
3. \(\sigma\): ancho ajustado de la distribución.
4. \(\kappa\): índice que controla la intensidad de la cola supratérmica.
5. \(C\): amplitud de normalización del ajuste.
6. \(v_{\rm th}\): escala térmica tridimensional calculada con las varianzas.

### 9. Corriente diamagnética

#### 9.1. Razón física

Un gradiente de presión perpendicular produce derivas opuestas de iones y
electrones y, por tanto, corriente. En estructuras Mirror, esta corriente ayuda
a sostener espacialmente las depresiones y aumentos del campo magnético.

#### 9.2. Fórmulas

$$
\mathbf J_{{\rm dia},s}
=\frac{\nabla P_{\perp,s}\times\mathbf B}{B^2}.
$$

En el plano \(YZ\), la componente dominante fuera del plano es:

$$
J_{{\rm dia},x,s}
=\frac{
(\partial P_{\perp,s}/\partial y)B_z
-(\partial P_{\perp,s}/\partial z)B_y
}{B^2},
$$

$$
J_{\rm dia,total}=J_{{\rm dia},i}+J_{{\rm dia},e}.
$$

Antes de calcular gradientes se aplica un filtro gaussiano para reducir ruido
estadístico PIC. Por ello, la corriente obtenida es un diagnóstico de
estructura coherente, no una medida de fluctuaciones celda a celda.

#### 9.3. Variables

1. \(P_{\perp,s}\): presión perpendicular de cada especie.
2. \(\nabla P_{\perp,s}\): gradiente espacial de presión.
3. \(\mathbf J_{{\rm dia},s}\): corriente diamagnética.
4. \(y,z\): coordenadas del plano de simulación.

### 10. Flujo de calor

#### 10.1. Razón física

El flujo de calor mide transporte de energía térmica. Permite determinar si la
relajación de la anisotropía solamente redistribuye energía entre direcciones o
también la transporta espacialmente.

#### 10.2. Fórmulas

El diagnóstico basado directamente en partículas usa el tercer momento
central:

$$
\mathbf c_p=\mathbf v_p-\langle\mathbf v\rangle,
\qquad
c_p^2=\mathbf c_p\cdot\mathbf c_p,
$$

$$
q_{\parallel}^{(p)}
=\frac{m}{2}\langle c_p^2c_{\parallel,p}\rangle_w,
\qquad
q_{\perp}^{(p)}
=\frac{m}{2}\langle c_p^2c_{\perp,p}\rangle_w.
$$

En el código,
\(c_{\perp,p}=\sqrt{c_{x,p}^2+c_{y,p}^2}\). Por tanto,
\(q_{\perp}^{(p)}\) mide una magnitud perpendicular positiva y no una
componente vectorial firmada. La definición vectorial completa sería
\(\mathbf q=(m/2)\langle c^2\mathbf c\rangle\).

Los mapas construidos con momentos de fluido son proxies de transporte:

$$
v_\parallel=\mathbf u\cdot\hat{\mathbf b},
\qquad
\mathbf v_\perp=\mathbf u-v_\parallel\hat{\mathbf b},
$$

$$
q_\parallel^{({\rm proxy})}=P_\parallel v_\parallel,
\qquad
q_\perp^{({\rm proxy})}=P_\perp|\mathbf v_\perp|.
$$

Los mapas de momentos no contienen el tercer momento completo y por eso no
deben interpretarse como el flujo de calor cinético exacto. El cálculo de
partículas es el diagnóstico físicamente más directo.

#### 10.3. Variables

1. \(\mathbf c_p\): velocidad peculiar respecto al flujo medio.
2. \(c_{\parallel,p}\), \(c_{\perp,p}\): componentes peculiar paralela y
   perpendicular.
3. \(\langle\cdot\rangle_w\): promedio ponderado por pesos de partículas.
4. \(\mathbf u\): velocidad macroscópica.
5. \(q_\parallel,q_\perp\): transporte de energía térmica paralelo y
   perpendicular.

### 11. Correlaciones espaciales

#### 11.1. Razón física

Las correlaciones comprueban si anisotropía, campo, densidad y corriente
pertenecen a la misma estructura física. Por ejemplo, una anticorrelación entre
densidad y magnitud de campo es una firma esperada de estructuras Mirror.

#### 11.2. Fórmula

Para dos mapas \(X\) e \(Y\), el código usa el coeficiente de Pearson:

$$
r_{XY}
=\frac{\sum_j(X_j-\bar X)(Y_j-\bar Y)}
{\sqrt{\sum_j(X_j-\bar X)^2}
 \sqrt{\sum_j(Y_j-\bar Y)^2}}.
$$

Se calculan, entre otras:

$$
r(A,\delta B),\quad
r(A,B),\quad
r(A,J_{\rm dia}),\quad
r(A,\rho_i).
$$

#### 11.3. Interpretación

1. \(r=1\): correlación lineal positiva perfecta.
2. \(r=-1\): anticorrelación lineal perfecta.
3. \(r\approx0\): ausencia de relación lineal; no descarta una relación no
   lineal.

### 12. Balance de energía

#### 12.1. Razón física

El seguimiento energético comprueba que el crecimiento de los campos procede
de la energía de las partículas y permite detectar errores numéricos o
inconsistencias entre snapshots.

#### 12.2. Fórmulas

$$
E_{\rm bulk}
=\frac{m}{2}|\langle\mathbf v\rangle|^2,
$$

$$
E_{\rm thermal}
=\frac{m}{2}
\left\langle|\mathbf v-\langle\mathbf v\rangle|^2\right\rangle,
$$

$$
E_{\delta B}
=\frac{1}{2}\langle(B-B_0)^2\rangle,
$$

$$
E_{\rm total}
=E_{\rm bulk}+E_{\rm thermal}+E_{\delta B},
\qquad
\epsilon_E(t)=\frac{E_{\rm total}(t)-E_{\rm total}(0)}
{E_{\rm total}(0)}.
$$

Este es un balance diagnóstico de las cantidades disponibles, no la energía
electromagnética total completa: no incluye explícitamente toda la energía del
campo eléctrico ni todas las especies en cada término.

#### 12.3. Variables

1. \(E_{\rm bulk}\): energía cinética del flujo medio.
2. \(E_{\rm thermal}\): energía cinética térmica.
3. \(E_{\delta B}\): energía de la fluctuación magnética.
4. \(E_{\rm total}\): suma diagnóstica.
5. \(\epsilon_E\): variación relativa respecto al primer snapshot.

### 13. Validación de momentos

#### 13.1. Razón física

Antes de interpretar una inestabilidad se verifica que la distribución
realmente fue inicializada con la densidad, deriva, temperatura y anisotropía
solicitadas. Esta prueba separa un problema de inicialización de un efecto
físico posterior.

#### 13.2. Fórmulas

$$
n_{\rm medido}
=\frac{N_p\,C_{\rm ori}}{N_{\rm celdas}},
$$

$$
\langle v_j\rangle_w
=\frac{\sum_p w_p v_{j,p}}{\sum_p w_p},
$$

$$
T_j=m\,\operatorname{Var}_w(v_j),
\qquad
v_{{\rm th},j}=\sqrt{\frac{T_j}{m}},
$$

$$
\operatorname{error\ relativo}
=100\frac{|X_{\rm medido}-X_{\rm esperado}|}{|X_{\rm esperado}|}.
$$

#### 13.3. Variables

1. \(N_p\): número de macropartículas de la especie.
2. \(C_{\rm ori}\): factor de peso `cori` usado por PSC.
3. \(N_{\rm celdas}\): número total de celdas.
4. \(w_p\): peso de cada macropartícula.
5. \(X\): cualquier magnitud validada.

### 14. Correspondencia entre scripts y diagnósticos

1. `anisotropy_analysis.py`: secciones 1 a 4; calcula presión térmica,
   proyección sobre el campo local, \(A_s\), \(\beta_{\parallel,s}\), umbrales
   y Brazil plots.
2. `fluctuationofmagneticfiel.py`: sección 5; genera mapas de fluctuaciones
   magnéticas normalizadas.
3. `mirror_physics.py`: secciones 5 y 9; visualiza estructuras magnéticas
   Mirror y corriente asociada.
4. `spectral_analysis.py`: sección 7; calcula FFT, PSD, espectro radial,
   pendiente y modos dominantes.
5. `plot_prt.py`: sección 8; construye VDF 2D, evolución de distribuciones y
   comparación Maxwelliana/Kappa. Las visualizaciones 3D cualitativas quedan en
   `legacy/` y no forman parte del flujo mantenido.
6. `diamagnetic_current.py`: sección 9; calcula
   \(J_{{\rm dia},i}\), \(J_{{\rm dia},e}\) y la corriente total.
7. `heat_flux_analysis.py`: sección 10; calcula los proxies espaciales
   \(P_\parallel v_\parallel\) y \(P_\perp v_\perp\).
8. `physical_diagnostics.py`: integra las secciones 3 a 12, crea tablas,
   mapas, correlaciones, ajustes, tasa de crecimiento y balance energético.
9. `validate_moments.py`: sección 13; verifica densidad, deriva, temperatura
   y anisotropía iniciales usando archivos de partículas.
10. `compare_physical_cases.py`: compara las mismas magnitudes entre corridas;
    solo es físicamente válido si se mantienen la misma definición de
    anisotropía, especie impulsora y normalización temporal.
11. `data_reader.py`: no aplica una fórmula física; centraliza la lectura,
    ensamblado y selección de datasets HDF5.
12. `psc_units.py`: define masas, campo guía, frecuencias, escalas espaciales,
    temperaturas iniciales y conversión de pasos a \(\Omega_{ci}t\).

## Documentación técnica

Para estructura interna de archivos, datasets HDF5 y responsabilidades de cada
script, ver:

```text
CodeforAnalisys/ANALISIS_ESTRUCTURA.md
```
