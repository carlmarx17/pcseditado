---
name: psc-figure-conventions
description: Reglas de presentación para toda figura, tabla o documento del análisis PSC (CodeforAnalisys). Úsala siempre que se vaya a crear, editar o revisar un script que dibuje (cualquier *.py con matplotlib en CodeforAnalisys/, paper_figures.py, kappa_dynamics.py, plot_prt.py, physical_diagnostics.py, etc.), cuando el usuario pida "mejorar las gráficas", "una figura para el paper/tesis", una serie temporal de fluctuaciones, tasas de crecimiento, VDF o kappa, o un documento/reporte con resultados. Evita que el usuario tenga que repetir sus preferencias — en particular que nunca se dibuje el primer punto ni el arranque de ruido en gráficas de fluctuaciones contra el tiempo.
---

# Convenciones de figuras PSC

El dueño de la tesis ya pidió estas cosas más de una vez. Aplicarlas sin que
las recuerde. Los comentarios y etiquetas van en inglés; el chat en español.

## 1. Series temporales de fluctuaciones: sin el primer punto ni el arranque

**Regla.** En toda figura de una fluctuación contra el tiempo (dB, |dB|^2,
amplitud de un modo, energía de campo, rms en la ventana prt, E(k,t)) no se
dibuja t = 0 ni el arranque de ruido del quiet start.

**Por qué.** En t = 0 el campo es B0 uniforme: la fluctuación es redondeo y en
eje log estira la escala varias décadas. Después el ruido de partículas se
establece en unos tiempos de tránsito iónico: una subida casi vertical que no
es crecimiento ni piso de ruido, y es lo único que se ve en el borde izquierdo.
"Ahí nunca se ve nada."

**Cómo.** Una sola función, no máscaras a mano:

```python
shown = ps.measured_fluctuation(t)              # perfil activo (psc_units)
shown = ps.measured_fluctuation(t, settle)      # varios perfiles en una figura
```

- `settle` por defecto: `psc_units.noise_settling_time(2*pi/L_box)`, es decir
  2 / (k v_th,i) del modo fundamental de la caja (≈ 2.85 Omega_ci^-1 en los
  casos mirror moderate).
- Para el ajuste o la figura de un modo concreto usar el k de ese modo:
  `noise_settling_time(k_di)`.
- En scripts que comparan corridas de perfiles distintos (kappa_dynamics.py,
  paper_figures.py) calcular `settle` del perfil de cada corrida, no del
  perfil activo.
- Nunca escribir `t > 0` suelto para estas series. Si la figura oculta un
  intervalo, decirlo en una nota pequeña ("quiet-start noise build-up
  (t Omega_ci < X) not shown").
- Los ajustes de tasa de crecimiento tampoco usan esas muestras.

No aplica a magnitudes que no son fluctuaciones (anisotropía, temperatura,
índice kappa, energía de partículas): ahí t = 0 es el estado inicial y sí va.

## 2. Estilo

- Todo pasa por `plot_style` (`ps.apply()`, `ps.save(fig, path)`), tema `paper`.
  `ps.save` escribe PNG + PDF y el registro de QA.
- Después de generar, leer `figure_qa_*.jsonl` de la carpeta de salida y dejar
  `issues` vacío: sin paneles vacíos, sin textos superpuestos, sin leyendas o
  notas encima de los datos.
- Leyendas debajo o al lado del panel, no sobre las curvas. Una sola leyenda
  cuando varios paneles comparten series.
- Unidades físicas, nunca "code units": tiempo en 1/Omega_ci, longitudes en
  d_i, velocidades en v_A, campos en B0, corrientes en e n0 v_A, energías en
  B0^2/mu0.
- Colores Okabe-Ito en orden fijo: Maxwelliana, kappa 5, kappa 3.
- Ejes log con muchas etiquetas: solo décadas. Evitar offsets "(x10^-3)"
  automáticos; escalar el dato y decirlo en la etiqueta.
- Cortar el eje a la banda física (k d_i <= 2 para iones), no a la malla.

## 3. Distribuciones y kappa

- Un modelo solo se compara con datos de la misma varianza: Maxwelliana y kappa
  se dibujan con la sigma medida en ese instante, nunca con la temperatura
  inicial sobre un snapshot tardío.
- kappa se estima por máxima verosimilitud (`plasma_physics.kappa_mle`) o con
  el estimador de curtosis truncada en marco local (`kappa_eff`); el ajuste por
  mínimos cuadrados al histograma está sesgado.
- El índice que se cita es el del marco del campo local. Se grafica 1/kappa
  (0 = Maxwelliana), con kappa en el eje derecho.
- Para decidir "cola" contra "aplanamiento" usar medidas sin modelo: fracción
  más allá de 3 sigma respecto a la Gaussiana y f(u)/Gaussiana en u = dv/sigma(t).

## 4. Comparación con la literatura

- Decir siempre en qué convención se compara kappa con Maxwelliana: misma
  temperatura (estas corridas) o mismo núcleo (Lazar et al. 2015). El signo del
  efecto sobre gamma cambia entre una y otra.
- No afirmar el origen de un efecto que solo un control puede decidir
  (calentamiento numérico, tasa de fondo de kappa) antes de tener el control.

## 5. Entrega

- Código, scripts de job y documentación al repositorio; `analysis_results/`
  nunca.
- Tras cambiar código: tests (`pytest` en CodeforAnalisys), `graphify update .`
  y `enrich.py`. Commit directo a `main`, sin líneas de atribución de Claude.
