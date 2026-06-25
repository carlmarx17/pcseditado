# Plan de análisis: inestabilidades por anisotropía

Este documento resume qué debe comprobar el análisis para cada familia de
simulaciones. Los comandos concretos están en `CodeforAnalisys/README.md`.

## Variables centrales

Para cada especie:

```text
A = T_perp / T_parallel
beta_parallel = 2 P_parallel / |B|^2
```

En los scripts de análisis, `P_parallel` se proyecta sobre la dirección local de
`B`, no solo sobre el eje `z`. Esto evita interpretar como relajación térmica
una rotación local del campo.

## Mirror

Condición física:

```text
A_i > 1
beta_i_parallel * (A_i - 1) > 1
```

Firmas esperadas:

- crecimiento de fluctuaciones compresivas en `|B|`;
- estructuras tipo hoyo o espejo magnético;
- anticorrelación entre densidad y magnitud del campo;
- trayectoria global que se acerca al umbral marginal.

Casos:

```text
psc_mirror_bimaxwellian_strong, psc_mirror_bimaxwellian_moderate, psc_mirror_bimaxwellian_weak, psc_mirror_bikappa3, psc_mirror_bikappa5
```

## Firehose

Condición física:

```text
A_i < 1
beta_i_parallel * (1 - A_i) > 2
```

Firmas esperadas:

- crecimiento de fluctuaciones transversales;
- reducción del exceso de presión paralela;
- `A_i` aumenta hacia 1 si se usa `T_perp/T_parallel`;
- el inverso `T_parallel/T_perp` disminuye hacia 1.

Casos:

```text
psc_firehose_bimaxwellian_strong, psc_firehose_bimaxwellian_moderate, psc_firehose_bimaxwellian_weak, psc_firehose_bikappa3, psc_firehose_bikappa5
```

## Whistler

Condición práctica:

```text
A_e > 1 + 0.21 / beta_e_parallel^0.6
```

Firmas esperadas:

- crecimiento en escalas electrónicas;
- anisotropía electrónica decreciendo hacia el umbral;
- espectro dominado por modos compatibles con propagación paralela u oblicua.

Casos:

```text
psc_whistler_bimaxwellian_strong, psc_whistler_bimaxwellian_moderate, psc_whistler_bimaxwellian_weak
```

## Diagnósticos mínimos

Una corrida no debe evaluarse con una sola figura. El paquete de análisis debe
producir como mínimo:

```text
anisotropy_evolution.csv
evolucion_anisotropia.png
brazil_trayectoria.png
dominant_mode_evolution_xy.png
spectrum_2d_final_*.png
particle_anisotropy_evolution.csv
```

Para comparar casos, usar siempre:

- misma definición de `A`;
- misma especie impulsora;
- mismo intervalo temporal normalizado a `Omega_ci`;
- mismo criterio de selección de snapshots.
