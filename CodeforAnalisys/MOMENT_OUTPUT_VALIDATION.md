# Validación del output de momentos de PSC

Fecha: 2026-10-23. Ejecuciones locales nuevas, CPU, HDF5, salida instantánea del paso 0. Se compiló y ejecutó `psc_test_pressure_moment`; los resultados no son datos sintéticos.

## Resultado

`all_1st:txx..tzx` contiene el flujo de momento bruto `n m <u_i v_j>`, con `u=gamma v`. No está centrado ni dividido por n. En el límite no relativista es `n m <v_i v_j>`.

Para iones de carga +1, `n=rho`. La presión no relativista se obtiene por celda como `P_ij=t_ij-p_i*p_j/(n*m)` y la temperatura en unidades de energía como `k_B T_ij=P_ij/n`. La covarianza de velocidades es `P_ij/(n*m)`. Para otras cargas, usar `n=rho/q`.

| Prueba | Parámetros | Resultado |
| --- | --- | --- |
| Densidad variable | m=100, T0=.01, n=.5..1.5, sin flujo | RMS relativo del ajuste tzz=n*T0: 1.98%; del ajuste tzz=T0: 35.02% |
| Flujo variable | m=100, T0=.01, n=1, u_y amplitud .02 | RMS relativo del modelo bruto: .70%; del modelo central: 244.16% |
| Masa y densidad distintas | m=400, T0=.01, n=2, u_y amplitud .02 | Temperaturas medias recuperadas: Tx=.00997357, Ty=.00996523, Tz=.00998004 |

En las tres ejecuciones las temperaturas medias recuperadas en cada dirección difieren menos del 0.5% de .01. En la tercera, la covarianza yy es .0000249131: coincide con T/m, no con T. Esto comprueba la normalización de masa.

El RMS se normaliza por el RMS de cada modelo, por lo que los porcentajes comparan discrepancias relativas a modelos diferentes. El umbral de clasificación del script (10% y separación entre candidatos) es diagnóstico, no un intervalo de confianza.

## Correcciones y comprobaciones del análisis

- Se resta el flujo por celda antes del promedio sobre y; hacerlo después confunde flujos no resueltos con energía térmica.
- Se comparan predicciones cuantitativas y se admite un resultado inconcluso cuando no existe un ajuste suficiente o los modelos son indistinguibles.
- Tres pruebas de pytest verifican discriminación entre modelos, datos inválidos/degenerados y cancelación de flujos opuestos al promediar. Todas pasaron.
- El caso de simulación y su registro en CMake ya existían como cambios locales; no fue necesario modificarlos.

## Unificación de la conversión en el pipeline (2026-10-26)

- La conversión vive ahora en un solo lugar, `plasma_physics.central_pressure_tensor`.
  `anisotropy_analysis.process_snapshot`,
  `temperature_anisotropy_dispersion.compute_temperature_quantity` y
  `physical_diagnostics.moment_thermal_maps` tenían cada uno su propia copia de la
  fórmula —las tres correctas, pero una corrección futura habría tenido que
  aterrizar en cuatro sitios—. Las tres llaman ahora a la función compartida.
  `moment_fields_by_name` quita el path del dataset (`txx_i/p0/3d`) y `n_safe`
  expone la densidad enmascarada, para que el umbral de densidad no se reescriba
  en cada script.
- El run sintético (`synthetic_run.py`) escribe ahora un flujo de cizalla `u_y(z)`
  de media nula en `p_y` y `t_yy`, con `FLOW_AMPL = 1` (la amplitud en la que
  `n m u_y^2` llega a la presión paralela). La parte térmica no cambia,
  `P_yy = t_yy - p_y^2/(n m) = n T_perp`, así que un análisis correcto lee lo mismo
  que sin flujo. Antes `p = 0` hacía que el término de centrado fuera idénticamente
  cero: ningún test del pipeline distinguía entre restarlo y no restarlo.
- Las partículas del prt se dejan sin flujo a propósito. El sesgo por promediar la
  ventana entera con una sola media global es otra pregunta
  (`plot_prt._compute_particle_energies`) y sigue sin medir.
- Prueba de mutación: al borrar la resta de `central_pressure_tensor` fallan cuatro
  tests de `test_physical_consistency.py` y el `A_i` del run sintético pasa de
  1.999 a 2.249, por encima de la tolerancia +-0.1 con la que
  `test_pipeline_synthetic` lo verifica.

## Reproducción desde la raíz del repositorio

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPSC_USE_ADIOS2=OFF -DBUILD_TESTING=OFF -DUSE_CUDA=OFF -DUSE_VPIC=OFF
cmake --build build --target psc_test_pressure_moment -j 6
mkdir -p analysis_results/moment_output_audit/test1 analysis_results/moment_output_audit/test2 analysis_results/moment_output_audit/test3
(cd analysis_results/moment_output_audit/test1 && PSC_MOMENT_TEST=1 PSC_NMAX=1 ../../../build/src/psc_test_pressure_moment > run.log 2>&1)
(cd analysis_results/moment_output_audit/test2 && PSC_MOMENT_TEST=2 PSC_NMAX=1 ../../../build/src/psc_test_pressure_moment > run.log 2>&1)
(cd analysis_results/moment_output_audit/test3 && PSC_MOMENT_TEST=2 PSC_NMAX=1 PSC_MASS_RATIO=400 PSC_N0=2 ../../../build/src/psc_test_pressure_moment > run.log 2>&1)
MPLBACKEND=Agg .venv/bin/python CodeforAnalisys/moment_output_test.py --test1 analysis_results/moment_output_audit/test1 --test2 analysis_results/moment_output_audit/test2 --out analysis_results/moment_output_audit/verified.png
MPLBACKEND=Agg .venv/bin/python -m pytest -q CodeforAnalisys/test_moment_output_test.py
```

La tercera salida puede analizarse con `read_profiles(Path(...), 0, mass=400)` y `thermal_pressure`; dividir sus componentes por `prof['rho']` recupera la temperatura. Los datos, logs, figura y JSON quedan en `analysis_results/moment_output_audit/` (directorio ignorado por Git).

## Alcance

Se verificaron los iones del backend CPU, con velocidades pequeñas y salida HDF5 instantánea. No se ejecutaron GPU ni ADIOS2. Las fórmulas no relativistas no constituyen una validación de temperatura relativista. La inicialización fija la dispersión en u; por eso la comparación con T0 tiene pequeñas correcciones relativistas, además del ruido de partículas y del depósito en malla. El cálculo de presión de los demás análisis no cambió numéricamente: se unificó en
una sola función, conservando el umbral de densidad que tenía cada copia.
