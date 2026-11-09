# COSMA · revisión de tesis de las tres corridas moderadas

## Inventario antes de reservar nodos

Desde el repositorio actualizado en COSMA, listar qué existe por corrida y por versión:

```bash
.venv/bin/python CodeforAnalisys/reuse_energy.py inventory analysis_results
```

Para mirror, firehose y whistler periódicos en el plano yz, seleccionar el directorio de resultados de una corrida. La figura se reconstruye a partir de energía global, trabajo por especie y partición por momentos según lo disponible. Se indican los reservorios ausentes. Ejemplo con una corrida archivada:

```bash
.venv/bin/python CodeforAnalisys/reuse_energy.py figure   analysis_results/v6-reviewed-20261015/mirror_bikappa5_moderate --out revision_energia/k5
```

Si la pregunta física requiere el término de presión y falta en el archivo, usar el mismo comando con `--measure-pressure` en un nodo de cómputo. Lee campos y momentos en los pasos del trabajo guardado. El cálculo es serial y no abre partículas. No ejecuta detección de agujeros, espectros ni VDF. Revisar la cadencia de los campos, especialmente para whistler. La figura usa tΩci para todas las especies. Reconexión requiere su propia geometría y balance de frontera. El envío completo de abajo se reserva para regenerar toda la serie mirror.

## Envío completo

Desde el checkout en COSMA:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
git pull --ff-only origin main
sbatch cosma_jobs/analisis/tesis_mirror.sh
```

El envío usa tres nodos de COSMA7, la cuenta dp433 y el Python instalado en `/cosma7/data/dp433/dc-mart18/pcseditado/.venv/bin/python`. El directorio de envío selecciona el código que ejecutará Slurm. Mantener esa revisión mientras el trabajo está en cola o en ejecución. Los datos siguen en `anisotropy_adios2`. Las corridas de producción son 11596993 (Maxwell), 12063822 (κ=5) y 11618877 (κ=3).

Cada ejecución crea `pcseditado/analysis_results/tesis_mirror_JOBID/`. Abrir `review/REVISION.md`, `quality_report/index.html` y el log de Slurm. Los logs de cálculo por corrida quedan en `logs/`. Si se interrumpe un trabajo, exportar `OUT` con el directorio absoluto de ese trabajo y repetir el envío con el mismo código y los mismos datos. El pipeline comprueba la firma de las etapas antes de reanudar. Las comparaciones finales se regeneran. `RUN_ROOT`, `PYTHON` y `CODE_ROOT` permiten cambiar las rutas del sitio.

El script descubre un control por distribución con salidas de campo. Si hay varias historias, indicar la ruta absoluta con `CONTROL_0` (Maxwell), `CONTROL_1` (κ=5) o `CONTROL_2` (κ=3) antes de llamar a sbatch. Se admiten controles parciales desde la condición inicial. Una carpeta de un intento fallido sin campos se omite. Los tres casos de producción requieren el último paso esperado. Sólo se ejecutan análisis sobre archivos existentes.

## Qué puede defender la tesis

| Pregunta | Medición y corrección | Criterio de interpretación |
|---|---|---|
| ¿Qué cambió al variar κ? | Revisar los momentos iniciales de ambas especies, masa, β, anisotropía, ppc y procedencia de dt en los manifiestos. El inicializador mantenido usa la bi-κ multivariante en ambas especies, con temperatura de segundo momento fija. | Comparar Maxwell, κ=5 y κ=3 a igual temperatura. El contraste teórico a núcleo fijo cambia la temperatura. Para atribuir exclusivamente a la cola iónica hace falta una variación independiente de la distribución electrónica. Confirmar el cargador del ejecutable histórico. |
| ¿Qué rama crece? | Ajuste del coeficiente complejo por modo, fase lineal previa a saturación, frecuencia con signo, polarización, compresibilidad y resolución 2π/T. Comparar γ y ω con teoría paralela en el armónico observado y con momentos electrónicos e iónicos contemporáneos. | La identificación AIC requiere propagación y polarización compatibles. Informar por separado el armónico de la caja y el máximo teórico continuo. Un ajuste que no supera ruido o no resuelve la frecuencia queda abierto. |
| ¿Cuál es el origen de los agujeros? | `review/<caso>/deepest_holes.csv` contiene una fila por trayectoria en su mayor depresión medible, (B_z), contribución transversal exacta, fundamental paralelo, presión magnética, presiones térmicas y densidad. Las fracciones son firmadas y pueden quedar fuera de [0,1] si hay cancelación. Contrastar con máscaras, sensibilidad del umbral y censura por división o fusión. | La reconstrucción con el fundamental identifica cuánto contraste podría producir ese par de modos aislado. El resto incluye otros modos e interferencias. El balance local y la anticorrelación apoyan la descripción de estructura compresiva. No establecen el mecanismo que la generó. |
| ¿Hay evidencia de crecimiento mirror independiente? | `review/<caso>/oblique_phase.csv` usa la misma ventana lineal de referencia para todos los armónicos oblicuos guardados con índices absolutos ≤3. Mide pendiente de ln abs(Bz), coherencia y frecuencia en el marco de flujo iónico medio (aproximación no relativista), coespectro densidad–Bz y residuo de presión. Los coeficientes de toda la historia quedan en `07_structures/attribution_modes.csv`. | Comparar potencia con el fondo previo, variar la ventana, buscar persistencia y rastrear el modo hasta el objeto. Una fase casi fija en una serie corta o dominada por ruido no demuestra aperiodicidad. El código de teoría de esta entrega resuelve AIC paralela. La identificación cinética definitiva de mirror requiere contrastar con dispersión oblicua de la distribución real en los mismos k. |
| ¿La caja sesga el resultado? | En 20 di, Δk di=0.314. Con βi⊥=10 y ρi definido con √(2T⊥/mi), el primer k⊥ρi es 0.99. A 63.4° el primer par (ny,nz)=(2,1) ya tiene k⊥ρi≈1.99. | El muestreo espacial limita las ondas largas y oblicuas. Una caja mayor puede probar ese efecto si mantiene dx, dt, ppc y el mismo caso físico. Una familia firehose en otra caja cambia también la inestabilidad y sirve sólo como comparación. |
| ¿Dónde quedó la anisotropía? | `resonant_anisotropy_mirror_moderate_kappa/`, `kappa_dynamics_mirror_moderate_kappa/` y `trajectory_linear_theory_mirror_moderate_kappa/` comparan núcleo y cola, κ en campo local y crecimiento predicho con parámetros medidos. Examinar conteos efectivos en la cola y repetibilidad entre semillas. | vres=(ω−nΩi)/k∥ depende del armónico y del sentido de propagación. Contrastar la banda resonante con el ancho observado de la distribución. El valor del fundamental no fija un límite universal para todas las partículas. |
| ¿Es causal el vínculo entre ondas y evolución de κ? | Comparar VDF en marco local y global, campo en la ventana de partículas y campo de toda la caja. `wave_kappa_test/` compara el cambio de forma con potencia modal, tendencias temporales y validación por bloques. | Una tendencia común, un ajuste de alto R² o un error formal de partículas no establecen causalidad. Reportar sensibilidad temporal y límites por tener una realización de cada caso. |
| ¿Cuánta energía espuria aparece? | En la caja periódica sumar las seis componentes de energía de campo en la malla nativa y las cinéticas relativistas globales por especie de DiagEnergies. Reportar el residuo, su tasa por fase y su tamaño respecto a la energía liberada y a la ganancia electrónica. El cálculo por momentos incluye flujo, energía térmica paralela y perpendicular de ambas especies y todos los campos. | El cierre global mide creación o pérdida neta. La partición por momentos usa una aproximación no relativista, especialmente relevante para electrones calientes y colas κ. Usar la ventana de partículas para temperaturas. Verificar aliasing antes de interpretar trabajo J·E. |
| ¿Qué corrección térmica es justificable? | En cada control verificar coincidencia de distribución, energía térmica inicial, dx, ppc, dt, forma de partícula y solver. Comparar por volumen en la cobertura común y conservar curvas brutas. El CSV de energía incluye la corrección cuando la pareja es compatible. | La resta supone calentamiento aditivo. Temperaturas electrónicas parecidas en tres corridas motivan comprobar un efecto común. Sólo los controles emparejados y pruebas de convergencia respaldan cuantificar cuánto es numérico. Una corrección posterior tampoco reconstruye la trayectoria dinámica de una simulación conservativa. |

## Directrices para decidir qué entra en el manuscrito

1. Presentar primero la procedencia del caso, el ajuste del modo dominante, los momentos iniciales y el cierre energético por fases. Si el error rivaliza con el intercambio físico, retener como descriptivas las mediciones de transferencia y saturación.
2. Comparar γ a κ variable en el mismo k permitido y con la misma convención térmica. Añadir la evolución de βe y las tasas con los parámetros medidos. La comparación a núcleo fijo es un experimento teórico distinto.
3. Presentar estructuras como una población medida, con profundidades y balance locales. El catálogo de trayectorias originales permite inspeccionar perfiles representativos y comprobar la estabilidad de los resultados ante cambios del detector. Conservar los objetos indeterminados.
4. Para la dispersión entre corridas y las colas, separar error de ajuste, ruido de muestreo y variación entre realizaciones. Un control parcial restringe únicamente los tiempos cubiertos. Los ensayos de resolución existentes pueden analizarse con el trabajo separado `analisis_resolution_mirror.sh` cuando existan sus datos.

## Prioridades de la reunión del 9 de noviembre

**Primero, la figura de energía.** Cada corrida escribe `review/<caso>/heating_channels.png` y el CSV homónimo. El primer panel muestra las reservas globales. Los demás comparan el trabajo electromagnético con los cambios cinéticos y térmicos por especie en los mismos tiempos y unidades por volumen. Los términos de transferencia ya están contabilizados en las variaciones de energía. Sumarlos otra vez al total duplicaría la transferencia.

En una caja periódica, con promedios de volumen, las predicciones de cierre son

```text
Δ(Eeléctrica + Emagnética) = −Σ W_s          W_s = ∫ ⟨J_s·E⟩ dt
ΔK_s = W_s
ΔU_s ≈ Q_s                                 Q_s = −∫ ⟨P_s:∇u_s⟩ dt
ΔK_flujo,s ≈ W_s − Q_s
```

Las últimas dos relaciones usan el balance de momentos no relativista y el tensor completo de presión central, incluidos los términos fuera de la diagonal. El campo magnético desvía partículas y participa en el intercambio a través de la evolución de los campos. La fuerza magnética de Lorentz hace trabajo directo cero. El aumento de energía interna incluye compresión reversible. Un residuo en una especie mezcla posibles efectos de cadencia, colocación de las variables, truncación de momentos y error numérico. El cierre de la suma global es el diagnóstico independiente. La partición sigue el balance de Vlasov–Maxwell integrado en el dominio. [Yang et al. (2017)](https://doi.org/10.1103/PhysRevE.95.061201)

**Artículo 1.** El resultado primario es el crecimiento medido de AIC al variar κ a segundo momento fijo, en el mismo armónico. La condición inicial es Aᵢ=2 y βᵢ∥=5. Los valores relajados se presentan en la sección de saturación. La interpretación de la normalización a núcleo fijo queda como predicción teórica hasta disponer de un contraste PIC independiente (p. ej., κ=3 a núcleo fijo o un control Maxwelliano de β∥=10 que pruebe el efecto de elevar β). Para las bandas de velocidad, distinguir significación formal de muestreo, dispersión entre semillas y sesgo de caja. Los oblicuos de 63–68° siguen siendo candidatos. La figura energética acompaña tanto el ajuste lineal como la saturación.

**Artículo 2.** Estudiar κ como variable dinámica con un estimador declarado y su sensibilidad a marco, volumen y rango de velocidades. Un ajuste temporal de R² alto por sí solo permite tendencias comunes. Para atribuir la evolución a las ondas se necesitan controles que cubran el intervalo, contraste contra tendencia temporal, espectros y polarización resueltos, así como convergencia de la cola. El segundo incremento temporal y la hipótesis núcleo–halo son preguntas abiertas. Dos máximos espectrales por sí solos tampoco identifican dos poblaciones resonantes.

**Tesis.** El par firehose de 40 di aporta una comparación adicional en otra familia. Una prueba de tamaño de caja para mirror requiere conservar la misma física al variar L. Whistler y reconexión tienen capítulos propios. El envío `tesis_mirror.sh` procesa las tres corridas mirror moderadas y sus controles disponibles. Las nuevas simulaciones y las comprobaciones bibliográficas de prioridad quedan fuera del alcance de ese trabajo.
