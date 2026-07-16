# Análisis espectral de inestabilidades cinéticas de anisotropía de temperatura en simulaciones PIC full-cinéticas

## TL;DR
- El flujo de trabajo estándar es: (1) FFT espacial y espacio-temporal de los campos B/E de la caja periódica para construir mapas ω–k y espectros P(k), P(ω); (2) superposición de soluciones de teoría lineal Vlasov (WHAMP, NHDS, PLUME, DSHARK, LEOPARD, ALPS); (3) medida de γ(k) por ajuste exponencial de la energía magnética por modo; (4) discriminación mediante polarización, ω_r, compresibilidad, ángulo de propagación y curvas de umbral de Hellinger et al. (2006).
- Las cuatro inestabilidades iónicas se distinguen sin ambigüedad: **ion-cyclotron/EMIC** (propagante, ω_r finita, casi-paralela, izquierda, transversal), **firehose paralela** (propagante, casi-paralela, derecha, transversal), **firehose oblicua** (ω_r≈0, oblicua, quasi-lineal), **mirror** (ω_r≈0, oblicua, compresiva con δn anticorrelacionada a δ|B|). Las curvas de umbral usan T_⊥/T_∥ = 1 + a/(β_∥ − β₀)^b.
- Para whistler/electron-firehose/electron-mirror hace falta PIC full-cinético (VPIC, OSIRIS, SMILEI, iPIC3D, ACRONYM, EPOCH, P3D, PSC); los códigos hybrid-PIC (CAMELIA, dHybridR, HYPERS, Pegasus) tratan los electrones como fluido y NO resuelven modos de escala electrónica.

## Key Findings

1. **Los mapas ω–k se construyen con una FFT 2D en (x,t)** de una componente de campo, mirroring/ventaneo temporal para forzar periodicidad, y separación de ramas +k/−k según el signo de ω. Los límites de Nyquist son k_Ny = π/dx y ω_Ny = π/dt_out; el intervalo de guardado dt_out (no el paso de integración dt) suele ser el cuello de botella y produce aliasing en ω.

2. **γ(k) se mide ajustando el crecimiento exponencial** de la energía magnética de cada modo k: E_B(k,t) ∝ e^{2γ(k)t}, es decir γ(k) = ½ d/dt ln E_B(k,t) durante la fase lineal. La curva γ(k) medida se compara directamente con la teoría lineal.

3. **La discriminación se basa en cinco firmas concurrentes**: frecuencia real (propagante vs no propagante), ángulo θ_kB, polarización (izquierda/derecha, circular/lineal), compresibilidad (δB_∥ vs δB_⊥, δn–δ|B|) y posición en el plano (β_∥, T_⊥/T_∥) respecto a las curvas de umbral.

4. **Coeficientes de umbral verificados** de Hellinger, Trávníček, Kasper & Lazarus (2006), GRL 33, L09101, Tabla 1 (γ_max = 10⁻³ Ω_p). El resultado central de ese trabajo, en cita textual del abstract: *"In the slow solar wind, the observed proton temperature anisotropy seems to be constrained by oblique instabilities, by the mirror one and the oblique fire hose, contrary to the results of the linear theory which predicts a dominance of the proton cyclotron instability and the parallel fire hose."*

5. **Full-PIC vs hybrid-PIC**: la diferencia crítica es que sólo el full-PIC resuelve la física electrónica cinética (whistler anisotropy, electron firehose, electron mirror), a costa de resolver λ_D y ω_pe⁻¹.

## Details

### BLOQUE 1 — Metodología técnica de análisis espectral

#### 1.1 FFT de los campos y resolución espectral

La salida de una simulación PIC es una serie de campos en la malla, p. ej. B(x, y, t) y E(x, y, t), guardados cada dt_out en una caja periódica de tamaño L_x = N_x·dx. Como la caja es periódica, la base natural es la de Fourier discreta:

  B̃(k, t) = Σ_j B(x_j, t) e^{−i k x_j},  con k_m = 2π m / L_x, m = −N_x/2 … N_x/2.

- **Resolución en k**: Δk = 2π/L_x. Cuanto mayor la caja, más fina la rejilla en k.
- **Nyquist en k**: k_Ny = π/dx = π N_x / L_x. Estructuras con longitud de onda < 2·dx no se resuelven y se pliegan (aliasing en k).
- **Resolución en ω**: Δω = 2π/T, con T = N_t·dt_out la duración de la ventana analizada.
- **Nyquist en ω**: ω_Ny = π/dt_out. Aquí el parámetro relevante es dt_out (el intervalo de **guardado**), no el paso de integración del leapfrog. Si dt_out es demasiado grande, las ramas de alta frecuencia (p. ej. whistler cerca de ω_ce) se pliegan sobre frecuencias bajas: aliasing temporal. Por eso, para resolver EMIC basta dt_out ~ fracción de Ω_p⁻¹, pero para whistler hace falta dt_out ~ fracción de ω_ce⁻¹.

**Ventaneo (windowing).** La serie temporal finita equivale a multiplicar por una ventana rectangular, cuya transformada es un sinc que produce fuga espectral (leakage) y lóbulos laterales. Se aplican ventanas de Hanning o Hamming, W_H(t) = ½[1 − cos(2πt/(N−1))], que reducen los lóbulos laterales a costa de ensanchar el lóbulo principal (peor resolución, menos fuga). La ventana de Hanning, que toca cero en ambos extremos y elimina toda discontinuidad, es la elección por defecto en la mayoría de los casos por su buen compromiso resolución/fuga. En espacio, como la caja es periódica, muchas veces NO se ventanea (la periodicidad ya garantiza continuidad); el ventaneo se reserva para el eje temporal si la señal no es periódica en t. Una alternativa frecuente en la literatura PIC es el "mirroring": se refleja el campo en el eje temporal creando un array (2t, x, y) para forzar periodicidad antes de la FFT; esto introduce un modo espurio ("numerical mirror mode") en ω > ω_Ny/2 que hay que descartar.

**Zero-padding.** Rellenar con ceros antes de la FFT interpola la rejilla espectral (Δω aparente menor) pero NO añade información ni mejora la resolución real, que sigue fijada por T y L. Es útil para suavizar visualmente las ramas ω–k.

#### 1.2 Construcción y lectura de diagramas de dispersión ω–k

El procedimiento típico:

1. Se toma una componente de campo (p. ej. B_y o B_z, o combinaciones B_± = B_y ± iB_z para aislar polarización) a lo largo de la dirección de interés (paralela a B₀ para inestabilidades casi-paralelas; a lo largo de x para el diagrama general).
2. Se hace una FFT 2D: B(x, t) → B̃(k, ω). El módulo |B̃(k,ω)|² es la densidad espectral de potencia en el plano ω–k.
3. **Separación de modos +k/−k**: usando la FFT completa (compleja) se mantienen los cuatro cuadrantes (±k, ±ω). Un modo que se propaga hacia +x aparece como potencia en el cuadrante (k>0, ω>0) [y su conjugado (k<0, ω<0)]; uno hacia −x en (k>0, ω<0). Así se separan ondas contrapropagantes. Para polarización se usan las componentes B_± en vez de las cartesianas.
4. **Superposición de la teoría lineal**: sobre el mapa ω–k se dibujan las curvas ω_r(k) y bandas de γ(k)>0 obtenidas resolviendo la relación de dispersión Vlasov-Maxwell caliente con un solver lineal. La potencia acumulada debe concentrarse sobre las ramas teóricas inestables; el máximo de potencia debe coincidir con el máximo de γ(k).

**Solvers de teoría lineal** (raíces de la relación de dispersión de plasma caliente, notación de Stix):
- **WHAMP** (Rönnmark 1982): el clásico, bi-Maxwelliano, suma de Maxwellianas.
- **PLUME** (Klein, Howes & Brown 2025; extiende Quataert 1998) y **NHDS** (Verscharen & Chandran 2018): bi-Maxwellianos, rápidos (segundos), root-finding en el plano complejo ω = ω_r + iγ.
- **DSHARK** (Astfalk, Görler & Jenko 2015, JGR 120, 7107): bi-kappa, propagación oblicua.
- **LEOPARD** (Astfalk & Jenko 2017, JGR 122, 89): distribuciones gyrotrópicas arbitrarias en rejilla.
- **ALPS** (Verscharen et al. 2018, JPP; Klein & Verscharen 2025): "Arbitrary Linear Plasma Solver", acepta f₀ gyrotrópicas arbitrarias (incluso VDFs medidas por naves); mucho más lento (1–10 h en 32 núcleos) que PLUME (segundos).
- **PDRK/BO** (Xie & Xiao 2016): método matricial de autovalores que da todos los modos a la vez.

La función de dispersión de plasma Z(ξ) = π^{−1/2} ∫ e^{−x²}/(x−ξ) dx (Fried & Conte 1961) aparece en todas las relaciones; para modos paralelos el argumento es ξ_j^± = (ω ± Ω_j)/(k v_{th,j}).

#### 1.3 Espectros de potencia magnética y medida de γ

- **P(k)**: densidad espectral reducida integrando |B̃(k)|² sobre las direcciones no analizadas. P(ω) análogamente.
- **Espectro de fluctuaciones** δB²/B₀²: energía magnética normalizada, diagnóstico de saturación.
- **Espectros 2D (k_∥, k_⊥)**: se proyecta la potencia en el plano paralelo-perpendicular a B₀. La firehose oblicua y el mirror aparecen en k_⊥ ≠ 0; EMIC y firehose paralela en k_⊥ ≈ 0. Esencial para separar ramas oblicuas de paralelas.
- **Medida de la tasa de crecimiento γ(k)**: se transforma E(x)→E(k) por FFT en cada paso guardado; se sigue E_B(k,t); durante la fase lineal E_B(k,t) = E₀ + E₁ e^{2γ(k)t}, y se ajusta ln E_B(k,t) vs t con una recta cuya pendiente es 2γ(k). El γ máximo sobre k se compara con el máximo de la teoría lineal. La energía total transversal δB_⊥²(t) marca las fases: equilibración → crecimiento exponencial (lineal) → saturación (no lineal). Este método directo (ajuste exponencial de la amplitud del modo) es "físicamente transparente, sin restricción sobre la VDF 3D, eficiente en tiempo y memoria", según se describe habitualmente en la literatura PIC.

#### 1.4 Diagnósticos de polarización, helicidad y compresibilidad

- **Componente compresiva vs transversal**: se descompone δB en δB_∥ (a lo largo de B₀, compresiva) y δB_⊥ (transversal). La razón δB_∥²/(δB_∥²+δB_⊥²) es la **compresibilidad magnética**.
- **Helicidad magnética reducida σ_m** (Matthaeus & Goldstein 1982): σ_m(k) ∝ Im(B̃_y* B̃_z)/(|B̃_y|²+|B̃_z|²), en función del ángulo θ_kB. σ_m > 0 / < 0 discrimina la mano de la polarización. En viento solar σ_m separa las ondas Alfvén-cyclotron izquierdas casi-paralelas (θ_kB<30°) de los KAW/whistler derechos oblicuos (40°<θ_kB<140°).
- **Polarización**: circular derecha (whistler, firehose paralela), circular izquierda (EMIC/ion-cyclotron, electron firehose propagante), lineal (mirror, firehose oblicua no propagante). Se obtiene de la relación de fase entre B_y y B_z, o del signo de σ_m.
- **Ángulo de propagación θ_kB**: del vector k dominante respecto a B₀; convención habitual θ_kB > 30–50° = oblicuo (Camporeale & Burgess usan ~50°; Gary & Nishimura, Li & Habbal usan 30°).
- **Correlación δn–δ|B|**: coeficiente de correlación r(δn, δ|B|). Anticorrelación fuerte (r<0) es la firma del mirror (y del modo lento); correlación positiva del modo rápido.

### BLOQUE 2 — Criterios de discriminación

#### 2.1 Curvas de umbral en el plano (β_∥, T_⊥/T_∥)

La forma canónica (Gary; Hellinger et al. 2006; Bale et al. 2009) para el contorno de máxima tasa de crecimiento γ_max fijado (habitualmente 10⁻³ Ω_p):

  **T_⊥/T_∥ = 1 + a / (β_∥ − β₀)^b**

**Coeficientes verificados de Hellinger et al. (2006), Tabla 1** (γ_max = 10⁻³ Ω_p; condiciones: 0.01 ≤ β_∥p ≤ 30, 0.1 ≤ T_⊥p/T_∥p ≤ 10, β_e=1, ω_pe/ω_ce=100):

| Inestabilidad | a | b | β₀ | Rama |
|---|---|---|---|---|
| Ion-cyclotron (AIC/proton cyclotron) | 0.43 | 0.42 | −0.0004 | T_⊥>T_∥ |
| Mirror | 0.77 | 0.76 | −0.016 | T_⊥>T_∥ |
| Firehose paralela | −0.47 | 0.53 | 0.59 | T_∥>T_⊥ |
| Firehose oblicua | −1.4 | 1.0 | −0.11 | T_∥>T_⊥ |

Escritas de forma equivalente y verbatim como aparecen en la literatura derivada (Chandran et al. 2011; Bale et al. 2009): umbral mirror R_m = 1 + 0.77(β_p + 0.016)^{−0.76} y firehose oblicua R_f = 1 − 1.4(β_p + 0.11)^{−1}. Para T_⊥>T_∥ los coeficientes a>0 (cotas superiores); para T_∥>T_⊥, a<0 (cotas inferiores). β₀ puede ser negativo (lo es en tres de las cuatro; el término β₀ sólo es importante para las dos firehose). El umbral de ion-cyclotron también se ajusta con exponente b ≈ 0.4 (Gary & Lee 1994; Hellinger et al. 2006). El umbral de **whistler anisotropy electrónico** (Gary & Wang 1996, JGR 101, 10749, γ=0.01 Ω_e) es **T_⊥e/T_∥e = 1 + 0.27/β_∥e^{0.57}**; en general los modos electrónicos se ajustan con T_⊥e/T_∥e − 1 = S_e/β_∥e^{α_e}, con 0.1 ≲ S_e ≲ 1 y 0.5 ≲ α_e < 0.7. Para electron firehose se usa la misma forma con a<0.

En la práctica, el umbral de marginalidad se calcula fijando γ = 10⁻³ Ω_ci (Hellinger et al. 2006) — el valor que mejor concuerda con los datos del viento solar; con γ_max menores (hasta 10⁻¹³ Ω_i) el umbral, sobre todo a bajo β_∥, se desplaza considerablemente (Astfalk & Jenko 2016).

#### 2.2 Tabla de discriminación de las inestabilidades iónicas

| Criterio | Ion-cyclotron/EMIC | Firehose paralela | Firehose oblicua | Mirror |
|---|---|---|---|---|
| Anisotropía impulsora | T_⊥>T_∥ | T_∥>T_⊥ | T_∥>T_⊥ | T_⊥>T_∥ |
| ω_r | finita (~0.1–0.5 Ω_p) | finita | ≈0 (no propagante) | ≈0 (no propagante) |
| θ_kB | casi-paralelo (~0°) | casi-paralelo | oblicuo (~60–70°) | oblicuo (~60–80°) |
| Polarización | circular izquierda | circular derecha | lineal | lineal |
| Compresibilidad | baja (transversal) | baja (transversal) | intermedia | alta (δB_∥ dominante) |
| δn–δ|B| | — | correlación positiva | correlación positiva (lineal) | **anticorrelada** |
| k típico | k_∥ d_i ~ 0.5–1 | k_∥ d_i ~ 1 | k ρ_i ~ 0.3–0.5 | k ρ_i ~ 0.3–0.5 |
| Rama del modo | proton-cyclotron (extensión Alfvén a k_∥d_p≳1) | fast-magnetosonic/whistler | Alfvén | slow-mode/magnetosonic |
| Eficacia isotropizante | resonante | auto-limitada, débil | muy eficaz (auto-destructiva) | muy eficaz (estructuras coherentes) |

La anticorrelación mirror sigue **δn/n = −(T_⊥/T_∥ − 1)·δB/B** (Hasegawa 1969). El mirror genera estructuras compresivas coherentes no propagantes (magnetic holes/peaks) con δB_∥ ≫ δB_⊥, elongadas a lo largo de B₀ local; el parámetro de mirror C_M = β_i⊥(T_i⊥/T_i∥ − 1) > 1 indica inestabilidad. La firehose oblicua tiene evolución "auto-destructiva": genera modos no propagantes que dispersan protones, reducen la anisotropía, destruyen su propia rama no propagante y se convierten (mode conversion) en modos propagantes amortiguados (Hellinger & Matsumoto 2000, 2001). En el régimen de bajo β_∥ la firehose paralela domina linealmente; alrededor de β_∥ ~ 5 domina la oblicua.

En la práctica observacional y numérica, la firehose oblicua y el mirror (ambas no propagantes) acotan mejor los datos que la ion-cyclotron o la firehose paralela, pese a que estas últimas tengan mayor γ — un resultado establecido en Bale, Kasper, Howes et al. (2009), PRL 103, 211101, que muestra potencia de fluctuaciones magnéticas de escala girocinética realzada a lo largo de los umbrales de mirror y firehose oblicua, y compresibilidad magnética realzada a alto β_∥ sólo sobre el umbral mirror.

#### 2.3 Modos electrónicos (sólo full-PIC)

- **Whistler anisotropy** (T_⊥e>T_∥e): circular derecha, casi-paralelo (γ máximo en k×B₀=0), ω_r > Ω_p, k_∥ c/ω_pe ~ 1; longitud dominante λ_e = 2π A_e^{−1/2} c/ω_pe (Gary & Wang 1996; Gary & Karimabadi 2006). Impone cota superior a T_⊥e/T_∥e.
- **Electron mirror** (T_⊥e>T_∥e): oblicuo, ω_r=0, compresivo; δB_x débilmente anticorrelacionado con n_e; se estabiliza antes que el whistler (Gary & Karimabadi 2006; Hellinger et al. 2018, "Electron mirror instability: PIC simulations").
- **Electron firehose** (T_∥e>T_⊥e): dos ramas — la propagante (p-EFI, ω_r finita, circular izquierda, casi-paralela, componente B_y dominante) y la aperiódica/oblicua (a-EFI, ω_r=0, la de mayor γ, sólo a ángulos oblicuos con k×B₀≠0). Camporeale & Burgess (2008, JGR 113, A07107) confirmaron con PIC 2D que la a-EFI oblicua domina el crecimiento inicial; López et al. (2019, 2022), Innocenti et al. (2019) y Micera et al. (2020, 2021) lo reprodujeron.

#### 2.4 Códigos PIC full-cinéticos y hybrid-PIC

**Full-PIC** (electrones e iones cinéticos; resuelven whistler, electron firehose, electron mirror): **VPIC** (relativista, charge-conserving), **OSIRIS** (relativista, 3D masivamente paralelo), **SMILEI** (FDTD/Yee, Boris, MPI-OpenMP-GPU), **iPIC3D** (semi-implícito, conserva energía, permite Δt grandes en escalas MHD), **ACRONYM**, **EPOCH**, **P3D**, **PSC**, **PIConGPU**, **WarpX**. Requieren resolver la longitud de Debye λ_D y ω_pe⁻¹, por lo que suelen usar ratios m_p/m_e reducidos y ω_pe/ω_ce reducidos (p. ej. ω_pe/ω_ce=4) para ahorrar recursos (y así reducir la separación d_e/λ_D).

**Hybrid-PIC** (iones cinéticos vía PIC, electrones fluido sin masa, ley de Ohm generalizada): **CAMELIA** (Current Advance Method Et cycLIc leApfrog; Matthews 1994; Franci, Hellinger et al.), **dHybridR**, **HYPERS**, **Pegasus** (Kunz, Stone & Bai 2014). NO resuelven la física electrónica cinética: no capturan whistler ni electron firehose/mirror, pero resuelven perfectamente EMIC, firehose y mirror iónicos a escalas d_i y ρ_i con mucho menor coste, permitiendo cajas grandes. La comparación directa PIC-full vs hybrid la hicieron **Cerri, Franci, Califano, Landi & Hellinger (2017)**, J. Plasma Phys. (arXiv:1703.02443) y **Franci et al. (2019)**, Front. Astron. Space Sci. 6, 64, usando CAMELIA, el código Euleriano hybrid Vlasov-Maxwell HVM y el full-PIC OSIRIS: los espectros coinciden bien hasta escalas comparables a d_e, por debajo de las cuales el full-PIC se empina más (efectos de inercia y cinética electrónica ausentes en el modelo hybrid isotérmico sin masa).

#### 2.5 Papers clave con figuras ω–k, umbrales y espectros de potencia

- **Hellinger, Trávníček, Kasper & Lazarus (2006)**, GRL 33, L09101, "Solar wind proton temperature anisotropy: Linear theory and WIND/SWE observations" — curvas de umbral en el plano (β_∥, T_⊥/T_∥) ("Brazil plot"); Tabla 1 de coeficientes.
- **Bale, Kasper, Howes, Quataert, Salem & Sundkvist (2009)**, PRL 103, 211101, "Magnetic fluctuation power near proton temperature anisotropy instability thresholds in the solar wind" — mapas de potencia δB/B y compresibilidad sobre los umbrales.
- **Gary & Wang (1996)**, JGR 101, 10749, "Whistler instability: Electron anisotropy upper bound" — PIC 2D, umbral whistler electrónico.
- **Gary & Karimabadi (2006)**, JGR 111, A11224, "Linear theory of electron temperature anisotropy instabilities: Whistler, mirror, and Weibel".
- **Camporeale & Burgess (2008)**, JGR 113, A07107 — electron firehose PIC, ramas propagante/oblicua, figuras γ vs θ.
- **Hellinger, Trávníček, Decyk & Schriver (2014)**, JGR 119, 59, "Oblique electron fire hose instability: Particle-in-cell simulations".
- **Hellinger et al. (2018)**, "Electron mirror instability: Particle-in-cell simulations" (arXiv:1806.04987) — mapas δB(t, k) y δB(t, θ_kB) con las propiedades del modo más inestable de whistler y mirror superpuestas.
- **Riquelme, Quataert & Verscharen (2015)**, ApJ 800, 27, "Particle-in-cell simulations of continuously driven mirror and ion cyclotron instabilities in high beta astrophysical and heliospheric plasmas".
- **Kunz, Schekochihin & Stone (2014)**, PRL 112, 205003, "Firehose and Mirror Instabilities in a Collisionless Shearing Plasma" (hybrid Pegasus) — espectros k y evolución; escala de saturación firehose ∝ S^{1/2}.
- **Matteini, Landi, Hellinger & Velli (2006)**, JGR, firehose paralela en expanding box; **Hellinger & Trávníček (2008)**, JGR 113, A10109, firehose oblicua; **Hellinger (2017)** (arXiv:1701.03665) firehose en expanding box.
- **Franci et al. (2016)**, ApJ 833, 91 y (2018) — espectros P(k) sub-iónicos con CAMELIA.
- **López et al. / Micera et al. (2020, 2021)** — PIC de electron firehose oblicuo y whistler.

## Recommendations

1. **Primer paso — configura el guardado**: fija dt_out para que ω_Ny = π/dt_out cubra con margen la frecuencia máxima de interés (Ω_p para modos iónicos; ω_ce para whistler). Fija L lo bastante grande para que Δk = 2π/L resuelva k ρ_i ~ 0.5 con ≥10 puntos. Umbral que cambia la decisión: si buscas whistler/electron-firehose, dt_out < 0.1 ω_ce⁻¹; si sólo iónicos, dt_out ~ 0.05 Ω_p⁻¹ basta.
2. **Construye el diagnóstico ω–k con B_± complejas** para obtener polarización y separación +k/−k de una vez; superpón las ramas de NHDS/PLUME (bi-Maxwelliano rápido) y, si la VDF se desvía de bi-Maxwelliana en la fase no lineal, re-analiza con ALPS/LEOPARD/DSHARK (bi-kappa o gyrotrópica arbitraria). Guarda el mapa (k_∥, k_⊥) para separar ramas oblicuas de paralelas.
3. **Mide γ(k) por ajuste log-lineal** de E_B(k,t) en la fase lineal (ln E_B ∝ 2γt) y valida contra la teoría; si γ_medido < γ_teórico sistemáticamente, revisa ruido PIC (partículas por celda), amortiguamiento numérico y ventana de ajuste. Verifica que el pico de potencia ω–k coincide con el máximo de γ(k).
4. **Aplica el árbol de decisión** de la tabla 2.2: (a) ¿ω_r≈0? → mirror o firehose oblicua; distínguelas por el signo de la anisotropía (T_⊥>T_∥ = mirror; T_∥>T_⊥ = firehose oblicua) y por δn–δ|B| (anticorrelación fuerte + δB_∥ dominante = mirror). (b) ¿ω_r finita y casi-paralela? → EMIC (izquierda, T_⊥>T_∥) vs firehose paralela (derecha, T_∥>T_⊥). Confirma con la posición en el plano (β_∥, T_⊥/T_∥) respecto a las curvas de la Tabla 1.
5. **Elige el código según la física**: hybrid-PIC (CAMELIA/dHybridR/Pegasus) si sólo te interesan modos iónicos (más barato, cajas grandes, sin restricción de λ_D/ω_pe); full-PIC (VPIC/OSIRIS/SMILEI/iPIC3D) si necesitas whistler o inestabilidades electrónicas. Usa ratios m_p/m_e y ω_pe/ω_ce reducidos con cautela y documenta su efecto.

## Caveats

- Las curvas de umbral de Hellinger et al. (2006) suponen plasma bi-Maxwelliano homogéneo con β_e=1 y ω_pe/ω_ce=100; en simulaciones con otros parámetros o VDFs no-bi-Maxwellianas los umbrales se desplazan (Astfalk & Jenko 2016 para bi-kappa: las colas supratérmicas estabilizan la firehose oblicua y desplazan la paralela; Walters et al. 2023 para VDFs fuera de equilibrio).
- La discrepancia observacional conocida —los datos alcanzan los umbrales de mirror y firehose oblicua pero no los de ion-cyclotron y firehose paralela pese a su mayor γ— sigue sin explicación completa; se atribuye a la naturaleza no propagante de los modos oblicuos, a especies minoritarias (alfas) y drifts (Maruca et al. 2012), a la baja eficiencia de extracción de energía (Shoji et al. 2009) o al aplanamiento cuasi-lineal de la región resonante (Isenberg et al. 2013). Este contraste con la teoría lineal es explícito en Hellinger et al. (2006).
- El "numerical mirror mode" del mirroring temporal y los modos de rejilla (k cerca de k_Ny) son artefactos que hay que descartar explícitamente; en la técnica de mirroring, todo lo que aparece por encima de ω_Ny/2 es reflejo espurio.
- La distinción propagante/no propagante depende de la resolución en ω; con dt_out grande un modo con ω_r pequeña puede parecer no propagante por aliasing temporal. Comprueba que dt_out resuelve la ω_r esperada.
- Los ratios m_p/m_e y ω_pe/ω_ce reducidos del full-PIC alteran la separación de escalas d_e/λ_D y pueden distorsionar cuantitativamente las ramas y umbrales electrónicos; los coeficientes de umbral electrónico dependen de β_∥e y de la relación de masas efectiva.
- Los coeficientes de la Tabla 1 se citan a veces en formas algebraicamente equivalentes (con β₀ dentro del paréntesis con signo cambiado); verifica siempre la convención antes de trazar las curvas.