# De la reconexión a corridas grandes de inestabilidades por anisotropía

Análisis basado en (a) lectura directa de este repositorio (`src/psc_reconnection.cxx`,
`src/psc_anisotropy_case.hxx`, `src/include/setup_particles.hxx`, jobs de COSMA) y
(b) literatura verificada (enlaces al final). Lo que es lectura de código está marcado
**[código]**; lo verificado en papers, **[lit]**; lo que es práctica general no
verificada en una fuente concreta, **[general]**. Lo que no pude verificar lo digo.

---

## 1. Punto de partida: lo que ya hay en el repo **[código]**

Contrario a la premisa de "adaptar el deck de reconexión", el repo **ya contiene** un
framework de inestabilidades (`psc_anisotropy_case.hxx` + casos firehose/mirror
bi-Maxwellian y bi-kappa). La pregunta correcta ya no es "cómo convertir el Harris en
plasma homogéneo" (hecho), sino **si el setup actual escala bien y qué corregir antes
de quemar horas en COSMA**. Comparación de los dos decks:

| Parámetro | `psc_reconnection.cxx` | `psc_anisotropy_case.hxx` |
|---|---|---|
| Geometría | 2D `dim_yz`, doble Harris | 2D `dim_yz`, homogéneo, B₀ = ẑ |
| Caja | 25.6 × 51.2 d_i | 20 × 20 d_i (bigbox: 40 × 40) |
| Grilla | 256 × 512 → Δx = 0.1 d_i | 576² → Δx = 0.0347 d_i (bigbox 1152²) |
| m_i/m_e | 25 | 200 |
| ω_pe/Ω_ce | 2 | 12.5 (ver §5, naming de `vA_over_c`) |
| ppc (`nicell`) | 100 × 4 especies | 1000 × 2 especies |
| CFL | 0.99 | 0.95 |
| Distribución | κ = 3 multivariada | bi-Maxwellian o bi-kappa (T⊥ ≠ T∥ vía `npt.T[]`) |
| Fronteras | Periódicas | Periódicas |
| Balance | cada 500 | cada 2500 |
| Duración por defecto | nmax 10⁷ (tope manual) | nmax 1.2 × 10⁶ ≈ 158 Ω_ci⁻¹ (calculado abajo) |

La inicialización anisotrópica ya es correcta en su estructura: `npt.T[0]=T[1]=T⊥`,
`npt.T[2]=T∥` con B₀ en z, ejes de malla alineados con B₀ — que es el único caso en
que `T[3]` por ejes de malla equivale a bi-Maxwelliana giroscópica. Si algún día
inclinas B₀, hay que rotar la matriz de temperaturas a mano (el sampler no conoce B).

El sampler `createKappaMultivariate` **[código]** usa mezcla de escala
Gaussiana‑Gamma: `Y ~ Gamma(κ−0.5)`, `S = √((κ−1.5)/Y)`, `p_i = Z_i·S·√(T_i/m)`.
Eso genera la bi‑kappa estándar f ∝ [1 + Q/(κ−3/2)]^−(κ+1) en la convención
"temperature‑preserving": la varianza es exactamente T_i (E[S²] = 1). Es la
convención correcta para comparar con NHDS/LEOPARD/ALPS **si** les pasas la misma T
física; verifica qué convención de θ vs T usa cada solver (algunos parametrizan con
θ² = (1−3/(2κ))·2T/m).

## 2. Qué hace la literatura en corridas grandes **[lit]**

Parámetros extraídos de los papers (leídos, no de memoria):

**Micera et al. 2020, ApJ 893:130** (firehose protónico paralelo, full PIC
semi-implícito ECsim, 1D): caja L = 60 d_i elegida explícitamente para que quepan
**>20 longitudes de onda del modo más inestable**; Δx ≈ 0.074 d_i;
Δt = 0.5 ω_pe⁻¹; **10⁴ ppc por especie**; periódicas; masa realista;
ω_pe/Ω_ce = 63.24. Reportan que corridas con distinta resolución y ppc dan
resultados similares (test de convergencia explícito). Ojo: ECsim es
energy-conserving semi-implícito y **no** está obligado a resolver λ_De — PSC
explícito sí (§3).

**Hellinger et al. 2019, ApJ 883:178** (firehose vs turbulencia, híbrido expanding
box, 3D): grilla 512 × 512 × 256; Δx = Δy = 0.25 d_i, Δz = 0.5 d_i (caja
128 × 128 × 128 d_i); **400 ppc** (protones); Δt = 0.05 Ω_ci⁻¹ con subciclado del
campo B a Δt/10; resistividad η = 10⁻³ μ₀v_A²/ω_ci para evitar acumulación de
energía en la escala de grilla; expansión t_exp = 10⁴ Ω_ci⁻¹; periódicas. Híbrido:
sin escala electrónica que resolver, por eso pueden usar Δx de 0.25–0.5 d_i.

**Riquelme, Quataert & Verscharen 2015, ApJ 800:27** (mirror + IC driven por shear,
full PIC TRISTAN-MP, β ~ 1–100): el estado no lineal lo domina mirror; la
anisotropía satura cerca del umbral lineal de mirror; δB ~ 0.3⟨B⟩ en la fase
secular; μ deja de conservarse cuando δB ≳ 0.1⟨B⟩. No extraje su tabla de
resolución numérica — si necesitas sus Δx/ppc, hay que leer su sección 2 en detalle.

**Relevantes para tu caso kappa** (existencia verificada, setups no extraídos):
López et al. 2019, ApJL 873:L20 (firehose electrónico bi-kappa, PIC); López et al.
2022, ApJ 930:158 (firehose 2D PIC acoplando escalas p⁺/e⁻); "Hybrid Simulation and
Quasi-linear Theory of Bi-Kappa Proton Instabilities" (ApJ 2023); y un método de
rejection sampling para kappa en PIC (arXiv:2512.04272) contra el que puedes
contrastar tu sampler.

**Patrones de diseño que se repiten** **[general, consistente con lo anterior]**:

- 1D basta para modos con k ∥ B (firehose paralelo, EMIC paralelo); 2D es el mínimo
  para mirror y firehose oblicuo; 3D solo cuando compites modos paralelos vs
  oblicuos simultáneamente o añades turbulencia. Tu `dim_yz` con B₀ = ẑ captura
  k∥ (z) y k⊥ (y) en un plano: correcto para mirror y firehose oblicuo, con la
  limitación 2D de un solo plano de k.
- Caja: la regla operativa es L ≳ 10–20 λ_peak del modo dominante, es decir
  k_min = 2π/L ≲ k_peak/10–20. Full PIC iónico: cajas de 20–100 d_i. Tu 20 d_i da
  k_min·d_i ≈ 0.31 — para firehose/mirror con k_peak·d_i ~ 0.3–0.8 eso deja el pico
  apenas en el 1er–3er armónico: **poco**. El bigbox40 (k_min·d_i ≈ 0.157) es lo
  mínimo defendible; para el inverse cascade post-saturación y modos oblicuos de k
  bajo, más grande aún es mejor.
- ppc en full PIC explícito: 100–1000 típico, 10⁴ en 1D de lujo. Tus 1000 están bien
  situados; el ruido en energía escala ∝ 1/ppc y en amplitud ∝ 1/√ppc.
- Duración: crecimiento con γ/Ω_ci ~ 10⁻³–10⁻¹ según cercanía al umbral →
  saturación en decenas–cientos de Ω_ci⁻¹; la relajación cuasilineal hacia el umbral
  marginal (lo que se compara con el plano β∥–T⊥/T∥) requiere cientos a ~10³ Ω_ci⁻¹.

## 3. Numérica crítica al pasar de reconexión a inestabilidades

**Grid heating / λ_De** **[lit + código]**. PIC explícito con interpolación lineal
calienta numéricamente si Δx ≳ 3–3.5 λ_De (Birdsall & Langdon; ver también
arXiv:2606.25528 sobre termalización numérica y arXiv:2503.05123 sobre smoothing).
Números de tus decks:

- Reconexión: T_e = 1/48 → λ_De = 0.144 d_e; Δx = 0.5 d_e → **Δx/λ_De ≈ 3.5**. Al límite pero defendible.
- Anisotropía: T_e∥ = β_e∥·B₀²/2 = 0.0032 (β_e∥ = 1) → λ_De = 0.057 d_e;
  Δx = 0.491 d_e → **Δx/λ_De ≈ 8.7**. El comentario en `psc_anisotropy_case.hxx`
  dice "dx/lambda_De ~ 3.78", pero ese número solo sale usando una temperatura
  iónica (√T_i∥ con β_i = 5 da 3.9); con la λ_De **electrónica** — que es la que
  manda para grid heating — estás ~2.5× por encima del criterio clásico. Esto es lo
  primero que verificaría (checklist §6): puede estar inyectando calentamiento
  espurio en los electrones a lo largo de 10⁶ pasos, y un T_e(t) que sube solo
  contamina directamente tu plano β∥–T⊥/T∥.

Mitigaciones si el control run confirma heating: subir β_e∥ (electrones más
calientes → λ_De mayor), refinar grilla (caro: coste ∝ N²·pasos en 2D), o smoothing
de corriente. **No encontré en el PSC público un filtro binomial/smoothing de
corriente configurable** — si tu versión editada no lo añadió, no cuentes con él.

**Ruido de partículas y semilla de los modos** **[general + código]**. En un plasma
homogéneo la inestabilidad crece desde el ruido térmico de las macropartículas. Con
más ppc el piso de ruido baja (∝ 1/ppc en energía), la fase lineal dura más y el
ajuste de γ es más limpio; con pocos ppc los modos arrancan de amplitudes ya
no-lineales o el ruido tapa los γ pequeños. Detalle de tu código: PSC inicializa
todas las partículas en el **centro de la celda** (`x_cc`), no uniformemente — el
espectro inicial de ruido de densidad no es el de un plasma térmico y tarda ~ un
periodo de plasma en termalizar. No es un problema para γ (mide después de los
primeros Ω_ci⁻¹) pero explica transientes iniciales. Además `createKappaMultivariate`
usa `std::random_device` por hilo **[código]**: las corridas no son reproducibles
bit a bit; para comparar γ entre corridas idénticas considera una semilla fija.

**k discreto y comparación con teoría lineal** **[general]**. La caja periódica solo
admite k_n = 2πn/L. Tu γ medido para "el modo dominante" es el γ(k_n) del armónico
más cercano al pico teórico, no γ_max del continuo. Compara con NHDS/LEOPARD/ALPS
**evaluados exactamente en los k_n de tu caja** (y en tu dirección de k del plano
y-z), no con el máximo de la curva. Esta es la razón física de que caja pequeña ⇒
γ aparente menor y umbral aparente corrido — importa directamente para tus contornos
en el plano β∥ vs T⊥/T∥.

**CFL y dispersión** **[general + código]**. `cfl = 0.99` (reconexión) deja margen
casi nulo; el 0.95 del caso de anisotropía es lo habitual. Cerca del límite de
Courant el error de dispersión EM del esquema de Yee es máximo justo en los k altos;
para corridas de 10⁶ pasos con whistlers/EMIC en juego, 0.75–0.95 es más prudente.
Micera et al. usan un esquema distinto (semi-implícito), su Δt no es comparable.

**Fronteras e inicialización** — ya resuelto en tu caso: periódicas + homogéneo sin
drifts (el deck de reconexión necesitaba doble Harris precisamente para ser
periódico; el homogéneo no tiene esa restricción). La perturbación inicial de campo
tampoco se necesita: se siembra del ruido.

**Conservación y correctores** **[código]**. Marder cada 100 + check de Gauss cada
100 están activos en el caso de anisotropía (en el de reconexión el check de Gauss
está desactivado, intervalo negativo). `DiagEnergies` tiene default 0 en el header
(`PSC_ENERGIES_EVERY_DEFAULT 0`) aunque tu runbook dice 5000 vía entorno: para
inestabilidades esa serie temporal es tu diagnóstico principal (γ del crecimiento
de δB² sale de ahí gratis) — actívala SIEMPRE y con cadencia alta (50–100 pasos;
es barata, un reduce global).

## 4. Escalas de tiempo y costo (números de tu setup) **[código, aritmética]**

Con m_i/m_e = 200, B₀ = 0.08 (unidades ω_pe): Ω_ci⁻¹ = m_i/B₀ = 2500 ω_pe⁻¹.
Δt = 0.95 · (0.491/√2) ≈ 0.33 ω_pe⁻¹ → **~7600 pasos por Ω_ci⁻¹**.

- nmax 1.2 × 10⁶ ≈ 158 Ω_ci⁻¹: suficiente para crecimiento y saturación de drives
  moderados/fuertes (γ/Ω_ci ≳ 10⁻²); **corto** para drives débiles cerca del umbral
  y para la relajación cuasilineal larga. Calcula nmax por caso: t_fin ≈ 10/γ + 200–500 Ω_ci⁻¹.
- Salidas: fields cada 500 pasos = 0.066 Ω_ci⁻¹ (≈15 muestras por Ω_ci⁻¹ — sobra
  para γ; el criterio es ≥10 muestras por e-folding, es decir intervalo ≤ 1/(10γ)).
- Checkpoint cada 5000 pasos = 0.66 Ω_ci⁻¹ → 240 checkpoints en una corrida. Cada
  checkpoint serializa ~6.6 × 10⁸ partículas (≥20 GB): es mucho I/O. Con el límite
  de 48 h de cosma7-rp basta checkpointear cada ~2–4 h de wallclock (equivalente a
  cada 5–10 × 10⁴ pasos).

**Memoria**: N_prt = n_celdas × nicell × n_especies. Estándar: 576² × 1000 × 2 =
6.6 × 10⁸ partículas; a ~32–64 B/partícula (single precision + overhead de sorting)
→ 25–45 GB agregados + campos (despreciables en comparación). Bigbox40: ×4.

**Horas-núcleo** (fórmula, no promesa): coste ≈ N_prt × n_pasos / R, con
R ≈ 3–10 × 10⁶ particle-pushes/s/core en CPU **[general]**. Estándar:
6.6 × 10⁸ × 1.2 × 10⁶ / (5 × 10⁶) ≈ 4 × 10⁴ core-h (~40 h en 1024 ranks). Bigbox40:
~1.6 × 10⁵ core-h — consistente con tu job de 83 nodos × 28 × 48 h que ya prevé
reanudar desde checkpoint. La descomposición 48 × 48 con parches de 24² celdas está
bien; nota que en plasma **homogéneo** el load balancing casi no trabaja (a
diferencia de la lámina de Harris que concentra partículas): puedes subir
`PSC_BALANCE_INTERVAL` o desactivarlo y ahorrarte ese overhead.

**Diagnósticos — qué guardar**: prioriza (1) serie de energías densa (barata),
(2) campos B a cadencia fija para espectros δB(k,t) y ajuste de γ por modo,
(3) momentos (n, v, P⊥, P∥ por especie) a la misma cadencia — de P⊥/P∥ sale tu
trayectoria en el plano β∥–T⊥/T∥, (4) partículas crudas solo en subregión y rara vez
— tu deck ya restringe a la ventana 0.4–0.6 de la caja **[código]**, bien. Los f(v)
para comparar con la teoría kappa se reconstruyen de esos dumps escasos.

## 5. Checks específicos de tu código (lectura directa) **[código]**

1. **`vA_over_c` no es v_A/c.** El código hace `g.B0 = g.vA_over_c` con B en
   unidades donde Ω_ce = B. Eso fija ω_ce/ω_pe = 0.08 (ω_pe/Ω_ce = 12.5); el v_A/c
   físico resultante es B₀/√(m_i n) = 0.08/√200 ≈ 5.7 × 10⁻³. Las betas están bien
   (se definen desde B₀² directamente), pero cualquier interpretación de
   velocidades en unidades de "v_A del input" está mal por un factor √m_i.
2. **Δx/λ_De ≈ 8.7 con la λ_De electrónica** (§3) vs el 3.78 comentado en el
   header. Verificar con control run.
3. **`DiagEnergies` default 0** — asegúrate de que todos los jobs exportan
   `PSC_ENERGIES_EVERY` (y bájalo a 50–100).
4. **Semilla RNG no reproducible** (`std::random_device` en el sampler kappa).
5. **Checkpoint cada 5000 pasos** = I/O excesivo (§4).
6. **CFL**: 0.95 OK; no heredar el 0.99 del deck de reconexión.
7. El deck de reconexión inicializa kappa **isótropa** (T[0]=T[1]=T[2]); tu caso de
   anisotropía ya hace la bi-kappa correctamente vía T[] — nada que portar de vuelta.

## 6. Checklist de decisiones antes de la próxima campaña grande

1. Para cada punto (β∥, T⊥/T∥, κ): correr NHDS/LEOPARD/ALPS primero → k_peak,
   γ_max, dirección del modo. De ahí: L ≥ 10–20·(2π/k_peak) y nmax ≥
   (10/γ_max + 300 Ω_ci⁻¹)/Δt. La caja se decide con el solver, no al revés.
2. Control run isotrópico (T⊥/T∥ = 1, mismo todo): mide grid heating puro
   (T_e(t), T_i(t) seculares) y el piso de ruido δB²(k). Si T_e sube
   apreciablemente en ~100 Ω_ci⁻¹ → §3 mitigaciones antes de producir.
3. Convergencia en ppc (250/500/1000) y en Δx (×2) en caja pequeña, un solo punto
   físico, comparando γ del modo dominante. Micera et al. hacen exactamente esto.
4. γ por modo: fit exponencial de log|δB_k|² en la fase lineal, por cada armónico
   k_n; comparar con el solver evaluado en esos mismos k_n.
5. Energía total conservada a <1% en toda la corrida (con Marder activo el ΔE es
   diagnóstico, no corrección de energía).
6. Presupuesto: coste ≈ N_prt·n_pasos/R (§4); añade 20% por diagnósticos e I/O y
   planifica reanudaciones para >48 h.
7. Decidir 2D vs 3D por física, no por defecto: mirror vs EMIC en competencia
   (tu plano β∥–T⊥/T∥ con T⊥ > T∥) es sensible a la dimensionalidad; Riquelme et
   al. 2015 (2D/3D PIC) y Hellinger et al. 2019 (3D híbrido) son las referencias de
   contraste.

## No verificado / pendiente

- Detalles de resolución de Riquelme et al. 2015 y de los setups de López et al.
  2019/2022 (los papers existen y son los relevantes; no extraje sus tablas).
- Si el PSC público (o tu fork) tiene smoothing de corriente configurable: no lo
  encontré en la documentación ni en los fuentes que revisé.
- Propiedades exactas de conservación del pusher `1vbec` de PSC (es
  charge-conserving por construcción Villasenor–Buneman; sobre su comportamiento
  de grid heating no hay caracterización publicada que haya encontrado).
- El factor exacto "3.4–5" citado en tu header como límite seguro de Δx/λ_De: el
  criterio clásico de Birdsall & Langdon es del orden de 3; el valor preciso depende
  del orden de interpolación y del esquema. Trátalo como orden de magnitud.

## Apéndice: caso `psc_reconnection_comparable` **[código]**

Creado para la comparativa reconexión ↔ inestabilidades con parámetros igualados a
los casos de anisotropía (convención bigbox40):

| Parámetro | Valor | Estado |
|---|---|---|
| m_i/m_e | 200 | igualado |
| Caja | 40 × 40 d_i (hojas en ±10 d_i) | igualado |
| Grilla | 1152² → 28.8 celdas/d_i, Δx = 0.491 d_e | igualado |
| `nicell` | 1000 | igualado |
| CFL | 0.95 | igualado |
| κ | 3.0 | igualado |
| np | 48 × 48 (2304 ranks, parches 24²) | igualado |
| Salidas/checks/balance/energías | mismos intervalos y mismos env-overrides | igualado |
| ω_pe/Ω_ce | **2.0** (anisotropía: 12.5) | **diferencia deliberada** |

La razón de la única diferencia: en Harris el balance de presión fija
T_e = 1/(2(ω_pe/Ω_ce)²(1+T_i/T_e)); con 12.5 saldría λ_De = 0.023 d_e →
Δx/λ_De ≈ 21 (calentamiento de grilla seguro). Con 2.0: λ_De = 0.144 d_e →
Δx/λ_De = 3.4, igual de sano que los casos de anisotropía corregidos. Consecuencia:
comparar en unidades iónicas (d_i, Ω_ci, v_A, B₀), no en ω_pe ni en c.

Escalas del caso: Ω_ci⁻¹ = 400 ω_pe⁻¹ ≈ 1212 pasos (Δt ≈ 0.33 ω_pe⁻¹);
nmax por defecto 250 000 ≈ 206 Ω_ci⁻¹ (`PSC_NMAX` para cambiarlo). Partículas
≈ 1152² × 1000 × Σn ≈ 6–7 × 10⁸ (similar al caso estándar de anisotropía).
Costo ≈ 250k pasos: ~6× más barato por Ω_ci⁻¹ que anisotropía en pasos, total del
orden de 3–5 × 10⁴ core-h con R ~ 5 × 10⁶ pushes/s/core.

Advertencia encontrada al revisar los bigbox **[código]**: `psc_firehose_*_bigbox40`
tiene `PSC_DOMAIN_DI=40` en compile-time pero `ngrid` por defecto **576** (heredado
del header); la resolución correcta depende de exportar `PSC_NGRID=1152` en el job.
Si se lanza sin esa variable, corre en silencio a la mitad de resolución
(14.4 celdas/d_i, Δx/λ_De ×2 peor). El caso comparable de reconexión ya trae 1152
como default para evitar ese modo de fallo.

## Referencias consultadas

- Micera et al. 2020, ApJ 893:130 — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ab7faa) · [arXiv:1907.08502](https://arxiv.org/abs/1907.08502)
- Hellinger et al. 2019, ApJ 883:178 — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ab3e01) · [arXiv:1908.07760](https://arxiv.org/abs/1908.07760)
- Riquelme, Quataert & Verscharen 2015, ApJ 800:27 — [IOPscience](https://iopscience.iop.org/article/10.1088/0004-637X/800/1/27) · [arXiv:1402.0014](https://arxiv.org/abs/1402.0014)
- López et al. 2019, ApJL 873:L20; López et al. 2022, ApJ 930:158 (setups no extraídos)
- Hybrid simulation & QL theory of bi-kappa proton instabilities — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/aceb5b)
- Kappa sampling en PIC — [arXiv:2512.04272](https://arxiv.org/abs/2512.04272)
- Grid heating / termalización numérica — [arXiv:2606.25528](https://arxiv.org/abs/2606.25528) · smoothing: [arXiv:2503.05123](https://arxiv.org/abs/2503.05123) · [Finite spatial-grid effects (CPC)](https://www.sciencedirect.com/science/article/abs/pii/S001046552030268X)
- PSC: Germaschewski et al. 2016, JCP 318:305 — [arXiv:1310.7866](https://arxiv.org/abs/1310.7866) · [repo](https://github.com/psc-code/psc) · [docs](https://psc.readthedocs.io/en/latest/)
- Hellinger et al. 2006 (umbral marginal solar wind, contexto del plano β∥–T⊥/T∥); Agudelo Rueda et al. 2024, ApJ 971:109 (referencia base de tu deck de reconexión)
