
add_subdirectory(kg)
add_subdirectory(libpsc)

macro(add_psc_executable name)
  add_executable(${name})
  target_gtensor_sources(${name} PRIVATE ${name}.cxx)
  target_link_libraries(${name} psc)
endmacro(add_psc_executable)

if(NOT USE_CUDA)
  add_psc_executable(psc_bgk)
endif()

add_psc_executable(psc_bubble_yz)
add_psc_executable(psc_flatfoil_yz)
add_psc_executable(psc_flatfoil_yz_small)
add_psc_executable(psc_whistler)
add_psc_executable(psc_harris_xz)
add_psc_executable(psc_harris_yz)
add_psc_executable(psc_2d_shock)
add_psc_executable(psc_radiation)
add_psc_executable(psc_shock)
# --- Bi-Maxwellian anisotropy scan (1000 ppc, 576², mr=200) ---
# Mirror
add_psc_executable(psc_mirror_bimaxwellian_strong)
add_psc_executable(psc_mirror_bimaxwellian_moderate)
add_psc_executable(psc_mirror_bimaxwellian_weak)
# Firehose
add_psc_executable(psc_firehose_bimaxwellian_strong)
add_psc_executable(psc_firehose_bimaxwellian_moderate)
add_psc_executable(psc_firehose_bimaxwellian_weak)
# Whistler (electron-driven)
add_psc_executable(psc_whistler_bimaxwellian_strong)
add_psc_executable(psc_whistler_bimaxwellian_moderate)
add_psc_executable(psc_whistler_bimaxwellian_weak)

# --- Bi-Kappa anisotropy cases (1000 ppc, 576², mr=200) ---
add_psc_executable(psc_mirror_bikappa3)
# Mismo Ai/beta que psc_mirror_bimaxwellian_moderate, para comparar
# Bi-Maxwelliana vs Bi-Kappa a igualdad de anisotropia.
add_psc_executable(psc_mirror_bikappa3_moderate)
add_psc_executable(psc_mirror_bikappa5_moderate)
# Isotropic controls of the mirror moderate series (A_i = 1, same ion
# thermal energy): their electron heating is the numerical baseline.
add_psc_executable(psc_mirror_bimaxwellian_isotropic)
add_psc_executable(psc_mirror_bikappa5_isotropic)
add_psc_executable(psc_mirror_bikappa3_isotropic)
# Gemelo Bi-Kappa de psc_whistler_bimaxwellian_moderate.
add_psc_executable(psc_whistler_bikappa3_moderate)
add_psc_executable(psc_mirror_bikappa5)
add_psc_executable(psc_firehose_bikappa3)
add_psc_executable(psc_firehose_bikappa5)
# Gemelos de distribucion de psc_whistler_bimaxwellian_strong (mismos
# beta_e_par=0.5, Ae=3.0): serie whistler strong bi-Max / k=5 / k=3.
# Los jobs whistler refinan grilla/ppc por entorno (WHISTLER_PARAMETROS.md).
add_psc_executable(psc_whistler_bikappa3_strong)
add_psc_executable(psc_whistler_bikappa5_strong)
# Variante mi/me=800 de la serie whistler strong (PSC_MASS_RATIO es
# compile-time, como PSC_DOMAIN_DI en bigbox40): identica a su base salvo
# la razon de masas. Ver src/WHISTLER_PARAMETROS.md.
add_psc_executable(psc_whistler_bimaxwellian_strong_mr800)
add_psc_executable(psc_whistler_bikappa3_strong_mr800)
add_psc_executable(psc_whistler_bikappa5_strong_mr800)

# --- Firehose caja grande (40 d_i, misma fisica, ngrid=576) ---
# El tamano de caja es compile-time (#define PSC_DOMAIN_DI), no env,
# por eso son ejecutables aparte. Ver cosma_jobs/README.md.
add_psc_executable(psc_firehose_bimaxwellian_moderate_bigbox40)
add_psc_executable(psc_firehose_bikappa3_bigbox40)
# Gemelo Bi-Maxwelliano de psc_firehose_bikappa3_bigbox40 (mismo
# beta_i_par=10 y Ai=0.1): el par controlado para aislar el efecto de kappa.
add_psc_executable(psc_firehose_bimaxwellian_strong_bigbox40)
# Tercer miembro de la serie firehose strong 40 d_i (kappa=5).
add_psc_executable(psc_firehose_bikappa5_bigbox40)

add_psc_executable(psc_reconnection)
# Reconexion con parametros igualados a los casos de anisotropia
# (mi/me=200, 40x40 d_i, 1152^2, nicell=1000; ver ESCALADO_INESTABILIDADES.md)
add_psc_executable(psc_reconnection_comparable)
add_psc_executable(psc_turbulence_3D_kappa_3)
add_psc_executable(psc_turbulence_3D_maxwellian_3)

if(NOT USE_CUDA)
  install(
    TARGETS psc_bgk
    RUNTIME DESTINATION bin
  )
endif()

install(
  TARGETS psc_bubble_yz psc_flatfoil_yz psc_whistler psc_harris_xz psc_shock
  RUNTIME DESTINATION bin
)
