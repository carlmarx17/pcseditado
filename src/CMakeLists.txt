
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
# --- Bi-Maxwellian anisotropy scan (1500 ppc, 1024², mr=200) ---
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

# --- Bi-Kappa anisotropy cases (1500 ppc, 1024², mr=200) ---
add_psc_executable(psc_mirror_bikappa3)
# Mismo Ai/beta que psc_mirror_bimaxwellian_moderate, para comparar
# Bi-Maxwelliana vs Bi-Kappa a igualdad de anisotropia.
add_psc_executable(psc_mirror_bikappa3_moderate)
add_psc_executable(psc_mirror_bikappa5)
add_psc_executable(psc_firehose_bikappa3)
add_psc_executable(psc_firehose_bikappa5)

add_psc_executable(psc_reconnection)
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
