
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
add_psc_executable(psc_mirror_maxwellian)
add_psc_executable(psc_firehose_maxwellian)
add_psc_executable(psc_mirror_kappa)
add_psc_executable(psc_firehose_kappa)

# --- Bi-Maxwellian anisotropy scan (2000 ppc, 1408², mr=200) ---
# Mirror
add_psc_executable(psc_M_S_bM)   # Mirror Strong  βi∥=5.0 Ai=3.0
add_psc_executable(psc_M_M_bM)   # Mirror Moderate βi∥=5.0 Ai=2.0
add_psc_executable(psc_M_W_bM)   # Mirror Weak    βi∥=6.0 Ai=1.5
# Firehose
add_psc_executable(psc_F_S_bM)   # Firehose Strong  βi∥=10.0 Ai=0.1
add_psc_executable(psc_F_M_bM)   # Firehose Moderate βi∥=6.0  Ai=0.3
add_psc_executable(psc_F_W_bM)   # Firehose Weak    βi∥=3.0  Ai=0.6
# Whistler (electron-driven)
add_psc_executable(psc_W_S_bM)   # Whistler Strong  βe∥=0.5 Ae=3.0
add_psc_executable(psc_W_M_bM)   # Whistler Moderate βe∥=0.5 Ae=2.0
add_psc_executable(psc_W_W_bM)   # Whistler Weak    βe∥=0.5 Ae=1.5

add_psc_executable(psc_reconnection)

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
