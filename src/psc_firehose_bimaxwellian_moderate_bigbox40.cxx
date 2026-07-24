// ======================================================================
// psc_firehose_bimaxwellian_moderate_bigbox40 -
//   Firehose Moderate Bi-Maxwellian, caja 40 d_i (el doble de la
//   estandar de 20 d_i). Misma fisica que
//   psc_firehose_bimaxwellian_moderate (beta_i_parallel=6.0, Ai=0.3,
//   beta_e_parallel=1.0, Ae=1.0, mass_ratio=200, 1500 ppc).
//
//   Unica diferencia: PSC_DOMAIN_DI=40 en vez de 20. Con ngrid=576
//   (sin cambios) la resolucion pasa de ~28.8 a ~14.4 celdas/d_i.
//   El tamano de caja NO es overrideable por variable de entorno
//   (ver #ifndef PSC_DOMAIN_DI en psc_anisotropy_case.hxx), por eso
//   esto es un ejecutable aparte compilado con el define.
// ======================================================================

#define PSC_CASE_LABEL "firehose_bimaxwellian_moderate_bigbox40"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_firehose_bimaxwellian_moderate_bigbox40"

// Caja el doble de grande que la estandar (20 d_i).
#define PSC_DOMAIN_DI 40.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 6.0
#define PSC_TI_PERP_OVER_TI_PAR 0.3
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
