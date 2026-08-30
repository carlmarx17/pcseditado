// ======================================================================
// psc_firehose_bimaxwellian_strong_bigbox40 -
//   Firehose Strong Bi-Maxwellian, caja 40 d_i (el doble de la
//   estandar de 20 d_i). Misma fisica que
//   psc_firehose_bimaxwellian_strong (beta_i_parallel=10.0, Ai=0.1,
//   beta_e_parallel=1.0, Ae=1.0, mass_ratio=200, 1000 ppc).
//
//   Es el gemelo Bi-Maxwelliano de psc_firehose_bikappa3_bigbox40:
//   mismos beta y misma anisotropia, sin colas suprathermales. Con
//   este par la comparacion kappa-vs-Maxwelliana del firehose queda
//   controlada (el par contra firehose_bimaxwellian_moderate_bigbox40
//   no lo estaba: difiere en beta_i_par 6 vs 10 y en Ai 0.3 vs 0.1).
//
//   Unica diferencia con la base: PSC_DOMAIN_DI=40 en vez de 20. Se
//   corre con PSC_NGRID=1152 (ver cosma_jobs/simulacion) para mantener
//   la misma resolucion que los casos estandar (~28.8 celdas/d_i).
//   El tamano de caja NO es overrideable por variable de entorno
//   (ver #ifndef PSC_DOMAIN_DI en psc_anisotropy_case.hxx), por eso
//   esto es un ejecutable aparte compilado con el define.
// ======================================================================

#define PSC_CASE_LABEL "firehose_bimaxwellian_strong_bigbox40"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_firehose_bimaxwellian_strong_bigbox40"

// Caja el doble de grande que la estandar (20 d_i).
#define PSC_DOMAIN_DI 40.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 10.0
#define PSC_TI_PERP_OVER_TI_PAR 0.1
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
