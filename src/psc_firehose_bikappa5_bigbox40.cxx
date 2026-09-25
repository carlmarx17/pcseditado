// ======================================================================
// psc_firehose_bikappa5_bigbox40 -
//   Firehose Bi-Kappa (kappa=5), caja 40 d_i. Misma fisica que
//   psc_firehose_bikappa3_bigbox40 y psc_firehose_bimaxwellian_strong_bigbox40
//   (beta_i_parallel=10.0, Ai=0.1, beta_e_parallel=1.0, Ae=1.0,
//   mass_ratio=200, 1000 ppc); unicamente cambia kappa=3 por kappa=5.
//   Cierra la serie Bi-Maxwelliana / kappa=5 / kappa=3 en caja grande.
//
//   Se corre con PSC_NGRID=1152 (ver cosma_jobs/simulacion) para
//   mantener la misma resolucion que los casos estandar (~28.8
//   celdas/d_i). El tamano de caja NO es overrideable por variable de
//   entorno (ver #ifndef PSC_DOMAIN_DI en psc_anisotropy_case.hxx), por
//   eso esto es un ejecutable aparte compilado con el define.
// ======================================================================

#define PSC_CASE_LABEL "firehose_bikappa5_bigbox40"
#define PSC_DISTRIBUTION_LABEL "Bi-Kappa"
#define PSC_OUTPUT_BASENAME "prt_firehose_bikappa5_bigbox40"

// Caja el doble de grande que la estandar (20 d_i).
#define PSC_DOMAIN_DI 40.0

#define PSC_USE_KAPPA 1
#define PSC_KAPPA 5.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 10.0
#define PSC_TI_PERP_OVER_TI_PAR 0.1
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
