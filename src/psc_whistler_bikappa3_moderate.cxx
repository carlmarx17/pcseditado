// ======================================================================
// psc_whistler_bikappa3_moderate - Whistler Moderate Bi-Kappa (kappa=3)
//
// beta_i_parallel=1.0, Ai=Ti_perp/Ti_parallel=1.0
// beta_e_parallel=0.5, Ae=Te_perp/Te_parallel=2.0
// mass_ratio=200, 1000 ppc, 576x576
//
// Mismas condiciones fisicas que psc_whistler_bimaxwellian_moderate,
// unicamente cambiando la distribucion inicial a Bi-Kappa con kappa=3.
// Gemelo de distribucion de ese caso: aisla el efecto de las colas
// supratermicas sobre la inestabilidad whistler.
// ======================================================================

#define PSC_CASE_LABEL "whistler_bikappa3_moderate"
#define PSC_DISTRIBUTION_LABEL "Bi-Kappa"
#define PSC_OUTPUT_BASENAME "prt_whistler_bikappa3_moderate"

#define PSC_USE_KAPPA 1
#define PSC_KAPPA 3.0

#define PSC_BETA_E_PAR 0.5
#define PSC_BETA_I_PAR 1.0
#define PSC_TI_PERP_OVER_TI_PAR 1.0
#define PSC_TE_PERP_OVER_TE_PAR 2.0

#include "psc_anisotropy_case.hxx"
