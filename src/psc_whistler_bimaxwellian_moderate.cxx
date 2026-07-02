// ======================================================================
// psc_whistler_bimaxwellian_moderate - Whistler Moderate Bi-Maxwellian
//
// beta_i_parallel=1.0, Ai=Ti_perp/Ti_parallel=1.0
// beta_e_parallel=0.5, Ae=Te_perp/Te_parallel=2.0
// mass_ratio=200, 1500 ppc, 1024x1024
// ======================================================================

#define PSC_CASE_LABEL "whistler_bimaxwellian_moderate"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_whistler_bimaxwellian_moderate"

#define PSC_BETA_E_PAR 0.5
#define PSC_BETA_I_PAR 1.0
#define PSC_TI_PERP_OVER_TI_PAR 1.0
#define PSC_TE_PERP_OVER_TE_PAR 2.0

#include "psc_anisotropy_case.hxx"
