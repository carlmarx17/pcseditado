// ======================================================================
// psc_whistler_bimaxwellian_strong_mr800 - Whistler Strong Bi-Maxwellian, mi/me=800
//
// beta_i_parallel=1.0, Ai=Ti_perp/Ti_parallel=1.0
// beta_e_parallel=0.5, Ae=Te_perp/Te_parallel=3.0
// mass_ratio=800: mass-ratio variant of the matching mi/me=200 case;
// the only difference is PSC_MASS_RATIO (see WHISTLER_PARAMETROS.md).
// ======================================================================

#define PSC_CASE_LABEL "whistler_bimaxwellian_strong_mr800"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_whistler_bimaxwellian_strong_mr800"

#define PSC_BETA_E_PAR 0.5
#define PSC_BETA_I_PAR 1.0
#define PSC_TI_PERP_OVER_TI_PAR 1.0
#define PSC_TE_PERP_OVER_TE_PAR 3.0

#define PSC_MASS_RATIO 800.

#include "psc_anisotropy_case.hxx"
