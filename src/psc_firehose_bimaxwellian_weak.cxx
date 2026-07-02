// ======================================================================
// psc_firehose_bimaxwellian_weak - Firehose Weak Bi-Maxwellian
//
// beta_i_parallel=3.0, Ai=Ti_perp/Ti_parallel=0.6
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// mass_ratio=200, 1500 ppc, 1024x1024
// ======================================================================

#define PSC_CASE_LABEL "firehose_bimaxwellian_weak"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_firehose_bimaxwellian_weak"

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 3.0
#define PSC_TI_PERP_OVER_TI_PAR 0.6
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
