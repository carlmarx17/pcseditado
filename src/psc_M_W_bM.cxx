// ======================================================================
// psc_M_W_bM - Mirror Weak Bi-Maxwellian
//
// beta_i_parallel=6.0, Ai=Ti_perp/Ti_parallel=1.5
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// mass_ratio=200, 1000 ppc, 1408x1408
// ======================================================================

#define PSC_CASE_LABEL "M-W-bM"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_M_W_bM"

#define PSC_NMAX_DEFAULT 1650000
#define PSC_CHECKPOINT_EVERY_DEFAULT 0
#define PSC_STATS_EVERY 50

#define PSC_MASS_RATIO 200.
#define PSC_LAMBDA0 20.
#define PSC_VA_OVER_C 0.05
#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 6.0
#define PSC_TI_PERP_OVER_TI_PAR 1.5
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#define PSC_DOMAIN_DI 30.0
#define PSC_NGRID_DEFAULT 1408
#define PSC_NP_Y_DEFAULT 64
#define PSC_NP_Z_DEFAULT 16
#define PSC_NICELL_DEFAULT 1000

#define PSC_BALANCE_INTERVAL 700
#define PSC_FIELDS_EVERY_DEFAULT 1000
#define PSC_PARTICLES_EVERY_DEFAULT 10000
#define PSC_COLLISION_TEMP g.Te_perp

#include "psc_anisotropy_case.hxx"
