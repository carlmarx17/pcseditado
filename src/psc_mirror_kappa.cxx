// ======================================================================
// psc_mirror_kappa - Mirror Kappa
//
// beta_i_parallel=5.0, Ai=Ti_perp/Ti_parallel=3.0
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// Kappa is selected at compile time through PSC_KAPPA / PSC_KAPPA_SUFFIX.
// ======================================================================

#ifndef PSC_KAPPA
#define PSC_KAPPA 3.0
#endif

#ifndef PSC_KAPPA_SUFFIX
#define PSC_KAPPA_SUFFIX "kappa3"
#endif

#define PSC_CASE_LABEL "mirror_" PSC_KAPPA_SUFFIX
#define PSC_DISTRIBUTION_LABEL "Kappa"
#define PSC_OUTPUT_BASENAME "prt_mirror_" PSC_KAPPA_SUFFIX

#define PSC_USE_KAPPA 1
#define PSC_ALLOW_ENV_OVERRIDES 1

#define PSC_NMAX_DEFAULT 1800000
#define PSC_CHECKPOINT_EVERY_DEFAULT 7500
#define PSC_STATS_EVERY 50

#define PSC_MASS_RATIO 200.
#define PSC_LAMBDA0 20.
#define PSC_VA_OVER_C 0.05
#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 5.0
#define PSC_TI_PERP_OVER_TI_PAR 3.0
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#define PSC_DOMAIN_DI 20.0
#define PSC_NGRID_DEFAULT 1536
#define PSC_NP_Y_DEFAULT 64
#define PSC_NP_Z_DEFAULT 16
#define PSC_NICELL_DEFAULT 1000

#define PSC_BALANCE_INTERVAL 750
#define PSC_FIELDS_EVERY_DEFAULT 750
#define PSC_PARTICLES_EVERY_DEFAULT 1000
#define PSC_COLLISION_TEMP g.Te_perp

#include "psc_anisotropy_case.hxx"
