// ======================================================================
// psc_mirror_bikappa3_isotropic - Mirror Isotropic Control Bi-Kappa (kappa=3)
//
// beta_i_parallel=8.333333333333334 (25/3), Ai=Ti_perp/Ti_parallel=1.0
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// mass_ratio=200, 1000 ppc, 576x576, kappa=3 for both species
//
// Control of psc_mirror_bikappa3_moderate: same numerics, electrons and
// ion thermal energy, beta_i_par (1/2 + A_i) = 12.5, but isotropic and
// therefore mirror stable. The heating of its electrons is the numerical
// heating of the setup (dx/lambda_De = 8.68 at t = 0), which
// energy_audit.py subtracts from the anisotropic run.
// ======================================================================

#define PSC_CASE_LABEL "mirror_bikappa3_isotropic"
#define PSC_DISTRIBUTION_LABEL "Bi-Kappa"
#define PSC_OUTPUT_BASENAME "prt_mirror_bikappa3_isotropic"

#define PSC_USE_KAPPA 1
#define PSC_KAPPA 3.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 8.333333333333334
#define PSC_TI_PERP_OVER_TI_PAR 1.0
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
