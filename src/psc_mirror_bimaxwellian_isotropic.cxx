// ======================================================================
// psc_mirror_bimaxwellian_isotropic - Mirror Isotropic Control Bi-Maxwellian
//
// beta_i_parallel=8.333333333333334 (25/3), Ai=Ti_perp/Ti_parallel=1.0
// beta_e_parallel=1.0, Ae=Te_perp/Te_parallel=1.0
// mass_ratio=200, 1000 ppc, 10 d_i box run with PSC_NGRID=288 (dx = 0.0347 d_i)
//
// Control of psc_mirror_bimaxwellian_moderate: same numerics, electrons and
// ion thermal energy, beta_i_par (1/2 + A_i) = 12.5, but isotropic and
// therefore mirror stable. The heating of its electrons is the numerical
// heating of the setup (dx/lambda_De = 8.68 at t = 0), which
// energy_audit.py subtracts, per unit volume, from the anisotropic run.
// The heating is a local property of dx, dt and ppc, so the control uses a
// 10 d_i box (a quarter of the cells) with exactly the twins' resolution:
// run it with PSC_NGRID=288.
// ======================================================================

#define PSC_CASE_LABEL "mirror_bimaxwellian_isotropic"
#define PSC_DISTRIBUTION_LABEL "Bi-Maxwellian"
#define PSC_OUTPUT_BASENAME "prt_mirror_bimaxwellian_isotropic"

// Reduced box of the numerical control; dx (and so dt) equal the twins'.
#define PSC_DOMAIN_DI 10.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 8.333333333333334
#define PSC_TI_PERP_OVER_TI_PAR 1.0
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
