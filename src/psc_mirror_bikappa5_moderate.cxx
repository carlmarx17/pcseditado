// ======================================================================
// psc_mirror_bikappa5_moderate - Mirror Moderate Bi-Kappa (kappa=5)
//
// Mismas condiciones fisicas que psc_mirror_bimaxwellian_moderate y
// psc_mirror_bikappa3_moderate (beta_i_parallel=5.0,
// Ai=Ti_perp/Ti_parallel=2.0, beta_e_parallel=1.0,
// Ae=Te_perp/Te_parallel=1.0, mass_ratio=200, 1000 ppc, 576x576),
// unicamente cambiando kappa=3 por kappa=5. Cierra la serie
// Bi-Maxwelliana / kappa=5 / kappa=3 a igualdad de anisotropia, para
// ver como escala el efecto de las colas supratermicas con kappa.
// ======================================================================

#define PSC_CASE_LABEL "mirror_bikappa5_moderate"
#define PSC_DISTRIBUTION_LABEL "Bi-Kappa"
#define PSC_OUTPUT_BASENAME "prt_mirror_bikappa5_moderate"

#define PSC_USE_KAPPA 1
#define PSC_KAPPA 5.0

#define PSC_BETA_E_PAR 1.0
#define PSC_BETA_I_PAR 5.0
#define PSC_TI_PERP_OVER_TI_PAR 2.0
#define PSC_TE_PERP_OVER_TE_PAR 1.0

#include "psc_anisotropy_case.hxx"
