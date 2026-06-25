#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include "../libpsc/psc_heating/psc_heating_impl.hxx"

#include <cstdlib>
#include <string>

#ifndef PSC_USE_KAPPA
#define PSC_USE_KAPPA 0
#endif

#ifndef PSC_KAPPA
#define PSC_KAPPA 3.0
#endif

#ifndef PSC_ALLOW_ENV_OVERRIDES
#define PSC_ALLOW_ENV_OVERRIDES 1
#endif

#ifndef PSC_COLLISION_TEMP
#define PSC_COLLISION_TEMP g.Te_perp
#endif

// ======================================================================
// Particle kinds

enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS,
};

// ======================================================================
// PscFlatfoilParams

struct PscFlatfoilParams
{
  double BB;
  double Zi;
  double mass_ratio;
  double lambda0;

  double vA_over_c;
  double beta_e_par;
  double beta_i_par;
  double Ti_perp_over_Ti_par;
  double Te_perp_over_Te_par;
  double n;

  // calculated from the above
  double B0;
  double Te_par;
  double Te_perp;
  double Ti_par;
  double Ti_perp;
  double mi;
  double me;

  double d_i;
};

// ======================================================================
// Global parameters

namespace
{

PscFlatfoilParams g;

std::string read_checkpoint_filename;

PscParams psc_params;

int ngrid = PSC_NGRID_DEFAULT;
int np_y = PSC_NP_Y_DEFAULT;
int np_z = PSC_NP_Z_DEFAULT;
int nicell = PSC_NICELL_DEFAULT;
int fields_every = PSC_FIELDS_EVERY_DEFAULT;
int particles_every = PSC_PARTICLES_EVERY_DEFAULT;
int balance_interval = PSC_BALANCE_INTERVAL;
int continuity_every = 5000;
int energies_every = 5000;

} // namespace

// ======================================================================
// PSC configuration

using Dim = dim_yz;

using PscConfig = PscConfig1vbecSingle<Dim>;

using Writer = WriterDefault;

using MfieldsState = PscConfig::MfieldsState;
using Mparticles = PscConfig::Mparticles;
using Balance = PscConfig::Balance;
using Collision = PscConfig::Collision;
using Checks = PscConfig::Checks;
using Marder = PscConfig::Marder;
using OutputParticles = PscConfig::OutputParticles;

// ======================================================================

int envInt(const char* name, int fallback)
{
#if PSC_ALLOW_ENV_OVERRIDES
  if (const char* value = std::getenv(name)) {
    return std::stoi(value);
  }
#endif
  return fallback;
}

// ======================================================================
// setupParameters

void setupParameters()
{
  psc_params.nmax = envInt("PSC_NMAX", PSC_NMAX_DEFAULT);
  psc_params.cfl = 0.95;
  psc_params.write_checkpoint_every_step =
    envInt("PSC_CHECKPOINT_EVERY", PSC_CHECKPOINT_EVERY_DEFAULT);
  psc_params.stats_every = PSC_STATS_EVERY;

  ngrid = envInt("PSC_NGRID", ngrid);
  np_y = envInt("PSC_NP_Y", np_y);
  np_z = envInt("PSC_NP_Z", np_z);
  nicell = envInt("PSC_NICELL", nicell);
  fields_every = envInt("PSC_FIELDS_EVERY", fields_every);
  particles_every = envInt("PSC_PARTICLES_EVERY", particles_every);
  balance_interval = envInt("PSC_BALANCE_INTERVAL", balance_interval);
  continuity_every = envInt("PSC_CONTINUITY_EVERY", continuity_every);
  energies_every = envInt("PSC_ENERGIES_EVERY", energies_every);

  g.BB = 1.0;
  g.Zi = 1.;
  g.mass_ratio = PSC_MASS_RATIO;
  g.lambda0 = PSC_LAMBDA0;

  g.vA_over_c = PSC_VA_OVER_C;
  g.beta_e_par = PSC_BETA_E_PAR;
  g.beta_i_par = PSC_BETA_I_PAR;
  g.Ti_perp_over_Ti_par = PSC_TI_PERP_OVER_TI_PAR;
  g.Te_perp_over_Te_par = PSC_TE_PERP_OVER_TE_PAR;
  g.n = 1.0;

  g.mi = g.mass_ratio * g.Zi;
  g.me = 1.0;

  g.B0 = g.vA_over_c;
  g.Te_par = g.beta_e_par * sqr(g.B0) / 2.;
  g.Te_perp = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par = g.beta_i_par * sqr(g.B0) / 2.;
  g.Ti_perp = g.Ti_perp_over_Ti_par * g.Ti_par;
}

// ======================================================================
// setupGrid

Grid_t* setupGrid()
{
  g.d_i = std::sqrt(g.mass_ratio / g.n);

  double domain_size = PSC_DOMAIN_DI * g.d_i;

  Grid_t::Real3 LL = {1.0, domain_size, domain_size};
  Int3         gdims = {1, ngrid, ngrid};
  Int3         np    = {1, np_y, np_z};

  Grid_t::Domain domain{gdims, LL, -.5 * LL, np};

  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  mpi_printf(MPI_COMM_WORLD, "case = %s\n", PSC_CASE_LABEL);
  mpi_printf(MPI_COMM_WORLD, "distribution = %s\n", PSC_DISTRIBUTION_LABEL);
  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De (background) = %g\n",
             sqrt(PSC_COLLISION_TEMP));

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = nicell;

  double dt = psc_params.cfl * courant_length(domain);
  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) {
    ibn[0] = 0;
  }
  if (Dim::InvarY::value) {
    ibn[1] = 0;
  }
  if (Dim::InvarZ::value) {
    ibn[2] = 0;
  }

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ======================================================================
// initializeParticles

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts)
{
#if PSC_USE_KAPPA
  partitionAndSetupParticles(
    setup_particles, balance, grid_ptr, mprts,
    [&](int kind, Double3 pos, int patch, Int3 idx, psc_particle_np& np) {
      psc_particle_npt npt{};
      npt.kind = kind;
      switch (kind) {
        case MY_ION:
          npt.n = g.n;
          npt.T[0] = g.Ti_perp;
          npt.T[1] = g.Ti_perp;
          npt.T[2] = g.Ti_par;
          break;
        case MY_ELECTRON:
          npt.n = g.n;
          npt.T[0] = g.Te_perp;
          npt.T[1] = g.Te_perp;
          npt.T[2] = g.Te_par;
          break;
        default: assert(0);
      }
      np.n = npt.n;
      np.p = setup_particles.createKappaMultivariate(npt);
    });
#else
  partitionAndSetupParticles(setup_particles, balance, grid_ptr, mprts,
                             [&](int kind, Double3 crd, psc_particle_npt& npt) {
                               switch (kind) {
                                 case MY_ION:
                                   npt.n = g.n;
                                   npt.T[0] = g.Ti_perp;
                                   npt.T[1] = g.Ti_perp;
                                   npt.T[2] = g.Ti_par;
                                   break;
                                 case MY_ELECTRON:
                                   npt.n = g.n;
                                   npt.T[0] = g.Te_perp;
                                   npt.T[1] = g.Te_perp;
                                   npt.T[2] = g.Te_par;
                                   break;
                                 default: assert(0);
                               }
                             });
#endif
}

// ======================================================================
// initializeFields

void initializeFields(MfieldsState& mflds)
{
  setupFields(mflds, [&](int m, double crd[3]) {
    switch (m) {
      case HZ: return g.B0;
      default: return 0.;
    }
  });
}

// ======================================================================
// run

void run()
{
  mpi_printf(MPI_COMM_WORLD, "*** Setting up %s (%s) ...\n",
             PSC_CASE_LABEL, PSC_DISTRIBUTION_LABEL);

  setupParameters();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty()) {
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);
  }

  psc_params.balance_interval = balance_interval;
  Balance balance{3};

  psc_params.sort_interval = 10;

  int collision_interval = -10;
  double collision_nu =
    3.76 * std::pow(PSC_COLLISION_TEMP, 2.) / g.Zi / g.lambda0;
  Collision collision{grid, collision_interval, collision_nu};

  ChecksParams checks_params{};
  checks_params.continuity.check_interval = continuity_every;
  checks_params.continuity.err_threshold = 1e-4;
  checks_params.continuity.print_max_err_always = true;
  checks_params.continuity.dump_always = false;
  checks_params.gauss.check_interval = 100;
  checks_params.gauss.err_threshold = 1e-4;
  checks_params.gauss.print_max_err_always = true;
  checks_params.gauss.dump_always = false;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  double marder_diffusion = 0.9;
  int marder_loop = 3;
  bool marder_dump = false;
  psc_params.marder_interval = 100;
  Marder marder(grid, marder_diffusion, marder_loop, marder_dump);

  OutputFieldsItemParams outf_item_params{};
  OutputFieldsParams outf_params{};
  outf_item_params.pfield.out_interval = fields_every;
  outf_params.fields = outf_item_params;
  outf_params.moments = outf_item_params;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, outf_params};

  OutputParticlesParams outp_params{};
  outp_params.every_step = particles_every;
  outp_params.data_dir = ".";
  outp_params.basename = PSC_OUTPUT_BASENAME;
  outp_params.lo = {0, int(0.4 * ngrid), int(0.4 * ngrid)};
  outp_params.hi = {1, int(0.6 * ngrid), int(0.6 * ngrid)};
  OutputParticles outp{grid, outp_params};

  DiagEnergies oute{grid.comm(), energies_every};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_particles(grid);
#if PSC_USE_KAPPA
  setup_particles.kappa = PSC_KAPPA;
#endif
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;

  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_particles, balance, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto psc = makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts,
                                          balance, collision, checks, marder,
                                          diagnostics);

  psc.integrate();
}

// ======================================================================
// main

int main(int argc, char** argv)
{
  psc_init(argc, argv);

  if (const char* restart = std::getenv("PSC_RESTART")) {
    read_checkpoint_filename = restart;
  }

  run();

  psc_finalize();
  return 0;
}
