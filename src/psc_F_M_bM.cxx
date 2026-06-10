// psc_F_M_bM — Firehose Moderate Bi-Maxwellian
// βi∥=6.0  Ai=0.3  βe∥=1.0  Ae=1.0 | mr=200, 2000 ppc, 1408²

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>
#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"
#include "../libpsc/psc_heating/psc_heating_impl.hxx"

enum { MY_ELECTRON, MY_ION, N_MY_KINDS };

struct PscFlatfoilParams {
  double BB, Zi, mass_ratio, lambda0;
  double vA_over_c, beta_e_par, beta_i_par;
  double Ti_perp_over_Ti_par, Te_perp_over_Te_par, n;
  double B0, Te_par, Te_perp, Ti_par, Ti_perp, mi, me, d_i;
};

namespace { PscFlatfoilParams g; std::string read_checkpoint_filename; PscParams psc_params; }

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

void setupParameters() {
  psc_params.nmax = 1650000; psc_params.cfl = 0.95;
  psc_params.write_checkpoint_every_step = 0; psc_params.stats_every = 50;
  g.BB=1.0; g.Zi=1.; g.mass_ratio=200.; g.lambda0=20.;
  g.vA_over_c=0.05; g.beta_e_par=1.0; g.beta_i_par=6.0;
  g.Ti_perp_over_Ti_par=0.3; g.Te_perp_over_Te_par=1.0; g.n=1.0;
  g.mi=g.mass_ratio*g.Zi; g.me=1.0; g.B0=g.vA_over_c;
  g.Te_par=g.beta_e_par*sqr(g.B0)/2.; g.Te_perp=g.Te_perp_over_Te_par*g.Te_par;
  g.Ti_par=g.beta_i_par*sqr(g.B0)/2.; g.Ti_perp=g.Ti_perp_over_Ti_par*g.Ti_par;
}

Grid_t* setupGrid() {
  g.d_i = std::sqrt(g.mass_ratio / g.n);
  double domain_size = 20.0 * g.d_i;
  Grid_t::Real3 LL = {1.0, domain_size, domain_size};
  Int3 gdims = {1, 1408, 1408}; Int3 np = {1, 8, 8};
  Grid_t::Domain domain{gdims, LL, -.5 * LL, np};
  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};
  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De = %g\n", sqrt(g.Te_par));
  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = 2000;
  double dt = psc_params.cfl * courant_length(domain);
  Grid_t::Normalization norm{norm_params};
  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) ibn[0] = 0;
  if (Dim::InvarY::value) ibn[1] = 0;
  if (Dim::InvarZ::value) ibn[2] = 0;
  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts) {
  partitionAndSetupParticles(setup_particles, balance, grid_ptr, mprts,
    [&](int kind, Double3 crd, psc_particle_npt& npt) {
      switch (kind) {
        case MY_ION: npt.n=g.n; npt.T[0]=g.Ti_perp; npt.T[1]=g.Ti_perp; npt.T[2]=g.Ti_par; break;
        case MY_ELECTRON: npt.n=g.n; npt.T[0]=g.Te_perp; npt.T[1]=g.Te_perp; npt.T[2]=g.Te_par; break;
        default: assert(0);
      }
    });
}

void initializeFields(MfieldsState& mflds) {
  setupFields(mflds, [&](int m, double crd[3]) {
    switch (m) { case HZ: return g.B0; default: return 0.; }
  });
}

void run() {
  mpi_printf(MPI_COMM_WORLD, "*** Setting up F-M-bM (Firehose Moderate Bi-Maxwellian) ...\n");
  setupParameters();
  auto grid_ptr = setupGrid(); auto& grid = *grid_ptr;
  Mparticles mprts(grid); MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty()) read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);
  psc_params.balance_interval = 700; Balance balance{3};
  psc_params.sort_interval = 10;
  int collision_interval = -10;
  double collision_nu = 3.76 * std::pow(g.Te_perp, 2.) / g.Zi / g.lambda0;
  Collision collision{grid, collision_interval, collision_nu};
  ChecksParams checks_params{};
  checks_params.continuity.check_interval = 0; checks_params.continuity.err_threshold = 1e-4;
  checks_params.continuity.print_max_err_always = true; checks_params.continuity.dump_always = false;
  checks_params.gauss.check_interval = 100; checks_params.gauss.err_threshold = 1e-4;
  checks_params.gauss.print_max_err_always = true; checks_params.gauss.dump_always = false;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};
  double marder_diffusion = 0.9; int marder_loop = 3; bool marder_dump = false;
  psc_params.marder_interval = 100;
  Marder marder(grid, marder_diffusion, marder_loop, marder_dump);
  OutputFieldsItemParams outf_item_params{}; OutputFieldsParams outf_params{};
  outf_item_params.pfield.out_interval = 690; outf_item_params.tfield.out_interval = 690;
  outf_item_params.tfield.average_every = 138;
  outf_params.fields = outf_item_params; outf_params.moments = outf_item_params;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, outf_params};
  OutputParticlesParams outp_params{}; outp_params.every_step = 400;
  outp_params.data_dir = "."; outp_params.basename = "prt_F_M_bM";
  outp_params.lo = {0, int(0.3*1408), int(0.3*1408)};
  outp_params.hi = {1, int(0.7*1408), int(0.7*1408)};
  OutputParticles outp{grid, outp_params};
  int oute_interval = -100; DiagEnergies oute{grid.comm(), oute_interval};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);
  SetupParticles<Mparticles> setup_particles(grid);
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;
  if (read_checkpoint_filename.empty()) { initializeParticles(setup_particles, balance, grid_ptr, mprts); initializeFields(mflds); }
  auto psc = makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts, balance, collision, checks, marder, diagnostics);
  psc.integrate();
}

int main(int argc, char** argv) { psc_init(argc, argv); run(); psc_finalize(); return 0; }
