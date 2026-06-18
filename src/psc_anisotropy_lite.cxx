#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include "../libpsc/psc_heating/psc_heating_impl.hxx"

#include <cmath>
#include <cstdlib>
#include <string>

#ifndef PSC_LITE_CASE
#error "PSC_LITE_CASE must be defined"
#endif

enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS,
};

struct LiteParams
{
  double Zi = 1.;
  double mass_ratio = 200.;
  double vA_over_c = .05;
  double beta_e_par;
  double beta_i_par;
  double Ti_perp_over_Ti_par;
  double Te_perp_over_Te_par;
  double n = 1.;
  double B0;
  double Te_par;
  double Te_perp;
  double Ti_par;
  double Ti_perp;
  double d_i;
  double domain_di;
  double seed = 1.e-2;
  int nmax;
  int ngrid = 512;
  int nicell = 64;
  int checkpoint_every = 50000;
  int fields_every = 2000;
  int particles_every = 30000;
  const char* basename;
};

namespace
{
LiteParams g;
std::string read_checkpoint_filename;
PscParams psc_params;
} // namespace

int envInt(const char* name, int fallback)
{
  if (const char* value = std::getenv(name)) {
    return std::stoi(value);
  }
  return fallback;
}

double envDouble(const char* name, double fallback)
{
  if (const char* value = std::getenv(name)) {
    return std::stod(value);
  }
  return fallback;
}

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

void setupParameters()
{
  psc_params.cfl = .95;
  psc_params.stats_every = 500;
#if PSC_LITE_CASE == 1
  g.beta_i_par = 5.;
  g.Ti_perp_over_Ti_par = 3.;
  g.beta_e_par = 1.;
  g.Te_perp_over_Te_par = 1.;
  g.domain_di = 20.;
  g.nmax = 180000;
  g.basename = "prt_M_lite";
#elif PSC_LITE_CASE == 2
  g.beta_i_par = 10.;
  g.Ti_perp_over_Ti_par = .1;
  g.beta_e_par = 1.;
  g.Te_perp_over_Te_par = 1.;
  g.domain_di = 20.;
  g.nmax = 180000;
  g.basename = "prt_F_lite";
#elif PSC_LITE_CASE == 3
  g.beta_i_par = 1.;
  g.Ti_perp_over_Ti_par = 1.;
  g.beta_e_par = .5;
  g.Te_perp_over_Te_par = 3.;
  g.domain_di = 4.;
  g.nmax = 40000;
  g.checkpoint_every = 10000;
  g.fields_every = 500;
  g.particles_every = 10000;
  g.basename = "prt_W_lite";
#else
#error "Unknown PSC_LITE_CASE"
#endif

  g.nmax = envInt("PSC_NMAX", g.nmax);
  g.ngrid = envInt("PSC_NGRID", g.ngrid);
  g.nicell = envInt("PSC_NICELL", g.nicell);
  g.checkpoint_every =
    envInt("PSC_CHECKPOINT_EVERY", g.checkpoint_every);
  g.fields_every = envInt("PSC_FIELDS_EVERY", g.fields_every);
  g.particles_every = envInt("PSC_PARTICLES_EVERY", g.particles_every);
  g.seed = envDouble("PSC_SEED", g.seed);

  g.B0 = g.vA_over_c;
  g.Te_par = g.beta_e_par * sqr(g.B0) / 2.;
  g.Te_perp = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par = g.beta_i_par * sqr(g.B0) / 2.;
  g.Ti_perp = g.Ti_perp_over_Ti_par * g.Ti_par;
  g.d_i = std::sqrt(g.mass_ratio / g.n);
  psc_params.nmax = g.nmax;
  psc_params.write_checkpoint_every_step = g.checkpoint_every;
}

Grid_t* setupGrid()
{
  double domain_size = g.domain_di * g.d_i;
  Grid_t::Real3 LL = {1., domain_size, domain_size};
  Int3 gdims = {1, g.ngrid, g.ngrid};
  Int3 np = {1, 8, 8};
  Grid_t::Domain domain{gdims, LL, -.5 * LL, np};

  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = g.nicell;
  Grid_t::Normalization norm{norm_params};
  double dt = psc_params.cfl * courant_length(domain);

  Int3 ibn = {0, 2, 2};
  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts)
{
  partitionAndSetupParticles(
    setup_particles, balance, grid_ptr, mprts,
    [&](int kind, Double3, psc_particle_npt& npt) {
      npt.n = g.n;
      if (kind == MY_ION) {
        npt.T[0] = g.Ti_perp;
        npt.T[1] = g.Ti_perp;
        npt.T[2] = g.Ti_par;
      } else {
        npt.T[0] = g.Te_perp;
        npt.T[1] = g.Te_perp;
        npt.T[2] = g.Te_par;
      }
    });
}

void initializeFields(MfieldsState& mflds)
{
  const auto& grid = mflds.grid();
  const double Ly = grid.domain.length[1];
  const double Lz = grid.domain.length[2];

  setupFields(mflds, [&](int m, double crd[3]) {
    const double y = crd[1];
    const double z = crd[2];
#if PSC_LITE_CASE == 1
    // Oblique, divergence-free compressive seed for Mirror.
    const double ky = 4. * M_PI / Ly;
    const double kz = 2. * M_PI / Lz;
    const double kn = std::sqrt(ky * ky + kz * kz);
    if (m == HY) {
      return g.seed * g.B0 * (kz / kn) * std::cos(ky * y) * std::sin(kz * z);
    }
    if (m == HZ) {
      return g.B0 -
             g.seed * g.B0 * (ky / kn) * std::sin(ky * y) * std::cos(kz * z);
    }
#elif PSC_LITE_CASE == 2
    // Parallel transverse seed for Firehose.
    const double kz = 4. * M_PI / Lz;
    if (m == HX) {
      return g.seed * g.B0 * std::cos(kz * z);
    }
    if (m == HZ) {
      return g.B0;
    }
#elif PSC_LITE_CASE == 3
    // Circularly polarized parallel seed for electron Whistler.
    const double kz = 8. * M_PI / Lz;
    if (m == HX) {
      return g.seed * g.B0 * std::cos(kz * z);
    }
    if (m == HY) {
      return g.seed * g.B0 * std::sin(kz * z);
    }
    if (m == HZ) {
      return g.B0;
    }
#endif
    return 0.;
  });
}

void run()
{
  setupParameters();
  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;
  Mparticles mprts(grid);
  MfieldsState mflds(grid);

  if (!read_checkpoint_filename.empty()) {
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);
  }

  psc_params.balance_interval = 2000;
  psc_params.balance_mem_fraction = .9;
  Balance balance{3};
  psc_params.sort_interval = 20;

  int collision_interval = -1;
  Collision collision{grid, collision_interval, 1.e-6};

  ChecksParams checks_params{};
  checks_params.continuity.check_interval = 0;
  checks_params.gauss.check_interval = 1000;
  checks_params.gauss.err_threshold = 1.e-3;
  checks_params.gauss.print_max_err_always = false;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  psc_params.marder_interval = 200;
  Marder marder(grid, .9, 3, false);

  OutputFieldsItemParams outf_item_params{};
  OutputFieldsParams outf_params{};
  outf_item_params.pfield.out_interval = g.fields_every;
  outf_params.fields = outf_item_params;
  outf_params.moments = outf_item_params;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, outf_params};

  OutputParticlesParams outp_params{};
  outp_params.every_step = g.particles_every;
  outp_params.data_dir = ".";
  outp_params.basename = g.basename;
  outp_params.lo = {0, int(.475 * g.ngrid), int(.475 * g.ngrid)};
  outp_params.hi = {1, int(.525 * g.ngrid), int(.525 * g.ngrid)};
  OutputParticles outp{grid, outp_params};

  DiagEnergies oute{grid.comm(), -500};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_particles(grid);
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;
  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_particles, balance, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto psc = makePscIntegrator<PscConfig>(
    psc_params, *grid_ptr, mflds, mprts, balance, collision, checks, marder,
    diagnostics);
  psc.integrate();
}

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
