// ======================================================================
// psc_test_pressure_moment - diagnostic case, NOT part of the thesis matrix
//
// Answers two questions about what PSC writes as txx..tzx in `all_1st`:
//
//   (1) pressure or temperature?  Run with PSC_MOMENT_TEST=1: the density is
//       modulated along z, n(z) = n0 (1 + A sin(2 pi z / Lz)), while every
//       species is initialised with the SAME uniform isotropic temperature T0
//       and no drift.  A temperature output is flat in z; a pressure output
//       follows n(z), because P = n k T.
//
//   (2) raw second moment or central (thermal) one?  Run with
//       PSC_MOMENT_TEST=2: the density and the temperature are uniform, but a
//       bulk drift u_y(z) = u0 sin(2 pi z / Lz) is imposed on both species
//       (same velocity for both, so the net current stays zero).  A raw second
//       moment picks up n m u_y^2 and is modulated; a central moment is flat.
//
// Both tests only need the t=0 output; a few steps are run so that the answer
// can be checked for drift in time as well.
//
// Expected values with the defaults below (ion species, test 1):
//   tzz_i(z) = n(z) * T0,  txx_i(z) = tyy_i(z) = n(z) * T0
// and for test 2:
//   tyy_i(z) = n0 * (T0 + mi * u0^2 sin^2(2 pi z / Lz))
//   tyy_i - py_i^2/(n mi) = n0 * T0   (flat)
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include <cstdlib>
#include <string>

// ======================================================================
// Particle kinds

enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS,
};

// ======================================================================
// Parameters

struct PscMomentTestParams
{
  double mass_ratio;
  double T0;         // uniform temperature, both species, all directions
  double n0;         // mean density
  double n_ampl;     // relative amplitude of the density modulation
  double u_drift;    // amplitude of the bulk drift (test 2)
  double B0;         // background field along x (out of plane)
  double Lz;         // box length along z
  int test;          // 1 = density gradient, 2 = sheared bulk drift
};

namespace
{

PscMomentTestParams g;
PscParams psc_params;

int ngrid_z = 64;
int ngrid_y = 8;
int nicell = 500;
int nmax = 4;
int fields_every = 2;

} // namespace

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

// ======================================================================
// setupParameters

void setupParameters()
{
  g.test = envInt("PSC_MOMENT_TEST", 1);
  g.mass_ratio = envDouble("PSC_MASS_RATIO", 100.);
  g.T0 = envDouble("PSC_T0", 0.01);
  g.n0 = envDouble("PSC_N0", 1.0);
  g.n_ampl = envDouble("PSC_N_AMPL", 0.5);
  g.u_drift = envDouble("PSC_U_DRIFT", 0.02);
  g.B0 = envDouble("PSC_B0", 0.0);
  g.Lz = envDouble("PSC_LZ", 5.0);

  ngrid_z = envInt("PSC_NGRID_Z", ngrid_z);
  ngrid_y = envInt("PSC_NGRID_Y", ngrid_y);
  nicell = envInt("PSC_NICELL", nicell);
  nmax = envInt("PSC_NMAX", nmax);
  fields_every = envInt("PSC_FIELDS_EVERY", fields_every);

  // test 1 modulates the density only, test 2 the drift only
  if (g.test == 2) {
    g.n_ampl = 0.0;
  } else {
    g.u_drift = 0.0;
  }

  psc_params.nmax = nmax;
  psc_params.cfl = 0.95;
  psc_params.write_checkpoint_every_step = -1;
  psc_params.stats_every = 1;
}

// ======================================================================
// density / drift profiles

static inline double profile(double z)
{
  return std::sin(2. * M_PI * z / g.Lz);
}

static inline double density(double z)
{
  return g.n0 * (1. + g.n_ampl * profile(z));
}

static inline double drift(double z)
{
  return g.u_drift * profile(z);
}

// ======================================================================
// setupGrid

Grid_t* setupGrid()
{
  double dz = g.Lz / ngrid_z;
  double Ly = dz * ngrid_y; // square cells

  Grid_t::Real3 LL = {1.0, Ly, g.Lz};
  Int3 gdims = {1, ngrid_y, ngrid_z};
  Int3 np = {1, 1, 1};

  Grid_t::Domain domain{gdims, LL, {0., 0., 0.}, np};

  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {1., g.mass_ratio, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  mpi_printf(MPI_COMM_WORLD, "test = %d (%s)\n", g.test,
             g.test == 2 ? "sheared bulk drift, uniform n and T"
                         : "density gradient, uniform T");
  mpi_printf(MPI_COMM_WORLD,
             "grid = 1 x %d x %d, Lz = %g, dz = %g, nicell = %d, nmax = %d\n",
             ngrid_y, ngrid_z, g.Lz, dz, nicell, nmax);
  mpi_printf(MPI_COMM_WORLD,
             "T0 = %g, n0 = %g, n_ampl = %g, u_drift = %g, mi/me = %g\n", g.T0,
             g.n0, g.n_ampl, g.u_drift, g.mass_ratio);
  mpi_printf(MPI_COMM_WORLD, "lambda_De = %g (dz/lambda_De = %g)\n",
             std::sqrt(g.T0), dz / std::sqrt(g.T0));
  mpi_printf(MPI_COMM_WORLD, "predicted n*T0 range = [%g, %g]\n",
             g.n0 * (1. - g.n_ampl) * g.T0, g.n0 * (1. + g.n_ampl) * g.T0);

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = nicell;

  double dt = psc_params.cfl * courant_length(domain);
  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {0, 2, 2};

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ======================================================================
// initializeParticles

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts)
{
  partitionAndSetupParticles(setup_particles, balance, grid_ptr, mprts,
                             [&](int kind, Double3 crd, psc_particle_npt& npt) {
                               double z = crd[2];
                               npt.n = density(z);
                               npt.T[0] = g.T0;
                               npt.T[1] = g.T0;
                               npt.T[2] = g.T0;
                               npt.p[0] = 0.;
                               npt.p[1] = drift(z);
                               npt.p[2] = 0.;
                             });
}

// ======================================================================
// initializeFields

void initializeFields(MfieldsState& mflds)
{
  setupFields(mflds, [&](int m, double crd[3]) {
    switch (m) {
      case HX: return g.B0;
      default: return 0.;
    }
  });
}

// ======================================================================
// run

void run()
{
  mpi_printf(MPI_COMM_WORLD, "*** Setting up pressure-moment test ...\n");

  setupParameters();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);

  psc_params.balance_interval = -1;
  Balance balance{3};

  psc_params.sort_interval = -1;

  Collision collision{grid, -10, 0.};

  ChecksParams checks_params{};
  checks_params.continuity.check_interval = -1;
  checks_params.gauss.check_interval = -1;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  psc_params.marder_interval = -1;
  Marder marder(grid, 0.9, 3, false);

  OutputFieldsItemParams outf_item_params{};
  OutputFieldsParams outf_params{};
  outf_item_params.pfield.out_interval = fields_every;
  outf_params.fields = outf_item_params;
  outf_params.moments = outf_item_params;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, outf_params};

  OutputParticlesParams outp_params{};
  outp_params.every_step = -1;
  outp_params.data_dir = ".";
  outp_params.basename = "prt_test_pressure_moment";
  OutputParticles outp{grid, outp_params};

  DiagEnergies oute{grid.comm(), -1};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_particles(grid);
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;

  initializeParticles(setup_particles, balance, grid_ptr, mprts);
  initializeFields(mflds);

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
  run();
  psc_finalize();
  return 0;
}
