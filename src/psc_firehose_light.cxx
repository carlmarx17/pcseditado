// ======================================================================
// psc_firehose_light.cxx
//
// Firehose instability via bi-Maxwellian temperature anisotropy.
// Lightweight variant: designed to run on a local PC with low RAM.
//
// Physics:
//   B along Z (HZ), variations in Y-Z plane.
//   Trigger condition: Ti_perp/Ti_par < 1 - 2/beta_i_par  (firehose)
//   Here: Ti_perp/Ti_par = 0.1, beta_i_par = 10 => threshold = 0.8
//   => 0.1 << 0.8, so strongly driven firehose.
//
// Key low-RAM choices:
//   mass_ratio = 100  (me/mi = 1/100)
//   ppc        = 500
//   grid       = 64x64
//   nmax       = 20000
//   np patches = 2x2
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>
#include <mrc_params.h>
#include <cstdlib>
#include <cmath>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"
#include "writer_mrc.hxx"
#include <libpsc/psc_heating/psc_heating_impl.hxx>
#include "heating_spot_foil.hxx"

// ----------------------------------------------------------------------
// 1. Particle species
// ----------------------------------------------------------------------
enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS
};

// ----------------------------------------------------------------------
// 2. Physical parameters
// ----------------------------------------------------------------------
struct FirehoseLightParams
{
  double BB, Zi, mass_ratio, lambda0;
  double vA_over_c, beta_e_par, beta_i_par;
  double Ti_perp_over_Ti_par, Te_perp_over_Te_par, n;

  // Derived
  double B0, Te_par, Te_perp, Ti_par, Ti_perp, mi, me, d_i, d_e, lambda_De;
};

static FirehoseLightParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

namespace
{
// -----------------------------------------------------------------------
// Low-RAM parameters
// -----------------------------------------------------------------------
constexpr double kDomainSizeDi = 16.0;
constexpr int    kGridSizeYZ   = 64;
constexpr int    kPPC          = 500;
constexpr int    kFieldOutStep = 200;
constexpr int    kPrtOutStep   = 2000;

int envOrDefault(const char* name, int default_value)
{
  if (const char* v = std::getenv(name)) return std::atoi(v);
  return default_value;
}

bool hasEnv(const char* name) { return std::getenv(name) != nullptr; }
} // namespace

// ----------------------------------------------------------------------
// 3. Dimensionality: Y-Z plane, B along Z
// ----------------------------------------------------------------------
using Dim = dim_yz;
using PscConfig = PscConfig1vbecSingle<Dim>;

#include <kg/io.h>
#include "writer_adios2.hxx"

using Writer       = WriterADIOS2;
using MfieldsState = PscConfig::MfieldsState;
using Mparticles   = PscConfig::Mparticles;
using Balance      = PscConfig::Balance;
using Collision    = PscConfig::Collision;
using Checks       = PscConfig::Checks;
using Marder       = PscConfig::Marder;

// ADIOS2 particle output
struct OutputParticlesAdios2 {
  OutputParticlesAdios2(const Grid_t& grid, const OutputParticlesParams& params)
    : params_(params), comm_(grid.comm()) {}

  template <typename Mp>
  void operator()(Mp& mprts) {
    if (params_.every_step <= 0 || mprts.grid().timestep() % params_.every_step != 0) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s.%09d.bp",
             params_.data_dir, params_.basename, mprts.grid().timestep());
    auto io     = kg::io::IOAdios2{};
    auto writer = io.open(filename, kg::io::Mode::Write, comm_);
    writer.put("mprts", mprts);
    writer.close();
  }

  OutputParticlesParams params_;
  MPI_Comm              comm_;
};

using OutputParticles = OutputParticlesAdios2;
using Moment_n        = Moment_n_1st<Mparticles, MfieldsC>;

auto make_MfieldsMoment_n(const Grid_t& grid)
{
  return MfieldsC(grid, grid.kinds.size(), grid.ibn);
}

// ----------------------------------------------------------------------
// 4. Physical parameter setup — FIREHOSE conditions
// ----------------------------------------------------------------------
void setupParameters()
{
  // 20000 steps => enough to see firehose growth and saturation.
  // Firehose grows faster than mirror (shorter growth timescale at high beta).
  psc_params.nmax                    = envOrDefault("PSC_NMAX", 20000);
  psc_params.cfl                     = 0.95;
  psc_params.write_checkpoint_every_step = 5000;
  psc_params.stats_every             = 50;

  // ----------------------------------------------------------------------
  // FIREHOSE instability conditions:
  //   Threshold: Ti_perp/Ti_par < 1 - 2/beta_i_par
  //   With beta_i_par = 10: threshold = 1 - 0.2 = 0.8
  //   We use Ti_perp/Ti_par = 0.1 => strongly driven (factor ~8 below threshold)
  //
  //   Physical interpretation: parallel pressure greatly exceeds perpendicular
  //   => field-aligned Alfven wave becomes unstable (firehose).
  // ----------------------------------------------------------------------
  g.BB                   = 1.0;
  g.Zi                   = 1.0;
  g.mass_ratio           = 100.0; // me/mi = 1/100
  g.lambda0              = 20.0;

  g.vA_over_c            = 0.05;  // vA/c
  g.beta_e_par           = 1.0;
  g.beta_i_par           = 10.0;  // high parallel beta (strong firehose)
  g.Ti_perp_over_Ti_par  = 0.1;   // Ti_perp << Ti_par => firehose trigger
  g.Te_perp_over_Te_par  = 1.0;   // electrons isotropic
  g.n                    = 1.0;

  // Derived masses
  g.mi = g.mass_ratio * g.Zi;
  g.me = 1.0;

  // Magnetic field
  double rho = g.n * g.mi + g.n * g.me;
  double vA2 = g.vA_over_c * g.vA_over_c;
  g.B0       = std::sqrt(rho * vA2 / (1.0 - vA2));

  // Temperatures
  g.Te_par   = g.beta_e_par  * g.B0 * g.B0 / (2.0 * g.n);
  g.Te_perp  = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par   = g.beta_i_par  * g.B0 * g.B0 / (2.0 * g.n);
  g.Ti_perp  = g.Ti_perp_over_Ti_par * g.Ti_par;
}

// ----------------------------------------------------------------------
// 5. Grid setup — at least 1 cell per Debye length
// ----------------------------------------------------------------------
Grid_t* setupGrid()
{
  g.d_i       = std::sqrt(g.mass_ratio / g.n);
  g.d_e       = 1.0 / std::sqrt(g.n);
  // Use Te_par for Debye length since electrons are isotropic here
  g.lambda_De = std::sqrt(g.Te_par / g.n);

  const double domain_size = kDomainSizeDi * g.d_i;

  double dx_max = std::min(g.lambda_De, g.d_e);
  int req_cells = static_cast<int>(std::ceil(domain_size / dx_max));
  int grid_yz   = hasEnv("PSC_GRID_YZ")
                    ? envOrDefault("PSC_GRID_YZ", kGridSizeYZ)
                    : std::max(kGridSizeYZ, req_cells);
  int ppc       = envOrDefault("PSC_NICELL", kPPC);

  mpi_printf(MPI_COMM_WORLD,
             "[firehose_light] d_e=%.3g  d_i=%.3g  lambda_De=%.3g\n",
             g.d_e, g.d_i, g.lambda_De);
  mpi_printf(MPI_COMM_WORLD,
             "[firehose_light] domain=%.2f d_i  grid=%dx%d  ppc=%d\n",
             kDomainSizeDi, grid_yz, grid_yz, ppc);
  mpi_printf(MPI_COMM_WORLD,
             "[firehose_light] FIREHOSE: Ti_perp/Ti_par=%.2f  beta_i_par=%.1f"
             "  threshold=%.3f\n",
             g.Ti_perp_over_Ti_par, g.beta_i_par,
             1.0 - 2.0 / g.beta_i_par);

  Grid_t::Real3 LL{1., domain_size, domain_size};
  Int3 gd{1, grid_yz, grid_yz};
  Int3 np{1, 2, 2}; // 4 patches

  Grid_t::Domain dom{gd, LL, -.5 * LL, np};

  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION]      = {g.Zi,  g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1.,   1., "e"};

  auto npn    = Grid_t::NormalizationParams::dimensionless();
  npn.nicell  = ppc;

  double dt   = psc_params.cfl * courant_length(dom);
  Grid_t::Normalization norm{npn};

  Int3 ibn{2, 2, 2};
  if (Dim::InvarX::value) ibn[0] = 0;
  if (Dim::InvarY::value) ibn[1] = 0;
  if (Dim::InvarZ::value) ibn[2] = 0;

  return new Grid_t{dom, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// 6. Particle initialization — bi-Maxwellian (firehose: Ti_par >> Ti_perp)
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts)
{
  partitionAndSetupParticles(
    setup_p, bal, gptr, mprts,
    [&](int kind, Double3 /*pos*/, int /*patch*/, Int3 /*idx*/,
        psc_particle_np& np) {
      psc_particle_npt npt{};
      npt.kind = kind;
      if (kind == MY_ION) {
        npt.n    = g.n;
        npt.T[0] = g.Ti_perp; // x (perp to B)
        npt.T[1] = g.Ti_perp; // y (perp to B)
        npt.T[2] = g.Ti_par;  // z (par to B, Ti_par >> Ti_perp)
      } else {
        npt.n    = g.n;
        npt.T[0] = g.Te_perp;
        npt.T[1] = g.Te_perp;
        npt.T[2] = g.Te_par;
      }
      np.n = npt.n;
      np.p = setup_p.createMaxwellian(npt);
    });
}

// ----------------------------------------------------------------------
// 7. Field initialization — B along Z
// ----------------------------------------------------------------------
static constexpr int B_DIR = HZ;

void initializeFields(MfieldsState& mflds)
{
  setupFields(mflds, [&](int m, double[3]) {
    return m == B_DIR ? g.B0 : 0.;
  });
}

// ----------------------------------------------------------------------
// 8. Run
// ----------------------------------------------------------------------
void run()
{
  mpi_printf(MPI_COMM_WORLD,
             "*** Firehose Instability (Maxwellian, light PC run) ***\n");
  setupParameters();
  auto grid_ptr = setupGrid();
  auto& grid    = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  psc_params.balance_interval = 200;
  Balance bal{3};
  psc_params.sort_interval = 10;

  double nu  = 3.76 * std::pow(g.Te_par, 2.) / g.Zi / g.lambda0;
  Collision coll{grid, -10, nu};

  ChecksParams chkp{};
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 50;
  Marder marder(grid, 0.9, 3, false);

  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval  = kFieldOutStep;
  ofip.tfield.out_interval  = kFieldOutStep;
  ofip.tfield.average_every = kFieldOutStep / 4;

  OutputFieldsParams ofp{};
  ofp.fields  = ofip;
  ofp.moments = ofip;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  int window = std::min(32, grid.domain.gdims[1] / 2);
  OutputParticlesParams opp{};
  opp.every_step = kPrtOutStep;
  opp.data_dir   = ".";
  opp.basename   = "prt_firehose";
  opp.lo = {0, (grid.domain.gdims[1] - window) / 2,
               (grid.domain.gdims[2] - window) / 2};
  opp.hi = {0, (grid.domain.gdims[1] + window) / 2,
               (grid.domain.gdims[2] + window) / 2};
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), -100};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid);
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population         = MY_ION;

  auto mf_n = make_MfieldsMoment_n(grid);

  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_p, bal, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto integrator =
    makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts,
                                 bal, coll, checks, marder, diagnostics);
  integrator.integrate();
}

// ----------------------------------------------------------------------
// 9. main
// ----------------------------------------------------------------------
int main(int argc, char** argv)
{
  psc_init(argc, argv);

  const char* ckpt = nullptr;
  mrc_params_get_option_string("read_checkpoint", &ckpt);
  if (ckpt) read_checkpoint_filename = ckpt;

  run();
  psc_finalize();
  return 0;
}
