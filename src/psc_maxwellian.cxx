// ======================================================================
// psc_maxwellian.cxx
// Simulation of ion temperature-anisotropy-driven instabilities.
// Maxwellian case (bi-Maxwellian) variant of psc_temp_aniso.cxx
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"
#include "writer_mrc.hxx"
#include <libpsc/psc_heating/psc_heating_impl.hxx>
#include "heating_spot_foil.hxx"

// ----------------------------------------------------------------------
// 1. Particle kinds
// Define the species involved in the simulation.
// ----------------------------------------------------------------------
enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS
};

// ----------------------------------------------------------------------
// 2. Simulation parameters
// Global physical parameters for the mirror instability setup.
// ----------------------------------------------------------------------
struct PscFlatfoilParams
{
  double BB, Zi, mass_ratio, lambda0;
  double vA_over_c, beta_e_par, beta_i_par;
  double Ti_perp_over_Ti_par, Te_perp_over_Te_par, n;

  // Derived quantities
  double B0, Te_par, Te_perp, Ti_par, Ti_perp, mi, me, d_i;
};

static PscFlatfoilParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

namespace
{

constexpr double kResolvedDomainDe = 64.;
constexpr int kResolvedGridYZ = 512;
constexpr int kParticlesPerCell = 1000;
constexpr int kParticleOutputStride = 500;
constexpr int kParticleWindowCells = 64;

} // namespace

// ----------------------------------------------------------------------
// 3. Compile-time configuration
// ENFORCING 2D Y-Z PLANE: The instability requires variations along the
// magnetic field (Z-axis). Running in X-Y with B in Z suppresses the mode.
// ----------------------------------------------------------------------
using Dim = dim_yz;

#ifdef USE_CUDA
using PscConfig = PscConfig1vbecCuda<Dim>;
#else
using PscConfig = PscConfig1vbecSingle<Dim>;
#endif

#include <kg/io.h>
#include "writer_adios2.hxx"

using Writer = WriterADIOS2;
using MfieldsState = PscConfig::MfieldsState;
using Mparticles = PscConfig::Mparticles;
using Balance = PscConfig::Balance;
using Collision = PscConfig::Collision;
using Checks = PscConfig::Checks;
using Marder = PscConfig::Marder;

struct OutputParticlesAdios2 {
  OutputParticlesAdios2(const Grid_t& grid, const OutputParticlesParams& params)
    : params_(params), comm_(grid.comm()) {}

  template <typename Mp>
  void operator()(Mp& mprts) {
    if (params_.every_step <= 0 || mprts.grid().timestep() % params_.every_step != 0) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s.%09d.bp", params_.data_dir, params_.basename, mprts.grid().timestep());
    
    auto io = kg::io::IOAdios2{};
    auto writer = io.open(filename, kg::io::Mode::Write, comm_);
    writer.put("mprts", mprts);
    writer.close();
  }

  OutputParticlesParams params_;
  MPI_Comm comm_;
};

using OutputParticles = OutputParticlesAdios2;

// ----------------------------------------------------------------------
// 4. Moment selector setup
// ----------------------------------------------------------------------
template <typename Mp, typename Dm, typename Enable = void>
struct Moment_n_Selector
{
  using type = Moment_n_1st<Mp, MfieldsC>;
};

#ifdef USE_CUDA
template <typename Mp, typename Dm>
struct Moment_n_Selector<Mp, Dm,
                         typename std::enable_if<Mp::is_cuda::value>::type>
{
  using type = Moment_n_1st_cuda<Mp, Dm>;
};
#endif
using Moment_n = typename Moment_n_Selector<Mparticles, Dim>::type;

template <typename Moment>
auto make_MfieldsMoment_n(const Grid_t& grid)
{
  return MfieldsC(grid, grid.kinds.size(), grid.ibn);
}

#ifdef USE_CUDA
template <>
auto make_MfieldsMoment_n<Moment_n>(const Grid_t& grid)
{
  return HMFields({-grid.ibn, grid.domain.ldims + 2 * grid.ibn},
                  grid.kinds.size(), grid.n_patches());
}
#endif

// ----------------------------------------------------------------------
// 5. Parameter setup
// ----------------------------------------------------------------------
void setupParameters()
{
  psc_params.nmax = 80001;
  psc_params.cfl = 0.95;
  psc_params.write_checkpoint_every_step = 1000;
  psc_params.stats_every = 1;

  g.BB = 1.0;
  g.Zi = 1.0;
  g.mass_ratio = 64.0; // Artificial mass ratio to speed up simulation
  g.lambda0 = 20.0;

  // Campaign 1 from the documentation: vary kappa at fixed beta_i|| = 5, A = 3.
  g.vA_over_c = 0.1;
  g.beta_e_par = 1.0;
  g.beta_i_par = 5.0;
  g.Ti_perp_over_Ti_par = 3.0;
  g.Te_perp_over_Te_par = 1.0;
  g.n = 1.0;

  // Calculate derived thermal velocities and pressures with exact physics
  g.mi = g.mass_ratio * g.Zi;
  g.me = 1.0;
  
  double rho = g.n * g.mi + g.n * g.me;
  double vA2 = sqr(g.vA_over_c);
  g.B0 = std::sqrt(rho * vA2 / (1.0 - vA2));

  // Thermodynamic definitions dependent on exact density
  g.Te_par = g.beta_e_par * sqr(g.B0) / (2.0 * g.n);
  g.Te_perp = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par = g.beta_i_par * sqr(g.B0) / (2.0 * g.n);
  g.Ti_perp = g.Ti_perp_over_Ti_par * g.Ti_par;
}

// ----------------------------------------------------------------------
// 6. Grid setup
// ----------------------------------------------------------------------
Grid_t* setupGrid()
{
  // Enforce Debye resolution dynamically (> 3 points per Debye length)
  double lambda_De = std::sqrt(g.Te_perp);
  double max_dy = lambda_De / 3.0; // Enforce at least 3 cells per lambda_De
  int requiredGridYZ = std::ceil(kResolvedDomainDe / max_dy);
  int finalGridYZ = std::max(kResolvedGridYZ, requiredGridYZ);

  Grid_t::Real3 LL{1., kResolvedDomainDe, kResolvedDomainDe};
  Int3 gd{1, finalGridYZ, finalGridYZ};
  Int3 np{1, 4, 8};

  Grid_t::Domain dom{gd, LL, -.5 * LL, np};
  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  // Rigorous deep ion length dependent on density and charge squared:
  g.d_i = std::sqrt(kinds[MY_ION].m / (g.n * sqr(kinds[MY_ION].q)));

  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De = %g\n", sqrt(g.Te_perp));

  auto npn = Grid_t::NormalizationParams::dimensionless();
  npn.nicell = kParticlesPerCell;

  double dt = psc_params.cfl * courant_length(dom);
  Grid_t::Normalization norm{npn};

  Int3 ibn{2, 2, 2};
  if (Dim::InvarX::value)
    ibn[0] = 0;
  if (Dim::InvarY::value)
    ibn[1] = 0;
  if (Dim::InvarZ::value)
    ibn[2] = 0;

  return new Grid_t{dom, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// 7. Particle initialization
// Loads particles with the specified bi-Maxwellian velocity distributions.
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p, Balance& bal,
                         Grid_t*& gptr, Mparticles& mprts)
{
  partitionAndSetupParticles(
    setup_p, bal, gptr, mprts,
    [&](int kind, Double3 pos, int patch, Int3 idx, psc_particle_np& np) {
      psc_particle_npt npt{};
      npt.kind = kind;
      if (kind == MY_ION) {
        npt.n = g.n;
        npt.T[0] = g.Ti_perp;
        npt.T[1] = g.Ti_perp;
        npt.T[2] = g.Ti_par;
      } else {
        npt.n = g.n;
        npt.T[0] = g.Te_perp;
        npt.T[1] = g.Te_perp;
        npt.T[2] = g.Te_par;
      }
      np.n = npt.n;
      // CHANGED: Using Maxwellian instead of KappaMultivariate for the Maxwellian case
      np.p = setup_p.createMaxwellian(npt);
    });
}

// ----------------------------------------------------------------------
// 8. Field initialization
// Background magnetic field B0 must be set along the Z-axis (HZ)
// to allow k_parallel dynamics in the Y-Z plane.
// ----------------------------------------------------------------------
static constexpr int B_DIR = HZ;

void initializeFields(MfieldsState& mflds)
{
  setupFields(mflds, [&](int m, double[3]) { return m == B_DIR ? g.B0 : 0.; });
}

// ----------------------------------------------------------------------
// 9. Main Run Function
// ----------------------------------------------------------------------
void run()
{
  mpi_printf(MPI_COMM_WORLD,
             "*** Setting up Mirror Instability Simulation (Maxwellian Case)...\n");
  setupParameters();
  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  psc_params.balance_interval = 500;
  Balance bal{3};
  psc_params.sort_interval = 10;

  double nu = 3.76 * pow(g.Te_perp, 2.) / g.Zi / g.lambda0;
  Collision coll{grid, -10, nu};

  ChecksParams chkp{};
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 100;
  Marder marder(grid, 0.9, 3, false);

  // Output fields configuration: Save data every 500 time steps
  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval = 500;
  ofip.tfield.out_interval = 500;
  ofip.tfield.average_every = 50;

  OutputFieldsParams ofp{};
  ofp.fields = ofip;
  ofp.moments = ofip;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  // Save only a representative central window of particles to control I/O.
  OutputParticlesParams opp{};
  opp.every_step = kParticleOutputStride; // Reactivado usando ADIOS2!
  opp.data_dir = ".";
  opp.basename = "prt_maxwellian";
  opp.lo = {0, (grid.domain.gdims[1] - kParticleWindowCells) / 2,
            (grid.domain.gdims[2] - kParticleWindowCells) / 2};
  opp.hi = {0, (grid.domain.gdims[1] + kParticleWindowCells) / 2,
            (grid.domain.gdims[2] + kParticleWindowCells) / 2};
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), -100};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid);
  // setup_p.kappa is not needed for Maxwellian, but keeping these config for structure
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population = MY_ION;

  auto mf_n = make_MfieldsMoment_n<Moment_n>(grid);

  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_p, bal, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto integrator =
    makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts, bal, coll,
                                 checks, marder, diagnostics);

  integrator.integrate();
}

// ----------------------------------------------------------------------
// 10. Main entry point
// ----------------------------------------------------------------------
int main(int argc, char** argv)
{
  psc_init(argc, argv);
  run();
  psc_finalize();
  return 0;
}
