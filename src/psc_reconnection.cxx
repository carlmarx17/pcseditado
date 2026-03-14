// ======================================================================
// psc_reconnection.cxx
// Simulation of Magnetic Reconnection in Collisionless Plasma.
// Combines Harris current sheet field with Kappa temperature distributions.
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

// ----------------------------------------------------------------------
// 1. Particle kinds
// ----------------------------------------------------------------------
enum {
  MY_ELECTRON,
  MY_ION,
  MY_ELECTRON_BG,
  MY_ION_BG,
  N_MY_KINDS
};

// ----------------------------------------------------------------------
// 2. Simulation parameters
// ----------------------------------------------------------------------
struct PscReconnectionParams {
  double Lx_di, Ly_di, Lz_di; // Size of box in d_i
  double L_di;                // Sheet thickness / ion inertial length

  double theta;
  double dby_b0;              // Field perturbation
  double bg;                  // Guide field
  double Lpert_Lz;            // Perturbation wavelength ratio
  
  double mass_ratio;
  double Ti_Te, Tib_Ti, Teb_Te;
  double nb_n0;
  
  // Derived quantities
  double b0, d_i, wce, wci, wpe, wpi, wpe_wce;
  double L, Lx, Ly, Lz, Lpert, dby, dbz;
  double TTi, TTe;
};

static PscReconnectionParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

// ----------------------------------------------------------------------
// 3. Compile-time configuration
// Reconneciton in YZ plane
// ----------------------------------------------------------------------
using Dim = dim_yz;

#ifdef USE_CUDA
using PscConfig = PscConfig1vbecCuda<Dim>;
#else
using PscConfig = PscConfig1vbecSingle<Dim>;
#endif

using Writer = WriterDefault;
using MfieldsState    = PscConfig::MfieldsState;
using Mparticles      = PscConfig::Mparticles;
using Balance         = PscConfig::Balance;
using Collision       = PscConfig::Collision;
using Checks          = PscConfig::Checks;
using Marder          = PscConfig::Marder;
using OutputParticles = PscConfig::OutputParticles;

// ----------------------------------------------------------------------
// 4. Moment selector setup
// ----------------------------------------------------------------------
template <typename Mp, typename Dm, typename Enable = void>
struct Moment_n_Selector { using type = Moment_n_1st<Mp, MfieldsC>; };

#ifdef USE_CUDA
template <typename Mp, typename Dm>
struct Moment_n_Selector<Mp, Dm, typename std::enable_if<Mp::is_cuda::value>::type> {
  using type = Moment_n_1st_cuda<Mp, Dm>;
};
#endif
using Moment_n = typename Moment_n_Selector<Mparticles, Dim>::type;

template <typename Moment>
auto make_MfieldsMoment_n(const Grid_t &grid) {
  return MfieldsC(grid, grid.kinds.size(), grid.ibn);
}

#ifdef USE_CUDA
template <>
auto make_MfieldsMoment_n<Moment_n>(const Grid_t &grid) {
  return HMFields({-grid.ibn, grid.domain.ldims + 2*grid.ibn},
                  grid.kinds.size(), grid.n_patches());
}
#endif

// ----------------------------------------------------------------------
// 5. Parameter setup
// ----------------------------------------------------------------------
void setupParameters() {
  psc_params.nmax = 10000001; 
  psc_params.cfl  = 0.99;
  psc_params.write_checkpoint_every_step = 10000;
  psc_params.stats_every = 1;

  double me = 1;
  double ec = 1;
  double c = 1;
  double eps0 = 1;

  g.mass_ratio = 25.0; 

  g.Lx_di = 1.;
  g.Ly_di = 10.;
  g.Lz_di = 40.;
  g.L_di = 0.5;

  g.Ti_Te = 5.0;
  g.Tib_Ti = 0.333;
  g.Teb_Te = 0.333;
  g.nb_n0 = 0.05; // 5% background density
  
  g.bg = 0.0;     // no guide field
  g.theta = 0.0;  
  g.dby_b0 = 0.03; 
  g.Lpert_Lz = 1.0;

  g.wpe_wce = 2.0;

  g.TTe = me * sqr(c) / (2. * eps0 * sqr(g.wpe_wce) * (1. + g.Ti_Te));
  g.TTi = g.TTe * g.Ti_Te;

  g.wci = 1. / (g.mass_ratio * g.wpe_wce);
  g.wce = g.wci * g.mass_ratio;
  g.wpe = g.wce * g.wpe_wce;
  g.wpi = g.wpe / sqrt(g.mass_ratio);

  g.d_i = c / g.wpi;
  g.L = g.L_di * g.d_i;
  g.Lx = g.Lx_di * g.d_i;
  g.Ly = g.Ly_di * g.d_i;
  g.Lz = g.Lz_di * g.d_i;

  g.b0 = me * c * g.wce / ec;
  g.Lpert = g.Lpert_Lz * g.Lz;
  g.dby = g.dby_b0 * g.b0;
  g.dbz = -g.dby * g.Lpert / (2. * g.Ly);
}

// ----------------------------------------------------------------------
// 6. Grid setup
// ----------------------------------------------------------------------
Grid_t* setupGrid() {
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ELECTRON]    = {-1., 1., "e"};
  kinds[MY_ION]         = {1., g.mass_ratio, "i"};
  kinds[MY_ELECTRON_BG] = {-1., 1., "e_bg"};
  kinds[MY_ION_BG]      = {1., g.mass_ratio, "i_bg"};

  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De (background) = %g\n", sqrt(g.TTe));

  Grid_t::Real3 LL = {g.Lx_di * g.d_i, g.Ly_di * g.d_i, g.Lz_di * g.d_i};
  Int3 gdims = {1, 128, 512};
  Int3 np = {1, 1, 4}; // MPI DECOMPOSITION
  
  Grid_t::Domain domain{gdims, LL, {0, -.5 * LL[1], 0}, np};

  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC}
  };

  auto norm_params = Grid_t::NormalizationParams::dimensionless(); 
  norm_params.nicell = 100; 
  
  mprintf("dx %g %g %g\n", domain.dx[0], domain.dx[1], domain.dx[2]);
  double dt = psc_params.cfl * courant_length(domain);
  mprintf("dt %g cfl %g\n", dt, psc_params.cfl);

  Grid_t::Normalization norm{norm_params};
  
  Int3 ibn = {2, 2, 2}; 
  if(Dim::InvarX::value) ibn[0] = 0;
  if(Dim::InvarY::value) ibn[1] = 0; 
  if(Dim::InvarZ::value) ibn[2] = 0;
  
  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// 7. Particle initialization (using Kappa distributions)
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts) {
                         
  partitionAndSetupParticles(setup_p, bal, gptr, mprts,
    [&](int kind, Double3 crd, int patch, Int3 idx, psc_particle_np& np){
      psc_particle_npt npt{};
      npt.kind = kind;
      
      switch (kind) {
        case MY_ELECTRON: // drifting electrons
          npt.n = 1. / sqr(cosh(crd[1] / g.L));
          npt.p[0] = -2. * g.TTe / g.b0 / g.L;
          npt.T[0] = g.TTe; npt.T[1] = g.TTe; npt.T[2] = g.TTe;
          npt.kind = MY_ELECTRON;
          break;
        case MY_ION: // drifting ions
          npt.n = 1. / sqr(cosh(crd[1] / g.L));
          npt.p[0] = 2. * g.TTi / g.b0 / g.L;
          npt.T[0] = g.TTi; npt.T[1] = g.TTi; npt.T[2] = g.TTi;
          npt.kind = MY_ION;
          break;
        case MY_ELECTRON_BG: 
          npt.n = g.nb_n0;
          npt.p[0] = 0.;
          npt.T[0] = g.Teb_Te * g.TTe;
          npt.T[1] = g.Teb_Te * g.TTe;
          npt.T[2] = g.Teb_Te * g.TTe;
          npt.kind = MY_ELECTRON_BG;
          break;
        case MY_ION_BG:
          npt.n = g.nb_n0;
          npt.p[0] = 0.;
          npt.T[0] = g.Tib_Ti * g.TTi;
          npt.T[1] = g.Tib_Ti * g.TTi;
          npt.T[2] = g.Tib_Ti * g.TTi;
          npt.kind = MY_ION_BG;
          break;
        default: assert(0);
      }
      
      np.n = npt.n;
      // INITIALIZE WITH KAPPA DISTRIBUTION (from setup_p)
      np.p = setup_p.createKappaMultivariate(npt);
    });
}

// ----------------------------------------------------------------------
// 8. Field initialization (Harris sheet + Perturbation)
// ----------------------------------------------------------------------
void initializeFields(MfieldsState& mflds) {
  double b0 = g.b0, dby = g.dby, dbz = g.dbz;
  double L = g.L, Ly = g.Ly, Lz = g.Lz, Lpert = g.Lpert;
  double cs = cos(g.theta), sn = sin(g.theta);

  mprintf("L %g\n", L);
  setupFields(mflds, [&](int m, double crd[3]) {
    double y = crd[1], z = crd[2];

    switch (m) {
      case HX: return -sn * b0 * tanh(y / L) + b0 * g.bg;

      case HY:
        return dby * cos(M_PI * y / Ly) *
               sin(2.0 * M_PI * (z - 0.5 * Lz) / Lpert);

      case HZ:
        return cs * b0 * tanh(y / L) +
               dbz * cos(2. * M_PI * (z - .5 * Lz) / Lpert) *
                 sin(M_PI * y / Ly);

      // JXI: Optional, sometimes needed for correctness in boundaries
      default: return 0.;
    }
  });
}

// ----------------------------------------------------------------------
// 9. Main Run Function
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD, "*** Setting up Magnetic Reconnection w/ Kappa Simulation...\n");
  setupParameters();
  
  auto grid_ptr = setupGrid(); 
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if(!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  psc_params.balance_interval = 500;
  Balance bal{3};
  psc_params.sort_interval = 10;

  int collision_interval = 0;
  double collision_nu = 1e-10;
  Collision coll{grid, collision_interval, collision_nu};

  ChecksParams chkp{};
  chkp.continuity.check_interval = 0;
  chkp.gauss.check_interval = -100;
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 100;
  Marder marder(grid, 0.9, 3, false);

  // Output fields configuration
  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval = 100;
  ofip.tfield.out_interval = -4;
  ofip.tfield.average_every = 50;
  
  OutputFieldsParams ofp{};
  ofp.fields = ofip;
  ofp.moments = ofip;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  // Output particles configuration
  OutputParticlesParams opp{};
  opp.every_step = -4; 
  opp.data_dir = ".";
  opp.basename = "prt";
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), -100};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid, 4);
  setup_p.kappa = 3.0; // The parameter kappa for the distribution!
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population = MY_ION_BG;

  if(!read_checkpoint_filename.empty()) {
     // Checkpoint loaded
  } else {
    initializeParticles(setup_p, bal, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto integrator = makePscIntegrator<PscConfig>(psc_params,
    *grid_ptr, mflds, mprts, bal, coll, checks, marder, diagnostics);

  integrator.integrate();
}

// ----------------------------------------------------------------------
// 10. Main entry point
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
  psc_init(argc, argv);
  run();
  psc_finalize();
  return 0;
}
