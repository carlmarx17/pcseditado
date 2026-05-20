// ======================================================================
// psc_reconnection_mini.cxx
// Prueba LIGERA de reconexión magnética (Harris sheet).
// Propósito: Verificar que el código arranca, inicializa partículas/campos
//            y corre algunos pasos sin errores — no pretende física real.
//
// Diferencias vs. psc_reconnection.cxx:
//   - Grid: {1, 32, 128}   (en vez de 128×512)
//   - np:   {1, 1, 1}      (sin descomposición MPI)
//   - nicell: 4            (en vez de 100)
//   - nmax: 200            (en vez de 10 millones)
//   - write_checkpoint: desactivado
//   - output de campos cada 50 pasos, partículas desactivado
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
  double Lx_di, Ly_di, Lz_di;
  double L_di;
  double theta;
  double dby_b0;
  double bg;
  double Lpert_Lz;
  double mass_ratio;
  double Ti_Te, Tib_Ti, Teb_Te;
  double nb_n0;
  double b0, d_i, wce, wci, wpe, wpi, wpe_wce;
  double L, Lx, Ly, Lz, Lpert, dby, dbz;
  double TTi, TTe;
};

static PscReconnectionParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

// ----------------------------------------------------------------------
// 3. Compile-time configuration (YZ plane, single precision CPU)
// ----------------------------------------------------------------------
using Dim = dim_yz;

#ifdef USE_CUDA
using PscConfig = PscConfig1vbecCuda<Dim>;
#else
using PscConfig = PscConfig1vbecSingle<Dim>;
#endif

using Writer         = WriterDefault;
using MfieldsState   = PscConfig::MfieldsState;
using Mparticles     = PscConfig::Mparticles;
using Balance        = PscConfig::Balance;
using Collision      = PscConfig::Collision;
using Checks         = PscConfig::Checks;
using Marder         = PscConfig::Marder;
using OutputParticles= PscConfig::OutputParticles;

// ----------------------------------------------------------------------
// 4. Moment selector
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

// ----------------------------------------------------------------------
// 5. Parameter setup — valores MINI para prueba rápida
// ----------------------------------------------------------------------
void setupParameters() {
  psc_params.nmax  = 200;       // ← muy pocos pasos
  psc_params.cfl   = 0.99;
  psc_params.write_checkpoint_every_step = 0; // sin checkpoints
  psc_params.stats_every = 10;

  double me = 1, ec = 1, c = 1, eps0 = 1;

  g.mass_ratio = 25.0;

  // Dominio reducido: 10 × 10 d_i (en vez de 10 × 40)
  g.Lx_di = 1.;
  g.Ly_di = 10.;
  g.Lz_di = 10.;
  g.L_di  = 0.5;

  g.Ti_Te  = 5.0;
  g.Tib_Ti = 0.333;
  g.Teb_Te = 0.333;
  g.nb_n0  = 0.05;

  g.bg      = 0.0;
  g.theta   = 0.0;
  g.dby_b0  = 0.03;
  g.Lpert_Lz= 1.0;
  g.wpe_wce = 2.0;

  g.TTe = me * sqr(c) / (2. * eps0 * sqr(g.wpe_wce) * (1. + g.Ti_Te));
  g.TTi = g.TTe * g.Ti_Te;

  g.wci = 1. / (g.mass_ratio * g.wpe_wce);
  g.wce = g.wci * g.mass_ratio;
  g.wpe = g.wce * g.wpe_wce;
  g.wpi = g.wpe / sqrt(g.mass_ratio);

  g.d_i  = c / g.wpi;
  g.L    = g.L_di  * g.d_i;
  g.Lx   = g.Lx_di * g.d_i;
  g.Ly   = g.Ly_di * g.d_i;
  g.Lz   = g.Lz_di * g.d_i;

  g.b0    = me * c * g.wce / ec;
  g.Lpert = g.Lpert_Lz * g.Lz;
  g.dby   = g.dby_b0 * g.b0;
  g.dbz   = -g.dby * g.Lpert / (2. * g.Ly);
}

// ----------------------------------------------------------------------
// 6. Grid setup — MINI: 1×32×128, sin descomposición MPI
// ----------------------------------------------------------------------
Grid_t* setupGrid() {
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ELECTRON]    = {-1., 1.,          "e"   };
  kinds[MY_ION]         = { 1., g.mass_ratio, "i"   };
  kinds[MY_ELECTRON_BG] = {-1., 1.,          "e_bg"};
  kinds[MY_ION_BG]      = { 1., g.mass_ratio, "i_bg"};

  mpi_printf(MPI_COMM_WORLD, "[MINI] d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "[MINI] lambda_De (bg) = %g\n", sqrt(g.TTe));
  mpi_printf(MPI_COMM_WORLD, "[MINI] b0 = %g, L = %g\n", g.b0, g.L);

  Grid_t::Real3 LL = {g.Lx_di * g.d_i, g.Ly_di * g.d_i, g.Lz_di * g.d_i};

  // Grid pequeño: 1 × 32 × 128  (en vez de 1 × 128 × 512)
  Int3 gdims = {1, 32, 128};
  Int3 np    = {1,  1,   1};   // un solo patch MPI

  Grid_t::Domain domain{gdims, LL, {0, -.5 * LL[1], 0}, np};

  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC}
  };

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = 4;   // ← muy pocas partículas por celda para rapidez

  mprintf("[MINI] dx %g %g %g\n", domain.dx[0], domain.dx[1], domain.dx[2]);
  double dt = psc_params.cfl * courant_length(domain);
  mprintf("[MINI] dt %g   cfl %g\n", dt, psc_params.cfl);

  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) ibn[0] = 0;
  if (Dim::InvarY::value) ibn[1] = 0;
  if (Dim::InvarZ::value) ibn[2] = 0;

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// 7. Particle initialization (Maxwellian — sin Kappa para simplicidad)
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts) {
  partitionAndSetupParticles(setup_p, bal, gptr, mprts,
    [&](int kind, Double3 crd, int patch, Int3 idx, psc_particle_np& np) {
      psc_particle_npt npt{};
      npt.kind = kind;

      switch (kind) {
        case MY_ELECTRON:
          npt.n    = 1. / sqr(cosh(crd[1] / g.L));
          npt.p[0] = -2. * g.TTe / g.b0 / g.L;
          npt.T[0] = g.TTe; npt.T[1] = g.TTe; npt.T[2] = g.TTe;
          break;
        case MY_ION:
          npt.n    = 1. / sqr(cosh(crd[1] / g.L));
          npt.p[0] =  2. * g.TTi / g.b0 / g.L;
          npt.T[0] = g.TTi; npt.T[1] = g.TTi; npt.T[2] = g.TTi;
          break;
        case MY_ELECTRON_BG:
          npt.n    = g.nb_n0;
          npt.p[0] = 0.;
          npt.T[0] = g.Teb_Te * g.TTe;
          npt.T[1] = g.Teb_Te * g.TTe;
          npt.T[2] = g.Teb_Te * g.TTe;
          break;
        case MY_ION_BG:
          npt.n    = g.nb_n0;
          npt.p[0] = 0.;
          npt.T[0] = g.Tib_Ti * g.TTi;
          npt.T[1] = g.Tib_Ti * g.TTi;
          npt.T[2] = g.Tib_Ti * g.TTi;
          break;
        default: assert(0);
      }

      np.n = npt.n;
      np.p = setup_p.createMaxwellian(npt);
    });
}

// ----------------------------------------------------------------------
// 8. Field initialization (Harris sheet + perturbación)
// ----------------------------------------------------------------------
void initializeFields(MfieldsState& mflds) {
  double b0 = g.b0, dby = g.dby, dbz = g.dbz;
  double L = g.L, Ly = g.Ly, Lz = g.Lz, Lpert = g.Lpert;
  double cs = cos(g.theta), sn = sin(g.theta);

  mprintf("[MINI] L %g  b0 %g  dby %g  dbz %g\n", L, b0, dby, dbz);

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
      default: return 0.;
    }
  });
}

// ----------------------------------------------------------------------
// 9. Main run
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD,
    "=== [MINI] Prueba rapida de reconexion magnetica ===\n");
  mpi_printf(MPI_COMM_WORLD,
    "    Grid: 1x32x128  nicell=4  nmax=200\n");

  setupParameters();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);

  psc_params.balance_interval = 0; // sin balanceo
  Balance bal{3};
  psc_params.sort_interval = 10;

  Collision coll{grid, 0, 1e-10};

  ChecksParams chkp{};
  chkp.continuity.check_interval = 0;
  chkp.gauss.check_interval      = -50; // revisar Gauss cada 50 pasos
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 0; // sin Marder en prueba mini
  Marder marder(grid, 0.9, 3, false);

  // Desactivar TODO output de campos/momentos (evita crash de vista xdmf en grid mini)
  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval = -1; // DESACTIVADO
  ofip.tfield.out_interval = -1; // DESACTIVADO

  OutputFieldsParams ofp{};
  ofp.fields  = ofip;
  ofp.moments = ofip;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  // Sin output de partículas
  OutputParticlesParams opp{};
  opp.every_step = -1;
  opp.data_dir   = ".";
  opp.basename   = "prt_mini";
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), -1}; // DESACTIVADO en grid mini (crash gt::view con ibn[0]=0)

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid, 4);
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population = MY_ION_BG;

  initializeParticles(setup_p, bal, grid_ptr, mprts);
  initializeFields(mflds);

  mpi_printf(MPI_COMM_WORLD,
    "[MINI] Iniciando integracion por %d pasos...\n", psc_params.nmax);

  auto integrator = makePscIntegrator<PscConfig>(psc_params,
    *grid_ptr, mflds, mprts, bal, coll, checks, marder, diagnostics);

  integrator.integrate();

  mpi_printf(MPI_COMM_WORLD,
    "=== [MINI] Prueba completada exitosamente! ===\n");
}

// ----------------------------------------------------------------------
// 10. Entry point
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
  psc_init(argc, argv);
  run();
  psc_finalize();
  return 0;
}
