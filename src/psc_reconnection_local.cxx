// ======================================================================
// psc_reconnection_local.cxx
// Reconexión magnética — versión PC local para análisis.
//
// Configuración (balance RAM/velocidad en laptop):
//   Grid:   1 × 64 × 256   (YZ plane, Harris sheet)
//   nicell: 10              (partículas/celda)
//   nmax:   1000            (~1 τ_ci aproximado según dt)
//   output: pfd cada 50 pasos (campos E,B,J)
//   MPI:    1 rank (np = {1,1,1}) — sin descomposición, evita crash Balance
//
// Física: hoja Harris  Bz = b0·tanh(y/L)
//         Perturbación Bpert para desencadenar reconexión
//         mass_ratio=25, wpe/wce=2, Ti/Te=5
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

enum {
  MY_ELECTRON,
  MY_ION,
  MY_ELECTRON_BG,
  MY_ION_BG,
  N_MY_KINDS
};

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

// YZ plane (X invariante)
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
// Parámetros físicos (iguales al psc_reconnection completo)
// ----------------------------------------------------------------------
void setupParameters() {
  psc_params.nmax  = 5000;   // ~54 Ωᵢ⁻¹ → ver reconexión completa
  psc_params.cfl   = 0.99;
  psc_params.write_checkpoint_every_step = 1000;
  psc_params.stats_every = 50;

  double me = 1, ec = 1, c = 1, eps0 = 1;

  g.mass_ratio = 25.0;

  g.Lx_di = 1.;
  g.Ly_di = 10.;   // ancho de la caja en y
  g.Lz_di = 40.;   // largo en z (dirección de reconexión)
  g.L_di  = 0.5;   // grosor de la hoja Harris

  g.Ti_Te  = 5.0;
  g.Tib_Ti = 0.333;
  g.Teb_Te = 0.333;
  g.nb_n0  = 0.05;

  g.bg      = 0.0;   // sin campo guía
  g.theta   = 0.0;
  g.dby_b0  = 0.03;  // perturbación del 3%
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
// Grid: 1×64×256, 4 patches en Z (una capa Harris)
// Resolución: dy = Ly/64 ≈ 0.78 d_i, dz = Lz/256 ≈ 0.78 d_i
// ----------------------------------------------------------------------
Grid_t* setupGrid() {
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ELECTRON]    = {-1., 1.,          "e"   };
  kinds[MY_ION]         = { 1., g.mass_ratio, "i"   };
  kinds[MY_ELECTRON_BG] = {-1., 1.,          "e_bg"};
  kinds[MY_ION_BG]      = { 1., g.mass_ratio, "i_bg"};

  mpi_printf(MPI_COMM_WORLD, "=== psc_reconnection_local ===\n");
  mpi_printf(MPI_COMM_WORLD, "d_e=%.3g  d_i=%.3g  b0=%.3g  L=%.3g\n",
             1., g.d_i, g.b0, g.L);
  mpi_printf(MPI_COMM_WORLD, "TTe=%.3g  TTi=%.3g  wpe/wce=%.1f\n",
             g.TTe, g.TTi, g.wpe_wce);
  mpi_printf(MPI_COMM_WORLD, "lambda_De(bg)=%.3g\n", sqrt(g.TTe));

  Grid_t::Real3 LL = {g.Lx, g.Ly, g.Lz};

  // Grid local: 1×64×256, UN SOLO patch (sin descomposición MPI)
  Int3 gdims = {1, 64, 256};
  Int3 np    = {1,  1,   1};  // 1 patch → correr con mpirun -np 1

  Grid_t::Domain domain{gdims, LL, {0., -.5 * LL[1], 0.}, np};

  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_CONDUCTING_WALL, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_REFLECTING,      BND_PRT_PERIODIC}
  };

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = 10;  // 10 partículas/celda

  mprintf("dx %g %g %g\n", domain.dx[0], domain.dx[1], domain.dx[2]);
  double dt = psc_params.cfl * courant_length(domain);
  mprintf("dt=%g  → nmax=%d pasos = %.2f wci^-1\n",
          dt, psc_params.nmax, dt * psc_params.nmax * g.wci);

  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) ibn[0] = 0;
  if (Dim::InvarY::value) ibn[1] = 0;
  if (Dim::InvarZ::value) ibn[2] = 0;

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// Inicialización de partículas — Maxwellian estándar
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
// Campos: hoja Harris + perturbación magnética
// ----------------------------------------------------------------------
void initializeFields(MfieldsState& mflds) {
  double b0 = g.b0, dby = g.dby, dbz = g.dbz;
  double L = g.L, Ly = g.Ly, Lz = g.Lz, Lpert = g.Lpert;
  double cs = cos(g.theta), sn = sin(g.theta);

  mprintf("Fields: b0=%g  L=%g  dby=%g  dbz=%g\n", b0, L, dby, dbz);

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
// Run
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD,
    "\n=== Reconexion Magnetica LOCAL (PC) ===\n"
    "    Grid 64x256, nicell=10, nmax=5000 (~54 Oi^-1)\n"
    "    1 MPI rank, output pfd cada 50 pasos\n\n");

  setupParameters();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  psc_params.balance_interval = 0;   // sin rebalanceo automático en PC
  Balance bal{3};
  psc_params.sort_interval = 10;

  Collision coll{grid, 0, 1e-10};

  ChecksParams chkp{};
  chkp.continuity.check_interval = 0;
  chkp.gauss.check_interval      = -100;
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 100;
  Marder marder(grid, 0.9, 3, false);

  // Output de campos (E, B, J) cada 50 pasos
  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval = 50;
  ofip.tfield.out_interval = -1;

  // Momentos (densidad, corriente) cada 50 pasos
  OutputFieldsItemParams ofip_m{};
  ofip_m.pfield.out_interval = 50;
  ofip_m.tfield.out_interval = -1;

  OutputFieldsParams ofp{};
  ofp.fields  = ofip;
  ofp.moments = ofip_m;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  // Sin output de partículas (muy costoso)
  OutputParticlesParams opp{};
  opp.every_step = -1;
  opp.data_dir   = ".";
  opp.basename   = "prt_local";
  OutputParticles outp{grid, opp};

  // DiagEnergies desactivado (incompatible con ibn[0]=0 en dim_yz)
  DiagEnergies oute{grid.comm(), -1};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid, 4);
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population = MY_ION_BG;

  if (read_checkpoint_filename.empty()) {
    initializeParticles(setup_p, bal, grid_ptr, mprts);
    initializeFields(mflds);
  }

  mpi_printf(MPI_COMM_WORLD, "\n→ Iniciando integracion PIC...\n\n");

  auto integrator = makePscIntegrator<PscConfig>(psc_params,
    *grid_ptr, mflds, mprts, bal, coll, checks, marder, diagnostics);

  integrator.integrate();

  mpi_printf(MPI_COMM_WORLD,
    "\n=== Simulacion LOCAL completada! ===\n"
    "    Archivos pfd.*.h5 y pfd_moments.*.h5 guardados en ./\n\n");
}

int main(int argc, char** argv) {
  psc_init(argc, argv);
  run();
  psc_finalize();
  return 0;
}
