// ======================================================================
// psc_reconnection_comparable.cxx
// Reconexión magnética (doble Harris) con parámetros IGUALADOS a los
// casos de inestabilidades por anisotropía (psc_anisotropy_case.hxx)
// para comparativa directa:
//
//   IGUALADO a anisotropía (bigbox40):
//     mass_ratio      = 200
//     caja            = 40 x 40 d_i
//     grilla          = 1152 x 1152  (28.8 celdas/d_i, dx = 0.491 d_e)
//     nicell          = 1000
//     cfl             = 0.95
//     kappa           = 3.0
//     np (defecto)    = 48 x 48 (2304 ranks, parches 24x24)
//     fields cada 500, particles cada 10000 (ventana 0.4-0.6),
//     checkpoint 5000, balance 2500, continuity 5000, gauss 100,
//     energias cada 100, overrides por variables de entorno.
//
//   DIFERENCIA DELIBERADA (única):
//     wpe/wce = 2.0 (anisotropía usa 12.5). En Harris, el balance de
//     presión fija Te = 1/(2*(wpe/wce)^2*(1+Ti/Te)); con 12.5 saldría
//     Te ~ 5.3e-4 -> lambda_De ~ 0.023 d_e -> dx/lambda_De ~ 21:
//     calentamiento de grilla garantizado en PIC explícito. Con 2.0:
//     Te = 1/48 -> lambda_De = 0.144 d_e -> dx/lambda_De = 3.4 (límite
//     clásico). Comparar en unidades iónicas (d_i, Omega_ci, v_A), donde
//     wpe/wce entra sólo débilmente en la física a escala iónica.
//
//   Escalas resultantes (wpe/wce=2, mi/me=200):
//     d_i = 14.14 d_e ; Omega_ci^-1 = 400 wpe^-1 ~ 1212 pasos
//     (dt = 0.95*0.491/sqrt(2) ~ 0.33 wpe^-1)
//
// Física de la hoja idéntica a psc_reconnection.cxx:
// dos hojas en y = -Ly/4 y +Ly/4, totalmente periódico, perturbación
// sólo en la hoja de y = +Ly/4, kappa multivariada isótropa (kappa=3).
// Ref: Agudelo Rueda et al., ApJ 971, 109 (2024)
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include <cstdlib>
#include <string>

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
  double dby_b0;              // Field perturbation amplitude
  double bg;                  // Guide field (fraction of B0)
  double Lpert_Lz;            // Perturbation wavelength ratio

  double mass_ratio;
  double Ti_Te, Tib_Ti, Teb_Te;
  double nb_n0;

  // Derived quantities
  double b0, d_i, wce, wci, wpe, wpi, wpe_wce;
  double L, Lx, Ly, Lz, Lpert, dby;
  double TTi, TTe;
};

static PscReconnectionParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

// Run-control (mismos nombres de entorno que psc_anisotropy_case.hxx)
static int ngrid = 1152;            // 40 d_i * 28.8 celdas/d_i
static int np_y = 48;
static int np_z = 48;
static int nicell = 1000;
static int fields_every = 500;
static int particles_every = 10000;
static int balance_interval = 2500;
static int continuity_every = 5000;
static int energies_every = 100;

static int envInt(const char* name, int fallback)
{
  if (const char* value = std::getenv(name)) {
    return std::stoi(value);
  }
  return fallback;
}

// ----------------------------------------------------------------------
// 3. Compile-time configuration — igual que anisotropía
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
// 4. Parameter setup
// ----------------------------------------------------------------------
void setupParameters() {
  psc_params.nmax = envInt("PSC_NMAX", 250000); // ~206 Omega_ci^-1
  psc_params.cfl  = 0.95;                       // MATCH anisotropía
  psc_params.write_checkpoint_every_step =
    envInt("PSC_CHECKPOINT_EVERY", 5000);       // MATCH anisotropía
  psc_params.stats_every = 50;                  // MATCH anisotropía

  ngrid = envInt("PSC_NGRID", ngrid);
  np_y = envInt("PSC_NP_Y", np_y);
  np_z = envInt("PSC_NP_Z", np_z);
  nicell = envInt("PSC_NICELL", nicell);
  fields_every = envInt("PSC_FIELDS_EVERY", fields_every);
  particles_every = envInt("PSC_PARTICLES_EVERY", particles_every);
  balance_interval = envInt("PSC_BALANCE_INTERVAL", balance_interval);
  continuity_every = envInt("PSC_CONTINUITY_EVERY", continuity_every);
  energies_every = envInt("PSC_ENERGIES_EVERY", energies_every);

  double me = 1;
  double ec = 1;
  double c = 1;
  double eps0 = 1;

  g.mass_ratio = 200.0; // MATCH anisotropía (antes 25)

  g.Lx_di = 1.;
  g.Ly_di = 40.;    // MATCH bigbox40: hojas en ±10 d_i (separación 20 d_i)
  g.Lz_di = 40.;    // MATCH bigbox40
  g.L_di = 0.5;     // Sheet half-thickness (física de la hoja, sin cambio)

  g.Ti_Te = 5.0;
  g.Tib_Ti = 1.0;
  g.Teb_Te = 1.0;
  g.nb_n0 = 0.2;

  g.bg = 0.0;
  g.theta = 0.0;
  g.dby_b0 = 0.03;
  g.Lpert_Lz = 1.0;

  // ÚNICA diferencia deliberada con anisotropía (ver cabecera):
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
}

// ----------------------------------------------------------------------
// 5. Pressure balance verification (sin cambios)
// ----------------------------------------------------------------------
void verifyPressureBalance() {
  double b0 = g.b0, L = g.L, Ly = g.Ly;
  double TTi = g.TTi, TTe = g.TTe;
  double nb = g.nb_n0;

  mpi_printf(MPI_COMM_WORLD,
    "\n=== Pressure Balance Verification (double Harris sheet) ===\n");
  mpi_printf(MPI_COMM_WORLD,
    "  B0 = %g, L = %g, TTi = %g, TTe = %g, nb = %g\n",
    b0, L, TTi, TTe, nb);

  double y_positions[] = {
    -0.25 * Ly, -0.125 * Ly, 0.0, 0.125 * Ly, 0.25 * Ly, 0.4 * Ly,
  };

  double P_ref = -1;
  for (int i = 0; i < 6; i++) {
    double y = y_positions[i];

    double Bz = b0 * (tanh((y + 0.25*Ly) / L) - tanh((y - 0.25*Ly) / L) - 1.0);
    double Bx = b0 * g.bg;
    double B_sq = Bz*Bz + Bx*Bx;

    double n_harris = 1.0 / sqr(cosh((y + 0.25*Ly) / L))
                    + 1.0 / sqr(cosh((y - 0.25*Ly) / L));

    double P_plasma = (n_harris + nb) * (TTi + TTe);
    double P_mag = 0.5 * B_sq;
    double P_total = P_plasma + P_mag;

    if (P_ref < 0) P_ref = P_total;
    double err = fabs(P_total - P_ref) / P_ref * 100.0;

    mpi_printf(MPI_COMM_WORLD,
      "  y/Ly = %+6.3f: n_H = %.4f, |B| = %.4f, P_plasma = %.6f, "
      "P_mag = %.6f, P_total = %.6f (err = %.2f%%)\n",
      y / Ly, n_harris, sqrt(B_sq), P_plasma, P_mag, P_total, err);
  }
  mpi_printf(MPI_COMM_WORLD,
    "=== End Pressure Balance ===\n\n");
}

// ----------------------------------------------------------------------
// 6. Grid setup — FULLY PERIODIC
// ----------------------------------------------------------------------
Grid_t* setupGrid() {
  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ELECTRON]    = {-1., 1., "e"};
  kinds[MY_ION]         = {1., g.mass_ratio, "i"};
  kinds[MY_ELECTRON_BG] = {-1., 1., "e_bg"};
  kinds[MY_ION_BG]      = {1., g.mass_ratio, "i_bg"};

  mpi_printf(MPI_COMM_WORLD, "case = reconnection_comparable\n");
  mpi_printf(MPI_COMM_WORLD,
             "run = nmax %d, ngrid %d, nicell %d, np 1x%dx%d\n",
             psc_params.nmax, ngrid, np_y, np_z, nicell);
  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De (background) = %g\n", sqrt(g.TTe));

  Grid_t::Real3 LL = {g.Lx_di * g.d_i, g.Ly_di * g.d_i, g.Lz_di * g.d_i};
  Int3 gdims = {1, ngrid, ngrid};
  Int3 np = {1, np_y, np_z};

  // Domain centered at y=0: y in [-Ly/2, +Ly/2]
  Grid_t::Domain domain{gdims, LL, {0, -.5 * LL[1], 0}, np};

  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}
  };

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = nicell;

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
// 7. Particle initialization — kappa multivariada (sin cambios de física)
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts) {

  partitionAndSetupParticles(setup_p, bal, gptr, mprts,
    [&](int kind, Double3 crd, int patch, Int3 idx, psc_particle_np& np){
      psc_particle_npt npt{};
      npt.kind = kind;

      double y = crd[1];
      double Ly = g.Ly;

      double n_sheet1 = 1. / sqr(cosh((y + 0.25 * Ly) / g.L));
      double n_sheet2 = 1. / sqr(cosh((y - 0.25 * Ly) / g.L));
      double n_total_harris = n_sheet1 + n_sheet2;

      double drift_weight = 0.;
      if (n_total_harris > 1e-10) {
        drift_weight = (n_sheet1 - n_sheet2) / n_total_harris;
      }

      switch (kind) {
        case MY_ELECTRON:
          npt.n = n_total_harris;
          npt.p[0] = -2. * g.TTe / g.b0 / g.L * drift_weight;
          npt.T[0] = g.TTe; npt.T[1] = g.TTe; npt.T[2] = g.TTe;
          npt.kind = MY_ELECTRON;
          break;
        case MY_ION:
          npt.n = n_total_harris;
          npt.p[0] = 2. * g.TTi / g.b0 / g.L * drift_weight;
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
      np.p = setup_p.createKappaMultivariate(npt);
    });
}

// ----------------------------------------------------------------------
// 8. Field initialization (sin cambios de física)
// ----------------------------------------------------------------------
void initializeFields(MfieldsState& mflds) {
  double b0 = g.b0, dby = g.dby;
  double L = g.L, Ly = g.Ly, Lz = g.Lz, Lpert = g.Lpert;
  double cs = cos(g.theta), sn = sin(g.theta);
  double sigma = L;

  mprintf("Double Harris sheet: L=%g, Ly=%g, Lz=%g\n", L, Ly, Lz);
  mprintf("Perturbation: dby=%g, sigma=%g, Lpert=%g\n", dby, sigma, Lpert);

  setupFields(mflds, [&](int m, double crd[3]) {
    double y = crd[1], z = crd[2];

    double Bz_eq = cs * b0 * (tanh((y + 0.25*Ly) / L)
                             - tanh((y - 0.25*Ly) / L) - 1.0);
    double Bx_eq = -sn * b0 * (tanh((y + 0.25*Ly) / L)
                              - tanh((y - 0.25*Ly) / L) - 1.0)
                 + b0 * g.bg;

    double y_rel = (y - 0.25 * Ly) / sigma;
    double sech_val = 1.0 / cosh(y_rel);
    double kz = 2.0 * M_PI / Lpert;

    double pert_By = dby * sech_val
                   * sin(kz * (z - 0.5 * Lz));
    double pert_Bz = (dby * sigma * kz) * tanh(y_rel) * sech_val
                   * cos(kz * (z - 0.5 * Lz));

    switch (m) {
      case HX: return Bx_eq;
      case HY: return pert_By;
      case HZ: return Bz_eq + pert_Bz;
      default: return 0.;
    }
  });
}

// ----------------------------------------------------------------------
// 9. Main Run Function
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD,
    "*** Reconnection (double Harris) — parámetros comparables a anisotropía ***\n");
  setupParameters();

  verifyPressureBalance();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if(!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  psc_params.balance_interval = balance_interval; // MATCH anisotropía (2500)
  Balance bal{3};
  psc_params.sort_interval = 10;

  int collision_interval = -10;   // off, MATCH anisotropía
  double collision_nu = 1e-10;
  Collision coll{grid, collision_interval, collision_nu};

  ChecksParams chkp{};
  chkp.continuity.check_interval = continuity_every; // MATCH (5000)
  chkp.continuity.err_threshold = 1e-4;
  chkp.continuity.print_max_err_always = true;
  chkp.continuity.dump_always = false;
  chkp.gauss.check_interval = 100;                   // MATCH (100)
  chkp.gauss.err_threshold = 1e-4;
  chkp.gauss.print_max_err_always = true;
  chkp.gauss.dump_always = false;
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 100;
  Marder marder(grid, 0.9, 3, false);

  OutputFieldsItemParams ofip{};
  ofip.pfield.out_interval = fields_every; // MATCH (500)
  OutputFieldsParams ofp{};
  ofp.fields = ofip;
  ofp.moments = ofip;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  OutputParticlesParams opp{};
  opp.every_step = particles_every;        // MATCH (10000)
  opp.data_dir = ".";
  opp.basename = "prt_reconnection_comparable";
  opp.lo = {0, int(0.4 * ngrid), int(0.4 * ngrid)}; // MATCH ventana 0.4-0.6
  opp.hi = {1, int(0.6 * ngrid), int(0.6 * ngrid)};
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), energies_every};    // MATCH (activo)
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid, 4);
  setup_p.kappa = 3.0;                     // MATCH kappa=3
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

  if (const char* restart = std::getenv("PSC_RESTART")) {
    read_checkpoint_filename = restart;
  }

  run();
  psc_finalize();
  return 0;
}
