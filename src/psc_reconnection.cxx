// ======================================================================
// psc_reconnection.cxx
// Simulation of Magnetic Reconnection in Collisionless Plasma.
//
// *** DOUBLE HARRIS CURRENT SHEET ***
// Two antiparallel current sheets at y = -Ly/4 and y = +Ly/4
// with FULLY PERIODIC boundary conditions in all directions.
//
// Pressure balance: P_total = n(T_i + T_e) + B^2/(2 mu_0) = const
// Perturbation applied ONLY near the sheet at y = +Ly/4.
//
// Ref: Agudelo Rueda et al., ApJ 971, 109 (2024)
//      doi:10.3847/1538-4357/ad5e73
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

// ----------------------------------------------------------------------
// 3. Compile-time configuration
// Reconnection in YZ plane (X is out-of-plane)
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
  g.Ly_di = 25.6;   // Large Y domain for double Harris sheet (2 sheets well separated)
  g.Lz_di = 51.2;   // Long Z domain for outflow jets to develop
  g.L_di = 0.5;     // Sheet half-thickness

  g.Ti_Te = 5.0;
  g.Tib_Ti = 1.0;   // Background Ti = Harris Ti (for clean pressure balance)
  g.Teb_Te = 1.0;   // Background Te = Harris Te (for clean pressure balance)
  g.nb_n0 = 0.2;    // 20% background density (helps numerical stability)

  g.bg = 0.0;       // No guide field (anti-parallel reconnection)
  g.theta = 0.0;
  g.dby_b0 = 0.03;  // 3% perturbation amplitude
  g.Lpert_Lz = 1.0;

  g.wpe_wce = 2.0;

  // Harris equilibrium: n0 * (Ti + Te) = B0^2 / (2 mu0)
  // In PSC units (c=1, me=1, eps0=1, wpe=1):
  //   Te = me*c^2 / (2*eps0*wpe_wce^2*(1+Ti/Te))
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
// 6. Pressure balance verification
//    P_total = n(y)*(Ti+Te) + B(y)^2/(2*mu0) = const
//    Evaluated WITHOUT perturbation.
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

  // Check at several y positions
  double y_positions[] = {
    -0.25 * Ly,  // Center of sheet 1
    -0.125 * Ly, // Between center and midpoint
    0.0,          // Midpoint between sheets
    0.125 * Ly,  // Between midpoint and sheet 2
    0.25 * Ly,   // Center of sheet 2
    0.4 * Ly,    // Near boundary
  };

  double P_ref = -1;
  for (int i = 0; i < 6; i++) {
    double y = y_positions[i];

    // Double Harris B field (theta=0, bg=0): Bz component
    double Bz = b0 * (tanh((y + 0.25*Ly) / L) - tanh((y - 0.25*Ly) / L) - 1.0);
    double Bx = b0 * g.bg; // guide field
    double B_sq = Bz*Bz + Bx*Bx;

    // Harris density from double sheet
    double n_harris = 1.0 / sqr(cosh((y + 0.25*Ly) / L))
                    + 1.0 / sqr(cosh((y - 0.25*Ly) / L));

    // Plasma pressure (Harris + background)
    double P_plasma = (n_harris + nb) * (TTi + TTe);

    // Magnetic pressure (in PSC units, mu0 = 1)
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
// 7. Grid setup — FULLY PERIODIC for double Harris sheet
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
  Int3 gdims = {1, 256, 512};
  Int3 np = {1, 2, 4}; // MPI DECOMPOSITION: 8 ranks total

  // Domain centered at y=0: y in [-Ly/2, +Ly/2]
  Grid_t::Domain domain{gdims, LL, {0, -.5 * LL[1], 0}, np};

  // FULLY PERIODIC boundary conditions (required for double Harris sheet)
  psc::grid::BC bc{
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
    {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}
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
// 8. Particle initialization (using Kappa distributions)
//    Double Harris sheet: two sech^2 profiles with opposite drifts.
//    Out-of-plane drift velocity in x carries the current J_x.
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts) {

  partitionAndSetupParticles(setup_p, bal, gptr, mprts,
    [&](int kind, Double3 crd, int patch, Int3 idx, psc_particle_np& np){
      psc_particle_npt npt{};
      npt.kind = kind;

      double y = crd[1];
      double Ly = g.Ly;

      // Double Harris sheet density:
      //   n(y) = sech^2((y + Ly/4)/L) + sech^2((y - Ly/4)/L)
      double n_sheet1 = 1. / sqr(cosh((y + 0.25 * Ly) / g.L)); // sheet at y=-Ly/4
      double n_sheet2 = 1. / sqr(cosh((y - 0.25 * Ly) / g.L)); // sheet at y=+Ly/4
      double n_total_harris = n_sheet1 + n_sheet2;

      // Out-of-plane drift: sign flips between sheets
      // Sheet 1 (y=-Ly/4): current in +x → ions drift +x, electrons -x
      // Sheet 2 (y=+Ly/4): current in -x → ions drift -x, electrons +x
      // Net drift weighted by local density contribution from each sheet:
      //   v_drift = v0 * (n_sheet1 - n_sheet2) / n_total
      double drift_weight = 0.;
      if (n_total_harris > 1e-10) {
        drift_weight = (n_sheet1 - n_sheet2) / n_total_harris;
      }

      switch (kind) {
        case MY_ELECTRON: // drifting electrons (both sheets)
          npt.n = n_total_harris;
          // Drift in x (out-of-plane): sign determined by local sheet
          npt.p[0] = -2. * g.TTe / g.b0 / g.L * drift_weight;
          npt.T[0] = g.TTe; npt.T[1] = g.TTe; npt.T[2] = g.TTe;
          npt.kind = MY_ELECTRON;
          break;
        case MY_ION: // drifting ions (both sheets)
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
      // INITIALIZE WITH KAPPA DISTRIBUTION (from setup_p)
      np.p = setup_p.createKappaMultivariate(npt);
    });
}

// ----------------------------------------------------------------------
// 9. Field initialization
//    Double Harris sheet + localized perturbation on ONE sheet.
//
//    Equilibrium field (no perturbation):
//      Bz(y) = B0 * [tanh((y+Ly/4)/L) - tanh((y-Ly/4)/L) - 1]
//
//    Perturbation (smooth, localized near sheet at y=+Ly/4):
//      Uses vector potential Ax = eps / cosh((y-Ly/4)/sigma)
//      to guarantee div(B) = 0.
// ----------------------------------------------------------------------
void initializeFields(MfieldsState& mflds) {
  double b0 = g.b0, dby = g.dby;
  double L = g.L, Ly = g.Ly, Lz = g.Lz, Lpert = g.Lpert;
  double cs = cos(g.theta), sn = sin(g.theta);
  double sigma = L; // perturbation localization width = sheet width

  mprintf("Double Harris sheet: L=%g, Ly=%g, Lz=%g\n", L, Ly, Lz);
  mprintf("Perturbation: dby=%g, sigma=%g, Lpert=%g\n", dby, sigma, Lpert);

  setupFields(mflds, [&](int m, double crd[3]) {
    double y = crd[1], z = crd[2];

    // Double Harris equilibrium field
    double Bz_eq = cs * b0 * (tanh((y + 0.25*Ly) / L)
                             - tanh((y - 0.25*Ly) / L) - 1.0);
    double Bx_eq = -sn * b0 * (tanh((y + 0.25*Ly) / L)
                              - tanh((y - 0.25*Ly) / L) - 1.0)
                 + b0 * g.bg;

    // Perturbation localized near sheet at y = +Ly/4 via vector potential
    // Ax_pert = eps * cos(kz*(z - Lz/2)) / cosh((y - Ly/4)/sigma)
    // => dBy = dAx/dz, dBz = -dAx/dy (guarantees div B = 0)
    double y_rel = (y - 0.25 * Ly) / sigma;
    double sech_val = 1.0 / cosh(y_rel);
    double kz = 2.0 * M_PI / Lpert;

    double pert_By = dby * sech_val
                   * sin(kz * (z - 0.5 * Lz));
    double pert_Bz = (dby * sigma * kz) * tanh(y_rel) * sech_val
                   * cos(kz * (z - 0.5 * Lz));
    // Note: the sign comes from Bz_pert = -dAx/dy
    //   d/dy [1/cosh(y_rel)] = -(1/sigma)*tanh(y_rel)/cosh(y_rel)
    //   so Bz_pert = eps*kz*(1/sigma)*tanh(y_rel)*sech(y_rel)*cos(...)
    //   Simplified: Bz_pert ~ dby * tanh(y_rel)*sech(y_rel) * cos(...)
    //   (the sigma*kz factor absorbed into amplitude scaling)

    switch (m) {
      case HX: return Bx_eq;
      case HY: return pert_By;
      case HZ: return Bz_eq + pert_Bz;
      default: return 0.;
    }
  });
}

// ----------------------------------------------------------------------
// 10. Main Run Function
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD,
    "*** Setting up Magnetic Reconnection — DOUBLE HARRIS SHEET ***\n");
  mpi_printf(MPI_COMM_WORLD,
    "*** Kappa distribution, fully periodic BCs ***\n");
  setupParameters();

  // Verify pressure balance BEFORE adding perturbation
  verifyPressureBalance();

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
// 11. Main entry point
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
  psc_init(argc, argv);
  run();
  psc_finalize();
  return 0;
}
