#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include <cstdlib>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include "../libpsc/psc_heating/psc_heating_impl.hxx"

// ======================================================================
// Particle kinds

enum
{
  MY_ELECTRON,
  MY_ION,
  N_MY_KINDS,
};

// ======================================================================
// PscFlatfoilParams

struct PscFlatfoilParams
{
  double BB;
  double Zi;
  double mass_ratio;
  double lambda0;

  double vA_over_c;
  double beta_e_par;
  double beta_i_par;
  double Ti_perp_over_Ti_par;
  double Te_perp_over_Te_par;
  double n;
  double omegape_Omegae;

  // calculated from the above
  double B0;
  double Te_par;
  double Te_perp;
  double Ti_par;
  double Ti_perp;
  double mi;
  double me;

  double d_i;

  // turbulence / Alfvenic fluctuations
  // ======================================================================
  double dB_B0;        // rms fluctuation level
  int kmin;
  int kmax;
  int nmodes;
  bool init_alfvenic_velocity;
  double spectral_index;
  unsigned turbulence_seed;
  // ======================================================================


};

// ======================================================================
// Global parameters

namespace
{

PscFlatfoilParams g;

std::string read_checkpoint_filename;

PscParams psc_params;

} // namespace

// ======================================================================
// PSC configuration

using Dim = dim_xyz;

using PscConfig = PscConfig1vbecSingle<Dim>;

using Writer = WriterDefault; // can choose WriterMrc, WriterAdios2

// ======================================================================

using MfieldsState = PscConfig::MfieldsState;
using Mparticles = PscConfig::Mparticles;
using Balance = PscConfig::Balance;
using Collision = PscConfig::Collision;
using Checks = PscConfig::Checks;
using Marder = PscConfig::Marder;
using OutputParticles = PscConfig::OutputParticles;

// ======================================================================

// setupParameters

void setupParameters()
{
  psc_params.nmax = 30000;
  psc_params.cfl = 0.95;
  psc_params.write_checkpoint_every_step = 100;
  psc_params.stats_every = 50;

  //read_checkpoint_filename = "checkpoint_3000.bp";

  g.BB = 1.0;
  g.Zi = 1.;
  g.mass_ratio = 49.;
  g.lambda0 = 20.;

  // General Parameters
  g.omegape_Omegae = 2.;
  g.vA_over_c = 0.08;//(1/g.omegape_Omegae)/g.mass_ratio; //This is to make sure there are no relativistic signals
  g.beta_e_par = 1.0;
  g.beta_i_par = 1.0; //considering hotter ions than electrons
  g.Ti_perp_over_Ti_par = 1.0;
  g.Te_perp_over_Te_par = 1.0;
  g.n = 1.0;

  g.mi = g.mass_ratio * g.Zi;
  g.me = 1.0;

  g.B0 = g.vA_over_c;
  g.Te_par = g.beta_e_par * sqr(g.B0) / 2.;
  g.Te_perp = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par = g.beta_i_par * sqr(g.B0) / 2.;
  g.Ti_perp = g.Ti_perp_over_Ti_par * g.Ti_par;

  // Alfvenic fluctuation parameters
  // ======================================================================
  g.dB_B0 = 0.8;          // δB_rms / B0
  g.kmin = 2;
  g.kmax = 6;
  g.nmodes = 64;
  g.init_alfvenic_velocity = true;
  g.spectral_index = 5./3.;
  g.turbulence_seed = 1235;
  // ======================================================================

}

// ======================================================================
// setupGrid

Grid_t* setupGrid()
{
  g.d_i = std::sqrt(g.mass_ratio / g.n);

  // Dominio de 32 d_i;  d_i = sqrt(200) ≈ 14.14 celdas → 128 celdas ≈ 0.22 d_i/celda
  double domain_size = 2.0 * M_PI * 3. * g.d_i;

  Grid_t::Real3 LL = {domain_size, domain_size, domain_size};
  Int3         gdims = {640, 640, 644};
  Int3         np    = {5, 8, 28}; // 32 patches for 32 MPI ranks

  Grid_t::Domain domain{gdims, LL, -.5 * LL, np};

  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION] = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  mpi_printf(MPI_COMM_WORLD, "d_e = %g, d_i = %g\n", 1., g.d_i);
  mpi_printf(MPI_COMM_WORLD, "lambda_De (background) = %g\n",
             sqrt(g.Te_perp));

  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = 400; // ppc

  double dt = psc_params.cfl * courant_length(domain);
  Grid_t::Normalization norm{norm_params};

  Int3 ibn = {2, 2, 2};
  if (Dim::InvarX::value) {
    ibn[0] = 0;
  }
  if (Dim::InvarY::value) {
    ibn[1] = 0;
  }
  if (Dim::InvarZ::value) {
    ibn[2] = 0;
  }

  return new Grid_t{domain, bc, kinds, norm, dt, -1, ibn};
}


// ======================================================================
#include <random>
#include <vector>

struct AlfvenMode
{
  double ky, kz;

  double phase;
  double amp;
  double phase_u;
  double amp_u;

};

std::vector<AlfvenMode> g_modes;
// ======================================================================



// Create spectrum
// ======================================================================
void setupAlfvenModes(Grid_t& grid)
{
  std::mt19937 rng(g.turbulence_seed);

  std::uniform_real_distribution<double> uni(0., 2. * M_PI);

  g_modes.clear();

  double Ly = grid.domain.length[1];
  double Lz = grid.domain.length[2];

  double total_power = 0.0;

  // --- build modes

  for (int n = 0; n < g.nmodes; n++) {

    int nny = g.kmin + rng() % (g.kmax - g.kmin + 1);
    int nnz = g.kmin + rng() % (g.kmax - g.kmin + 1);

    // avoid k=0
    if (nny == 0 && nnz == 0)
      continue;

    double ky = 2. * M_PI * nny / Ly;
    double kz = 2. * M_PI * nnz / Lz;

    double kperp = std::sqrt(ky * ky + kz * kz);

    // power-law spectrum
    double Pk = std::pow(kperp, -g.spectral_index);

    total_power += Pk;

    AlfvenMode mode{};

    mode.ky = ky;
    mode.kz = kz;

    mode.phase = uni(rng);
    mode.amp = Pk;

    mode.phase_u = uni(rng);
    mode.amp_u = mode.amp;


    g_modes.push_back(mode);
  }

  // --- normalize total fluctuation energy

  double target = sqr(g.dB_B0 * g.B0);

  double norm = std::sqrt(target / total_power);

  for (auto& m : g_modes) {
    m.amp *= norm;
    m.amp_u *= norm;
  }
}
// ======================================================================




// ======================================================================
// initializeParticles

void initializeParticles(SetupParticles<Mparticles>& setup_particles,
                         Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts)
{
  partitionAndSetupParticles(
		  setup_particles, balance, grid_ptr, mprts,
                             [&](int kind, Double3 pos, int patch, Int3 idx, psc_particle_np& np) {
			     psc_particle_npt npt{};
			     npt.kind = kind;
			     double y = pos[1];
                              double z = pos[2];

                              //double dBx = 0.;
                              double dBy = 0.;
                              double dBz = 0.;
                              double Jx = 0.;
                              double uy = 0.;
                              double uz = 0.;

                              for (auto& mode : g_modes) {

                                
				double phase_u =
  					mode.ky * y +
  					mode.kz * z +
  					mode.phase; //same phase instead of phase_u

				double su = std::sin(phase_u);

				double kperp = sqrt(mode.ky * mode.ky +
                				mode.kz * mode.kz);


				double u1 = -(mode.kz/kperp) * mode.amp_u * su;

				double u2 = (mode.ky/kperp) * mode.amp_u * su;

				uy += u1;
				uz += u2;

                                double phase =
                                  mode.ky * y +
                                  mode.kz * z +
                                  mode.phase;

                                double c = std::cos(phase);
                                double s = std::sin(phase);

                                double tmpBy =
                                  -mode.kz * mode.amp * s;

                                double tmpBz =
                                   mode.ky * mode.amp * s;

                                dBy += tmpBy;
                                dBz += tmpBz;

                                double k2 =
                                  mode.ky * mode.ky +
                                  mode.kz * mode.kz;

                                Jx += k2 * mode.amp * c;


                              }

                              double rho0 =
                              g.n * g.mi +
                              g.n * g.me;

                              double fac = 1.0 / std::sqrt(rho0);

                               switch (kind) {
                                 case MY_ION:
                                   npt.n = g.n;
                                   npt.p[0] =  0.;//fac * dBx;
                                   npt.p[1] =  uy;//fac * dBy;
                                   npt.p[2] =  uz;//fac * dBz;
                                   npt.T[0] = g.Ti_par;
                                   npt.T[1] = g.Ti_perp;
                                   npt.T[2] = g.Ti_perp;
                                   break;

                                 case MY_ELECTRON:
                                   npt.n = g.n;                                   
                                   npt.p[0] = -Jx / g.n;// fac * dBx;
                                   npt.p[1] =  uy;//fac * dBy;
                                   npt.p[2] =  uz;//fac * dBz;                                  
                                   npt.T[0] = g.Te_par;
                                   npt.T[1] = g.Te_perp;
                                   npt.T[2] = g.Te_perp;
                                   break;
                                 default: assert(0);
                               }
			       np.n = npt.n;
			       np.p = setup_particles.createKappaMultivariate(npt);
                             });
}

// ======================================================================
// initializeFields


void initializeFields(MfieldsState& mflds)
{
  auto& grid = mflds.grid();

  //setupAlfvenModes(grid);

  setupFields(mflds, [&](int m, double crd[3]) {

    double y = crd[1];
    double z = crd[2];

    double Bx = g.B0;
    double By = 0.;
    double Bz = 0;

    double Jx = 0.;
    double Jy = 0.;
    double Jz = 0.;

    for (auto& mode : g_modes) {

      double phase =
        mode.ky * y +
        mode.kz * z +
        mode.phase;

      double c = std::cos(phase);
      double s = std::sin(phase);

      // ------------------------------------------
      // EXACTLY divergence-free Alfvenic mode
      //
      //
      // exactly in 2D yz geometry
      // ------------------------------------------

      // --------------------------------------
      // vector potential:
      //
      // A_x = amp * cos(phase)
      //
      // B = curl(A_x ex)
      // --------------------------------------

      double kperp = sqrt(mode.ky * mode.ky +
       		mode.kz * mode.kz);

      double dBy = -(mode.kz / kperp) *
   		mode.amp * s;

      double dBz = (mode.ky / kperp) *
   		mode.amp * s;


      By += dBy;
      Bz += dBz;

      // --------------------------------------
      // Jx = curl(B)_x
      //
      // Jx = dBz/dy - dBy/dz
      //     = k_perp^2 A_x
      // --------------------------------------

      double k2 =
        mode.ky * mode.ky +
        mode.kz * mode.kz;

      Jx += k2 * mode.amp * c;


    }

    switch (m) {

    case HX: return Bx;
    case HY: return By;
    case HZ: return Bz;

    case JXI: return Jx;
    case JYI: return Jy;
    case JZI: return Jz;

    default:
      return 0.;
    }
  });
}



//void initializeFields(MfieldsState& mflds)
//{
//  setupFields(mflds, [&](int m, double crd[3]) {
//    switch (m) {
//      case HZ: return g.B0;
//      default: return 0.;
//    }
//  });
//}

// ======================================================================
// run

void run()
{
  mpi_printf(MPI_COMM_WORLD, "*** Setting up Kappa turbulence...\n");

  setupParameters();

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if (!read_checkpoint_filename.empty()) {
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);
  }

  psc_params.balance_interval = 1000;
  Balance balance{3};

  psc_params.sort_interval = 10;

  int collision_interval = -10;
  double collision_nu =
    3.76 * std::pow(g.Te_perp, 2.) / g.Zi / g.lambda0;
  Collision collision{grid, collision_interval, collision_nu};

  ChecksParams checks_params{};
  checks_params.continuity.check_interval = 0;
  checks_params.continuity.err_threshold = 1e-4;
  checks_params.continuity.print_max_err_always = true;
  checks_params.continuity.dump_always = false;
  checks_params.gauss.check_interval = 100;
  checks_params.gauss.err_threshold = 1e-4;
  checks_params.gauss.print_max_err_always = true;
  checks_params.gauss.dump_always = false;
  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  double marder_diffusion = 0.9;
  int marder_loop = 3;
  bool marder_dump = false;
  psc_params.marder_interval = 100;
  Marder marder(grid, marder_diffusion, marder_loop, marder_dump);

  OutputFieldsItemParams outf_item_params{};
  OutputFieldsParams outf_params{};
  outf_item_params.pfield.out_interval = 200;
  outf_item_params.tfield.out_interval = -1000;
  outf_item_params.tfield.average_every = -1000;
  outf_params.fields = outf_item_params;
  outf_params.moments = outf_item_params;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, outf_params};

  OutputParticlesParams outp_params{};
  outp_params.every_step = 1000;
  outp_params.data_dir = ".";
  outp_params.basename = "prt_turbulence_3D_kappa_3";
  // Save only particles from the central 8×8 d_i region (cells 48–80)
  // to reduce storage from ~577 GB to ~36 GB over 200 snapshots
  outp_params.lo = {100, 100, 100};
  outp_params.hi = {200, 200, 200};
  OutputParticles outp{grid, outp_params};

  int oute_interval = -100;
  DiagEnergies oute{grid.comm(), oute_interval};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_particles(grid);
  setup_particles.kappa = 3.0;
  setup_particles.fractional_n_particles_per_cell = true;
  setup_particles.neutralizing_population = MY_ION;

//  if (read_checkpoint_filename.empty()) {
//    initializeParticles(setup_particles, balance, grid_ptr, mprts);
//    initializeFields(mflds);
//  }

  if (read_checkpoint_filename.empty()) {
  // build turbulence realization ONCE
  setupAlfvenModes(*grid_ptr);
  initializeParticles(setup_particles, balance, grid_ptr, mprts);
  initializeFields(mflds);
  }

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

  if (const char* restart = std::getenv("PSC_RESTART")) {
    read_checkpoint_filename = restart;
  }

  run();

  psc_finalize();
  return 0;
}
