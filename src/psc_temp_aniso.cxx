// ======================================================================
// psc_temp_aniso.cxx
// Simulation of Ion Mirror Instability in Collisionless Plasma.
//
// Configuration: mass_ratio=200, lambda_De resolved (>=3 pts/cell),
// 72 MPI ranks (feynman-00, np{1,8,9}), BP5+ZFP output,
// SLURM streaming pipeline (purge_and_analyze.slurm).
// ======================================================================
//
// GRID PHYSICS NOTES:
//   vA/c = 0.10  =>  B0 = 0.10
//   beta_e_par = 1.0  =>  Te_par = beta_e * B0^2/2 = 0.005
//   lambda_De  = sqrt(Te_perp) = sqrt(0.005) ~ 0.0707  [c/wpe]
//   Delta_cell = 0.023 < lambda_De/3 = 0.0236  => 3.07 pts/lambda_De  [OK]
//   d_i        = sqrt(mass_ratio) = sqrt(200) ~ 14.14  [c/wpe]
//   Omega_i    = B0/mi = 0.1/200 = 5e-4  [wpe]
//   dt_CFL     ~ 0.95 * 0.023 / sqrt(2) ~ 0.01545  [1/wpe]
//   nmax=800000 covers t ~ 12360/wpe ~ 6.2 Omega_i^{-1}  (ion gyro periods)
//
// STORAGE BUDGET (7 GB on /scratchsan):
//   Grid: 4096x4608 cells, float32, 6 EM vars -> 452 MB/snapshot raw
//   ZFP compression (accuracy=1e-4): ~90 MB/snapshot compressed
//   Fields out every 1000 steps -> 800 snapshots -> ~72 GB raw -> ~14 GB ZFP
//   *** SLURM pipeline (purge_and_analyze) runs AFTER sim, processes all
//       snapshots at once and deletes BP5. Peak disk = full run output.
//   With pipeline keeping <80 snapshots on disk at a time -> ~7 GB peak.
//   Checkpoints: 1 at a time (lossless bzip2, ~4 GB each), deleted after new.
//   Particles: 10 snapshots total, subsampled 0.5% -> negligible.
// ======================================================================

#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"
#include "writer_adios2.hxx"
#include <libpsc/psc_heating/psc_heating_impl.hxx>
#include "heating_spot_foil.hxx"

// ----------------------------------------------------------------------
// 1. Particle kinds
// Define the species involved in the simulation.
// ----------------------------------------------------------------------
enum { MY_ELECTRON, MY_ION, N_MY_KINDS };

// ----------------------------------------------------------------------
// 2. Simulation parameters
// Global physical parameters for the mirror instability setup.
// ----------------------------------------------------------------------
struct PscFlatfoilParams {
  double BB, Zi, mass_ratio, lambda0;
  double vA_over_c, beta_e_par, beta_i_par;
  double Ti_perp_over_Ti_par, Te_perp_over_Te_par, n;
  
  // Derived quantities
  double B0, Te_par, Te_perp, Ti_par, Ti_perp, mi, me, d_i;
};

static PscFlatfoilParams g;
static std::string read_checkpoint_filename;
static PscParams psc_params;

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

using Writer = WriterADIOS2;
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
// SEGMENTED PIPELINE DESIGN:
//   The full run (800 000 steps) is split into 12 segments of 70 000 steps.
//   Each segment writes 70 field snapshots × ~90 MB ZFP = ~6.3 GB < 7 GB.
//   The SLURM pipeline:
//     1. Runs sim segment (sets PSC_NMAX = next boundary)
//     2. Runs purge_and_analyze.slurm (analyze + delete BP5) IN PARALLEL
//        with the NEXT simulation segment (--dependency=after, not afterok)
//     3. Cycle repeats until step 800 000.
//   PSC_NMAX env var sets the end step for each segment without recompiling.
//   write_checkpoint_every_step = -70000 ensures checkpoint at segment end.
void setupParameters() {
  // Default nmax (full run). Overridden per segment by PSC_NMAX env var.
  psc_params.nmax = 800000;
  const char* env_nmax = std::getenv("PSC_NMAX");
  if (env_nmax && std::atoi(env_nmax) > 0) {
    psc_params.nmax = std::atoi(env_nmax);
    mpi_printf(MPI_COMM_WORLD,
               "[PSC_NMAX] Overriding nmax: %d (from env)\n",
               psc_params.nmax);
  }

  psc_params.cfl = 0.95;
  // Checkpoint EVERY 70 000 steps (= segment length), keep only latest.
  // Ensures a valid restart point at every segment boundary.
  psc_params.write_checkpoint_every_step = -70000;
  psc_params.stats_every = 500;

  g.BB = 1.0;
  g.Zi = 1.0;
  g.mass_ratio = 200.0; // Realistic mass ratio (proton/electron)
  g.lambda0    = 20.0;

  // Physical parameters targeting ion mirror instability
  g.vA_over_c           = 0.10;
  g.beta_e_par          = 1.0;   // lambda_De = sqrt(0.005) ~ 0.0707
  g.beta_i_par          = 10.0;  // High beta: strong mirror drive
  g.Ti_perp_over_Ti_par = 3.5;   // T_perp > T_par => mirror unstable
  g.Te_perp_over_Te_par = 1.0;   // Isotropic electrons
  g.n = 1.0;

  // Derived quantities
  g.B0      = g.vA_over_c;
  g.Te_par  = g.beta_e_par  * sqr(g.B0) / 2.0;
  g.Te_perp = g.Te_perp_over_Te_par * g.Te_par;
  g.Ti_par  = g.beta_i_par  * sqr(g.B0) / 2.0;
  g.Ti_perp = g.Ti_perp_over_Ti_par * g.Ti_par;
  g.mi      = g.mass_ratio;
  g.me      = 1.0 / g.mass_ratio;
}

// ----------------------------------------------------------------------
// 6. Grid setup — 72 MPI cores, lambda_De >= 3 points
// ----------------------------------------------------------------------
// RESOLUTION ANALYSIS (mass_ratio = 200):
//   lambda_De = sqrt(Te_perp) = sqrt(0.005) ~ 0.0707  [c/wpe]
//   Delta_cell = LL_Y/NY = 94.208/4096 = 0.023003
//   pts per lambda_De = 0.0707/0.023 = 3.07  ✓ (>= 3)
//
//   d_i = sqrt(200) ~ 14.14  [c/wpe]
//   Domain Y: 94.2 c/wpe = 6.66 d_i
//   Domain Z: 105.98 c/wpe = 7.49 d_i
//
//   MPI layout: np{1,8,9} = 72 ranks, each patch 512x512 cells
// -----------------------------------------------------------------------
Grid_t* setupGrid() {
  // Physical domain in units of c/omega_pe
  // Delta_cell = 0.023 => lambda_De/Delta_cell = 3.07  (RESOLVED)
  const double DCELL = 0.023003;
  const int    NY    = 4096;    // divisible by 8 patches -> 512x512 each
  const int    NZ    = 4608;    // divisible by 9 patches -> 512x512 each

  Grid_t::Real3 LL{1., NY * DCELL, NZ * DCELL};  // {1, 94.208, 105.997}
  Int3 gd{1, NY, NZ};
  Int3 np{1, 8, 9};   // 72 MPI ranks

  Grid_t::Domain dom{gd, LL, -.5*LL, np};
  psc::grid::BC bc{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                   {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  Grid_t::Kinds kinds(N_MY_KINDS);
  kinds[MY_ION]      = {g.Zi, g.mass_ratio * g.Zi, "i"};
  kinds[MY_ELECTRON] = {-1., 1., "e"};

  g.d_i = sqrt(kinds[MY_ION].m / kinds[MY_ION].q);

  const double lambda_De = sqrt(g.Te_perp);
  const double dcell_y   = LL[1] / NY;
  mpi_printf(MPI_COMM_WORLD, "=== Grid Resolution ===\n");
  mpi_printf(MPI_COMM_WORLD, "  d_e = 1.0, d_i = %g\n", g.d_i);
  mpi_printf(MPI_COMM_WORLD, "  lambda_De = %g\n", lambda_De);
  mpi_printf(MPI_COMM_WORLD, "  Delta_cell = %g\n", dcell_y);
  mpi_printf(MPI_COMM_WORLD, "  pts/lambda_De = %g  (need >= 3)\n",
             lambda_De / dcell_y);
  mpi_printf(MPI_COMM_WORLD, "  Domain: %.1f x %.1f c/wpe  =  %.2f x %.2f d_i\n",
             LL[1], LL[2], LL[1]/g.d_i, LL[2]/g.d_i);

  auto npn = Grid_t::NormalizationParams::dimensionless();
  npn.nicell = 50;  // 50 ppc -- feasible with high-RAM server (~45 GB for particles)

  double dt = psc_params.cfl * courant_length(dom);
  mpi_printf(MPI_COMM_WORLD, "  dt = %g  =>  nmax=%d covers t=%.0f/wpe = %.2f/Omega_i\n",
             dt, psc_params.nmax,
             dt * psc_params.nmax,
             dt * psc_params.nmax * (g.BB / g.mi));
  Grid_t::Normalization norm{npn};

  Int3 ibn{2, 2, 2};
  if (Dim::InvarX::value) ibn[0] = 0;
  if (Dim::InvarY::value) ibn[1] = 0;
  if (Dim::InvarZ::value) ibn[2] = 0;

  return new Grid_t{dom, bc, kinds, norm, dt, -1, ibn};
}

// ----------------------------------------------------------------------
// 7. Particle initialization
// Loads particles with the specified bi-Maxwellian velocity distributions.
// ----------------------------------------------------------------------
void initializeParticles(SetupParticles<Mparticles>& setup_p,
                         Balance& bal, Grid_t*& gptr, Mparticles& mprts) {
  partitionAndSetupParticles(setup_p, bal, gptr, mprts,
    [&](int kind, Double3 pos, int patch, Int3 idx, psc_particle_np& np){
      psc_particle_npt npt{};
      npt.kind = kind;
      if(kind == MY_ION) {
        npt.n = g.n;
        npt.T[0] = g.Ti_perp; npt.T[1] = g.Ti_perp; npt.T[2] = g.Ti_par;
      } else {
        npt.n = g.n;
        npt.T[0] = g.Te_perp; npt.T[1] = g.Te_perp; npt.T[2] = g.Te_par;
      }
      np.n = npt.n;
      np.p = setup_p.createKappaMultivariate(npt);
    });
}

// ----------------------------------------------------------------------
// 8. Field initialization
// Background magnetic field B0 must be set along the Z-axis (HZ) 
// to allow k_parallel dynamics in the Y-Z plane.
// ----------------------------------------------------------------------
static constexpr int B_DIR = HZ; 

void initializeFields(MfieldsState& mflds) {
  setupFields(mflds, [&](int m, double[3]){ return m == B_DIR ? g.B0 : 0.; });
}

// ----------------------------------------------------------------------
// 9. Main Run Function
// ----------------------------------------------------------------------
void run() {
  mpi_printf(MPI_COMM_WORLD, "*** Setting up Mirror Instability Simulation...\n");
  setupParameters();
  auto grid_ptr = setupGrid(); 
  auto& grid = *grid_ptr;

  Mparticles mprts(grid);
  MfieldsState mflds(grid);
  if(!read_checkpoint_filename.empty())
    read_checkpoint(read_checkpoint_filename, grid, mprts, mflds);

  // Balance every 2000 steps (expensive with large grid -- do rarely)
  psc_params.balance_interval = 2000;
  Balance bal{3};
  psc_params.sort_interval = 50;

  double nu = 3.76 * pow(g.Te_perp, 2.) / g.Zi / g.lambda0;
  Collision coll{grid, -10, nu};

  ChecksParams chkp{};
  Checks checks{grid, MPI_COMM_WORLD, chkp};

  psc_params.marder_interval = 500;
  Marder marder(grid, 0.9, 3, false);

  // ── Output configuration ──────────────────────────────────────────────
  // Storage budget: 7 GB on /scratchsan. With ZFP (accuracy=1e-4 ~5:1):
  //
  //   Fields (6 EM vars, 4096×4608, float32):
  //     Raw : 452 MB/snapshot
  //     ZFP : ~90 MB/snapshot
  //     Every 1000 steps → 800 snapshots → ~72 GB raw → ~14 GB ZFP
  //     SLURM pipeline streams & deletes; peak on disk ≈ 80 snaps ≈ 7.2 GB
  //
  //   Moments (every 4000 steps → 200 snapshots, ~60 MB ZFP each → ~12 GB)
  //     DISABLED (tfield) — only pfd_moments every 4000 steps if needed.
  //
  //   Particles: 10 snapshots (every 80000 steps), 0.5% subsample → tiny.
  //
  //   Checkpoints: rolling — only 1 kept at a time by purge script.
  //     Each checkpoint ~4 GB (lossless bzip2) — written every 200000 steps.
  // -------------------------------------------------------------------------
  OutputFieldsItemParams ofip_fields{};
  ofip_fields.pfield.out_interval  = 1000; // 800 snapshots total
  ofip_fields.tfield.out_interval  = -1;   // time-averaged fields: DISABLED
  ofip_fields.tfield.average_every = 500;

  OutputFieldsItemParams ofip_moments{};
  ofip_moments.pfield.out_interval = 4000; // 200 snapshots of moments
  ofip_moments.tfield.out_interval = -1;   // DISABLED

  OutputFieldsParams ofp{};
  ofp.fields   = ofip_fields;
  ofp.moments  = ofip_moments;
  OutputFields<MfieldsState, Mparticles, Dim, Writer> outf{grid, ofp};

  // Particles: 10 snapshots across the full run (every 80000 steps).
  // purge_and_analyze.slurm subsamples 0.5% → compact HDF5, deletes .bp.
  OutputParticlesParams opp{};
  opp.every_step = 80000;
  opp.data_dir   = ".";
  opp.basename   = "prt";
  OutputParticles outp{grid, opp};

  DiagEnergies oute{grid.comm(), -500};
  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  SetupParticles<Mparticles> setup_p(grid);
  setup_p.kappa = 3.0; // The parameter kappa
  setup_p.fractional_n_particles_per_cell = true;
  setup_p.neutralizing_population = MY_ION;

  auto mf_n = make_MfieldsMoment_n<Moment_n>(grid);

  if(read_checkpoint_filename.empty()) {
    initializeParticles(setup_p, bal, grid_ptr, mprts);
    initializeFields(mflds);
  }

  auto integrator = makePscIntegrator<PscConfig>(psc_params,
    *grid_ptr, mflds, mprts,
    bal, coll, checks, marder, diagnostics);
    
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
