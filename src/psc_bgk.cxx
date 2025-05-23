
#include <psc.hxx>
#include <setup_fields.hxx>
#include <setup_particles.hxx>

#include "DiagnosticsDefault.h"
#include "OutputFieldsDefault.h"
#include "psc_config.hxx"

#include "psc_bgk_util/bgk_params.hxx"
#include "psc_bgk_util/table.hxx"
#include "psc_bgk_util/params_parser.hxx"

// ======================================================================
// PSC configuration
//
// This sets up compile-time configuration for the code, in particular
// what data structures and algorithms to use
//
// EDIT to change order / floating point type / cuda / 2d/3d

using Dim = dim_yz;
#ifdef USE_CUDA
using PscConfig = PscConfig1vbecCuda<Dim>;
#else
using PscConfig = PscConfig1vbecDouble<Dim>;
#endif

// ----------------------------------------------------------------------

using BgkMfields = PscConfig::Mfields;
using MfieldsState = PscConfig::MfieldsState;
using Mparticles = PscConfig::Mparticles;
using Balance = PscConfig::Balance;
using Collision = PscConfig::Collision;
using Checks = PscConfig::Checks;
using Marder = PscConfig::Marder;
using OutputParticles = PscConfig::OutputParticles;

// ======================================================================
// Global parameters

namespace
{
// table of initial conditions
Table* ic_table;

std::string read_checkpoint_filename;

// Parameters specific to BGK simulation
PscBgkParams g;

// General PSC parameters (see include/psc.hxx),
PscParams psc_params;

} // namespace

// ======================================================================
// setupParameters

void setupParameters(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " path/to/params\nExiting."
              << std::endl;
    exit(1);
  }
  std::string path_to_params(argv[1]);
  ParsedParams parsedParams(path_to_params);
  ic_table = new Table(parsedParams.get<std::string>("path_to_data"));
  g.loadParams(parsedParams, *ic_table);

  psc_params.nmax = parsedParams.get<int>("nmax");
  psc_params.stats_every = parsedParams.get<int>("stats_every");
  psc_params.cfl = parsedParams.getOrDefault<double>("cfl", .75);

  psc_params.write_checkpoint_every_step =
    parsedParams.getOrDefault<int>("checkpoint_every", 0);
  if (parsedParams.getOrDefault<bool>("read_checkpoint", false))
    read_checkpoint_filename =
      parsedParams.get<std::string>("path_to_checkpoint");

  std::ifstream src(path_to_params, std::ios::binary);
  std::ofstream dst("params_record.txt", std::ios::binary);
  dst << src.rdbuf();

  if (g.n_grid_3 > 1 && typeid(Dim) != typeid(dim_xyz)) {
    LOG_ERROR("3D runs require Dim = dim_xyz\n");
    exit(1);
  } else if (g.n_grid == 1 && typeid(Dim) != typeid(dim_yz)) {
    LOG_ERROR("2D runs require Dim = dim_yz\n");
    exit(1);
  }
}

// ======================================================================
// setupGrid
//
// This helper function is responsible for setting up the "Grid",
// which is really more than just the domain and its decomposition, it
// also encompasses PC normalization parameters, information about the
// particle kinds, etc.

Grid_t* setupGrid()
{
  auto domain = Grid_t::Domain{
    {g.n_grid_3, g.n_grid, g.n_grid},           // # grid points
    {g.box_size_3, g.box_size, g.box_size},     // physical lengths
    {0., -.5 * g.box_size, -.5 * g.box_size},   // *offset* for origin
    {g.n_patches_3, g.n_patches, g.n_patches}}; // # patches

  auto bc =
    psc::grid::BC{{BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                  {BND_FLD_PERIODIC, BND_FLD_PERIODIC, BND_FLD_PERIODIC},
                  {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC},
                  {BND_PRT_PERIODIC, BND_PRT_PERIODIC, BND_PRT_PERIODIC}};

  auto kinds = Grid_t::Kinds(NR_KINDS);
  kinds[KIND_ELECTRON] = {g.q_e, g.m_e, "e"};
  kinds[KIND_ION] = {g.q_i, g.m_i, "i"};

  // lambda_D = v_e / omega_pe = v_e = beta
  mpi_printf(MPI_COMM_WORLD, "lambda_D = %g\n", g.beta);

  // --- generic setup
  auto norm_params = Grid_t::NormalizationParams::dimensionless();
  norm_params.nicell = g.nicell;

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
// writeGT

template <typename GT>
void writeGT(const GT& gt, const Grid_t& grid, const std::string& name,
             const std::vector<std::string>& compNames)
{
  WriterMRC writer;
  writer.open(name);
  writer.begin_step(grid.timestep(), grid.timestep() * grid.dt);
  writer.write(gt, grid, name, compNames);
  writer.end_step();
  writer.close();
}

// ----------------------------------------------------------------------
// writeMF

template <typename MF>
void writeMF(MF&& mfld, const std::string& name,
             const std::vector<std::string>& compNames)
{
  writeGT(psc::mflds::interior(mfld.grid(), mfld.gt()), mfld.grid(), name,
          compNames);
}

// ======================================================================
// helper methods

inline double getCoord(double crd)
{
  if (crd < -g.box_size / 2)
    return crd + g.box_size;
  if (crd > g.box_size / 2)
    return crd - g.box_size;
  return crd;
}

template <int LEN>
inline void setAll(double (&vals)[LEN], double newval)
{
  for (int i = 0; i < LEN; i++)
    vals[i] = newval;
}

inline double getIonDensity(double rho)
{
  if (!g.do_ion)
    return g.n_i;
  double potential = ic_table->get_interpolated("Psi", "rho", rho);
  return std::exp(-potential / g.T_i);
}

// ======================================================================
// v_phi_cdf
// Cumulative distribution function for azimuthal electron velocity

double v_phi_cdf(double v_phi, double rho)
{
  double A_phi = ic_table->has_column("A_phi")
                   ? ic_table->get_interpolated("A_phi", "rho", rho)
                   : .5 * g.Hx * rho;

  // convert units from psc to paper
  v_phi /= g.beta;
  rho /= g.beta;
  A_phi /= g.beta;

  double gamma = 1 + 8 * g.k * sqr(rho);
  double alpha =
    1 - g.h0 / std::sqrt(gamma) * std::exp(-4 * g.k * sqr(A_phi * rho) / gamma);

  double mean0 = 0;
  double stdev0 = 1;

  double mean1 = 8 * g.k * A_phi * sqr(rho) / gamma;
  double stdev1 = 1 / std::sqrt(gamma);

  double m0 = (1 + std::erf((v_phi - mean0) / (stdev0 * std::sqrt(2)))) / 2;
  double m1 = (1 + std::erf((v_phi - mean1) / (stdev1 * std::sqrt(2)))) / 2;

  return m0 / alpha + m1 * (1 - 1 / alpha);
}

struct pdist
{
  pdist(double y, double z, double rho)
    : y{y},
      z{z},
      rho{rho},
      v_phi_dist{[=](double v_phi) { return v_phi_cdf(v_phi, rho); }},
      v_rho_dist{0, g.beta},
      v_x_dist{0, g.beta}
  {}

  Double3 operator()()
  {
    double v_phi = v_phi_dist.get();
    double v_rho = v_rho_dist.get();
    double v_x = v_x_dist.get();

    double coef = g.v_e_coef * (g.reverse_v ? -1 : 1) *
                  (g.reverse_v_half && y < 0 ? -1 : 1);
    double p_x = coef * g.m_e * v_x;
    double p_y = coef * g.m_e * (v_phi * -z + v_rho * y) / rho;
    double p_z = coef * g.m_e * (v_phi * y + v_rho * z) / rho;
    return Double3{p_x, p_y, p_z};
  }

private:
  double y, z, rho;
  rng::InvertedCdf<double> v_phi_dist;
  rng::Normal<double> v_rho_dist;
  rng::Normal<double> v_x_dist;
};

struct pdist_case4
{
  pdist_case4(double p_background, double y, double z, double rho)
    : p_background{p_background},
      y{y},
      z{z},
      rho{rho},
      v_phi_dist{g.beta * 8.0 * g.k * sqr(rho / g.beta) *
                   ((ic_table->has_column("A_phi")
                       ? ic_table->get_interpolated("A_phi", "rho", rho)
                       : 0.5 * g.Hx * rho) /
                    g.beta) /
                   (1.0 + 8.0 * g.k * sqr(rho / g.beta)),
                 g.beta / sqrt(1.0 + 8.0 * g.k * sqr(rho / g.beta))},
      v_rho_dist{0, g.beta},
      v_x_dist{g.beta * 2.0 * g.xi * g.A_x0 / (1.0 + 2.0 * g.xi),
               g.beta / sqrt(1.0 + 2.0 * g.xi)},
      simple_dist{0.0, g.beta},
      uniform{0.0, 1.0}
  {}

  Double3 operator()()
  {
    double coef = g.v_e_coef * (g.reverse_v ? -1 : 1) *
                  (g.reverse_v_half && y < 0 ? -1 : 1);
    if (rho == 0) {
      //  p_y and p_z reduce to same case at rho=0, but not p_x
      double p_x =
        coef * g.m_e *
        (uniform.get() < p_background ? simple_dist : v_x_dist).get();
      double p_y = coef * g.m_e * simple_dist.get();
      double p_z = coef * g.m_e * simple_dist.get();
      return Double3{p_x, p_y, p_z};
    }

    double v_phi, v_rho, v_x;

    if (uniform.get() < p_background) {
      v_phi = simple_dist.get();
      v_rho = simple_dist.get();
      v_x = simple_dist.get();
    } else {
      v_phi = v_phi_dist.get();
      v_rho = v_rho_dist.get();
      v_x = v_x_dist.get();
    }

    double p_x = coef * g.m_e * v_x;
    double p_y = coef * g.m_e * (v_phi * -z + v_rho * y) / rho;
    double p_z = coef * g.m_e * (v_phi * y + v_rho * z) / rho;
    return Double3{p_x, p_y, p_z};
  }

private:
  double p_background, y, z, rho;
  rng::Normal<double> v_phi_dist;
  rng::Normal<double> v_rho_dist;
  rng::Normal<double> v_x_dist;
  rng::Normal<double> simple_dist;
  rng::Uniform<double> uniform;
};

// ======================================================================
// initializeParticles

void initializeParticles(Balance& balance, Grid_t*& grid_ptr, Mparticles& mprts,
                         BgkMfields& divGradPhi)
{
  SetupParticles<Mparticles> setup_particles(*grid_ptr);
  setup_particles.centerer = centering::Centerer(centering::NC);

  auto&& qDensity = -psc::mflds::interior(divGradPhi.grid(), divGradPhi.gt());

  auto init_np = [&](int kind, Double3 crd, int p, Int3 idx,
                     psc_particle_np& np) {
    double y = getCoord(crd[1]);
    double z = getCoord(crd[2]);
    double rho = sqrt(sqr(y) + sqr(z));

    double Ti = g.T_i;
    switch (kind) {

      case KIND_ELECTRON:
        np.n = (qDensity(idx[0], idx[1], idx[2], 0, p) -
                getIonDensity(rho) * g.q_i) /
               g.q_e;

        if (g.xi != 0) {
          // case 4: sum of two maxwellians, where one has easy moments and the
          // other's moments are given in input file

          double psi_cs =
            ic_table->get_interpolated("Psi", "rho", rho) / sqr(g.beta);
          double n_background = exp(psi_cs);
          double p_background = n_background / np.n;
          np.p = pdist_case4(p_background, y, z, rho);

          break;
        }

        if (rho == 0) {
          double Te = ic_table->get_interpolated("Te", "rho", rho);
          np.p = setup_particles.createMaxwellian(
            {np.kind, np.n, {0, 0, 0}, {Te, Te, Te}, np.tag});
        } else if (g.maxwellian) {
          double Te = ic_table->get_interpolated("Te", "rho", rho);
          double vphi = ic_table->get_interpolated("v_phi", "rho", rho);
          double coef = g.v_e_coef * (g.reverse_v ? -1 : 1) *
                        (g.reverse_v_half && y < 0 ? -1 : 1);

          double pz = coef * g.m_e * vphi * y / rho;
          double py = coef * g.m_e * -vphi * z / rho;
          np.p = setup_particles.createMaxwellian(
            {np.kind, np.n, {0, py, pz}, {Te, Te, Te}, np.tag});
        } else {
          np.p = pdist(y, z, rho);
        }
        break;

      case KIND_ION:
        np.n = getIonDensity(rho);
        np.p = setup_particles.createMaxwellian(
          {np.kind, np.n, {0, 0, 0}, {Ti, Ti, Ti}, np.tag});
        break;

      default: assert(false);
    }
  };

  partitionAndSetupParticles(setup_particles, balance, grid_ptr, mprts,
                             init_np);
}

// ======================================================================
// fillGhosts

template <typename MF>
void fillGhosts(MF& mfld, int compBegin, int compEnd)
{
  Bnd_ bnd{};
  bnd.fill_ghosts(mfld, compBegin, compEnd);
}

// ======================================================================
// initializePhi

void initializePhi(BgkMfields& phi)
{
  setupScalarField(
    phi, centering::Centerer(centering::NC), [&](int m, double crd[3]) {
      double rho = sqrt(sqr(getCoord(crd[1])) + sqr(getCoord(crd[2])));
      return ic_table->get_interpolated("Psi", "rho", rho);
    });

  writeMF(phi, "phi", {"phi"});
}

// ======================================================================
// initializeGradPhi

void initializeGradPhi(BgkMfields& phi, BgkMfields& gradPhi)
{
  auto&& grad = psc::item::grad_ec(phi.gt(), phi.grid());
  psc::mflds::interior(gradPhi.grid(), gradPhi.storage()) = grad;

  fillGhosts(gradPhi, 0, 3);

  writeMF(gradPhi, "grad_phi", {"gradx", "grady", "gradz"});
}

// ======================================================================
// initializeDivGradPhi

void initializeDivGradPhi(BgkMfields& gradPhi, BgkMfields& divGradPhi)
{
  auto&& divGrad = psc::item::div_nc(gradPhi.grid(), gradPhi.gt());
  psc::mflds::interior(divGradPhi.grid(), divGradPhi.storage()) = divGrad;

  fillGhosts(divGradPhi, 0, 1);

  writeMF(divGradPhi, "div_grad_phi", {"divgrad"});
}

// ======================================================================
// initializeFields

void initializeFields(MfieldsState& mflds, BgkMfields& gradPhi)
{
  if (ic_table->has_column("B_x")) {
    setupFields(mflds, [&](int m, double crd[3]) {
      double y = getCoord(crd[1]);
      double z = getCoord(crd[2]);
      double rho = sqrt(sqr(y) + sqr(z));
      switch (m) {
        case HX: return ic_table->get_interpolated("B_x", "rho", rho);
        default: return 0.;
      }
    });
  } else {
    setupFields(mflds, [&](int m, double crd[3]) {
      switch (m) {
        case HX: return g.Hx;
        default: return 0.;
      }
    });
  }

  // initialize E separately
  mflds.storage().view(_all, _all, _all, _s(EX, EX + 3)) = -gradPhi.gt();
}

// ======================================================================
// run

static void run(int argc, char** argv)
{
  mpi_printf(MPI_COMM_WORLD, "*** Setting up...\n");

  // ----------------------------------------------------------------------
  // setup various parameters first

  setupParameters(argc, argv);

  // ----------------------------------------------------------------------
  // Set up grid, state fields, particles

  auto grid_ptr = setupGrid();
  auto& grid = *grid_ptr;
  MfieldsState mflds{grid};
  Mparticles mprts{grid};
  BgkMfields phi{grid, 1, mflds.ibn()};
  BgkMfields gradPhi{grid, 3, mflds.ibn()};
  BgkMfields divGradPhi{grid, 1, mflds.ibn()};

  // ----------------------------------------------------------------------
  // Set up various objects needed to run this case

  // -- Balance
  psc_params.balance_interval = 0;
  Balance balance{.1};

  // -- Sort
  psc_params.sort_interval = 10;

  // -- Collision
  int collision_interval = 0;
  double collision_nu = .1;
  Collision collision{grid, collision_interval, collision_nu};

  // -- Checks
  ChecksParams checks_params{};
  checks_params.gauss.check_interval = g.gauss_every;
  // checks_params.gauss.dump_always = true;
  checks_params.gauss.err_threshold = 1e-5;

  Checks checks{grid, MPI_COMM_WORLD, checks_params};

  // -- Marder correction
  double marder_diffusion = 0.9;
  int marder_loop = 3;
  bool marder_dump = false;
  psc_params.marder_interval = 5;
  Marder marder(grid, marder_diffusion, marder_loop, marder_dump);

  // ----------------------------------------------------------------------
  // Set up output
  //
  // FIXME, this really is too complicated and not very flexible

  // -- output fields
  OutputFieldsParams outf_params{};
  outf_params.fields.pfield.out_interval = g.fields_every;
  outf_params.moments.pfield.out_interval = g.moments_every;
  OutputFields<MfieldsState, Mparticles, Dim> outf{grid, outf_params};

  // -- output particles
  OutputParticlesParams outp_params{};
  outp_params.every_step = g.particles_every;
  outp_params.data_dir = ".";
  outp_params.basename = "prt";
  OutputParticles outp{grid, outp_params};

  int oute_interval = -100;
  DiagEnergies oute{grid.comm(), oute_interval};

  auto diagnostics = makeDiagnosticsDefault(outf, outp, oute);

  // ----------------------------------------------------------------------
  // Set up initial conditions

  if (read_checkpoint_filename.empty()) {
    initializePhi(phi);
    initializeGradPhi(phi, gradPhi);
    initializeDivGradPhi(gradPhi, divGradPhi);
    initializeParticles(balance, grid_ptr, mprts, divGradPhi);
    initializeFields(mflds, gradPhi);
  } else {
    read_checkpoint(read_checkpoint_filename, *grid_ptr, mprts, mflds);
  }

  delete ic_table;

  // ----------------------------------------------------------------------
  // Hand off to PscIntegrator to run the simulation

  auto psc =
    makePscIntegrator<PscConfig>(psc_params, *grid_ptr, mflds, mprts, balance,
                                 collision, checks, marder, diagnostics);

  psc.integrate();
}

// ======================================================================
// main

int main(int argc, char** argv)
{
  // psc_init(argc, argv);
  // FIXME restore whatever previous functionality there was with options
  int temp = 1;
  psc_init(temp, argv);

  run(argc, argv);

  psc_finalize();
  return 0;
}
