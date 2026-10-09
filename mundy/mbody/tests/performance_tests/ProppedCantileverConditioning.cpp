// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

/// \file
/// \brief The propped cantilever's Schur operator against its number of segments, unpreconditioned and preconditioned
/// by its self-mobility Jacobi or algebraic multigrid: the condition number, and the CG iterations of its solves.
///
/// Usage: ProppedCantileverConditioning --num-segments N [--conditioning] [--iterations] [--preconditioner
///          none|jacobi|amg] [--dt DT] [--links fixed_length|stiff_springs] [--lanczos-tol TOL]
///          [--lanczos-max-iters M] [--steps S] [--warm-start] [--cg-tol TOL] [--cg-max-iters M]
///          [--coarse-max-size C] [--max-levels L] [--smoother jacobi|sgs|chebyshev] [--prefix P] [--timings]
///
/// The cantilever is that of ProppedCantileverMatchesHenckyBarChain, and its Schur operator is A = dt B^T M B + K^-1.
/// - --conditioning bounds the spectrum of P A at the initial configuration by Lanczos, which must converge, and appends
///   a row to <prefix>_conditioning.csv.
/// - --iterations solves A x = b by preconditioned CG at the start of each of S + 1 steps, b = psi + dt B^T u_free being
///   a step's first right-hand side, from x = 0 or, with --warm-start, from the last step's solution; the steps between
///   are taken by the integrator. It appends a row per step to <prefix>_iterations.csv.
/// --timings adds the setup and solve times.

// External
#include <Kokkos_Core.hpp>
#include <stk_util/parallel/Parallel.hpp>

// C++ core
#include <algorithm>    // for std::max
#include <chrono>       // for std::chrono::steady_clock
#include <fstream>      // for std::ofstream, std::ifstream
#include <iomanip>      // for std::setprecision
#include <iostream>     // for std::cout, std::cerr
#include <sstream>      // for std::ostringstream
#include <stdexcept>    // for std::invalid_argument, std::runtime_error
#include <string>       // for std::string, std::stod, std::stoul
#include <type_traits>  // for std::is_same_v
#include <utility>      // for std::make_pair

// Mundy
#include <MundyMath_config.hpp>                        // for HAVE_MUNDYMATH_{MUELU,TPETRA,KOKKOSKERNELS}
#include <mundy_math/eigenvalues.hpp>                  // for mundy::{make_lanczos_state, solve_eigen_problem, ...}
#include <mundy_math/linear_system.hpp>                // for mundy::{CGConfig, make_linear_system, solve_linear_system}
#include <mundy_math/pgd.hpp>                          // for mundy::PGDConfig
#include <mundy_math/residuals.hpp>                    // for mundy::RelativeL2Residual
#include <mundy_mbody/KokkosMbody.hpp>                 // for mundy::mbody::{RodViews, make_constraint_set, ...}
#include <mundy_mbody/KokkosMbodyPreconditioners.hpp>  // for mundy::mbody::{SelfMobilityJacobi, SelfMobilityAMG}
#include <mundy_utils/rng.hpp>                         // for mundy::make_philox

namespace mundy {

namespace mbody {

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using backend_t = KokkosBackend<ExecSpace>;
using view_t = Kokkos::View<double*, ExecSpace::memory_space>;

//! \name Options
//@{

struct Options {
  size_t num_segments = 0;
  bool conditioning = false;
  bool iterations = false;
  std::string preconditioner = "jacobi";
  double dt = 100.0;
  std::string links = "fixed_length";
  double lanczos_tol = 1e-3;
  unsigned lanczos_max_iters = 200000;
  unsigned steps = 2;
  bool warm_start = false;
  double cg_tol = 1e-8;
  unsigned cg_max_iters = 1u << 30;
  unsigned coarse_max_size = 20;
  unsigned max_levels = 12;
  std::string smoother = "sgs";
  std::string prefix = "propped_cantilever";
  bool timings = false;
};

void print_usage(std::ostream& os) {
  os << "Usage: ProppedCantileverConditioning --num-segments N [--conditioning] [--iterations]\n"
     << "         [--preconditioner none|jacobi|amg] [--dt DT] [--links fixed_length|stiff_springs]\n"
     << "         [--lanczos-tol TOL] [--lanczos-max-iters M] [--steps S] [--warm-start] [--cg-tol TOL]\n"
     << "         [--cg-max-iters M] [--coarse-max-size C] [--max-levels L] [--smoother jacobi|sgs|chebyshev]\n"
     << "         [--prefix P] [--timings]\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--conditioning") {
      options.conditioning = true;
      continue;
    }
    if (key == "--iterations") {
      options.iterations = true;
      continue;
    }
    if (key == "--warm-start") {
      options.warm_start = true;
      continue;
    }
    if (key == "--timings") {
      options.timings = true;
      continue;
    }
    if (i + 1 == argc) {
      throw std::invalid_argument("ProppedCantileverConditioning: " + key + " needs a value");
    }
    const std::string value = argv[++i];
    if (key == "--num-segments") {
      options.num_segments = std::stoul(value);
    } else if (key == "--preconditioner") {
      options.preconditioner = value;
    } else if (key == "--dt") {
      options.dt = std::stod(value);
    } else if (key == "--links") {
      options.links = value;
    } else if (key == "--lanczos-tol") {
      options.lanczos_tol = std::stod(value);
    } else if (key == "--lanczos-max-iters") {
      options.lanczos_max_iters = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--steps") {
      options.steps = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--cg-tol") {
      options.cg_tol = std::stod(value);
    } else if (key == "--cg-max-iters") {
      options.cg_max_iters = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--coarse-max-size") {
      options.coarse_max_size = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--max-levels") {
      options.max_levels = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--smoother") {
      options.smoother = value;
    } else if (key == "--prefix") {
      options.prefix = value;
    } else {
      throw std::invalid_argument("ProppedCantileverConditioning: unknown option " + key);
    }
  }
  if (options.num_segments < 2 || (options.links != "fixed_length" && options.links != "stiff_springs") ||
      !(options.conditioning || options.iterations)) {
    throw std::invalid_argument("ProppedCantileverConditioning: invalid options");
  }
  return options;
}
//@}

//! \name The propped cantilever
//@{
// As in UnitTestKokkosMbody: a sphere chain of N segments over L = 8, held at rods 0 and 1, bent by a midspan load
// P = 0.01 onto an anchored sphere at its tip. Spacing a = L / N, radius 0.15 a, bend stiffness EI / a with EI = 5,
// and links of rest length a, rigid or stiff springs of 1e5 / a^2.

constexpr double kLength = 8.0;
constexpr double kBendingStiffness = 5.0;
constexpr double kLoad = 0.01;

/// \brief Anchor a holds rod at target.
void set_fixed_position(const FixedPositionViews<HostExecSpace>& anchors, int a, int rod, const Vector3d& target) {
  anchors.rod(a) = rod;
  anchors.target_point(a) = target;
  anchors.body_offset(a) = Vector3d{0.0, 0.0, 0.0};
  anchors.compliance(a) = Vector3d{0.0, 0.0, 0.0};
}

/// \brief The cantilever's rods, under their loads, and its constraints, with links of family Link.
template <typename Link>
auto make_propped_cantilever(size_t num_segments) {
  const double spacing = kLength / static_cast<double>(num_segments);
  const double radius = 0.15 * spacing;
  const size_t num_chain = num_segments + 2;
  const int midspan = static_cast<int>(num_segments / 2 + 1);
  const int tip = static_cast<int>(num_chain - 1);
  const int obstacle = static_cast<int>(num_chain);

  // The chain on a shallow arc, so every bend angle has a gradient, and the obstacle at its tip
  RodViews<HostExecSpace> rods(num_chain + 1);
  for (size_t k = 0; k < num_chain; ++k) {
    const double z = spacing * static_cast<double>(k);
    const double arc = std::max(z - spacing, 0.0);
    rods.center(k) = Vector3d{0.0, 1e-6 * arc * arc, z};
    rods.radius(k) = radius;
  }
  rods.center(obstacle) = Vector3d{0.0, -2.0 * radius, Vector3d(rods.center(tip))[2]};
  rods.radius(obstacle) = radius;
  for (size_t k = 0; k <= num_chain; ++k) {
    rods.orientation(k) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.length(k) = 0.0;
    rods.force(k) = Vector3d{0.0, 0.0, 0.0};
    rods.torque(k) = Vector3d{0.0, 0.0, 0.0};
    rods.velocity(k) = Vector3d{0.0, 0.0, 0.0};
    rods.omega(k) = Vector3d{0.0, 0.0, 0.0};
  }
  rods.force(midspan) = Vector3d{0.0, -kLoad, 0.0};

  // Links from rod 1 on, and a bend spring at each interior rod
  Link links(num_chain - 2);
  for (size_t k = 1; k + 1 < num_chain; ++k) {
    links.rod_i(k - 1) = static_cast<int>(k);
    links.rod_j(k - 1) = static_cast<int>(k + 1);
    links.rest_length(k - 1) = spacing;
    if constexpr (std::is_same_v<Link, LinearSpringViews<HostExecSpace>>) {
      links.spring_constant(k - 1) = 1.0e5 / (spacing * spacing);
    }
  }
  TriplePointAngularSpringViews<HostExecSpace> bends(num_chain - 2);
  for (size_t k = 0; k + 2 < num_chain; ++k) {
    bends.rod_i(k) = static_cast<int>(k);
    bends.rod_j(k) = static_cast<int>(k + 2);
    bends.rod_k(k) = static_cast<int>(k + 1);
    bends.rest_angle(k) = Kokkos::numbers::pi_v<double>;
    bends.spring_constant(k) = kBendingStiffness / spacing;
  }

  // The wall at rods 0 and 1, the anchored obstacle, and the tip's contact with it
  FixedPositionViews<HostExecSpace> anchors(3);
  set_fixed_position(anchors, 0, 0, Vector3d(rods.center(0)));
  set_fixed_position(anchors, 1, 1, Vector3d(rods.center(1)));
  set_fixed_position(anchors, 2, obstacle, Vector3d(rods.center(obstacle)));
  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = tip;
  contacts.rod_j(0) = obstacle;
  return std::make_pair(rods, make_constraint_set(links, bends, anchors, contacts));
}
//@}

//! \name The Schur system
//@{

/// \brief A copy of src in Space's memory that never shares storage with it, even within one memory space.
template <typename Space, typename T>
auto copy_to(const T& src) {
  const auto out = create_mirror(Space{}, src);
  deep_copy(out, src);
  return out;
}

/// \brief A copy of rods' force/torque, kept apart because a step adds the constraint forces into rods'.
view_t copy_load(const RodViews<ExecSpace>& rods) {
  const view_t load("load", rods.force_torque_view().extent(0));
  Kokkos::deep_copy(load, rods.force_torque_view());
  return load;
}

/// \brief force/torque := load and velocity/omega := 0, the state a step expects at its start.
void reset_rod_state(const RodViews<ExecSpace>& rods, const view_t& load) {
  Kokkos::deep_copy(rods.force_torque_view(), load);
  Kokkos::deep_copy(rods.velocity_omega_view(), 0.0);
}

/// \brief A linearization's Schur operator A = dt B^T M B + K^-1.
template <typename Linearization>
struct SchurOp {
  const Linearization* linearization;

  size_t domain_size() const {
    return linearization->num_rows();
  }
  size_t range_size() const {
    return linearization->num_rows();
  }
  void apply(const view_t& x, view_t& y) const {
    linearization->apply(x, y);
  }
};

/// \brief Hand use the linearization of rods under constraints at a step's start, and the step's first Schur right-hand
/// side b = psi + dt B^T u_free.
template <typename Model, typename... Families, typename Use>
void with_schur_system(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                       const Model& model, double dt, const Use& use) {
  const auto mobility = model.make_mobility(rods);
  using mobility_t = std::remove_cvref_t<decltype(mobility)>;
  impl::LinearizationWorkspace<ExecSpace, mobility_t, NoPreconditioner, Families...> workspace(
      impl::make_constraint_index_map(constraints), rods.size(), mobility, dt, NoPreconditioner{});
  const auto step =
      impl::make_step_data(rods, constraints, mobility, dt, PGDConfig<double>{}, CGConfig<double>{}, workspace);
  impl::linearize(step, workspace, rods, constraints);
  const auto schur_operator = impl::make_schur_operator(step, workspace.geometry());
  const auto linearization = workspace.schur_linearization(step, schur_operator);

  const size_t n = linearization.num_rows();
  view_t b("b", n);
  view_t b_rate("b_rate", n);
  const auto bt = impl::make_block_rate_op<ExecSpace>(workspace.geometry(), rods.size());
  auto bt_workspace = backend_t::make_workspace(bt);
  Kokkos::deep_copy(b, workspace.psi());
  backend_t::apply(bt, step.u_free, b_rate, bt_workspace);
  backend_t::axpby(dt, b_rate, 1.0, b);
  use(linearization, b);
}
//@}

//! \name Output
//@{

/// \brief x in scientific notation to four significant figures.
std::string sci(double x) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(3) << x;
  return os.str();
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// \brief An output stream appending to path, which gets header first if it is new.
std::ofstream open_csv(const std::string& path, const std::string& header) {
  const bool is_new = !std::ifstream(path).good();
  std::ofstream csv(path, std::ios::app);
  csv << std::setprecision(12);
  if (is_new) {
    csv << header << "\n";
  }
  return csv;
}

constexpr const char* kIdentityHeader = "Links,NumSegments,NumRows,Dt,Preconditioner,Smoother,CoarseMaxSize,MaxLevels";

std::string identity(const Options& options, size_t num_rows) {
  std::ostringstream os;
  os << options.links << "," << options.num_segments << "," << num_rows << "," << options.dt << ","
     << options.preconditioner << "," << options.smoother << "," << options.coarse_max_size << ","
     << options.max_levels;
  return os.str();
}
//@}

//! \name Measurements
//@{

/// \brief The preconditioner seconds of one setup: A_self's assembly (multigrid only) and the whole setup.
struct SetupSeconds {
  double assembly = 0.0;
  double setup = 0.0;
};

/// \brief policy's preconditioner, set up at linearization, timed into seconds.
template <typename Policy, typename Linearization>
auto set_up_preconditioner(const Policy& policy, const Linearization& linearization, SetupSeconds& seconds) {
  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();
#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
  if constexpr (std::is_same_v<Policy, SelfMobilityAMG>) {
    using matrix_t = typename SelfMobilityAMGOp<ExecSpace>::matrix_t;
    const auto row_rods = impl::make_row_rods<ExecSpace>(linearization.jacobian(), linearization.num_rows());
    const auto matrix = impl::make_shared_rod_pattern<matrix_t>(row_rods);
    impl::fill_self_mobility_values<ExecSpace>(linearization.dt(), linearization.jacobian(), linearization.mobility(),
                                               linearization.compliance(), matrix);
    Kokkos::fence();
    seconds.assembly = seconds_since(start);
    start = std::chrono::steady_clock::now();
  }
#endif
  auto preconditioner = policy.make_preconditioner(linearization);
  preconditioner.update(linearization);
  Kokkos::fence();
  seconds.setup = seconds_since(start);
  return preconditioner;
}

/// \brief Bound the spectrum of P A at linearization, P policy's preconditioner (A's spectrum for NoPreconditioner),
/// and write the conditioning row.
template <typename Policy, typename Linearization>
void measure_conditioning(const Options& options, const Policy& policy, const Linearization& linearization) {
  const size_t n = linearization.num_rows();
  const view_t start("start", n);
  Kokkos::parallel_for(
      "ProppedCantileverConditioning::start", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const size_t i) {
        openrand::Philox rng = make_philox(7, i);
        start(i) = rng.uniform<double>(-1.0, 1.0);
      });
  using host_array_t = Kokkos::View<double*, Kokkos::HostSpace>;
  auto state = make_lanczos_state(view_t(start), view_t("v_prev", n), view_t("z", n), view_t("Az", n),
                                  host_array_t("alpha", options.lanczos_max_iters),
                                  host_array_t("beta", options.lanczos_max_iters));
  const LanczosConfig<double> config{.max_iters = options.lanczos_max_iters, .tol = options.lanczos_tol};
  const auto problem = make_eigen_problem<backend_t>(SchurOp<Linearization>{&linearization});

  SetupSeconds setup;
  LanczosResult<double> result;
  double lanczos_seconds = 0.0;
  if constexpr (std::is_same_v<Policy, NoPreconditioner>) {
    const auto start_time = std::chrono::steady_clock::now();
    result = solve_eigen_problem(problem, make_lanczos_strategy(config), state);
    Kokkos::fence();
    lanczos_seconds = seconds_since(start_time);
  } else {
    const auto preconditioner = set_up_preconditioner(policy, linearization, setup);
    const auto start_time = std::chrono::steady_clock::now();
    result = solve_eigen_problem(problem, make_lanczos_strategy(config, preconditioner), state);
    Kokkos::fence();
    lanczos_seconds = seconds_since(start_time);
  }
  if (!result.converged) {
    throw std::runtime_error("ProppedCantileverConditioning: Lanczos did not converge within " +
                             std::to_string(options.lanczos_max_iters) + " iterations (relative Ritz residual " +
                             sci(result.residual) + ")");
  }
  const double condition_number = result.largest_eigenvalue / result.smallest_eigenvalue;

  std::ofstream csv = open_csv(options.prefix + "_conditioning.csv",
                               std::string(kIdentityHeader) + ",SmallestEigenvalue,LargestEigenvalue,ConditionNumber" +
                                   (options.timings ? ",AssemblySeconds,SetupSeconds,LanczosSeconds" : ""));
  csv << identity(options, n) << "," << result.smallest_eigenvalue << "," << result.largest_eigenvalue << ","
      << condition_number;
  std::cout << options.preconditioner << " conditioning: N " << sci(static_cast<double>(options.num_segments))
            << ", rows " << sci(static_cast<double>(n)) << ", dt " << sci(options.dt) << ", smallest eigenvalue "
            << sci(result.smallest_eigenvalue) << ", largest eigenvalue " << sci(result.largest_eigenvalue)
            << ", condition number " << sci(condition_number);
  if (options.timings) {
    csv << "," << setup.assembly << "," << setup.setup << "," << lanczos_seconds;
    std::cout << ", setup " << sci(setup.setup) << " s (A_self assembly " << sci(setup.assembly) << " s), Lanczos "
              << sci(lanczos_seconds) << " s";
  }
  csv << "\n";
  std::cout << std::endl;
}

/// \brief Solve A x = b by CG to a relative residual of options' tolerance, preconditioned by preconditioner unless it
/// is NoPreconditioner, from the guess in x.
template <typename Linearization, typename Precond>
CGResult<double> solve_schur(const Options& options, const Linearization& linearization, const view_t& b,
                             const view_t& x, const Precond& preconditioner) {
  const size_t n = linearization.num_rows();
  const auto system = make_linear_system<backend_t>(SchurOp<Linearization>{&linearization}, view_t(b));
  auto state = make_cg_state(view_t(x), view_t("r", n), view_t("p", n), view_t("Ap", n));
  const CGConfig<double> config{options.cg_max_iters, options.cg_tol};
  CGResult<double> result;
  if constexpr (std::is_same_v<Precond, NoPreconditioner>) {
    result = solve_linear_system(system, make_cg_solution_strategy(RelativeL2Residual{}, config), state);
  } else {
    result = solve_linear_system(system, make_cg_solution_strategy(RelativeL2Residual{}, config, preconditioner), state);
  }
  if (!result.converged) {
    throw std::runtime_error("ProppedCantileverConditioning: CG did not converge within " +
                             std::to_string(options.cg_max_iters) + " iterations (relative residual " +
                             sci(result.residual) + ")");
  }
  return result;
}

/// \brief The policy that moves the cantilever between measured steps, the same for every measured preconditioner.
auto stepping_policy() {
#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
  MueLuConfig<double> config;
  config.coarse_max_size = 20;
  config.max_levels = 12;
  config.smoother = MueLuSmoother::SYMMETRIC_GAUSS_SEIDEL;
  return SelfMobilityAMG{config};
#else
  return SelfMobilityJacobi{};
#endif
}

/// \brief At the start of each of options.steps + 1 steps, solve the step's first Schur system preconditioned by
/// policy's preconditioner, cold or warm-started from the last step's solution, and write a row per step.
template <typename Policy, typename Constraints, typename Model>
void measure_iterations(const Options& options, const Policy& policy, const RodViews<ExecSpace>& rods,
                        const Constraints& constraints, const Model& model, const view_t& load) {
  MixedLCPConfig lcp_cfg;
  lcp_cfg.max_cg_iters = 1u << 30;
  lcp_cfg.cg_tol = 1e-10;
  lcp_cfg.outer_tol = 1e-10;
  const MixedSLCPConfig step_cfg{lcp_cfg, 50, 1e-9, 1e-9};
  const auto integrator = make_mixed_slcp_integrator(rods, constraints, model, options.dt, stepping_policy());

  std::ofstream csv =
      open_csv(options.prefix + "_iterations.csv",
               std::string(kIdentityHeader) + ",WarmStart,Step,CgTol,Iterations" +
                   (options.timings ? ",AssemblySeconds,SetupSeconds,SolveSeconds" : ""));
  view_t previous;
  for (unsigned step = 0; step <= options.steps; ++step) {
    reset_rod_state(rods, load);
    with_schur_system(rods, constraints, model, options.dt, [&](const auto& linearization, const view_t& b) {
      const size_t n = linearization.num_rows();
      const view_t x("x", n);
      const bool warm = options.warm_start && previous.extent(0) == n;
      if (warm) {
        Kokkos::deep_copy(x, previous);
      }
      SetupSeconds setup;
      CGResult<double> result;
      double solve_seconds = 0.0;
      if constexpr (std::is_same_v<Policy, NoPreconditioner>) {
        const auto start = std::chrono::steady_clock::now();
        result = solve_schur(options, linearization, b, x, NoPreconditioner{});
        Kokkos::fence();
        solve_seconds = seconds_since(start);
      } else {
        const auto preconditioner = set_up_preconditioner(policy, linearization, setup);
        const auto start = std::chrono::steady_clock::now();
        result = solve_schur(options, linearization, b, x, preconditioner);
        Kokkos::fence();
        solve_seconds = seconds_since(start);
      }
      previous = x;

      csv << identity(options, n) << "," << warm << "," << step << "," << options.cg_tol << "," << result.num_iters;
      std::cout << options.preconditioner << " iterations: N " << sci(static_cast<double>(options.num_segments))
                << ", rows " << sci(static_cast<double>(n)) << ", dt " << sci(options.dt) << ", step "
                << sci(static_cast<double>(step)) << (warm ? ", warm" : ", cold") << ", iterations "
                << sci(static_cast<double>(result.num_iters));
      if (options.timings) {
        csv << "," << setup.assembly << "," << setup.setup << "," << solve_seconds;
        std::cout << ", setup " << sci(setup.setup) << " s (A_self assembly " << sci(setup.assembly) << " s), solve "
                  << sci(solve_seconds) << " s";
      }
      csv << "\n";
      std::cout << std::endl;
    });

    if (step < options.steps) {
      reset_rod_state(rods, load);
      const MixedSLCPResult result = solve_step(integrator, step_cfg);
      MUNDY_THROW_REQUIRE(result.accepted_lcp_result.converged, std::runtime_error,
                          "ProppedCantileverConditioning: a step's mixed LCP solve failed to converge.");
      advance_rods(rods, options.dt);
    }
  }
}
//@}

//! \name A run
//@{

/// \brief The run's measurements with policy's preconditioner and links of family Link.
template <typename Link, typename Policy>
void run(const Options& options, const Policy& policy) {
  const auto [rods_host, constraints_host] = make_propped_cantilever<Link>(options.num_segments);
  const auto rods = copy_to<ExecSpace>(rods_host);
  const auto constraints = copy_to<ExecSpace>(constraints_host);
  const LocalDragMobility model{.viscosity = 1.0};
  const view_t load = copy_load(rods);

  if (options.conditioning) {
    reset_rod_state(rods, load);
    with_schur_system(rods, constraints, model, options.dt, [&](const auto& linearization, const view_t&) {
      measure_conditioning(options, policy, linearization);
    });
  }
  if (options.iterations) {
    measure_iterations(options, policy, rods, constraints, model, load);
  }
}

/// \brief The run for options' preconditioner, with links of family Link.
template <typename Link>
void run_for_preconditioner(const Options& options) {
  if (options.preconditioner == "none") {
    run<Link>(options, NoPreconditioner{});
  } else if (options.preconditioner == "jacobi") {
    run<Link>(options, SelfMobilityJacobi{});
  } else if (options.preconditioner == "amg") {
#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
    MueLuConfig<double> config;
    config.coarse_max_size = options.coarse_max_size;
    config.max_levels = options.max_levels;
    if (options.smoother == "sgs") {
      config.smoother = MueLuSmoother::SYMMETRIC_GAUSS_SEIDEL;
    } else if (options.smoother == "chebyshev") {
      config.smoother = MueLuSmoother::CHEBYSHEV;
    } else if (options.smoother != "jacobi") {
      throw std::invalid_argument("ProppedCantileverConditioning: unknown smoother " + options.smoother);
    }
    run<Link>(options, SelfMobilityAMG{config});
#else
    throw std::invalid_argument("ProppedCantileverConditioning: amg needs MueLu, Tpetra and KokkosKernels");
#endif
  } else {
    throw std::invalid_argument("ProppedCantileverConditioning: unknown preconditioner " + options.preconditioner);
  }
}
//@}

}  // namespace

}  // namespace mbody

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    const mundy::mbody::Options options = mundy::mbody::parse_options(argc, argv);
    if (options.links == "fixed_length") {
      mundy::mbody::run_for_preconditioner<mundy::mbody::FixedLengthViews<mundy::mbody::HostExecSpace>>(options);
    } else {
      mundy::mbody::run_for_preconditioner<mundy::mbody::LinearSpringViews<mundy::mbody::HostExecSpace>>(options);
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << "\n";
    mundy::mbody::print_usage(std::cerr);
    status = 1;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << "\n";
    status = 1;
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return status;
}
