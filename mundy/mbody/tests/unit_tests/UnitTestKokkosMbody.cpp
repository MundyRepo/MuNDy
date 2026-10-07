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
/// \brief Unit tests for mbody's multibody time integrators and the operators and geometry kernels behind them.

// External
#include <gtest/gtest.h>  // for TEST, EXPECT_NEAR, etc

#include <Kokkos_Core.hpp>  // for Kokkos::View, Kokkos::parallel_for, Kokkos::parallel_reduce

// C++ core
#include <algorithm>  // for std::max
#include <bit>        // for std::bit_cast
#include <cmath>      // for std::abs, std::sqrt, std::pow, std::exp, std::log, std::atan2
#include <cstdint>    // for uint64_t
#include <cstring>    // for std::memcmp
#include <limits>     // for std::numeric_limits
#include <map>        // for std::map
#include <random>     // for std::mt19937, std::uniform_real_distribution
#include <stdexcept>  // for std::invalid_argument, std::runtime_error
#include <string>     // for std::to_string
#include <utility>    // for std::swap
#include <vector>     // for std::vector

// Mundy
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix
#include <mundy_math/eigenvalues.hpp>      // for mundy::make_eigen_problem, mundy::solve_eigen_problem
#include <mundy_math/preconditioners.hpp>  // for mundy::{NoPreconditioner, JacobiPreconditioner}
#include <mundy_mbody/KokkosMbody.hpp>     // for mundy::mbody::{make_mixed_lcp_integrator, solve_step, advance_rods}

namespace mundy {

namespace mbody {

namespace {

//! \name Device staging
//@{

// Library calls run on TestExecSpace; inputs are built and results checked on the host in HostExecSpace containers.
using TestExecSpace = Kokkos::DefaultExecutionSpace;
using TestMemSpace = TestExecSpace::memory_space;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

/// \brief One mixed LCP step of size dt on TestExecSpace for host inputs, which are updated in place.
template <typename Model, typename... Families>
MixedLCPResult solve_on_device(const RodViews<HostExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                               const Model& mobility_model, double dt, const MixedLCPConfig& cfg) {
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const MixedLCPResult result = solve_step(make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt), cfg);
  deep_copy(rods, rods_d);
  deep_copy(constraints, constraints_d);
  return result;
}

/// \brief A device vector with one entry per constraint row of family, for a geometry kernel to fill.
template <typename FamilyViews>
Kokkos::View<double*, TestMemSpace> make_constraint_values(const FamilyViews& family) {
  return Kokkos::View<double*, TestMemSpace>("constraint_values", family.num_rows());
}

/// \brief A copy of src in Space's memory that never shares storage with it, even within one memory space.
template <typename Space, typename T>
auto copy_to(const T& src) {
  const auto out = create_mirror(Space{}, src);
  deep_copy(out, src);
  return out;
}

/// \brief How many entries of two equally long vectors differ in value; -0.0 and +0.0 are equal.
template <typename ViewA, typename ViewB>
size_t count_value_differences(const ViewA& a_view, const ViewB& b_view) {
  const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_view);
  const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b_view);
  MUNDY_THROW_REQUIRE(a.extent(0) == b.extent(0), std::invalid_argument, "count_value_differences: length mismatch.");
  size_t differences = 0;
  for (size_t i = 0; i < a.extent(0); ++i) {
    differences += (a(i) != b(i)) ? 1 : 0;
  }
  return differences;
}

/// \brief How many entries of two equally long vectors differ in their bit patterns.
template <typename ViewA, typename ViewB>
size_t count_bit_differences(const ViewA& a_view, const ViewB& b_view) {
  const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a_view);
  const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b_view);
  MUNDY_THROW_REQUIRE(a.extent(0) == b.extent(0), std::invalid_argument, "count_bit_differences: length mismatch.");
  size_t differences = 0;
  for (size_t i = 0; i < a.extent(0); ++i) {
    differences += (std::memcmp(&a(i), &b(i), sizeof(a(i))) != 0) ? 1 : 0;
  }
  return differences;
}

//@}

//! \name A caller's mobility
//@{

/// \brief Every rod moves as an isotropic sphere: velocity = m force and omega = m_rot torque.
///
/// It has only a plain apply: no fused scaled apply, no workspace, and no self-mobility block.
template <typename Space>
struct IsotropicMobilityOp {
  using view_t = Kokkos::View<double*, typename Space::memory_space>;

  double m;
  double m_rot;
  size_t num_rods;

  size_t domain_size() const {
    return 6 * num_rods;
  }
  size_t range_size() const {
    return 6 * num_rods;
  }
  view_t make_domain_vector() const {
    return view_t("isotropic_domain", domain_size());
  }
  view_t make_range_vector() const {
    return view_t("isotropic_range", range_size());
  }
  void apply(const view_t& force_torque, view_t& vel_omega) const {
    const double m_trans = m, m_rotation = m_rot;
    Kokkos::parallel_for(
        "IsotropicMobilityOp::apply", Kokkos::RangePolicy<Space>(0, domain_size()),
        KOKKOS_LAMBDA(const int i) { vel_omega(i) = (i % 6 < 3 ? m_trans : m_rotation) * force_torque(i); });
  }
};

/// \brief The isotropic mobility of every rod.
struct IsotropicMobility {
  double m;
  double m_rot;

  template <typename Space>
  IsotropicMobilityOp<Space> make_mobility(const RodViews<Space>& rods) const {
    return IsotropicMobilityOp<Space>{m, m_rot, rods.size()};
  }
};

//@}

//! \name A caller's preconditioners
//@{

/// \brief Jacobi with d = 1, under which preconditioned CG performs plain CG's arithmetic.
///
/// d is NaN until the first update, so a solve that precedes it fails.
template <typename Space>
struct UnitJacobiOp : JacobiPreconditioner<KokkosBackend<Space>, Kokkos::View<double*, typename Space::memory_space>> {
  using view_t = Kokkos::View<double*, typename Space::memory_space>;

  explicit UnitJacobiOp(size_t num_rows)
      : JacobiPreconditioner<KokkosBackend<Space>, view_t>(KokkosBackend<Space>{}, make_nan_view(num_rows)) {
  }

  template <typename Linearization>
  void update(const Linearization&) {
    Kokkos::deep_copy(this->diag(), 1.0);
  }

  static view_t make_nan_view(size_t num_rows) {
    const view_t d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "unit_jacobi_d"), num_rows);
    Kokkos::deep_copy(d, std::numeric_limits<double>::quiet_NaN());
    return d;
  }
};

/// \brief Jacobi with d = 1.
struct UnitJacobi {
  template <typename Linearization>
  UnitJacobiOp<typename Linearization::execution_space> make_preconditioner(const Linearization& linearization) const {
    return UnitJacobiOp<typename Linearization::execution_space>(linearization.num_rows());
  }
};

/// \brief At one update, SelfMobilityJacobi's d and the diagonal of dt B^T M B + K^-1 probed by unit vectors.
struct ProbedDiagonal {
  std::vector<double> self_mobility_jacobi;
  std::vector<double> probed;
};

/// \brief SelfMobilityJacobi, recording at each update its d and the diagonal it should equal.
template <typename Space>
struct ProbingSelfMobilityJacobiOp : SelfMobilityJacobiOp<Space> {
  std::vector<ProbedDiagonal>* records;

  template <typename Linearization>
  void update(const Linearization& linearization) {
    SelfMobilityJacobiOp<Space>::update(linearization);
    const size_t num_rows = linearization.num_rows();
    const auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, this->diagonal());
    Kokkos::View<double*, typename Space::memory_space> e("e", num_rows), column("column", num_rows);
    ProbedDiagonal record{std::vector<double>(num_rows), std::vector<double>(num_rows)};
    for (size_t i = 0; i < num_rows; ++i) {
      Kokkos::deep_copy(e, 0.0);
      Kokkos::deep_copy(Kokkos::subview(e, i), 1.0);
      linearization.apply(e, column);
      record.self_mobility_jacobi[i] = d(i);
      record.probed[i] = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, column)(i);
    }
    records->push_back(record);
  }
};

/// \brief SelfMobilityJacobi, recording into records at each update.
struct ProbingSelfMobilityJacobi {
  std::vector<ProbedDiagonal>* records;

  template <typename Linearization>
  auto make_preconditioner(const Linearization& linearization) const {
    return ProbingSelfMobilityJacobiOp<typename Linearization::execution_space>{
        SelfMobilityJacobi{}.make_preconditioner(linearization), records};
  }
};

//@}

//! \name Compile-time contracts
//@{

static_assert(ViewsContainer<RodViews<HostExecSpace>> && ViewsContainer<LinearSpringViews<HostExecSpace>> &&
                  ViewsContainer<AngularSpringViews<HostExecSpace>> && ViewsContainer<PinViews<HostExecSpace>> &&
                  ViewsContainer<FixedLengthViews<HostExecSpace>> &&
                  ViewsContainer<TriplePointAngularSpringViews<HostExecSpace>> &&
                  ViewsContainer<FixedPositionViews<HostExecSpace>> && ViewsContainer<FixedPoseViews<HostExecSpace>> &&
                  ViewsContainer<ContactViews<HostExecSpace>>,
              "rods and every constraint family must be views containers");
static_assert(!ViewsContainer<ConstraintSet<PinViews<HostExecSpace>>>, "a constraint set is copied family by family");
static_assert(ConstraintFamily<LinearSpringViews<HostExecSpace>> &&
                  ConstraintFamily<AngularSpringViews<HostExecSpace>> && ConstraintFamily<PinViews<HostExecSpace>> &&
                  ConstraintFamily<FixedLengthViews<HostExecSpace>> &&
                  ConstraintFamily<TriplePointAngularSpringViews<HostExecSpace>> &&
                  ConstraintFamily<FixedPositionViews<HostExecSpace>> &&
                  ConstraintFamily<FixedPoseViews<HostExecSpace>> && ConstraintFamily<ContactViews<HostExecSpace>>,
              "every constraint family must satisfy ConstraintFamily");
static_assert(!ConstraintFamily<RodViews<HostExecSpace>>, "rods are bodies, not constraints");
static_assert(MobilityModel<LocalDragMobility, TestExecSpace> && HasSelfMobility<LocalDragMobilityOp<TestExecSpace>>,
              "local drag is a mobility model whose mobility has self blocks");
static_assert(MobilityModel<IsotropicMobility, TestExecSpace> && !HasSelfMobility<IsotropicMobilityOp<TestExecSpace>>,
              "a caller's mobility model needs only a plain apply");

/// \brief Whether make_constraint_set accepts families of these types.
template <typename... Families>
concept FormsConstraintSet = requires(const Families&... families) { make_constraint_set(families...); };

static_assert(FormsConstraintSet<PinViews<HostExecSpace>, ContactViews<HostExecSpace>>,
              "distinct families sharing an execution space form a constraint set");
static_assert(!FormsConstraintSet<PinViews<HostExecSpace>, PinViews<HostExecSpace>>,
              "a family type may appear in a constraint set only once");
static_assert(std::is_same_v<HostExecSpace, Kokkos::Serial> ||
                  !FormsConstraintSet<PinViews<HostExecSpace>, ContactViews<Kokkos::Serial>>,
              "the families of a constraint set must share one execution space");

//@}

//! \name Rod setup
//@{

/// \brief Two rods with the given poses and a shared radius and length.
RodViews<HostExecSpace> make_two_rod_system(const Vector3d& center_i, const Quaterniond& orientation_i,
                                            const Vector3d& center_j, const Quaterniond& orientation_j,
                                            double radius = 0.2, double length = 1.0) {
  RodViews<HostExecSpace> rods(2);

  rods.center(0) = center_i;
  rods.center(1) = center_j;
  rods.orientation(0) = orientation_i;
  rods.orientation(1) = orientation_j;
  rods.radius(0) = rods.radius(1) = radius;
  rods.length(0) = rods.length(1) = length;
  return rods;
}

/// \brief A copy of rods moved by (velocity, omega) * eps.
RodViews<HostExecSpace> perturb_rods(const RodViews<HostExecSpace>& rods,
                                     const Kokkos::View<double*, Kokkos::HostSpace>& vel_omega, double eps) {
  RodViews<HostExecSpace> out(rods.size());
  for (size_t i = 0; i < rods.size(); ++i) {
    const Vector3d vel = rod_velocity(vel_omega, static_cast<int>(i));
    const Vector3d omega = rod_omega(vel_omega, static_cast<int>(i));
    out.center(i) = rods.center(i) + eps * vel;
    out.radius(i) = rods.radius(i);
    out.length(i) = rods.length(i);

    const double omega_norm = norm(omega);
    if (omega_norm > 1e-14) {
      const Quaterniond dq = axis_angle_to_quaternion(omega / omega_norm, omega_norm * eps);
      out.orientation(i) = dq * rods.orientation(i);
    } else {
      out.orientation(i) = rods.orientation(i);
    }
  }
  return out;
}

/// \brief Zero every rod's force, torque, velocity and omega.
void zero_rod_state(const RodViews<HostExecSpace>& rods) {
  for (size_t i = 0; i < rods.size(); ++i) {
    rods.force(i) = Vector3d{0.0, 0.0, 0.0};
    rods.torque(i) = Vector3d{0.0, 0.0, 0.0};
    rods.velocity(i) = Vector3d{0.0, 0.0, 0.0};
    rods.omega(i) = Vector3d{0.0, 0.0, 0.0};
  }
}

/// \brief A two-rod velocity/omega vector.
Kokkos::View<double*, Kokkos::HostSpace> make_vel_omega(const Vector3d& vel_i, const Vector3d& omega_i,
                                                        const Vector3d& vel_j, const Vector3d& omega_j) {
  Kokkos::View<double*, Kokkos::HostSpace> v("vel_omega", 12);
  rod_velocity(v, 0) = vel_i;
  rod_omega(v, 0) = omega_i;
  rod_velocity(v, 1) = vel_j;
  rod_omega(v, 1) = omega_j;
  return v;
}

//@}

//! \name Anchor setup
//@{

/// \brief Anchor a of anchors holds the point body_offset of rod at target.
void set_fixed_position(const FixedPositionViews<HostExecSpace>& anchors, int a, int rod, const Vector3d& target,
                        const Vector3d& body_offset = Vector3d{0.0, 0.0, 0.0},
                        const Vector3d& compliance = Vector3d{0.0, 0.0, 0.0}) {
  anchors.rod(a) = rod;
  anchors.target_point(a) = target;
  anchors.body_offset(a) = body_offset;
  anchors.compliance(a) = compliance;
}

/// \brief Anchor a of anchors holds rod's point body_offset at target_point and its orientation at target_orientation.
void set_fixed_pose(const FixedPoseViews<HostExecSpace>& anchors, int a, int rod, const Vector3d& target_point,
                    const Quaterniond& target_orientation, const Vector3d& body_offset = Vector3d{0.0, 0.0, 0.0},
                    const Vector3d& position_compliance = Vector3d{0.0, 0.0, 0.0},
                    const Vector3d& orientation_compliance = Vector3d{0.0, 0.0, 0.0}) {
  anchors.rod(a) = rod;
  anchors.target_point(a) = target_point;
  anchors.target_orientation(a) = target_orientation;
  anchors.body_offset(a) = body_offset;
  anchors.position_compliance(a) = position_compliance;
  anchors.orientation_compliance(a) = orientation_compliance;
}

//@}

//! \name Time stepping
//@{

// Time loops stage once, step on TestExecSpace, and copy back only what they check.

/// \brief A copy of rods' force/torque, kept apart because a step adds the constraint forces into rods'.
template <typename Space>
Kokkos::View<double*, typename Space::memory_space> copy_load(const RodViews<Space>& rods) {
  Kokkos::View<double*, typename Space::memory_space> load("load", rods.force_torque_view().extent(0));
  Kokkos::deep_copy(load, rods.force_torque_view());
  return load;
}

/// \brief force/torque := load and velocity/omega := 0, the state a step expects at its start.
template <typename Space>
void reset_rod_state(const RodViews<Space>& rods, const Kokkos::View<double*, typename Space::memory_space>& load) {
  Kokkos::deep_copy(rods.force_torque_view(), load);
  Kokkos::deep_copy(rods.velocity_omega_view(), 0.0);
}

/// \brief One step of integrator under a constant external load, after which its rods move.
template <typename Integrator, typename Config>
auto step_rods(const Integrator& integrator, const Config& cfg,
               const Kokkos::View<double*, typename Integrator::execution_space::memory_space>& load) {
  reset_rod_state(integrator.rods(), load);
  const auto result = solve_step(integrator, cfg);
  advance_rods(integrator.rods(), integrator.dt());
  return result;
}

/// \brief The largest distance or angle any rod moved over its last step, dt max(|v|, |omega|).
template <typename Space>
double max_step_displacement(const RodViews<Space>& rods, double dt) {
  double max_speed = 0.0;
  Kokkos::parallel_reduce(
      "max_step_displacement", Kokkos::RangePolicy<Space>(0, rods.size()),
      KOKKOS_LAMBDA(const int i, double& m) {
        m = Kokkos::max(m, Kokkos::max(norm(rods.velocity(i)), norm(rods.omega(i))));
      },
      Kokkos::Max<double>(max_speed));
  return dt * max_speed;
}

/// \brief Whether a mixed LCP step's solve converged.
bool lcp_converged(const MixedLCPResult& result) {
  return result.converged;
}

/// \brief Whether the mixed LCP solve of a mixed SLCP step's returned linearization converged.
bool lcp_converged(const MixedSLCPResult& result) {
  return result.accepted_lcp_result.converged;
}

/// \brief An integrator over integrator's rods, constraints, mobility model, step size and preconditioner policy, with
/// storage of its own: its steps start from zero.
template <typename Integrator>
Integrator with_fresh_storage(const Integrator& integrator) {
  return Integrator(integrator.rods(), integrator.constraints(), integrator.mobility_model(), integrator.dt(),
                    integrator.preconditioner_policy());
}

/// \brief Step integrator until no rod moves farther than settled_step, or turns through a larger angle, in one step,
/// each step through integrator's storage if held_storage and through storage of its own otherwise.
///
/// Returns whether that happened within max_steps.
template <typename Integrator, typename Config>
bool step_until_settled(const Integrator& integrator, const Config& cfg,
                        const Kokkos::View<double*, typename Integrator::execution_space::memory_space>& load,
                        double settled_step, int max_steps, bool held_storage = true) {
  for (int step = 0; step < max_steps; ++step) {
    const auto result =
        held_storage ? step_rods(integrator, cfg, load) : step_rods(with_fresh_storage(integrator), cfg, load);
    MUNDY_THROW_REQUIRE(lcp_converged(result), std::runtime_error,
                        "step_until_settled: a step's mixed LCP solve failed to converge.");
    if (max_step_displacement(integrator.rods(), integrator.dt()) <= settled_step) {
      return true;
    }
  }
  return false;
}

//@}

//! \name Finite differences
//@{

/// \brief Expect each row of rate_op applied to vel_omega to match a central difference of value_of.
template <typename ValueOf, typename RateOp>
void expect_rate_matches_finite_difference(const ValueOf& value_of, const RodViews<HostExecSpace>& rods,
                                           const RateOp& rate_op,
                                           const Kokkos::View<double*, Kokkos::HostSpace>& vel_omega, double tol) {
  constexpr double eps = 1e-6;
  const Kokkos::View<double*, Kokkos::HostSpace> value_plus = value_of(perturb_rods(rods, vel_omega, eps));
  const Kokkos::View<double*, Kokkos::HostSpace> value_minus = value_of(perturb_rods(rods, vel_omega, -eps));

  Kokkos::View<double*, TestMemSpace> rate_d("rate", rate_op.range_size());
  rate_op.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, vel_omega), rate_d);
  const auto rate = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate_d);

  for (size_t row = 0; row < rate.extent(0); ++row) {
    EXPECT_NEAR(rate(row), (value_plus(row) - value_minus(row)) / (2.0 * eps), tol) << "row " << row;
  }
}

//@}

//! \name Dense linear algebra
//@{

/// \brief An operator as a dense row-major matrix, one unit-vector apply per column.
template <typename Op>
std::vector<std::vector<double>> materialize_dense(const Op& op) {
  const size_t rows = op.range_size();
  const size_t cols = op.domain_size();
  std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
  Kokkos::View<double*, TestMemSpace> unit("unit", cols), out("out", rows);
  for (size_t j = 0; j < cols; ++j) {
    Kokkos::deep_copy(unit, 0.0);
    Kokkos::deep_copy(Kokkos::subview(unit, j), 1.0);
    op.apply(unit, out);
    const auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    for (size_t i = 0; i < rows; ++i) {
      dense[i][j] = out_h(i);
    }
  }
  return dense;
}

// Row-major dense matrices for reference solves whose size is known only at run time.
using DenseMat = std::vector<std::vector<double>>;

DenseMat dense_matmul(const DenseMat& A, const DenseMat& B) {
  const size_t m = A.size(), k = A.empty() ? 0 : A[0].size(), n = B.empty() ? 0 : B[0].size();
  DenseMat C(m, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < m; ++i) {
    for (size_t p = 0; p < k; ++p) {
      const double a = A[i][p];
      for (size_t j = 0; j < n; ++j) {
        C[i][j] += a * B[p][j];
      }
    }
  }
  return C;
}

DenseMat dense_transpose(const DenseMat& A) {
  const size_t m = A.size(), n = A.empty() ? 0 : A[0].size();
  DenseMat T(n, std::vector<double>(m, 0.0));
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

std::vector<double> dense_matvec(const DenseMat& A, const std::vector<double>& x) {
  std::vector<double> y(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = 0; j < x.size(); ++j) {
      y[i] += A[i][j] * x[j];
    }
  }
  return y;
}

/// \brief [A B], for A and B with the same number of rows.
DenseMat dense_hcat(const DenseMat& A, const DenseMat& B) {
  DenseMat C = A;
  for (size_t i = 0; i < C.size(); ++i) {
    C[i].insert(C[i].end(), B[i].begin(), B[i].end());
  }
  return C;
}

/// \brief sqrt(sum_ij A_ij^2).
double dense_frobenius_norm(const DenseMat& A) {
  double sum = 0.0;
  for (const auto& row : A) {
    for (const double entry : row) {
      sum += entry * entry;
    }
  }
  return std::sqrt(sum);
}

/// \brief The x solving A x = b, by Gaussian elimination with partial pivoting.
std::vector<double> dense_solve(DenseMat A, std::vector<double> b) {
  const size_t n = A.size();
  for (size_t col = 0; col < n; ++col) {
    size_t piv = col;
    for (size_t r = col + 1; r < n; ++r) {
      if (std::abs(A[r][col]) > std::abs(A[piv][col])) {
        piv = r;
      }
    }
    std::swap(A[col], A[piv]);
    std::swap(b[col], b[piv]);
    const double d = A[col][col];
    for (size_t r = 0; r < n; ++r) {
      if (r == col) {
        continue;
      }
      const double f = A[r][col] / d;
      for (size_t c = col; c < n; ++c) {
        A[r][c] -= f * A[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    x[i] = b[i] / A[i][i];
  }
  return x;
}

/// \brief ||A^-1||_F, an upper bound on ||A^-1||_2, from one solve per column.
double dense_inverse_frobenius_norm(const DenseMat& A) {
  const size_t n = A.size();
  DenseMat inverse(n, std::vector<double>(n, 0.0));
  for (size_t col = 0; col < n; ++col) {
    std::vector<double> unit(n, 0.0);
    unit[col] = 1.0;
    const std::vector<double> x = dense_solve(A, unit);
    for (size_t row = 0; row < n; ++row) {
      inverse[row][col] = x[row];
    }
  }
  return dense_frobenius_norm(inverse);
}

//@}

//! \name Shared fixtures
//@{

constexpr size_t kChainNumRods = 6;

/// \brief The inputs to one mixed LCP step of size dt.
template <typename... Families>
struct SolveInput {
  RodViews<HostExecSpace> rods;
  ConstraintSet<Families...> constraints;
  LocalDragMobility mobility_model;
  double dt;
  MixedLCPConfig cfg;
};

/// \brief A chain of rods joined by linear and angular springs, slightly stretched and bent, loaded at both ends.
///
/// Each rod is turned a little from its neighbour so adjacent tangents are never parallel and every bend axis is
/// defined.
SolveInput<LinearSpringViews<HostExecSpace>, AngularSpringViews<HostExecSpace>> make_chain_problem(
    double spring_constant) {
  RodViews<HostExecSpace> rods(kChainNumRods);
  const double spacing = 1.6;
  for (size_t k = 0; k < kChainNumRods; ++k) {
    rods.center(k) = Vector3d{0.0, 0.0, spacing * static_cast<double>(k)};
    rods.orientation(k) = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.15 * static_cast<double>(k));
    rods.radius(k) = 0.2;
    rods.length(k) = 1.0;
    rods.force(k) = Vector3d{0.0, 0.0, 0.0};
    rods.torque(k) = Vector3d{0.0, 0.0, 0.0};
    rods.velocity(k) = Vector3d{0.0, 0.0, 0.0};
    rods.omega(k) = Vector3d{0.0, 0.0, 0.0};
  }
  rods.force(0) = Vector3d{0.3, -0.1, 0.0};
  rods.torque(kChainNumRods - 1) = Vector3d{0.0, 0.2, -0.1};

  const size_t num_links = kChainNumRods - 1;
  LinearSpringViews<HostExecSpace> linear_springs(num_links);
  AngularSpringViews<HostExecSpace> angular_springs(num_links);
  for (size_t k = 0; k < num_links; ++k) {
    linear_springs.rod_i(k) = static_cast<int>(k);
    linear_springs.rod_j(k) = static_cast<int>(k + 1);
    linear_springs.rest_length(k) = spacing - 0.1;  // slight initial stretch
    linear_springs.spring_constant(k) = spring_constant;

    angular_springs.rod_i(k) = static_cast<int>(k);
    angular_springs.rod_j(k) = static_cast<int>(k + 1);
    angular_springs.rest_angle(k) = 0.1;  // actual twist per link is 0.15, so a small initial bend
    angular_springs.spring_constant(k) = spring_constant;
  }

  MixedLCPConfig cfg;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-10;
  cfg.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.outer_tol = 1e-10;
  return {rods, make_constraint_set(linear_springs, angular_springs), LocalDragMobility{.viscosity = 1.0}, 0.5, cfg};
}

/// \brief Sum over every spring of 0.5 k (stretch or bend)^2, from rod poses.
template <typename Space>
double elastic_energy(const RodViews<Space>& rods, const LinearSpringViews<Space>& lin_springs,
                      const AngularSpringViews<Space>& ang_springs) {
  double linear_energy = 0.0;
  Kokkos::parallel_reduce(
      "linear_spring_energy", Kokkos::RangePolicy<Space>(0, lin_springs.size()),
      KOKKOS_LAMBDA(const int k, double& sum) {
        const int i = lin_springs.rod_i(k);
        const int j = lin_springs.rod_j(k);
        const double stretch = norm(rods.center(j) - rods.center(i)) - lin_springs.rest_length(k);
        sum += 0.5 * lin_springs.spring_constant(k) * stretch * stretch;
      },
      linear_energy);
  double angular_energy = 0.0;
  Kokkos::parallel_reduce(
      "angular_spring_energy", Kokkos::RangePolicy<Space>(0, ang_springs.size()),
      KOKKOS_LAMBDA(const int k, double& sum) {
        const int i = ang_springs.rod_i(k);
        const int j = ang_springs.rod_j(k);
        const Vector3d tangent_i = rods.orientation(i) * Vector3d{0.0, 0.0, 1.0};
        const Vector3d tangent_j = rods.orientation(j) * Vector3d{0.0, 0.0, 1.0};
        const double bend = minor_angle(tangent_i, tangent_j) - ang_springs.rest_angle(k);
        sum += 0.5 * ang_springs.spring_constant(k) * bend * bend;
      },
      angular_energy);
  return linear_energy + angular_energy;
}

/// \brief A straight chain of spheres along z with a bend spring at each interior vertex.
///
/// Callers add their own linear springs and anchors.
struct BendChain {
  RodViews<HostExecSpace> rods;
  TriplePointAngularSpringViews<HostExecSpace> springs;
};

BendChain make_bend_chain(size_t num_spheres, double spacing, double k_ang) {
  BendChain chain;
  chain.rods = RodViews<HostExecSpace>(num_spheres);
  for (size_t k = 0; k < num_spheres; ++k) {
    chain.rods.center(k) = Vector3d{0.0, 0.0, spacing * static_cast<double>(k)};
    chain.rods.orientation(k) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    chain.rods.radius(k) = 0.15 * spacing;
    chain.rods.length(k) = 0.0;
  }

  const size_t num_interior = num_spheres - 2;
  chain.springs = TriplePointAngularSpringViews<HostExecSpace>(num_interior);
  for (size_t k = 0; k < num_interior; ++k) {
    const size_t vertex = k + 1;
    chain.springs.rod_i(k) = static_cast<int>(vertex - 1);
    chain.springs.rod_j(k) = static_cast<int>(vertex + 1);
    chain.springs.rod_k(k) = static_cast<int>(vertex);
    chain.springs.rest_angle(k) = Kokkos::numbers::pi_v<double>;
    chain.springs.spring_constant(k) = k_ang;
  }
  return chain;
}

/// \brief Seven spheres in a row on x, in contact with their neighbours: six pressed together and one left free.
///
/// Spheres 0-5 are 0.384 to 0.416 apart against a contact distance of 0.4, and loads on 0 and 5 press them together.
/// Sphere 6 sits 0.05 clear of sphere 5, which is pulled away from it, so their contact never pushes.
SolveInput<ContactViews<HostExecSpace>> make_sphere_row_problem() {
  constexpr size_t kNumPressed = 6;
  RodViews<HostExecSpace> rods(kNumPressed + 1);
  for (size_t k = 0; k < kNumPressed; ++k) {
    rods.center(k) = Vector3d{(0.38 + 0.004 * static_cast<double>(k)) * static_cast<double>(k), 0.0, 0.0};
  }
  rods.center(kNumPressed) = Vector3d(rods.center(kNumPressed - 1)) + Vector3d{0.45, 0.0, 0.0};
  for (size_t k = 0; k < rods.size(); ++k) {
    rods.orientation(k) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(k) = 0.2;
    rods.length(k) = 0.0;
  }
  zero_rod_state(rods);
  rods.force(0) = Vector3d{1.0, 0.0, 0.0};
  rods.force(kNumPressed - 1) = Vector3d{-1.0, 0.0, 0.0};

  ContactViews<HostExecSpace> contacts(kNumPressed);
  for (size_t k = 0; k < kNumPressed; ++k) {
    contacts.rod_i(k) = static_cast<int>(k);
    contacts.rod_j(k) = static_cast<int>(k + 1);
  }

  MixedLCPConfig cfg;
  cfg.outer_tol = 1e-10;
  return {rods, make_constraint_set(contacts), LocalDragMobility{.viscosity = 1.0}, 0.3, cfg};
}

//@}

//! \name Copying between memory spaces
//@{

/// \brief view(i) := offset + i + 1 for every entry, distinct and nonzero across views given distinct offsets.
template <typename HostView>
void fill_distinct(const HostView& view, int offset) {
  for (size_t i = 0; i < view.extent(0); ++i) {
    view(i) = static_cast<typename HostView::value_type>(offset + static_cast<int>(i) + 1);
  }
}

/// \brief c copied to TestExecSpace and back, through fresh allocations on both sides.
template <typename T>
auto round_trip(const T& c) {
  return copy_to<HostExecSpace>(copy_to<TestExecSpace>(c));
}

TEST(Mbody, CreateMirrorCopiesEveryField) {
  RodViews<HostExecSpace> rods(2);
  fill_distinct(rods.center_view(), 0);
  fill_distinct(rods.orientation_view(), 100);
  fill_distinct(rods.radius_view(), 200);
  fill_distinct(rods.length_view(), 300);
  fill_distinct(rods.force_torque_view(), 400);
  fill_distinct(rods.velocity_omega_view(), 500);
  const auto rods_back = round_trip(rods);
  EXPECT_EQ(count_bit_differences(rods_back.center_view(), rods.center_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_back.orientation_view(), rods.orientation_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_back.radius_view(), rods.radius_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_back.length_view(), rods.length_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_back.force_torque_view(), rods.force_torque_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_back.velocity_omega_view(), rods.velocity_omega_view()), 0u);

  LinearSpringViews<HostExecSpace> linear(2);
  fill_distinct(linear.rod_i_view(), 0);
  fill_distinct(linear.rod_j_view(), 100);
  fill_distinct(linear.rest_length_view(), 200);
  fill_distinct(linear.spring_constant_view(), 300);
  fill_distinct(linear.lambda_view(), 400);
  const auto linear_back = round_trip(linear);
  EXPECT_EQ(count_bit_differences(linear_back.rod_i_view(), linear.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(linear_back.rod_j_view(), linear.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(linear_back.rest_length_view(), linear.rest_length_view()), 0u);
  EXPECT_EQ(count_bit_differences(linear_back.spring_constant_view(), linear.spring_constant_view()), 0u);
  EXPECT_EQ(count_bit_differences(linear_back.lambda_view(), linear.lambda_view()), 0u);

  AngularSpringViews<HostExecSpace> angular(2);
  fill_distinct(angular.rod_i_view(), 0);
  fill_distinct(angular.rod_j_view(), 100);
  fill_distinct(angular.rest_angle_view(), 200);
  fill_distinct(angular.spring_constant_view(), 300);
  fill_distinct(angular.lambda_view(), 400);
  const auto angular_back = round_trip(angular);
  EXPECT_EQ(count_bit_differences(angular_back.rod_i_view(), angular.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(angular_back.rod_j_view(), angular.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(angular_back.rest_angle_view(), angular.rest_angle_view()), 0u);
  EXPECT_EQ(count_bit_differences(angular_back.spring_constant_view(), angular.spring_constant_view()), 0u);
  EXPECT_EQ(count_bit_differences(angular_back.lambda_view(), angular.lambda_view()), 0u);

  TriplePointAngularSpringViews<HostExecSpace> triple(2);
  fill_distinct(triple.rod_i_view(), 0);
  fill_distinct(triple.rod_j_view(), 100);
  fill_distinct(triple.rod_k_view(), 200);
  fill_distinct(triple.rest_angle_view(), 300);
  fill_distinct(triple.spring_constant_view(), 400);
  fill_distinct(triple.lambda_view(), 500);
  const auto triple_back = round_trip(triple);
  EXPECT_EQ(count_bit_differences(triple_back.rod_i_view(), triple.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(triple_back.rod_j_view(), triple.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(triple_back.rod_k_view(), triple.rod_k_view()), 0u);
  EXPECT_EQ(count_bit_differences(triple_back.rest_angle_view(), triple.rest_angle_view()), 0u);
  EXPECT_EQ(count_bit_differences(triple_back.spring_constant_view(), triple.spring_constant_view()), 0u);
  EXPECT_EQ(count_bit_differences(triple_back.lambda_view(), triple.lambda_view()), 0u);

  FixedPositionViews<HostExecSpace> positions(2);
  fill_distinct(positions.rod_view(), 0);
  fill_distinct(positions.target_point_view(), 100);
  fill_distinct(positions.body_offset_view(), 200);
  fill_distinct(positions.compliance_view(), 300);
  fill_distinct(positions.lambda_view(), 400);
  const auto positions_back = round_trip(positions);
  EXPECT_EQ(count_bit_differences(positions_back.rod_view(), positions.rod_view()), 0u);
  EXPECT_EQ(count_bit_differences(positions_back.target_point_view(), positions.target_point_view()), 0u);
  EXPECT_EQ(count_bit_differences(positions_back.body_offset_view(), positions.body_offset_view()), 0u);
  EXPECT_EQ(count_bit_differences(positions_back.compliance_view(), positions.compliance_view()), 0u);
  EXPECT_EQ(count_bit_differences(positions_back.lambda_view(), positions.lambda_view()), 0u);

  FixedPoseViews<HostExecSpace> poses(2);
  fill_distinct(poses.rod_view(), 0);
  fill_distinct(poses.target_point_view(), 100);
  fill_distinct(poses.target_orientation_view(), 200);
  fill_distinct(poses.body_offset_view(), 300);
  fill_distinct(poses.compliance_view(), 400);
  fill_distinct(poses.lambda_view(), 500);
  const auto poses_back = round_trip(poses);
  EXPECT_EQ(count_bit_differences(poses_back.rod_view(), poses.rod_view()), 0u);
  EXPECT_EQ(count_bit_differences(poses_back.target_point_view(), poses.target_point_view()), 0u);
  EXPECT_EQ(count_bit_differences(poses_back.target_orientation_view(), poses.target_orientation_view()), 0u);
  EXPECT_EQ(count_bit_differences(poses_back.body_offset_view(), poses.body_offset_view()), 0u);
  EXPECT_EQ(count_bit_differences(poses_back.compliance_view(), poses.compliance_view()), 0u);
  EXPECT_EQ(count_bit_differences(poses_back.lambda_view(), poses.lambda_view()), 0u);

  PinViews<HostExecSpace> pins(2);
  fill_distinct(pins.rod_i_view(), 0);
  fill_distinct(pins.rod_j_view(), 100);
  fill_distinct(pins.body_offset_i_view(), 200);
  fill_distinct(pins.body_offset_j_view(), 300);
  fill_distinct(pins.lambda_view(), 400);
  const auto pins_back = round_trip(pins);
  EXPECT_EQ(count_bit_differences(pins_back.rod_i_view(), pins.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(pins_back.rod_j_view(), pins.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(pins_back.body_offset_i_view(), pins.body_offset_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(pins_back.body_offset_j_view(), pins.body_offset_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(pins_back.lambda_view(), pins.lambda_view()), 0u);

  FixedLengthViews<HostExecSpace> lengths(2);
  fill_distinct(lengths.rod_i_view(), 0);
  fill_distinct(lengths.rod_j_view(), 100);
  fill_distinct(lengths.body_offset_i_view(), 200);
  fill_distinct(lengths.body_offset_j_view(), 300);
  fill_distinct(lengths.rest_length_view(), 400);
  fill_distinct(lengths.lambda_view(), 500);
  const auto lengths_back = round_trip(lengths);
  EXPECT_EQ(count_bit_differences(lengths_back.rod_i_view(), lengths.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(lengths_back.rod_j_view(), lengths.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(lengths_back.body_offset_i_view(), lengths.body_offset_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(lengths_back.body_offset_j_view(), lengths.body_offset_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(lengths_back.rest_length_view(), lengths.rest_length_view()), 0u);
  EXPECT_EQ(count_bit_differences(lengths_back.lambda_view(), lengths.lambda_view()), 0u);

  ContactViews<HostExecSpace> contacts(2);
  fill_distinct(contacts.rod_i_view(), 0);
  fill_distinct(contacts.rod_j_view(), 100);
  fill_distinct(contacts.lambda_view(), 200);
  const auto contacts_back = round_trip(contacts);
  EXPECT_EQ(count_bit_differences(contacts_back.rod_i_view(), contacts.rod_i_view()), 0u);
  EXPECT_EQ(count_bit_differences(contacts_back.rod_j_view(), contacts.rod_j_view()), 0u);
  EXPECT_EQ(count_bit_differences(contacts_back.lambda_view(), contacts.lambda_view()), 0u);
}

// create_mirror allocates even within one memory space, where create_mirror_view aliases.
TEST(Mbody, CreateMirrorNeverAliases) {
  RodViews<HostExecSpace> rods(2);
  fill_distinct(rods.center_view(), 0);
  PinViews<HostExecSpace> pins(1);
  fill_distinct(pins.lambda_view(), 0);
  const auto constraints = make_constraint_set(pins);

  const auto rods_mirror = create_mirror(HostExecSpace{}, rods);
  const auto constraints_mirror = create_mirror(HostExecSpace{}, constraints);
  EXPECT_NE(rods_mirror.center_view().data(), rods.center_view().data());
  EXPECT_NE(get<PinViews<HostExecSpace>>(constraints_mirror).lambda_view().data(), pins.lambda_view().data());
  for (size_t i = 0; i < rods_mirror.center_view().extent(0); ++i) {
    EXPECT_EQ(rods_mirror.center_view()(i), 0.0) << "entry " << i;
  }

  EXPECT_EQ(create_mirror_view(HostExecSpace{}, rods).center_view().data(), rods.center_view().data());
  EXPECT_EQ(get<PinViews<HostExecSpace>>(create_mirror_view(HostExecSpace{}, constraints)).lambda_view().data(),
            pins.lambda_view().data());
}

//@}

//! \name Constraint Jacobians
//@{

// Each constraint's rate operator against a central difference of the constraint value its geometry kernel computes.

TEST(Mbody, ContactJacobianMatchesFiniteDifference) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{0.9, 0.3, 0.1},
                          axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.4));

  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 0;
  contacts.rod_j(0) = 1;

  const auto contacts_d = create_mirror_view_and_copy(TestExecSpace{}, contacts);
  auto sep0 = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), contacts_d, sep0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto sep = make_constraint_values(contacts);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), contacts_d, sep);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.3, -0.1, 0.2}, Vector3d{0.1, 0.2, -0.3}, Vector3d{-0.2, 0.4, 0.1}, Vector3d{-0.3, 0.1, 0.2});

  const impl::PairForceOpT<TestExecSpace> rate_op(geo, rods.size());
  expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-6);
}

TEST(Mbody, LinearSpringJacobianMatchesFiniteDifference) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{1.5, 0.5, -0.4},
                          axis_angle_to_quaternion(Vector3d{1.0, 0.0, 0.0}, 0.7));

  LinearSpringViews<HostExecSpace> springs(1);
  springs.rod_i(0) = 0;
  springs.rod_j(0) = 1;
  springs.rest_length(0) = 1.0;
  springs.spring_constant(0) = 2.0;

  const auto springs_d = create_mirror_view_and_copy(TestExecSpace{}, springs);
  auto b0 = make_constraint_values(springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(springs);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.1, 0.2, 0.05}, Vector3d{0.4, -0.2, 0.1}, Vector3d{-0.3, 0.1, -0.2}, Vector3d{0.2, 0.3, -0.1});

  const impl::PairForceOpT<TestExecSpace> rate_op(geo, rods.size());
  expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-6);
}

TEST(Mbody, AngularSpringJacobianMatchesFiniteDifference) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.2),
                          Vector3d{1.2, 0.0, 0.0}, axis_angle_to_quaternion(Vector3d{0.3, 1.0, 0.2}, 0.9));

  AngularSpringViews<HostExecSpace> springs(1);
  springs.rod_i(0) = 0;
  springs.rod_j(0) = 1;
  springs.rest_angle(0) = 0.5;
  springs.spring_constant(0) = 3.0;

  const auto springs_d = create_mirror_view_and_copy(TestExecSpace{}, springs);
  auto b0 = make_constraint_values(springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(springs);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.0, 0.0, 0.0}, Vector3d{0.3, -0.1, 0.2}, Vector3d{0.0, 0.0, 0.0}, Vector3d{-0.2, 0.4, 0.1});

  const impl::PairForceOpT<TestExecSpace> rate_op(geo, rods.size());
  expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-6);
}

TEST(Mbody, TriplePointAngularSpringJacobianMatchesFiniteDifference) {
  RodViews<HostExecSpace> rods(3);
  rods.center(0) = Vector3d{0.3, -0.2, 0.1};
  rods.center(1) = Vector3d{1.1, 0.4, -0.3};
  rods.center(2) = Vector3d{-0.2, 0.9, 0.5};  // the vertex
  for (size_t i = 0; i < 3; ++i) {
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};  // unused: the constraint depends on positions only
    rods.radius(i) = 0.2;
    rods.length(i) = 0.0;
  }

  TriplePointAngularSpringViews<HostExecSpace> springs(1);
  springs.rod_i(0) = 0;
  springs.rod_j(0) = 1;
  springs.rod_k(0) = 2;
  springs.rest_angle(0) = 1.2;
  springs.spring_constant(0) = 3.0;

  const auto springs_d = create_mirror_view_and_copy(TestExecSpace{}, springs);
  auto b0 = make_constraint_values(springs);
  const impl::TripleGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(springs);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };

  Kokkos::View<double*, Kokkos::HostSpace> vel_omega("vel_omega", 18);
  rod_velocity(vel_omega, 0) = Vector3d{0.3, -0.1, 0.2};
  rod_velocity(vel_omega, 1) = Vector3d{-0.2, 0.4, 0.1};
  rod_velocity(vel_omega, 2) = Vector3d{0.1, 0.2, -0.3};

  const impl::TripleForceOpT<TestExecSpace> rate_op(geo, rods.size());
  expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-6);
}

// <B x, y> == <x, B^T y> for random x and y. Rows share owners, as one constraint's rows on one rod do, so a
// forward operator that overwrites a rod's block instead of accumulating into it breaks the identity.
TEST(Mbody, SingleForceOpIsExactAdjoint) {
  constexpr size_t kNumRods = 4;
  constexpr size_t kNumRows = 9;
  constexpr size_t kGenDim = 6 * kNumRods;
  const int owners[kNumRows] = {0, 0, 0, 2, 2, 2, 3, 3, 1};  // rod 0 x3, rod 2 x3, rod 3 x2, rod 1 x1

  std::mt19937 rng(20260925);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const impl::SingleGeometry<HostExecSpace> geo_h(kNumRows);
  for (size_t p = 0; p < kNumRows; ++p) {
    geo_h.owner_view()(p) = owners[p];
    geo_h.force(static_cast<int>(p)) = Vector3d{dist(rng), dist(rng), dist(rng)};
    geo_h.torque(static_cast<int>(p)) = Vector3d{dist(rng), dist(rng), dist(rng)};
  }
  const impl::SingleGeometry<TestExecSpace> geo(
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.owner_view()),
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.jacobian_view()));

  const impl::SingleForceOp<TestExecSpace> B(geo, kNumRods);
  const impl::SingleForceOpT<TestExecSpace> BT(geo, kNumRods);

  Kokkos::View<double*, Kokkos::HostSpace> x("x", kNumRows), y("y", kGenDim);
  for (size_t p = 0; p < kNumRows; ++p) {
    x(p) = dist(rng);
  }
  for (size_t i = 0; i < kGenDim; ++i) {
    y(i) = dist(rng);
  }

  Kokkos::View<double*, TestMemSpace> bx_d("bx", kGenDim), bty_d("bty", kNumRows);
  B.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, x), bx_d);
  BT.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, y), bty_d);
  const auto bx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bx_d);
  const auto bty = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bty_d);

  double lhs = 0.0;
  double rhs = 0.0;
  for (size_t i = 0; i < kGenDim; ++i) {
    lhs += bx(i) * y(i);
  }
  for (size_t p = 0; p < kNumRows; ++p) {
    rhs += x(p) * bty(p);
  }

  // Summation order is the only difference between the two sides.
  EXPECT_NEAR(lhs, rhs, 1e-12 * std::max(1.0, std::abs(lhs)));
}

// The pair peer of the identity above. Rows share owner pairs, as a pin's three rows do.
TEST(Mbody, PairAdjoint) {
  constexpr size_t kNumRods = 4;
  constexpr size_t kNumRows = 8;
  constexpr size_t kGenDim = 6 * kNumRods;
  const int owners_i[kNumRows] = {0, 0, 0, 2, 2, 2, 1, 3};
  const int owners_j[kNumRows] = {1, 1, 1, 3, 3, 3, 3, 0};

  std::mt19937 rng(20261002);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const impl::PairGeometry<HostExecSpace> geo_h(kNumRows);
  for (size_t p = 0; p < kNumRows; ++p) {
    const int row = static_cast<int>(p);
    geo_h.owner_i_view()(p) = owners_i[p];
    geo_h.owner_j_view()(p) = owners_j[p];
    geo_h.force_i(row) = Vector3d{dist(rng), dist(rng), dist(rng)};
    geo_h.torque_i(row) = Vector3d{dist(rng), dist(rng), dist(rng)};
    geo_h.force_j(row) = Vector3d{dist(rng), dist(rng), dist(rng)};
    geo_h.torque_j(row) = Vector3d{dist(rng), dist(rng), dist(rng)};
  }
  const impl::PairGeometry<TestExecSpace> geo(
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.owner_i_view()),
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.owner_j_view()),
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.jacobian_i_view()),
      Kokkos::create_mirror_view_and_copy(TestMemSpace{}, geo_h.jacobian_j_view()));

  const impl::PairForceOp<TestExecSpace> B(geo, kNumRods);
  const impl::PairForceOpT<TestExecSpace> BT(geo, kNumRods);

  Kokkos::View<double*, Kokkos::HostSpace> x("x", kNumRows), y("y", kGenDim);
  for (size_t p = 0; p < kNumRows; ++p) {
    x(p) = dist(rng);
  }
  for (size_t i = 0; i < kGenDim; ++i) {
    y(i) = dist(rng);
  }

  Kokkos::View<double*, TestMemSpace> bx_d("bx", kGenDim), bty_d("bty", kNumRows);
  B.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, x), bx_d);
  BT.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, y), bty_d);
  const auto bx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bx_d);
  const auto bty = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bty_d);

  double lhs = 0.0;
  double rhs = 0.0;
  for (size_t i = 0; i < kGenDim; ++i) {
    lhs += bx(i) * y(i);
  }
  for (size_t p = 0; p < kNumRows; ++p) {
    rhs += x(p) * bty(p);
  }

  // Summation order is the only difference between the two sides.
  EXPECT_NEAR(lhs, rhs, 1e-12 * std::max(1.0, std::abs(lhs)));
}

// The anchor is off the rod centre and the rod is turned, so the torque rows r_world x e_c are nonzero.
TEST(Mbody, FixedPositionJacobianMatchesFiniteDifference) {
  const Quaterniond tilt = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.7);
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{0.3, -0.2, 1.4}, tilt);
  zero_rod_state(rods);

  FixedPositionViews<HostExecSpace> anchors(1);
  set_fixed_position(anchors, 0, /*rod=*/1, /*target=*/Vector3d{0.1, 0.1, 1.0},
                     /*body_offset=*/Vector3d{0.15, -0.1, 0.5});

  const auto anchors_d = create_mirror_view_and_copy(TestExecSpace{}, anchors);
  auto b0 = make_constraint_values(anchors);
  const impl::SingleGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), anchors_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(anchors);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), anchors_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}, Vector3d{0.4, -0.3, 0.2}, Vector3d{-0.2, 0.5, 0.3});

  const impl::SingleForceOpT<TestExecSpace> rate_op(geo, rods.size());
  expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-8);
}

// The inverse left Jacobian against a central difference of the rotation vector. The angles straddle the switch
// between its closed form and its series.
TEST(Mbody, RotationVectorJacobianMatchesFiniteDifference) {
  const Quaterniond target = axis_angle_to_quaternion(Vector3d{0.3, -0.5, 0.8}, 0.9);
  const Vector3d omega{0.4, -0.3, 0.25};

  for (const double error_angle : {1.5, 0.8, 0.2, 2.0e-2, 5.0e-3}) {
    const Quaterniond orientation = axis_angle_to_quaternion(Vector3d{0.2, 0.7, -0.4}, error_angle) * target;
    const Vector3d rotation_vector = quaternion_to_rotation_vector(orientation * inverse(target));
    const Matrix3<double> jacobian = impl::rotation_vector_jacobian(rotation_vector);

    constexpr double eps = 1e-7;
    const double omega_norm = norm(omega);
    const Quaterniond forward = axis_angle_to_quaternion(omega / omega_norm, omega_norm * eps);
    const Quaterniond backward = axis_angle_to_quaternion(omega / omega_norm, -omega_norm * eps);
    const Vector3d finite_difference = (quaternion_to_rotation_vector(forward * orientation * inverse(target)) -
                                        quaternion_to_rotation_vector(backward * orientation * inverse(target))) /
                                       (2.0 * eps);

    EXPECT_NEAR(norm(jacobian * omega - finite_difference), 0.0, 1e-7) << "error_angle=" << error_angle;
  }
}

// Both halves of a fixed pose match a central difference at any pose error. The sweep reaches 1.5 rad, where the
// identity in place of the inverse left Jacobian would be tens of percent off.
TEST(Mbody, FixedPoseJacobianMatchesFiniteDifference) {
  const Quaterniond rod_orientation = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.7);
  const Vector3d error_axis{0.0, 0.0, 1.0};

  for (const double error_angle : {1.5, 0.8, 0.4, 0.1}) {
    RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                       Vector3d{0.3, -0.2, 1.4}, rod_orientation);
    zero_rod_state(rods);

    // The target orientation makes the orientation error exactly error_angle about error_axis.
    FixedPoseViews<HostExecSpace> anchors(1);
    set_fixed_pose(anchors, 0, /*rod=*/1, /*target_point=*/Vector3d{0.1, 0.1, 1.0},
                   /*target_orientation=*/inverse(axis_angle_to_quaternion(error_axis, error_angle)) * rod_orientation,
                   /*body_offset=*/Vector3d{0.15, -0.1, 0.5});

    const auto anchors_d = create_mirror_view_and_copy(TestExecSpace{}, anchors);
    auto b0 = make_constraint_values(anchors);
    const impl::SingleGeometry<TestExecSpace> geo =
        impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), anchors_d, b0);

    auto value_of = [&](const RodViews<HostExecSpace>& r) {
      auto b = make_constraint_values(anchors);
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), anchors_d, b);
      return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
    };

    const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
        Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}, Vector3d{0.4, -0.3, 0.2}, Vector3d{-0.2, 0.5, 0.3});

    SCOPED_TRACE("error_angle=" + std::to_string(error_angle));
    const impl::SingleForceOpT<TestExecSpace> rate_op(geo, rods.size());
    expect_rate_matches_finite_difference(value_of, rods, rate_op, vel_omega, 1e-8);
  }
}

// Both rods turned, offset and spinning, so every lever-arm term is live. A central difference at eps = 1e-6 of
// O(1) values errs by eps^2 |f'''| / 6 + eps_mach |f| / eps, about 2e-10.
TEST(Mbody, HolonomicJacobians) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, axis_angle_to_quaternion(Vector3d{0.6, 0.0, 0.8}, 0.6),
                          Vector3d{0.9, -0.4, 1.3}, axis_angle_to_quaternion(Vector3d{0.0, 0.8, -0.6}, 1.1));
  zero_rod_state(rods);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.3, -0.2, 0.1}, Vector3d{-0.4, 0.2, 0.5}, Vector3d{-0.1, 0.4, -0.3}, Vector3d{0.2, -0.3, 0.4});
  const Vector3d offset_i{0.1, -0.2, 0.45};
  const Vector3d offset_j{-0.15, 0.05, -0.4};

  // Pin
  PinViews<HostExecSpace> pins(1);
  pins.rod_i(0) = 0;
  pins.rod_j(0) = 1;
  pins.body_offset_i(0) = offset_i;
  pins.body_offset_j(0) = offset_j;
  const auto pins_d = create_mirror_view_and_copy(TestExecSpace{}, pins);
  auto pin_b0 = make_constraint_values(pins);
  const impl::PairGeometry<TestExecSpace> pin_geo = impl::compute_geometry(rods_d, pins_d, pin_b0);

  auto pin_value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(pins);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), pins_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };
  {
    SCOPED_TRACE("pin");
    const impl::PairForceOpT<TestExecSpace> rate_op(pin_geo, rods.size());
    expect_rate_matches_finite_difference(pin_value_of, rods, rate_op, vel_omega, 1e-8);
  }

  // Fixed length
  FixedLengthViews<HostExecSpace> lengths(1);
  lengths.rod_i(0) = 0;
  lengths.rod_j(0) = 1;
  lengths.body_offset_i(0) = offset_i;
  lengths.body_offset_j(0) = offset_j;
  lengths.rest_length(0) = 1.2;
  const auto lengths_d = create_mirror_view_and_copy(TestExecSpace{}, lengths);
  auto length_b0 = make_constraint_values(lengths);
  const impl::PairGeometry<TestExecSpace> length_geo = impl::compute_geometry(rods_d, lengths_d, length_b0);

  auto length_value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(lengths);
    impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), lengths_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  };
  {
    SCOPED_TRACE("fixed length");
    const impl::PairForceOpT<TestExecSpace> rate_op(length_geo, rods.size());
    expect_rate_matches_finite_difference(length_value_of, rods, rate_op, vel_omega, 1e-8);
  }
}

//@}

//! \name Constraint packing
//@{

// Every family non-empty, each at a distinct prime size, so swapping any two moves an offset, and passed out of packing
// order: each block packs pairs, then triples, then single-body rows, each in the set's order, and contacts form the
// unilateral block. Also fixes the rows per entry: three per pin and fixed position, six per fixed pose.
TEST(Mbody, ConstraintIndexMapPacksEveryFamily) {
  const auto constraints = make_constraint_set(
      FixedPoseViews<HostExecSpace>(11), TriplePointAngularSpringViews<HostExecSpace>(5),
      LinearSpringViews<HostExecSpace>(2), ContactViews<HostExecSpace>(19), PinViews<HostExecSpace>(13),
      FixedPositionViews<HostExecSpace>(7), AngularSpringViews<HostExecSpace>(3), FixedLengthViews<HostExecSpace>(17));

  const auto index_map = impl::make_constraint_index_map(constraints);

  EXPECT_EQ(index_map.range<ContactViews<HostExecSpace>>().begin, 0u);
  EXPECT_EQ(index_map.range<ContactViews<HostExecSpace>>().size(), 19u);
  EXPECT_EQ(index_map.num_unilateral, 19u);

  EXPECT_EQ(index_map.range<LinearSpringViews<HostExecSpace>>().begin, 0u);
  EXPECT_EQ(index_map.range<LinearSpringViews<HostExecSpace>>().size(), 2u);
  EXPECT_EQ(index_map.range<PinViews<HostExecSpace>>().begin, 2u);
  EXPECT_EQ(index_map.range<PinViews<HostExecSpace>>().size(), 39u);
  EXPECT_EQ(index_map.range<AngularSpringViews<HostExecSpace>>().begin, 41u);
  EXPECT_EQ(index_map.range<AngularSpringViews<HostExecSpace>>().size(), 3u);
  EXPECT_EQ(index_map.range<FixedLengthViews<HostExecSpace>>().begin, 44u);
  EXPECT_EQ(index_map.range<FixedLengthViews<HostExecSpace>>().size(), 17u);
  EXPECT_EQ(index_map.range<TriplePointAngularSpringViews<HostExecSpace>>().begin, 61u);
  EXPECT_EQ(index_map.range<TriplePointAngularSpringViews<HostExecSpace>>().size(), 5u);
  EXPECT_EQ(index_map.range<FixedPoseViews<HostExecSpace>>().begin, 66u);
  EXPECT_EQ(index_map.range<FixedPoseViews<HostExecSpace>>().size(), 66u);
  EXPECT_EQ(index_map.range<FixedPositionViews<HostExecSpace>>().begin, 132u);
  EXPECT_EQ(index_map.range<FixedPositionViews<HostExecSpace>>().size(), 21u);
  EXPECT_EQ(index_map.num_bilateral, 153u);
}

//@}

//! \name Single-family solves
//@{

// One constraint through a full mixed LCP step. A single spring reduces the Schur complement to the scalar
// equation (B^T M B + 1/k) y = -b0, and a single contact to the scalar LCP lambda = max(0, -sep0 / A). The scalars come
// from the same operators the step uses, so these check its assembly of the operators, not the operators.
// The spring solves run CG to cg_tol = 1e-14, and B^T M B + 1/k >= 1/k bounds the multiplier's error by k cg_tol.

/// \brief The quadratic form B^T M B of a single constraint, as a scalar.
template <typename OpBT, typename OpM, typename OpB>
double scalar_quadratic_form(const OpBT& BT, const OpM& M, const OpB& B) {
  MUNDY_THROW_REQUIRE(B.domain_size() == 1, std::invalid_argument,
                      "scalar_quadratic_form: B must have exactly one column.");
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto btmb = make_quadratic_form<backend_t>(BT, M, B);

  Kokkos::View<double*, TestMemSpace> one("one", 1), result("result", 1);
  Kokkos::deep_copy(one, 1.0);
  auto workspace = backend_t::make_workspace(btmb);
  backend_t::apply(btmb, one, result, workspace);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result)(0);
}

// Two rods joined by one linear spring.
TEST(Mbody, LinearSpringOnlyMatchesScalarSchurComplement) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
  zero_rod_state(rods);

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = 2.0;

  const auto constraints = make_constraint_set(lin_springs);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 1.0;
  MixedLCPConfig cfg;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(lin_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, lin_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0 / (btmb + 1.0 / lin_springs.spring_constant(0));
  EXPECT_NEAR(lin_springs.lambda(0), y_expected, 1e-12);

  // Internal force: equal and opposite on the two rods.
  const Vector3d total_force = rods.force(0) + rods.force(1);
  const Vector3d total_torque = rods.torque(0) + rods.torque(1);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  EXPECT_NEAR(norm(total_torque), 0.0, 1e-9);
}

// A caller's own mobility drives the step. With every rod an isotropic sphere, a spring joining two rods' centers has
// B^T M B = 2 m exactly, so y = -b0 / (2 m dt + 1/k), and each rod moves at exactly m times its force.
TEST(Mbody, CallerMobilityMatchesClosedFormSpring) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
  zero_rod_state(rods);

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = 2.0;

  const auto constraints = make_constraint_set(lin_springs);

  const IsotropicMobility mobility_model{.m = 0.7, .m_rot = 1.3};
  const double dt = 0.5;
  MixedLCPConfig cfg;
  cfg.cg_tol = 1e-14;

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  EXPECT_TRUE(result.converged);

  // y to k cg_tol plus rounding; velocity = M W bit for bit
  const double y_expected = -(1.6 - 1.0) / (2.0 * mobility_model.m * dt + 1.0 / lin_springs.spring_constant(0));
  EXPECT_NEAR(lin_springs.lambda(0), y_expected, 1e-13);
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(rods.velocity(i)[k], mobility_model.m * rods.force(i)[k]) << "rod " << i << ", component " << k;
      EXPECT_EQ(rods.omega(i)[k], mobility_model.m_rot * rods.torque(i)[k]) << "rod " << i << ", component " << k;
    }
  }
}

// Two rods joined by one angular spring.
TEST(Mbody, AngularSpringOnlyMatchesScalarSchurComplement) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 1.6},
                          axis_angle_to_quaternion(Vector3d{1.0, 0.0, 0.0}, 0.3));
  zero_rod_state(rods);

  AngularSpringViews<HostExecSpace> ang_springs(1);
  ang_springs.rod_i(0) = 0;
  ang_springs.rod_j(0) = 1;
  ang_springs.rest_angle(0) = 0.0;
  ang_springs.spring_constant(0) = 2.0;

  const auto constraints = make_constraint_set(ang_springs);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 1.0;
  MixedLCPConfig cfg;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(ang_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, ang_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0 / (btmb + 1.0 / ang_springs.spring_constant(0));
  EXPECT_NEAR(ang_springs.lambda(0), y_expected, 1e-12);

  // Internal torque and no force: both cancel between the two rods.
  const Vector3d total_force = rods.force(0) + rods.force(1);
  const Vector3d total_torque = rods.torque(0) + rods.torque(1);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  EXPECT_NEAR(norm(total_torque), 0.0, 1e-9);
}

// Three spheres bent away from straight at the middle one, joined by one triple-point angular spring.
TEST(Mbody, TriplePointAngularSpringOnlyMatchesScalarSchurComplement) {
  RodViews<HostExecSpace> rods(3);
  rods.center(0) = Vector3d{0.0, 0.0, 0.0};
  rods.center(1) = Vector3d{0.0, 0.3, 1.0};  // the vertex
  rods.center(2) = Vector3d{0.1, 0.0, 2.1};
  for (int i = 0; i < 3; ++i) {
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(i) = 0.2;
    rods.length(i) = 0.0;
  }
  zero_rod_state(rods);

  TriplePointAngularSpringViews<HostExecSpace> triple_springs(1);
  triple_springs.rod_i(0) = 0;
  triple_springs.rod_j(0) = 2;
  triple_springs.rod_k(0) = 1;
  triple_springs.rest_angle(0) = Kokkos::numbers::pi_v<double>;
  triple_springs.spring_constant(0) = 2.0;

  const auto constraints = make_constraint_set(triple_springs);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 1.0;
  MixedLCPConfig cfg;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(triple_springs);
  const impl::TripleGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, triple_springs), b0_d);
  const impl::TripleForceOp<TestExecSpace> B(geo, rods.size());
  const impl::TripleForceOpT<TestExecSpace> BT(geo, rods.size());
  const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0 / (btmb + 1.0 / triple_springs.spring_constant(0));
  EXPECT_NEAR(triple_springs.lambda(0), y_expected, 1e-12);

  // Internal and position-only: the three forces cancel and there is no torque.
  const Vector3d total_force = rods.force(0) + rods.force(1) + rods.force(2);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(norm(rods.torque(i)), 0.0, 1e-12) << "rod " << i;
  }
}

struct ContactOnlyCaseResult {
  double lambda;
  double sep0;
  double A;
  bool converged;
};

/// \brief One contact between two parallel rods gap_x apart: the solved multiplier and its closed form's scalars.
ContactOnlyCaseResult run_contact_only_case(double gap_x, double radius) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{gap_x, 0.0, 0.0},
                          Quaterniond{1.0, 0.0, 0.0, 0.0}, radius, 1.0);
  zero_rod_state(rods);

  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 0;
  contacts.rod_j(0) = 1;

  const auto constraints = make_constraint_set(contacts);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 1.0;
  MixedLCPConfig cfg;
  cfg.outer_tol = 1e-12;

  // Scalar A := D^T M D
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto sep0 = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, contacts), sep0);
  const impl::PairForceOp<TestExecSpace> D(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> DT(geo, rods.size());
  const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);
  const double A_value = scalar_quadratic_form(DT, M, D);
  const double sep0_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep0)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  return ContactOnlyCaseResult{contacts.lambda(0), sep0_value, A_value, result.converged};
}

// Overlapping rods: lambda = -sep0 / A > 0.
TEST(Mbody, ContactOnlyActiveMatchesScalarLCP) {
  const ContactOnlyCaseResult r = run_contact_only_case(/*gap_x=*/0.3, /*radius=*/0.2);
  EXPECT_TRUE(r.converged);
  ASSERT_LT(r.sep0, 0.0) << "test setup should start penetrating";

  // PGD stops at |A lambda + sep0| <= outer_tol = 1e-12, so the error is at most 1e-12 / A, about 2.5e-12.
  const double lambda_expected = -r.sep0 / r.A;
  EXPECT_GT(r.lambda, 0.0);
  EXPECT_NEAR(r.lambda, lambda_expected, 1e-11);
}

// Separated rods: lambda = 0.
TEST(Mbody, ContactOnlyInactiveMatchesScalarLCP) {
  const ContactOnlyCaseResult r = run_contact_only_case(/*gap_x=*/1.0, /*radius=*/0.2);
  EXPECT_TRUE(r.converged);
  ASSERT_GT(r.sep0, 0.0) << "test setup should start separated";
  EXPECT_NEAR(r.lambda, 0.0, 1e-6);
}

// outer_tol bounds every contact's linearized end-of-step separation Phi + dt D^T U: within outer_tol of zero where the
// contact pushes, and at least -outer_tol where it does not. Recomputing the separation from the returned velocities
// rounds differently from the solve, by about 1e-16.
TEST(Mbody, OuterTolBoundsEndOfStepSeparations) {
  const auto p = make_sphere_row_problem();
  const auto& contacts = get<ContactViews<HostExecSpace>>(p.constraints);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  auto sep0_d = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, contacts), sep0_d);
  const impl::PairForceOpT<TestExecSpace> DT(geo, p.rods.size());
  const auto sep0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep0_d);

  for (const double outer_tol : {1e-2, 1e-6, 1e-12}) {
    // Solve
    const auto rods = copy_to<HostExecSpace>(p.rods);
    const auto constraints = copy_to<HostExecSpace>(p.constraints);
    MixedLCPConfig cfg = p.cfg;
    cfg.outer_tol = outer_tol;
    const MixedLCPResult result = solve_on_device(rods, constraints, p.mobility_model, p.dt, cfg);
    ASSERT_TRUE(result.converged) << "outer_tol " << outer_tol << ": " << result;

    // End-of-step separations
    Kokkos::View<double*, TestMemSpace> rate_d("rate", contacts.size());
    DT.apply(create_mirror_view_and_copy(TestExecSpace{}, rods.velocity_omega_view()), rate_d);
    const auto rate = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate_d);
    const auto& solved_contacts = get<ContactViews<HostExecSpace>>(constraints);
    size_t num_pushing = 0;
    for (size_t k = 0; k < contacts.size(); ++k) {
      const double separation = sep0(k) + p.dt * rate(k);
      if (solved_contacts.lambda(k) > 0.0) {
        ++num_pushing;
        EXPECT_LE(std::abs(separation), outer_tol + 1e-15) << "outer_tol " << outer_tol << ", contact " << k;
      } else {
        EXPECT_GE(separation, -outer_tol - 1e-15) << "outer_tol " << outer_tol << ", contact " << k;
      }
    }
    EXPECT_EQ(num_pushing, contacts.size() - 1) << "outer_tol " << outer_tol;
  }
}

// Empty families are zero-column operators, and a solve over them converges at once.

// No constraints at all: the velocities are exactly the free motion M F_ext.
TEST(Mbody, ZeroSpringsZeroContactsMatchesRawMobility) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
  zero_rod_state(rods);
  rods.force(0) = Vector3d{0.3, -0.2, 0.1};
  rods.torque(0) = Vector3d{0.1, 0.05, -0.1};
  rods.force(1) = Vector3d{-0.2, 0.1, 0.05};
  rods.torque(1) = Vector3d{0.05, -0.1, 0.1};

  const ConstraintSet<> constraints;

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 1.0;
  MixedLCPConfig cfg;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);
  Kokkos::View<double*, TestMemSpace> vel_omega_expected_d("vel_omega_expected", 6 * rods.size());
  M.apply(rods_d.force_torque_view(), vel_omega_expected_d);
  const auto vel_omega_expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vel_omega_expected_d);

  const auto slcp_rods_d = copy_to<TestExecSpace>(rods);
  const MixedLCPResult result = solve_on_device(rods, constraints, mobility_model, dt, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  for (size_t i = 0; i < rods.size(); ++i) {
    const Vector3d vel_expected = rod_velocity(vel_omega_expected, static_cast<int>(i));
    const Vector3d omega_expected = rod_omega(vel_omega_expected, static_cast<int>(i));
    EXPECT_NEAR(norm(rods.velocity(i) - vel_expected), 0.0, 1e-12);
    EXPECT_NEAR(norm(rods.omega(i) - omega_expected), 0.0, 1e-12);
  }

  // Mixed SLCP: the same free motion, accepted at its first linearization
  const MixedSLCPResult slcp = solve_step(make_mixed_slcp_integrator(slcp_rods_d, constraints, mobility_model, dt),
                                          MixedSLCPConfig{cfg, 50, 1e-9, 1e-9});
  EXPECT_TRUE(slcp.converged) << slcp;
  EXPECT_EQ(slcp.num_iters, 1u);
  EXPECT_EQ(slcp.accepted_lcp_result.num_iters, result.num_iters);
  EXPECT_EQ(slcp.accepted_lcp_result.converged, result.converged);
  const auto slcp_rods = copy_to<HostExecSpace>(slcp_rods_d);
  EXPECT_EQ(count_bit_differences(slcp_rods.velocity_omega_view(), rods.velocity_omega_view()), 0u);
}

// The Schur complement's CG on a 0x0 system: zero iterations and zero residual.
TEST(Mbody, EmptySpringBlockSchurComplementConvergesInZeroIterations) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const LinearSpringViews<TestExecSpace> lin_springs(0);
  const AngularSpringViews<TestExecSpace> ang_springs(0);

  const auto springs = make_constraint_set(lin_springs, ang_springs);
  const auto index_map = impl::make_constraint_index_map(springs);
  const auto block = impl::make_block_geometry<ConstraintType::BILATERAL, TestExecSpace>(index_map);
  const Kokkos::View<double*, TestMemSpace> b0("b0", index_map.num_bilateral);
  impl::compute_block_geometry<ConstraintType::BILATERAL>(rods_d, springs, index_map, block, b0);
  const impl::PairGeometry<TestExecSpace>& spring_geo = ::mundy::get<0>(block.groups);

  const impl::PairForceOp<TestExecSpace> B(spring_geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(spring_geo, rods.size());
  const LocalDragMobilityOp<TestExecSpace> M(1.0, rods_d);
  const Kokkos::View<double*, TestMemSpace> kinv_diag("kinv_diag", 0);

  using backend_t = KokkosBackend<TestExecSpace>;
  const auto btmb_plus_kinv =
      make_sum_op<backend_t>(make_quadratic_form<backend_t>(BT, M, B), make_diagonal_op<backend_t>(kinv_diag));
  const CGConfig<double> cg_cfg{.max_iters = 200, .tol = 1e-10};
  const auto S = make_cg_inv_op<backend_t>(btmb_plus_kinv, cg_cfg);

  Kokkos::View<double*, TestMemSpace> rhs("rhs", 0), out("out", 0);
  S.apply(rhs, out);

  EXPECT_TRUE(S.last_result().converged);
  EXPECT_EQ(S.last_result().num_iters, 0u);
  EXPECT_NEAR(S.last_result().residual, 0.0, 1e-15);
}

//@}

//! \name Multi-family solves
//@{

// Several families in one solve, against a dense solve of the same Schur complement.

/// \brief The multipliers and velocity/omega of one step.
struct DenseStep {
  std::vector<double> y;
  std::vector<double> vel_omega;
};

/// \brief The inverse Schur complement dt B^T M B + K^-1.
///
/// Its mobility is dt M, the displacement per unit force over one step.
DenseMat dense_schur_matrix(const DenseMat& B, const DenseMat& M, const std::vector<double>& kinv, double dt) {
  DenseMat schur = dense_matmul(dense_transpose(B), dense_matmul(M, B));
  for (size_t r = 0; r < schur.size(); ++r) {
    for (size_t c = 0; c < schur.size(); ++c) {
      schur[r][c] *= dt;
    }
    schur[r][r] += kinv[r];
  }
  return schur;
}

/// \brief One step solved densely: b = b0 + dt B^T M F_ext, (dt B^T M B + K^-1) y = -b, and v = M (F_ext + B y).
DenseStep dense_schur_step(const DenseMat& B, const DenseMat& M, const std::vector<double>& b0,
                           const std::vector<double>& kinv, const std::vector<double>& force_torque_ext, double dt) {
  const std::vector<double> b_rate = dense_matvec(dense_transpose(B), dense_matvec(M, force_torque_ext));
  std::vector<double> neg_b(b0.size());
  for (size_t r = 0; r < b0.size(); ++r) {
    neg_b[r] = -(b0[r] + dt * b_rate[r]);
  }

  DenseStep step;
  step.y = dense_solve(dense_schur_matrix(B, M, kinv, dt), neg_b);
  std::vector<double> total_force_torque = dense_matvec(B, step.y);
  for (size_t i = 0; i < total_force_torque.size(); ++i) {
    total_force_torque[i] += force_torque_ext[i];
  }
  step.vel_omega = dense_matvec(M, total_force_torque);
  return step;
}

// A chain of linear and angular springs packed into one y-block.
TEST(Mbody, ChainMatchesIndependentDenseSolve) {
  const auto p = make_chain_problem(/*spring_constant=*/3.0);
  const auto& lin_springs = get<LinearSpringViews<HostExecSpace>>(p.constraints);
  const auto& ang_springs = get<AngularSpringViews<HostExecSpace>>(p.constraints);

  // Dense reference, built before the step updates the inputs; B's columns follow the y-block order
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto lin_springs_d = create_mirror_view_and_copy(TestExecSpace{}, lin_springs);
  const auto ang_springs_d = create_mirror_view_and_copy(TestExecSpace{}, ang_springs);
  auto b0_lin_d = make_constraint_values(lin_springs_d);
  auto b0_ang_d = make_constraint_values(ang_springs_d);
  const impl::PairForceOp<TestExecSpace> B_lin(impl::compute_geometry(rods_d, lin_springs_d, b0_lin_d), kChainNumRods);
  const impl::PairForceOp<TestExecSpace> B_ang(impl::compute_geometry(rods_d, ang_springs_d, b0_ang_d), kChainNumRods);
  const LocalDragMobilityOp<TestExecSpace> M = p.mobility_model.make_mobility(rods_d);
  const auto b0_lin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_lin_d);
  const auto b0_ang = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_ang_d);

  std::vector<double> b0, kinv;
  for (size_t k = 0; k < b0_lin.extent(0); ++k) {
    b0.push_back(b0_lin(k));
    kinv.push_back(1.0 / lin_springs.spring_constant(k));
  }
  for (size_t k = 0; k < b0_ang.extent(0); ++k) {
    b0.push_back(b0_ang(k));
    kinv.push_back(1.0 / ang_springs.spring_constant(k));
  }
  const auto force_torque_ext = p.rods.force_torque_view();
  const DenseStep expected = dense_schur_step(
      dense_hcat(materialize_dense(B_lin), materialize_dense(B_ang)), materialize_dense(M), b0, kinv,
      std::vector<double>(force_torque_ext.data(), force_torque_ext.data() + force_torque_ext.size()), p.dt);

  // Solve
  const MixedLCPResult result = solve_on_device(p.rods, p.constraints, p.mobility_model, p.dt, p.cfg);
  EXPECT_TRUE(result.converged);

  // CG stops at a residual of cg_tol = 1e-10, and K^-1 = I/3 bounds the error in y, and so in v, by about 3e-10.
  const size_t num_links = kChainNumRods - 1;
  for (size_t i = 0; i < num_links; ++i) {
    EXPECT_NEAR(lin_springs.lambda(i), expected.y[i], 1e-9) << "linear spring " << i;
    EXPECT_NEAR(ang_springs.lambda(i), expected.y[num_links + i], 1e-9) << "angular spring " << i;
  }
  for (size_t i = 0; i < expected.vel_omega.size(); ++i) {
    EXPECT_NEAR(p.rods.velocity_omega_view()(i), expected.vel_omega[i], 1e-9) << "entry " << i;
  }
}

// A family a set does not hold and an empty family it does hold contribute nothing: the chain solves to the same values
// with only its springs as with every family present and the others empty, for either sequence length. Serial execution
// fixes the order of every atomic sum; an empty block may still turn a -0.0 into +0.0, so values, not bit patterns, are
// compared.
TEST(Mbody, AbsentFamiliesMatchEmptyFamilies) {
  const auto p = make_chain_problem(/*spring_constant=*/3.0);

  for (const unsigned max_iters : {1u, 50u}) {
    const MixedSLCPConfig cfg{p.cfg, max_iters, 1e-9, 1e-9};
    const auto lin_springs = copy_to<Kokkos::Serial>(get<LinearSpringViews<HostExecSpace>>(p.constraints));
    const auto ang_springs = copy_to<Kokkos::Serial>(get<AngularSpringViews<HostExecSpace>>(p.constraints));
    const auto springs_only = make_constraint_set(lin_springs, ang_springs);
    const auto every_family =
        make_constraint_set(lin_springs, ang_springs, PinViews<Kokkos::Serial>(0), FixedLengthViews<Kokkos::Serial>(0),
                            TriplePointAngularSpringViews<Kokkos::Serial>(0), FixedPositionViews<Kokkos::Serial>(0),
                            FixedPoseViews<Kokkos::Serial>(0), ContactViews<Kokkos::Serial>(0));
    const auto rods_reduced = copy_to<Kokkos::Serial>(p.rods);
    const auto rods_full = copy_to<Kokkos::Serial>(p.rods);

    // Solve
    const MixedSLCPResult reduced =
        solve_step(make_mixed_slcp_integrator(rods_reduced, springs_only, p.mobility_model, p.dt), cfg);
    const auto lin_lambda_reduced = copy_to<Kokkos::Serial>(lin_springs).lambda_view();
    const auto ang_lambda_reduced = copy_to<Kokkos::Serial>(ang_springs).lambda_view();
    const MixedSLCPResult full =
        solve_step(make_mixed_slcp_integrator(rods_full, every_family, p.mobility_model, p.dt), cfg);
    ASSERT_TRUE(lcp_converged(reduced)) << reduced;
    if (max_iters > 1) {
      ASSERT_TRUE(reduced.converged) << reduced;
      ASSERT_GE(reduced.num_iters, 2u) << "the sequence re-linearizes the springs";
    }

    // The same step
    EXPECT_EQ(full.num_iters, reduced.num_iters) << "max_iters=" << max_iters;
    EXPECT_EQ(full.residual, reduced.residual) << "max_iters=" << max_iters;
    EXPECT_EQ(full.converged, reduced.converged) << "max_iters=" << max_iters;
    EXPECT_EQ(full.accepted_lcp_result.num_iters, reduced.accepted_lcp_result.num_iters) << "max_iters=" << max_iters;
    EXPECT_EQ(full.accepted_lcp_result.residual, reduced.accepted_lcp_result.residual) << "max_iters=" << max_iters;
    EXPECT_EQ(count_value_differences(rods_reduced.force_torque_view(), rods_full.force_torque_view()), 0u)
        << "max_iters=" << max_iters;
    EXPECT_EQ(count_value_differences(rods_reduced.velocity_omega_view(), rods_full.velocity_omega_view()), 0u)
        << "max_iters=" << max_iters;
    EXPECT_EQ(count_value_differences(lin_lambda_reduced, lin_springs.lambda_view()), 0u) << "max_iters=" << max_iters;
    EXPECT_EQ(count_value_differences(ang_lambda_reduced, ang_springs.lambda_view()), 0u) << "max_iters=" << max_iters;
  }
}

// A simply supported beam: two anchors on different rods, coupled through the springs between them. B's columns act
// on two distinct bodies, so it keeps full column rank.
//
// The reference transposes B densely, so it also checks each family's rate operator against its force operator. The
// chain starts on a shallow arc so every bend angle has a gradient (see solve_static_bend) and the bend rows resist
// the load.
TEST(Mbody, ChainHeldAtBothEndsKeepsBothAnchors) {
  constexpr size_t num_segments = 8;
  constexpr size_t num_spheres = num_segments + 1;
  const double L = 8.0, EI = 5.0, load = 0.005, pre_bend = 1e-3;
  const double spacing = L / static_cast<double>(num_segments);

  for (const double dt : {0.05, 0.5, 5.0}) {
    BendChain chain = make_bend_chain(num_spheres, spacing, EI / spacing);
    RodViews<HostExecSpace> rods = chain.rods;
    for (size_t k = 0; k < num_spheres; ++k) {
      const double z = spacing * static_cast<double>(k);
      rods.center(k) = Vector3d{0.0, pre_bend * z * z, z};
    }

    LinearSpringViews<HostExecSpace> lin_springs(num_segments);
    for (size_t k = 0; k < num_segments; ++k) {
      lin_springs.rod_i(k) = static_cast<int>(k);
      lin_springs.rod_j(k) = static_cast<int>(k + 1);
      lin_springs.rest_length(k) = spacing;
      lin_springs.spring_constant(k) = 1.0e5 / (spacing * spacing);
    }

    const int last = static_cast<int>(num_spheres - 1);
    FixedPositionViews<HostExecSpace> supports(2);
    set_fixed_position(supports, 0, /*rod=*/0, /*target=*/Vector3d(rods.center(0)));
    set_fixed_position(supports, 1, /*rod=*/last, /*target=*/Vector3d(rods.center(last)));

    const auto constraints = make_constraint_set(lin_springs, chain.springs, supports);

    const LocalDragMobility mobility_model{.viscosity = 1.0};
    MixedLCPConfig cfg;
    cfg.max_cg_iters = 1000;
    cfg.cg_tol = 1e-12;
    cfg.max_outer_iters = 1;

    rods.force(num_spheres / 2) = Vector3d{0.0, -load, 0.0};

    // Dense reference, built before the step updates the inputs; B's columns follow the y-block order
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
    const auto lin_springs_d = create_mirror_view_and_copy(TestExecSpace{}, lin_springs);
    const auto triple_springs_d = create_mirror_view_and_copy(TestExecSpace{}, chain.springs);
    const auto supports_d = create_mirror_view_and_copy(TestExecSpace{}, supports);
    auto b0_lin_d = make_constraint_values(lin_springs_d);
    auto b0_triple_d = make_constraint_values(triple_springs_d);
    auto b0_fixed_d = make_constraint_values(supports_d);
    const impl::PairForceOp<TestExecSpace> B_lin(impl::compute_geometry(rods_d, lin_springs_d, b0_lin_d), num_spheres);
    const impl::TripleForceOp<TestExecSpace> B_triple(impl::compute_geometry(rods_d, triple_springs_d, b0_triple_d),
                                                      num_spheres);
    const impl::SingleForceOp<TestExecSpace> B_fixed(impl::compute_geometry(rods_d, supports_d, b0_fixed_d),
                                                     num_spheres);
    const LocalDragMobilityOp<TestExecSpace> M = mobility_model.make_mobility(rods_d);

    const auto b0_lin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_lin_d);
    const auto b0_triple = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_triple_d);
    const auto b0_fixed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_fixed_d);

    std::vector<double> b0, kinv;
    for (size_t k = 0; k < b0_lin.extent(0); ++k) {
      b0.push_back(b0_lin(k));
      kinv.push_back(1.0 / lin_springs.spring_constant(k));
    }
    for (size_t k = 0; k < b0_triple.extent(0); ++k) {
      b0.push_back(b0_triple(k));
      kinv.push_back(1.0 / chain.springs.spring_constant(k));
    }
    for (size_t k = 0; k < b0_fixed.extent(0); ++k) {
      b0.push_back(b0_fixed(k));
      kinv.push_back(0.0);  // rigid
    }
    const auto force_torque_ext = rods.force_torque_view();
    const DenseStep expected = dense_schur_step(
        dense_hcat(dense_hcat(materialize_dense(B_lin), materialize_dense(B_triple)), materialize_dense(B_fixed)),
        materialize_dense(M), b0, kinv,
        std::vector<double>(force_torque_ext.data(), force_torque_ext.data() + force_torque_ext.size()), dt);

    // Solve
    ASSERT_TRUE(solve_on_device(rods, constraints, mobility_model, dt, cfg).converged) << "dt=" << dt;

    std::vector<double> y;
    for (size_t k = 0; k < lin_springs.size(); ++k) {
      y.push_back(lin_springs.lambda(k));
    }
    for (size_t k = 0; k < chain.springs.size(); ++k) {
      y.push_back(chain.springs.lambda(k));
    }
    for (size_t k = 0; k < supports.size(); ++k) {
      for (int c = 0; c < 3; ++c) {
        y.push_back(supports.lambda(k)[c]);
      }
    }
    ASSERT_EQ(y.size(), expected.y.size());
    for (size_t r = 0; r < y.size(); ++r) {
      EXPECT_NEAR(y[r], expected.y[r], 1e-9) << "multiplier " << r << " at dt=" << dt;
    }
    for (size_t i = 0; i < 6 * num_spheres; ++i) {
      EXPECT_NEAR(rods.velocity_omega_view()(i), expected.vel_omega[i], 1e-12) << "entry " << i << " at dt=" << dt;
    }
    EXPECT_NEAR(norm(rods.velocity(0)), 0.0, 1e-12) << "dt=" << dt;
    EXPECT_NEAR(norm(rods.velocity(num_spheres - 1)), 0.0, 1e-12) << "dt=" << dt;
  }
}

// Pins and a fixed length are rigid rows (K^-1 = 0) beside a spring and an anchor. Rod 0 is held, a spring joins it to
// rod 1, rod 1's end is pinned to rod 2's, and a fixed length joins rods 2 and 3, so the rigid columns stay
// independent. CG stops at ||r||_2 <= cg_tol, so ||y - y*||_2 <= cg_tol ||A^-1||_2 with A = dt B^T M B + K^-1, and
// v = M (F_ext + B y) adds a factor ||M B||_2. Frobenius norms bound both.
TEST(Mbody, HolonomicDenseStep) {
  constexpr size_t kNumRods = 4;
  const Vector3d tilt_axis{0.6, 0.8, 0.0};
  const double tilts[kNumRods] = {0.0, 0.3, -0.5, 0.8};
  RodViews<HostExecSpace> rods(kNumRods);
  for (size_t k = 0; k < kNumRods; ++k) {
    const double s = static_cast<double>(k);
    rods.center(k) = Vector3d{0.1 * s, -0.05 * s, 1.1 * s};
    rods.orientation(k) = axis_angle_to_quaternion(tilt_axis, tilts[k]);
    rods.radius(k) = 0.2;
    rods.length(k) = 1.0;
  }
  zero_rod_state(rods);
  rods.force(2) = Vector3d{0.3, -0.2, 0.1};
  rods.force(3) = Vector3d{-0.1, 0.25, 0.2};
  rods.torque(3) = Vector3d{0.05, -0.1, 0.08};

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = 3.0;
  PinViews<HostExecSpace> pins(1);
  pins.rod_i(0) = 1;
  pins.rod_j(0) = 2;
  pins.body_offset_i(0) = Vector3d{0.0, 0.0, 0.5};
  pins.body_offset_j(0) = Vector3d{0.0, 0.0, -0.5};
  FixedLengthViews<HostExecSpace> lengths(1);
  lengths.rod_i(0) = 2;
  lengths.rod_j(0) = 3;
  lengths.body_offset_i(0) = Vector3d{0.1, 0.0, 0.4};
  lengths.body_offset_j(0) = Vector3d{0.0, -0.1, -0.4};
  lengths.rest_length(0) = 0.35;
  FixedPositionViews<HostExecSpace> anchors(1);
  set_fixed_position(anchors, 0, /*rod=*/0, Vector3d(rods.center(0)));
  const auto constraints = make_constraint_set(lin_springs, pins, lengths, anchors);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 0.3;
  MixedLCPConfig cfg;
  cfg.max_cg_iters = 1000;
  cfg.cg_tol = 1e-10;
  cfg.max_outer_iters = 1;

  // Dense reference, built before the step updates the inputs; B's columns follow the y-block order
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto lin_springs_d = create_mirror_view_and_copy(TestExecSpace{}, lin_springs);
  const auto pins_d = create_mirror_view_and_copy(TestExecSpace{}, pins);
  const auto lengths_d = create_mirror_view_and_copy(TestExecSpace{}, lengths);
  const auto anchors_d = create_mirror_view_and_copy(TestExecSpace{}, anchors);
  auto b0_lin_d = make_constraint_values(lin_springs_d);
  auto b0_pin_d = make_constraint_values(pins_d);
  auto b0_length_d = make_constraint_values(lengths_d);
  auto b0_fixed_d = make_constraint_values(anchors_d);
  const impl::PairForceOp<TestExecSpace> B_lin(impl::compute_geometry(rods_d, lin_springs_d, b0_lin_d), kNumRods);
  const impl::PairForceOp<TestExecSpace> B_pin(impl::compute_geometry(rods_d, pins_d, b0_pin_d), kNumRods);
  const impl::PairForceOp<TestExecSpace> B_length(impl::compute_geometry(rods_d, lengths_d, b0_length_d), kNumRods);
  const impl::SingleForceOp<TestExecSpace> B_fixed(impl::compute_geometry(rods_d, anchors_d, b0_fixed_d), kNumRods);
  const LocalDragMobilityOp<TestExecSpace> M_op = mobility_model.make_mobility(rods_d);

  std::vector<double> b0, kinv;
  const auto append_rows = [&b0, &kinv](const Kokkos::View<double*, TestMemSpace>& rows, double compliance) {
    const auto rows_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rows);
    for (size_t r = 0; r < rows_h.extent(0); ++r) {
      b0.push_back(rows_h(r));
      kinv.push_back(compliance);
    }
  };
  append_rows(b0_lin_d, 1.0 / lin_springs.spring_constant(0));
  append_rows(b0_pin_d, 0.0);
  append_rows(b0_length_d, 0.0);
  append_rows(b0_fixed_d, 0.0);

  const DenseMat B = dense_hcat(
      dense_hcat(dense_hcat(materialize_dense(B_lin), materialize_dense(B_pin)), materialize_dense(B_length)),
      materialize_dense(B_fixed));
  const DenseMat M = materialize_dense(M_op);
  const auto force_torque_ext = rods.force_torque_view();
  const DenseStep expected = dense_schur_step(
      B, M, b0, kinv, std::vector<double>(force_torque_ext.data(), force_torque_ext.data() + force_torque_ext.size()),
      dt);
  const double y_bound = cfg.cg_tol * dense_inverse_frobenius_norm(dense_schur_matrix(B, M, kinv, dt));
  const double v_bound = dense_frobenius_norm(dense_matmul(M, B)) * y_bound;

  // Solve
  ASSERT_TRUE(solve_on_device(rods, constraints, mobility_model, dt, cfg).converged);

  std::vector<double> y{lin_springs.lambda(0)};
  for (int c = 0; c < 3; ++c) {
    y.push_back(pins.lambda(0)[c]);
  }
  y.push_back(lengths.lambda(0));
  for (int c = 0; c < 3; ++c) {
    y.push_back(anchors.lambda(0)[c]);
  }
  ASSERT_EQ(y.size(), expected.y.size());
  for (size_t r = 0; r < y.size(); ++r) {
    EXPECT_NEAR(y[r], expected.y[r], y_bound) << "multiplier " << r;
  }
  for (size_t i = 0; i < 6 * kNumRods; ++i) {
    EXPECT_NEAR(rods.velocity_omega_view()(i), expected.vel_omega[i], v_bound) << "entry " << i;
  }
}

//@}

//! \name Anchors
//@{

// Constraints holding a point or pose of a rod at a target.

// A rigid anchor reaches its target in one step wherever its rows are linear in the step, and at rest its multipliers
// are the reactions to the load, at every dt. A center position anchor's rows are linear in the step, and so are a pose
// anchor's orientation rows, the rotation vector of the error: the step they solve for turns about that vector, along
// which it changes linearly. The pose anchor's point rows turn with the rod, so only a step whose rows hold at its end
// reaches that point, to within length_tol.
TEST(Mbody, RigidAnchorsReachTheirTargets) {
  const Vector3d position_target{0.4, -0.3, 1.1};
  const Vector3d pose_target_point{-1.4, 0.7, -0.9};
  const Quaterniond pose_target_orientation = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.6);
  const Vector3d body_offset{0.0, 0.0, 0.45};
  const Vector3d load{0.7, 0.25, -0.5};

  for (const double dt : {0.005, 0.5, 2.0, 20.0}) {
    for (const unsigned max_iters : {1u, 50u}) {
      RodViews<HostExecSpace> rods(2);
      rods.center(0) = position_target + Vector3d{-0.35, 0.2, 0.15};  // both start off target
      rods.center(1) = pose_target_point + Vector3d{-0.3, 0.25, 0.1};
      for (int i = 0; i < 2; ++i) {
        rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};
        rods.radius(i) = 0.2;
      }
      rods.length(0) = 1.0;
      rods.length(1) = 0.9;

      FixedPositionViews<HostExecSpace> position_anchors(1);
      set_fixed_position(position_anchors, 0, /*rod=*/0, position_target);
      FixedPoseViews<HostExecSpace> pose_anchors(1);
      set_fixed_pose(pose_anchors, 0, /*rod=*/1, pose_target_point, pose_target_orientation, body_offset);
      const auto constraints = make_constraint_set(position_anchors, pose_anchors);

      const LocalDragMobility mobility_model{.viscosity = 1.0};
      MixedSLCPConfig cfg;
      cfg.inner_lcp_config.max_cg_iters = 500;
      cfg.inner_lcp_config.cg_tol = 1e-14;
      cfg.inner_lcp_config.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
      cfg.max_iters = max_iters;
      cfg.length_tol = 1e-11;
      cfg.angle_tol = 1e-11;

      zero_rod_state(rods);
      rods.force(0) = load;
      rods.force(1) = load;
      const auto rods_d = copy_to<TestExecSpace>(rods);
      const auto constraints_d = copy_to<TestExecSpace>(constraints);
      const auto load_d = copy_load(rods_d);
      const auto integrator = make_mixed_slcp_integrator(rods_d, constraints_d, mobility_model, dt);

      // One step
      const MixedSLCPResult first = step_rods(integrator, cfg, load_d);
      ASSERT_TRUE(lcp_converged(first)) << first << " at dt=" << dt;
      deep_copy(rods, rods_d);
      const Vector3d step_rotation_error =
          quaternion_to_rotation_vector(rods.orientation(1) * inverse(pose_target_orientation));
      EXPECT_NEAR(norm(rods.center(0) - position_target), 0.0, 1e-10) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(step_rotation_error), 0.0, 1e-10) << "dt=" << dt << " max_iters=" << max_iters;
      if (max_iters > 1) {
        ASSERT_TRUE(first.converged) << first << " at dt=" << dt;
        EXPECT_NEAR(norm(rods.center(1) + rods.orientation(1) * body_offset - pose_target_point), 0.0, cfg.length_tol)
            << "dt=" << dt;
      }

      // At rest
      ASSERT_TRUE(step_until_settled(integrator, cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000))
          << "not settled at dt=" << dt << " max_iters=" << max_iters;
      reset_rod_state(rods_d, load_d);
      ASSERT_TRUE(solve_step(integrator, cfg).converged) << "dt=" << dt << " max_iters=" << max_iters;
      deep_copy(rods, rods_d);
      deep_copy(constraints, constraints_d);
      const Vector3d r_world = rods.orientation(1) * body_offset;
      const Vector3d rest_rotation_error =
          quaternion_to_rotation_vector(rods.orientation(1) * inverse(pose_target_orientation));
      EXPECT_NEAR(norm(rods.velocity(0)), 0.0, 1e-10) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(rods.velocity(1)), 0.0, 1e-10) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(position_anchors.lambda(0) + load), 0.0, 1e-10) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(rods.center(1) + r_world - pose_target_point), 0.0, 1e-9)
          << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(rest_rotation_error), 0.0, 1e-9) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(pose_anchors.position_lambda(0) + load), 0.0, 1e-9) << "dt=" << dt << " max_iters=" << max_iters;
      EXPECT_NEAR(norm(pose_anchors.orientation_lambda(0) - cross(r_world, load)), 0.0, 1e-9)
          << "dt=" << dt << " max_iters=" << max_iters;
    }
  }
}

// Soft anchors settle at an offset of exactly compliance * load, componentwise, at any dt. At the fixed point v = 0,
// so F_ext + B y = 0 and K^-1 y = -b0, and a centre anchor's B is the identity on its force rows.
//
// The orientation rows are exact only for a torque about one axis, where the rotation vector is parallel to the
// torque and the inverse left Jacobian acts as the identity on it. Every compliance differs from every other, so a
// swap between the two families or between rows shows up as a wrong offset.
TEST(Mbody, CompliantAnchorsSettleAtComplianceTimesLoad) {
  const Vector3d position_compliance{0.01, 0.02, 0.04};
  const Vector3d position_load{0.7, -0.4, 0.25};
  const Vector3d pose_compliance{0.03, 0.05, 0.015};
  const Vector3d pose_load{-0.3, 0.6, 0.45};
  const Vector3d orientation_compliance{0.02, 0.06, 0.035};
  const Vector3d position_target{1.0, -0.5, 0.25};
  const Vector3d pose_target{-1.0, 0.75, -0.25};
  const Quaterniond pose_orientation_target{1.0, 0.0, 0.0, 0.0};
  const double torque = 0.3;

  for (int torque_axis = 0; torque_axis < 3; ++torque_axis) {
    Vector3d applied_torque{0.0, 0.0, 0.0};
    applied_torque[torque_axis] = torque;

    for (const double dt : {0.05, 0.5, 5.0}) {
      RodViews<HostExecSpace> rods(2);
      rods.center(0) = position_target;
      rods.center(1) = pose_target;
      for (int i = 0; i < 2; ++i) {
        rods.orientation(i) = pose_orientation_target;
        rods.radius(i) = 0.2;
        rods.length(i) = 0.0;
      }

      const Vector3d centre{0.0, 0.0, 0.0};
      FixedPositionViews<HostExecSpace> position_anchors(1);
      set_fixed_position(position_anchors, 0, /*rod=*/0, position_target, centre, position_compliance);
      FixedPoseViews<HostExecSpace> pose_anchors(1);
      set_fixed_pose(pose_anchors, 0, /*rod=*/1, pose_target, pose_orientation_target, centre, pose_compliance,
                     orientation_compliance);
      const auto constraints = make_constraint_set(position_anchors, pose_anchors);

      const LocalDragMobility mobility_model{.viscosity = 1.0};
      MixedLCPConfig cfg;
      cfg.max_cg_iters = 500;
      cfg.cg_tol = 1e-14;
      cfg.max_outer_iters = 1;

      rods.force(0) = position_load;
      rods.force(1) = pose_load;
      rods.torque(1) = applied_torque;
      const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
      const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
      const auto load_d = copy_load(rods_d);
      ASSERT_TRUE(step_until_settled(make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt), cfg, load_d,
                                     /*settled_step=*/1e-12, /*max_steps=*/1000))
          << "not settled about axis " << torque_axis << " at dt=" << dt;
      deep_copy(rods, rods_d);

      const Vector3d position_offset = rods.center(0) - position_target;
      const Vector3d pose_offset = rods.center(1) - pose_target;
      const Vector3d rotation_error =
          quaternion_to_rotation_vector(rods.orientation(1) * inverse(pose_orientation_target));
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(position_offset[c], position_compliance[c] * position_load[c], 1e-9)
            << "position row " << c << " at dt=" << dt;
        EXPECT_NEAR(pose_offset[c], pose_compliance[c] * pose_load[c], 1e-9) << "pose row " << c << " at dt=" << dt;
        EXPECT_NEAR(rotation_error[c], orientation_compliance[c] * applied_torque[c], 1e-9)
            << "orientation row " << c << " about axis " << torque_axis << " at dt=" << dt;
      }
    }
  }
}

// One sphere held, the other driven onto it along the line of centres. At the fixed point the contact multiplier
// balances the push and the anchor reaction is exactly minus the contact force. The contact needs PGD iterations,
// so max_outer_iters keeps its default.
TEST(Mbody, ContactAgainstAnchoredRodBalancesItsReaction) {
  constexpr double radius = 0.2;
  const double push = 0.5;
  const Vector3d anchor_target{0.0, 0.0, 0.0};

  RodViews<HostExecSpace> rods(2);
  rods.center(0) = anchor_target;
  rods.center(1) = Vector3d{0.0, 0.0, 2.0 * radius + 0.1};  // starts clear, driven into contact
  for (int i = 0; i < 2; ++i) {
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(i) = radius;
    rods.length(i) = 0.0;
  }

  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 0;
  contacts.rod_j(0) = 1;
  FixedPositionViews<HostExecSpace> anchors(1);
  set_fixed_position(anchors, 0, /*rod=*/0, anchor_target);
  const auto constraints = make_constraint_set(contacts, anchors);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 0.5;
  MixedLCPConfig cfg;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.outer_tol = 1e-12;

  rods.force(1) = Vector3d{0.0, 0.0, -push};
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  ASSERT_TRUE(step_until_settled(make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt), cfg, load_d,
                                 /*settled_step=*/1e-12, /*max_steps=*/1000));
  deep_copy(rods, rods_d);
  deep_copy(constraints, constraints_d);

  EXPECT_NEAR(norm(rods.center(0) - anchor_target), 0.0, 1e-10);
  EXPECT_NEAR(norm(rods.center(1) - rods.center(0)), 2.0 * radius, 1e-9);
  EXPECT_NEAR(contacts.lambda(0), push, 1e-9);
  // The driven sphere sits above, so the contact presses the held one downward and the anchor pulls up.
  EXPECT_NEAR(norm(Vector3d(anchors.lambda(0)) - Vector3d{0.0, 0.0, push}), 0.0, 1e-9);
}

//@}

//! \name Time integration
//@{

// Repeated steps. One linear spring between two rods relaxes as ds/dt = -A k s with A = B^T M B, and backward Euler
// steps it exactly as s_n = s0 / (1 + dt/tau)^n with tau = 1/(A k). A comes from the drag formula, not from the
// mobility operator.

/// \brief The parallel inverse drag coefficient of a rod.
double expected_inv_drag_para(double radius, double length, double viscosity) {
  const double lprime = length + 2.0 * radius;
  const double p = lprime / (2.0 * radius);
  const double log_p = std::log(p);
  const double inv_p = 1.0 / p;
  const double inv_p2 = inv_p * inv_p;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  return (log_p - 0.207 + 0.98 * inv_p - 0.133 * inv_p2) / lprime / (2.0 * pi * viscosity);
}

/// \brief The perpendicular inverse drag coefficient of a rod.
double expected_inv_drag_perp(double radius, double length, double viscosity) {
  const double lprime = length + 2.0 * radius;
  const double p = lprime / (2.0 * radius);
  const double log_p = std::log(p);
  const double inv_p = 1.0 / p;
  const double inv_p2 = inv_p * inv_p;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  return (log_p + 0.839 + 0.185 * inv_p + 0.233 * inv_p2) / lprime / (4.0 * pi * viscosity);
}

/// \brief The rotational inverse drag coefficient of a rod.
double expected_inv_drag_rot(double radius, double length, double viscosity) {
  const double lprime = length + 2.0 * radius;
  const double p = lprime / (2.0 * radius);
  const double log_p = std::log(p);
  const double inv_p = 1.0 / p;
  const double inv_p2 = inv_p * inv_p;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  return 3.0 * (log_p - 0.662 + 0.917 * inv_p - 0.05 * inv_p2) / (lprime * lprime * lprime) / (pi * viscosity);
}

/// \brief Sum over the linear springs of center distance minus rest length.
template <typename Space>
double summed_stretch(const RodViews<Space>& rods, const LinearSpringViews<Space>& lin_springs) {
  double stretch = 0.0;
  Kokkos::parallel_reduce(
      "summed_stretch", Kokkos::RangePolicy<Space>(0, lin_springs.size()),
      KOKKOS_LAMBDA(const int k, double& sum) {
        sum += norm(rods.center(lin_springs.rod_j(k)) - rods.center(lin_springs.rod_i(k))) - lin_springs.rest_length(k);
      },
      stretch);
  return stretch;
}

/// \brief A relaxation's summed stretch before and after each step, and each step's result.
struct Relaxation {
  std::vector<double> stretch;
  std::vector<MixedSLCPResult> results;
};

/// \brief Two rods joined by one linear spring, stepped num_steps times with at most max_iters linearizations each.
Relaxation run_relaxation(double spring_constant, double dt, int num_steps, unsigned max_iters, double radius = 0.2,
                          double length = 1.0, double viscosity = 1.0) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 1.6},
                          Quaterniond{1.0, 0.0, 0.0, 0.0}, radius, length);

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = spring_constant;

  const auto constraints = make_constraint_set(lin_springs);

  const LocalDragMobility mobility_model{.viscosity = viscosity};
  MixedSLCPConfig cfg;
  cfg.inner_lcp_config.max_cg_iters = 500;
  cfg.inner_lcp_config.cg_tol = 1e-12;
  cfg.inner_lcp_config.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.max_iters = max_iters;
  cfg.length_tol = 1e-9;
  cfg.angle_tol = 1e-9;

  zero_rod_state(rods);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto& lin_springs_d = get<LinearSpringViews<TestExecSpace>>(constraints_d);
  const auto load_d = copy_load(rods_d);
  const auto integrator = make_mixed_slcp_integrator(rods_d, constraints_d, mobility_model, dt);

  Relaxation relaxation{{summed_stretch(rods_d, lin_springs_d)}, {}};
  for (int step = 0; step < num_steps; ++step) {
    relaxation.results.push_back(step_rods(integrator, cfg, load_d));
    EXPECT_TRUE(lcp_converged(relaxation.results.back())) << relaxation.results.back() << " at step " << step;
    relaxation.stretch.push_back(summed_stretch(rods_d, lin_springs_d));
  }
  return relaxation;
}

// The stretch follows the backward-Euler closed form exactly, and the continuous decay s0 exp(-t/tau) closely at this
// modest dt/tau.
TEST(Mbody, SpringRelaxationMatchesBackwardEulerAndExponentialDecay) {
  const double radius = 0.2, length = 1.0, viscosity = 1.0, spring_constant = 2.0;
  const double A = 2.0 * expected_inv_drag_para(radius, length, viscosity);  // both rods move
  const double tau = 1.0 / (A * spring_constant);

  const double dt = 0.2 * tau;
  const int num_steps = 10;
  const std::vector<double> stretch =
      run_relaxation(spring_constant, dt, num_steps, /*max_iters=*/1, radius, length, viscosity).stretch;

  const double s0 = stretch[0];
  for (int n = 0; n <= num_steps; ++n) {
    const double backward_euler = s0 / std::pow(1.0 + dt / tau, n);
    EXPECT_NEAR(stretch[n], backward_euler, 1e-8 + 1e-6 * std::abs(backward_euler)) << "step " << n;

    const double continuous = s0 * std::exp(-n * dt / tau);
    EXPECT_NEAR(stretch[n], continuous, 0.05 * s0) << "step " << n << " (vs. continuous exp(-t/tau))";
  }
}

// From dt = 0.1 tau to 50 tau the stretch follows the closed form and decays monotonically, never overshooting zero,
// for either sequence length. The rods stay on the z axis, so the spring keeps its direction through every step and its
// first linearization holds at the end of the step: every sequence accepts it.
TEST(Mbody, SpringRelaxationStableAcrossWideDtSweep) {
  const double radius = 0.2, length = 1.0, viscosity = 1.0, spring_constant = 2.0;
  const double A = 2.0 * expected_inv_drag_para(radius, length, viscosity);
  const double tau = 1.0 / (A * spring_constant);

  for (const double dt_over_tau : {0.1, 1.0, 5.0, 20.0, 50.0}) {
    for (const unsigned max_iters : {1u, 50u}) {
      const double dt = dt_over_tau * tau;
      const int num_steps = 6;
      const Relaxation relaxation =
          run_relaxation(spring_constant, dt, num_steps, max_iters, radius, length, viscosity);
      const std::vector<double>& stretch = relaxation.stretch;

      const double s0 = stretch[0];
      for (int n = 0; n <= num_steps; ++n) {
        const double backward_euler = s0 / std::pow(1.0 + dt / tau, n);
        EXPECT_NEAR(stretch[n], backward_euler, 1e-8 + 1e-6 * std::abs(backward_euler))
            << "dt/tau=" << dt_over_tau << " max_iters=" << max_iters << " step " << n;
      }
      for (int n = 1; n <= num_steps; ++n) {
        EXPECT_LE(std::abs(stretch[n]), std::abs(stretch[n - 1]) + 1e-12)
            << "dt/tau=" << dt_over_tau << " max_iters=" << max_iters << ": stretch grew at step " << n
            << " (unstable)";
        EXPECT_GE(stretch[n], 0.0) << "dt/tau=" << dt_over_tau << " max_iters=" << max_iters
                                   << ": stretch went negative at step " << n
                                   << " (should decay monotonically toward zero, never overshoot)";
      }
      for (size_t n = 0; n < relaxation.results.size(); ++n) {
        EXPECT_TRUE(relaxation.results[n].converged)
            << relaxation.results[n] << " at dt/tau=" << dt_over_tau << " max_iters=" << max_iters << " step " << n;
        EXPECT_EQ(relaxation.results[n].num_iters, 1u)
            << "dt/tau=" << dt_over_tau << " max_iters=" << max_iters << " step " << n;
      }
    }
  }
}

/// \brief lambda_max(K B^T M B) for the springs of p, from K^1/2 B^T M B K^1/2, which shares its spectrum.
template <typename... Families>
double spring_network_stiffness(const SolveInput<Families...>& p) {
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto& lin_springs = get<LinearSpringViews<HostExecSpace>>(p.constraints);
  const auto& ang_springs = get<AngularSpringViews<HostExecSpace>>(p.constraints);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto lin_springs_d = create_mirror_view_and_copy(TestExecSpace{}, lin_springs);
  const auto ang_springs_d = create_mirror_view_and_copy(TestExecSpace{}, ang_springs);
  const auto springs_d = make_constraint_set(lin_springs_d, ang_springs_d);
  const auto index_map = impl::make_constraint_index_map(springs_d);
  const auto block = impl::make_block_geometry<ConstraintType::BILATERAL, TestExecSpace>(index_map);
  const Kokkos::View<double*, TestMemSpace> b0("b0", index_map.num_bilateral);
  impl::compute_block_geometry<ConstraintType::BILATERAL>(rods_d, springs_d, index_map, block, b0);
  const impl::PairGeometry<TestExecSpace>& geo = ::mundy::get<0>(block.groups);
  const impl::PairForceOp<TestExecSpace> B(geo, p.rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, p.rods.size());
  const LocalDragMobilityOp<TestExecSpace> M = p.mobility_model.make_mobility(rods_d);

  // Spring rows are packed linear then angular, matching the pair group.
  const size_t num_linear = lin_springs.size();
  const size_t num_springs = geo.size();
  Kokkos::View<double*, Kokkos::HostSpace> sqrt_k("sqrt_k", num_springs), q0("q0", num_springs);
  std::mt19937 rng(20261001);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t row = 0; row < num_springs; ++row) {
    sqrt_k(row) =
        std::sqrt(row < num_linear ? lin_springs.spring_constant(row) : ang_springs.spring_constant(row - num_linear));
    q0(row) = dist(rng);
  }
  const auto K_half = make_diagonal_op<backend_t>(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, sqrt_k));
  const auto S = make_quadratic_form<backend_t>(K_half, make_quadratic_form<backend_t>(BT, M, B), K_half);

  auto prob = make_eigen_problem<backend_t>(S);
  auto state = make_power_state(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, q0),
                                Kokkos::View<double*, TestMemSpace>("z", num_springs),
                                Kokkos::View<double*, TestMemSpace>("r", num_springs));
  const auto strat = make_power_strategy(RelativeL2Residual{}, PowerConfig<double>{.max_iters = 100000, .tol = 1e-10});
  const auto result = solve_eigen_problem(prob, strat, state);
  EXPECT_TRUE(result.converged) << result;
  return result.eigenvalue;
}

// Explicit Euler on the linearized springs is stable only for dt < dt_crit = 2 / lambda_max(K B^T M B). On both sides
// of dt_crit every backward-Euler step lowers the elastic energy, since it maps K^1/2 b to
// (I + dt K^1/2 B^T M B K^1/2)^-1 K^1/2 b. A converged sequence is backward Euler on the springs' nonlinear gradient
// flow, whose step is guaranteed to lower the energy only where it minimizes E(x) + |x - x_k|^2 / (2 dt) in the metric
// of M^-1, so the decrease is checked here rather than assumed; an unconverged sequence returns the first
// linearization.
TEST(Mbody, ChainStableAcrossExplicitStabilityLimit) {
  const double dt_crit = 2.0 / spring_network_stiffness(make_chain_problem(/*spring_constant=*/3.0));

  for (const double cfl : {0.5, 0.9, 1.1, 2.0, 10.0, 100.0}) {
    for (const unsigned max_iters : {1u, 50u}) {
      auto p = make_chain_problem(/*spring_constant=*/3.0);
      p.dt = cfl * dt_crit;
      p.cfg.cg_tol = 1e-13;
      const MixedSLCPConfig cfg{p.cfg, max_iters, 1e-9, 1e-9};

      zero_rod_state(p.rods);  // no external load: the springs relax toward rest
      const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
      const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
      const auto& lin_springs_d = get<LinearSpringViews<TestExecSpace>>(constraints_d);
      const auto& ang_springs_d = get<AngularSpringViews<TestExecSpace>>(constraints_d);
      const auto load_d = copy_load(rods_d);
      const auto integrator = make_mixed_slcp_integrator(rods_d, constraints_d, p.mobility_model, p.dt);

      std::vector<double> energy{elastic_energy(rods_d, lin_springs_d, ang_springs_d)};
      for (int step = 0; step < 40; ++step) {
        const MixedSLCPResult result = step_rods(integrator, cfg, load_d);
        ASSERT_TRUE(lcp_converged(result))
            << result << " at cfl=" << cfl << " max_iters=" << max_iters << " step " << step;
        energy.push_back(elastic_energy(rods_d, lin_springs_d, ang_springs_d));
      }

      // Round-off floor relative to the initial energy, which the largest steps decay toward.
      const double floor = 1e-10 * energy.front();
      for (size_t n = 0; n + 1 < energy.size(); ++n) {
        EXPECT_LE(energy[n + 1], energy[n] + floor)
            << "cfl=" << cfl << " max_iters=" << max_iters << ": energy rose at step " << n + 1;
      }
      EXPECT_LT(energy.back(), energy.front()) << "cfl=" << cfl << " max_iters=" << max_iters;
    }
  }
}

//@}

//! \name Sequential linearization
//@{

// MixedSLCPIntegrator: the bilateral rows re-linearized at the configurations its iterates move the rods to.

/// \brief Five rods under every bilateral family and one pressing contact, all force directions turning with the rods.
///
/// Rods 0-3 are held by a pose anchor and joined by a spring, a pin and a fixed length; rod 3 presses onto a sphere
/// held by a position anchor.
auto make_fallback_problem() {
  constexpr size_t kNumRods = 5;
  const Vector3d tilt_axis{0.6, 0.8, 0.0};
  const double tilts[4] = {0.0, 0.3, -0.5, 0.8};

  RodViews<HostExecSpace> rods(kNumRods);
  for (size_t k = 0; k < 4; ++k) {
    const double s = static_cast<double>(k);
    rods.center(k) = Vector3d{0.1 * s, -0.05 * s, 1.1 * s};
    rods.orientation(k) = axis_angle_to_quaternion(tilt_axis, tilts[k]);
    rods.radius(k) = 0.2;
    rods.length(k) = 1.0;
  }
  rods.center(4) = Vector3d(rods.center(3)) + Vector3d{0.35, 0.0, 0.0};  // within 0.35 of rod 3's centerline
  rods.orientation(4) = Quaterniond{1.0, 0.0, 0.0, 0.0};
  rods.radius(4) = 0.2;
  rods.length(4) = 0.0;
  zero_rod_state(rods);
  rods.force(2) = Vector3d{0.3, -0.2, 0.1};
  rods.force(3) = Vector3d{0.4, 0.25, 0.2};
  rods.torque(3) = Vector3d{0.05, -0.1, 0.08};

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = 3.0;
  PinViews<HostExecSpace> pins(1);
  pins.rod_i(0) = 1;
  pins.rod_j(0) = 2;
  pins.body_offset_i(0) = Vector3d{0.0, 0.0, 0.5};
  pins.body_offset_j(0) = Vector3d{0.0, 0.0, -0.5};
  FixedLengthViews<HostExecSpace> lengths(1);
  lengths.rod_i(0) = 2;
  lengths.rod_j(0) = 3;
  lengths.body_offset_i(0) = Vector3d{0.1, 0.0, 0.4};
  lengths.body_offset_j(0) = Vector3d{0.0, -0.1, -0.4};
  lengths.rest_length(0) = 0.35;
  FixedPoseViews<HostExecSpace> pose_anchors(1);
  set_fixed_pose(pose_anchors, 0, /*rod=*/0, Vector3d(rods.center(0)), Quaterniond(rods.orientation(0)),
                 /*body_offset=*/Vector3d{0.0, 0.0, -0.5});
  FixedPositionViews<HostExecSpace> position_anchors(1);
  set_fixed_position(position_anchors, 0, /*rod=*/4, Vector3d(rods.center(4)));
  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 3;
  contacts.rod_j(0) = 4;

  const double dt = 0.3;
  MixedLCPConfig cfg;
  cfg.max_cg_iters = 1000;
  cfg.cg_tol = 1e-10;
  cfg.outer_tol = 1e-10;
  return SolveInput{rods, make_constraint_set(lin_springs, pins, lengths, pose_anchors, position_anchors, contacts),
                    LocalDragMobility{.viscosity = 1.0}, dt, cfg};
}

// A sequence that cannot converge returns its first linearization, which is the mixed LCP step, bit for bit. Serial
// execution fixes the order of every atomic sum, so the two solves round identically.
TEST(Mbody, SlcpFallsBackToFirstLinearization) {
  const auto p = make_fallback_problem();
  const auto rods_lcp = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_lcp = copy_to<Kokkos::Serial>(p.constraints);
  const auto rods_slcp = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_slcp = copy_to<Kokkos::Serial>(p.constraints);

  // Solve
  const MixedLCPResult lcp =
      solve_step(make_mixed_lcp_integrator(rods_lcp, constraints_lcp, p.mobility_model, p.dt), p.cfg);
  const MixedSLCPResult slcp =
      solve_step(make_mixed_slcp_integrator(rods_slcp, constraints_slcp, p.mobility_model, p.dt),
                 MixedSLCPConfig{p.cfg, 3, 1e-300, 1e-300});
  ASSERT_FALSE(slcp.converged) << slcp;
  EXPECT_GE(slcp.num_iters, 2u);
  ASSERT_GT(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                get<ContactViews<Kokkos::Serial>>(constraints_lcp).lambda_view())(0),
            0.0)
      << "the contact should press";

  // The returned step
  EXPECT_EQ(slcp.accepted_lcp_result.num_iters, lcp.num_iters);
  EXPECT_EQ(std::bit_cast<uint64_t>(slcp.accepted_lcp_result.residual), std::bit_cast<uint64_t>(lcp.residual));
  EXPECT_EQ(slcp.accepted_lcp_result.converged, lcp.converged);
  EXPECT_EQ(count_bit_differences(rods_slcp.force_torque_view(), rods_lcp.force_torque_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_slcp.velocity_omega_view(), rods_lcp.velocity_omega_view()), 0u);
  EXPECT_EQ(count_bit_differences(get<ContactViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<ContactViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<LinearSpringViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<LinearSpringViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<PinViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<PinViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedLengthViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<FixedLengthViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedPositionViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<FixedPositionViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedPoseViews<Kokkos::Serial>>(constraints_slcp).lambda_view(),
                                  get<FixedPoseViews<Kokkos::Serial>>(constraints_lcp).lambda_view()),
            0u);
}

/// \brief The label and size of every Kokkos allocation f makes, with its multiplicity.
template <typename F>
std::map<std::pair<std::string, uint64_t>, size_t> record_allocations(F&& f) {
  static std::map<std::pair<std::string, uint64_t>, size_t>* recording = nullptr;
  std::map<std::pair<std::string, uint64_t>, size_t> allocations;
  recording = &allocations;
  Kokkos::Tools::Experimental::set_init_callback(
      [](const int, const uint64_t, const uint32_t, Kokkos_Profiling_KokkosPDeviceInfo*) {});
  Kokkos::Tools::Experimental::set_allocate_data_callback([](const Kokkos_Profiling_SpaceHandle, const char* label,
                                                             const void*,
                                                             const uint64_t size) { ++(*recording)[{label, size}]; });
  f();
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_init_callback(nullptr);
  recording = nullptr;
  return allocations;
}

/// \brief A mixed SLCP step of size dt through an integrator of its own on TestExecSpace copies of rods and
/// constraints, and the allocations it makes.
template <typename Model, typename Policy = NoPreconditioner, typename... Families>
std::pair<MixedSLCPResult, std::map<std::pair<std::string, uint64_t>, size_t>> slcp_allocations(
    const RodViews<HostExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const MixedSLCPConfig& cfg, const Policy& preconditioner_policy = Policy{}) {
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(constraints);
  MixedSLCPResult result;
  const auto allocations = record_allocations([&] {
    result =
        solve_step(make_mixed_slcp_integrator(rods_d, constraints_d, mobility_model, dt, preconditioner_policy), cfg);
  });
  return {result, allocations};
}

/// \brief A mixed LCP step of size dt through an integrator of its own on TestExecSpace copies of rods and
/// constraints, and the allocations it makes.
template <typename Model, typename... Families>
std::pair<MixedLCPResult, std::map<std::pair<std::string, uint64_t>, size_t>> lcp_allocations(
    const RodViews<HostExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const MixedLCPConfig& cfg) {
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(constraints);
  MixedLCPResult result;
  const auto allocations = record_allocations(
      [&] { result = solve_step(make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt), cfg); });
  return {result, allocations};
}

// A solve allocates only before its first iteration: solves that differ in how many SLCP iterates or PGD iterations
// they take make the same allocations. An unrecorded solve first sizes Kokkos' own scratch.
TEST(Mbody, IterationsDoNotAllocate) {
  // Both blocks, and the bilateral block alone, by sequences of different lengths
  const auto p = make_fallback_problem();
  const auto& c = p.constraints;
  const auto bilateral_only =
      make_constraint_set(get<LinearSpringViews<HostExecSpace>>(c), get<PinViews<HostExecSpace>>(c),
                          get<FixedLengthViews<HostExecSpace>>(c), get<FixedPoseViews<HostExecSpace>>(c),
                          get<FixedPositionViews<HostExecSpace>>(c));
  slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  const auto [both_short, both_short_allocations] =
      slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-5, 1e-5});
  const auto [both_long, both_long_allocations] =
      slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  ASSERT_TRUE(both_short.converged && both_long.converged) << both_short << "\n" << both_long;
  ASSERT_LT(both_short.num_iters, both_long.num_iters);
  EXPECT_EQ(both_short_allocations, both_long_allocations);

  // Both blocks, with the Schur complement preconditioned by its self-mobility Jacobi
  slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9}, SelfMobilityJacobi{});
  const auto [jacobi_short, jacobi_short_allocations] =
      slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-5, 1e-5}, SelfMobilityJacobi{});
  const auto [jacobi_long, jacobi_long_allocations] =
      slcp_allocations(p.rods, c, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9}, SelfMobilityJacobi{});
  ASSERT_TRUE(jacobi_short.converged && jacobi_long.converged) << jacobi_short << "\n" << jacobi_long;
  ASSERT_LT(jacobi_short.num_iters, jacobi_long.num_iters);
  EXPECT_EQ(jacobi_short_allocations, jacobi_long_allocations);

  slcp_allocations(p.rods, bilateral_only, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  const auto [bilateral_short, bilateral_short_allocations] =
      slcp_allocations(p.rods, bilateral_only, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-5, 1e-5});
  const auto [bilateral_long, bilateral_long_allocations] =
      slcp_allocations(p.rods, bilateral_only, p.mobility_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  ASSERT_TRUE(bilateral_short.converged && bilateral_long.converged) << bilateral_short << "\n" << bilateral_long;
  ASSERT_LT(bilateral_short.num_iters, bilateral_long.num_iters);
  EXPECT_EQ(bilateral_short_allocations, bilateral_long_allocations);

  // The bilateral block alone under a caller's mobility, which has only a plain apply
  const IsotropicMobility isotropic_model{.m = 0.25, .m_rot = 1.0};
  slcp_allocations(p.rods, bilateral_only, isotropic_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  const auto [isotropic_short, isotropic_short_allocations] =
      slcp_allocations(p.rods, bilateral_only, isotropic_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-5, 1e-5});
  const auto [isotropic_long, isotropic_long_allocations] =
      slcp_allocations(p.rods, bilateral_only, isotropic_model, p.dt, MixedSLCPConfig{p.cfg, 50, 1e-9, 1e-9});
  ASSERT_TRUE(isotropic_short.converged && isotropic_long.converged) << isotropic_short << "\n" << isotropic_long;
  ASSERT_LT(isotropic_short.num_iters, isotropic_long.num_iters);
  EXPECT_EQ(isotropic_short_allocations, isotropic_long_allocations);

  // The unilateral block alone, by PGD to different tolerances
  const auto row = make_sphere_row_problem();
  MixedLCPConfig loose = row.cfg;
  loose.outer_tol = 1e-4;
  MixedLCPConfig tight = row.cfg;
  tight.outer_tol = 1e-12;
  lcp_allocations(row.rods, row.constraints, row.mobility_model, row.dt, tight);
  const auto [unilateral_loose, unilateral_loose_allocations] =
      lcp_allocations(row.rods, row.constraints, row.mobility_model, row.dt, loose);
  const auto [unilateral_tight, unilateral_tight_allocations] =
      lcp_allocations(row.rods, row.constraints, row.mobility_model, row.dt, tight);
  ASSERT_TRUE(unilateral_loose.converged && unilateral_tight.converged) << unilateral_loose << "\n" << unilateral_tight;
  ASSERT_LT(unilateral_loose.num_iters, unilateral_tight.num_iters);
  EXPECT_EQ(unilateral_loose_allocations, unilateral_tight_allocations);
}

/// \brief A pendulum's angle from -y before and after each step, its constraint error after each, and their results.
struct PendulumRun {
  std::vector<double> theta;
  std::vector<double> constraint_error;
  std::vector<MixedSLCPResult> results;
};

constexpr double kPendulumTheta0 = 1.2;
constexpr double kPendulumArm = 1.0;  // pivot to loaded point
constexpr double kPendulumCgTol = 1e-14;
constexpr LocalDragMobility kPendulumMobilityModel{.viscosity = 1.0};

/// \brief A pendulum step by a sequence of at most max_iters linearizations.
MixedSLCPConfig make_pendulum_config(unsigned max_iters, double length_tol) {
  MixedSLCPConfig cfg;
  cfg.inner_lcp_config.max_cg_iters = 200;
  cfg.inner_lcp_config.cg_tol = kPendulumCgTol;
  cfg.inner_lcp_config.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.max_iters = max_iters;
  cfg.length_tol = length_tol;
  cfg.angle_tol = length_tol;
  return cfg;
}

/// \brief A sphere held by a fixed length from an anchored sphere, stepped num_steps times at dt = h tau, with the
/// Schur complement preconditioned by its self-mobility Jacobi if preconditioned, through one integrator's storage held
/// across the steps if held_storage and through storage of each step's own otherwise.
///
/// The bob, at L = kPendulumArm under a constant force f, turns in the plane normal to its axis, where its mobility is
/// m I, so tau = L / (m f). Its constraint error is |r| - L.
PendulumRun run_bob_pendulum(double h, int num_steps, unsigned max_iters, double length_tol, bool preconditioned,
                             bool held_storage) {
  const double radius = 0.2, f = 0.5, L = kPendulumArm;
  const double tau = L / (expected_inv_drag_perp(radius, 0.0, kPendulumMobilityModel.viscosity) * f);

  RodViews<HostExecSpace> rods(2);
  rods.center(0) = Vector3d{0.0, 0.0, 0.0};
  rods.center(1) = Vector3d{L * std::sin(kPendulumTheta0), -L * std::cos(kPendulumTheta0), 0.0};
  for (int i = 0; i < 2; ++i) {
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(i) = radius;
    rods.length(i) = 0.0;
  }
  zero_rod_state(rods);
  rods.force(1) = Vector3d{0.0, -f, 0.0};

  FixedPositionViews<HostExecSpace> anchors(1);
  set_fixed_position(anchors, 0, /*rod=*/0, Vector3d{0.0, 0.0, 0.0});
  FixedLengthViews<HostExecSpace> lengths(1);
  lengths.rod_i(0) = 0;
  lengths.rod_j(0) = 1;
  lengths.rest_length(0) = L;

  const double dt = h * tau;
  const MixedSLCPConfig cfg = make_pendulum_config(max_iters, length_tol);
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(make_constraint_set(anchors, lengths));
  const auto load_d = copy_load(rods_d);

  PendulumRun run{{kPendulumTheta0}, {}, {}};
  const auto run_steps = [&](const auto& policy) {
    const auto integrator = make_mixed_slcp_integrator(rods_d, constraints_d, kPendulumMobilityModel, dt, policy);
    for (int step = 0; step < num_steps; ++step) {
      run.results.push_back(held_storage ? step_rods(integrator, cfg, load_d)
                                         : step_rods(with_fresh_storage(integrator), cfg, load_d));
      deep_copy(rods, rods_d);
      const Vector3d r = rods.center(1) - rods.center(0);
      run.theta.push_back(std::atan2(r[0], -r[1]));
      run.constraint_error.push_back(norm(r) - L);
    }
  };
  if (preconditioned) {
    run_steps(SelfMobilityJacobi{});
  } else {
    run_steps(NoPreconditioner{});
  }
  return run;
}

/// \brief A rod pinned at one end to an anchored sphere, stepped num_steps times at dt = h tau, with the Schur
/// complement preconditioned by its self-mobility Jacobi if preconditioned.
///
/// The rod, of half length a = kPendulumArm under a constant force f at its center, turns about its pinned end in the
/// plane of its axis and the force, so tau = (m_perp + m_rot a^2) / (a m_perp m_rot f); theta is the angle of its
/// axis. Its constraint error is the distance between the pinned points.
PendulumRun run_rod_pendulum(double h, int num_steps, unsigned max_iters, double length_tol, bool preconditioned) {
  const double radius = 0.1, f = 1.0, a = kPendulumArm;
  const double m_perp = expected_inv_drag_perp(radius, 2.0 * a, kPendulumMobilityModel.viscosity);
  const double m_rot = expected_inv_drag_rot(radius, 2.0 * a, kPendulumMobilityModel.viscosity);
  const double tau = (m_perp + m_rot * a * a) / (a * m_perp * m_rot * f);
  const Vector3d axis_body{0.0, 0.0, 1.0};

  RodViews<HostExecSpace> rods(2);
  rods.center(0) = Vector3d{0.0, 0.0, 0.0};
  rods.orientation(0) = Quaterniond{1.0, 0.0, 0.0, 0.0};
  rods.radius(0) = radius;
  rods.length(0) = 0.0;
  rods.center(1) = Vector3d{a * std::sin(kPendulumTheta0), -a * std::cos(kPendulumTheta0), 0.0};
  rods.orientation(1) = axis_angle_to_quaternion(Vector3d{std::cos(kPendulumTheta0), std::sin(kPendulumTheta0), 0.0},
                                                 Kokkos::numbers::pi_v<double> / 2.0);  // axis from pivot to center
  rods.radius(1) = radius;
  rods.length(1) = 2.0 * a;
  zero_rod_state(rods);
  rods.force(1) = Vector3d{0.0, -f, 0.0};

  FixedPositionViews<HostExecSpace> anchors(1);
  set_fixed_position(anchors, 0, /*rod=*/0, Vector3d{0.0, 0.0, 0.0});
  PinViews<HostExecSpace> pins(1);
  pins.rod_i(0) = 0;
  pins.rod_j(0) = 1;
  pins.body_offset_i(0) = Vector3d{0.0, 0.0, 0.0};
  pins.body_offset_j(0) = -a * axis_body;

  const double dt = h * tau;
  const MixedSLCPConfig cfg = make_pendulum_config(max_iters, length_tol);
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(make_constraint_set(anchors, pins));
  const auto load_d = copy_load(rods_d);

  PendulumRun run{{kPendulumTheta0}, {}, {}};
  for (int step = 0; step < num_steps; ++step) {
    run.results.push_back(
        preconditioned
            ? step_rods(
                  make_mixed_slcp_integrator(rods_d, constraints_d, kPendulumMobilityModel, dt, SelfMobilityJacobi{}),
                  cfg, load_d)
            : step_rods(make_mixed_slcp_integrator(rods_d, constraints_d, kPendulumMobilityModel, dt), cfg, load_d));
    deep_copy(rods, rods_d);
    const Vector3d t = rods.orientation(1) * axis_body;
    run.theta.push_back(std::atan2(t[0], -t[1]));
    run.constraint_error.push_back(norm(rods.center(1) - a * t - rods.center(0)));
  }
  return run;
}

/// \brief The two realizations of a pendulum.
enum class Pendulum { BOB, PINNED_ROD };

/// \brief A realization's name, for failure messages.
const char* pendulum_name(Pendulum pendulum) {
  return pendulum == Pendulum::BOB ? "bob" : "pinned rod";
}

/// \brief A pendulum of either realization, stepped num_steps times at dt = h tau, with the Schur complement
/// preconditioned by its self-mobility Jacobi if preconditioned.
PendulumRun run_pendulum(Pendulum pendulum, double h, int num_steps, unsigned max_iters, double length_tol,
                         bool preconditioned) {
  return pendulum == Pendulum::BOB
             ? run_bob_pendulum(h, num_steps, max_iters, length_tol, preconditioned, /*held_storage=*/false)
             : run_rod_pendulum(h, num_steps, max_iters, length_tol, preconditioned);
}

// For the bob, with h = dt / tau, both step maps are exact. A single linearization moves it along the tangent at the
// start of the step, and a converged sequence ends where the constraint holds with its force directed along it there:
//   single:   theta' = theta - atan(h sin theta),          |r'| = L sqrt(1 + h^2 sin^2 theta)
//   sequence: tan theta' = sin theta / (cos theta + h),    |r'| = L
// The single step's Schur residual, at most cg_tol in length, bounds each linearized row at the end of the step, so the
// bob and the anchored sphere each sit within cg_tol of it. An accepted sequence's force direction changes the step by
// at most sqrt(2) length_tol in the plane, which turns the bob by at most sqrt(2) length_tol / |r + dt m F|.
//
// At h = 5, |r + dt m F| >= (h - 1) L, so the sequence would contract at a rate |L - |r + dt m F|| / L >= 3: it stops
// early, and the step is the single linearization.
//
// Preconditioning the Schur complement, or starting its solves from the previous step's through an integrator's held
// storage, changes how CG reaches cg_tol, not the bound, so every map holds with and without either.
TEST(Mbody, PendulumBobStepsFollowExactMaps) {
  const double h = 0.1, L = kPendulumArm, length_tol = 1e-11;
  const int num_steps = 10;

  for (const bool preconditioned : {false, true}) {
    for (const bool held_storage : {false, true}) {
      // Single linearization
      const PendulumRun single =
          run_bob_pendulum(h, num_steps, /*max_iters=*/1, length_tol, preconditioned, held_storage);
      for (int k = 0; k < num_steps; ++k) {
        const double theta = single.theta[k];
        EXPECT_TRUE(lcp_converged(single.results[k]))
            << single.results[k] << " at step " << k << " preconditioned=" << preconditioned
            << " held_storage=" << held_storage;
        EXPECT_NEAR(single.theta[k + 1], theta - std::atan(h * std::sin(theta)), 2.0 * kPendulumCgTol / L)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
        EXPECT_NEAR(single.constraint_error[k], L * std::sqrt(1.0 + h * h * std::sin(theta) * std::sin(theta)) - L,
                    2.0 * kPendulumCgTol)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
      }

      // Sequence
      const PendulumRun sequence =
          run_bob_pendulum(h, num_steps, /*max_iters=*/50, length_tol, preconditioned, held_storage);
      for (int k = 0; k < num_steps; ++k) {
        const double theta = sequence.theta[k];
        const double p_norm = L * std::sqrt(1.0 + 2.0 * h * std::cos(theta) + h * h);
        ASSERT_TRUE(sequence.results[k].converged)
            << sequence.results[k] << " at step " << k << " preconditioned=" << preconditioned
            << " held_storage=" << held_storage;
        EXPECT_GE(sequence.results[k].num_iters, 2u)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
        EXPECT_LE(std::abs(sequence.constraint_error[k]), length_tol)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
        EXPECT_NEAR(sequence.theta[k + 1], std::atan2(std::sin(theta), std::cos(theta) + h),
                    std::sqrt(2.0) * length_tol / p_norm)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
      }

      // Beyond the convergence regime
      const double h_beyond = 5.0;
      const PendulumRun beyond =
          run_bob_pendulum(h_beyond, 12, /*max_iters=*/50, length_tol, preconditioned, held_storage);
      for (size_t k = 0; k < beyond.results.size(); ++k) {
        const double theta = beyond.theta[k];
        EXPECT_TRUE(lcp_converged(beyond.results[k]))
            << beyond.results[k] << " at step " << k << " preconditioned=" << preconditioned
            << " held_storage=" << held_storage;
        EXPECT_NEAR(beyond.theta[k + 1], theta - std::atan(h_beyond * std::sin(theta)), 2.0 * kPendulumCgTol / L)
            << "step " << k << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
      }
    }
  }
}

// A pendulum under a constant force follows d theta/dt = -sin(theta) / tau, so theta(T) = 2 atan(tan(theta0 / 2) e^-T)
// with T in units of tau, in two realizations: a bob on a fixed length and a rod pinned at one end, which also turns
// (see run_bob_pendulum and run_rod_pendulum). A single linearization moves the loaded point along the tangent at the
// start of the step, forward Euler on theta to second order in the step; a converged sequence ends where the constraint
// holds with its force directed along it there, backward Euler to second order.
//
// The first-order error coefficients c = (theta_N - theta(T)) / h of the two schemes tend to -E and +E, with
// E = sin(theta(T)) ln(sin theta0 / sin theta(T)) / 2, for either realization. Richardson extrapolation 2 c(2N) - c(N)
// removes the O(h) term of c, so in the asymptotic range what remains is below |c(N) - c(2N)|; a per-step angle error
// delta adds at most 9 N^2 delta / T. Preconditioning the Schur complement leaves delta's bound unchanged.
TEST(Mbody, PendulumRefinesToEulerErrorCoefficients) {
  const double T = 1.5;
  const double theta_T = 2.0 * std::atan(std::tan(kPendulumTheta0 / 2.0) * std::exp(-T));
  const double E = 0.5 * std::sin(theta_T) * std::log(std::sin(kPendulumTheta0) / std::sin(theta_T));
  const double length_tol = 1e-12;
  const int num_steps[3] = {200, 400, 800};

  for (const bool preconditioned : {false, true}) {
    for (const Pendulum pendulum : {Pendulum::BOB, Pendulum::PINNED_ROD}) {
      for (const unsigned max_iters : {1u, 50u}) {
        const bool sequence = max_iters > 1;

        // Every step at every level
        double coefficient[3];
        for (int level = 0; level < 3; ++level) {
          const double h = T / num_steps[level];
          const PendulumRun run = run_pendulum(pendulum, h, num_steps[level], max_iters, length_tol, preconditioned);
          coefficient[level] = (run.theta.back() - theta_T) / h;
          for (size_t k = 0; k < run.results.size(); ++k) {
            ASSERT_TRUE(lcp_converged(run.results[k]))
                << run.results[k] << " for the " << pendulum_name(pendulum) << " at N=" << num_steps[level] << " step "
                << k << " preconditioned=" << preconditioned;
            if (sequence) {
              ASSERT_TRUE(run.results[k].converged)
                  << run.results[k] << " for the " << pendulum_name(pendulum) << " at N=" << num_steps[level]
                  << " step " << k << " preconditioned=" << preconditioned;
              EXPECT_LE(std::abs(run.constraint_error[k]), length_tol)
                  << "for the " << pendulum_name(pendulum) << " at N=" << num_steps[level] << " step " << k
                  << " preconditioned=" << preconditioned;
            }
          }
        }

        // Refinement
        const double delta =
            sequence ? std::sqrt(2.0) * length_tol / kPendulumArm : 2.0 * kPendulumCgTol / kPendulumArm;
        for (int level = 0; level + 1 < 3; ++level) {
          const double n = num_steps[level];
          const double richardson = 2.0 * coefficient[level + 1] - coefficient[level];
          EXPECT_NEAR(richardson, sequence ? E : -E,
                      std::abs(coefficient[level] - coefficient[level + 1]) + 9.0 * n * n * delta / T)
              << "N=" << num_steps[level] << " for the " << pendulum_name(pendulum) << " at max_iters=" << max_iters
              << " preconditioned=" << preconditioned;
        }
      }
    }
  }
}

// Beyond the convergence regime, at h = 5, each realization's sequence stops before max_iters once its contraction
// predicts it cannot converge, and returns its first linearization unconverged.
TEST(Mbody, PendulumSequencesStopEarlyBeyondConvergence) {
  const double h = 5.0, length_tol = 1e-12;
  const unsigned max_iters = 50;
  for (const Pendulum pendulum : {Pendulum::BOB, Pendulum::PINNED_ROD}) {
    const PendulumRun run = run_pendulum(pendulum, h, 12, max_iters, length_tol, /*preconditioned=*/false);
    for (size_t k = 0; k < run.results.size(); ++k) {
      EXPECT_TRUE(lcp_converged(run.results[k])) << run.results[k] << " for the " << pendulum_name(pendulum);
      EXPECT_FALSE(run.results[k].converged) << "for the " << pendulum_name(pendulum) << " at step " << k;
      EXPECT_LT(run.results[k].num_iters, max_iters) << "for the " << pendulum_name(pendulum) << " at step " << k;
    }
  }
}

/// \brief The largest |psi + K^-1 lambda| over family's rows at rods' configuration, over its length rows and angle
/// rows.
template <typename Space, typename Family>
impl::LengthAngleMax constitutive_residual(const RodViews<Space>& rods, const Family& family) {
  Kokkos::View<double*, typename Space::memory_space> psi("psi", family.num_rows());
  impl::compute_geometry(rods, family, psi);
  const auto lambda = family.lambda_view();
  double length_max = 0.0;
  double angle_max = 0.0;
  Kokkos::parallel_reduce(
      "constitutive_residual_length", Kokkos::RangePolicy<Space>(0, family.num_rows()),
      KOKKOS_LAMBDA(const int row, double& m) {
        if (Family::row_unit(row % Family::rows_per_entry) == RowUnit::LENGTH) {
          m = Kokkos::max(m, Kokkos::abs(psi(row) + family.row_compliance(row) * lambda(row)));
        }
      },
      Kokkos::Max<double>(length_max));
  Kokkos::parallel_reduce(
      "constitutive_residual_angle", Kokkos::RangePolicy<Space>(0, family.num_rows()),
      KOKKOS_LAMBDA(const int row, double& m) {
        if (Family::row_unit(row % Family::rows_per_entry) == RowUnit::ANGLE) {
          m = Kokkos::max(m, Kokkos::abs(psi(row) + family.row_compliance(row) * lambda(row)));
        }
      },
      Kokkos::Max<double>(angle_max));
  return impl::LengthAngleMax{Kokkos::max(length_max, 0.0), Kokkos::max(angle_max, 0.0)};
}

// Every bilateral family, rigid and compliant, in two groups of rods joined by nothing, under forces and torques, with
// the families passed out of packing order. Rods 0-2 are clamped at the bottom end of the first by a rigid pose anchor
// and pinned end to end, and sphere 3 hangs from the top of rod 2 by a fixed length. Rods 4-7 carry a spring of each
// kind and compliant position and pose anchors, all preloaded. Turning the rods moves the attached points at second
// order in the step, so a single linearization leaves the pins open and the linear spring off its law; at the end of an
// accepted SLCP step every row follows its constitutive law, psi + K^-1 lambda = 0, to within its tolerance.
TEST(Mbody, SlcpHoldsBilateralRowsAtStepEnd) {
  constexpr size_t kNumRods = 8;
  const Vector3d tilt_axis{0.6, 0.8, 0.0};
  const Vector3d half{0.0, 0.0, 0.5};
  RodViews<HostExecSpace> rods(kNumRods);
  for (size_t k = 0; k < kNumRods; ++k) {
    rods.radius(k) = 0.2;
    rods.length(k) = 1.0;
  }

  // Rigid group: rods 0-2 pinned end to end above the clamp, sphere 3 hung from the top of rod 2
  const double rigid_tilts[3] = {0.0, 0.3, -0.4};
  Vector3d bottom{0.0, 0.0, 0.0};
  for (size_t k = 0; k < 3; ++k) {
    const Quaterniond orientation = axis_angle_to_quaternion(tilt_axis, rigid_tilts[k]);
    rods.orientation(k) = orientation;
    rods.center(k) = bottom + orientation * half;
    bottom = Vector3d(rods.center(k)) + orientation * half;
  }
  rods.center(3) = bottom + Vector3d{0.3, 0.0, 0.4};
  rods.orientation(3) = Quaterniond{1.0, 0.0, 0.0, 0.0};
  rods.length(3) = 0.0;

  // Compliant group: rods 4-7, away from the rigid group
  const double compliant_tilts[4] = {0.0, 0.3, -0.4, 0.2};
  const Vector3d centers[4] = {{4.0, 0.0, 0.0}, {4.1, 0.0, 1.2}, {4.3, 0.1, 2.4}, {4.2, 0.4, 3.5}};
  for (size_t k = 0; k < 4; ++k) {
    rods.center(4 + k) = centers[k];
    rods.orientation(4 + k) = axis_angle_to_quaternion(tilt_axis, compliant_tilts[k]);
  }

  zero_rod_state(rods);
  rods.force(2) = Vector3d{0.4, -0.2, 0.1};
  rods.torque(2) = Vector3d{0.05, 0.1, -0.08};
  rods.force(3) = Vector3d{0.1, 0.2, -0.3};
  rods.force(7) = Vector3d{0.3, -0.25, 0.1};
  rods.torque(6) = Vector3d{0.04, -0.06, 0.05};

  PinViews<HostExecSpace> pins(2);
  for (size_t k = 0; k < 2; ++k) {
    pins.rod_i(k) = static_cast<int>(k);
    pins.rod_j(k) = static_cast<int>(k + 1);
    pins.body_offset_i(k) = half;
    pins.body_offset_j(k) = -half;
  }
  FixedLengthViews<HostExecSpace> lengths(1);
  lengths.rod_i(0) = 2;
  lengths.rod_j(0) = 3;
  lengths.body_offset_i(0) = half;
  lengths.body_offset_j(0) = Vector3d{0.0, 0.0, 0.0};
  lengths.rest_length(0) = 0.5;
  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 4;
  lin_springs.rod_j(0) = 5;
  lin_springs.rest_length(0) = 1.1;
  lin_springs.spring_constant(0) = 4.0;
  AngularSpringViews<HostExecSpace> ang_springs(1);
  ang_springs.rod_i(0) = 5;
  ang_springs.rod_j(0) = 6;
  ang_springs.rest_angle(0) = 0.5;
  ang_springs.spring_constant(0) = 3.0;
  TriplePointAngularSpringViews<HostExecSpace> triple_springs(1);
  triple_springs.rod_i(0) = 5;
  triple_springs.rod_j(0) = 7;
  triple_springs.rod_k(0) = 6;
  triple_springs.rest_angle(0) = minor_angle(centers[1] - centers[2], centers[3] - centers[2]) - 0.2;
  triple_springs.spring_constant(0) = 2.0;
  FixedPositionViews<HostExecSpace> position_anchors(1);
  set_fixed_position(position_anchors, 0, /*rod=*/7, centers[3] + rods.orientation(7) * half + Vector3d{0.05, 0.0, 0.0},
                     half, /*compliance=*/Vector3d{0.02, 0.03, 0.01});
  FixedPoseViews<HostExecSpace> pose_anchors(2);
  set_fixed_pose(pose_anchors, 0, /*rod=*/0, Vector3d{0.0, 0.0, 0.0}, Quaterniond(rods.orientation(0)), -half);
  set_fixed_pose(pose_anchors, 1, /*rod=*/4, centers[0] - rods.orientation(4) * half,
                 axis_angle_to_quaternion(Vector3d{0.0, 0.0, 1.0}, 0.1) * Quaterniond(rods.orientation(4)), -half,
                 /*position_compliance=*/Vector3d{0.01, 0.02, 0.015},
                 /*orientation_compliance=*/Vector3d{0.03, 0.02, 0.04});
  const auto constraints =
      make_constraint_set(pose_anchors, triple_springs, pins, lin_springs, position_anchors, lengths, ang_springs);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 0.5;
  MixedSLCPConfig cfg;
  cfg.inner_lcp_config.max_cg_iters = 500;
  cfg.inner_lcp_config.cg_tol = 1e-13;
  cfg.inner_lcp_config.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.length_tol = 1e-8;
  cfg.angle_tol = 1e-8;

  for (const unsigned max_iters : {1u, 50u}) {
    cfg.max_iters = max_iters;

    // Step
    const auto rods_d = copy_to<TestExecSpace>(rods);
    const auto constraints_d = copy_to<TestExecSpace>(constraints);
    const MixedSLCPResult result =
        step_rods(make_mixed_slcp_integrator(rods_d, constraints_d, mobility_model, dt), cfg, copy_load(rods_d));
    ASSERT_TRUE(lcp_converged(result)) << result << " at max_iters=" << max_iters;

    // Every row at the end of the step
    const impl::LengthAngleMax pin_residual =
        constitutive_residual(rods_d, get<PinViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax length_residual =
        constitutive_residual(rods_d, get<FixedLengthViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax lin_residual =
        constitutive_residual(rods_d, get<LinearSpringViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax ang_residual =
        constitutive_residual(rods_d, get<AngularSpringViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax triple_residual =
        constitutive_residual(rods_d, get<TriplePointAngularSpringViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax position_residual =
        constitutive_residual(rods_d, get<FixedPositionViews<TestExecSpace>>(constraints_d));
    const impl::LengthAngleMax pose_residual =
        constitutive_residual(rods_d, get<FixedPoseViews<TestExecSpace>>(constraints_d));
    if (max_iters > 1) {
      ASSERT_TRUE(result.converged) << result;
      EXPECT_GE(result.num_iters, 2u);
      EXPECT_LE(pin_residual.length, cfg.length_tol);
      EXPECT_LE(length_residual.length, cfg.length_tol);
      EXPECT_LE(lin_residual.length, cfg.length_tol);
      EXPECT_LE(ang_residual.angle, cfg.angle_tol);
      EXPECT_LE(triple_residual.angle, cfg.angle_tol);
      EXPECT_LE(position_residual.length, cfg.length_tol);
      EXPECT_LE(pose_residual.length, cfg.length_tol);
      EXPECT_LE(pose_residual.angle, cfg.angle_tol);
    } else {
      EXPECT_GT(pin_residual.length, 100.0 * cfg.length_tol);
      EXPECT_GT(lin_residual.length, 100.0 * cfg.length_tol);
    }
  }
}

//@}

//! \name Integrator storage
//@{

// An integrator holds the storage its steps write, and a new integrator can take another's: steps through it carry
// their solutions to the next step.

/// \brief The allocations of two steps through a mixed SLCP integrator of size-dt steps that has stepped once,
/// unrecorded, on TestExecSpace copies of rods and constraints.
template <typename Model, typename Policy, typename... Families>
std::map<std::pair<std::string, uint64_t>, size_t> held_slcp_allocations(const RodViews<HostExecSpace>& rods,
                                                                         const ConstraintSet<Families...>& constraints,
                                                                         const Model& mobility_model, double dt,
                                                                         const MixedSLCPConfig& cfg,
                                                                         const Policy& preconditioner_policy) {
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(constraints);
  const auto load_d = copy_load(rods_d);
  const auto integrator = make_mixed_slcp_integrator(rods_d, constraints_d, mobility_model, dt, preconditioner_policy);
  reset_rod_state(rods_d, load_d);
  solve_step(integrator, cfg);
  return record_allocations([&] {
    for (int step = 0; step < 2; ++step) {
      reset_rod_state(rods_d, load_d);
      solve_step(integrator, cfg);
    }
  });
}

/// \brief The allocations of two steps through a mixed LCP integrator of size-dt steps that has stepped once,
/// unrecorded, on TestExecSpace copies of rods and constraints.
template <typename Model, typename... Families>
std::map<std::pair<std::string, uint64_t>, size_t> held_lcp_allocations(const RodViews<HostExecSpace>& rods,
                                                                        const ConstraintSet<Families...>& constraints,
                                                                        const Model& mobility_model, double dt,
                                                                        const MixedLCPConfig& cfg) {
  const auto rods_d = copy_to<TestExecSpace>(rods);
  const auto constraints_d = copy_to<TestExecSpace>(constraints);
  const auto load_d = copy_load(rods_d);
  const auto integrator = make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt);
  reset_rod_state(rods_d, load_d);
  solve_step(integrator, cfg);
  return record_allocations([&] {
    for (int step = 0; step < 2; ++step) {
      reset_rod_state(rods_d, load_d);
      solve_step(integrator, cfg);
    }
  });
}

/// \brief The label and size of each allocation, for failure messages.
std::string allocation_labels(const std::map<std::pair<std::string, uint64_t>, size_t>& allocations) {
  std::string labels;
  for (const auto& [allocation, count] : allocations) {
    labels += allocation.first + " (" + std::to_string(allocation.second) + " B) x" + std::to_string(count) + "\n";
  }
  return labels;
}

// A step through an integrator allocates nothing once a first step has sized its storage: both blocks,
// unpreconditioned and preconditioned; the bilateral block alone under a caller's mobility, which has only a plain
// apply; and the unilateral block alone.
TEST(Mbody, IntegratorStepsDoNotAllocate) {
  const auto p = make_fallback_problem();
  const auto& c = p.constraints;
  const auto bilateral_only =
      make_constraint_set(get<LinearSpringViews<HostExecSpace>>(c), get<PinViews<HostExecSpace>>(c),
                          get<FixedLengthViews<HostExecSpace>>(c), get<FixedPoseViews<HostExecSpace>>(c),
                          get<FixedPositionViews<HostExecSpace>>(c));
  const MixedSLCPConfig cfg{p.cfg, 50, 1e-9, 1e-9};
  const auto row = make_sphere_row_problem();

  // Step
  const auto both = held_slcp_allocations(p.rods, c, p.mobility_model, p.dt, cfg, NoPreconditioner{});
  const auto both_jacobi = held_slcp_allocations(p.rods, c, p.mobility_model, p.dt, cfg, SelfMobilityJacobi{});
  const auto bilateral_isotropic = held_slcp_allocations(
      p.rods, bilateral_only, IsotropicMobility{.m = 0.25, .m_rot = 1.0}, p.dt, cfg, NoPreconditioner{});
  const auto unilateral = held_lcp_allocations(row.rods, row.constraints, row.mobility_model, row.dt, row.cfg);

  // No allocation
  EXPECT_TRUE(both.empty()) << allocation_labels(both);
  EXPECT_TRUE(both_jacobi.empty()) << allocation_labels(both_jacobi);
  EXPECT_TRUE(bilateral_isotropic.empty()) << allocation_labels(bilateral_isotropic);
  EXPECT_TRUE(unilateral.empty()) << allocation_labels(unilateral);
}

// An integrator given storage sized for another layout of rods and constraints makes storage of its own. Integrators
// alternating between the fallback problem and the same problem without its contact, each taking the last one's
// storage, step exactly as integrators with storage of their own, bit for bit. Serial execution fixes the order of
// every atomic sum.
TEST(Mbody, IntegratorRebuildsStorageOfAnotherLayout) {
  const auto p = make_fallback_problem();
  const auto& c = p.constraints;
  const auto contact_free =
      make_constraint_set(get<LinearSpringViews<HostExecSpace>>(c), get<PinViews<HostExecSpace>>(c),
                          get<FixedLengthViews<HostExecSpace>>(c), get<FixedPoseViews<HostExecSpace>>(c),
                          get<FixedPositionViews<HostExecSpace>>(c), ContactViews<HostExecSpace>(0));
  const MixedSLCPConfig cfg{p.cfg, 50, 1e-9, 1e-9};
  auto storage =
      make_mixed_slcp_integrator(copy_to<Kokkos::Serial>(p.rods), copy_to<Kokkos::Serial>(c), p.mobility_model, p.dt)
          .workspace();

  for (int step = 0; step < 3; ++step) {
    const auto& constraints = step == 1 ? contact_free : c;
    const auto rods_passed = copy_to<Kokkos::Serial>(p.rods);
    const auto constraints_passed = copy_to<Kokkos::Serial>(constraints);
    const auto rods_own = copy_to<Kokkos::Serial>(p.rods);
    const auto constraints_own = copy_to<Kokkos::Serial>(constraints);

    // Step
    const auto passed = make_mixed_slcp_integrator(rods_passed, constraints_passed, p.mobility_model, p.dt,
                                                   NoPreconditioner{}, storage);
    storage = passed.workspace();
    const MixedSLCPResult through_passed = solve_step(passed, cfg);
    const MixedSLCPResult through_own =
        solve_step(make_mixed_slcp_integrator(rods_own, constraints_own, p.mobility_model, p.dt), cfg);
    ASSERT_TRUE(through_own.converged) << through_own << " at step " << step;

    // The step
    EXPECT_EQ(through_passed.num_iters, through_own.num_iters) << "step " << step;
    EXPECT_EQ(std::bit_cast<uint64_t>(through_passed.residual), std::bit_cast<uint64_t>(through_own.residual))
        << "step " << step;
    EXPECT_EQ(through_passed.accepted_lcp_result.num_iters, through_own.accepted_lcp_result.num_iters)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(rods_passed.force_torque_view(), rods_own.force_torque_view()), 0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(rods_passed.velocity_omega_view(), rods_own.velocity_omega_view()), 0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<ContactViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<ContactViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<LinearSpringViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<LinearSpringViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<PinViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<PinViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<FixedLengthViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<FixedLengthViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<FixedPoseViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<FixedPoseViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
    EXPECT_EQ(count_bit_differences(get<FixedPositionViews<Kokkos::Serial>>(constraints_passed).lambda_view(),
                                    get<FixedPositionViews<Kokkos::Serial>>(constraints_own).lambda_view()),
              0u)
        << "step " << step;
  }
}

// Storage passed to a new integrator serves that integrator's step size and rods, not those it was made with:
//   - integrators of two step sizes in turn, each taking the last one's storage, meet the closed form of a spring
//   between
//     isotropic spheres, the force y = -(|r| - L) / (2 m dt + 1/k);
//   - integrators over rods copied into new views for every step, each taking the last one's storage, step exactly as
//     one integrator over rods kept in one set of views, bit for bit, on Serial.
TEST(Mbody, IntegratorStorageServesTheNextIntegrator) {
  // Two step sizes
  {
    RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                       Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
    zero_rod_state(rods);
    LinearSpringViews<HostExecSpace> lin_springs(1);
    lin_springs.rod_i(0) = 0;
    lin_springs.rod_j(0) = 1;
    lin_springs.rest_length(0) = 1.0;
    lin_springs.spring_constant(0) = 2.0;
    const IsotropicMobility mobility_model{.m = 0.7, .m_rot = 1.3};
    MixedLCPConfig cfg;
    cfg.cg_tol = 1e-14;

    const auto rods_d = copy_to<TestExecSpace>(rods);
    const auto constraints_d = copy_to<TestExecSpace>(make_constraint_set(lin_springs));
    const auto load_d = copy_load(rods_d);
    auto storage = make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, 0.5).workspace();
    for (const double dt : {0.5, 2.0}) {
      const auto integrator =
          make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt, NoPreconditioner{}, storage);
      storage = integrator.workspace();
      reset_rod_state(rods_d, load_d);
      const MixedLCPResult result = solve_step(integrator, cfg);
      EXPECT_TRUE(result.converged) << result << " at dt=" << dt;
      const double y = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, get<LinearSpringViews<TestExecSpace>>(constraints_d).lambda_view())(0);
      EXPECT_NEAR(y, -(1.6 - 1.0) / (2.0 * mobility_model.m * dt + 1.0 / lin_springs.spring_constant(0)), 1e-13)
          << "dt=" << dt;
    }
  }

  // New views for every step
  {
    const auto p = make_fallback_problem();
    const MixedSLCPConfig cfg{p.cfg, 50, 1e-9, 1e-9};
    const auto rods_kept = copy_to<Kokkos::Serial>(p.rods);
    const auto constraints_kept = copy_to<Kokkos::Serial>(p.constraints);
    const auto load_kept = copy_load(rods_kept);
    const auto kept = make_mixed_slcp_integrator(rods_kept, constraints_kept, p.mobility_model, p.dt);
    const auto rods_copied = copy_to<Kokkos::Serial>(p.rods);
    const auto constraints_copied = copy_to<Kokkos::Serial>(p.constraints);
    const auto load_copied = copy_load(rods_copied);
    auto storage = make_mixed_slcp_integrator(rods_copied, constraints_copied, p.mobility_model, p.dt).workspace();

    for (int step = 0; step < 4; ++step) {
      ASSERT_TRUE(lcp_converged(step_rods(kept, cfg, load_kept))) << "step " << step;
      const auto rods_new = copy_to<Kokkos::Serial>(rods_copied);
      const auto copied =
          make_mixed_slcp_integrator(rods_new, constraints_copied, p.mobility_model, p.dt, NoPreconditioner{}, storage);
      storage = copied.workspace();
      ASSERT_TRUE(lcp_converged(step_rods(copied, cfg, load_copied))) << "step " << step;
      deep_copy(rods_copied, rods_new);
    }

    EXPECT_EQ(count_bit_differences(rods_copied.center_view(), rods_kept.center_view()), 0u);
    EXPECT_EQ(count_bit_differences(rods_copied.orientation_view(), rods_kept.orientation_view()), 0u);
    EXPECT_EQ(count_bit_differences(rods_copied.force_torque_view(), rods_kept.force_torque_view()), 0u);
    EXPECT_EQ(count_bit_differences(rods_copied.velocity_omega_view(), rods_kept.velocity_omega_view()), 0u);
    EXPECT_EQ(count_bit_differences(get<PinViews<Kokkos::Serial>>(constraints_copied).lambda_view(),
                                    get<PinViews<Kokkos::Serial>>(constraints_kept).lambda_view()),
              0u);
    EXPECT_EQ(count_bit_differences(get<ContactViews<Kokkos::Serial>>(constraints_copied).lambda_view(),
                                    get<ContactViews<Kokkos::Serial>>(constraints_kept).lambda_view()),
              0u);
  }
}

//@}

//! \name Chain mechanics
//@{

// A sphere chain's response against beam theory. With spacing a, a linear spring of stiffness k_lin gives
// EA = k_lin a and a bend joint of stiffness k_ang gives EI = k_ang a: the Hencky bar chain. Bending uses
// triple-point springs, since a pairwise angular spring does not resist a sub-chain sliding sideways.

/// \brief The measured EA = F L / dL of an axial chain of length L pulled apart by tip_force at both ends.
double run_axial_chain_EA(size_t num_segments, double L, double k_lin, double tip_force) {
  const size_t num_spheres = num_segments + 1;
  const double spacing = L / static_cast<double>(num_segments);

  RodViews<HostExecSpace> rods(num_spheres);
  for (size_t k = 0; k < num_spheres; ++k) {
    rods.center(k) = Vector3d{0.0, 0.0, spacing * static_cast<double>(k)};
    rods.orientation(k) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(k) = 0.2;
    rods.length(k) = 0.0;
  }
  zero_rod_state(rods);

  LinearSpringViews<HostExecSpace> lin_springs(num_segments);
  for (size_t k = 0; k < num_segments; ++k) {
    lin_springs.rod_i(k) = static_cast<int>(k);
    lin_springs.rod_j(k) = static_cast<int>(k + 1);
    lin_springs.rest_length(k) = spacing;
    lin_springs.spring_constant(k) = k_lin;
  }
  const auto constraints = make_constraint_set(lin_springs);

  const LocalDragMobility mobility_model{.viscosity = 1.0};
  const double dt = 2.0;
  MixedLCPConfig cfg;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate

  rods.force(0) = Vector3d{0.0, 0.0, -tip_force};
  rods.force(num_spheres - 1) = Vector3d{0.0, 0.0, tip_force};
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  const auto integrator = make_mixed_lcp_integrator(rods_d, constraints_d, mobility_model, dt);

  const int num_steps = 150;
  for (int step = 0; step < num_steps; ++step) {
    EXPECT_TRUE(step_rods(integrator, cfg, load_d).converged) << "num_segments=" << num_segments << " step " << step;
  }
  deep_copy(rods, rods_d);

  const double L_final = norm(rods.center(num_spheres - 1) - rods.center(0));
  return tip_force * L / (L_final - L);
}

// Every segment of an axial chain carries the same tension, so EA = k_lin a is exact at every segment count.
TEST(Mbody, ChainAxialStiffnessConvergesWithSegmentCount) {
  const double L = 8.0, k_lin = 3.0, tip_force = 0.3;
  for (const size_t num_segments : {2, 4, 8, 16}) {
    const double spacing = L / static_cast<double>(num_segments);
    const double EA_expected = k_lin * spacing;
    const double EA_measured = run_axial_chain_EA(num_segments, L, k_lin, tip_force);
    EXPECT_NEAR(EA_measured, EA_expected, 1e-3 * EA_expected) << "num_segments=" << num_segments;
  }
}

/// \brief The free spheres' transverse displacements at static equilibrium under f_ext, solved directly.
///
/// Without contacts, a settled chain satisfies F_ext + B y = 0 with each bend multiplier y = -k_ang theta.
/// For small transverse displacements u the bend angles are linear, theta = J u, with J the bend springs' rate operator
/// applied to the free spheres' y-velocities, so equilibrium is the linear system k_ang J^T J u = f_ext.
///
/// A bend angle has no gradient at straight, so J is taken on a shallow arc of amplitude pre_bend, which the answer
/// does not depend on. With the arc in the y-z plane the bend axis is x, so the y-displacements decouple exactly.
///
/// CantileverSettlesToStaticBend checks that stepping settles to this equilibrium.
std::vector<double> solve_static_bend(size_t num_spheres, double spacing, double k_ang, size_t first_free,
                                      const std::vector<double>& f_ext, double pre_bend = 1e-6) {
  BendChain chain = make_bend_chain(num_spheres, spacing, k_ang);
  for (size_t k = 0; k < num_spheres; ++k) {
    const double z = spacing * static_cast<double>(k);
    chain.rods.center(k) = Vector3d{0.0, pre_bend * z * z, z};
  }

  const size_t num_interior = chain.springs.size();
  const size_t num_free = f_ext.size();
  auto b0 = make_constraint_values(chain.springs);
  const impl::TripleGeometry<TestExecSpace> geo =
      impl::compute_geometry(create_mirror_view_and_copy(TestExecSpace{}, chain.rods),
                             create_mirror_view_and_copy(TestExecSpace{}, chain.springs), b0);
  const impl::TripleForceOpT<TestExecSpace> op_t(geo, num_spheres);

  DenseMat jacobian(num_interior, std::vector<double>(num_free, 0.0));
  Kokkos::View<double*, TestMemSpace> vel("vel", 6 * num_spheres), rate("rate", num_interior);
  auto vel_h = Kokkos::create_mirror_view(vel);
  for (size_t col = 0; col < num_free; ++col) {
    Kokkos::deep_copy(vel_h, 0.0);
    rod_velocity(vel_h, static_cast<int>(first_free + col)) = Vector3d{0.0, 1.0, 0.0};
    Kokkos::deep_copy(vel, vel_h);
    op_t.apply(vel, rate);
    const auto rate_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate);
    for (size_t row = 0; row < num_interior; ++row) {
      jacobian[row][col] = rate_h(row);
    }
  }

  DenseMat stiffness = dense_matmul(dense_transpose(jacobian), jacobian);
  for (auto& row : stiffness) {
    for (auto& entry : row) {
      entry *= k_ang;
    }
  }
  return dense_solve(stiffness, f_ext);
}

/// \brief Cantilever tip deflection. Spheres 0 and 1 are the wall, holding its position and slope.
double run_static_cantilever_tip(size_t num_segments, double L, double EI, double tip_force) {
  const double spacing = L / static_cast<double>(num_segments);
  std::vector<double> f_ext(num_segments, 0.0);
  f_ext.back() = tip_force;
  return solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, f_ext).back();
}

/// \brief Simply supported midspan deflection. Spheres 0 and N are the supports; num_segments must be even.
double run_static_simply_supported_midspan(size_t num_segments, double L, double EI, double load) {
  const double spacing = L / static_cast<double>(num_segments);
  const size_t midspan = num_segments / 2 - 1;  // free spheres are 1..N-1
  std::vector<double> f_ext(num_segments - 1, 0.0);
  f_ext[midspan] = load;
  return solve_static_bend(num_segments + 1, spacing, EI / spacing, 1, f_ext)[midspan];
}

// A Hencky bar chain of N segments under a tip load deflects exactly
//
//   tip_N = F L^3 (N+1)(2N+1) / (6 N^2 EI),
//
// first order in the spacing: its departure from Euler-Bernoulli's F L^3 / (3 EI) is (3N+1)/(2N^2).
TEST(Mbody, ChainBendingMatchesHenckyBarChain) {
  const double L = 8.0, EI = 5.0, tip_force = 0.005;
  const double continuum = tip_force * L * L * L / (3.0 * EI);

  double finest_scaled_error = 0.0;
  for (const size_t num_segments : {4, 7, 11, 16, 25, 50, 100, 200}) {
    const double n = static_cast<double>(num_segments);
    const double hencky = tip_force * L * L * L * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n * EI);
    const double tip = run_static_cantilever_tip(num_segments, L, EI, tip_force);

    EXPECT_NEAR(tip, hencky, 1e-6 * hencky) << "num_segments=" << num_segments;
    finest_scaled_error = n * std::abs(tip - continuum) / continuum;
  }

  // N rel_err = (3N+1)/(2N), which falls to 3/2
  EXPECT_NEAR(finest_scaled_error, 1.5, 0.005);
}

// Supported at both ends under a midspan load, the chain deflects exactly
//
//   delta = P L^3 / (48 EI) (1 + 2/N^2),
//
// second order in the spacing. The cantilever loses an order to its wall holding two adjacent spheres.
TEST(Mbody, SimplySupportedBendingMatchesHenckyBarChain) {
  const double L = 8.0, EI = 5.0, load = 0.005;
  const double continuum = load * L * L * L / (48.0 * EI);

  double finest_scaled_error = 0.0;
  for (const size_t num_segments : {4, 8, 16, 32, 64, 128}) {
    const double n = static_cast<double>(num_segments);
    const double hencky = continuum * (1.0 + 2.0 / (n * n));
    const double midspan = run_static_simply_supported_midspan(num_segments, L, EI, load);

    EXPECT_NEAR(midspan, hencky, 1e-6 * hencky) << "num_segments=" << num_segments;
    finest_scaled_error = n * n * std::abs(midspan - continuum) / continuum;
  }

  // N^2 rel_err is exactly 2 at every resolution
  EXPECT_NEAR(finest_scaled_error, 2.0, 1e-4);
}

/// \brief A cantilever chain in rods 0 to num_segments + 1 of num_rods rods, joined by links of family Link.
///
/// Links join neighbours from rod 1 on and a bend spring sits at each interior rod; callers hold rods 0 and 1 as the
/// wall, which a link between them could only duplicate, and set any rods past the chain. Linear-spring links are
/// stiff, k = 1e5 / spacing^2; fixed-length links are rigid. The free rods start on a shallow arc so every bend angle
/// has a gradient on the first step, and the settled state does not depend on it.
template <typename Link>
SolveInput<Link, TriplePointAngularSpringViews<HostExecSpace>> make_cantilever(size_t num_segments, double L, double EI,
                                                                               size_t num_rods, double dt) {
  const double spacing = L / static_cast<double>(num_segments);
  const size_t num_chain = num_segments + 2;
  MUNDY_THROW_REQUIRE(num_rods >= num_chain, std::invalid_argument, "make_cantilever: num_rods must hold the chain.");
  const BendChain chain = make_bend_chain(num_chain, spacing, EI / spacing);

  RodViews<HostExecSpace> rods(num_rods);
  for (size_t k = 0; k < num_chain; ++k) {
    const double z = spacing * static_cast<double>(k);
    const double arc = std::max(z - spacing, 0.0);
    rods.center(k) = Vector3d{0.0, 1e-6 * arc * arc, z};
    rods.orientation(k) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(k) = chain.rods.radius(k);
    rods.length(k) = 0.0;
  }
  zero_rod_state(rods);

  Link links(num_chain - 2);
  for (size_t k = 1; k + 1 < num_chain; ++k) {
    links.rod_i(k - 1) = static_cast<int>(k);
    links.rod_j(k - 1) = static_cast<int>(k + 1);
    links.rest_length(k - 1) = spacing;
    if constexpr (std::is_same_v<Link, LinearSpringViews<HostExecSpace>>) {
      links.spring_constant(k - 1) = 1.0e5 / (spacing * spacing);
    }
  }

  MixedLCPConfig cfg;
  cfg.max_cg_iters = 1000;
  // Tight enough that solver error sits far below the settled chain's geometric nonlinearity.
  cfg.cg_tol = 1e-14;
  cfg.outer_tol = 1e-12;
  return {rods, make_constraint_set(links, chain.springs), LocalDragMobility{.viscosity = 1.0}, dt, cfg};
}

/// \brief The settled state of a cantilever under a tip load.
struct SettledCantilever {
  bool settled;
  std::vector<double> displacement;  // transverse, of the free rods 2 to num_segments + 1
};

/// \brief A cantilever with links of family Link settled under a transverse tip load.
///
/// Each step is a sequence of at most max_iters linearizations.
template <typename Link>
SettledCantilever run_settled_cantilever(size_t num_segments, double L, double EI, double tip_force, double dt,
                                         unsigned max_iters) {
  const size_t num_chain = num_segments + 2;
  const int tip = static_cast<int>(num_chain - 1);
  auto p = make_cantilever<Link>(num_segments, L, EI, num_chain, dt);
  p.rods.force(tip) = Vector3d{0.0, tip_force, 0.0};

  FixedPositionViews<HostExecSpace> wall(2);
  set_fixed_position(wall, 0, /*rod=*/0, Vector3d(p.rods.center(0)));
  set_fixed_position(wall, 1, /*rod=*/1, Vector3d(p.rods.center(1)));
  const auto constraints = make_constraint_set(get<Link>(p.constraints),
                                               get<TriplePointAngularSpringViews<HostExecSpace>>(p.constraints), wall);
  const MixedSLCPConfig cfg{p.cfg, max_iters, 1e-9, 1e-9};

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  SettledCantilever result;
  result.settled = step_until_settled(make_mixed_slcp_integrator(rods_d, constraints_d, p.mobility_model, p.dt), cfg,
                                      load_d, /*settled_step=*/1e-12, /*max_steps=*/1000);
  deep_copy(p.rods, rods_d);

  for (size_t k = 2; k < num_chain; ++k) {
    result.displacement.push_back(p.rods.center(k)[1]);
  }
  return result;
}

// Without contacts, the settled chain is the static equilibrium solve_static_bend computes, with stiff-spring or rigid
// links and for either sequence length: a settled step does not move the rods, so every linearization is at the same
// configuration. Every free sphere's deflection matches it up to the chain's nonlinear geometry, which at this load
// moves the deflection by about 3e-5 of the tip's, shrinking as P^2.
TEST(Mbody, CantileverSettlesToStaticBend) {
  const double L = 8.0, EI = 5.0, tip_force = 1e-3;
  const char* const link_names[2] = {"stiff springs", "fixed lengths"};
  for (const size_t num_segments : {4, 8, 16}) {
    const double spacing = L / static_cast<double>(num_segments);
    std::vector<double> f_ext(num_segments, 0.0);
    f_ext.back() = tip_force;
    const std::vector<double> expected = solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, f_ext);

    for (const unsigned max_iters : {1u, 50u}) {
      const SettledCantilever by_link[2] = {run_settled_cantilever<LinearSpringViews<HostExecSpace>>(
                                                num_segments, L, EI, tip_force, /*dt=*/100.0, max_iters),
                                            run_settled_cantilever<FixedLengthViews<HostExecSpace>>(
                                                num_segments, L, EI, tip_force, /*dt=*/100.0, max_iters)};
      for (int link = 0; link < 2; ++link) {
        const SettledCantilever& r = by_link[link];
        ASSERT_TRUE(r.settled) << link_names[link] << " at num_segments=" << num_segments << " max_iters=" << max_iters;
        for (size_t k = 0; k < expected.size(); ++k) {
          EXPECT_NEAR(r.displacement[k], expected[k], 1e-4 * expected.back())
              << "free sphere " << k << " with " << link_names[link] << " at num_segments=" << num_segments
              << " max_iters=" << max_iters;
        }
      }
    }
  }
}

/// \brief The settled state of a propped cantilever.
struct ProppedCantilever {
  bool settled;
  double contact_force;
  double tip_gap;
};

/// \brief A cantilever with links of family Link whose tip rests on an anchored sphere, settled under a midspan load.
///
/// Each step is a sequence of at most max_iters linearizations, with the Schur complement preconditioned by its
/// self-mobility Jacobi if preconditioned, through one workspace held across the steps if held_storage.
template <typename Link>
ProppedCantilever run_propped_cantilever(size_t num_segments, double L, double EI, double load, double dt,
                                         unsigned max_iters, bool preconditioned, bool held_storage) {
  const size_t num_chain = num_segments + 2;
  const int midspan = static_cast<int>(num_segments / 2 + 1);
  const int tip = static_cast<int>(num_chain - 1);
  const int obstacle = static_cast<int>(num_chain);
  auto p = make_cantilever<Link>(num_segments, L, EI, num_chain + 1, dt);

  const double radius = p.rods.radius(0);
  p.rods.center(obstacle) = Vector3d{0.0, -2.0 * radius, p.rods.center(tip)[2]};  // touches the straight tip
  p.rods.orientation(obstacle) = Quaterniond{1.0, 0.0, 0.0, 0.0};
  p.rods.radius(obstacle) = radius;
  p.rods.length(obstacle) = 0.0;
  p.rods.force(midspan) = Vector3d{0.0, -load, 0.0};

  FixedPositionViews<HostExecSpace> anchors(3);
  set_fixed_position(anchors, 0, /*rod=*/0, Vector3d(p.rods.center(0)));
  set_fixed_position(anchors, 1, /*rod=*/1, Vector3d(p.rods.center(1)));
  set_fixed_position(anchors, 2, obstacle, Vector3d(p.rods.center(obstacle)));
  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = tip;
  contacts.rod_j(0) = obstacle;
  const auto constraints = make_constraint_set(
      get<Link>(p.constraints), get<TriplePointAngularSpringViews<HostExecSpace>>(p.constraints), anchors, contacts);
  const MixedSLCPConfig cfg{p.cfg, max_iters, 1e-9, 1e-9};

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  const auto settle = [&](const auto& policy) {
    return step_until_settled(make_mixed_slcp_integrator(rods_d, constraints_d, p.mobility_model, p.dt, policy), cfg,
                              load_d, /*settled_step=*/1e-12, /*max_steps=*/1000, held_storage);
  };
  const bool settled = preconditioned ? settle(SelfMobilityJacobi{}) : settle(NoPreconditioner{});
  deep_copy(p.rods, rods_d);
  deep_copy(constraints, constraints_d);

  const double tip_gap = norm(p.rods.center(tip) - p.rods.center(obstacle)) - 2.0 * radius;
  return ProppedCantilever{settled, contacts.lambda(0), tip_gap};
}

// A cantilever whose tip rests on an anchored sphere, under a midspan load P. The contact props the tip, and the
// chain's flexibility gives its reaction exactly,
//
//   R_N = P (N+2)(5N+2) / (8 (N+1)(2N+1)),
//
// first order in the spacing toward Euler-Bernoulli's 5P/16, with stiff-spring or rigid links and for either sequence
// length, with and without the Schur complement preconditioned by its self-mobility Jacobi, and with a workspace per
// step or one held across the steps. The steps keep the chain's geometry nonlinear, which at this load moves the
// reaction by at most about 1e-6 of itself, shrinking as P^2.
TEST(Mbody, ProppedCantileverMatchesHenckyBarChain) {
  const double L = 8.0, EI = 5.0, load = 0.01;
  const double continuum = 5.0 * load / 16.0;
  const char* const link_names[2] = {"stiff springs", "fixed lengths"};

  for (const bool preconditioned : {false, true}) {
    for (const bool held_storage : {false, true}) {
      for (const unsigned max_iters : {1u, 50u}) {
        for (int link = 0; link < 2; ++link) {
          double finest_scaled_error = 0.0;
          double finest_scaled_error_expected = 0.0;
          for (const size_t num_segments : {4, 8, 16, 32}) {
            const double n = static_cast<double>(num_segments);
            const double hencky = load * (n + 2.0) * (5.0 * n + 2.0) / (8.0 * (n + 1.0) * (2.0 * n + 1.0));
            const ProppedCantilever r = link == 0 ? run_propped_cantilever<LinearSpringViews<HostExecSpace>>(
                                                        num_segments, L, EI, load,
                                                        /*dt=*/100.0, max_iters, preconditioned, held_storage)
                                                  : run_propped_cantilever<FixedLengthViews<HostExecSpace>>(
                                                        num_segments, L, EI, load,
                                                        /*dt=*/100.0, max_iters, preconditioned, held_storage);

            ASSERT_TRUE(r.settled) << link_names[link] << " at num_segments=" << num_segments
                                   << " max_iters=" << max_iters << " preconditioned=" << preconditioned
                                   << " held_storage=" << held_storage;
            EXPECT_NEAR(r.contact_force, hencky, 3e-6 * hencky)
                << link_names[link] << " at num_segments=" << num_segments << " max_iters=" << max_iters
                << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
            EXPECT_NEAR(r.tip_gap, 0.0, 1e-12)
                << link_names[link] << " at num_segments=" << num_segments << " max_iters=" << max_iters
                << " preconditioned=" << preconditioned << " held_storage=" << held_storage;
            finest_scaled_error = n * (r.contact_force - continuum) / continuum;
            finest_scaled_error_expected = (9.0 * n + 3.0) / (10.0 * n + 15.0 + 5.0 / n);
          }

          // N rel_err = (9N + 3) / (10N + 15 + 5/N), which falls to 9/10
          EXPECT_NEAR(finest_scaled_error, finest_scaled_error_expected, 1e-4)
              << link_names[link] << " at max_iters=" << max_iters << " preconditioned=" << preconditioned
              << " held_storage=" << held_storage;
        }
      }
    }
  }
}

// The pre-bend must not reach the answer: amplitudes a decade apart agree on both boundary value problems.
TEST(Mbody, BendLinearizationIsIndependentOfPreBend) {
  const double L = 8.0, EI = 5.0, load = 0.005;
  constexpr size_t num_segments = 16;
  const double spacing = L / static_cast<double>(num_segments);

  std::vector<double> cantilever(num_segments, 0.0);
  cantilever.back() = load;
  const double tip_fine = solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, cantilever, 1e-7).back();
  const double tip_coarse = solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, cantilever, 1e-6).back();
  EXPECT_NEAR(tip_fine, tip_coarse, 1e-6 * std::abs(tip_fine));

  const size_t midspan = num_segments / 2 - 1;
  std::vector<double> supported(num_segments - 1, 0.0);
  supported[midspan] = load;
  const double mid_fine = solve_static_bend(num_segments + 1, spacing, EI / spacing, 1, supported, 1e-7)[midspan];
  const double mid_coarse = solve_static_bend(num_segments + 1, spacing, EI / spacing, 1, supported, 1e-6)[midspan];
  EXPECT_NEAR(mid_fine, mid_coarse, 1e-6 * std::abs(mid_fine));
}

//@}

//! \name Solver building blocks
//@{

// A rod's self-mobility block is its block of M: a unit force or torque on rod r moves rod r by that block's column and
// no other rod at all. The block and M's apply form the same products; a fused multiply-add may move them by an ulp.
TEST(Mbody, LocalDragSelfMobilityIsItsBlockOfM) {
  RodViews<HostExecSpace> rods(3);
  rods.center(0) = Vector3d{0.0, 0.0, 0.0};
  rods.orientation(0) = axis_angle_to_quaternion(Vector3d{0.6, 0.8, 0.0}, 0.7);
  rods.radius(0) = 0.2;
  rods.length(0) = 1.0;
  rods.center(1) = Vector3d{2.0, 0.5, -1.0};
  rods.orientation(1) = axis_angle_to_quaternion(Vector3d{0.0, 0.6, 0.8}, 2.3);
  rods.radius(1) = 0.15;
  rods.length(1) = 0.0;
  rods.center(2) = Vector3d{-1.0, 3.0, 0.5};
  rods.orientation(2) = axis_angle_to_quaternion(Vector3d{1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}, 1.9);
  rods.radius(2) = 0.05;
  rods.length(2) = 3.0;
  const LocalDragMobilityOp<HostExecSpace> M = LocalDragMobility{.viscosity = 0.8}.make_mobility(rods);

  Kokkos::View<double*, HostExecSpace::memory_space> e("e", 18), column("column", 18);
  for (int r = 0; r < 3; ++r) {
    const Matrix<double, 6, 6> block = M.self_mobility(r);
    double scale = 0.0;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        scale = std::max(scale, std::abs(block(i, j)));
      }
    }
    for (int c = 0; c < 6; ++c) {
      Kokkos::deep_copy(e, 0.0);
      e(6 * r + c) = 1.0;
      M.apply(e, column);
      for (int k = 0; k < 18; ++k) {
        if (k / 6 == r) {
          EXPECT_NEAR(column(k), block(k % 6, c), 4.0 * std::numeric_limits<double>::epsilon() * scale)
              << "rod " << r << ", entry (" << k % 6 << ", " << c << ")";
        } else {
          EXPECT_EQ(column(k), 0.0) << "rod " << r << " moves rod " << k / 6;
        }
      }
    }
  }
}

// SelfMobilityJacobi's d is the diagonal of dt B^T M B + K^-1, exact under local drag, which couples no two rods: on
// rows coupling one, two and three rods, with and without torque, and with and without compliance (the fallback
// problem with a triple-point angular spring added), and at every solve of a sequence. Probing by unit vectors applies
// the operator CG inverts. The two sum the same terms dt v^T M_b v and K^-1 in different orders, each with a few dozen
// roundings, and local drag's self blocks are well conditioned, so they agree to a few dozen ulps of d.
TEST(Mbody, SelfMobilityJacobiIsTheDiagonal) {
  const auto p = make_fallback_problem();
  const auto& c = p.constraints;
  TriplePointAngularSpringViews<HostExecSpace> bend(1);
  bend.rod_i(0) = 2;
  bend.rod_j(0) = 4;
  bend.rod_k(0) = 3;
  bend.rest_angle(0) = 1.5;
  bend.spring_constant(0) = 2.0;
  const auto constraints =
      make_constraint_set(get<LinearSpringViews<HostExecSpace>>(c), get<PinViews<HostExecSpace>>(c),
                          get<FixedLengthViews<HostExecSpace>>(c), get<FixedPoseViews<HostExecSpace>>(c),
                          get<FixedPositionViews<HostExecSpace>>(c), bend, get<ContactViews<HostExecSpace>>(c));

  // Solve
  std::vector<ProbedDiagonal> records;
  const auto rods_d = copy_to<TestExecSpace>(p.rods);
  const auto constraints_d = copy_to<TestExecSpace>(constraints);
  const MixedSLCPResult result = solve_step(
      make_mixed_slcp_integrator(rods_d, constraints_d, p.mobility_model, p.dt, ProbingSelfMobilityJacobi{&records}),
      MixedSLCPConfig{p.cfg, 3, 1e-300, 1e-300});
  ASSERT_GE(result.num_iters, 2u) << result;
  ASSERT_EQ(records.size(), result.num_iters) << "one update per solve";

  // Every row at every solve
  for (size_t k = 0; k < records.size(); ++k) {
    ASSERT_EQ(records[k].probed.size(), 15u);
    for (size_t i = 0; i < records[k].probed.size(); ++i) {
      EXPECT_NEAR(records[k].self_mobility_jacobi[i], records[k].probed[i],
                  32.0 * std::numeric_limits<double>::epsilon() * records[k].probed[i])
          << "row " << i << " at solve " << k;
    }
  }
}

// A caller's policy reaches CG, updated before every solve: with d = 1, preconditioned CG performs plain CG's
// arithmetic, so the sequence reproduces the unpreconditioned one bit for bit, while d is NaN until the policy's first
// update. Serial execution fixes the order of every atomic sum.
TEST(Mbody, UnitJacobiIsUnpreconditioned) {
  const auto p = make_fallback_problem();
  const MixedSLCPConfig cfg{p.cfg, 50, 1e-9, 1e-9};
  const auto rods_plain = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_plain = copy_to<Kokkos::Serial>(p.constraints);
  const auto rods_unit = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_unit = copy_to<Kokkos::Serial>(p.constraints);

  // Solve
  const MixedSLCPResult plain =
      solve_step(make_mixed_slcp_integrator(rods_plain, constraints_plain, p.mobility_model, p.dt), cfg);
  const MixedSLCPResult unit =
      solve_step(make_mixed_slcp_integrator(rods_unit, constraints_unit, p.mobility_model, p.dt, UnitJacobi{}), cfg);
  ASSERT_TRUE(plain.converged) << plain;
  ASSERT_GE(plain.num_iters, 2u);

  // The step
  EXPECT_EQ(unit.num_iters, plain.num_iters);
  EXPECT_EQ(std::bit_cast<uint64_t>(unit.residual), std::bit_cast<uint64_t>(plain.residual));
  EXPECT_EQ(unit.accepted_lcp_result.num_iters, plain.accepted_lcp_result.num_iters);
  EXPECT_EQ(std::bit_cast<uint64_t>(unit.accepted_lcp_result.residual),
            std::bit_cast<uint64_t>(plain.accepted_lcp_result.residual));
  EXPECT_EQ(count_bit_differences(rods_unit.force_torque_view(), rods_plain.force_torque_view()), 0u);
  EXPECT_EQ(count_bit_differences(rods_unit.velocity_omega_view(), rods_plain.velocity_omega_view()), 0u);
  EXPECT_EQ(count_bit_differences(get<ContactViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<ContactViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<LinearSpringViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<LinearSpringViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<PinViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<PinViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedLengthViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<FixedLengthViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedPositionViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<FixedPositionViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
  EXPECT_EQ(count_bit_differences(get<FixedPoseViews<Kokkos::Serial>>(constraints_unit).lambda_view(),
                                  get<FixedPoseViews<Kokkos::Serial>>(constraints_plain).lambda_view()),
            0u);
}

// CGInvOp against the dense inverse of a random SPD system B^T M B + K^-1.

TEST(Mbody, CGInvOpMatchesDenseInverse) {
  constexpr int kNumConfig = 4;     // configuration-space dimension
  constexpr int kNumBilateral = 3;  // NY: number of bilateral constraints

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  // Random SPD M = R^T R + diag_boost.
  Matrix<double, kNumConfig, kNumConfig> M_dense;
  {
    Matrix<double, kNumConfig, kNumConfig> R;
    for (int i = 0; i < kNumConfig; ++i) {
      for (int j = 0; j < kNumConfig; ++j) {
        R(i, j) = u(rng);
      }
    }
    M_dense = transpose(R) * R;
    for (int i = 0; i < kNumConfig; ++i) {
      M_dense(i, i) += 5.0;
    }
  }

  // Random B, random positive Kinv diagonal.
  Matrix<double, kNumConfig, kNumBilateral> B_dense;
  for (int i = 0; i < kNumConfig; ++i) {
    for (int j = 0; j < kNumBilateral; ++j) {
      B_dense(i, j) = u(rng);
    }
  }

  auto Kinv_dense = Matrix<double, kNumBilateral, kNumBilateral>::zeros();
  Kokkos::View<double*, TestMemSpace> kinv_diag("kinv_diag", kNumBilateral);
  auto kinv_diag_h = Kokkos::create_mirror_view(kinv_diag);
  for (int i = 0; i < kNumBilateral; ++i) {
    const double val = 0.5 + std::abs(u(rng));
    Kinv_dense(i, i) = val;
    kinv_diag_h(i) = val;
  }
  Kokkos::deep_copy(kinv_diag, kinv_diag_h);

  // Dense reference: S = (B^T M B + K^-1)^-1
  const auto S_dense = inverse(transpose(B_dense) * M_dense * B_dense + Kinv_dense);

  // The same system as matrix-free operators
  Kokkos::View<double**, TestMemSpace> M("M", kNumConfig, kNumConfig);
  Kokkos::View<double**, TestMemSpace> B("B", kNumConfig, kNumBilateral);
  Kokkos::View<double**, TestMemSpace> Bt("Bt", kNumBilateral, kNumConfig);
  auto M_h = Kokkos::create_mirror_view(M);
  auto B_h = Kokkos::create_mirror_view(B);
  auto Bt_h = Kokkos::create_mirror_view(Bt);
  for (int i = 0; i < kNumConfig; ++i) {
    for (int j = 0; j < kNumConfig; ++j) {
      M_h(i, j) = M_dense(i, j);
    }
  }
  for (int i = 0; i < kNumConfig; ++i) {
    for (int j = 0; j < kNumBilateral; ++j) {
      B_h(i, j) = B_dense(i, j);
      Bt_h(j, i) = B_dense(i, j);
    }
  }
  Kokkos::deep_copy(M, M_h);
  Kokkos::deep_copy(B, B_h);
  Kokkos::deep_copy(Bt, Bt_h);

  using backend_t = KokkosBackend<TestExecSpace>;
  const auto btmb = make_quadratic_form<backend_t>(Bt, M, B);
  const auto kinv_op = make_diagonal_op<backend_t>(kinv_diag);
  const auto sum_op = make_sum_op<backend_t>(btmb, kinv_op);
  const CGConfig<double> cg_cfg{.max_iters = 200, .tol = 1e-10};
  const auto cg_inv = make_cg_inv_op<backend_t>(sum_op, cg_cfg);

  Kokkos::View<double*, TestMemSpace> rhs("rhs", kNumBilateral);
  Kokkos::View<double*, TestMemSpace> out("out", kNumBilateral);
  auto rhs_h = Kokkos::create_mirror_view(rhs);
  for (int i = 0; i < kNumBilateral; ++i) {
    rhs_h(i) = u(rng);
  }
  Kokkos::deep_copy(rhs, rhs_h);

  cg_inv.apply(rhs, out);
  EXPECT_TRUE(cg_inv.last_result().converged);

  Vector<double, kNumBilateral> rhs_vec;
  for (int i = 0; i < kNumBilateral; ++i) {
    rhs_vec[i] = rhs_h(i);
  }
  const auto expected = S_dense * rhs_vec;

  const auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (int i = 0; i < kNumBilateral; ++i) {
    EXPECT_NEAR(out_h(i), expected[i], 1e-6);
  }
}

//@}

}  // namespace

}  // namespace mbody

}  // namespace mundy
