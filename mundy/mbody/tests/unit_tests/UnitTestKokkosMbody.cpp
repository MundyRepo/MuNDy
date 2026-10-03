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
/// \brief Unit tests for mbody::solve_mixed_lcp, solve_mixed_slcp, and the operators and geometry kernels behind them.

// External
#include <gtest/gtest.h>  // for TEST, EXPECT_NEAR, etc

#include <Kokkos_Core.hpp>  // for Kokkos::View, Kokkos::parallel_for, Kokkos::parallel_reduce

// C++ core
#include <algorithm>  // for std::max
#include <bit>        // for std::bit_cast
#include <cmath>      // for std::abs, std::sqrt, std::pow, std::exp, std::log, std::atan2
#include <cstdint>    // for uint64_t
#include <cstring>    // for std::memcmp
#include <random>     // for std::mt19937, std::uniform_real_distribution
#include <stdexcept>  // for std::invalid_argument, std::runtime_error
#include <string>     // for std::to_string
#include <utility>    // for std::swap
#include <vector>     // for std::vector

// Mundy
#include <mundy_math/Matrix.hpp>        // for mundy::Matrix
#include <mundy_math/eigenvalues.hpp>   // for mundy::make_eigen_problem, mundy::solve_eigen_problem
#include <mundy_mbody/KokkosMbody.hpp>  // for mundy::mbody::{solve_mixed_lcp, solve_mixed_slcp, advance_rods}

namespace mundy {

namespace mbody {

namespace {

//! \name Device staging
//@{

// Library calls run on TestExecSpace; inputs are built and results checked on the host in HostExecSpace containers.
using TestExecSpace = Kokkos::DefaultExecutionSpace;
using TestMemSpace = TestExecSpace::memory_space;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

/// \brief solve_mixed_lcp() on TestExecSpace for host inputs, which are updated in place.
template <typename... Families>
MixedLCPResult solve_on_device(const RodViews<HostExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                               const MixedLCPConfig& cfg) {
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const MixedLCPResult result = solve_mixed_lcp(rods_d, constraints_d, cfg);
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

/// \brief A copy of rods' force/torque, kept apart because solve_mixed_lcp() adds the constraint forces into rods'.
template <typename Space>
Kokkos::View<double*, typename Space::memory_space> copy_load(const RodViews<Space>& rods) {
  Kokkos::View<double*, typename Space::memory_space> load("load", rods.force_torque_view().extent(0));
  Kokkos::deep_copy(load, rods.force_torque_view());
  return load;
}

/// \brief force/torque := load and velocity/omega := 0, the state solve_mixed_lcp() expects at the start of a step.
template <typename Space>
void reset_rod_state(const RodViews<Space>& rods, const Kokkos::View<double*, typename Space::memory_space>& load) {
  Kokkos::deep_copy(rods.force_torque_view(), load);
  Kokkos::deep_copy(rods.velocity_omega_view(), 0.0);
}

/// \brief One backward-Euler step under a constant external load.
template <typename Space, typename... Families>
MixedLCPResult step_rods(const RodViews<Space>& rods, const ConstraintSet<Families...>& constraints,
                         const MixedLCPConfig& cfg, const Kokkos::View<double*, typename Space::memory_space>& load) {
  reset_rod_state(rods, load);
  const MixedLCPResult result = solve_mixed_lcp(rods, constraints, cfg);
  advance_rods(rods, cfg.dt);
  return result;
}

/// \brief One step under a constant external load with its bilateral rows held at its end.
template <typename Space, typename... Families>
MixedSLCPResult step_rods(const RodViews<Space>& rods, const ConstraintSet<Families...>& constraints,
                          const MixedSLCPConfig& cfg, const Kokkos::View<double*, typename Space::memory_space>& load) {
  reset_rod_state(rods, load);
  const MixedSLCPResult result = solve_mixed_slcp(rods, constraints, cfg);
  advance_rods(rods, cfg.inner_lcp_config.dt);
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

/// \brief Step until no rod moves farther than settled_step, or turns through a larger angle, in one step.
///
/// Returns whether that happened within max_steps.
template <typename Space, typename... Families>
bool step_until_settled(const RodViews<Space>& rods, const ConstraintSet<Families...>& constraints,
                        const MixedLCPConfig& cfg, const Kokkos::View<double*, typename Space::memory_space>& load,
                        double settled_step, int max_steps) {
  for (int step = 0; step < max_steps; ++step) {
    MUNDY_THROW_REQUIRE(step_rods(rods, constraints, cfg, load).converged, std::runtime_error,
                        "step_until_settled: a step failed to converge.");
    if (max_step_displacement(rods, cfg.dt) <= settled_step) {
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

/// \brief The inputs to one solve_mixed_lcp().
template <typename... Families>
struct SolveInput {
  RodViews<HostExecSpace> rods;
  ConstraintSet<Families...> constraints;
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
  cfg.dt = 0.5;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-10;
  cfg.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.outer_tol = 1e-10;
  return {rods, make_constraint_set(linear_springs, angular_springs), cfg};
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

  Kokkos::View<int*, Kokkos::HostSpace> owner("owner", kNumRows);
  for (size_t p = 0; p < kNumRows; ++p) {
    owner(p) = owners[p];
  }

  std::mt19937 rng(20260925);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const impl::SingleGeometry<HostExecSpace> geo_h(owner);
  for (size_t p = 0; p < kNumRows; ++p) {
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

  Kokkos::View<int*, Kokkos::HostSpace> owner_i("owner_i", kNumRows), owner_j("owner_j", kNumRows);
  for (size_t p = 0; p < kNumRows; ++p) {
    owner_i(p) = owners_i[p];
    owner_j(p) = owners_j[p];
  }

  std::mt19937 rng(20261002);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const impl::PairGeometry<HostExecSpace> geo_h(owner_i, owner_j);
  for (size_t p = 0; p < kNumRows; ++p) {
    const int row = static_cast<int>(p);
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

// One constraint through the full solve_mixed_lcp(). A single spring reduces the Schur complement to the scalar
// equation (B^T M B + 1/k) y = -b0, and a single contact to the scalar LCP lambda = max(0, -sep0 / A). The scalars come
// from the same operators solve_mixed_lcp() uses, so these check its assembly of the operators, not the operators.
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

  MixedLCPConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(lin_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, lin_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, cfg);
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

  MixedLCPConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(ang_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, ang_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, cfg);
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

  MixedLCPConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;
  cfg.cg_tol = 1e-14;

  // Scalar B^T M B
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(triple_springs);
  const impl::TripleGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, triple_springs), b0_d);
  const impl::TripleForceOp<TestExecSpace> B(geo, rods.size());
  const impl::TripleForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const double btmb = scalar_quadratic_form(BT, M, B);
  const double b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, cfg);
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

  MixedLCPConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;
  cfg.outer_tol = 1e-12;

  // Scalar A := D^T M D
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto sep0 = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, contacts), sep0);
  const impl::PairForceOp<TestExecSpace> D(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> DT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const double A_value = scalar_quadratic_form(DT, M, D);
  const double sep0_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep0)(0);

  // Solve
  const MixedLCPResult result = solve_on_device(rods, constraints, cfg);
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

  MixedLCPConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  Kokkos::View<double*, TestMemSpace> vel_omega_expected_d("vel_omega_expected", 6 * rods.size());
  M.apply(rods_d.force_torque_view(), vel_omega_expected_d);
  const auto vel_omega_expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vel_omega_expected_d);

  const MixedLCPResult result = solve_on_device(rods, constraints, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  for (size_t i = 0; i < rods.size(); ++i) {
    const Vector3d vel_expected = rod_velocity(vel_omega_expected, static_cast<int>(i));
    const Vector3d omega_expected = rod_omega(vel_omega_expected, static_cast<int>(i));
    EXPECT_NEAR(norm(rods.velocity(i) - vel_expected), 0.0, 1e-12);
    EXPECT_NEAR(norm(rods.omega(i) - omega_expected), 0.0, 1e-12);
  }
}

// The Schur complement's CG on a 0x0 system: zero iterations and zero residual.
TEST(Mbody, EmptySpringBlockSchurComplementConvergesInZeroIterations) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const LinearSpringViews<TestExecSpace> lin_springs(0);
  const AngularSpringViews<TestExecSpace> ang_springs(0);

  auto b0_lin = make_constraint_values(lin_springs);
  auto b0_ang = make_constraint_values(ang_springs);
  const impl::PairGeometry<TestExecSpace> lin_geo = impl::compute_geometry(rods_d, lin_springs, b0_lin);
  const impl::PairGeometry<TestExecSpace> ang_geo = impl::compute_geometry(rods_d, ang_springs, b0_ang);
  const impl::PairGeometry<TestExecSpace> spring_geo = impl::concat_pair_geometry(lin_geo, ang_geo);

  const impl::PairForceOp<TestExecSpace> B(spring_geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(spring_geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(1.0, rods_d);
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

  // Dense reference, built before solve_mixed_lcp() updates the inputs; B's columns follow the y-block order
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto lin_springs_d = create_mirror_view_and_copy(TestExecSpace{}, lin_springs);
  const auto ang_springs_d = create_mirror_view_and_copy(TestExecSpace{}, ang_springs);
  auto b0_lin_d = make_constraint_values(lin_springs_d);
  auto b0_ang_d = make_constraint_values(ang_springs_d);
  const impl::PairForceOp<TestExecSpace> B_lin(impl::compute_geometry(rods_d, lin_springs_d, b0_lin_d), kChainNumRods);
  const impl::PairForceOp<TestExecSpace> B_ang(impl::compute_geometry(rods_d, ang_springs_d, b0_ang_d), kChainNumRods);
  const impl::LocalDragMobilityOp<TestExecSpace> M(p.cfg.viscosity, rods_d);
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
      std::vector<double>(force_torque_ext.data(), force_torque_ext.data() + force_torque_ext.size()), p.cfg.dt);

  // Solve
  const MixedLCPResult result = solve_on_device(p.rods, p.constraints, p.cfg);
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
// with only its springs as with every family present and the others empty. Serial execution fixes the order of every
// atomic sum; an empty block may still turn a -0.0 into +0.0, so values, not bit patterns, are compared.
TEST(Mbody, AbsentFamiliesMatchEmptyFamilies) {
  const auto p = make_chain_problem(/*spring_constant=*/3.0);
  const auto lin_springs = copy_to<Kokkos::Serial>(get<LinearSpringViews<HostExecSpace>>(p.constraints));
  const auto ang_springs = copy_to<Kokkos::Serial>(get<AngularSpringViews<HostExecSpace>>(p.constraints));
  const auto springs_only = make_constraint_set(lin_springs, ang_springs);
  const auto every_family =
      make_constraint_set(lin_springs, ang_springs, PinViews<Kokkos::Serial>(0), FixedLengthViews<Kokkos::Serial>(0),
                          TriplePointAngularSpringViews<Kokkos::Serial>(0), FixedPositionViews<Kokkos::Serial>(0),
                          FixedPoseViews<Kokkos::Serial>(0), ContactViews<Kokkos::Serial>(0));

  // Mixed LCP
  const auto rods_reduced = copy_to<Kokkos::Serial>(p.rods);
  const auto rods_full = copy_to<Kokkos::Serial>(p.rods);
  ASSERT_TRUE(solve_mixed_lcp(rods_reduced, springs_only, p.cfg).converged);
  const auto lin_lambda_reduced = copy_to<Kokkos::Serial>(lin_springs).lambda_view();
  const auto ang_lambda_reduced = copy_to<Kokkos::Serial>(ang_springs).lambda_view();
  ASSERT_TRUE(solve_mixed_lcp(rods_full, every_family, p.cfg).converged);
  EXPECT_EQ(count_value_differences(rods_reduced.force_torque_view(), rods_full.force_torque_view()), 0u);
  EXPECT_EQ(count_value_differences(rods_reduced.velocity_omega_view(), rods_full.velocity_omega_view()), 0u);
  EXPECT_EQ(count_value_differences(lin_lambda_reduced, lin_springs.lambda_view()), 0u);
  EXPECT_EQ(count_value_differences(ang_lambda_reduced, ang_springs.lambda_view()), 0u);

  // Mixed SLCP, which re-linearizes the springs
  const MixedSLCPConfig slcp_cfg{p.cfg, 50, 1e-9, 1e-9};
  const auto slcp_rods_reduced = copy_to<Kokkos::Serial>(p.rods);
  const auto slcp_rods_full = copy_to<Kokkos::Serial>(p.rods);
  const MixedSLCPResult reduced = solve_mixed_slcp(slcp_rods_reduced, springs_only, slcp_cfg);
  ASSERT_TRUE(reduced.converged) << reduced;
  ASSERT_GE(reduced.num_iters, 2u);
  const MixedSLCPResult full = solve_mixed_slcp(slcp_rods_full, every_family, slcp_cfg);
  EXPECT_EQ(full.num_iters, reduced.num_iters);
  EXPECT_EQ(full.residual, reduced.residual);
  EXPECT_EQ(count_value_differences(slcp_rods_reduced.force_torque_view(), slcp_rods_full.force_torque_view()), 0u);
  EXPECT_EQ(count_value_differences(slcp_rods_reduced.velocity_omega_view(), slcp_rods_full.velocity_omega_view()), 0u);
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

    MixedLCPConfig cfg;
    cfg.dt = dt;
    cfg.viscosity = 1.0;
    cfg.max_cg_iters = 1000;
    cfg.cg_tol = 1e-12;
    cfg.max_outer_iters = 1;

    rods.force(num_spheres / 2) = Vector3d{0.0, -load, 0.0};

    // Dense reference, built before solve_mixed_lcp() updates the inputs; B's columns follow the y-block order
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
    const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);

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
    ASSERT_TRUE(solve_on_device(rods, constraints, cfg).converged) << "dt=" << dt;

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

  MixedLCPConfig cfg;
  cfg.dt = 0.3;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 1000;
  cfg.cg_tol = 1e-10;
  cfg.max_outer_iters = 1;

  // Dense reference, built before solve_mixed_lcp() updates the inputs; B's columns follow the y-block order
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
  const impl::LocalDragMobilityOp<TestExecSpace> M_op(cfg.viscosity, rods_d);

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
      cfg.dt);
  const double y_bound = cfg.cg_tol * dense_inverse_frobenius_norm(dense_schur_matrix(B, M, kinv, cfg.dt));
  const double v_bound = dense_frobenius_norm(dense_matmul(M, B)) * y_bound;

  // Solve
  ASSERT_TRUE(solve_on_device(rods, constraints, cfg).converged);

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

// A rigid anchor reaches its target in one step, and its multiplier is the reaction to the load, at every dt.
TEST(Mbody, FixedPositionHoldsItsTargetAtAnyDt) {
  const Vector3d target{0.4, -0.3, 1.1};
  const Vector3d load{0.7, 0.25, -0.5};

  for (const double dt : {0.005, 0.5, 2.0, 20.0}) {
    RodViews<HostExecSpace> rods(1);
    rods.center(0) = target + Vector3d{-0.35, 0.2, 0.15};  // starts off target
    rods.orientation(0) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(0) = 0.2;
    rods.length(0) = 1.0;

    FixedPositionViews<HostExecSpace> anchors(1);
    set_fixed_position(anchors, 0, /*rod=*/0, target);
    const auto constraints = make_constraint_set(anchors);

    MixedLCPConfig cfg;
    cfg.dt = dt;
    cfg.viscosity = 1.0;
    cfg.max_cg_iters = 200;
    cfg.cg_tol = 1e-14;
    cfg.max_outer_iters = 1;

    zero_rod_state(rods);
    rods.force(0) = load;
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
    const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
    const auto load_d = copy_load(rods_d);

    ASSERT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged) << "dt=" << dt;
    deep_copy(rods, rods_d);
    EXPECT_NEAR(norm(rods.center(0) - target), 0.0, 1e-10) << "dt=" << dt;

    reset_rod_state(rods_d, load_d);
    ASSERT_TRUE(solve_mixed_lcp(rods_d, constraints_d, cfg).converged) << "dt=" << dt;
    deep_copy(rods, rods_d);
    deep_copy(constraints, constraints_d);
    EXPECT_NEAR(norm(rods.velocity(0)), 0.0, 1e-10) << "dt=" << dt;
    EXPECT_NEAR(norm(anchors.lambda(0) + load), 0.0, 1e-10) << "dt=" << dt;
  }
}

// A rigid pose anchor holds its point and orientation, with multipliers equal to the reaction to the load.
TEST(Mbody, FixedPoseHoldsItsTarget) {
  const Vector3d target_point{0.4, -0.3, 1.1};
  const Quaterniond target_orientation = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.6);
  const Vector3d body_offset{0.0, 0.0, 0.45};
  const Vector3d load{0.7, 0.25, -0.5};

  for (const double dt : {0.05, 0.5, 2.0}) {
    RodViews<HostExecSpace> rods(1);
    rods.center(0) = target_point + Vector3d{-0.3, 0.25, 0.1};
    rods.orientation(0) = Quaterniond{1.0, 0.0, 0.0, 0.0};  // starts both displaced and misoriented
    rods.radius(0) = 0.2;
    rods.length(0) = 0.9;

    FixedPoseViews<HostExecSpace> anchors(1);
    set_fixed_pose(anchors, 0, /*rod=*/0, target_point, target_orientation, body_offset);
    const auto constraints = make_constraint_set(anchors);

    MixedLCPConfig cfg;
    cfg.dt = dt;
    cfg.viscosity = 1.0;
    cfg.max_cg_iters = 500;
    cfg.cg_tol = 1e-14;
    cfg.max_outer_iters = 1;

    zero_rod_state(rods);
    rods.force(0) = load;
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
    const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
    const auto load_d = copy_load(rods_d);
    ASSERT_TRUE(step_until_settled(rods_d, constraints_d, cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000))
        << "not settled at dt=" << dt;
    deep_copy(rods, rods_d);
    deep_copy(constraints, constraints_d);

    const Vector3d r_world = rods.orientation(0) * body_offset;
    EXPECT_NEAR(norm(rods.center(0) + r_world - target_point), 0.0, 1e-9) << "dt=" << dt;
    EXPECT_NEAR(norm(quaternion_to_rotation_vector(rods.orientation(0) * inverse(target_orientation))), 0.0, 1e-9)
        << "dt=" << dt;
    EXPECT_NEAR(norm(anchors.position_lambda(0) + load), 0.0, 1e-9) << "dt=" << dt;
    EXPECT_NEAR(norm(anchors.orientation_lambda(0) - cross(r_world, load)), 0.0, 1e-9) << "dt=" << dt;
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

      MixedLCPConfig cfg;
      cfg.dt = dt;
      cfg.viscosity = 1.0;
      cfg.max_cg_iters = 500;
      cfg.cg_tol = 1e-14;
      cfg.max_outer_iters = 1;

      rods.force(0) = position_load;
      rods.force(1) = pose_load;
      rods.torque(1) = applied_torque;
      const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
      const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
      const auto load_d = copy_load(rods_d);
      ASSERT_TRUE(step_until_settled(rods_d, constraints_d, cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000))
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

  MixedLCPConfig cfg;
  cfg.dt = 0.5;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.outer_tol = 1e-12;

  rods.force(1) = Vector3d{0.0, 0.0, -push};
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  ASSERT_TRUE(step_until_settled(rods_d, constraints_d, cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000));
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

/// \brief Two rods joined by one linear spring, stepped num_steps times; the stretch before and after each step.
std::vector<double> run_relaxation(double spring_constant, double dt, int num_steps, double radius = 0.2,
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

  MixedLCPConfig cfg;
  cfg.dt = dt;
  cfg.viscosity = viscosity;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate

  zero_rod_state(rods);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto& lin_springs_d = get<LinearSpringViews<TestExecSpace>>(constraints_d);
  const auto load_d = copy_load(rods_d);

  std::vector<double> stretch{summed_stretch(rods_d, lin_springs_d)};
  for (int step = 0; step < num_steps; ++step) {
    EXPECT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged) << "step " << step;
    stretch.push_back(summed_stretch(rods_d, lin_springs_d));
  }
  return stretch;
}

// The stretch follows the backward-Euler closed form exactly, and the continuous decay s0 exp(-t/tau) closely at this
// modest dt/tau.
TEST(Mbody, SpringRelaxationMatchesBackwardEulerAndExponentialDecay) {
  const double radius = 0.2, length = 1.0, viscosity = 1.0, spring_constant = 2.0;
  const double A = 2.0 * expected_inv_drag_para(radius, length, viscosity);  // both rods move
  const double tau = 1.0 / (A * spring_constant);

  const double dt = 0.2 * tau;
  const int num_steps = 10;
  const std::vector<double> stretch = run_relaxation(spring_constant, dt, num_steps, radius, length, viscosity);

  const double s0 = stretch[0];
  for (int n = 0; n <= num_steps; ++n) {
    const double backward_euler = s0 / std::pow(1.0 + dt / tau, n);
    EXPECT_NEAR(stretch[n], backward_euler, 1e-8 + 1e-6 * std::abs(backward_euler)) << "step " << n;

    const double continuous = s0 * std::exp(-n * dt / tau);
    EXPECT_NEAR(stretch[n], continuous, 0.05 * s0) << "step " << n << " (vs. continuous exp(-t/tau))";
  }
}

// From dt = 0.1 tau to 50 tau the stretch follows the closed form and decays monotonically, never overshooting zero.
TEST(Mbody, SpringRelaxationStableAcrossWideDtSweep) {
  const double radius = 0.2, length = 1.0, viscosity = 1.0, spring_constant = 2.0;
  const double A = 2.0 * expected_inv_drag_para(radius, length, viscosity);
  const double tau = 1.0 / (A * spring_constant);

  for (const double dt_over_tau : {0.1, 1.0, 5.0, 20.0, 50.0}) {
    const double dt = dt_over_tau * tau;
    const int num_steps = 6;
    const std::vector<double> stretch = run_relaxation(spring_constant, dt, num_steps, radius, length, viscosity);

    const double s0 = stretch[0];
    for (int n = 0; n <= num_steps; ++n) {
      const double backward_euler = s0 / std::pow(1.0 + dt / tau, n);
      EXPECT_NEAR(stretch[n], backward_euler, 1e-8 + 1e-6 * std::abs(backward_euler))
          << "dt/tau=" << dt_over_tau << " step " << n;
    }
    for (int n = 1; n <= num_steps; ++n) {
      EXPECT_LE(std::abs(stretch[n]), std::abs(stretch[n - 1]) + 1e-12)
          << "dt/tau=" << dt_over_tau << ": stretch grew at step " << n << " (unstable)";
      EXPECT_GE(stretch[n], 0.0) << "dt/tau=" << dt_over_tau << ": stretch went negative at step " << n
                                 << " (should decay monotonically toward zero, never overshoot)";
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
  auto b0_lin = make_constraint_values(lin_springs_d);
  auto b0_ang = make_constraint_values(ang_springs_d);
  const impl::PairGeometry<TestExecSpace> geo = impl::concat_pair_geometry(
      impl::compute_geometry(rods_d, lin_springs_d, b0_lin), impl::compute_geometry(rods_d, ang_springs_d, b0_ang));
  const impl::PairForceOp<TestExecSpace> B(geo, p.rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, p.rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(p.cfg.viscosity, rods_d);

  // Spring rows are packed linear then angular, matching the concatenated geometry.
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
// (I + dt K^1/2 B^T M B K^1/2)^-1 K^1/2 b.
TEST(Mbody, ChainStableAcrossExplicitStabilityLimit) {
  const double dt_crit = 2.0 / spring_network_stiffness(make_chain_problem(/*spring_constant=*/3.0));

  for (const double cfl : {0.5, 0.9, 1.1, 2.0, 10.0, 100.0}) {
    auto p = make_chain_problem(/*spring_constant=*/3.0);
    p.cfg.dt = cfl * dt_crit;
    p.cfg.cg_tol = 1e-13;

    zero_rod_state(p.rods);  // no external load: the springs relax toward rest
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
    const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
    const auto& lin_springs_d = get<LinearSpringViews<TestExecSpace>>(constraints_d);
    const auto& ang_springs_d = get<AngularSpringViews<TestExecSpace>>(constraints_d);
    const auto load_d = copy_load(rods_d);

    std::vector<double> energy{elastic_energy(rods_d, lin_springs_d, ang_springs_d)};
    for (int step = 0; step < 40; ++step) {
      ASSERT_TRUE(step_rods(rods_d, constraints_d, p.cfg, load_d).converged) << "cfl=" << cfl << " step " << step;
      energy.push_back(elastic_energy(rods_d, lin_springs_d, ang_springs_d));
    }

    // Round-off floor relative to the initial energy, which the largest steps decay toward.
    const double floor = 1e-10 * energy.front();
    for (size_t n = 0; n + 1 < energy.size(); ++n) {
      EXPECT_LE(energy[n + 1], energy[n] + floor) << "cfl=" << cfl << ": energy rose at step " << n + 1;
    }
    EXPECT_LT(energy.back(), energy.front()) << "cfl=" << cfl;
  }
}

//@}

//! \name Sequential linearization
//@{

// solve_mixed_slcp: the bilateral rows re-linearized at the configurations its iterates move the rods to.

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

  MixedLCPConfig cfg;
  cfg.dt = 0.3;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 1000;
  cfg.cg_tol = 1e-10;
  cfg.outer_tol = 1e-10;
  return SolveInput{rods, make_constraint_set(lin_springs, pins, lengths, pose_anchors, position_anchors, contacts),
                    cfg};
}

// A sequence that cannot converge returns its first linearization, which is the mixed LCP step, bit for bit. Serial
// execution fixes the order of every atomic sum, so the two solves round identically.
TEST(Mbody, Fallback) {
  const auto p = make_fallback_problem();
  const auto rods_lcp = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_lcp = copy_to<Kokkos::Serial>(p.constraints);
  const auto rods_slcp = copy_to<Kokkos::Serial>(p.rods);
  const auto constraints_slcp = copy_to<Kokkos::Serial>(p.constraints);

  // Solve
  const MixedLCPResult lcp = solve_mixed_lcp(rods_lcp, constraints_lcp, p.cfg);
  const MixedSLCPResult slcp = solve_mixed_slcp(rods_slcp, constraints_slcp, MixedSLCPConfig{p.cfg, 3, 1e-300, 1e-300});
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

// A bob held at length L from an anchored sphere under a constant force F turns in the plane normal to its axis, where
// its mobility is m I. At the end of the step it sits at r_k + dt m F projected radially onto |r| = L, so with theta
// measured from -y, tan theta_{k+1} = sin theta_k / (cos theta_k + dt/tau), tau = L / (m f).
//
// An accepted iterate's force direction changes the step by at most sqrt(2) length_tol in the plane, which turns the
// bob by at most sqrt(2) length_tol / |r_k + dt m F|, to first order in the iterate's angular error.
TEST(Mbody, PendulumStep) {
  const double radius = 0.2, L = 1.0, f = 0.5, theta0 = 1.2;
  const double m = expected_inv_drag_perp(radius, 0.0, 1.0);
  const double tau = L / (m * f);

  RodViews<HostExecSpace> rods(2);
  rods.center(0) = Vector3d{0.0, 0.0, 0.0};
  rods.center(1) = Vector3d{L * std::sin(theta0), -L * std::cos(theta0), 0.0};
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
  const auto constraints = make_constraint_set(anchors, lengths);

  MixedSLCPConfig cfg;
  cfg.inner_lcp_config.dt = 0.1 * tau;
  cfg.inner_lcp_config.viscosity = 1.0;
  cfg.inner_lcp_config.max_cg_iters = 200;
  cfg.inner_lcp_config.cg_tol = 1e-14;
  cfg.inner_lcp_config.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate
  cfg.max_iters = 50;
  cfg.length_tol = 1e-11;
  cfg.angle_tol = 1e-11;

  // Step
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  const MixedSLCPResult result = step_rods(rods_d, constraints_d, cfg, load_d);
  deep_copy(rods, rods_d);
  ASSERT_TRUE(result.converged) << result;
  EXPECT_GE(result.num_iters, 2u) << "the first linearization leaves |r| - L at second order in the step";

  // Radial projection
  const double h = cfg.inner_lcp_config.dt / tau;
  const double p_norm = L * std::sqrt(1.0 + 2.0 * h * std::cos(theta0) + h * h);
  const Vector3d r = rods.center(1) - rods.center(0);
  EXPECT_NEAR(norm(r), L, cfg.length_tol);
  EXPECT_NEAR(std::atan2(r[0], -r[1]), std::atan2(std::sin(theta0), std::cos(theta0) + h),
              std::sqrt(2.0) * cfg.length_tol / p_norm);
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

  MixedLCPConfig cfg;
  cfg.dt = 2.0;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.max_outer_iters = 1;  // no contacts, so PGD has nothing to iterate

  rods.force(0) = Vector3d{0.0, 0.0, -tip_force};
  rods.force(num_spheres - 1) = Vector3d{0.0, 0.0, tip_force};
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);

  const int num_steps = 150;
  for (int step = 0; step < num_steps; ++step) {
    EXPECT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged)
        << "num_segments=" << num_segments << " step " << step;
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
/// CantileverSettlesToStaticBend checks that solve_mixed_lcp() settles to this equilibrium.
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

/// \brief A cantilever chain in rods 0 to num_segments + 1 of num_rods rods.
///
/// Linear springs join neighbours and a bend spring sits at each interior rod; callers hold rods 0 and 1 as the wall
/// and set any rods past the chain. The free rods start on a shallow arc so every bend angle has a gradient on the
/// first step, and the settled state does not depend on it.
SolveInput<LinearSpringViews<HostExecSpace>, TriplePointAngularSpringViews<HostExecSpace>> make_cantilever(
    size_t num_segments, double L, double EI, size_t num_rods) {
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

  LinearSpringViews<HostExecSpace> lin_springs(num_chain - 1);
  for (size_t k = 0; k + 1 < num_chain; ++k) {
    lin_springs.rod_i(k) = static_cast<int>(k);
    lin_springs.rod_j(k) = static_cast<int>(k + 1);
    lin_springs.rest_length(k) = spacing;
    lin_springs.spring_constant(k) = 1.0e5 / (spacing * spacing);  // effectively inextensible
  }

  MixedLCPConfig cfg;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 1000;
  // Tight enough that solver error sits far below the settled chain's geometric nonlinearity.
  cfg.cg_tol = 1e-14;
  cfg.outer_tol = 1e-12;
  return {rods, make_constraint_set(lin_springs, chain.springs), cfg};
}

/// \brief The settled state of a cantilever under a tip load.
struct SettledCantilever {
  bool settled;
  std::vector<double> displacement;  // transverse, of the free rods 2 to num_segments + 1
};

/// \brief A cantilever settled under a transverse tip load.
SettledCantilever run_settled_cantilever(size_t num_segments, double L, double EI, double tip_force, double dt) {
  const size_t num_chain = num_segments + 2;
  const int tip = static_cast<int>(num_chain - 1);
  auto p = make_cantilever(num_segments, L, EI, num_chain);
  p.rods.force(tip) = Vector3d{0.0, tip_force, 0.0};

  FixedPositionViews<HostExecSpace> wall(2);
  set_fixed_position(wall, 0, /*rod=*/0, Vector3d(p.rods.center(0)));
  set_fixed_position(wall, 1, /*rod=*/1, Vector3d(p.rods.center(1)));
  const auto constraints = make_constraint_set(get<LinearSpringViews<HostExecSpace>>(p.constraints),
                                               get<TriplePointAngularSpringViews<HostExecSpace>>(p.constraints), wall);
  p.cfg.dt = dt;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  SettledCantilever result;
  result.settled = step_until_settled(rods_d, constraints_d, p.cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000);
  deep_copy(p.rods, rods_d);

  for (size_t k = 2; k < num_chain; ++k) {
    result.displacement.push_back(p.rods.center(k)[1]);
  }
  return result;
}

// Without contacts, the settled chain is the static equilibrium solve_static_bend computes. Every free sphere's
// deflection matches it up to solve_mixed_lcp()'s nonlinear geometry, which at this load moves the deflection by about
// 3e-5 of the tip's, shrinking as P^2.
TEST(Mbody, CantileverSettlesToStaticBend) {
  const double L = 8.0, EI = 5.0, tip_force = 1e-3;
  for (const size_t num_segments : {4, 8, 16}) {
    const double spacing = L / static_cast<double>(num_segments);
    std::vector<double> f_ext(num_segments, 0.0);
    f_ext.back() = tip_force;
    const std::vector<double> expected = solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, f_ext);
    const SettledCantilever r = run_settled_cantilever(num_segments, L, EI, tip_force, /*dt=*/100.0);

    ASSERT_TRUE(r.settled) << "num_segments=" << num_segments;
    for (size_t k = 0; k < expected.size(); ++k) {
      EXPECT_NEAR(r.displacement[k], expected[k], 1e-4 * expected.back())
          << "free sphere " << k << " at num_segments=" << num_segments;
    }
  }
}

/// \brief The settled state of a propped cantilever.
struct ProppedCantilever {
  bool settled;
  double contact_force;
  double tip_gap;
};

/// \brief A cantilever whose tip rests on an anchored sphere, settled under a midspan load.
ProppedCantilever run_propped_cantilever(size_t num_segments, double L, double EI, double load, double dt) {
  const size_t num_chain = num_segments + 2;
  const int midspan = static_cast<int>(num_segments / 2 + 1);
  const int tip = static_cast<int>(num_chain - 1);
  const int obstacle = static_cast<int>(num_chain);
  auto p = make_cantilever(num_segments, L, EI, num_chain + 1);

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
  const auto constraints =
      make_constraint_set(get<LinearSpringViews<HostExecSpace>>(p.constraints),
                          get<TriplePointAngularSpringViews<HostExecSpace>>(p.constraints), anchors, contacts);
  p.cfg.dt = dt;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  const bool settled =
      step_until_settled(rods_d, constraints_d, p.cfg, load_d, /*settled_step=*/1e-12, /*max_steps=*/1000);
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
// first order in the spacing toward Euler-Bernoulli's 5P/16. solve_mixed_lcp() keeps the chain's geometry nonlinear,
// which at this load moves the reaction by at most about 1e-6 of itself, shrinking as P^2.
TEST(Mbody, ProppedCantileverMatchesHenckyBarChain) {
  const double L = 8.0, EI = 5.0, load = 0.01;
  const double continuum = 5.0 * load / 16.0;

  double finest_scaled_error = 0.0;
  double finest_scaled_error_expected = 0.0;
  for (const size_t num_segments : {4, 8, 16, 32}) {
    const double n = static_cast<double>(num_segments);
    const double hencky = load * (n + 2.0) * (5.0 * n + 2.0) / (8.0 * (n + 1.0) * (2.0 * n + 1.0));
    const ProppedCantilever r = run_propped_cantilever(num_segments, L, EI, load, /*dt=*/100.0);

    ASSERT_TRUE(r.settled) << "num_segments=" << num_segments;
    EXPECT_NEAR(r.contact_force, hencky, 3e-6 * hencky) << "num_segments=" << num_segments;
    EXPECT_NEAR(r.tip_gap, 0.0, 1e-12) << "num_segments=" << num_segments;
    finest_scaled_error = n * (r.contact_force - continuum) / continuum;
    finest_scaled_error_expected = (9.0 * n + 3.0) / (10.0 * n + 15.0 + 5.0 / n);
  }

  // N rel_err = (9N + 3) / (10N + 15 + 5/N), which falls to 9/10
  EXPECT_NEAR(finest_scaled_error, finest_scaled_error_expected, 1e-4);
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
