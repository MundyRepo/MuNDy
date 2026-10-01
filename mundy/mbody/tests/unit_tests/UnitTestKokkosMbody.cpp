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
/// \brief Correctness tests (T1-T10) for the rod/spring/contact MCQPP solver in KokkosMbody.hpp.
///
/// See that file for the underlying math and KokkosMbodyImpl.hpp for the operator/geometry machinery
/// several of these tests exercise directly (white-box, via mundy::mbody::impl::...). T1-T6 and T9
/// are all single-step (call solve() once, or exercise impl:: pieces directly); T7, T8, and T10 are
/// multi-step time integration -- see their own section comments.

// C++ core
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// GTest
#include <gtest/gtest.h>

// Mundy
#include <mundy_math/Matrix.hpp>
#include <mundy_math/eigenvalues.hpp>
#include <mundy_math/lcp.hpp>
#include <mundy_mbody/KokkosMbody.hpp>

// ==========================================================================================================
// Stage 1 tests: T1 -- Jacobian consistency via finite differences.
//
// For each constraint type, perturb every rod's pose by (velocity, omega) * eps, recompute the
// geometry kernel's constraint value (separation / stretch / angle), and compare the finite
// difference against PairForceOpT's analytical rate. This is the acceptance test for
// impl::PairForceOp/PairForceOpT before anything is built on top of them.
// ==========================================================================================================
namespace mundy {

namespace mbody {

namespace {

// Every library call runs on TestExecSpace. Inputs are built and results inspected on the host, in containers typed on
// HostExecSpace, and staged across with create_mirror_view_and_copy / deep_copy.
using TestExecSpace = Kokkos::DefaultExecutionSpace;
using TestMemSpace = TestExecSpace::memory_space;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

// solve() on TestExecSpace against device copies of host-staged inputs; the rods' and every family's state
// (forces, velocities, multipliers) is copied back.
PGDResult<double> solve_on_device(const RodViews<HostExecSpace>& rods, const ConstraintSet<HostExecSpace>& constraints,
                                  const SolveConfig& cfg) {
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const PGDResult<double> result = solve(rods_d, constraints_d, cfg);
  deep_copy(rods, rods_d);
  deep_copy(constraints, constraints_d);
  return result;
}

// A constraint-value vector sized for one family. The geometry kernels fill a caller-owned view, so
// the white-box tests that drive a kernel directly (rather than through solve()) size one here.
template <typename FamilyViews>
Kokkos::View<double*, TestMemSpace> make_constraint_values(const FamilyViews& family) {
  return Kokkos::View<double*, TestMemSpace>("constraint_values", family.num_constraints());
}

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

// Perturb every rod's pose by (vel, omega) * eps (host-only helper for finite-difference checks).
RodViews<HostExecSpace> perturb_rods(const RodViews<HostExecSpace>& rods,
                                     const Kokkos::View<double*, Kokkos::HostSpace>& vel_omega, double eps) {
  RodViews<HostExecSpace> out = make_two_rod_system(rods.center(0), rods.orientation(0), rods.center(1),
                                                    rods.orientation(1), rods.radius(0), rods.length(0));
  for (size_t i = 0; i < rods.size(); ++i) {
    const Vector3d vel = rod_velocity(vel_omega, static_cast<int>(i));
    const Vector3d omega = rod_omega(vel_omega, static_cast<int>(i));
    out.center(i) = rods.center(i) + eps * vel;

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

void zero_rod_state(const RodViews<HostExecSpace>& rods) {
  for (size_t i = 0; i < rods.size(); ++i) {
    rods.force(i) = Vector3d{0.0, 0.0, 0.0};
    rods.torque(i) = Vector3d{0.0, 0.0, 0.0};
    rods.velocity(i) = Vector3d{0.0, 0.0, 0.0};
    rods.omega(i) = Vector3d{0.0, 0.0, 0.0};
  }
}

// Time loops stage their rods and constraints once, step them in place on TestExecSpace, and copy back only what they
// assert on.

// rods' current force/torque, in storage of its own: solve() accumulates the constraint forces into rods'.
template <typename Space>
Kokkos::View<double*, typename Space::memory_space> copy_load(const RodViews<Space>& rods) {
  Kokkos::View<double*, typename Space::memory_space> load("load", rods.force_torque_view().extent(0));
  Kokkos::deep_copy(load, rods.force_torque_view());
  return load;
}

// force/torque := load and velocity/omega := 0, the state solve() expects on entry to a step.
template <typename Space>
void reset_rod_state(const RodViews<Space>& rods, const Kokkos::View<double*, typename Space::memory_space>& load) {
  Kokkos::deep_copy(rods.force_torque_view(), load);
  Kokkos::deep_copy(rods.velocity_omega_view(), 0.0);
}

// Advance every rod over dt: center += dt * velocity, orientation rotated by omega * dt.
template <typename Space>
void advance_rods(const RodViews<Space>& rods, double dt) {
  Kokkos::parallel_for(
      "advance_rods", Kokkos::RangePolicy<Space>(0, rods.size()), KOKKOS_LAMBDA(const int i) {
        rods.center(i) = rods.center(i) + dt * rods.velocity(i);
        auto orientation = rods.orientation(i);
        rotate_quaternion(orientation, Vector3d(rods.omega(i)), dt);
      });
}

// One backward-Euler step under a constant external load.
template <typename Space>
PGDResult<double> step_rods(const RodViews<Space>& rods, const ConstraintSet<Space>& constraints,
                            const SolveConfig& cfg, const Kokkos::View<double*, typename Space::memory_space>& load) {
  reset_rod_state(rods, load);
  const PGDResult<double> result = solve(rods, constraints, cfg);
  advance_rods(rods, cfg.dt);
  return result;
}

// The largest distance or angle any rod moved over its last step: dt * max(|velocity|, |omega|).
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

// Central finite difference of `value_of(rods)` with respect to (vel,omega)*eps, compared against
// PairForceOpT's analytical rate at vel_omega.
void expect_rate_matches_finite_difference(std::function<double(const RodViews<HostExecSpace>&)> value_of,
                                           const RodViews<HostExecSpace>& rods,
                                           const impl::PairGeometry<TestExecSpace>& geo,
                                           const Kokkos::View<double*, Kokkos::HostSpace>& vel_omega, double tol) {
  constexpr double eps = 1e-6;
  const RodViews<HostExecSpace> rods_plus = perturb_rods(rods, vel_omega, eps);
  const RodViews<HostExecSpace> rods_minus = perturb_rods(rods, vel_omega, -eps);
  const double finite_diff_rate = (value_of(rods_plus) - value_of(rods_minus)) / (2.0 * eps);

  impl::PairForceOpT<TestExecSpace> op_t(geo, rods.size());
  Kokkos::View<double*, TestMemSpace> rate("rate", geo.size());
  op_t.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, vel_omega), rate);

  EXPECT_NEAR(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate)(0), finite_diff_rate, tol);
}

Kokkos::View<double*, Kokkos::HostSpace> make_vel_omega(const Vector3d& vel_i, const Vector3d& omega_i,
                                                        const Vector3d& vel_j, const Vector3d& omega_j) {
  Kokkos::View<double*, Kokkos::HostSpace> v("vel_omega", 12);
  rod_velocity(v, 0) = vel_i;
  rod_omega(v, 0) = omega_i;
  rod_velocity(v, 1) = vel_j;
  rod_omega(v, 1) = omega_j;
  return v;
}

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
      impl::compute_contact_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), contacts_d, sep0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto sep = make_constraint_values(contacts);
    impl::compute_contact_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), contacts_d, sep);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep)(0);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.3, -0.1, 0.2}, Vector3d{0.1, 0.2, -0.3}, Vector3d{-0.2, 0.4, 0.1}, Vector3d{-0.3, 0.1, 0.2});

  expect_rate_matches_finite_difference(value_of, rods, geo, vel_omega, 1e-6);
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
      impl::compute_linear_spring_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(springs);
    impl::compute_linear_spring_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b)(0);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.1, 0.2, 0.05}, Vector3d{0.4, -0.2, 0.1}, Vector3d{-0.3, 0.1, -0.2}, Vector3d{0.2, 0.3, -0.1});

  expect_rate_matches_finite_difference(value_of, rods, geo, vel_omega, 1e-6);
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
      impl::compute_angular_spring_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  auto value_of = [&](const RodViews<HostExecSpace>& r) {
    auto b = make_constraint_values(springs);
    impl::compute_angular_spring_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b)(0);
  };

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.0, 0.0, 0.0}, Vector3d{0.3, -0.1, 0.2}, Vector3d{0.0, 0.0, 0.0}, Vector3d{-0.2, 0.4, 0.1});

  expect_rate_matches_finite_difference(value_of, rods, geo, vel_omega, 1e-6);
}

// Three-body, position-only bend spring (TriplePointAngularSpringViews): the angle is measured at
// rod_k (the vertex) between the position vectors to rod_i and rod_j, so unlike the two above this
// needs a genuine 3-rod system and doesn't fit expect_rate_matches_finite_difference's 2-rod-specific
// helpers -- written out directly instead of forcing that abstraction to fit.
TEST(Mbody, TriplePointAngularSpringJacobianMatchesFiniteDifference) {
  RodViews<HostExecSpace> rods(3);
  rods.center(0) = Vector3d{0.3, -0.2, 0.1};
  rods.center(1) = Vector3d{1.1, 0.4, -0.3};
  rods.center(2) = Vector3d{-0.2, 0.9, 0.5};  // the vertex
  for (size_t i = 0; i < 3; ++i) {
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};  // irrelevant: this constraint has no torque
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
  const impl::TripleGeometry<TestExecSpace> geo = impl::compute_triple_point_angular_spring_geometry(
      create_mirror_view_and_copy(TestExecSpace{}, rods), springs_d, b0);

  Kokkos::View<double*, Kokkos::HostSpace> vel_omega("vel_omega", 18);
  rod_velocity(vel_omega, 0) = Vector3d{0.3, -0.1, 0.2};
  rod_velocity(vel_omega, 1) = Vector3d{-0.2, 0.4, 0.1};
  rod_velocity(vel_omega, 2) = Vector3d{0.1, 0.2, -0.3};
  // omega left zero: unused by this constraint (no dependence on any rod's orientation).

  constexpr double eps = 1e-6;
  auto perturbed_b0 = [&](double sign) {
    RodViews<HostExecSpace> r(3);
    for (size_t i = 0; i < 3; ++i) {
      r.center(i) = rods.center(i) + sign * eps * rod_velocity(vel_omega, static_cast<int>(i));
      r.orientation(i) = rods.orientation(i);
      r.radius(i) = rods.radius(i);
      r.length(i) = rods.length(i);
    }
    auto b = make_constraint_values(springs);
    impl::compute_triple_point_angular_spring_geometry(create_mirror_view_and_copy(TestExecSpace{}, r), springs_d, b);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b)(0);
  };
  const double finite_diff_rate = (perturbed_b0(1.0) - perturbed_b0(-1.0)) / (2.0 * eps);

  const impl::TripleForceOpT<TestExecSpace> op_t(geo, rods.size());
  Kokkos::View<double*, TestMemSpace> rate("rate", 1);
  op_t.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, vel_omega), rate);

  EXPECT_NEAR(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate)(0), finite_diff_rate, 1e-6);
}

// The unary operator pair has no constraint family behind it yet, but its defining property is
// already checkable and is exact: B and B^T are adjoint, so <B x, y> == <x, B^T y> for every x, y.
// Rows deliberately share owners here -- one row per constrained degree of freedom of the same rod
// is the normal shape of a single-body constraint, and sharing is what separates accumulating into a
// rod's generalized block from overwriting it (the reason SingleForceOp's add is atomic). A forward
// operator that overwrote, or indexed the wrong block, breaks the identity; one that is merely
// mis-scaled does not, which is what the finite-difference tests above are for.
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

  // Exact algebra: only summation order separates the two sides, so the floor is round-off.
  EXPECT_NEAR(lhs, rhs, 1e-12 * std::max(1.0, std::abs(lhs)));
}

// The anchor sits on a material point well off the rod centre, and the rod is turned away from the
// identity, so the torque rows r_world x e_c are genuinely exercised. A centred anchor leaves them
// zero and would check nothing past the identity block -- exactly the degenerate case that makes an
// axis-aligned shortcut look sufficient.
TEST(Mbody, FixedPositionJacobianMatchesFiniteDifference) {
  const Quaterniond tilt = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.7);
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{0.3, -0.2, 1.4}, tilt);
  zero_rod_state(rods);

  FixedPositionViews<HostExecSpace> anchors(1);
  anchors.rod(0) = 1;
  anchors.body_offset(0) = Vector3d{0.15, -0.1, 0.5};
  anchors.target_point(0) = Vector3d{0.1, 0.1, 1.0};
  anchors.compliance(0) = Vector3d{0.0, 0.0, 0.0};

  const auto anchors_d = create_mirror_view_and_copy(TestExecSpace{}, anchors);
  auto b0 = make_constraint_values(anchors);
  const impl::SingleGeometry<TestExecSpace> geo =
      impl::compute_fixed_position_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), anchors_d, b0);

  const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
      Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}, Vector3d{0.4, -0.3, 0.2}, Vector3d{-0.2, 0.5, 0.3});

  const impl::SingleForceOpT<TestExecSpace> op_t(geo, rods.size());
  Kokkos::View<double*, TestMemSpace> rate_d("rate", anchors.num_constraints());
  op_t.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, vel_omega), rate_d);

  constexpr double eps = 1e-6;
  auto b0_plus_d = make_constraint_values(anchors);
  auto b0_minus_d = make_constraint_values(anchors);
  impl::compute_fixed_position_geometry(
      create_mirror_view_and_copy(TestExecSpace{}, perturb_rods(rods, vel_omega, eps)), anchors_d, b0_plus_d);
  impl::compute_fixed_position_geometry(
      create_mirror_view_and_copy(TestExecSpace{}, perturb_rods(rods, vel_omega, -eps)), anchors_d, b0_minus_d);
  const auto rate = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate_d);
  const auto b0_plus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_plus_d);
  const auto b0_minus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_minus_d);

  for (size_t c = 0; c < anchors.num_constraints(); ++c) {
    EXPECT_NEAR(rate(c), (b0_plus(c) - b0_minus(c)) / (2.0 * eps), 1e-8) << "row " << c;
  }
}

// Both identities a boundary condition imposed *inside* the implicit solve satisfies and one imposed
// by discarding the anchored body's velocity *after* it does not: the anchored point reaches its
// target exactly in a single step, and the multiplier is the support reaction balancing the applied
// load. Neither depends on dt, so the sweep spans four decades; the post-hoc clamp's error is O(dt)
// and would fail both at the wide end while passing at the narrow one.
TEST(Mbody, FixedPositionHoldsItsTargetAtAnyDt) {
  const Vector3d target{0.4, -0.3, 1.1};
  const Vector3d load{0.7, 0.25, -0.5};

  for (const double dt : {0.005, 0.5, 2.0, 20.0}) {
    RodViews<HostExecSpace> rods(1);
    rods.center(0) = target + Vector3d{-0.35, 0.2, 0.15};  // starts displaced from where it must end up
    rods.orientation(0) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(0) = 0.2;
    rods.length(0) = 1.0;

    ConstraintSet<HostExecSpace> constraints;
    constraints.fixed_positions = FixedPositionViews<HostExecSpace>(1);
    constraints.fixed_positions.rod(0) = 0;
    constraints.fixed_positions.target_point(0) = target;
    constraints.fixed_positions.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
    constraints.fixed_positions.compliance(0) = Vector3d{0.0, 0.0, 0.0};  // rigid

    SolveConfig cfg;
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
    ASSERT_TRUE(solve(rods_d, constraints_d, cfg).converged) << "dt=" << dt;
    deep_copy(rods, rods_d);
    deep_copy(constraints, constraints_d);
    EXPECT_NEAR(norm(rods.velocity(0)), 0.0, 1e-10) << "dt=" << dt;
    EXPECT_NEAR(norm(constraints.fixed_positions.lambda(0) + load), 0.0, 1e-10) << "dt=" << dt;
  }
}

// The matrix the pose kernel's orientation rows are built from, checked on its own against a central
// difference of the rotation vector it differentiates. The pose test below covers only errors above
// the series cutoff, so the angles here straddle it: the coefficient on [theta]_x^2 switches from its
// closed form to its series there, and the two must agree across the seam.
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

// Both halves of a fixed pose are exact. The position rows' Jacobian is exact by construction; the
// orientation rows use SO(3)'s inverse left Jacobian rather than the identity usually substituted for
// it, so they match a finite difference to round-off at any pose error rather than only near zero.
// The sweep runs out to 1.5 rad, where the identity is tens of percent wrong.
TEST(Mbody, FixedPoseJacobianMatchesFiniteDifference) {
  const Quaterniond rod_orientation = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.7);
  const Vector3d error_axis{0.0, 0.0, 1.0};

  for (const double error_angle : {1.5, 0.8, 0.4, 0.1}) {
    RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                       Vector3d{0.3, -0.2, 1.4}, rod_orientation);
    zero_rod_state(rods);

    FixedPoseViews<HostExecSpace> anchors(1);
    anchors.rod(0) = 1;
    anchors.body_offset(0) = Vector3d{0.15, -0.1, 0.5};
    anchors.target_point(0) = Vector3d{0.1, 0.1, 1.0};
    // Chosen so the orientation error is exactly error_angle about error_axis.
    anchors.target_orientation(0) = inverse(axis_angle_to_quaternion(error_axis, error_angle)) * rod_orientation;
    anchors.position_compliance(0) = Vector3d{0.0, 0.0, 0.0};
    anchors.orientation_compliance(0) = Vector3d{0.0, 0.0, 0.0};

    const auto anchors_d = create_mirror_view_and_copy(TestExecSpace{}, anchors);
    auto b0 = make_constraint_values(anchors);
    const impl::SingleGeometry<TestExecSpace> geo =
        impl::compute_fixed_pose_geometry(create_mirror_view_and_copy(TestExecSpace{}, rods), anchors_d, b0);

    const Kokkos::View<double*, Kokkos::HostSpace> vel_omega = make_vel_omega(
        Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}, Vector3d{0.4, -0.3, 0.2}, Vector3d{-0.2, 0.5, 0.3});
    const impl::SingleForceOpT<TestExecSpace> op_t(geo, rods.size());
    Kokkos::View<double*, TestMemSpace> rate_d("rate", anchors.num_constraints());
    op_t.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, vel_omega), rate_d);

    constexpr double eps = 1e-6;
    auto b0_plus_d = make_constraint_values(anchors);
    auto b0_minus_d = make_constraint_values(anchors);
    impl::compute_fixed_pose_geometry(create_mirror_view_and_copy(TestExecSpace{}, perturb_rods(rods, vel_omega, eps)),
                                      anchors_d, b0_plus_d);
    impl::compute_fixed_pose_geometry(create_mirror_view_and_copy(TestExecSpace{}, perturb_rods(rods, vel_omega, -eps)),
                                      anchors_d, b0_minus_d);
    const auto rate = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rate_d);
    const auto b0_plus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_plus_d);
    const auto b0_minus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_minus_d);

    for (size_t row = 0; row < anchors.num_constraints(); ++row) {
      EXPECT_NEAR(rate(row), (b0_plus(row) - b0_minus(row)) / (2.0 * eps), 1e-8)
          << "row " << row << " at error_angle=" << error_angle;
    }
  }
}

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

    ConstraintSet<HostExecSpace> constraints;
    constraints.fixed_poses = FixedPoseViews<HostExecSpace>(1);
    constraints.fixed_poses.rod(0) = 0;
    constraints.fixed_poses.target_point(0) = target_point;
    constraints.fixed_poses.target_orientation(0) = target_orientation;
    constraints.fixed_poses.body_offset(0) = body_offset;
    constraints.fixed_poses.position_compliance(0) = Vector3d{0.0, 0.0, 0.0};
    constraints.fixed_poses.orientation_compliance(0) = Vector3d{0.0, 0.0, 0.0};

    SolveConfig cfg;
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
    for (int step = 0; step < 60; ++step) {
      ASSERT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged) << "dt=" << dt << " step " << step;
    }
    deep_copy(rods, rods_d);
    deep_copy(constraints, constraints_d);

    const Vector3d r_world = rods.orientation(0) * body_offset;
    EXPECT_NEAR(norm(rods.center(0) + r_world - target_point), 0.0, 1e-9) << "dt=" << dt;
    EXPECT_NEAR(norm(quaternion_to_rotation_vector(rods.orientation(0) * inverse(target_orientation))), 0.0, 1e-9)
        << "dt=" << dt;
    EXPECT_NEAR(norm(constraints.fixed_poses.position_lambda(0) + load), 0.0, 1e-9) << "dt=" << dt;
    EXPECT_NEAR(norm(constraints.fixed_poses.orientation_lambda(0) - cross(r_world, load)), 0.0, 1e-9) << "dt=" << dt;
  }
}

// ==========================================================================================================
// Stage 2 test: T2 -- mundy::CGInvOp validated against a dense-inverse ground truth.
//
// Builds a random SPD system B^T M B + Kinv (the same structure UnitTestConvex.cpp's own
// RandomMixedCongruentCCQP fixture exercises, but never solve-tests) via make_quadratic_form +
// make_sum_op + make_diagonal_op, and checks CGInvOp's matrix-free apply(rhs) against the
// same system's dense inverse(...) applied to the same rhs.
// ==========================================================================================================

TEST(Mbody, CGInvOpMatchesDenseInverse) {
  constexpr int kNumConfig = 4;     // NZ: size of the intermediate (configurational) space
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

  // Dense ground truth: S = (B^T M B + Kinv)^{-1}.
  const auto S_dense = inverse(transpose(B_dense) * M_dense * B_dense + Kinv_dense);

  // The equivalent Kokkos::View-based operator pipeline (dense-matrix LinearOps, exercising
  // KokkosBackend's BLAS-gemv DenseMatView path, exactly like UnitTestConvex.cpp's own Kokkos
  // fixtures).
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

// ==========================================================================================================
// Stage 3 tests: T3/T4 -- the full mundy::mbody::solve() pipeline against exactly-solvable analytical cases.
// ==========================================================================================================

// T3: two rods, one linear spring, zero contacts, zero external load. With x empty, the mixed
// problem is a single scalar equation (B^T M B + 1/k) y = -b0. The scalar B^T M B is evaluated
// here using the same operator building blocks solve() uses internally (make_quadratic_form over
// B/M/B^T built from the same geometry kernel), so this checks that solve()'s own composition of
// those blocks (CG for a 1-dof system, PGD over an empty x-block, the y* recovery formula, and the
// output write-back) is self-consistent.
TEST(Mbody, LinearSpringOnlyMatchesScalarSchurComplement) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
  zero_rod_state(rods);

  LinearSpringViews<HostExecSpace> lin_springs(1);
  lin_springs.rod_i(0) = 0;
  lin_springs.rod_j(0) = 1;
  lin_springs.rest_length(0) = 1.0;
  lin_springs.spring_constant(0) = 2.0;

  ConstraintSet<HostExecSpace> constraints;
  constraints.linear_springs = lin_springs;

  SolveConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  // Independent scalar ground truth for B^T M B, from the same building blocks solve() uses.
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(lin_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_linear_spring_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, lin_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const auto btmb_op = make_quadratic_form<backend_t>(BT, M, B);

  Kokkos::View<double*, TestMemSpace> ones("ones", 1), btmb_result_d("btmb_result", 1);
  Kokkos::deep_copy(ones, 1.0);
  auto ws = backend_t::make_workspace(btmb_op);
  backend_t::apply(btmb_op, ones, btmb_result_d, ws);
  const auto b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d);
  const auto btmb_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, btmb_result_d);

  const PGDResult<double> result = solve_on_device(rods, constraints, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0(0) / (btmb_result(0) + 1.0 / lin_springs.spring_constant(0));
  EXPECT_NEAR(lin_springs.lambda(0), y_expected, 1e-6);

  // Momentum conservation: the two rods' forces (and torques) must be equal and opposite.
  const Vector3d total_force = rods.force(0) + rods.force(1);
  const Vector3d total_torque = rods.torque(0) + rods.torque(1);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  EXPECT_NEAR(norm(total_torque), 0.0, 1e-9);
}

struct ContactOnlyCaseResult {
  double lambda;
  double sep0;
  double A;
  bool converged;
};

// Shared setup for T4: two parallel rods (offset laterally by gap_x, so the spherocylinder-vs-
// spherocylinder distance/normal is unambiguous), zero springs, zero external load.
ContactOnlyCaseResult run_contact_only_case(double gap_x, double radius) {
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{gap_x, 0.0, 0.0},
                          Quaterniond{1.0, 0.0, 0.0, 0.0}, radius, 1.0);
  zero_rod_state(rods);

  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 0;
  contacts.rod_j(0) = 1;

  ConstraintSet<HostExecSpace> constraints;
  constraints.contacts = contacts;

  SolveConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  // Independent scalar ground truth for A := D^T M D, from the same building blocks solve() uses.
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto sep0 = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_contact_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, contacts), sep0);
  const impl::PairForceOp<TestExecSpace> D(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> DT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const auto A_op = make_quadratic_form<backend_t>(DT, M, D);

  Kokkos::View<double*, TestMemSpace> ones("ones", 1), A_result("A_result", 1);
  Kokkos::deep_copy(ones, 1.0);
  auto ws = backend_t::make_workspace(A_op);
  backend_t::apply(A_op, ones, A_result, ws);

  const double sep0_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sep0)(0);
  const double A_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A_result)(0);

  const PGDResult<double> result = solve_on_device(rods, constraints, cfg);
  return ContactOnlyCaseResult{contacts.lambda(0), sep0_value, A_value, result.converged};
}

// T4 (active branch): overlapping rods -- closed-form 1-DOF LCP solution lambda = -sep0/A > 0.
TEST(Mbody, ContactOnlyActiveMatchesScalarLCP) {
  const ContactOnlyCaseResult r = run_contact_only_case(/*gap_x=*/0.3, /*radius=*/0.2);
  EXPECT_TRUE(r.converged);
  ASSERT_LT(r.sep0, 0.0) << "test setup should start penetrating";

  const double lambda_expected = -r.sep0 / r.A;
  EXPECT_GT(r.lambda, 0.0);
  EXPECT_NEAR(r.lambda, lambda_expected, 1e-4);
}

// T4 (inactive branch): separated rods -- closed-form 1-DOF LCP solution lambda = 0, and
// complementarity slackness lambda * sep0 = 0 holds trivially.
TEST(Mbody, ContactOnlyInactiveMatchesScalarLCP) {
  const ContactOnlyCaseResult r = run_contact_only_case(/*gap_x=*/1.0, /*radius=*/0.2);
  EXPECT_TRUE(r.converged);
  ASSERT_GT(r.sep0, 0.0) << "test setup should start separated";
  EXPECT_NEAR(r.lambda, 0.0, 1e-6);
  EXPECT_NEAR(r.lambda * r.sep0, 0.0, 1e-6);
}

// ==========================================================================================================
// T9: sizing -- every constraint block (contacts, linear springs, angular springs) is independently
// optional, so B (springs) and/or D (contacts) can each be a genuine zero-column operator, and the
// Schur complement S := (B^T M B + K^{-1})^{-1} can be a genuine 0x0 SPD system. Neither
// mundy::CGInvOp nor the outer PGD solve special-cases this: both must fall out of the general
// zero-extent-view code paths (Kokkos parallel_for/parallel_reduce over an empty range, matrix-free
// operators with domain/range size 0) automatically, converging trivially -- zero iterations, exactly
// zero residual -- rather than iterating needlessly, misreporting non-convergence, or (worse)
// touching out-of-bounds memory.
// ==========================================================================================================

// T9(a): zero springs *and* zero contacts -- the most degenerate case, where the outer PGD's x-block
// is empty (0 contacts) *and* the inner Schur complement's y-block is empty (0 springs)
// simultaneously. With nothing to solve for, vel_omega must come out exactly equal to the
// unconstrained free-motion prediction V_ext + M * F_ext -- no constraint correction whatsoever.
TEST(Mbody, ZeroSpringsZeroContactsMatchesRawMobility) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});
  zero_rod_state(rods);
  rods.force(0) = Vector3d{0.3, -0.2, 0.1};
  rods.torque(0) = Vector3d{0.1, 0.05, -0.1};
  rods.force(1) = Vector3d{-0.2, 0.1, 0.05};
  rods.torque(1) = Vector3d{0.05, -0.1, 0.1};

  const ConstraintSet<HostExecSpace> constraints;

  SolveConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  Kokkos::View<double*, TestMemSpace> vel_omega_expected_d("vel_omega_expected", 6 * rods.size());
  M.apply(rods_d.force_torque_view(), vel_omega_expected_d);
  const auto vel_omega_expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vel_omega_expected_d);

  const PGDResult<double> result = solve_on_device(rods, constraints, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  for (size_t i = 0; i < rods.size(); ++i) {
    const Vector3d vel_expected = rod_velocity(vel_omega_expected, static_cast<int>(i));
    const Vector3d omega_expected = rod_omega(vel_omega_expected, static_cast<int>(i));
    EXPECT_NEAR(norm(rods.velocity(i) - vel_expected), 0.0, 1e-12);
    EXPECT_NEAR(norm(rods.omega(i) - omega_expected), 0.0, 1e-12);
  }
}

// T9(b): angular springs only, zero contacts -- the exact analogue of
// LinearSpringOnlyMatchesScalarSchurComplement above, but for the *other* spring type, verifying that
// path through the same reduced-Schur-complement machinery (this is otherwise only exercised
// structurally, by T1's Jacobian-only AngularSpringJacobianMatchesFiniteDifference, never through a
// full solve()).
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

  ConstraintSet<HostExecSpace> constraints;
  constraints.angular_springs = ang_springs;

  SolveConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  // Independent scalar ground truth for B^T M B, from the same building blocks solve() uses.
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(ang_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_angular_spring_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, ang_springs), b0_d);
  const impl::PairForceOp<TestExecSpace> B(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const auto btmb_op = make_quadratic_form<backend_t>(BT, M, B);

  Kokkos::View<double*, TestMemSpace> ones("ones", 1), btmb_result_d("btmb_result", 1);
  Kokkos::deep_copy(ones, 1.0);
  auto ws = backend_t::make_workspace(btmb_op);
  backend_t::apply(btmb_op, ones, btmb_result_d, ws);
  const auto b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d);
  const auto btmb_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, btmb_result_d);

  const PGDResult<double> result = solve_on_device(rods, constraints, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0(0) / (btmb_result(0) + 1.0 / ang_springs.spring_constant(0));
  EXPECT_NEAR(ang_springs.lambda(0), y_expected, 1e-6);

  // Angular springs contribute equal-and-opposite torques and exactly zero force -- both should
  // cancel exactly between the two rods.
  const Vector3d total_force = rods.force(0) + rods.force(1);
  const Vector3d total_torque = rods.torque(0) + rods.torque(1);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  EXPECT_NEAR(norm(total_torque), 0.0, 1e-9);
}

// Triple-point angular springs only, zero contacts: the same scalar Schur complement as the linear and
// angular springs above, for three spheres bent away from straight at their middle one.
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

  ConstraintSet<HostExecSpace> constraints;
  constraints.triple_springs = triple_springs;

  SolveConfig cfg;
  cfg.dt = 1.0;
  cfg.viscosity = 1.0;

  // Independent scalar ground truth for B^T M B, from the same building blocks solve() uses.
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto b0_d = make_constraint_values(triple_springs);
  const impl::TripleGeometry<TestExecSpace> geo = impl::compute_triple_point_angular_spring_geometry(
      rods_d, create_mirror_view_and_copy(TestExecSpace{}, triple_springs), b0_d);
  const impl::TripleForceOp<TestExecSpace> B(geo, rods.size());
  const impl::TripleForceOpT<TestExecSpace> BT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);
  const auto btmb_op = make_quadratic_form<backend_t>(BT, M, B);

  Kokkos::View<double*, TestMemSpace> ones("ones", 1), btmb_result_d("btmb_result", 1);
  Kokkos::deep_copy(ones, 1.0);
  auto ws = backend_t::make_workspace(btmb_op);
  backend_t::apply(btmb_op, ones, btmb_result_d, ws);
  const auto b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_d);
  const auto btmb_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, btmb_result_d);

  const PGDResult<double> result = solve_on_device(rods, constraints, cfg);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.num_iters, 0u) << "empty (0-dim) contact block should need zero PGD iterations";

  const double y_expected = -b0(0) / (btmb_result(0) + 1.0 / triple_springs.spring_constant(0));
  EXPECT_NEAR(triple_springs.lambda(0), y_expected, 1e-6);

  // The spring is internal and position-only: its three forces cancel exactly and it exerts no torque.
  const Vector3d total_force = rods.force(0) + rods.force(1) + rods.force(2);
  EXPECT_NEAR(norm(total_force), 0.0, 1e-9);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(norm(rods.torque(i)), 0.0, 1e-12) << "rod " << i;
  }
}

// T9(c): zero springs, contacts present (the exact setup ContactOnlyActive/InactiveMatchesScalarLCP
// already exercise for solve()-level correctness) -- made explicit here as a direct, white-box check
// that the Schur complement's own CG, given a literal 0x0 operator (B has zero columns, K^{-1} has
// zero entries), converges immediately rather than iterating, timing out, or misreporting failure.
TEST(Mbody, EmptySpringBlockSchurComplementConvergesInZeroIterations) {
  RodViews<HostExecSpace> rods = make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0},
                                                     Vector3d{0.0, 0.0, 1.6}, Quaterniond{1.0, 0.0, 0.0, 0.0});

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const ConstraintSet<TestExecSpace> empty;

  auto b0_lin = make_constraint_values(empty.linear_springs);
  auto b0_ang = make_constraint_values(empty.angular_springs);
  const impl::PairGeometry<TestExecSpace> lin_geo =
      impl::compute_linear_spring_geometry(rods_d, empty.linear_springs, b0_lin);
  const impl::PairGeometry<TestExecSpace> ang_geo =
      impl::compute_angular_spring_geometry(rods_d, empty.angular_springs, b0_ang);
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

// ==========================================================================================================
// Stage 4 tests: T5 (chain of rods vs. an independent dense linear solve) and T6 (stability /
// consistency sweep).
// ==========================================================================================================

constexpr size_t kChainNumRods = 6;
constexpr size_t kChainNumSprings = 2 * (kChainNumRods - 1);  // linear + angular, one pair each link
constexpr size_t kChainRodSpaceDim = 6 * kChainNumRods;

// Everything one solve() call consumes. Shared by the chain and grid fixtures below.
struct SolveInput {
  RodViews<HostExecSpace> rods;
  ConstraintSet<HostExecSpace> constraints;
  SolveConfig cfg;
};

// A chain rod0-spring-rod1-spring-...-rod(N-1): each rod is twisted a bit relative to its neighbor
// (so adjacent tangents are never parallel, keeping the angular-spring axis well-defined), with a
// small initial stretch/bend offset and nonzero external load on the end rods.
SolveInput make_chain_problem(double spring_constant) {
  SolveInput p;
  p.rods = RodViews<HostExecSpace>(kChainNumRods);

  const double spacing = 1.6;
  for (size_t k = 0; k < kChainNumRods; ++k) {
    p.rods.center(k) = Vector3d{0.0, 0.0, spacing * static_cast<double>(k)};
    p.rods.orientation(k) = axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.15 * static_cast<double>(k));
    p.rods.radius(k) = 0.2;
    p.rods.length(k) = 1.0;
    p.rods.force(k) = Vector3d{0.0, 0.0, 0.0};
    p.rods.torque(k) = Vector3d{0.0, 0.0, 0.0};
    p.rods.velocity(k) = Vector3d{0.0, 0.0, 0.0};
    p.rods.omega(k) = Vector3d{0.0, 0.0, 0.0};
  }
  p.rods.force(0) = Vector3d{0.3, -0.1, 0.0};
  p.rods.torque(kChainNumRods - 1) = Vector3d{0.0, 0.2, -0.1};

  const size_t num_links = kChainNumRods - 1;
  p.constraints.linear_springs = LinearSpringViews<HostExecSpace>(num_links);
  p.constraints.angular_springs = AngularSpringViews<HostExecSpace>(num_links);

  for (size_t k = 0; k < num_links; ++k) {
    p.constraints.linear_springs.rod_i(k) = static_cast<int>(k);
    p.constraints.linear_springs.rod_j(k) = static_cast<int>(k + 1);
    p.constraints.linear_springs.rest_length(k) = spacing - 0.1;  // slight initial stretch
    p.constraints.linear_springs.spring_constant(k) = spring_constant;

    p.constraints.angular_springs.rod_i(k) = static_cast<int>(k);
    p.constraints.angular_springs.rod_j(k) = static_cast<int>(k + 1);
    p.constraints.angular_springs.rest_angle(k) = 0.1;  // actual twist per link is 0.15, so a small initial bend
    p.constraints.angular_springs.spring_constant(k) = spring_constant;
  }

  p.cfg.dt = 0.5;
  p.cfg.viscosity = 1.0;
  p.cfg.max_cg_iters = 500;
  p.cfg.cg_tol = 1e-10;
  p.cfg.max_outer_iters = 1;  // x has zero contacts -> the outer PGD loop has nothing to iterate on
  p.cfg.outer_tol = 1e-10;
  return p;
}

// Sum of 0.5 * k * (stretch or bend-angle)^2 over every spring, computed from rod poses rather than from the solver's
// multipliers.
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

// Materialize any operator's action as a dense (row-major) matrix via repeated unit-vector applies.
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

// Minimal dense linear algebra over row-major std::vector matrices, for the cross-checks whose system
// size is a runtime value rather than a template parameter.
using DenseMat = std::vector<std::vector<double>>;

DenseMat dense_matmul(const DenseMat& A, const DenseMat& B) {
  const size_t m = A.size(), k = A.empty() ? 0 : A[0].size(), n = B.empty() ? 0 : B[0].size();
  DenseMat C(m, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < m; ++i)
    for (size_t p = 0; p < k; ++p) {
      const double a = A[i][p];
      for (size_t j = 0; j < n; ++j) C[i][j] += a * B[p][j];
    }
  return C;
}

DenseMat dense_transpose(const DenseMat& A) {
  const size_t m = A.size(), n = A.empty() ? 0 : A[0].size();
  DenseMat T(n, std::vector<double>(m, 0.0));
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) T[j][i] = A[i][j];
  return T;
}

std::vector<double> dense_matvec(const DenseMat& A, const std::vector<double>& x) {
  std::vector<double> y(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
  return y;
}

// [A B]: B's columns appended to A's. Both have the same number of rows.
DenseMat dense_hcat(const DenseMat& A, const DenseMat& B) {
  DenseMat C = A;
  for (size_t i = 0; i < C.size(); ++i) C[i].insert(C[i].end(), B[i].begin(), B[i].end());
  return C;
}

// Solve A x = b by Gaussian elimination with partial pivoting (A is square, copied and destroyed).
std::vector<double> dense_solve(DenseMat A, std::vector<double> b) {
  const size_t n = A.size();
  for (size_t col = 0; col < n; ++col) {
    size_t piv = col;
    for (size_t r = col + 1; r < n; ++r)
      if (std::abs(A[r][col]) > std::abs(A[piv][col])) piv = r;
    std::swap(A[col], A[piv]);
    std::swap(b[col], b[piv]);
    const double d = A[col][col];
    for (size_t r = 0; r < n; ++r) {
      if (r == col) continue;
      const double f = A[r][col] / d;
      for (size_t c = col; c < n; ++c) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n, 0.0);
  for (size_t i = 0; i < n; ++i) x[i] = b[i] / A[i][i];
  return x;
}

// T5: cross-check mundy::mbody::solve() on a multi-spring chain against an independent dense
// linear solve. B and M are materialized (via materialize_dense, above) from the same
// impl::PairForceOp / impl::LocalDragMobilityOp building blocks solve() itself uses -- this is the
// same "reuse the already finite-difference-validated (T1) operators, but solve the resulting
// linear system by a completely different, non-iterative algorithm" strategy as T3/T4, just at
// multi-spring scale. What's newly exercised here, beyond T3/T4, is (a) concatenating linear+
// angular springs into one consistent y-block index space at N>1, and (b) mundy::CGInvOp/
// conjugate_gradient's iterative accuracy on a genuinely multi-dimensional (kChainNumSprings-dof),
// coupled SPD system.
TEST(Mbody, ChainMatchesIndependentDenseSolve) {
  SolveInput p = make_chain_problem(/*spring_constant=*/3.0);

  // Independent dense reference, built from the rods/springs BEFORE solve() mutates them.
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
  auto b0_lin = make_constraint_values(constraints_d.linear_springs);
  auto b0_ang = make_constraint_values(constraints_d.angular_springs);
  const impl::PairGeometry<TestExecSpace> lin_geo =
      impl::compute_linear_spring_geometry(rods_d, constraints_d.linear_springs, b0_lin);
  const impl::PairGeometry<TestExecSpace> ang_geo =
      impl::compute_angular_spring_geometry(rods_d, constraints_d.angular_springs, b0_ang);
  const impl::PairGeometry<TestExecSpace> spring_geo = impl::concat_pair_geometry(lin_geo, ang_geo);
  const auto b0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, impl::concat_vectors(b0_lin, b0_ang));

  const impl::PairForceOp<TestExecSpace> B(spring_geo, kChainNumRods);
  const impl::PairForceOpT<TestExecSpace> BT(spring_geo, kChainNumRods);
  const impl::LocalDragMobilityOp<TestExecSpace> M(p.cfg.viscosity, rods_d);

  const std::vector<std::vector<double>> B_dense = materialize_dense(B);  // kChainRodSpaceDim x kChainNumSprings
  const std::vector<std::vector<double>> M_dense = materialize_dense(M);  // kChainRodSpaceDim x kChainRodSpaceDim

  // b = b0 + dt * B^T (V_ext + M F_ext); V_ext = 0 here, F_ext read from p.rods.
  Kokkos::View<double*, Kokkos::HostSpace> force_torque_ext("force_torque_ext", kChainRodSpaceDim);
  for (size_t i = 0; i < kChainNumRods; ++i) {
    rod_force(force_torque_ext, static_cast<int>(i)) = p.rods.force(i);
    rod_torque(force_torque_ext, static_cast<int>(i)) = p.rods.torque(i);
  }
  Kokkos::View<double*, TestMemSpace> m_force_torque_ext("m_force_torque_ext", kChainRodSpaceDim);
  M.apply(Kokkos::create_mirror_view_and_copy(TestMemSpace{}, force_torque_ext), m_force_torque_ext);
  Kokkos::View<double*, TestMemSpace> b_rate_d("b_rate", kChainNumSprings);
  BT.apply(m_force_torque_ext, b_rate_d);
  const auto b_rate = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b_rate_d);

  Vector<double, kChainNumSprings> b_vec;
  for (size_t i = 0; i < kChainNumSprings; ++i) {
    b_vec[i] = b0(i) + p.cfg.dt * b_rate(i);
  }

  // (dt * B^T M B + Kinv), assembled by plain triple loops -- no KokkosBlas, no mundy::CGInvOp. The
  // Schur complement's "M" is dt * mobility (it maps a constraint force to the displacement it causes
  // over the step, not to a velocity -- see solve()'s M_dt), so M_dense (the raw, instantaneous
  // force->velocity mobility) needs that same dt scaling applied here to match.
  Matrix<double, kChainNumSprings, kChainNumSprings> btmb_plus_kinv =
      Matrix<double, kChainNumSprings, kChainNumSprings>::zeros();
  for (size_t i = 0; i < kChainNumSprings; ++i) {
    for (size_t j = 0; j < kChainNumSprings; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < kChainRodSpaceDim; ++k) {
        double mb_kj = 0.0;
        for (size_t l = 0; l < kChainRodSpaceDim; ++l) {
          mb_kj += M_dense[k][l] * B_dense[l][j];
        }
        sum += B_dense[k][i] * mb_kj;
      }
      btmb_plus_kinv(i, j) = p.cfg.dt * sum;
    }
  }
  for (size_t i = 0; i < kChainNumSprings / 2; ++i) {
    btmb_plus_kinv(i, i) += 1.0 / p.constraints.linear_springs.spring_constant(i);
  }
  for (size_t i = kChainNumSprings / 2; i < kChainNumSprings; ++i) {
    btmb_plus_kinv(i, i) += 1.0 / p.constraints.angular_springs.spring_constant(i - kChainNumSprings / 2);
  }

  const Vector<double, kChainNumSprings> y_expected = -1.0 * (inverse(btmb_plus_kinv) * b_vec);

  const PGDResult<double> result = solve_on_device(p.rods, p.constraints, p.cfg);
  EXPECT_TRUE(result.converged);

  const size_t num_links = kChainNumRods - 1;
  for (size_t i = 0; i < num_links; ++i) {
    EXPECT_NEAR(p.constraints.linear_springs.lambda(i), y_expected[i], 1e-4);
  }
  for (size_t i = 0; i < num_links; ++i) {
    EXPECT_NEAR(p.constraints.angular_springs.lambda(i), y_expected[num_links + i], 1e-4);
  }
}

// T6(a): a much stiffer chain (approaching a rigid bilateral limit) still converges, with the
// inner CG's iteration count staying well within budget.
TEST(Mbody, StiffChainStaysStable) {
  SolveInput p = make_chain_problem(/*spring_constant=*/1.0e6);
  const PGDResult<double> result = solve_on_device(p.rods, p.constraints, p.cfg);
  EXPECT_TRUE(result.converged);
  for (size_t i = 0; i < p.constraints.linear_springs.size(); ++i) {
    EXPECT_TRUE(std::isfinite(p.constraints.linear_springs.lambda(i)));
  }
  for (size_t i = 0; i < p.constraints.angular_springs.size(); ++i) {
    EXPECT_TRUE(std::isfinite(p.constraints.angular_springs.lambda(i)));
  }
}

// lambda_max(K B^T M B) over the springs of p: explicit Euler on the linearized spring network is stable only for
// dt < 2 / lambda_max. Evaluated matrix-free as the symmetric K^1/2 B^T M B K^1/2, which shares its spectrum.
double spring_network_stiffness(const SolveInput& p) {
  using backend_t = KokkosBackend<TestExecSpace>;
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
  auto b0_lin = make_constraint_values(constraints_d.linear_springs);
  auto b0_ang = make_constraint_values(constraints_d.angular_springs);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::concat_pair_geometry(impl::compute_linear_spring_geometry(rods_d, constraints_d.linear_springs, b0_lin),
                                 impl::compute_angular_spring_geometry(rods_d, constraints_d.angular_springs, b0_ang));
  const impl::PairForceOp<TestExecSpace> B(geo, p.rods.size());
  const impl::PairForceOpT<TestExecSpace> BT(geo, p.rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(p.cfg.viscosity, rods_d);

  // Spring rows are packed linear then angular, matching the concatenated geometry.
  const size_t num_linear = p.constraints.linear_springs.size();
  const size_t num_springs = geo.size();
  Kokkos::View<double*, Kokkos::HostSpace> sqrt_k("sqrt_k", num_springs), q0("q0", num_springs);
  std::mt19937 rng(20261001);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t row = 0; row < num_springs; ++row) {
    sqrt_k(row) = std::sqrt(row < num_linear ? p.constraints.linear_springs.spring_constant(row)
                                             : p.constraints.angular_springs.spring_constant(row - num_linear));
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

// T6(b): backward Euler is stable at any step size. Forward Euler on the same linearized spring network is stable only
// for dt < dt_crit = 2 / lambda_max(K B^T M B), so the sweep crosses dt / dt_crit = 1, and on both sides of it every
// implicit step must lower the elastic energy: the step maps K^1/2 b to (I + dt K^1/2 B^T M B K^1/2)^-1 K^1/2 b.
TEST(Mbody, ChainStableAcrossExplicitStabilityLimit) {
  const double dt_crit = 2.0 / spring_network_stiffness(make_chain_problem(/*spring_constant=*/3.0));

  for (const double cfl : {0.5, 0.9, 1.1, 2.0, 10.0, 100.0}) {
    SolveInput p = make_chain_problem(/*spring_constant=*/3.0);
    p.cfg.dt = cfl * dt_crit;
    p.cfg.cg_tol = 1e-13;

    zero_rod_state(p.rods);  // no external load: the springs relax toward rest
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
    const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
    const auto load_d = copy_load(rods_d);

    std::vector<double> energy{elastic_energy(rods_d, constraints_d.linear_springs, constraints_d.angular_springs)};
    for (int step = 0; step < 40; ++step) {
      ASSERT_TRUE(step_rods(rods_d, constraints_d, p.cfg, load_d).converged) << "cfl=" << cfl << " step " << step;
      energy.push_back(elastic_energy(rods_d, constraints_d.linear_springs, constraints_d.angular_springs));
    }

    // Round-off floor relative to the initial energy, which the largest steps decay toward.
    const double floor = 1e-10 * energy.front();
    for (size_t n = 0; n + 1 < energy.size(); ++n) {
      EXPECT_LE(energy[n + 1], energy[n] + floor) << "cfl=" << cfl << ": energy rose at step " << n + 1;
    }
    EXPECT_LT(energy.back(), energy.front()) << "cfl=" << cfl;
  }
}

// T6(c): the outer PGD solve converges to the same answer regardless of a deliberately bad initial
// guess -- exercised directly at the solve_mixed_cqpp level (bypassing mundy::mbody::solve()'s
// always-zero-init convention) on the T4 active-contact case.
TEST(Mbody, BadInitialGuessStillConvergesToSameAnswer) {
  const double gap_x = 0.3;
  const double radius = 0.2;
  RodViews<HostExecSpace> rods =
      make_two_rod_system(Vector3d{0.0, 0.0, 0.0}, Quaterniond{1.0, 0.0, 0.0, 0.0}, Vector3d{gap_x, 0.0, 0.0},
                          Quaterniond{1.0, 0.0, 0.0, 0.0}, radius, 1.0);
  zero_rod_state(rods);

  ContactViews<HostExecSpace> contacts(1);
  contacts.rod_i(0) = 0;
  contacts.rod_j(0) = 1;

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  auto sep0 = make_constraint_values(contacts);
  const impl::PairGeometry<TestExecSpace> geo =
      impl::compute_contact_geometry(rods_d, create_mirror_view_and_copy(TestExecSpace{}, contacts), sep0);
  const impl::PairForceOp<TestExecSpace> D(geo, rods.size());
  const impl::PairForceOpT<TestExecSpace> DT(geo, rods.size());
  const impl::LocalDragMobilityOp<TestExecSpace> M(1.0, rods_d);

  using backend_t = KokkosBackend<TestExecSpace>;
  const auto lcp = make_lcp<backend_t>(DT, M, D, sep0);
  const PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-10};
  const auto pgd = make_pgd_solution_strategy(cfg);

  Kokkos::View<double*, TestMemSpace> x_zero("x_zero", 1), grad0("grad0", 1), x_tmp0("x_tmp0", 1),
      grad_tmp0("grad_tmp0", 1);
  Kokkos::deep_copy(x_zero, 0.0);
  auto state_zero = make_pgd_state(x_zero, grad0, x_tmp0, grad_tmp0);
  const PGDResult<double> result_zero = solve_lcp(lcp, pgd, state_zero);

  Kokkos::View<double*, TestMemSpace> x_bad("x_bad", 1), grad1("grad1", 1), x_tmp1("x_tmp1", 1),
      grad_tmp1("grad_tmp1", 1);
  Kokkos::deep_copy(x_bad, 99.99);
  auto state_bad = make_pgd_state(x_bad, grad1, x_tmp1, grad_tmp1);
  const PGDResult<double> result_bad = solve_lcp(lcp, pgd, state_bad);

  EXPECT_TRUE(result_zero.converged);
  EXPECT_TRUE(result_bad.converged);
  const double x_zero_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_zero)(0);
  const double x_bad_value = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_bad)(0);
  EXPECT_NEAR(x_zero_value, x_bad_value, 1e-6);
  EXPECT_GT(x_zero_value, 0.0);  // sanity: this is the active (overlapping) branch
}

// ==========================================================================================================
// Stage 5 tests: T7 -- multi-step time integration. T1-T6 all call solve() exactly once, so none of
// them exercise repeated time-stepping. These step the scalar spring system and check the trajectory
// against the exact backward-Euler discretization -- the dt-consistency (the Schur-complement "M" is
// dt * mobility) that a single-step test cannot see.
//
// Both tests reduce to the same scalar system as T3 (two rods along z, one linear spring, zero
// contacts/angular springs/external load), so the analytical reference is exact, not approximate: with
// A := B^T M B (M the raw, instantaneous mobility) and tau := 1/(A*k), backward Euler on the resulting
// scalar ODE ds/dt = -A*k*s gives the closed form s_n = s0 / (1 + dt/tau)^n. A is computed here directly
// from the slender-body drag formula LocalDragMobilityOp::apply implements, reproduced independently
// rather than by constructing/calling the operator, so this is a check against known mechanics, not a
// tautology.
// ==========================================================================================================

double expected_inv_drag_para(double radius, double length, double viscosity) {
  const double lprime = length + 2.0 * radius;
  const double p = lprime / (2.0 * radius);
  const double log_p = std::log(p);
  const double inv_p = 1.0 / p;
  const double inv_p2 = inv_p * inv_p;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  return (log_p - 0.207 + 0.98 * inv_p - 0.133 * inv_p2) / lprime / (2.0 * pi * viscosity);
}

// Sum over the linear springs of (center distance - rest length).
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

// Two rods along z (T3's setup), one linear spring, no contacts/angular springs/external load. Takes
// `num_steps` backward-Euler steps and returns the stretch after each, with index 0 the initial
// (pre-stepping) stretch.
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

  ConstraintSet<HostExecSpace> constraints;
  constraints.linear_springs = lin_springs;

  SolveConfig cfg;
  cfg.dt = dt;
  cfg.viscosity = viscosity;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.max_outer_iters = 1;  // no contacts -> nothing for the outer PGD loop to do

  zero_rod_state(rods);
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);

  std::vector<double> stretch{summed_stretch(rods_d, constraints_d.linear_springs)};
  for (int step = 0; step < num_steps; ++step) {
    EXPECT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged) << "step " << step;
    stretch.push_back(summed_stretch(rods_d, constraints_d.linear_springs));
  }
  return stretch;
}

// T7(a): stepping the scalar spring system reproduces the exact backward-Euler discretization of
// ds/dt = -A*k*s, s_n = s0/(1+dt/tau)^n -- the precise closed form solve() is supposed to implement,
// not just "decays a plausible amount" -- and, since dt/tau is modest here, is also close to the
// continuous solution s0*exp(-t/tau), the physical relaxation this discretization approximates.
TEST(Mbody, SpringRelaxationMatchesBackwardEulerAndExponentialDecay) {
  const double radius = 0.2, length = 1.0, viscosity = 1.0, spring_constant = 2.0;
  const double A = 2.0 * expected_inv_drag_para(radius, length, viscosity);  // both rods move; T3's setup
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

// T7(b): backward Euler is unconditionally A-stable -- bounded and monotonically decaying for *any*
// dt > 0, however large relative to the system's own relaxation time tau. Sweeps dt from a small
// fraction of tau up to 50 tau and checks both that the exact backward-Euler formula holds at every
// step and that the trajectory never oscillates or grows.
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

// ==========================================================================================================
// Stage 6 test: T8 -- full-stack integration test. A 2D grid of spheres (zero-length rods), linear
// springs on every row/column/diagonal edge (full triangulation, so the lattice can't shear/rack
// without stretching a spring), angular (bending) springs along rows only. Angular springs constrain a
// single per-rod tangent axis (orientation * e_z), so a genuine 2D grid can't give both row- and
// column-neighbors an independent bending constraint without a second constrained axis that doesn't
// exist yet -- rows get real bending, columns are stretch-only. A random kick pushes many pairs into
// overlap; contacts (every pair within a generous cutoff of the post-kick configuration) have to
// resolve every one of them, every step, while the spring network relaxes -- the only test here that
// runs contacts and springs together at more than a handful of rods.
// ==========================================================================================================

size_t grid_index(size_t row, size_t col, size_t num_cols) {
  return row * num_cols + col;
}

// Builds a num_rows x num_cols grid of spheres (zero-length rods) at the given spacing, fully
// triangulated with linear springs (row/column/both diagonals), angular (bending) springs along rows
// only, and a contact for every pair within contact_cutoff of each other. Every rod starts with the
// same orientation (tangent along the row/+x axis), so row-neighbor rest angles are all zero. A
// uniform random kick in [-kick_magnitude, kick_magnitude]^3 is applied to every center *before* the
// contact list is built (so contacts reflect the actual post-kick configuration, not the pristine
// grid) -- the rest lengths/angles above are still the perfect grid's, so the kick alone is what gets
// relaxed away.
SolveInput make_grid_problem(size_t num_rows, size_t num_cols, double spacing, double radius, double spring_constant,
                             double kick_magnitude, unsigned seed) {
  SolveInput p;
  const size_t num_spheres = num_rows * num_cols;
  p.rods = RodViews<HostExecSpace>(num_spheres);

  const Quaterniond row_orientation =
      axis_angle_to_quaternion(Vector3d{0.0, 1.0, 0.0}, 0.5 * Kokkos::numbers::pi_v<double>);
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col < num_cols; ++col) {
      const size_t idx = grid_index(row, col, num_cols);
      p.rods.center(idx) = Vector3d{static_cast<double>(col) * spacing, static_cast<double>(row) * spacing, 0.0};
      p.rods.orientation(idx) = row_orientation;
      p.rods.radius(idx) = radius;
      p.rods.length(idx) = 0.0;
    }
  }
  zero_rod_state(p.rods);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> kick(-kick_magnitude, kick_magnitude);
  for (size_t idx = 0; idx < num_spheres; ++idx) {
    p.rods.center(idx) = p.rods.center(idx) + Vector3d{kick(rng), kick(rng), kick(rng)};
  }

  // Linear springs: every row edge, column edge, and both diagonals of every unit cell.
  std::vector<std::pair<size_t, size_t>> lin_pairs;
  std::vector<double> lin_rest_lengths;
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col < num_cols; ++col) {
      const size_t idx = grid_index(row, col, num_cols);
      if (col + 1 < num_cols) {
        lin_pairs.push_back({idx, grid_index(row, col + 1, num_cols)});
        lin_rest_lengths.push_back(spacing);
      }
      if (row + 1 < num_rows) {
        lin_pairs.push_back({idx, grid_index(row + 1, col, num_cols)});
        lin_rest_lengths.push_back(spacing);
      }
      if (row + 1 < num_rows && col + 1 < num_cols) {
        lin_pairs.push_back({idx, grid_index(row + 1, col + 1, num_cols)});
        lin_rest_lengths.push_back(spacing * std::sqrt(2.0));
        lin_pairs.push_back({grid_index(row, col + 1, num_cols), grid_index(row + 1, col, num_cols)});
        lin_rest_lengths.push_back(spacing * std::sqrt(2.0));
      }
    }
  }
  p.constraints.linear_springs = LinearSpringViews<HostExecSpace>(lin_pairs.size());
  for (size_t k = 0; k < lin_pairs.size(); ++k) {
    p.constraints.linear_springs.rod_i(k) = static_cast<int>(lin_pairs[k].first);
    p.constraints.linear_springs.rod_j(k) = static_cast<int>(lin_pairs[k].second);
    p.constraints.linear_springs.rest_length(k) = lin_rest_lengths[k];
    p.constraints.linear_springs.spring_constant(k) = spring_constant;
  }

  // Angular (bending) springs: row edges only -- see the section comment for why.
  std::vector<std::pair<size_t, size_t>> ang_pairs;
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col + 1 < num_cols; ++col) {
      ang_pairs.push_back({grid_index(row, col, num_cols), grid_index(row, col + 1, num_cols)});
    }
  }
  p.constraints.angular_springs = AngularSpringViews<HostExecSpace>(ang_pairs.size());
  for (size_t k = 0; k < ang_pairs.size(); ++k) {
    p.constraints.angular_springs.rod_i(k) = static_cast<int>(ang_pairs[k].first);
    p.constraints.angular_springs.rod_j(k) = static_cast<int>(ang_pairs[k].second);
    p.constraints.angular_springs.rest_angle(k) = 0.0;  // every rod starts with the same orientation
    p.constraints.angular_springs.spring_constant(k) = spring_constant;
  }

  // Contacts: every pair within a generous cutoff of the post-kick configuration. The test's own
  // overlap check (max_overlap, below) verifies *every* pair regardless of this list, so an
  // insufficient cutoff here would show up as a loud failure, not a silent miss.
  const double contact_cutoff = 1.8 * spacing;
  std::vector<std::pair<size_t, size_t>> contact_pairs;
  for (size_t i = 0; i < num_spheres; ++i) {
    for (size_t j = i + 1; j < num_spheres; ++j) {
      if (norm(p.rods.center(j) - p.rods.center(i)) < contact_cutoff) {
        contact_pairs.push_back({i, j});
      }
    }
  }
  p.constraints.contacts = ContactViews<HostExecSpace>(contact_pairs.size());
  for (size_t k = 0; k < contact_pairs.size(); ++k) {
    p.constraints.contacts.rod_i(k) = static_cast<int>(contact_pairs[k].first);
    p.constraints.contacts.rod_j(k) = static_cast<int>(contact_pairs[k].second);
  }
  p.cfg.dt = 0.1;
  p.cfg.viscosity = 1.0;
  p.cfg.max_cg_iters = 300;
  p.cfg.cg_tol = 1e-6;
  p.cfg.max_outer_iters = 500;
  p.cfg.outer_tol = 1e-6;
  return p;
}

// Brute-force over every pair, independent of whatever contact list the solver was given: positive
// if any two spheres overlap (by that much), non-positive if none do.
template <typename Space>
double max_overlap(const RodViews<Space>& rods) {
  const int num_rods = static_cast<int>(rods.size());
  double worst_gap = 0.0;
  Kokkos::parallel_reduce(
      "max_overlap", Kokkos::RangePolicy<Space>(0, num_rods * num_rods),
      KOKKOS_LAMBDA(const int pair, double& min_gap) {
        const int i = pair / num_rods;
        const int j = pair % num_rods;
        if (i < j) {
          const double gap = norm(rods.center(j) - rods.center(i)) - rods.radius(i) - rods.radius(j);
          min_gap = Kokkos::min(min_gap, gap);
        }
      },
      Kokkos::Min<double>(worst_gap));
  return -worst_gap;
}

TEST(Mbody, GridOfSpheresStaysOverlapFreeAndRelaxes) {
  SolveInput p = make_grid_problem(/*num_rows=*/4, /*num_cols=*/4, /*spacing=*/1.0, /*radius=*/0.35,
                                   /*spring_constant=*/3.0, /*kick_magnitude=*/0.35, /*seed=*/1234);

  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, p.rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, p.constraints);
  const auto load_d = copy_load(rods_d);
  ASSERT_GT(max_overlap(rods_d), 0.0) << "test setup should start with a real overlap somewhere";

  const int num_steps = 80;
  std::vector<double> energy_trace;
  energy_trace.push_back(elastic_energy(rods_d, constraints_d.linear_springs, constraints_d.angular_springs));

  for (int step = 0; step < num_steps; ++step) {
    ASSERT_TRUE(step_rods(rods_d, constraints_d, p.cfg, load_d).converged) << "step " << step;
    ASSERT_LE(max_overlap(rods_d), 1e-4) << "overlap at step " << step;
    energy_trace.push_back(elastic_energy(rods_d, constraints_d.linear_springs, constraints_d.angular_springs));
  }

  // Eventually relaxes: elastic energy shortly after the kick vs. at the end should have dropped
  // substantially, not just be bouncing around the same scale.
  const double energy_after_kick = energy_trace[1];
  const double energy_final = energy_trace.back();
  EXPECT_LT(energy_final, 0.1 * energy_after_kick)
      << "energy_after_kick=" << energy_after_kick << " energy_final=" << energy_final;

  // Stays stable throughout, not just at the end: energy should never blow up past its initial scale.
  const double energy_ceiling = 2.0 * energy_trace[0];
  for (size_t n = 0; n < energy_trace.size(); ++n) {
    EXPECT_LT(energy_trace[n], energy_ceiling) << "energy diverged at step " << n;
  }
}

// ==========================================================================================================
// T10: chain of spheres -- axial stiffness (Young's modulus) and cantilever bending (Euler-Bernoulli).
//
// T5/T6/T8 already exercise a chain/grid of springs structurally (Jacobians, dense-solve cross-check,
// stability under a kick); this section instead checks the chain's *aggregate mechanical response*
// against two textbook continuum predictions, closing the physics-analysis loop the same way T7 did
// for single-spring relaxation.
//
// T10(a) uses linear springs alone (one per adjacent pair, as in the chain and grid fixtures above);
// T10(b) adds TriplePointAngularSpringViews for bending -- one per *interior* sphere (vertex = that
// sphere, outer points = its two neighbors), NOT the pairwise/orientation-based AngularSpringViews
// (see TriplePointAngularSpringViews's own doc comment for why: that type's bend is a function of two
// rods' independent orientation DOFs, which has no first-order resistance to a whole sub-chain
// translating sideways relative to its neighbors -- a real shear zero-mode, fine for T5/T6/T8's
// qualitative checks but wrong for a quantitative bending profile).
//
// Both discrete-to-continuum correspondences (EA := k_lin*spacing, EI := k_ang*spacing) are derived
// the same way, symmetrically: a single-element force/moment balance, not a chain-wide or
// energy-density argument (which can hide boundary-dependent effects). One spring under tension F
// stretches by F/k_lin; strain over its length `a` is F/(k_lin*a), matching continuum strain F/(EA)
// gives EA=k_lin*a. One joint under an applied moment M rotates (from straight) by M/k_ang; that
// same rotation is, geometrically, the turning angle of a curve sampled at spacing `a`, i.e.
// approximately kappa*a for curvature kappa -- so kappa*a = M/k_ang, matching the continuum
// constitutive relation M=EI*kappa gives EI=k_ang*a. This is exactly the classical Hencky bar-chain
// model's correspondence (rigid segments joined by rotational springs of stiffness EI/a) for bending,
// and the standard "springs in series" result for axial. Both are per-element statements, independent
// of how many elements make up a particular chain or how its ends are handled.
//
// What *is* chain- (and boundary-) dependent is how well a finite chain of such elements approximates
// the continuum solution of a *specific* boundary value problem -- e.g. the textbook cantilever
// deflection y(x) = F*(3*L*x^2 - x^3)/(6*EI), tip deflection y(L) = F*L^3/(3*EI). T10(a) shows the
// axial correspondence is exact at *any* segment count (no curvature/higher-derivative structure to
// discretize); T10(b) shows the bending correspondence reaching the continuum formula only in the
// segment-count limit, and reaching the discrete chain's own exact deflection at every resolution
// along the way -- a finite-difference error, not a wrong per-element coefficient.
// ==========================================================================================================

// Runs the axial-only chain (N=num_segments+1 spheres, physical length L fixed, spacing = L/N
// shrinking as N grows) to static equilibrium under an equal-and-opposite pulling force at the two
// ends, and returns the measured effective axial stiffness EA_measured := F*L/dL. See T10(a) below
// for why this is expected to be *exact* at every resolution, not just in a many-segment limit.
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
  ConstraintSet<HostExecSpace> constraints;
  constraints.linear_springs = lin_springs;

  SolveConfig cfg;
  cfg.dt = 2.0;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.max_outer_iters = 1;  // no contacts -> nothing for the outer PGD loop to do

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

// T10(a): sweep the number of segments (fixed physical length L, fixed spring constant k_lin, so
// spacing = L/N shrinks as N grows) and check the measured EA matches EA_expected := k_lin*spacing at
// *every* resolution tested, not just asymptotically: unlike bending (T10(b)), an axial chain has no
// curvature/higher-derivative structure -- every segment sees exactly the same tension F (force
// balance on each interior sphere, regardless of segment count), so "springs in series" is exact at
// any N. This is the axial analogue of T10(b)'s convergence sweep, checking the SAME kind of claim
// (does the discrete spring constant reproduce the continuum modulus) the same way, for symmetry.
TEST(Mbody, ChainAxialStiffnessConvergesWithSegmentCount) {
  const double L = 8.0, k_lin = 3.0, tip_force = 0.3;
  for (const size_t num_segments : {2, 4, 8, 16}) {
    const double spacing = L / static_cast<double>(num_segments);
    const double EA_expected = k_lin * spacing;
    const double EA_measured = run_axial_chain_EA(num_segments, L, k_lin, tip_force);
    EXPECT_NEAR(EA_measured, EA_expected, 1e-3 * EA_expected) << "num_segments=" << num_segments;
  }
}

// T10(b): a chain of triple-point angular springs (bending), clamped at one end, plus a transverse
// point force at the free end -- the classic Euler-Bernoulli cantilever setup. Bending here uses
// TriplePointAngularSpringViews, not the pairwise/orientation-based AngularSpringViews: one spring per
// *interior* sphere (vertex = that sphere, outer points = its two immediate neighbors), matching the
// standard bead-chain/discrete-elastic-rod bending convention (a sliding window of 3 consecutive
// nodes) -- see compute_triple_point_angular_spring_geometry.
//
// A "clamped" (zero displacement *and* zero slope) boundary needs *two* held spheres, not one: sphere
// 1 fixes the wall position and sphere 0 fixes the wall tangent, as the direction from sphere 0 to
// sphere 1.
//
// Unlike T10(a), a bending chain does not match the continuum Euler-Bernoulli formula at finite
// resolution. The per-spring force law (checked against finite differences in
// TriplePointAngularSpringJacobianMatchesFiniteDifference) and the EI=k_ang*spacing correspondence
// (re-derived via a single-joint moment-curvature balance, the same way T10(a)'s EA=k_lin*spacing is
// derived via a single-spring force balance) are exact statements about *one* spring. A chain of them
// solves its own discrete problem, whose tip deflection is exactly F L^3 (N+1)(2N+1) / (6 N^2 EI) and
// approaches the continuum value at first order in the spacing.

// A straight chain of spheres along z with a bend spring at each interior vertex, and nothing else:
// callers that relax it add their own linear springs and anchors.
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

// The chain's static equilibrium for a given set of free spheres, solved directly with no time
// stepping: linearize the bend constraint, giving b0(u) = J u with
// J(p, col) := d(theta_p)/d(y of sphere first_free + col), then solve k_ang J^T J u = f_ext for the
// free spheres' transverse displacements. Dense std::vector algebra rather than a fixed-size Matrix
// keeps the segment count a runtime value, which the refinement studies need out past N=100.
//
// A bend angle has no gradient at the straight rest configuration: along any transverse ray it
// behaves like |delta|, so it has a directional derivative in every direction but no single linear
// map. Linearizing therefore has to happen slightly off straight, which is what pre_bend is for --
// it is an artifact of differentiating this constraint, not a property of the chain, so it lives
// here rather than in make_bend_chain. A constant-curvature arc keeps every vertex's angle
// resolvable, the end ones included, and the result does not depend on the amplitude, which
// BendLinearizationIsIndependentOfPreBend checks.
//
// Restricting to each free sphere's y-component alone is exact, not an approximation: with the arc in
// the y-z plane the bend axis is exactly the x-axis, so the three axes decouple.
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
  const impl::TripleGeometry<TestExecSpace> geo = impl::compute_triple_point_angular_spring_geometry(
      create_mirror_view_and_copy(TestExecSpace{}, chain.rods),
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

// Cantilever: spheres 0 and 1 are the wall, 2..N+1 are free, and the load is at the free end.
double run_static_bending_tip_dynamic(size_t num_segments, double L, double EI, double tip_force) {
  const double spacing = L / static_cast<double>(num_segments);
  std::vector<double> f_ext(num_segments, 0.0);
  f_ext.back() = tip_force;
  return solve_static_bend(num_segments + 2, spacing, EI / spacing, 2, f_ext).back();
}

// Simply supported: spheres 0 and N are the supports, 1..N-1 are free, and the load is at midspan.
// num_segments must be even so midspan lands on a sphere.
double run_static_simply_supported_midspan(size_t num_segments, double L, double EI, double load) {
  const double spacing = L / static_cast<double>(num_segments);
  const size_t midspan = num_segments / 2 - 1;  // free spheres are 1..N-1
  std::vector<double> f_ext(num_segments - 1, 0.0);
  f_ext[midspan] = load;
  return solve_static_bend(num_segments + 1, spacing, EI / spacing, 1, f_ext)[midspan];
}

// T10(b): the discrete cantilever has an exact closed form, so this does not have to settle for
// "the error shrinks". Summing each joint's rotation M_j/k_ang against its lever arm, with
// k_ang = EI/spacing, a Hencky bar chain of N segments under a tip load F deflects
//
//   tip_N = F L^3 (N+1)(2N+1) / (6 N^2 EI)
//
// whose relative departure from Euler-Bernoulli's F L^3 / (3 EI) is exactly (3N+1)/(2N^2). The
// scheme is therefore first order in the spacing with error constant 3/2, both derived rather than
// fitted, and the static equilibrium must reproduce the formula at every resolution, not just
// approach it.
TEST(Mbody, ChainBendingMatchesHenckyBarChain) {
  const double L = 8.0, EI = 5.0, tip_force = 0.005;
  const double continuum = tip_force * L * L * L / (3.0 * EI);

  double finest_scaled_error = 0.0;
  for (const size_t num_segments : {4, 7, 11, 16, 25, 50, 100, 200}) {
    const double n = static_cast<double>(num_segments);
    const double hencky = tip_force * L * L * L * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n * EI);
    const double tip = run_static_bending_tip_dynamic(num_segments, L, EI, tip_force);

    EXPECT_NEAR(tip, hencky, 1e-6 * hencky) << "num_segments=" << num_segments;
    finest_scaled_error = n * std::abs(tip - continuum) / continuum;
  }

  // Read as a convergence rate, N * rel_err = (3N+1)/(2N), which falls monotonically to 3/2.
  EXPECT_NEAR(finest_scaled_error, 1.5, 0.005);
}

// The only exercise of the full five-family packing order, with every family non-empty at a distinct
// prime size so any transposition moves an offset. Pins the row multipliers num_constraints() applies
// as well: three rows per fixed position, six per fixed pose.
TEST(Mbody, ConstraintIndexMapPacksEveryFamily) {
  ConstraintSet<HostExecSpace> constraints;
  constraints.linear_springs = LinearSpringViews<HostExecSpace>(2);
  constraints.angular_springs = AngularSpringViews<HostExecSpace>(3);
  constraints.triple_springs = TriplePointAngularSpringViews<HostExecSpace>(5);
  constraints.fixed_positions = FixedPositionViews<HostExecSpace>(7);
  constraints.fixed_poses = FixedPoseViews<HostExecSpace>(11);

  const ConstraintIndexMap index_map = make_constraint_index_map(constraints);

  EXPECT_EQ(index_map.linear_springs.begin, 0u);
  EXPECT_EQ(index_map.linear_springs.size(), 2u);
  EXPECT_EQ(index_map.angular_springs.begin, 2u);
  EXPECT_EQ(index_map.angular_springs.size(), 3u);
  EXPECT_EQ(index_map.triple_springs.begin, 5u);
  EXPECT_EQ(index_map.triple_springs.size(), 5u);
  EXPECT_EQ(index_map.fixed_positions.begin, 10u);
  EXPECT_EQ(index_map.fixed_positions.size(), 21u);
  EXPECT_EQ(index_map.fixed_poses.begin, 31u);
  EXPECT_EQ(index_map.fixed_poses.size(), 66u);
  EXPECT_EQ(index_map.total, 97u);
}

// Soft anchors. At a fixed point every body has v = 0, so F_ext + B y = 0 and K^-1 y = -b0, and a centre
// anchor's B is the identity on its force rows. Started at its target, each anchored rod therefore settles
// at an offset of exactly compliance * load, componentwise, with no dependence on dt.
//
// The orientation rows are exact only for a torque about a single axis: the rod then turns about that
// axis, so the rotation vector is parallel to the torque and the inverse left Jacobian acts as the
// identity on it. Hence the loop over axes rather than one general torque.
//
// Both single-arity families are non-empty in one solve, so their adjacent y-block ranges are both live.
// Every compliance here differs from every other, which makes a swap between the two families -- or
// between rows inside one -- show up as a wrong offset.
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

  // Settled once no rod moves farther than this, or turns through a larger angle, in one step.
  constexpr double settled_step = 1e-12;
  constexpr int max_steps = 1000;

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

      ConstraintSet<HostExecSpace> constraints;
      constraints.fixed_positions = FixedPositionViews<HostExecSpace>(1);
      constraints.fixed_positions.rod(0) = 0;
      constraints.fixed_positions.target_point(0) = position_target;
      constraints.fixed_positions.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
      constraints.fixed_positions.compliance(0) = position_compliance;

      constraints.fixed_poses = FixedPoseViews<HostExecSpace>(1);
      constraints.fixed_poses.rod(0) = 1;
      constraints.fixed_poses.target_point(0) = pose_target;
      constraints.fixed_poses.target_orientation(0) = pose_orientation_target;
      constraints.fixed_poses.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
      constraints.fixed_poses.position_compliance(0) = pose_compliance;
      constraints.fixed_poses.orientation_compliance(0) = orientation_compliance;

      SolveConfig cfg;
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
      int steps = 0;
      while (steps < max_steps) {
        ASSERT_TRUE(step_rods(rods_d, constraints_d, cfg, load_d).converged) << "dt=" << dt << " step " << steps;
        ++steps;
        if (max_step_displacement(rods_d, dt) <= settled_step) {
          break;
        }
      }
      ASSERT_LT(steps, max_steps) << "not settled about axis " << torque_axis << " at dt=" << dt;
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

// Contacts alongside an anchor, which nothing else does: the x-block appears only in the contact-only
// and grid tests, so the reduced problem H = D^T M D - D^T M B S B^T M D has never been formed with
// rigid anchor rows (K^-1 = 0) inside S. Unlike the other anchor tests this needs the outer PGD to
// iterate, so max_outer_iters stays at its default rather than being pinned to 1.
//
// One sphere is held; the other is driven onto it along the line of centres. At the fixed point the
// driven sphere's balance fixes the contact multiplier, and the held sphere's balance makes the anchor
// reaction exactly minus the contact force it absorbs -- an identity spanning the two blocks.
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

  ConstraintSet<HostExecSpace> constraints;
  constraints.contacts = ContactViews<HostExecSpace>(1);
  constraints.contacts.rod_i(0) = 0;
  constraints.contacts.rod_j(0) = 1;
  constraints.fixed_positions = FixedPositionViews<HostExecSpace>(1);
  constraints.fixed_positions.rod(0) = 0;
  constraints.fixed_positions.target_point(0) = anchor_target;
  constraints.fixed_positions.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
  constraints.fixed_positions.compliance(0) = Vector3d{0.0, 0.0, 0.0};  // rigid

  SolveConfig cfg;
  cfg.dt = 0.5;
  cfg.viscosity = 1.0;
  cfg.max_cg_iters = 500;
  cfg.cg_tol = 1e-12;
  cfg.outer_tol = 1e-12;

  rods.force(1) = Vector3d{0.0, 0.0, -push};
  const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
  const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
  const auto load_d = copy_load(rods_d);
  for (int step = 0; step < 100; ++step) {
    step_rods(rods_d, constraints_d, cfg, load_d);
  }
  deep_copy(rods, rods_d);
  deep_copy(constraints, constraints_d);

  EXPECT_NEAR(norm(rods.center(0) - anchor_target), 0.0, 1e-10);
  EXPECT_NEAR(norm(rods.center(1) - rods.center(0)), 2.0 * radius, 1e-9);
  EXPECT_NEAR(constraints.contacts.lambda(0), push, 1e-9);
  // The driven sphere sits above, so the contact presses the held one downward and the anchor pulls up.
  EXPECT_NEAR(norm(Vector3d(constraints.fixed_positions.lambda(0)) - Vector3d{0.0, 0.0, push}), 0.0, 1e-9);
}

// Anchors on distinct rods are the ordinary case; two on one rod duplicate that rod's constraint
// columns and leave the bilateral block rank deficient. solve() rejects the latter, but only in debug
// builds, so the detection itself is exercised here where it runs in any build.
TEST(Mbody, DoublyAnchoredRodsAreDetected) {
  constexpr size_t kNumRods = 3;
  const auto anchor_on = [](int rod) {
    FixedPositionViews<HostExecSpace> anchors(1);
    anchors.rod(0) = rod;
    anchors.target_point(0) = Vector3d{0.0, 0.0, 0.0};
    anchors.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
    anchors.compliance(0) = Vector3d{0.0, 0.0, 0.0};
    return anchors;
  };

  ConstraintSet<HostExecSpace> both_ends;
  both_ends.fixed_positions = FixedPositionViews<HostExecSpace>(2);
  for (int a = 0; a < 2; ++a) {
    both_ends.fixed_positions.rod(a) = 2 * a;  // rods 0 and 2
    both_ends.fixed_positions.target_point(a) = Vector3d{0.0, 0.0, 0.0};
    both_ends.fixed_positions.body_offset(a) = Vector3d{0.0, 0.0, 0.0};
    both_ends.fixed_positions.compliance(a) = Vector3d{0.0, 0.0, 0.0};
  }
  EXPECT_EQ(impl::count_doubly_anchored_rods(create_mirror_view_and_copy(TestExecSpace{}, both_ends), kNumRods), 0u);

  ConstraintSet<HostExecSpace> twice_on_one;
  twice_on_one.fixed_positions = FixedPositionViews<HostExecSpace>(2);
  for (int a = 0; a < 2; ++a) {
    twice_on_one.fixed_positions.rod(a) = 1;  // both on rod 1
    twice_on_one.fixed_positions.target_point(a) = Vector3d{0.0, 0.0, 0.0};
    twice_on_one.fixed_positions.body_offset(a) = Vector3d{0.0, 0.0, 0.0};
    twice_on_one.fixed_positions.compliance(a) = Vector3d{0.0, 0.0, 0.0};
  }
  EXPECT_EQ(impl::count_doubly_anchored_rods(create_mirror_view_and_copy(TestExecSpace{}, twice_on_one), kNumRods), 1u);

  // A position and a pose on one rod collide the same way: both constrain its translational block.
  ConstraintSet<HostExecSpace> mixed;
  mixed.fixed_positions = anchor_on(1);
  mixed.fixed_poses = FixedPoseViews<HostExecSpace>(1);
  mixed.fixed_poses.rod(0) = 1;
  mixed.fixed_poses.target_point(0) = Vector3d{0.0, 0.0, 0.0};
  mixed.fixed_poses.target_orientation(0) = Quaterniond{1.0, 0.0, 0.0, 0.0};
  mixed.fixed_poses.body_offset(0) = Vector3d{0.0, 0.0, 0.0};
  mixed.fixed_poses.position_compliance(0) = Vector3d{0.0, 0.0, 0.0};
  mixed.fixed_poses.orientation_compliance(0) = Vector3d{0.0, 0.0, 0.0};
  EXPECT_EQ(impl::count_doubly_anchored_rods(create_mirror_view_and_copy(TestExecSpace{}, mixed), kNumRods), 1u);

#ifndef NDEBUG
  // solve() enforces this with a debug-only assert, so the throw is reachable only where that assert
  // is compiled in. The counting above is what runs everywhere.
  RodViews<HostExecSpace> rods(kNumRods);
  for (size_t i = 0; i < kNumRods; ++i) {
    rods.center(i) = Vector3d{0.0, 0.0, static_cast<double>(i)};
    rods.orientation(i) = Quaterniond{1.0, 0.0, 0.0, 0.0};
    rods.radius(i) = 0.2;
    rods.length(i) = 0.0;
  }
  zero_rod_state(rods);
  SolveConfig cfg;
  EXPECT_ANY_THROW(solve_on_device(rods, twice_on_one, cfg));
#endif
}

// The pre-bend exists only so a constraint with no gradient at its rest state can be differentiated
// at all, so it must not reach the answer. Two amplitudes a decade apart, on both boundary value
// problems, have to agree far more tightly than either differs from its continuum limit.
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

// The other classical boundary value problem for the same chain, and a sharper probe of the
// discretization than the cantilever. Supported at both ends under a midspan load, the discrete
// Hencky deflection is exactly
//
//   delta = P L^3 / (48 EI) * (1 + 2/N^2)
//
// by the same sum of joint moments against their lever arms. Unlike the cantilever's (3N+1)/(2N^2)
// this is SECOND order with constant exactly 2 -- same elements, same springs, different boundary
// treatment, which places the cantilever's lost order in how its wall holds two adjacent nodes rather
// than in the element.
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

  // Read as a convergence rate, N^2 * rel_err is exactly 2 at every resolution, not just in the limit.
  EXPECT_NEAR(finest_scaled_error, 2.0, 1e-4);
}

// Two anchors on two *different* rods coupled through the springs between them, which is the ordinary
// multi-anchor case and the one a simply supported beam needs. B's columns act on two distinct bodies'
// translational blocks, so it keeps full column rank, unlike two anchors placed on one rod.
//
// One solve against a dense solve of the same Schur complement: every multiplier and velocity matches,
// and both anchored spheres are held at zero velocity. The reference transposes B densely, so it also
// checks each family's rate operator against its force operator. The chain starts on a shallow arc so
// every bend angle has a gradient (see solve_static_bend) and the bend rows resist the load.
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

    FixedPositionViews<HostExecSpace> supports(2);
    const Vector3d left = Vector3d(rods.center(0));
    const Vector3d right = Vector3d(rods.center(num_spheres - 1));
    for (int a = 0; a < 2; ++a) {
      supports.rod(a) = (a == 0) ? 0 : static_cast<int>(num_spheres - 1);
      supports.target_point(a) = (a == 0) ? left : right;
      supports.body_offset(a) = Vector3d{0.0, 0.0, 0.0};
      supports.compliance(a) = Vector3d{0.0, 0.0, 0.0};
    }

    ConstraintSet<HostExecSpace> constraints;
    constraints.linear_springs = lin_springs;
    constraints.triple_springs = chain.springs;
    constraints.fixed_positions = supports;

    SolveConfig cfg;
    cfg.dt = dt;
    cfg.viscosity = 1.0;
    cfg.max_cg_iters = 1000;
    cfg.cg_tol = 1e-12;
    cfg.max_outer_iters = 1;

    rods.force(num_spheres / 2) = Vector3d{0.0, -load, 0.0};

    // Dense reference, built from the rods and constraints before solve() mutates them. Columns of B follow
    // the y-block packing order: linear springs, triple springs, fixed positions.
    const auto rods_d = create_mirror_view_and_copy(TestExecSpace{}, rods);
    const auto constraints_d = create_mirror_view_and_copy(TestExecSpace{}, constraints);
    auto b0_lin_d = make_constraint_values(constraints_d.linear_springs);
    auto b0_triple_d = make_constraint_values(constraints_d.triple_springs);
    auto b0_fixed_d = make_constraint_values(constraints_d.fixed_positions);
    const impl::PairForceOp<TestExecSpace> B_lin(
        impl::compute_linear_spring_geometry(rods_d, constraints_d.linear_springs, b0_lin_d), num_spheres);
    const impl::TripleForceOp<TestExecSpace> B_triple(
        impl::compute_triple_point_angular_spring_geometry(rods_d, constraints_d.triple_springs, b0_triple_d),
        num_spheres);
    const impl::SingleForceOp<TestExecSpace> B_fixed(
        impl::compute_fixed_position_geometry(rods_d, constraints_d.fixed_positions, b0_fixed_d), num_spheres);
    const impl::LocalDragMobilityOp<TestExecSpace> M(cfg.viscosity, rods_d);

    const DenseMat B_dense =
        dense_hcat(dense_hcat(materialize_dense(B_lin), materialize_dense(B_triple)), materialize_dense(B_fixed));
    const DenseMat BT_dense = dense_transpose(B_dense);
    const DenseMat M_dense = materialize_dense(M);
    const auto b0_lin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_lin_d);
    const auto b0_triple = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_triple_d);
    const auto b0_fixed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b0_fixed_d);

    std::vector<double> b0, kinv;
    for (size_t k = 0; k < b0_lin.extent(0); ++k) {
      b0.push_back(b0_lin(k));
      kinv.push_back(1.0 / constraints.linear_springs.spring_constant(k));
    }
    for (size_t k = 0; k < b0_triple.extent(0); ++k) {
      b0.push_back(b0_triple(k));
      kinv.push_back(1.0 / constraints.triple_springs.spring_constant(k));
    }
    for (size_t k = 0; k < b0_fixed.extent(0); ++k) {
      b0.push_back(b0_fixed(k));
      kinv.push_back(0.0);  // rigid
    }

    // b = b0 + dt B^T M F_ext and (dt B^T M B + K^-1) y = -b, then v = M (F_ext + B y).
    std::vector<double> force_torque_ext(6 * num_spheres, 0.0);
    for (size_t i = 0; i < 6 * num_spheres; ++i) {
      force_torque_ext[i] = rods.force_torque_view()(i);
    }
    const std::vector<double> b_rate = dense_matvec(BT_dense, dense_matvec(M_dense, force_torque_ext));
    std::vector<double> neg_b(b0.size());
    for (size_t r = 0; r < b0.size(); ++r) {
      neg_b[r] = -(b0[r] + dt * b_rate[r]);
    }
    DenseMat schur = dense_matmul(BT_dense, dense_matmul(M_dense, B_dense));
    for (size_t r = 0; r < schur.size(); ++r) {
      for (size_t c = 0; c < schur.size(); ++c) {
        schur[r][c] *= dt;
      }
      schur[r][r] += kinv[r];
    }
    const std::vector<double> y_expected = dense_solve(schur, neg_b);
    std::vector<double> total_force_torque = dense_matvec(B_dense, y_expected);
    for (size_t i = 0; i < total_force_torque.size(); ++i) {
      total_force_torque[i] += force_torque_ext[i];
    }
    const std::vector<double> vel_omega_expected = dense_matvec(M_dense, total_force_torque);

    ASSERT_TRUE(solve_on_device(rods, constraints, cfg).converged) << "dt=" << dt;

    std::vector<double> y;
    for (size_t k = 0; k < constraints.linear_springs.size(); ++k) y.push_back(constraints.linear_springs.lambda(k));
    for (size_t k = 0; k < constraints.triple_springs.size(); ++k) y.push_back(constraints.triple_springs.lambda(k));
    for (size_t k = 0; k < 2; ++k) {
      for (int c = 0; c < 3; ++c) y.push_back(constraints.fixed_positions.lambda(k)[c]);
    }
    ASSERT_EQ(y.size(), y_expected.size());
    for (size_t r = 0; r < y.size(); ++r) {
      EXPECT_NEAR(y[r], y_expected[r], 1e-9) << "multiplier " << r << " at dt=" << dt;
    }
    for (size_t i = 0; i < 6 * num_spheres; ++i) {
      EXPECT_NEAR(rods.velocity_omega_view()(i), vel_omega_expected[i], 1e-12) << "entry " << i << " at dt=" << dt;
    }
    EXPECT_NEAR(norm(rods.velocity(0)), 0.0, 1e-12) << "dt=" << dt;
    EXPECT_NEAR(norm(rods.velocity(num_spheres - 1)), 0.0, 1e-12) << "dt=" << dt;
  }
}

}  // namespace

}  // namespace mbody

}  // namespace mundy
