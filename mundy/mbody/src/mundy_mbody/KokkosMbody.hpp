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

#ifndef MUNDY_MBODY_KOKKOSMBODY_HPP_
#define MUNDY_MBODY_KOKKOSMBODY_HPP_

// C++ core
#include <algorithm>  // for std::max
#include <cmath>      // for std::log
#include <ostream>    // for std::ostream

// Mundy
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/cqpp.hpp>
#include <mundy_math/lcp.hpp>
#include <mundy_math/linear_ops.hpp>
#include <mundy_math/linear_system.hpp>
#include <mundy_mbody/KokkosMbodyTypes.hpp>
#include <mundy_mbody/impl/KokkosMbodyImpl.hpp>

namespace mundy {

namespace mbody {

//! \name Solve configurations and results
//@{

/// \brief Configuration for a mixed LCP step: the step, the outer (PGD) solve, and the inner (CG) solve.
struct MixedLCPConfig {
  double dt = 1.0;
  double viscosity = 1.0;
  unsigned max_outer_iters = 1000;
  double outer_tol = 1e-6;
  unsigned max_cg_iters = 200;
  double cg_tol = 1e-8;
};

/// \brief Result of a mixed LCP step: the contact solve's iteration count, final residual, and whether it converged.
struct MixedLCPResult {
  unsigned num_iters{0};
  double residual{0.0};
  bool converged{false};
};

/// \brief Write a MixedLCPResult to an ostream.
inline std::ostream& operator<<(std::ostream& os, const MixedLCPResult& result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}

/// \brief Configuration for a mixed SLCP step: its inner mixed LCPs and the sequence of them.
///
/// An iterate is accepted once, at the configuration it moves the rods to, every bilateral row's residual
/// psi + K^-1 y and the displacement its constraint force directions would change by are within length_tol (rows and
/// displacements measured in length) and angle_tol (in radians). Neither can be met below its floor: the inner
/// cg_tol, and about 1e-8 rad for angles measured near 0 or pi.
struct MixedSLCPConfig {
  MixedLCPConfig inner_lcp_config;
  unsigned max_iters = 20;
  double length_tol = 1e-6;
  double angle_tol = 1e-6;
};

/// \brief Result of a mixed SLCP step: the linearization count, the returned iterate's residual, and convergence.
///
/// residual is the returned iterate's largest acceptance residual over the tolerance it is tested against, so
/// converged is residual <= 1, and accepted_lcp_result is that iterate's inner LCP result. Unconverged, the returned
/// iterate is the first linearization; num_iters < max_iters then means the sequence stopped once it was predicted not
/// to converge within max_iters.
struct MixedSLCPResult {
  unsigned num_iters{0};
  double residual{0.0};
  bool converged{false};
  MixedLCPResult accepted_lcp_result{};
};

/// \brief Write a MixedSLCPResult to an ostream.
inline std::ostream& operator<<(std::ostream& os, const MixedSLCPResult& result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged
     << ", accepted_lcp_result: {" << result.accepted_lcp_result << "}";
  return os;
}
//@}

/// \brief Move every rod through one step of its velocity, to the configuration C^k (+) G^k dt U.
///
/// Centers move by dt v. Orientations turn by exp(dt omega / 2) (x) q, the exact rotation under an angular velocity
/// held over the step, whose rate at dt = 0 is G^k omega.
template <typename ExecSpace>
void advance_rods(const RodViews<ExecSpace>& rods, double dt) {
  Kokkos::parallel_for(
      "advance_rods", Kokkos::RangePolicy<ExecSpace>(0, rods.size()), KOKKOS_LAMBDA(const int i) {
        rods.center(i) = rods.center(i) + dt * rods.velocity(i);
        auto orientation = rods.orientation(i);
        rotate_quaternion(orientation, Vector3d(rods.omega(i)), dt);
      });
}

namespace impl {

/// \brief The parts of a step that no linearization changes.
///
/// The contact block is linearized at the start of the step, C^k: its Jacobian and q = Phi(C^k) + dt D^T U_free.
/// The mobility is that of C^k, and U_free = V_ext + M F_ext is the velocity without constraint forces.
template <typename ExecSpace>
struct StepData {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;

  ConstraintIndexMap index_map;
  size_t num_rods;
  size_t num_contacts;
  PairGeometry<ExecSpace> contact_geo;
  view_t q;
  view_t kinv;
  LocalDragMobilityOp<ExecSpace> mobility;
  view_t u_free;
};

/// \brief The step data of rods and constraints at their current configuration.
template <typename ExecSpace>
StepData<ExecSpace> make_step_data(const RodViews<ExecSpace>& rods, const ConstraintSet<ExecSpace>& constraints,
                                   const MixedLCPConfig& cfg) {
  using view_t = typename StepData<ExecSpace>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;

  const size_t num_rods = rods.size();
  const size_t num_contacts = constraints.contacts.size();
  MUNDY_THROW_ASSERT(count_doubly_anchored_rods(constraints, num_rods) == 0, std::invalid_argument,
                     "mbody: a rod carries more than one fixed-position or fixed-pose anchor, leaving the bilateral "
                     "block rank deficient.");

  view_t q("q", num_contacts);
  const PairGeometry<ExecSpace> contact_geo = compute_contact_geometry(rods, constraints.contacts, q);
  const LocalDragMobilityOp<ExecSpace> mobility(cfg.viscosity, rods);

  view_t u_free("u_free", 6 * num_rods);
  Kokkos::deep_copy(u_free, rods.velocity_omega_view());
  view_t m_force_torque_ext("m_force_torque_ext", 6 * num_rods);
  mobility.apply(rods.force_torque_view(), m_force_torque_ext);
  backend_t::axpby(1.0, m_force_torque_ext, 1.0, u_free);

  if (num_contacts > 0) {
    view_t gap_rate("gap_rate", num_contacts);
    PairForceOpT<ExecSpace>(contact_geo, num_rods).apply(u_free, gap_rate);
    backend_t::axpby(cfg.dt, gap_rate, 1.0, q);
  }

  return StepData<ExecSpace>{make_constraint_index_map(constraints), num_rods, num_contacts, contact_geo, q,
                             compliance_diagonal(constraints),       mobility, u_free};
}

/// \brief The bilateral map B of a bilateral geometry, from multipliers to center-of-mass force and torque.
template <typename ExecSpace>
auto make_bilateral_force_op(const BilateralGeometry<ExecSpace>& geo, size_t num_rods) {
  using backend_t = KokkosBackend<ExecSpace>;
  return make_concat_domain_op<backend_t>(
      make_concat_domain_op<backend_t>(PairForceOp<ExecSpace>(geo.pairs, num_rods),
                                       TripleForceOp<ExecSpace>(geo.triples, num_rods)),
      SingleForceOp<ExecSpace>(geo.singles, num_rods));
}

/// \brief The bilateral map B^T, from center-of-mass translational and rotational velocity to constraint rates.
template <typename ExecSpace>
auto make_bilateral_rate_op(const BilateralGeometry<ExecSpace>& geo, size_t num_rods) {
  using backend_t = KokkosBackend<ExecSpace>;
  return make_concat_range_op<backend_t>(
      make_concat_range_op<backend_t>(PairForceOpT<ExecSpace>(geo.pairs, num_rods),
                                      TripleForceOpT<ExecSpace>(geo.triples, num_rods)),
      SingleForceOpT<ExecSpace>(geo.singles, num_rods));
}

/// \brief One linearization's solution and the wrench and velocity it produces.
///
/// x and y are the contact magnitudes and bilateral multipliers, bilateral_wrench is B y, wrench is W = D x + B y,
/// m_wrench is M W, and velocity is U_free + M W.
template <typename ExecSpace>
struct LinearizedStep {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;

  view_t x;
  view_t y;
  view_t bilateral_wrench;
  view_t wrench;
  view_t m_wrench;
  view_t velocity;
  MixedLCPResult result;
};

/// \brief Solve the mixed LCP with the step's contact block and the bilateral block linearized at geo.
///
/// The bilateral linear term is b = psi + pull_scale B^T pull_velocity; the contact solve starts from x_start.
template <typename ExecSpace>
LinearizedStep<ExecSpace> solve_linearization(
    const StepData<ExecSpace>& step, const BilateralGeometry<ExecSpace>& geo,
    const Kokkos::View<double*, typename ExecSpace::memory_space>& psi,
    const Kokkos::View<double*, typename ExecSpace::memory_space>& pull_velocity, double pull_scale,
    const Kokkos::View<double*, typename ExecSpace::memory_space>& x_start, const MixedLCPConfig& cfg) {
  using view_t = typename LinearizedStep<ExecSpace>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;

  const size_t num_rods = step.num_rods;
  const size_t num_contacts = step.num_contacts;
  const size_t num_bilateral = step.index_map.total;
  const bool has_contacts = num_contacts > 0;
  const bool has_bilateral = num_bilateral > 0;

  const PairForceOp<ExecSpace> D(step.contact_geo, num_rods);
  const PairForceOpT<ExecSpace> DT(step.contact_geo, num_rods);
  const auto B = make_bilateral_force_op(geo, num_rods);
  const auto BT = make_bilateral_rate_op(geo, num_rods);
  // The mixed CQPP's "M" is dt * mobility: it maps a constraint force to the displacement it causes over the step,
  // not to a velocity. The velocity this step returns uses raw M.
  const auto M_dt = make_scaled_op<backend_t>(cfg.dt, step.mobility);

  view_t b("b", num_bilateral);
  Kokkos::deep_copy(b, psi);
  if (has_bilateral) {
    view_t b_rate("b_rate", num_bilateral);
    BT.apply(pull_velocity, b_rate);
    backend_t::axpby(pull_scale, b_rate, 1.0, b);
  }

  // Schur complement S := (B^T M B + K^{-1})^{-1}, realized via matrix-free CG.
  const auto btmb_plus_kinv =
      make_sum_op<backend_t>(make_quadratic_form<backend_t>(BT, M_dt, B), make_diagonal_op<backend_t>(step.kinv));
  const CGConfig<double> cg_cfg{cfg.max_cg_iters, cfg.cg_tol};
  const auto S = make_cg_inv_op<backend_t>(btmb_plus_kinv, cg_cfg);

  // x* (contact force magnitudes): the reduced CQPP, which without bilateral rows is the contact LCP. Without
  // contacts x* is empty, and its projected residual is zero before any iteration.
  LinearizedStep<ExecSpace> out;
  out.x = view_t("x", num_contacts);
  Kokkos::deep_copy(out.x, x_start);
  out.result = MixedLCPResult{0, 0.0, 0.0 <= cfg.outer_tol};
  if (has_contacts) {
    view_t grad("grad", num_contacts);
    view_t x_tmp("x_tmp", num_contacts);
    view_t grad_tmp("grad_tmp", num_contacts);

    auto pgd = make_pgd_solution_strategy(PGDConfig<double>{.max_iters = cfg.max_outer_iters, .tol = cfg.outer_tol});
    auto pgd_state = make_pgd_state(out.x, grad, x_tmp, grad_tmp);
    PGDResult<double> pgd_result;
    if (has_bilateral) {
      const auto mcqpp =
          make_mixed_cqpp<backend_t>(DT, M_dt, D, step.q, B, S, BT, b, LowerBoundSpace<double>{.lower_bound = 0.0});
      pgd_result = solve_mixed_cqpp(mcqpp, pgd, pgd_state);
    } else {
      pgd_result = solve_lcp(make_lcp<backend_t>(DT, M_dt, D, step.q), pgd, pgd_state);
    }
    out.result = MixedLCPResult{pgd_result.num_iters, pgd_result.residual, pgd_result.converged};
  }
  MUNDY_THROW_REQUIRE(out.result.converged, std::runtime_error, "mbody: outer PGD solve failed to converge.");

  // y* = -S (b + B^T M D x*), and the constraint force/torque D x* + B y*.
  view_t Dx("Dx", 6 * num_rods);
  if (has_contacts) {
    D.apply(out.x, Dx);
  }
  out.y = view_t("y", num_bilateral);
  out.bilateral_wrench = view_t("bilateral_wrench", 6 * num_rods);
  if (has_bilateral) {
    view_t y_rhs("y_rhs", num_bilateral);
    if (has_contacts) {
      view_t MDx("MDx", 6 * num_rods);
      M_dt.apply(Dx, MDx);
      BT.apply(MDx, y_rhs);
      backend_t::axpby(1.0, b, 1.0, y_rhs);  // y_rhs = b + B^T M D x*
    } else {
      Kokkos::deep_copy(y_rhs, b);
    }
    S.apply(y_rhs, out.y);
    backend_t::axpby(-1.0, out.y, 0.0, out.y);  // y := -y
    B.apply(out.y, out.bilateral_wrench);
  }
  out.wrench = view_t("wrench", 6 * num_rods);
  Kokkos::deep_copy(out.wrench, out.bilateral_wrench);
  if (has_contacts) {
    backend_t::axpby(1.0, Dx, 1.0, out.wrench);
  }

  out.m_wrench = view_t("m_wrench", 6 * num_rods);
  step.mobility.apply(out.wrench, out.m_wrench);
  out.velocity = view_t("velocity", 6 * num_rods);
  Kokkos::deep_copy(out.velocity, step.u_free);
  backend_t::axpby(1.0, out.m_wrench, 1.0, out.velocity);
  return out;
}

/// \brief Apply a linearization to rods and constraints.
///
/// force/torque gains W, velocity/omega becomes U_free + M W, and every family receives its multipliers.
template <typename ExecSpace>
void write_step(const RodViews<ExecSpace>& rods, const ConstraintSet<ExecSpace>& constraints,
                const ConstraintIndexMap& index_map, const LinearizedStep<ExecSpace>& step) {
  auto force_torque = rods.force_torque_view();
  KokkosBackend<ExecSpace>::axpby(1.0, step.wrench, 1.0, force_torque);
  Kokkos::deep_copy(rods.velocity_omega_view(), step.velocity);

  Kokkos::deep_copy(constraints.contacts.lambda_view(), step.x);
  Kokkos::deep_copy(constraints.linear_springs.lambda_view(), subrange(step.y, index_map.linear_springs));
  Kokkos::deep_copy(constraints.angular_springs.lambda_view(), subrange(step.y, index_map.angular_springs));
  Kokkos::deep_copy(constraints.pins.lambda_view(), subrange(step.y, index_map.pins));
  Kokkos::deep_copy(constraints.fixed_lengths.lambda_view(), subrange(step.y, index_map.fixed_lengths));
  Kokkos::deep_copy(constraints.triple_springs.lambda_view(), subrange(step.y, index_map.triple_springs));
  Kokkos::deep_copy(constraints.fixed_positions.lambda_view(), subrange(step.y, index_map.fixed_positions));
  Kokkos::deep_copy(constraints.fixed_poses.lambda_view(), subrange(step.y, index_map.fixed_poses));
}

/// \brief An iterate's largest acceptance residual over its tolerance, at the configuration it moves the rods to.
///
/// The residuals are |psi + K^-1 y| there and M (B' y - B y), the displacement by which the bilateral force
/// directions there would change the step, where B and B' map multipliers to center-of-mass force and torque at the
/// iterate's linearization point and at that configuration. B y is the iterate's bilateral wrench.
template <typename ExecSpace>
double slcp_merit(const StepData<ExecSpace>& step, const LinearizedStep<ExecSpace>& iterate,
                  const BilateralGeometry<ExecSpace>& trial_geo,
                  const Kokkos::View<double*, typename ExecSpace::memory_space>& trial_psi,
                  const MixedSLCPConfig& cfg) {
  using view_t = typename StepData<ExecSpace>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;
  const size_t num_rods = step.num_rods;

  const LengthAngleMax rows = max_bilateral_residual<ExecSpace>(trial_psi, step.kinv, iterate.y, step.index_map);

  view_t wrench_change("wrench_change", 6 * num_rods);
  make_bilateral_force_op(trial_geo, num_rods).apply(iterate.y, wrench_change);
  backend_t::axpby(-1.0, iterate.bilateral_wrench, 1.0, wrench_change);
  view_t displacement_change("displacement_change", 6 * num_rods);
  step.mobility.apply(cfg.inner_lcp_config.dt, wrench_change, 0.0, displacement_change);
  const LengthAngleMax moved = max_displacement<ExecSpace>(displacement_change, num_rods);

  return std::max({rows.length / cfg.length_tol, rows.angle / cfg.angle_tol, moved.length / cfg.length_tol,
                   moved.angle / cfg.angle_tol});
}

}  // namespace impl

/// \brief One linearly implicit step of a multibody system: a mixed LCP over its contacts and bilateral constraints.
///
/// Solves, for contact force magnitudes x >= 0 and bilateral multipliers y in R^m:
///   x*, y* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// via the Schur complement S := (B^T M B + K^{-1})^{-1}:
///   H := D^T M D - D^T M B S B^T M D,  g := q - D^T M B S b
///   x* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y* = -S (b + B^T M D x*)
///
/// Every constraint is linearized at the start of the step, C^k: q = Phi(C^k) + dt D^T U_free and
/// b = psi(C^k) + dt B^T U_free, with U_free = V_ext + M F_ext. D maps contact force magnitudes to center-of-mass force
/// and torque, and D^T maps center-of-mass translational and rotational velocity to the rate of change of separation.
/// B and B^T do the same for the concatenated bilateral multipliers and the rates of change of their constraint
/// values, in the packing order the constraint index map fixes. K^{-1} is the per-constraint compliance, zero for a
/// rigidly held one. M is the local-drag rod mobility, dt M in the problem above. S is SPD and only its apply-action
/// is cheap, so it is realized by a matrix-free CG, not an explicit inverse.
///
/// On entry rods' force/torque is the external load F_ext and velocity/omega the imposed velocity V_ext. On exit
/// force/torque is F_ext + D x* + B y*, velocity/omega is U_free + M (D x* + B y*), every family's lambda holds its
/// multipliers, and the rods have not moved. advance_rods can be used to perform the consistent time integration, after
/// which each constraint holds to first order in dt.
///
/// A boundary condition belongs in the constraint set rather than being imposed on rods afterwards.
/// A body held by a constraint contributes its reaction to B y, which cancels out of the relaxation's
/// fixed point exactly; a body held by discarding its velocity after the solve leaves that reaction
/// outside B, and the fixed point then carries an error of order dt times the body's mobility.
template <typename ExecSpace>
MixedLCPResult solve_mixed_lcp(const RodViews<ExecSpace>& rods, const ConstraintSet<ExecSpace>& constraints,
                               const MixedLCPConfig& cfg) {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;

  const impl::StepData<ExecSpace> step = impl::make_step_data(rods, constraints, cfg);
  view_t psi("psi", step.index_map.total);
  const impl::BilateralGeometry<ExecSpace> geo =
      impl::compute_bilateral_geometry(rods, constraints, step.index_map, psi);
  if (step.num_contacts == 0 && step.index_map.total == 0) {
    Kokkos::deep_copy(rods.velocity_omega_view(), step.u_free);
    return MixedLCPResult{0, 0.0, 0.0 <= cfg.outer_tol};
  }

  const impl::LinearizedStep<ExecSpace> linearized =
      impl::solve_linearization(step, geo, psi, step.u_free, cfg.dt, view_t("x_start", step.num_contacts), cfg);
  impl::write_step(rods, constraints, step.index_map, linearized);
  return linearized.result;
}

/// \brief One step of a multibody system whose bilateral constraints hold at its end: a sequence of mixed LCPs.
///
/// The step's configuration is the push-forward C(W) = C^k (+) G^k (dt V_ext + M (F_ext + W)) of the constraint wrench
/// W. Contacts are linearized at C^k throughout, so they hold to the accuracy of that linearization. The bilateral rows
/// are linearized afresh at each iterate's configuration C_n, starting from C_0 = C^k: iterate n solves the mixed LCP
/// with B_n = B(C_n) and
///   b_n = psi(C_n) - dt B_n^T M W_{n-1}   (n >= 1),
/// the linear model of psi at C(W) about C_n, and moves the rods to C_{n+1} = C(W_n). Iterate 0 is solve_mixed_lcp's
/// step.
///
/// At a converged iterate, C* = C(W*) satisfies psi(C*) + K^{-1} y* = 0 for every bilateral row and
/// W* = D x* + B(C*) y*: rigid rows hold exactly and compliant ones follow their constitutive law at the end of the
/// step. An iterate is accepted once both hold at C_{n+1} to the configured tolerances. The sequence stops unconverged
/// at max_iters iterates, or earlier once its contraction predicts it cannot converge by then, and the step is then
/// iterate 0. The iteration contracts at a rate of about dt times the mobility times the constraints' curvature
/// weighted by their multipliers, so it converges only where that is below one.
///
/// On entry rods' force/torque is the external load F_ext and velocity/omega the imposed velocity V_ext. On exit
/// force/torque is F_ext + W, velocity/omega is U_free + M W, every family's lambda holds its multipliers, all of the
/// returned iterate, and the rods have not moved. advance_rods can be used to perform the consistent time integration,
/// which for a converged iterate reaches the configuration at which it was accepted.
template <typename ExecSpace>
MixedSLCPResult solve_mixed_slcp(const RodViews<ExecSpace>& rods, const ConstraintSet<ExecSpace>& constraints,
                                 const MixedSLCPConfig& cfg) {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;
  MUNDY_THROW_REQUIRE(cfg.max_iters >= 1, std::invalid_argument, "mbody::solve_mixed_slcp: max_iters must be >= 1.");
  MUNDY_THROW_REQUIRE(cfg.length_tol > 0.0 && cfg.angle_tol > 0.0, std::invalid_argument,
                      "mbody::solve_mixed_slcp: length_tol and angle_tol must be positive.");
  const MixedLCPConfig& lcp_cfg = cfg.inner_lcp_config;

  const impl::StepData<ExecSpace> step = impl::make_step_data(rods, constraints, lcp_cfg);
  const ConstraintIndexMap& index_map = step.index_map;
  view_t psi("psi", index_map.total);
  const impl::BilateralGeometry<ExecSpace> geo = impl::compute_bilateral_geometry(rods, constraints, index_map, psi);
  if (step.num_contacts == 0 && index_map.total == 0) {
    Kokkos::deep_copy(rods.velocity_omega_view(), step.u_free);
    return MixedSLCPResult{1, 0.0, true, MixedLCPResult{0, 0.0, 0.0 <= lcp_cfg.outer_tol}};
  }

  const impl::LinearizedStep<ExecSpace> first =
      impl::solve_linearization(step, geo, psi, step.u_free, lcp_cfg.dt, view_t("x_start", step.num_contacts), lcp_cfg);

  // Trial configurations C_{n+1} = C^k (+) G^k dt U_n, in storage of their own: the mobility reads rods' poses.
  RodViews<ExecSpace> trial(step.num_rods);
  Kokkos::deep_copy(trial.radius_view(), rods.radius_view());
  Kokkos::deep_copy(trial.length_view(), rods.length_view());

  impl::LinearizedStep<ExecSpace> iterate = first;
  double first_merit = 0.0;
  double previous_merit = 0.0;
  for (unsigned num_iters = 1;; ++num_iters) {
    Kokkos::deep_copy(trial.center_view(), rods.center_view());
    Kokkos::deep_copy(trial.orientation_view(), rods.orientation_view());
    Kokkos::deep_copy(trial.velocity_omega_view(), iterate.velocity);
    advance_rods(trial, lcp_cfg.dt);
    view_t trial_psi("trial_psi", index_map.total);
    const impl::BilateralGeometry<ExecSpace> trial_geo =
        impl::compute_bilateral_geometry(trial, constraints, index_map, trial_psi);

    const double merit = impl::slcp_merit(step, iterate, trial_geo, trial_psi, cfg);
    if (num_iters == 1) {
      first_merit = merit;
    }
    if (merit <= 1.0) {
      impl::write_step(rods, constraints, index_map, iterate);
      return MixedSLCPResult{num_iters, merit, true, iterate.result};
    }

    // Stop once the contraction observed so far cannot bring the merit to 1 within max_iters.
    const unsigned remaining = cfg.max_iters - num_iters;
    const bool hopeless =
        num_iters >= 2 && static_cast<double>(remaining) * std::log(merit / previous_merit) + std::log(merit) > 0.0;
    if (remaining == 0 || hopeless) {
      impl::write_step(rods, constraints, index_map, first);
      return MixedSLCPResult{num_iters, first_merit, false, first.result};
    }
    previous_merit = merit;

    iterate = impl::solve_linearization(step, trial_geo, trial_psi, iterate.m_wrench, -lcp_cfg.dt, iterate.x, lcp_cfg);
  }
}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODY_HPP_
