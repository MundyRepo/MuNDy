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
#include <cmath>        // for std::log
#include <ostream>      // for std::ostream
#include <type_traits>  // for std::is_same_v

// Mundy
#include <mundy_math/linear_system.hpp>  // for mundy::CGConfig
#include <mundy_math/pgd.hpp>            // for mundy::PGDConfig
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

/// \brief Result of a mixed LCP step: the unilateral solve's iteration count, final residual, and whether it converged.
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

/// \brief One linearly implicit step of a multibody system: a mixed LCP over its unilateral and bilateral constraints.
///
/// Solves, for unilateral multipliers x >= 0, such as contact force magnitudes, and bilateral multipliers y in R^m:
///   x*, y* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// via the Schur complement S := (B^T M B + K^{-1})^{-1}:
///   H := D^T M D - D^T M B S B^T M D,  g := q - D^T M B S b
///   x* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y* = -S (b + B^T M D x*)
///
/// Every constraint is linearized at the start of the step, C^k: q = Phi(C^k) + dt D^T U_free and
/// b = psi(C^k) + dt B^T U_free, with U_free = V_ext + M F_ext. D maps the unilateral multipliers to center-of-mass
/// force and torque, and D^T maps center-of-mass translational and rotational velocity to the rates of change of their
/// constraint values, such as a contact's separation. B and B^T do the same for the bilateral multipliers and the
/// rates of change of their constraint values. K^{-1} is the per-constraint compliance, zero for a rigidly held one.
/// The rigid rows must be independent: two of them holding the same degree of freedom, such as two anchors on one rod,
/// leave S undefined. M is the local-drag rod mobility, dt M in the problem above. S is SPD and only its apply-action
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
template <typename ExecSpace, typename... Families>
MixedLCPResult solve_mixed_lcp(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                               const MixedLCPConfig& cfg) {
  static_assert((std::is_same_v<typename Families::execution_space, ExecSpace> && ...),
                "mbody::solve_mixed_lcp: rods and every constraint family must share one execution space.");
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;

  const PGDConfig<double> pgd_cfg{cfg.max_outer_iters, cfg.outer_tol};
  const CGConfig<double> cg_cfg{cfg.max_cg_iters, cfg.cg_tol};
  const impl::StepData<ExecSpace, Families...> step = impl::make_step_data(rods, constraints, cfg.dt, cfg.viscosity);
  view_t psi("psi", step.index_map.num_bilateral);
  const auto geo = impl::make_block_geometry<ConstraintType::BILATERAL, ExecSpace>(step.index_map);
  impl::compute_block_geometry<ConstraintType::BILATERAL>(rods, constraints, step.index_map, geo, psi);
  if (step.index_map.num_unilateral == 0 && step.index_map.num_bilateral == 0) {
    Kokkos::deep_copy(rods.velocity_omega_view(), step.u_free);
    return MixedLCPResult{0, 0.0, 0.0 <= cfg.outer_tol};
  }

  const impl::LinearizedStep<ExecSpace> linearized =
      impl::solve_linearization(step, geo, psi, impl::Displacement<ExecSpace>{step.u_free, cfg.dt},
                                view_t("x_start", step.index_map.num_unilateral), pgd_cfg, cg_cfg);
  impl::write_step(rods, constraints, step.index_map, linearized);
  return MixedLCPResult{linearized.result.num_iters, linearized.result.residual, linearized.result.converged};
}

/// \brief One step of a multibody system whose bilateral constraints hold at its end: a sequence of mixed LCPs.
///
/// The step's configuration is the push-forward C(W) = C^k (+) G^k (dt V_ext + M (F_ext + W)) of the constraint wrench
/// W. The unilateral rows are linearized at C^k throughout, so they hold to the accuracy of that linearization. The
/// bilateral rows are linearized afresh at each iterate's configuration C_n, starting from C_0 = C^k: iterate n solves
/// the mixed LCP with B_n = B(C_n) and
///   b_n = psi(C_n) - dt B_n^T M W_{n-1}   (n >= 1),
/// the linear model of psi at C(W) about C_n, and moves the rods to C_{n+1} = C(W_n). Iterate 0 is solve_mixed_lcp's
/// step, and the rigid rows must be independent as there.
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
template <typename ExecSpace, typename... Families>
MixedSLCPResult solve_mixed_slcp(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                                 const MixedSLCPConfig& cfg) {
  static_assert((std::is_same_v<typename Families::execution_space, ExecSpace> && ...),
                "mbody::solve_mixed_slcp: rods and every constraint family must share one execution space.");
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;
  MUNDY_THROW_REQUIRE(cfg.max_iters >= 1, std::invalid_argument, "mbody::solve_mixed_slcp: max_iters must be >= 1.");
  MUNDY_THROW_REQUIRE(cfg.length_tol > 0.0 && cfg.angle_tol > 0.0, std::invalid_argument,
                      "mbody::solve_mixed_slcp: length_tol and angle_tol must be positive.");
  const MixedLCPConfig& lcp_cfg = cfg.inner_lcp_config;

  const PGDConfig<double> pgd_cfg{lcp_cfg.max_outer_iters, lcp_cfg.outer_tol};
  const CGConfig<double> cg_cfg{lcp_cfg.max_cg_iters, lcp_cfg.cg_tol};
  const impl::StepData<ExecSpace, Families...> step =
      impl::make_step_data(rods, constraints, lcp_cfg.dt, lcp_cfg.viscosity);
  const impl::ConstraintIndexMap<Families...>& index_map = step.index_map;
  view_t psi("psi", index_map.num_bilateral);
  const auto geo = impl::make_block_geometry<ConstraintType::BILATERAL, ExecSpace>(index_map);
  impl::compute_block_geometry<ConstraintType::BILATERAL>(rods, constraints, index_map, geo, psi);
  if (index_map.num_unilateral == 0 && index_map.num_bilateral == 0) {
    Kokkos::deep_copy(rods.velocity_omega_view(), step.u_free);
    return MixedSLCPResult{1, 0.0, true, MixedLCPResult{0, 0.0, 0.0 <= lcp_cfg.outer_tol}};
  }

  const impl::LinearizedStep<ExecSpace> first =
      impl::solve_linearization(step, geo, psi, impl::Displacement<ExecSpace>{step.u_free, lcp_cfg.dt},
                                view_t("x_start", index_map.num_unilateral), pgd_cfg, cg_cfg);
  const Kokkos::View<RowUnit*, typename ExecSpace::memory_space> row_units = impl::make_row_units<ExecSpace>(index_map);

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
    impl::compute_block_geometry<ConstraintType::BILATERAL>(trial, constraints, index_map, geo, psi);

    const double merit = impl::slcp_merit(step, iterate, geo, psi, row_units, cfg.length_tol, cfg.angle_tol);
    if (num_iters == 1) {
      first_merit = merit;
    }
    if (merit <= 1.0) {
      impl::write_step(rods, constraints, index_map, iterate);
      return MixedSLCPResult{
          num_iters, merit, true,
          MixedLCPResult{iterate.result.num_iters, iterate.result.residual, iterate.result.converged}};
    }

    // Stop once the contraction observed so far cannot bring the merit to 1 within max_iters.
    const unsigned remaining = cfg.max_iters - num_iters;
    const bool hopeless =
        num_iters >= 2 && static_cast<double>(remaining) * std::log(merit / previous_merit) + std::log(merit) > 0.0;
    if (remaining == 0 || hopeless) {
      impl::write_step(rods, constraints, index_map, first);
      return MixedSLCPResult{num_iters, first_merit, false,
                             MixedLCPResult{first.result.num_iters, first.result.residual, first.result.converged}};
    }
    previous_merit = merit;

    iterate = impl::solve_linearization(step, geo, psi, impl::Displacement<ExecSpace>{iterate.m_wrench, -lcp_cfg.dt},
                                        iterate.x, pgd_cfg, cg_cfg);
  }
}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODY_HPP_
