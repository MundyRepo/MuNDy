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
#include <ostream>  // for std::ostream
#include <utility>  // for std::move

// Mundy
#include <mundy_math/linear_system.hpp>                // for mundy::CGConfig
#include <mundy_math/pgd.hpp>                          // for mundy::PGDConfig
#include <mundy_math/preconditioners.hpp>              // for mundy::NoPreconditioner
#include <mundy_mbody/KokkosMbodyMobility.hpp>         // for mundy::mbody::MobilityModel
#include <mundy_mbody/KokkosMbodyPreconditioners.hpp>  // for mundy::mbody::SelfMobilityJacobi
#include <mundy_mbody/KokkosMbodyTypes.hpp>
#include <mundy_mbody/impl/KokkosMbodyImpl.hpp>

namespace mundy {

namespace mbody {

//! \name Solve configurations and results
//@{

/// \brief Numerics of a mixed LCP step: the outer (PGD) solve and the inner (CG) solve.
///
/// outer_tol is a length. At the end of the step (to first order in dt), no two bodies overlap by more than outer_tol,
/// and no two bodies that push on each other are more than outer_tol apart.
///
/// outer_tol must be at least cg_tol in a step with both contacts and bilateral constraints. cg_tol bounds the solve
/// for the bilateral constraint forces y: at the end of the step (to first order in dt), the bilateral constraints'
/// residuals psi + K^{-1} y, each a length or an angle in radians, have L2 norm at most cg_tol. The contacts see that
/// error through the bodies' motion and cannot be resolved more finely than it.
struct MixedLCPConfig {
  unsigned max_outer_iters = 1000;
  double outer_tol = 1e-6;
  unsigned max_cg_iters = 200;
  double cg_tol = 1e-8;
};

/// \brief Result of a mixed LCP step: the unilateral solve's iteration count, final residual, and whether it converged.
///
/// residual is the largest end-of-step overlap between two bodies, or gap between two that push on each other: the
/// length outer_tol bounds.
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

/// \brief Numerics of a mixed SLCP step: its inner mixed LCPs and the sequence of them.
///
/// An iterate is accepted once, at the configuration it moves the rods to, every bilateral row's residual
/// psi + K^-1 y and the displacement its constraint force directions would change by are within length_tol (rows and
/// displacements measured in length) and angle_tol (in radians). Neither can be met below its floor: the inner
/// cg_tol, which each must therefore be at least, and about 1e-8 rad for angles measured near 0 or pi.
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

/// \brief Numerics of a mixed ReLCP step: its inner mixed LCPs and the sequence of them.
///
/// An iterate is accepted once, at the configuration it moves the rods to, no ReLCP pair overlaps by more than
/// length_tol. With relinearize_bilateral, every bilateral row's residual psi + K^-1 y and the displacement its
/// constraint force directions would change the step by must also be within length_tol (rows and displacements measured
/// in length) and angle_tol (in radians). length_tol must be at least the inner outer_tol, the floor of every
/// unilateral row; re-linearizing, both tolerances must be at least the inner cg_tol.
struct MixedReLCPConfig {
  MixedLCPConfig inner_lcp_config;
  unsigned max_iters = 20;
  double length_tol = 1e-6;
  double angle_tol = 1e-6;
  bool relinearize_bilateral = false;
};

/// \brief Result of a mixed ReLCP step: the linearization count, the returned iterate's residual, and convergence.
///
/// residual is the returned iterate's largest acceptance residual over the tolerance it is tested against, so
/// converged is residual <= 1, and accepted_lcp_result is that iterate's inner LCP result. Unconverged, the returned
/// iterate is the first linearization.
struct MixedReLCPResult {
  unsigned num_iters{0};
  double residual{0.0};
  bool converged{false};
  MixedLCPResult accepted_lcp_result{};
};

/// \brief Write a MixedReLCPResult to an ostream.
inline std::ostream& operator<<(std::ostream& os, const MixedReLCPResult& result) {
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
  impl::advance_poses(rods, dt);
}

//! \name Multibody time integrators
//@{

/// \brief The multibody time integrator by linearly implicit steps of size dt, each solved as one mixed LCP over the
/// unilateral and bilateral constraints.
///
/// Each step solves, for unilateral multipliers x >= 0, such as contact force magnitudes, and bilateral multipliers
/// y in R^m:
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
/// leave S undefined. M is the rods' mobility at C^k, mobility_model.make_mobility(rods), and dt M appears in the
/// problem above. S is SPD and only its apply-action is cheap, so it is realized by a matrix-free CG, not an explicit
/// inverse, preconditioned by the preconditioner policy (NoPreconditioner leaves it unpreconditioned).
///
/// A boundary condition belongs in the constraint set rather than being imposed on rods afterwards.
/// A body held by a constraint contributes its reaction to B y, which cancels out of the relaxation's
/// fixed point exactly; a body held by discarding its velocity after the solve leaves that reaction
/// outside B, and the fixed point then carries an error of order dt times the body's mobility.
///
/// The integrator holds handles to its rods and constraints, which share their data, so each step sees every value
/// written through them; a different set of views (a resized family, a new contact list, other rods) needs a new
/// integrator. It holds copies of the mobility model and the preconditioner policy, and the storage its steps write.
/// Reusing the integrator reuses that storage, and each step's solves start from the previous step's solutions and from
/// an estimate the storage holds for each constraint row, so its steps meet the same tolerances as a fresh
/// integrator's, but need not equal them. A family's rows keep their meaning from step to step while its entries do:
/// rewriting a family's entries, or passing storage to another constraint list of the same layout, leaves that estimate
/// misaligned, and steps still meet their tolerances but may take more iterations. Copies of an integrator share its
/// storage, so an integrator and its copies must not be used by more than one in-flight solve_step concurrently; make
/// an integrator per concurrent solve.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
class MixedLCPIntegrator
    : public impl::IntegratorData<
          ExecSpace, Model, Policy,
          impl::MixedLCPWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>, Families...> {
 public:
  using impl::IntegratorData<
      ExecSpace, Model, Policy,
      impl::MixedLCPWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>,
      Families...>::IntegratorData;
};

/// \brief The multibody time integrator by steps of size dt whose bilateral constraints hold at their end, each solved
/// as a sequence of mixed LCPs.
///
/// A step's configuration is the push-forward C(W) = C^k (+) G^k (dt V_ext + M (F_ext + W)) of the constraint wrench
/// W. The unilateral rows are linearized at C^k throughout, so they hold to the accuracy of that linearization. The
/// bilateral rows are linearized afresh at each iterate's configuration C_n, starting from C_0 = C^k: iterate n solves
/// the mixed LCP with B_n = B(C_n) and
///   b_n = psi(C_n) - dt B_n^T M W_{n-1}   (n >= 1),
/// the linear model of psi at C(W) about C_n, and moves the rods to C_{n+1} = C(W_n). Iterate 0 is
/// MixedLCPIntegrator's step, and the rigid rows must be independent as there.
///
/// At a converged iterate, C* = C(W*) satisfies psi(C*) + K^{-1} y* = 0 for every bilateral row and
/// W* = D x* + B(C*) y*: rigid rows hold exactly and compliant ones follow their constitutive law at the end of the
/// step. An iterate is accepted once both hold at C_{n+1} to the configured tolerances. The sequence stops unconverged
/// at max_iters iterates, or earlier once its contraction predicts it cannot converge by then, and the step is then
/// iterate 0. The iteration contracts at a rate of about dt times the mobility times the constraints' curvature
/// weighted by their multipliers, so it converges only where that is below one.
///
/// It holds its rods, constraints, mobility model, preconditioner policy and storage as MixedLCPIntegrator does. Copies
/// of an integrator share its storage, so an integrator and its copies must not be used by more than one in-flight
/// solve_step concurrently; make an integrator per concurrent solve.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
class MixedSLCPIntegrator
    : public impl::IntegratorData<
          ExecSpace, Model, Policy,
          impl::SequenceWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>, Families...> {
 public:
  using impl::IntegratorData<
      ExecSpace, Model, Policy,
      impl::SequenceWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>,
      Families...>::IntegratorData;
};

/// \brief The multibody time integrator by steps of size dt whose ReLCP contacts end without overlap, each solved as a
/// sequence of mixed LCPs.
///
/// A step's configuration is the push-forward C(W) = C^k (+) G^k (dt V_ext + M (F_ext + W)) of the constraint wrench
/// W. Iterate 0 is MixedLCPIntegrator's step, every row linearized at C^k. Iterate n moves the rods to
/// C_{n+1} = C(W_n), where each ReLCP family's rule gives its pairs' separations. Every pair overlapping there by more
/// than length_tol gains a unilateral row of its separation, with
///   q = Phi(C_{n+1}) - dt J^T M W_n,
/// the linear model about C_{n+1} of the separation at C(W). Within a step, rows are neither removed nor re-linearized,
/// so iterate n + 1 constrains all that iterate n did, and other unilateral families keep their rows from C^k. With
/// relinearize_bilateral the bilateral rows are linearized afresh at C_{n+1} as MixedSLCPIntegrator's are; otherwise
/// they stay at C^k, as MixedLCPIntegrator's do.
///
/// An iterate is accepted once no ReLCP pair overlaps at C_{n+1} by more than length_tol and, re-linearizing, the
/// bilateral rows hold there. The sequence stops unconverged at max_iters iterates, and the step is then iterate 0.
/// A new row's separation is linearized about C_{n+1}, which must lie where its rule's separation is smooth: a step
/// that moves a pair from contact to an overlap of about the rods' radius can leave its rows without a solution, and
/// the unilateral solve then fails. A step generating more rows than its storage holds grows the storage, which later
/// steps keep.
///
/// It holds its rods, constraints, mobility model, preconditioner policy and storage as MixedLCPIntegrator does. Copies
/// of an integrator share its storage, so an integrator and its copies must not be used by more than one in-flight
/// solve_step concurrently; make an integrator per concurrent solve.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
class MixedReLCPIntegrator
    : public impl::IntegratorData<
          ExecSpace, Model, Policy,
          impl::SequenceWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>, Families...> {
 public:
  using impl::IntegratorData<
      ExecSpace, Model, Policy,
      impl::SequenceWorkspace<ExecSpace, impl::mobility_of_t<Model, ExecSpace>, Policy, Families...>,
      Families...>::IntegratorData;
};

/// \brief The mixed LCP integrator of rods under constraints, moving with mobility_model's mobility by steps of dt,
/// with the Schur complement preconditioned by preconditioner_policy, in storage of its own.
template <typename ExecSpace, typename Model, typename Policy = NoPreconditioner, typename... Families>
MixedLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_lcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy = Policy{}) {
  return MixedLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                   preconditioner_policy);
}

/// \brief The mixed LCP integrator above, in workspace's storage if it fits rods and constraints: the same rod count
/// and row count for every family.
///
/// Otherwise the integrator has storage of its own. Reused storage keeps its solutions and estimates, from which the
/// next step's solves start, and its preconditioner op, made from the policy it was built with.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_lcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy,
    typename MixedLCPIntegrator<ExecSpace, Model, Policy, Families...>::workspace_t workspace) {
  return MixedLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                   preconditioner_policy, std::move(workspace));
}

/// \brief The mixed SLCP integrator of rods under constraints, moving with mobility_model's mobility by steps of dt,
/// with each Schur complement preconditioned by preconditioner_policy, in storage of its own.
template <typename ExecSpace, typename Model, typename Policy = NoPreconditioner, typename... Families>
MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_slcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy = Policy{}) {
  return MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                    preconditioner_policy);
}

/// \brief The mixed SLCP integrator above, in workspace's storage if it fits rods and constraints, as for
/// make_mixed_lcp_integrator.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_slcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy,
    typename MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...>::workspace_t workspace) {
  return MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                    preconditioner_policy, std::move(workspace));
}

/// \brief The mixed ReLCP integrator of rods under constraints, moving with mobility_model's mobility by steps of dt,
/// with each Schur complement preconditioned by preconditioner_policy, in storage of its own.
template <typename ExecSpace, typename Model, typename Policy = NoPreconditioner, typename... Families>
MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_relcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy = Policy{}) {
  return MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                     preconditioner_policy);
}

/// \brief The mixed ReLCP integrator above, in workspace's storage if it fits rods and constraints, as for
/// make_mixed_lcp_integrator.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...> make_mixed_relcp_integrator(
    const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints, const Model& mobility_model,
    double dt, const Policy& preconditioner_policy,
    typename MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...>::workspace_t workspace) {
  return MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...>(rods, constraints, mobility_model, dt,
                                                                     preconditioner_policy, std::move(workspace));
}

/// \brief One step of integrator at its rods' current configuration C^k.
///
/// On entry the rods' force/torque is the external load F_ext and velocity/omega the imposed velocity V_ext, and every
/// unilateral family's lambda holds the multipliers the step's unilateral solve starts from: the last step's, which
/// each step writes there, or zero for a cold start. On exit force/torque is F_ext + D x* + B y*, velocity/omega is
/// U_free + M (D x* + B y*), every family's lambda holds its multipliers, and the rods have not moved.
/// advance_rods(rods, dt) performs the consistent time integration, after which each constraint holds to first order in
/// dt.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedLCPResult solve_step(const MixedLCPIntegrator<ExecSpace, Model, Policy, Families...>& integrator,
                          const MixedLCPConfig& cfg) {
  const RodViews<ExecSpace>& rods = integrator.rods();
  const ConstraintSet<Families...>& constraints = integrator.constraints();
  const double dt = integrator.dt();
  auto& workspace = integrator.workspace();

  const PGDConfig<double> pgd_cfg{cfg.max_outer_iters, cfg.outer_tol};
  const CGConfig<double> cg_cfg{cfg.max_cg_iters, cfg.cg_tol};
  const auto step = impl::make_step_data(rods, constraints, integrator.mobility_model().make_mobility(rods), dt,
                                         pgd_cfg, cg_cfg, workspace.linearization);
  if (step.index_map.num_unilateral == 0 && step.index_map.num_bilateral == 0) {
    Kokkos::deep_copy(rods.velocity_omega_view(), step.u_free);
    return MixedLCPResult{0, 0.0, 0.0 <= cfg.outer_tol};
  }
  MUNDY_THROW_ASSERT(
      step.index_map.num_unilateral == 0 || step.index_map.num_bilateral == 0 || cfg.outer_tol >= cfg.cg_tol,
      std::invalid_argument, "mbody::solve_step: outer_tol must be at least cg_tol.");

  impl::linearize(step, workspace.linearization, rods, constraints);
  impl::read_unilateral_multipliers<ExecSpace>(constraints, step.index_map, workspace.x_start);
  impl::solve_linearization(step, workspace.linearization, impl::Displacement<ExecSpace>{step.u_free, dt},
                            workspace.x_start, workspace.linearized);
  impl::write_step(rods, constraints, step.index_map, workspace.linearized);
  const PGDResult<double>& result = workspace.linearized.result;
  return MixedLCPResult{result.num_iters, result.residual, result.converged};
}

/// \brief One step of integrator at its rods' current configuration C^k.
///
/// On entry the rods' force/torque is the external load F_ext and velocity/omega the imposed velocity V_ext, and every
/// unilateral family's lambda holds the multipliers the first iterate's unilateral solve starts from, as for a mixed
/// LCP step. On exit force/torque is F_ext + W, velocity/omega is U_free + M W, every family's lambda holds its
/// multipliers, all of the returned iterate, and the rods have not moved. advance_rods(rods, dt) performs the
/// consistent time integration, which for a converged iterate reaches the configuration at which it was accepted.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedSLCPResult solve_step(const MixedSLCPIntegrator<ExecSpace, Model, Policy, Families...>& integrator,
                           const MixedSLCPConfig& cfg) {
  MUNDY_THROW_REQUIRE(cfg.max_iters >= 1, std::invalid_argument, "mbody::solve_step: max_iters must be >= 1.");
  MUNDY_THROW_REQUIRE(cfg.length_tol > 0.0 && cfg.angle_tol > 0.0, std::invalid_argument,
                      "mbody::solve_step: length_tol and angle_tol must be positive.");
  const RodViews<ExecSpace>& rods = integrator.rods();
  auto& workspace = integrator.workspace();
  const MixedLCPConfig& lcp_cfg = cfg.inner_lcp_config;
  const impl::ConstraintIndexMap<Families...>& index_map = workspace.lcp.linearization.index_map();
  MUNDY_THROW_ASSERT(
      index_map.num_unilateral == 0 || index_map.num_bilateral == 0 || lcp_cfg.outer_tol >= lcp_cfg.cg_tol,
      std::invalid_argument, "mbody::solve_step: outer_tol must be at least cg_tol.");
  MUNDY_THROW_ASSERT(
      index_map.num_bilateral == 0 || (cfg.length_tol >= lcp_cfg.cg_tol && cfg.angle_tol >= lcp_cfg.cg_tol),
      std::invalid_argument, "mbody::solve_step: length_tol and angle_tol must be at least cg_tol.");

  const impl::SequenceResult result = impl::solve_sequence(
      rods, integrator.constraints(), integrator.mobility_model().make_mobility(rods), integrator.dt(),
      PGDConfig<double>{lcp_cfg.max_outer_iters, lcp_cfg.outer_tol},
      CGConfig<double>{lcp_cfg.max_cg_iters, lcp_cfg.cg_tol}, cfg.max_iters, /*stop_when_hopeless=*/true, workspace,
      impl::RelinearizeBilateral{cfg.length_tol, cfg.angle_tol});
  const PGDResult<double>& accepted = result.accepted_lcp_result;
  return MixedSLCPResult{result.num_iters, result.residual, result.converged,
                         MixedLCPResult{accepted.num_iters, accepted.residual, accepted.converged}};
}

/// \brief One step of integrator at its rods' current configuration C^k.
///
/// On entry and exit the rods and constraints are as for a mixed SLCP step: every family's lambda holds its multipliers
/// of the returned iterate, which for a ReLCP family are those of its pairs' rows from the start of the step. The
/// returned iterate's constraint force includes every row it solved. advance_rods(rods, dt) performs the consistent
/// time integration, which for a converged iterate reaches the configuration at which it was accepted.
template <typename ExecSpace, typename Model, typename Policy, typename... Families>
MixedReLCPResult solve_step(const MixedReLCPIntegrator<ExecSpace, Model, Policy, Families...>& integrator,
                            const MixedReLCPConfig& cfg) {
  MUNDY_THROW_REQUIRE(cfg.max_iters >= 1, std::invalid_argument, "mbody::solve_step: max_iters must be >= 1.");
  MUNDY_THROW_REQUIRE(cfg.length_tol > 0.0 && cfg.angle_tol > 0.0, std::invalid_argument,
                      "mbody::solve_step: length_tol and angle_tol must be positive.");
  const RodViews<ExecSpace>& rods = integrator.rods();
  auto& workspace = integrator.workspace();
  const MixedLCPConfig& lcp_cfg = cfg.inner_lcp_config;
  const impl::ConstraintIndexMap<Families...>& index_map = workspace.lcp.linearization.index_map();
  MUNDY_THROW_ASSERT(
      index_map.num_unilateral == 0 || index_map.num_bilateral == 0 || lcp_cfg.outer_tol >= lcp_cfg.cg_tol,
      std::invalid_argument, "mbody::solve_step: outer_tol must be at least cg_tol.");
  MUNDY_THROW_ASSERT(index_map.num_unilateral == 0 || cfg.length_tol >= lcp_cfg.outer_tol, std::invalid_argument,
                     "mbody::solve_step: length_tol must be at least outer_tol.");
  MUNDY_THROW_ASSERT(!cfg.relinearize_bilateral || index_map.num_bilateral == 0 ||
                         (cfg.length_tol >= lcp_cfg.cg_tol && cfg.angle_tol >= lcp_cfg.cg_tol),
                     std::invalid_argument, "mbody::solve_step: length_tol and angle_tol must be at least cg_tol.");

  const PGDConfig<double> pgd_cfg{lcp_cfg.max_outer_iters, lcp_cfg.outer_tol};
  const CGConfig<double> cg_cfg{lcp_cfg.max_cg_iters, lcp_cfg.cg_tol};
  const auto mobility = integrator.mobility_model().make_mobility(rods);
  const impl::AugmentContacts augment_contacts{cfg.length_tol};
  const impl::SequenceResult result =
      cfg.relinearize_bilateral
          ? impl::solve_sequence(rods, integrator.constraints(), mobility, integrator.dt(), pgd_cfg, cg_cfg,
                                 cfg.max_iters, /*stop_when_hopeless=*/false, workspace,
                                 impl::RelinearizeBilateral{cfg.length_tol, cfg.angle_tol}, augment_contacts)
          : impl::solve_sequence(rods, integrator.constraints(), mobility, integrator.dt(), pgd_cfg, cg_cfg,
                                 cfg.max_iters, /*stop_when_hopeless=*/false, workspace, augment_contacts);
  const PGDResult<double>& accepted = result.accepted_lcp_result;
  return MixedReLCPResult{result.num_iters, result.residual, result.converged,
                          MixedLCPResult{accepted.num_iters, accepted.residual, accepted.converged}};
}
//@}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODY_HPP_
