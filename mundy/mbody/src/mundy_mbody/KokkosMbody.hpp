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

#ifndef MUNDY_MESH_PERFORMANCE_TESTS_KOKKOS_MBODY_HPP_
#define MUNDY_MESH_PERFORMANCE_TESTS_KOKKOS_MBODY_HPP_

// Mundy
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/cqpp.hpp>
#include <mundy_math/linear_ops.hpp>
#include <mundy_math/linear_system.hpp>

#include "KokkosMbodyImpl.hpp"
#include "KokkosMbodyTypes.hpp"

namespace mundy {

namespace mbody {

/// \brief Fully implicit mixed complementarity + bilateral-constraint time integration of a multibody system.
///
/// Achieved via a mixed constrained quadratic programming (MCQPP) solver over a set of rods and the
/// constraints acting on them.
///
/// Solves, for contact force magnitudes x >= 0 and bilateral multipliers y in R^m:
///   x*, y* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// via the Schur complement S := (B^T M B + K^{-1})^{-1}:
///   H := D^T M D - D^T M B S B^T M D,  g := q - D^T M B S b
///   x* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y* = -S (b + B^T M D x*)
///
/// D/D^T map contact force magnitudes <-> generalized rod force/torque; B/B^T do the same for the
/// concatenated bilateral multipliers, whose packing order the constraint index map fixes. K^{-1} is
/// the per-constraint compliance, zero for a rigidly held one. M is the local-drag rod mobility. S is
/// SPD and only its apply-action is cheap, so it is realized by a matrix-free CG, not an explicit
/// inverse.
///
/// A boundary condition belongs in the constraint set rather than being imposed on rods afterwards.
/// A body held by a constraint contributes its reaction to B y, which cancels out of the relaxation's
/// fixed point exactly; a body held by discarding its velocity after the solve leaves that reaction
/// outside B, and the fixed point then carries an error of order dt times the body's mobility.
template <typename ExecSpace>
PGDResult<double> solve(const RodViews<ExecSpace>& rods, const ConstraintSet<ExecSpace>& constraints,
                        const SolveConfig& cfg) {
  using memory_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, memory_space>;
  using backend_t = KokkosBackend<ExecSpace>;

  const auto& lin_springs = constraints.linear_springs;
  const auto& ang_springs = constraints.angular_springs;
  const auto& triple_springs = constraints.triple_springs;
  const auto& fixed_positions = constraints.fixed_positions;
  const auto& fixed_poses = constraints.fixed_poses;
  const auto& contacts = constraints.contacts;

  const size_t num_rods = rods.size();
  const size_t num_contacts = contacts.size();
  const ConstraintIndexMap index_map = make_constraint_index_map(constraints);

  MUNDY_THROW_ASSERT(impl::count_doubly_anchored_rods(constraints, num_rods) == 0, std::invalid_argument,
                     "mbody::solve: a rod carries more than one fixed-position or fixed-pose anchor, leaving the "
                     "bilateral block rank deficient.");

  // Contact (x-block) and spring (y-block, linear + axis-angular + triple-point-angular
  // concatenated) geometry.
  view_t sep0("sep0", num_contacts);
  const impl::PairGeometry<ExecSpace> contact_geo = impl::compute_contact_geometry(rods, contacts, sep0);

  // One allocation for the whole y-block; each family's kernel fills its own rows of it.
  view_t b0("b0", index_map.total);
  const impl::PairGeometry<ExecSpace> lin_geo =
      impl::compute_linear_spring_geometry(rods, lin_springs, impl::subrange(b0, index_map.linear_springs));
  const impl::PairGeometry<ExecSpace> ang_geo =
      impl::compute_angular_spring_geometry(rods, ang_springs, impl::subrange(b0, index_map.angular_springs));
  const impl::PairGeometry<ExecSpace> pair_spring_geo = impl::concat_pair_geometry(lin_geo, ang_geo);
  const impl::TripleGeometry<ExecSpace> triple_geo = impl::compute_triple_point_angular_spring_geometry(
      rods, triple_springs, impl::subrange(b0, index_map.triple_springs));
  const impl::SingleGeometry<ExecSpace> fixed_position_geo = impl::compute_fixed_position_geometry(
      rods, fixed_positions, impl::subrange(b0, index_map.fixed_positions));
  const impl::SingleGeometry<ExecSpace> fixed_pose_geo =
      impl::compute_fixed_pose_geometry(rods, fixed_poses, impl::subrange(b0, index_map.fixed_poses));
  const impl::SingleGeometry<ExecSpace> single_geo =
      impl::concat_single_geometry(fixed_position_geo, fixed_pose_geo);
  const view_t kinv_diag = impl::concat_vectors(impl::reciprocal<ExecSpace>(lin_springs.spring_constant_view()),
                                                impl::reciprocal<ExecSpace>(ang_springs.spring_constant_view()),
                                                impl::reciprocal<ExecSpace>(triple_springs.spring_constant_view()),
                                                fixed_positions.compliance_view(), fixed_poses.compliance_view());

  // Operators.
  const impl::PairForceOp<ExecSpace> D(contact_geo, num_rods);
  const impl::PairForceOpT<ExecSpace> DT(contact_geo, num_rods);
  const impl::PairForceOp<ExecSpace> B_pairs(pair_spring_geo, num_rods);
  const impl::PairForceOpT<ExecSpace> BT_pairs(pair_spring_geo, num_rods);
  const impl::TripleForceOp<ExecSpace> B_triple(triple_geo, num_rods);
  const impl::TripleForceOpT<ExecSpace> BT_triple(triple_geo, num_rods);
  const impl::SingleForceOp<ExecSpace> B_single(single_geo, num_rods);
  const impl::SingleForceOpT<ExecSpace> BT_single(single_geo, num_rods);
  const auto B_springs = make_concat_domain_op<backend_t>(B_pairs, B_triple);
  const auto BT_springs = make_concat_range_op<backend_t>(BT_pairs, BT_triple);
  const auto B = make_concat_domain_op<backend_t>(B_springs, B_single);
  const auto BT = make_concat_range_op<backend_t>(BT_springs, BT_single);
  const impl::LocalDragMobilityOp<ExecSpace> M(cfg.viscosity, rods);
  // The mixed CQPP's "M" is dt * mobility: it maps a constraint force to the displacement it causes
  // over the step, not to a velocity. The other uses of M below (free-velocity prediction, final
  // write-back) want the instantaneous force->velocity relation and stay on raw M.
  const auto M_dt = make_scaled_op<backend_t>(cfg.dt, M);

  // vel_omega := V_ext + M F_ext, in rods' own storage: force_torque is F_ext (read-only below) and
  // velocity_omega is V_ext on entry, updated in place to the pre-constraint velocity/omega.
  view_t force_torque = rods.force_torque_view();
  view_t vel_omega = rods.velocity_omega_view();

  view_t m_force_torque_ext("m_force_torque_ext", 6 * num_rods);
  M.apply(force_torque, m_force_torque_ext);
  backend_t::axpby(1.0, m_force_torque_ext, 1.0, vel_omega);

  // q := sep0 + dt * D^T vel_omega,  b := b0 + dt * B^T vel_omega.
  view_t q("q", num_contacts);
  DT.apply(vel_omega, q);
  backend_t::axpby(cfg.dt, q, 1.0, sep0);
  view_t& q_vec = sep0;

  view_t b_rate("b_rate", index_map.total);
  BT.apply(vel_omega, b_rate);
  backend_t::axpby(cfg.dt, b_rate, 1.0, b0);
  view_t& b_vec = b0;

  // Schur complement S := (B^T M B + K^{-1})^{-1}, realized via matrix-free CG.
  const auto btmb_plus_kinv =
      make_sum_op<backend_t>(make_quadratic_form<backend_t>(BT, M_dt, B), make_diagonal_op<backend_t>(kinv_diag));
  const CGConfig<double> cg_cfg{cfg.max_cg_iters, cfg.cg_tol};
  const auto S = make_cg_inv_op<backend_t>(btmb_plus_kinv, cg_cfg);

  // Reduced CQPP + PGD solve for x* (contact force magnitudes).
  const auto mcqpp =
      make_mixed_cqpp<backend_t>(DT, M_dt, D, q_vec, B, S, BT, b_vec, LowerBoundSpace<double>{.lower_bound = 0.0});

  view_t x("x", num_contacts);
  view_t grad("grad", num_contacts);
  view_t x_tmp("x_tmp", num_contacts);
  view_t grad_tmp("grad_tmp", num_contacts);
  Kokkos::deep_copy(x, 0.0);

  auto pgd = make_pgd_solution_strategy(PGDConfig<double>{.max_iters = cfg.max_outer_iters, .tol = cfg.outer_tol});
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);
  const PGDResult<double> result = solve_mixed_cqpp(mcqpp, pgd, pgd_state);
  MUNDY_THROW_REQUIRE(result.converged, std::runtime_error, "mbody::solve: outer PGD solve failed to converge.");

  // Recover y* = -S (b + B^T M D x*) -- solve_mixed_cqpp only returns the x-block result.
  view_t Dx("Dx", 6 * num_rods);
  D.apply(x, Dx);
  view_t MDx("MDx", 6 * num_rods);
  M_dt.apply(Dx, MDx);
  view_t y_rhs("y_rhs", index_map.total);
  BT.apply(MDx, y_rhs);
  backend_t::axpby(1.0, b_vec, 1.0, y_rhs);  // y_rhs = b + B^T M D x*

  view_t y("y", index_map.total);
  S.apply(y_rhs, y);
  backend_t::axpby(-1.0, y, 0.0, y);  // y := -y

  // Apply x*, y* back: force/torque += D x* + B y*; velocity/omega += M (D x* + B y*).
  view_t By("By", 6 * num_rods);
  B.apply(y, By);
  view_t total_force_torque("total_force_torque", 6 * num_rods);
  backend_t::axpby(1.0, Dx, 0.0, total_force_torque);
  backend_t::axpby(1.0, By, 1.0, total_force_torque);

  view_t m_total("m_total", 6 * num_rods);
  M.apply(total_force_torque, m_total);

  // force_torque/vel_omega alias rods' own storage, so these mutate rods' force/torque and
  // velocity/omega in place.
  backend_t::axpby(1.0, total_force_torque, 1.0, force_torque);
  backend_t::axpby(1.0, m_total, 1.0, vel_omega);

  Kokkos::deep_copy(contacts.lambda_view(), x);
  Kokkos::deep_copy(lin_springs.lambda_view(), impl::subrange(y, index_map.linear_springs));
  Kokkos::deep_copy(ang_springs.lambda_view(), impl::subrange(y, index_map.angular_springs));
  Kokkos::deep_copy(triple_springs.lambda_view(), impl::subrange(y, index_map.triple_springs));
  Kokkos::deep_copy(fixed_positions.lambda_view(), impl::subrange(y, index_map.fixed_positions));
  Kokkos::deep_copy(fixed_poses.lambda_view(), impl::subrange(y, index_map.fixed_poses));

  return result;
}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MESH_PERFORMANCE_TESTS_KOKKOS_MBODY_HPP_
