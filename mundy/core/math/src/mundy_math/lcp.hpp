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

#ifndef MUNDY_MATH_LCP_HPP_
#define MUNDY_MATH_LCP_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/convex_spaces.hpp>  // for mundy::LowerBoundSpace
#include <mundy_math/cqpp.hpp>        // for mundy::CQPP, mundy::solve_cqpp -- used ONLY by the reduction method below
#include <mundy_math/linear_ops.hpp>  // for mundy::make_quadratic_form
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::{workspace_for_t, make_workspace, vector_value_type}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>  // for mundy::storage
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

/// \brief Linear complementarity problem (LCP) formulation
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x  s.t. x in R^n, x >= 0
template <typename Backend, typename LinearOp, typename QVector,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class LCP {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using workspace_t = Workspace;
  using value_type = impl::vector_value_type<q_vector_t>;

  LCP(Backend, LinearOp&& A, QVector&& q)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), workspace_(impl::make_workspace(A_.get())) {
  }

  LCP(Backend, LinearOp&& A, QVector&& q, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), workspace_(std::move(workspace)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  q_vector_storage_t q_;
  mutable workspace_t workspace_;
};

/// \brief Reduce an LCP to the equivalent non-negativity-constrained CQPP (x >= 0).
template <class Backend, class LinearOp, class QVector>
KOKKOS_FUNCTION auto to_cqpp(const LCP<Backend, LinearOp, QVector>& P) {
  using value_type = typename LCP<Backend, LinearOp, QVector>::value_type;
  static constexpr LowerBoundSpace Rn_plus{static_cast<value_type>(0)};
  auto A_copy = P.A();
  auto q_copy = P.q();
  return CQPP(P.backend(), std::move(A_copy), std::move(q_copy), Rn_plus, P.workspace());
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <typename Backend, typename LinearOp, typename QVector>
LCP(Backend, LinearOp&&, QVector&&) -> LCP<Backend, LinearOp, QVector>;

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
LCP(Backend, LinearOp&&, QVector&&, const Workspace&) -> LCP<Backend, LinearOp, QVector, Workspace>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <typename Backend, typename LinearOp, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q) {
  return LCP(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  return LCP(Backend{}, std::move(A), std::forward<QVector>(q));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                     UVector&& u) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  auto workspace = A.make_workspace(std::forward<FVector>(f), std::forward<UVector>(u));
  return LCP(Backend{}, std::move(A), std::forward<QVector>(q), std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q, Workspace&& workspace) {
  return LCP(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q), std::forward<Workspace>(workspace));
}
//@}

/// \brief Solve a linear complementarity problem (LCP): `0 <= A x + q _|_ x >= 0`.
///
/// Methods for an LCP (this overload implements method 1):
///   -# Reduction to CQPP + a CQPP strategy -- the only method today. `to_cqpp` maps the LCP onto a
///      non-negativity-constrained CQPP (the `x >= 0` orthant appears only at reduction; the LCP type
///      itself carries no convex space), then a CQPP method such as PGD solves it via \ref solve_cqpp.
///      This is why the header includes `cqpp.hpp`; the return type is that strategy's result
///      (\ref PGDResult for PGD).
///   -# Lemke pivoting (planned) -- native, operates on the LCP directly, no `cqpp.hpp` dependency.
///   -# Interior-point (planned) -- native, no `cqpp.hpp` dependency.
///
/// \param prob The linear complementarity problem to solve.
/// \param strat The CQPP solution strategy to apply to the reduced problem.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Problem& p) {
  { to_cqpp(p) };
})
KOKKOS_FUNCTION auto solve_lcp(const Problem& prob, const Strategy& strat, State& state) {
  auto cqpp_prob = to_cqpp(prob);
  return solve_cqpp(cqpp_prob, strat, state);
}

}  // namespace mundy

#endif  // MUNDY_MATH_LCP_HPP_
