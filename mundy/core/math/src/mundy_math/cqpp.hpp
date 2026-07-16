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

#ifndef MUNDY_MATH_CQPP_HPP_
#define MUNDY_MATH_CQPP_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <ostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/Matrix.hpp>     // for mundy::is_matrix_v
#include <mundy_math/Tolerance.hpp>  // for mundy::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::Vector
#include <mundy_math/cmath.hpp>
#include <mundy_math/convex_spaces.hpp>  // for mundy::ValidConvexSpace and the *Space types
#include <mundy_math/linear_ops.hpp>     // for mundy::{QuadraticFormOp, MixedReducedOp, CongruentMixedReducedOp, ...}
#include <mundy_math/pgd.hpp>        // for the PGD method (PGDStrategy, PGDResult, ...) -- see "Methods for CQPP" below
#include <mundy_math/residuals.hpp>  // for the projected-gradient residual policies
#include <mundy_math/solver_backends.hpp>  // for mundy::{Backend, Workspace, concepts, ...}
#include <mundy_utils/reference_wrapper.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>  // for mundy::storage, mundy::store
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Problems + state
//@{
// Every Problem type below pairs a LinearOp with a mutable workspace (its cached scratch state for evaluating
// that operator -- see Backend::apply and the operator's own Workspace type). That workspace is mutated during
// solve_cqpp/solve_mixed_cqpp, so a single Problem instance must not be used by more than one in-flight solve
// concurrently; construct a fresh Problem per solve if solves may run concurrently.

/// \brief Constrained quadratic programming problem (CQPP) formulation
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOp, typename QVector, ValidConvexSpace ConvexSpace,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class CQPP {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using vector_t = typename vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using workspace_t = Workspace;
  using value_type = impl::vector_value_type<vector_t>;

  CQPP(Backend, LinearOp&& A, QVector&& q, const space_t& space)
      : A_(std::forward<LinearOp>(A)),
        q_(std::forward<QVector>(q)),
        space_(space),
        workspace_(impl::make_workspace(A_.get())) {
  }

  CQPP(Backend, LinearOp&& A, QVector&& q, const space_t& space, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), space_(space), workspace_(std::move(workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating A (see Backend::apply and Op::Workspace types).
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  vector_storage_t q_;
  space_t space_;
  mutable workspace_t workspace_;
};

/// \brief Mixed constrained quadratic programming problem (MCQPP) formulation
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D,
/// q, BT, S, B, b).
template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          ValidConvexSpace ConvexSpace, typename WorkspaceA = impl::workspace_for_t<std::remove_cvref_t<LinearOpA>>,
          typename WorkspaceL = impl::workspace_for_t<std::remove_cvref_t<LinearOpL>>>
class MCQPP {
 public:
  using backend_t = Backend;
  using linear_op_storage_a_t = ::mundy::storage<LinearOpA>;
  using linear_op_storage_l_t = ::mundy::storage<LinearOpL>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using f_vector_storage_t = ::mundy::storage<FVector>;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using f_vector_t = typename f_vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using a_workspace_t = WorkspaceA;
  using l_workspace_t = WorkspaceL;
  using value_type = impl::vector_value_type<q_vector_t>;

  MCQPP(Backend, LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b, const space_t& space)
      : A_(std::forward<LinearOpA>(A)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        a_workspace_(impl::make_workspace(A_.get())),
        l_workspace_(impl::make_workspace(L_.get())) {
  }

  MCQPP(Backend, LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b, const space_t& space,
        a_workspace_t a_workspace, l_workspace_t l_workspace)
      : A_(std::forward<LinearOpA>(A)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        a_workspace_(std::move(a_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return f_b_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating A and L (see Backend::apply and Op::Workspace
  /// types).
  KOKKOS_INLINE_FUNCTION a_workspace_t& a_workspace() const { return a_workspace_; }
  KOKKOS_INLINE_FUNCTION l_workspace_t& l_workspace() const { return l_workspace_; }
  // clang-format on

 private:
  linear_op_storage_a_t A_;
  q_vector_storage_t q_;
  linear_op_storage_l_t L_;
  f_vector_storage_t f_b_;
  space_t space_;
  mutable a_workspace_t a_workspace_;
  mutable l_workspace_t l_workspace_;
};

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, ValidConvexSpace ConvexSpace,
          typename WorkspaceDT = impl::workspace_for_t<std::remove_cvref_t<LinearOpDT>>,
          typename WorkspaceM = impl::workspace_for_t<std::remove_cvref_t<LinearOpM>>,
          typename WorkspaceD = impl::workspace_for_t<std::remove_cvref_t<LinearOpD>>,
          typename WorkspaceL = impl::workspace_for_t<std::remove_cvref_t<LinearOpL>>>
class CongruentMCQPP {
 public:
  using backend_t = Backend;
  using linear_op_storage_dt_t = ::mundy::storage<LinearOpDT>;
  using linear_op_storage_m_t = ::mundy::storage<LinearOpM>;
  using linear_op_storage_d_t = ::mundy::storage<LinearOpD>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_storage_l_t = ::mundy::storage<LinearOpL>;
  using f_vector_storage_t = ::mundy::storage<FVector>;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using dt_workspace_t = WorkspaceDT;
  using m_workspace_t = WorkspaceM;
  using d_workspace_t = WorkspaceD;
  using l_workspace_t = WorkspaceL;
  using value_type = impl::vector_value_type<q_vector_t>;

  CongruentMCQPP(Backend, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L, FVector&& f_b,
                 const space_t& space)
      : DT_(std::forward<LinearOpDT>(DT)),
        M_(std::forward<LinearOpM>(M)),
        D_(std::forward<LinearOpD>(D)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        dt_workspace_(impl::make_workspace(DT_.get())),
        m_workspace_(impl::make_workspace(M_.get())),
        d_workspace_(impl::make_workspace(D_.get())),
        l_workspace_(impl::make_workspace(L_.get())) {
  }

  CongruentMCQPP(Backend, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L, FVector&& f_b,
                 const space_t& space, dt_workspace_t dt_workspace, m_workspace_t m_workspace,
                 d_workspace_t d_workspace, l_workspace_t l_workspace)
      : DT_(std::forward<LinearOpDT>(DT)),
        M_(std::forward<LinearOpM>(M)),
        D_(std::forward<LinearOpD>(D)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        dt_workspace_(std::move(dt_workspace)),
        m_workspace_(std::move(m_workspace)),
        d_workspace_(std::move(d_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return DT_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return M_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return D_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return f_b_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating DT, M, D, and L (see Backend::apply and Op::Workspace
  /// types).
  KOKKOS_INLINE_FUNCTION dt_workspace_t& dt_workspace() const { return dt_workspace_; }
  KOKKOS_INLINE_FUNCTION m_workspace_t& m_workspace() const { return m_workspace_; }
  KOKKOS_INLINE_FUNCTION d_workspace_t& d_workspace() const { return d_workspace_; }
  KOKKOS_INLINE_FUNCTION l_workspace_t& l_workspace() const { return l_workspace_; }
  // clang-format on

 private:
  linear_op_storage_dt_t DT_;
  linear_op_storage_m_t M_;
  linear_op_storage_d_t D_;
  q_vector_storage_t q_;
  linear_op_storage_l_t L_;
  f_vector_storage_t f_b_;
  space_t space_;
  mutable dt_workspace_t dt_workspace_;
  mutable m_workspace_t m_workspace_;
  mutable d_workspace_t d_workspace_;
  mutable l_workspace_t l_workspace_;
};

template <class Backend, class LinearOpA, class QVector, class LinearOpL, class FVector, class ConvexSpace,
          class AWorkspace, class LWorkspace>
KOKKOS_FUNCTION auto to_cqpp(
    const MCQPP<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace, AWorkspace, LWorkspace>& P) {
  // get the type of P no ref
  using value_type = std::remove_reference_t<decltype(P)>::value_type;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  // H owns an independent copy of A and L, so it stays valid after P is destroyed.
  auto A_copy = P.A();
  auto L_copy = P.L();
  auto H = MixedReducedOp(backend_t{}, std::move(A_copy), std::move(L_copy));

  auto g = backend_t::make_vector_like(P.q());
  auto a_workspace = P.a_workspace();
  auto l_workspace = P.l_workspace();

  backend_t::apply(P.A(), P.f_b(), g, a_workspace);                                     // g = A f_b
  backend_t::axpby(static_cast<value_type>(1), P.q(), static_cast<value_type>(-1), g);  // g = q - A f_b

  auto ax = backend_t::make_range_vector(P.A());
  auto lax = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(ax), std::move(lax), std::move(a_workspace), std::move(l_workspace));
  return CQPP(backend_t{}, std::move(H), std::move(g), P.space(), workspace);
}

template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD, class QVector, class LinearOpL,
          class FVector, class ConvexSpace, class DTWorkspace, class MWorkspace, class DWorkspace, class LWorkspace>
KOKKOS_FUNCTION auto to_cqpp(
    const CongruentMCQPP<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace,
                         DTWorkspace, MWorkspace, DWorkspace, LWorkspace>& P) {
  using value_type = std::remove_reference_t<decltype(P)>::value_type;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  // H owns an independent copy of DT, M, D, and L, so it stays valid after P is destroyed.
  auto DT_copy = P.DT();
  auto M_copy = P.M();
  auto D_copy = P.D();
  auto L_copy = P.L();
  auto H =
      CongruentMixedReducedOp(backend_t{}, std::move(DT_copy), std::move(M_copy), std::move(D_copy), std::move(L_copy));

  auto g = backend_t::make_vector_like(P.q());
  auto m_f_b = backend_t::make_range_vector(P.M());
  auto dt_workspace = P.dt_workspace();
  auto m_workspace = P.m_workspace();
  auto d_workspace = P.d_workspace();
  auto l_workspace = P.l_workspace();

  backend_t::apply(P.M(), P.f_b(), m_f_b, m_workspace);
  backend_t::apply(P.DT(), m_f_b, g, dt_workspace);                                     // g = D^T M f_b
  backend_t::axpby(static_cast<value_type>(1), P.q(), static_cast<value_type>(-1), g);  // g = q - D^T M f_b

  auto dx = backend_t::make_range_vector(P.D());
  auto mdx = backend_t::make_range_vector(P.M());
  auto lmdx = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(dx), std::move(mdx), std::move(lmdx), std::move(dt_workspace),
                                    std::move(m_workspace), std::move(d_workspace), std::move(l_workspace));
  return CQPP(backend_t{}, std::move(H), std::move(g), P.space(), workspace);
}
//@}

/// \brief The strategy interface the solve_cqpp driver requires: initialize/iterate/done/result. Any method that
/// solves a CQPP (PGD today, see pgd.hpp) satisfies this. Structurally generic across iterative methods.
template <class Strategy, class Problem, class State>
concept CQPPSolverStrategy = requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

/// \brief Deduction guide for CQPP
template <typename Backend, typename LinearOp, typename QVector, ValidConvexSpace ConvexSpace>
CQPP(Backend, LinearOp&&, QVector&&, const ConvexSpace&) -> CQPP<Backend, LinearOp, QVector, ConvexSpace>;

template <typename Backend, typename LinearOp, typename QVector, ValidConvexSpace ConvexSpace, typename Workspace>
CQPP(Backend, LinearOp&&, QVector&&, const ConvexSpace&, const Workspace&)
    -> CQPP<Backend, LinearOp, QVector, ConvexSpace, Workspace>;

/// \brief Deduction guide for MCQPP
template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          ValidConvexSpace ConvexSpace>
MCQPP(Backend, LinearOpA&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&)
    -> MCQPP<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
MCQPP(Backend, LinearOpA&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&, const AWorkspace&, const LWorkspace&)
    -> MCQPP<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace, AWorkspace, LWorkspace>;

/// \brief Deduction guide for CongruentMCQPP
template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, ValidConvexSpace ConvexSpace>
CongruentMCQPP(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&)
    -> CongruentMCQPP<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, ValidConvexSpace ConvexSpace, typename DTWorkspace, typename MWorkspace,
          typename DWorkspace, typename LWorkspace>
CongruentMCQPP(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&,
               const DTWorkspace&, const MWorkspace&, const DWorkspace&, const LWorkspace&)
    -> CongruentMCQPP<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace, DTWorkspace,
                      MWorkspace, DWorkspace, LWorkspace>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

template <typename Backend, typename LinearOp, typename QVector, ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space) {
  return CQPP(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, ConvexSpace&& space) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  return CQPP(Backend{}, std::move(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          ValidConvexSpace ConvexSpace, typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                      UVector&& u, ConvexSpace&& space) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  auto workspace = A.make_workspace(std::forward<FVector>(f), std::forward<UVector>(u));
  return CQPP(Backend{}, std::move(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space),
              std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, ValidConvexSpace ConvexSpace, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space, Workspace&& workspace) {
  return CQPP(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space),
              std::forward<Workspace>(workspace));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b,
                                            ConvexSpace&& space) {
  return MCQPP(Backend{}, std::forward<LinearOpA>(A), std::forward<QVector>(q), std::forward<LinearOpL>(L),
               std::forward<FVector>(f_b), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q,    //
                                            LinearOpL&& L, FVector&& f_b,  //
                                            ConvexSpace&& space,           //
                                            AWorkspace&& a_workspace,      //
                                            LWorkspace&& l_workspace) {
  return MCQPP(Backend{}, std::forward<LinearOpA>(A), std::forward<QVector>(q), std::forward<LinearOpL>(L),
               std::forward<FVector>(f_b), std::forward<ConvexSpace>(space), std::forward<AWorkspace>(a_workspace),
               std::forward<LWorkspace>(l_workspace));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L,
                                            FVector&& f_b, ConvexSpace&& space) {
  return CongruentMCQPP(Backend{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M), std::forward<LinearOpD>(D),
                        std::forward<QVector>(q), std::forward<LinearOpL>(L), std::forward<FVector>(f_b),
                        std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpB, typename LinearOpS, typename LinearOpBT, typename BVector, ValidConvexSpace ConvexSpace>
KOKKOS_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpB&& B,
                                     LinearOpS&& S, LinearOpBT&& BT, BVector&& b, ConvexSpace&& space) {
  using backend_t = Backend;

  // Backend::apply/make_range_vector operate on the raw operators directly -- no need to wrap B/S/b in storage
  // just to read from them here. B and S are only forwarded (not read again) after this, into make_quadratic_form,
  // which is the one place that decides how they end up stored.
  auto tmp = backend_t::make_range_vector(S);
  auto f_b = backend_t::make_range_vector(B);
  backend_t::apply(S, b, tmp);
  backend_t::apply(B, tmp, f_b);

  auto L = make_quadratic_form<backend_t>(std::forward<LinearOpB>(B), std::forward<LinearOpS>(S),
                                          std::forward<LinearOpBT>(BT));

  return CongruentMCQPP(backend_t{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                        std::forward<LinearOpD>(D), std::forward<QVector>(q), std::move(L), std::move(f_b),
                        std::forward<ConvexSpace>(space));
}

/// \brief Solve a constrained quadratic programming problem (CQPP): `x* = argmin_{x in Omega} 0.5 x^T A x + q^T x`.
///
/// Methods for a CQPP:
///   -# Projected-gradient descent (PGD) -- the only method today. Its engine (\ref PGDStrategy,
///      \ref make_pgd_solution_strategy, \ref PGDResult) lives in the shared capability header
///      `pgd.hpp` because PGD also solves an LCP by reduction (see \ref solve_lcp); this verb drives it.
///      The return type is the strategy's result (\ref PGDResult for PGD).
///   -# ADMM (planned) -- inline here if unique to CQPP, or a shared capability header if reused.
///   -# Active-set (planned).
///
/// MCQPP / CongruentMCQPP reduce to a CQPP via `to_cqpp` and are solved through \ref solve_mixed_cqpp.
///
/// \param prob The constrained quadratic programming problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(CQPPSolverStrategy<Strategy, Problem, State>)
KOKKOS_FUNCTION auto solve_cqpp(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  auto result = strat.result(state);
  MUNDY_THROW_ASSERT(!result.converged || impl::workspace_is_committed(prob.workspace()), std::logic_error,
                     "solve_cqpp: converged solution requires committed operator workspace.");
  return result;
}

/// \brief Solve a mixed constrained convex quadratic programming problem (MCQPP)
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D,
/// q, BT, S, B, b).
///
/// \param prob The mixed constrained convex quadratic programming problem to solve.
/// \param strat The solution strategy to use (for the reduced CQPP in x and the linear solve in y).
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Problem& p) {
  { to_cqpp(p) };
})
KOKKOS_FUNCTION auto solve_mixed_cqpp(const Problem& prob, const Strategy& strat, State& state) {
  // Convert MCQPP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

}  // namespace mundy

#endif  // MUNDY_MATH_CQPP_HPP_
