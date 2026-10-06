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

#ifndef MUNDY_MATH_LINEAR_SYSTEM_HPP_
#define MUNDY_MATH_LINEAR_SYSTEM_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <ostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/cmath.hpp>            // for mundy::sqrt
#include <mundy_math/preconditioners.hpp>  // for mundy::{NoPreconditioner, Preconditioner}
#include <mundy_math/residuals.hpp>        // for the residual policies (L2Residual, RelativeL2Residual, ...)
#include <mundy_math/solver_backends.hpp>  // for mundy::{Backend, Workspace, concepts, ...}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Solve result
//@{

/// \brief Result of a linear-system solve: iteration count, final residual, and whether it converged.
template <class Scalar>
struct CGResult {
  using value_type = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  bool converged{false};
};

/// \brief Write a CGResult to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const CGResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}
//@}

template <typename Scalar>
struct CGConfig {
  using value_type = Scalar;

  unsigned max_iters{200};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};  // compared directly against whatever ResidualPolicy
                                                     // reports -- no unit baked in here, that's the policy's job
};

/// \brief The linear system A x = b for a square operator A.
///
/// Pairs the operator with its right-hand side and a mutable workspace; unconstrained (no convex space). A must be
/// square; CG (this header) additionally requires it to be SPD.
template <typename Backend, typename LinearOp, typename RhsVector,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
MUNDY_REQUIRES(LinearOperator<Backend, std::remove_cvref_t<LinearOp>, std::remove_cvref_t<RhsVector>,
                              std::remove_cvref_t<RhsVector>>&& VectorBackend<Backend, std::remove_cvref_t<RhsVector>>)
class LinearSystem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using rhs_vector_storage_t = ::mundy::storage<RhsVector>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using rhs_vector_t = typename rhs_vector_storage_t::value_type;
  using workspace_t = Workspace;
  using value_type = impl::vector_value_type<rhs_vector_t>;

  KOKKOS_INLINE_FUNCTION
  LinearSystem(Backend, LinearOp&& A, RhsVector&& b)
      : A_(std::forward<LinearOp>(A)), b_(std::forward<RhsVector>(b)), workspace_(impl::make_workspace(A_.get())) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "LinearSystem: operator must be square.");
  }

  KOKKOS_INLINE_FUNCTION
  LinearSystem(Backend, LinearOp&& A, RhsVector&& b, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), b_(std::forward<RhsVector>(b)), workspace_(std::move(workspace)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "LinearSystem: operator must be square.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& b() const { return b_.get(); }
  /// \brief Cached scratch state for evaluating A (mutated during a solve).
  ///
  /// One Problem must not back two concurrent solves; construct a fresh Problem per concurrent solve.
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  rhs_vector_storage_t b_;
  mutable workspace_t workspace_;
};

/// \brief The CG solve state: the x/r/p/Ap vectors and the iteration scalars.
///
/// r_dot_z() (r^T z for z = P r the preconditioned residual, r^T r without a preconditioner) and residual() (whatever
/// ResidualPolicy reports) are tracked separately and must not be conflated: the recurrence always needs r^T z for
/// alpha/beta regardless of which norm the caller convergence-tests against; only residual() is pluggable.
template <class Scalar, class XVector, class RVector, class PVector, class ApVector>
class CGState {
 public:
  using value_type = Scalar;

  KOKKOS_INLINE_FUNCTION
  CGState(XVector&& x, RVector&& r, PVector&& p, ApVector&& ap)
      : x_(std::forward<XVector>(x)),
        r_(std::forward<RVector>(r)),
        p_(std::forward<PVector>(p)),
        ap_(std::forward<ApVector>(ap)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& x()        { return x_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& x()  const { return x_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& r()        { return r_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& r()  const { return r_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& p()        { return p_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& p()  const { return p_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& Ap()       { return ap_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& Ap() const { return ap_.get(); }

  KOKKOS_INLINE_FUNCTION unsigned& iter()            { return iter_; }
  KOKKOS_INLINE_FUNCTION unsigned  iter()       const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()        { return converged_; }
  KOKKOS_INLINE_FUNCTION bool      converged()  const { return converged_; }
  // residual(): ResidualPolicy's output; convergence-test only.
  KOKKOS_INLINE_FUNCTION value_type& residual()       { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type  residual() const { return residual_; }
  // r_dot_z(): r^T z; drives the CG recurrence only. See CGStrategy::iterate.
  KOKKOS_INLINE_FUNCTION value_type& r_dot_z()        { return r_dot_z_; }
  KOKKOS_INLINE_FUNCTION value_type  r_dot_z()  const { return r_dot_z_; }
  // clang-format on

 private:
  ::mundy::storage<XVector> x_;
  ::mundy::storage<RVector> r_;
  ::mundy::storage<PVector> p_;
  ::mundy::storage<ApVector> ap_;
  unsigned iter_{0};
  bool converged_{false};
  Scalar residual_{0};
  Scalar r_dot_z_{0};
};

/// \brief The CG strategy: initialize/iterate/done/result over (Problem, State), preconditioned by Precond.
///
/// alpha/beta are fixed by the conjugate-direction recurrence, so there is no step policy -- only how the
/// residual is measured (ResidualPolicy, a VectorResidualPolicy) and the preconditioner P (a Preconditioner) are
/// pluggable. The residual measured is that of A x = b, with or without P. CG starts from alpha x for the caller's
/// guess x, alpha = x^T b / x^T A x, the multiple of x nearest the solution in the A-norm, wherever x^T A x > 0.
template <class ResidualPolicy, class Config, class Precond = NoPreconditioner>
class CGStrategy {
 public:
  using value_type = typename Config::value_type;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using preconditioner_t = Precond;
  using result_t = CGResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  CGStrategy(residual_policy_t resid, config_t cfg = {}) : resid_(resid), cfg_(cfg) {
  }

  KOKKOS_INLINE_FUNCTION
  CGStrategy(residual_policy_t resid, config_t cfg, Precond&& precond)
      : resid_(resid), cfg_(cfg), precond_(std::forward<Precond>(precond)) {
  }

  KOKKOS_INLINE_FUNCTION const auto& precond() const {
    return precond_.get();
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);
    static_assert(Preconditioner<preconditioner_t, backend_t, std::remove_cvref_t<decltype(state.r())>>,
                  "CGStrategy: Precond must be a Preconditioner.");
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    // Ap = A x for the caller's guess x (zero for a cold start, a previous solution for a warm start), then
    // x0 = alpha x.
    backend_t::apply(prob.A(), state.x(), state.Ap(), workspace);
    const value_type x_A_x = backend_t::template dot<value_type>(state.x(), state.Ap());
    if (x_A_x > zero) {
      const value_type alpha = backend_t::template dot<value_type>(state.x(), prob.b()) / x_A_x;
      backend_t::axpby(alpha, state.x(), zero, state.x());
      backend_t::axpby(alpha, state.Ap(), zero, state.Ap());
    }
    backend_t::deep_copy(state.r(), prob.b());
    backend_t::axpby(-one, state.Ap(), one, state.r());  // r = b - A x0
    if constexpr (is_preconditioned) {
      backend_t::apply(precond(), state.r(), state.Ap());  // z = P r, held in Ap until the next A p
      backend_t::deep_copy(state.p(), state.Ap());
      state.r_dot_z() = backend_t::template dot<value_type>(state.r(), state.Ap());
    } else {
      backend_t::deep_copy(state.p(), state.r());
      state.r_dot_z() = backend_t::template dot<value_type>(state.r(), state.r());
    }

    state.residual() = measure(backend, state, prob.b());
    state.iter() = 0;
    state.converged() = state.residual() <= static_cast<value_type>(cfg_.tol);
    if (state.converged()) {
      impl::workspace_commit(workspace);
    }
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);
    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    backend_t::apply(prob.A(), state.p(), state.Ap(), workspace);
    const value_type r_dot_z_old = state.r_dot_z();
    const value_type p_Ap = backend_t::template dot<value_type>(state.p(), state.Ap());
    // p^T A p not positive: A is not positive definite on the Krylov space, so CG stops unconverged.
    if (!(p_Ap > static_cast<value_type>(0))) {
      return true;
    }

    const value_type alpha = r_dot_z_old / p_Ap;
    backend_t::axpby(alpha, state.p(), one, state.x());
    backend_t::axpby(-alpha, state.Ap(), one, state.r());

    if constexpr (!is_preconditioned) {
      state.r_dot_z() = backend_t::template dot<value_type>(state.r(), state.r());
    }
    state.residual() = measure(backend, state, prob.b());
    ++state.iter();

    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    if constexpr (is_preconditioned) {
      backend_t::apply(precond(), state.r(), state.Ap());  // z = P r, held in Ap until the next A p
      state.r_dot_z() = backend_t::template dot<value_type>(state.r(), state.Ap());
      backend_t::axpby(one, state.Ap(), state.r_dot_z() / r_dot_z_old, state.p());
    } else {
      backend_t::axpby(one, state.r(), state.r_dot_z() / r_dot_z_old, state.p());
    }
    return false;
  }

  template <class State>
  KOKKOS_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_FUNCTION result_t result(const State& state) const {
    return {state.iter(), state.residual(), state.converged()};
  }

 private:
  static constexpr bool is_preconditioned = !std::same_as<std::remove_cvref_t<Precond>, NoPreconditioner>;

  template <class Backend, class State, class BVector>
  KOKKOS_FUNCTION value_type measure(const Backend& backend, const State& state, const BVector& b) const {
    using r_vector_t = std::remove_cvref_t<decltype(state.r())>;
    static_assert(VectorResidualPolicy<residual_policy_t, Backend, r_vector_t, BVector>,
                  "CGStrategy: ResidualPolicy must be a VectorResidualPolicy.");
    if constexpr (!is_preconditioned && std::same_as<residual_policy_t, L2Residual> &&
                  std::same_as<impl::vector_value_type<r_vector_t>, value_type>) {
      return sqrt(state.r_dot_z());
    } else {
      return resid_(backend, state.r(), b);
    }
  }

  residual_policy_t resid_;
  config_t cfg_;
  ::mundy::storage<Precond> precond_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <class Backend, class LinearOp, class RhsVector>
LinearSystem(Backend, LinearOp&&, RhsVector&&) -> LinearSystem<Backend, LinearOp, RhsVector>;

template <class Backend, class LinearOp, class RhsVector, class Workspace>
LinearSystem(Backend, LinearOp&&, RhsVector&&, const Workspace&)
    -> LinearSystem<Backend, LinearOp, RhsVector, Workspace>;

template <class XVector, class RVector, class PVector, class ApVector>
CGState(XVector&&, RVector&&, PVector&&, ApVector&&)
    -> CGState<impl::vector_value_type<XVector>, XVector, RVector, PVector, ApVector>;

template <class ResidualPolicy, class Config>
CGStrategy(ResidualPolicy, Config = {}) -> CGStrategy<ResidualPolicy, Config>;

template <class ResidualPolicy, class Config, class Precond>
CGStrategy(ResidualPolicy, Config, Precond&&) -> CGStrategy<ResidualPolicy, Config, Precond>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <class Backend, class LinearOp, class RhsVector>
KOKKOS_INLINE_FUNCTION auto make_linear_system(LinearOp&& A, RhsVector&& b) {
  return LinearSystem(Backend{}, std::forward<LinearOp>(A), std::forward<RhsVector>(b));
}

template <class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_cg_solution_strategy(ResidualPolicy&& residual_policy,
                                                      const CGConfig<Scalar>& cfg = {}) {
  return CGStrategy(std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class ResidualPolicy, class Scalar, class Precond>
KOKKOS_INLINE_FUNCTION auto make_cg_solution_strategy(ResidualPolicy&& residual_policy, const CGConfig<Scalar>& cfg,
                                                      Precond&& precond) {
  return CGStrategy(std::forward<ResidualPolicy>(residual_policy), cfg, std::forward<Precond>(precond));
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_cg_solution_strategy(const CGConfig<Scalar>& cfg = {}) {
  return CGStrategy(L2Residual{}, cfg);
}
//
template <class XVector, class RVector, class PVector, class ApVector>
KOKKOS_INLINE_FUNCTION auto make_cg_state(XVector&& x, RVector&& r, PVector&& p, ApVector&& ap) {
  return CGState(std::forward<XVector>(x), std::forward<RVector>(r), std::forward<PVector>(p),
                 std::forward<ApVector>(ap));
}
//@}

/// \brief Solve A x = b for SPD A via CG.
///
/// state.x() is the initial guess on entry (caller decides cold vs warm start) and the solution on exit. A
/// converged result implies the operator's workspace is committed.
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
})
KOKKOS_FUNCTION auto solve_linear_system(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  auto result = strat.result(state);
  MUNDY_THROW_ASSERT(!result.converged || impl::workspace_is_committed(prob.workspace()), std::logic_error,
                     "solve_linear_system: converged solution requires committed operator workspace.");
  return result;
}

//! \name Inverse operator
//@{
// One layer above solve_linear_system: an operator whose apply() is itself a linear solve, so an SPD operator's
// inverse can slot into operator algebra (e.g. a Schur complement) wherever a LinearOperator is expected.

/// \brief Wraps an SPD operator as its inverse: apply(rhs, out) solves op * out = rhs via matrix-free CG.
///
/// x/r/p/Ap and the operator's own workspace are allocated once at construction and reused; only the lightweight
/// per-call CGState/LinearSystem wrappers are rebuilt in apply(). 
///
/// The first apply is always a cold start; thereafter \p warm_start starts each solve from the previous solution.
///
/// \p Precond, when not NoPreconditioner, preconditions every solve.
template <typename Backend, typename Op, typename Precond = NoPreconditioner>
class CGInvOp {
 public:
  using backend_t = Backend;
  using x_vector_t = decltype(Backend::make_domain_vector(std::declval<const Op&>()));
  using range_vector_t = decltype(Backend::make_range_vector(std::declval<const Op&>()));
  using op_workspace_t = impl::workspace_for_t<Op>;
  using value_type = impl::vector_value_type<x_vector_t>;
  using config_t = CGConfig<value_type>;

  KOKKOS_FUNCTION
  CGInvOp(Backend, Op&& op, const config_t& cfg, bool warm_start = false)
      : CGInvOp(Backend{}, std::forward<Op>(op), cfg, NoPreconditioner{}, warm_start) {
  }

  KOKKOS_FUNCTION
  CGInvOp(Backend, Op&& op, const config_t& cfg, Precond&& precond, bool warm_start = false)
      : op_storage_(std::forward<Op>(op)),
        precond_storage_(std::forward<Precond>(precond)),
        cfg_(cfg),
        warm_start_(warm_start),
        x_(Backend::make_domain_vector(op_storage_.get())),
        r_(Backend::make_range_vector(op_storage_.get())),
        p_(Backend::make_range_vector(op_storage_.get())),
        ap_(Backend::make_range_vector(op_storage_.get())),
        op_workspace_(impl::make_workspace(op_storage_.get())) {
    MUNDY_THROW_ASSERT(Backend::domain_size(op_storage_.get()) == Backend::range_size(op_storage_.get()),
                       std::invalid_argument, "CGInvOp: operator must be square.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION size_t domain_size() const { return Backend::domain_size(op_storage_.get()); }
  KOKKOS_INLINE_FUNCTION size_t range_size() const { return Backend::range_size(op_storage_.get()); }
  KOKKOS_INLINE_FUNCTION static constexpr size_t static_domain_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_domain_size<Op>();
  }
  KOKKOS_INLINE_FUNCTION static constexpr size_t static_range_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_range_size<Op>();
  }
  KOKKOS_INLINE_FUNCTION auto make_domain_vector() const { return Backend::make_domain_vector(op_storage_.get()); }
  KOKKOS_INLINE_FUNCTION auto make_range_vector() const { return Backend::make_range_vector(op_storage_.get()); }
  // clang-format on

  /// out := op^{-1} rhs, via CG.
  template <class RhsVector, class OutVector>
  KOKKOS_FUNCTION void apply(const RhsVector& rhs, OutVector& out) const {
    constexpr value_type zero = static_cast<value_type>(0);
    if (!warm_start_ || first_apply_) {
      Backend::axpby(zero, x_, zero, x_);  // cold start: x0 = 0
    }
    // else: leave x_ at whatever it held after the previous solve (warm start).
    first_apply_ = false;

    auto prob = LinearSystem(Backend{}, op_storage_.get(), rhs, op_workspace_);
    auto state = CGState(x_, r_, p_, ap_);
    auto strat = CGStrategy(L2Residual{}, cfg_, precond_storage_.get());
    last_result_ = solve_linear_system(prob, strat, state);

    // A non-converged CG would silently return a wrong answer with no way for us to inform the caller, so we throw.
    MUNDY_THROW_REQUIRE(last_result_.converged || last_result_.num_iters == cfg_.max_iters, std::runtime_error,
                        "CGInvOp: CG stopped early: p^T A p <= 0, so the operator is not positive definite.");
    MUNDY_THROW_REQUIRE(last_result_.converged, std::runtime_error, "CGInvOp: inner CG solve failed to converge.");
    Backend::deep_copy(out, x_);
  }

  KOKKOS_INLINE_FUNCTION const CGResult<value_type>& last_result() const {
    return last_result_;
  }

 private:
  ::mundy::storage<Op> op_storage_;
  ::mundy::storage<Precond> precond_storage_;
  config_t cfg_;
  bool warm_start_;
  mutable bool first_apply_ = true;
  mutable x_vector_t x_;
  mutable range_vector_t r_;
  mutable range_vector_t p_;
  mutable range_vector_t ap_;
  mutable op_workspace_t op_workspace_;
  mutable CGResult<value_type> last_result_{};
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <class Backend, class Op, class Scalar>
CGInvOp(Backend, Op&&, const CGConfig<Scalar>&, bool = false) -> CGInvOp<Backend, Op>;

template <class Backend, class Op, class Scalar, class Precond>
MUNDY_REQUIRES(Preconditioner<Precond, Backend,
                              decltype(Backend::make_domain_vector(std::declval<const std::remove_cvref_t<Op>&>()))>)
CGInvOp(Backend, Op&&, const CGConfig<Scalar>&, Precond&&, bool = false) -> CGInvOp<Backend, Op, Precond>;
#endif  // DOXYGEN_SHOULD_SKIP_THIS

template <class Backend, class Op, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_cg_inv_op(Op&& op, const CGConfig<Scalar>& cfg, bool warm_start = false) {
  return CGInvOp(Backend{}, std::forward<Op>(op), cfg, warm_start);
}

template <class Backend, class Op, class Scalar, class Precond>
MUNDY_REQUIRES(Preconditioner<Precond, Backend,
                              decltype(Backend::make_domain_vector(std::declval<const std::remove_cvref_t<Op>&>()))>)
KOKKOS_INLINE_FUNCTION auto make_cg_inv_op(Op&& op, const CGConfig<Scalar>& cfg, Precond&& precond,
                                           bool warm_start = false) {
  return CGInvOp(Backend{}, std::forward<Op>(op), cfg, std::forward<Precond>(precond), warm_start);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_LINEAR_SYSTEM_HPP_
