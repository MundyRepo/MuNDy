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

#ifndef MUNDY_MATH_EIGENVALUES_HPP_
#define MUNDY_MATH_EIGENVALUES_HPP_

/// \file eigenvalues.hpp
/// \brief Extremal eigenvalues of a square operator via the power method.
///
/// The power method iterates q <- A q / ||A q|| from a caller-supplied start vector and reports the Rayleigh quotient
/// lambda = q . A q of the unit iterate. It converges to the eigenvalue of largest magnitude at rate
/// |lambda_2 / lambda_1| per iteration, provided the start vector has a component along its eigenvector.

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/Tolerance.hpp>        // for mundy::get_relaxed_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::sqrt
#include <mundy_math/linear_ops.hpp>       // for mundy::make_shifted_op
#include <mundy_math/residuals.hpp>        // for the vector and change residual policies
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::{vector_value_type, workspace_commit, ...}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Solve result
//@{

/// \brief Result of a power-method solve: iteration count, final residual, eigenvalue, and whether it converged.
template <class Scalar>
struct PowerResult {
  using value_type = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  Scalar eigenvalue{0};
  bool converged{false};
};

/// \brief Write a PowerResult to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const PowerResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", eigenvalue: " << result.eigenvalue
     << ", converged?: " << result.converged;
  return os;
}

/// \brief The dominant (largest-magnitude) end of a spectrum and the end opposite it.
template <class Scalar>
struct EigenBounds {
  using value_type = Scalar;

  PowerResult<Scalar> dominant;
  PowerResult<Scalar> opposite;
};
//@}

template <typename Scalar>
struct PowerConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};  // compared directly against whatever ResidualPolicy reports
};

/// \brief The eigenvalue problem A q = lambda q for a square operator A, paired with a mutable workspace.
template <typename Backend, typename LinearOp,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class EigenProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using workspace_t = Workspace;

  KOKKOS_INLINE_FUNCTION
  EigenProblem(Backend, LinearOp&& A) : A_(std::forward<LinearOp>(A)), workspace_(impl::make_workspace(A_.get())) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "EigenProblem: operator must be square.");
  }

  KOKKOS_INLINE_FUNCTION
  EigenProblem(Backend, LinearOp&& A, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), workspace_(std::move(workspace)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "EigenProblem: operator must be square.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  /// \brief Cached scratch state for evaluating A (mutated during a solve).
  ///
  /// One Problem must not back two concurrent solves; construct a fresh Problem per concurrent solve.
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  mutable workspace_t workspace_;
};

/// \brief The power-method state: the q/z/r vectors and the iteration scalars.
///
/// q is the start vector on entry and the unit eigenvector estimate on exit; z holds A q and r the eigen residual
/// A q - lambda q. eigenvalue() is the Rayleigh quotient of the q that residual() was measured at, and +infinity
/// before the first iterate.
template <class Scalar, class QVector, class ZVector, class RVector>
class PowerState {
 public:
  using value_type = Scalar;

  KOKKOS_INLINE_FUNCTION
  PowerState(QVector&& q, ZVector&& z, RVector&& r)
      : q_(std::forward<QVector>(q)), z_(std::forward<ZVector>(z)), r_(std::forward<RVector>(r)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& q()       { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& z()       { return z_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& z() const { return z_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& r()       { return r_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& r() const { return r_.get(); }

  KOKKOS_INLINE_FUNCTION unsigned&   iter()             { return iter_; }
  KOKKOS_INLINE_FUNCTION unsigned    iter()       const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&       converged()        { return converged_; }
  KOKKOS_INLINE_FUNCTION bool        converged()  const { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type& residual()         { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type  residual()   const { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type& eigenvalue()       { return eigenvalue_; }
  KOKKOS_INLINE_FUNCTION value_type  eigenvalue() const { return eigenvalue_; }
  // clang-format on

 private:
  ::mundy::storage<QVector> q_;
  ::mundy::storage<ZVector> z_;
  ::mundy::storage<RVector> r_;
  unsigned iter_{0};
  bool converged_{false};
  Scalar residual_{0};
  Scalar eigenvalue_{0};
};

/// \brief The power-method strategy: initialize/iterate/done/result over (Problem, State).
///
/// The iteration itself is fixed; only how convergence is measured is pluggable. ResidualPolicy is either a
/// VectorResidualPolicy, measuring r = A q - lambda q against A q, or a ChangeResidualPolicy, measuring lambda against
/// its previous estimate.
template <class ResidualPolicy, class Config>
class PowerStrategy {
 public:
  using value_type = typename Config::value_type;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = PowerResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  PowerStrategy(residual_policy_t resid, config_t cfg = {}) : resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize([[maybe_unused]] const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    constexpr value_type zero = static_cast<value_type>(0);

    const value_type q_norm = sqrt(backend_t::template dot<value_type>(state.q(), state.q()));
    MUNDY_THROW_REQUIRE(q_norm > zero, std::invalid_argument, "PowerStrategy: the start vector must be nonzero.");
    backend_t::axpby(zero, state.q(), static_cast<value_type>(1) / q_norm, state.q());

    state.iter() = 0;
    state.converged() = false;
    state.residual() = zero;
    state.eigenvalue() = Kokkos::Experimental::infinity_v<value_type>;
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    backend_t::apply(prob.A(), state.q(), state.z(), workspace);  // z = A q
    const value_type lambda_prev = state.eigenvalue();
    const value_type lambda = backend_t::template dot<value_type>(state.q(), state.z());  // q is unit
    backend_t::deep_copy(state.r(), state.z());
    backend_t::axpby(-lambda, state.q(), one, state.r());  // r = A q - lambda q

    state.eigenvalue() = lambda;
    state.residual() = measure(backend, state, lambda, lambda_prev);
    ++state.iter();

    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    const value_type z_norm = sqrt(backend_t::template dot<value_type>(state.z(), state.z()));
    MUNDY_THROW_REQUIRE(z_norm > zero, std::runtime_error,
                        "PowerStrategy: A q = 0 before convergence; the iterate has no direction to continue along.");
    backend_t::deep_copy(state.q(), state.z());
    backend_t::axpby(zero, state.q(), one / z_norm, state.q());  // q = A q / ||A q||
    return false;
  }

  template <class State>
  KOKKOS_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_FUNCTION result_t result(const State& state) const {
    return {state.iter(), state.residual(), state.eigenvalue(), state.converged()};
  }

 private:
  template <class Backend, class State>
  KOKKOS_FUNCTION value_type measure(const Backend& backend, const State& state, value_type lambda,
                                     value_type lambda_prev) const {
    using r_vector_t = std::remove_cvref_t<decltype(state.r())>;
    using z_vector_t = std::remove_cvref_t<decltype(state.z())>;
    if constexpr (VectorResidualPolicy<residual_policy_t, Backend, r_vector_t, z_vector_t>) {
      return resid_(backend, state.r(), state.z());
    } else {
      static_assert(ChangeResidualPolicy<residual_policy_t, value_type>,
                    "PowerStrategy: ResidualPolicy must be a VectorResidualPolicy or a ChangeResidualPolicy.");
      return resid_(lambda, lambda_prev);
    }
  }

  residual_policy_t resid_;
  config_t cfg_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <class Backend, class LinearOp>
EigenProblem(Backend, LinearOp&&) -> EigenProblem<Backend, LinearOp>;

template <class Backend, class LinearOp, class Workspace>
EigenProblem(Backend, LinearOp&&, const Workspace&) -> EigenProblem<Backend, LinearOp, Workspace>;

template <class QVector, class ZVector, class RVector>
PowerState(QVector&&, ZVector&&, RVector&&) -> PowerState<impl::vector_value_type<QVector>, QVector, ZVector, RVector>;

template <class ResidualPolicy, class Config>
PowerStrategy(ResidualPolicy, Config = {}) -> PowerStrategy<ResidualPolicy, Config>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <class Backend, class LinearOp>
KOKKOS_INLINE_FUNCTION auto make_eigen_problem(LinearOp&& A) {
  return EigenProblem(Backend{}, std::forward<LinearOp>(A));
}

template <class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_power_strategy(ResidualPolicy&& residual_policy, const PowerConfig<Scalar>& cfg = {}) {
  return PowerStrategy(std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_power_strategy(const PowerConfig<Scalar>& cfg = {}) {
  return PowerStrategy(L2Residual{}, cfg);
}
//
template <class QVector, class ZVector, class RVector>
KOKKOS_INLINE_FUNCTION auto make_power_state(QVector&& q, ZVector&& z, RVector&& r) {
  return PowerState(std::forward<QVector>(q), std::forward<ZVector>(z), std::forward<RVector>(r));
}
//@}

/// \brief The dominant (largest-magnitude) eigenpair of A via the power method.
///
/// state.q() is the start vector on entry and the unit eigenvector on exit. A converged result implies the operator's
/// workspace is committed.
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
})
KOKKOS_FUNCTION auto solve_eigen_problem(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  auto result = strat.result(state);
  MUNDY_THROW_ASSERT(!result.converged || impl::workspace_is_committed(prob.workspace()), std::logic_error,
                     "solve_eigen_problem: converged solution requires committed operator workspace.");
  return result;
}

/// \brief Both ends of the spectrum of a symmetric A via the shifted power method.
///
/// With lambda_d the dominant eigenvalue, the dominant eigenvalue of A - lambda_d I is lambda_opp - lambda_d, where
/// lambda_opp is the opposite end of the spectrum, so a second power iteration on the shifted operator recovers it.
/// Each state's q() is its start vector on entry and its unit eigenvector on exit.
template <class Problem, class Strategy, class DominantState, class OppositeState>
KOKKOS_FUNCTION auto solve_eigen_bounds(const Problem& prob, const Strategy& strat, DominantState& dominant_state,
                                        OppositeState& opposite_state) {
  using backend_t = decltype(prob.backend());
  using value_type = typename Strategy::value_type;

  const auto dominant = solve_eigen_problem(prob, strat, dominant_state);
  MUNDY_THROW_REQUIRE(dominant.converged, std::runtime_error,
                      "solve_eigen_bounds: the dominant eigenvalue, which sets the shift, did not converge.");

  const auto shifted_prob = make_eigen_problem<backend_t>(make_shifted_op<backend_t>(dominant.eigenvalue, prob.A()));
  auto opposite = solve_eigen_problem(shifted_prob, strat, opposite_state);
  opposite.eigenvalue += dominant.eigenvalue;

  return EigenBounds<value_type>{dominant, opposite};
}

}  // namespace mundy

#endif  // MUNDY_MATH_EIGENVALUES_HPP_
