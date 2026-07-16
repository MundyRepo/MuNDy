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

#ifndef MUNDY_MATH_PGD_HPP_
#define MUNDY_MATH_PGD_HPP_

/// \file pgd.hpp
/// \brief Projected-gradient descent (PGD), a solver method shared by more than one problem.
///
/// PGD is a capability header, not a problem header: it holds the projected-gradient method (Config/State/Strategy,
/// its step policy, its result) with no problem type and no solve_* verb. A problem that PGD applies to #includes
/// this header and provides its own solve_* verb (e.g. solve_cqpp in cqpp.hpp). PGD consumes the ProjectedProblem
/// interface below, so any problem satisfying that interface can be driven by it without PGD naming the problem.

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <ostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/Tolerance.hpp>        // for mundy::get_zero_tolerance<T>, get_relaxed_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::abs
#include <mundy_math/residuals.hpp>        // for the default projected-gradient residual policy
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::{vector_value_type, workspace_commit}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>  // for mundy::storage

namespace mundy {

//! \name Projected-gradient descent (PGD) result
//@{

/// \brief Result of a projected-gradient solve: iteration count, final residual, and whether it converged.
///
/// This is the result type of the PGD method, not of any one problem. Every problem PGD solves (CQPP, and LCP via
/// reduction) reports its solve as a PGDResult; a different method would define its own result type.
template <class Scalar>
struct PGDResult {
  using value_type = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  bool converged{false};
};

/// \brief Write a PGDResult to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const PGDResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}
//@}

//! \name Projected-gradient descent (PGD) problem interface
//@{

/// \brief The problem interface projected-gradient descent consumes: a backend, an operator A and linear term q
/// defining the quadratic 0.5 x^T A x + q^T x, a convex space to project onto, and a mutable operator workspace.
///
/// A CQPP satisfies this. An LCP does NOT (and must not): its constraint space appears only under reduction to a
/// CQPP, so an LCP is solved by PGD via to_cqpp rather than natively.
template <class Problem>
concept ProjectedProblem = requires(const Problem& p) {
  p.backend();
  p.A();
  p.q();
  p.space();
  p.workspace();
};
//@}

//! \name Projected-gradient descent (PGD) method
//@{

/// \brief Barzilai-Borwein step-size policy for projected-gradient descent.
struct BBStepStrategy {
  template <typename Backend, typename XOldVector, typename GradOldVector, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_value_type<XVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XOldVector& x_old,
                                             const GradOldVector& grad_old,  //
                                             const XVector& x, const GradVector& grad) const {
    using value_type = ReductionScalar;

    value_type num = Backend::template diff_dot<value_type>(x, x_old);  // (x - x_old) dot (x - x_old)
    value_type denom =
        Backend::template diff_dot<value_type>(x, x_old, grad, grad_old);  // (x - x_old) dot (grad - grad_old)

    // Avoid division by zero
    constexpr value_type eps = get_zero_tolerance<value_type>() * static_cast<value_type>(10);
    denom += eps * (abs(denom) < eps);

    return num / denom;
  }
};  // BBStepStrategy

template <typename Scalar>
struct PGDConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

template <class Scalar, class XVector, class GradVector, class XTmpVector, class GradTmpVector>
class PGDState {
 public:
  using value_type = Scalar;
  using x_vector_storage_t = ::mundy::storage<XVector>;
  using grad_vector_storage_t = ::mundy::storage<GradVector>;
  using x_tmp_vector_storage_t = ::mundy::storage<XTmpVector>;
  using grad_tmp_vector_storage_t = ::mundy::storage<GradTmpVector>;

  KOKKOS_INLINE_FUNCTION
  PGDState(XVector&& x, GradVector&& g, XTmpVector&& x_tmp, GradTmpVector&& g_tmp)
      : x_(std::forward<XVector>(x)),
        g_(std::forward<GradVector>(g)),
        x_tmp_(std::forward<XTmpVector>(x_tmp)),
        g_tmp_(std::forward<GradTmpVector>(g_tmp)) {
  }

  // Accessors (const/non-const as needed)
  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& x()      { return x_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& x() const{ return x_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& grad()      { return g_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& grad() const{ return g_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& x_tmp()      { return x_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& x_tmp() const{ return x_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& grad_tmp()      { return g_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& grad_tmp() const{ return g_tmp_.get(); }
  // clang-format on

  // Iteration locals with accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION unsigned& iter()         { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()    { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type& residual()     { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type& step_size()        { return step_size_; }

  KOKKOS_INLINE_FUNCTION unsigned  iter()    const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool      converged() const { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type  residual() const  { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type  step_size()    const  { return step_size_; }
  // clang-format on

 private:
  x_vector_storage_t x_;
  grad_vector_storage_t g_;
  x_tmp_vector_storage_t x_tmp_;
  grad_tmp_vector_storage_t g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  value_type residual_{0};
  value_type step_size_{1};
};

template <class StepPolicy, class ResidualPolicy, class Config>
class PGDStrategy {
 public:
  using value_type = typename Config::value_type;

  using step_policy_t = StepPolicy;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = PGDResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(step_policy_t step, residual_policy_t resid, config_t cfg = {}) : step_(step), resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  MUNDY_REQUIRES(ProjectedProblem<Problem>)
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    // x_tmp = x
    backend_t::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    backend_t::apply(prob.A(), state.x_tmp(), state.grad_tmp(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad_tmp());

    // Dai-Fletcher Sec. 5 initial step
    state.residual() = resid_(backend, state.x_tmp(), state.grad_tmp(), prob.space());

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<value_type>(cfg_.tol));
    if (state.converged()) {
      state.step_size() = one;
      // If already converged, copy grad_tmp to grad
      backend_t::deep_copy(state.grad(), state.grad_tmp());
      impl::workspace_commit(workspace);
    } else {
      state.step_size() = one / state.residual();
    }
  }

  template <class Problem, class State>
  MUNDY_REQUIRES(ProjectedProblem<Problem>)
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x = Proj(x_tmp - step_size * grad_tmp)
    backend_t::wrapped_axpbyz(one, state.x_tmp(), -state.step_size(), state.grad_tmp(), state.x(), prob.space());

    // grad = A x + q
    backend_t::apply(prob.A(), state.x(), state.grad(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad());

    // residual & test
    state.residual() = resid_(backend, state.x(), state.grad(), prob.space());
    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    // update step size and roll x_tmp/grad_tmp forward
    state.step_size() = step_(backend, state.x_tmp(), state.grad_tmp(), state.x(), state.grad());
    backend_t::deep_copy(state.x_tmp(), state.x());
    backend_t::deep_copy(state.grad_tmp(), state.grad());
    ++state.iter();
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
  step_policy_t step_;
  residual_policy_t resid_;
  config_t cfg_;
};
//@}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

/// \brief Deduction guide for PGDConfig
template <typename Scalar>
PGDConfig(unsigned, Scalar) -> PGDConfig<Scalar>;

/// \brief Deduction guide for PGDState
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
PGDState(XVector&&, GradVector&&, XTmpVector&&, GradTmpVector&&)
    -> PGDState<impl::vector_value_type<XVector>, XVector, GradVector, XTmpVector, GradTmpVector>;

/// \brief Deduction guide for PGDStrategy
template <class StepPolicy, class ResidualPolicy, class Config>
PGDStrategy(StepPolicy, ResidualPolicy, Config = {}) -> PGDStrategy<StepPolicy, ResidualPolicy, Config>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <class StepPolicy, class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(StepPolicy&& step_policy,          //
                                                       ResidualPolicy&& residual_policy,  //
                                                       const PGDConfig<Scalar>& cfg = {}) {
  return PGDStrategy(std::forward<StepPolicy>(step_policy), std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const PGDConfig<Scalar>& cfg = {}) {
  using DefaultStepPolicy = BBStepStrategy;
  using DefaultResidualPolicy = LinfNormProjectedDiffResidual;
  return PGDStrategy(DefaultStepPolicy{}, DefaultResidualPolicy{}, cfg);
}
//
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
KOKKOS_INLINE_FUNCTION auto make_pgd_state(XVector&& x,         //
                                           GradVector&& grad,   //
                                           XTmpVector&& x_tmp,  //
                                           GradTmpVector&& grad_tmp) {
  return PGDState(std::forward<XVector>(x), std::forward<GradVector>(grad), std::forward<XTmpVector>(x_tmp),
                  std::forward<GradTmpVector>(grad_tmp));
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_PGD_HPP_
