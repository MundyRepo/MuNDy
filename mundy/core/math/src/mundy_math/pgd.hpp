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
/// its step strategies, its result) with no problem type and no solve_* verb. A problem that PGD applies to #includes
/// this header and provides its own solve_* verb (e.g. solve_cqpp in cqpp.hpp). PGD consumes the ProjectedProblem
/// interface below, so any problem satisfying that interface can be driven by it without PGD naming the problem.

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <cstddef>  // for size_t
#include <ostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/NumTraits.hpp>        // for mundy::NumTraits
#include <mundy_math/Tolerance.hpp>        // for mundy::get_relaxed_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::{min, max}
#include <mundy_math/convex_spaces.hpp>    // for mundy::LowerBoundSpace
#include <mundy_math/eigenvalues.hpp>      // for mundy::{EigenProblem, PowerConfig, solve_eigen_problem}
#include <mundy_math/residuals.hpp>        // for the default projected-gradient residual policy
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::{vector_value_type, workspace_commit}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>       // for mundy::storage
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

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

template <typename Scalar>
struct PGDConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

/// \brief The iterate, its gradient, the previous ones, the iteration's progress, and StepState, the memory its step
/// strategy keeps between steps.
template <class Scalar, class XVector, class GradVector, class XTmpVector, class GradTmpVector, class StepState>
class PGDState {
 public:
  using value_type = Scalar;
  using step_state_t = StepState;
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
  KOKKOS_INLINE_FUNCTION unsigned&   iter()          { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&       converged()     { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type& residual()      { return residual_; }
  KOKKOS_INLINE_FUNCTION StepState&  step_state()    { return step_state_; }

  KOKKOS_INLINE_FUNCTION unsigned          iter()       const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool              converged()  const { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type        residual()   const { return residual_; }
  KOKKOS_INLINE_FUNCTION const StepState&  step_state() const { return step_state_; }
  // clang-format on

 private:
  x_vector_storage_t x_;
  grad_vector_storage_t g_;
  x_tmp_vector_storage_t x_tmp_;
  grad_tmp_vector_storage_t g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  value_type residual_{0};
  StepState step_state_{};
};

/// \brief A step strategy for projected-gradient descent: how each iterate is taken.
///
/// initialize(prob, state) is called once x_tmp holds the starting point x_0, grad_tmp holds A x_0 + q, and x_0 is not
/// converged; it sets state.step_state(). step(prob, state) takes x_tmp and grad_tmp as the current iterate x_k and its
/// gradient, writes the next iterate, a point of prob's space, into x and its gradient A x + q into grad, and updates
/// state.step_state() for the next step. A strategy names its memory as step_state_t<Scalar>.
///
/// A strategy that updates grad by recurrence, so that it equals A x + q only up to rounding, declares
/// static constexpr bool accumulates_gradient = true; PGD then recomputes A x + q before it accepts an iterate.
template <class S, class Problem, class State>
concept PGDStepStrategy = ProjectedProblem<Problem> && requires(const S& s, const Problem& prob, State& state) {
  typename S::template step_state_t<typename State::value_type>;
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.step(prob, state) } -> std::same_as<void>;
};

namespace impl {
/// \brief Whether StepStrategy declares that it updates grad by recurrence.
template <class StepStrategy>
inline constexpr bool accumulates_gradient_v = requires { requires StepStrategy::accumulates_gradient; };

/// \brief 1 where v is nonzero and 0 where it is zero.
template <class Scalar>
struct NonzeroIndicator {
  KOKKOS_INLINE_FUNCTION constexpr Scalar operator()(const Scalar& v) const {
    return v != Scalar(0) ? Scalar(1) : Scalar(0);
  }
};

/// \brief The first step length from x_tmp, whose gradient is grad_tmp and projected residual state.residual().
///
/// From a warm start, the exact minimizer along the projected gradient g_p, g_p^T g_p / g_p^T A g_p. From x = 0,
/// Dai & Fletcher's 1 / residual instead: an exact step there can pin every later Barzilai-Borwein step at 1 / diag(A)
/// wherever the entries with a descent direction do not couple through A. x and grad serve as scratch; x is restored.
template <class Problem, class State>
KOKKOS_FUNCTION typename State::value_type initial_step_size(const Problem& prob, State& state) {
  using value_type = typename State::value_type;
  using backend_t = decltype(prob.backend());
  constexpr value_type zero = static_cast<value_type>(0);
  constexpr value_type one = static_cast<value_type>(1);
  if (!(backend_t::template dot<value_type>(state.x_tmp(), state.x_tmp()) > zero)) {
    return one / state.residual();
  }

  // grad = g_p: g where a trial step x - g / residual leaves the projection, 0 where it projects back onto x
  backend_t::wrapped_axpbyz(one, state.x_tmp(), -one / state.residual(), state.grad_tmp(), state.grad(), prob.space());
  backend_t::axpby(-one, state.x_tmp(), one, state.grad());
  backend_t::wrapped_axpbyz(one, state.grad(), zero, state.grad(), state.grad(), NonzeroIndicator<value_type>{});
  backend_t::elementwise_mul(state.grad(), state.grad_tmp(), state.grad());

  // x = A g_p, then restored
  backend_t::apply(prob.A(), state.grad(), state.x(), prob.workspace());
  const value_type gp_gp = backend_t::template dot<value_type>(state.grad(), state.grad());
  const value_type gp_A_gp = backend_t::template dot<value_type>(state.grad(), state.x());
  MUNDY_THROW_ASSERT(gp_A_gp >= -get_zero_tolerance<value_type>(), std::invalid_argument,
                     "PGD: g_p^T A g_p < -tolerance, so A is not positive semi-definite.");
  backend_t::deep_copy(state.x(), state.x_tmp());
  return gp_A_gp > zero ? gp_gp / gp_A_gp : one / state.residual();
}

/// \brief x := Proj(x_tmp - step_size grad_tmp) and grad := A x + q.
template <class Problem, class State>
KOKKOS_FUNCTION void projected_step(const Problem& prob, State& state, typename State::value_type step_size) {
  using value_type = typename State::value_type;
  using backend_t = decltype(prob.backend());
  constexpr value_type one = static_cast<value_type>(1);
  backend_t::wrapped_axpbyz(one, state.x_tmp(), -step_size, state.grad_tmp(), state.x(), prob.space());
  backend_t::apply(prob.A(), state.x(), state.grad(), prob.workspace());
  backend_t::axpby(one, prob.q(), one, state.grad());
}

/// \brief The Barzilai-Borwein step length from the last step, x_tmp to x.
///
/// With s = x - x_tmp and y = grad - grad_tmp = A s, the step is s^T s / s^T A s, the inverse of A's Rayleigh quotient
/// along s. A positive semi-definite A makes s^T A s non-negative; where it is zero (s = 0, or A has no curvature along
/// s), current_step is kept.
template <class Backend, class State>
KOKKOS_FUNCTION typename State::value_type bb_step_size(const State& state,
                                                        typename State::value_type current_step) {
  using value_type = typename State::value_type;
  const value_type ss = Backend::template diff_dot<value_type>(state.x(), state.x_tmp());  // s^T s
  const value_type sy =
      Backend::template diff_dot<value_type>(state.x(), state.x_tmp(), state.grad(), state.grad_tmp());  // s^T A s
  MUNDY_THROW_ASSERT(sy >= -get_zero_tolerance<value_type>(), std::invalid_argument,
                     "PGD: s^T A s < -tolerance, so A is not positive semi-definite.");
  return sy > static_cast<value_type>(0) ? ss / sy : current_step;
}
}  // namespace impl

/// \brief The step length a Barzilai-Borwein strategy carries between steps.
template <class Scalar>
struct BBStepState {
  Scalar step_size{1};
};

/// \brief Barzilai-Borwein steps, each projected trial iterate taken.
///
/// The first step length is the exact one along the projected gradient from a warm start and 1 / residual from zero;
/// each later one is the inverse Rayleigh quotient s^T s / s^T A s along the previous step.
struct BBStepStrategy {
  template <class Scalar>
  using step_state_t = BBStepState<Scalar>;

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    state.step_state().step_size = impl::initial_step_size(prob, state);
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void step(const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    auto& step_size = state.step_state().step_size;
    impl::projected_step(prob, state, step_size);
    step_size = impl::bb_step_size<backend_t>(state, step_size);
  }
};

/// \brief The step length and recent objective values a nonmonotone Barzilai-Borwein strategy carries between steps.
template <class Scalar, size_t Memory>
struct NonmonotoneBBStepState {
  Scalar step_size{1};
  Kokkos::Array<Scalar, Memory> recent_objective{};
  unsigned num_recorded{0};
};

/// \brief Barzilai-Borwein steps under a nonmonotone line search (Grippo, Lampariello and Lucidi; as in spectral
/// projected gradient).
///
/// The trial iterate is the Barzilai-Borwein one, as BBStepStrategy takes. It is taken if the objective
/// f = 0.5 x^T A x + q^T x there is at most the largest f of the last Memory iterates plus sufficient_decrease
/// grad^T d, for d the step to it. Otherwise the iterate is the minimizer of f along d, a point between the two. The
/// test lets f rise between iterates, as Barzilai-Borwein steps need to, and the iterates converge where projected
/// Barzilai-Borwein steps alone can cycle. The starting point must lie in the problem's space.
template <size_t Memory = 10>
struct NonmonotoneBBStepStrategy {
  static_assert(Memory >= 1, "NonmonotoneBBStepStrategy: Memory must be at least 1.");

  double sufficient_decrease = 1e-4;

  template <class Scalar>
  using step_state_t = NonmonotoneBBStepState<Scalar, Memory>;

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto& memory = state.step_state();
    memory.step_size = impl::initial_step_size(prob, state);
    memory.num_recorded = 0;
    record(memory, objective(prob, state.x_tmp(), state.grad_tmp()));
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void step(const Problem& prob, State& state) const {
    using value_type = typename State::value_type;
    using backend_t = decltype(prob.backend());
    constexpr value_type one = static_cast<value_type>(1);
    constexpr value_type half = static_cast<value_type>(0.5);
    auto& memory = state.step_state();

    impl::projected_step(prob, state, memory.step_size);
    const value_type grad_d = backend_t::template dot<value_type>(state.x(), state.grad_tmp()) -
                              backend_t::template dot<value_type>(state.x_tmp(), state.grad_tmp());
    const value_type d_A_d =
        backend_t::template diff_dot<value_type>(state.x(), state.x_tmp(), state.grad(), state.grad_tmp());
    value_type f = objective(prob, state.x(), state.grad());
    // f along d is f_k + t grad_d + t^2 d_A_d / 2. A rejected d descends (grad_d < 0) and curves (d_A_d > 0), so its
    // minimizer t lies in (0, 1); rounding can break both where d all but vanishes, and the trial iterate is taken.
    const bool descends = grad_d < static_cast<value_type>(0) && d_A_d > static_cast<value_type>(0);
    if (descends && f > reference(memory) + static_cast<value_type>(sufficient_decrease) * grad_d) {
      const value_type t = -grad_d / d_A_d;
      backend_t::axpby(one - t, state.x_tmp(), t, state.x());
      backend_t::axpby(one - t, state.grad_tmp(), t, state.grad());
      f = latest(memory) + t * grad_d + half * t * t * d_A_d;
    }
    record(memory, f);
    memory.step_size = impl::bb_step_size<backend_t>(state, memory.step_size);
  }

 private:
  /// \brief f(x) = 0.5 x^T (grad + q), for grad = A x + q.
  template <class Problem, class XVector, class GradVector>
  KOKKOS_FUNCTION static auto objective(const Problem& prob, const XVector& x, const GradVector& grad) {
    using value_type = impl::vector_value_type<XVector>;
    using backend_t = decltype(prob.backend());
    return static_cast<value_type>(0.5) * (backend_t::template dot<value_type>(x, grad) +
                                           backend_t::template dot<value_type>(x, prob.q()));
  }

  template <class Scalar>
  KOKKOS_FUNCTION static void record(NonmonotoneBBStepState<Scalar, Memory>& memory, Scalar f) {
    memory.recent_objective[memory.num_recorded % Memory] = f;
    ++memory.num_recorded;
  }

  template <class Scalar>
  KOKKOS_FUNCTION static Scalar latest(const NonmonotoneBBStepState<Scalar, Memory>& memory) {
    return memory.recent_objective[(memory.num_recorded - 1) % Memory];
  }

  template <class Scalar>
  KOKKOS_FUNCTION static Scalar reference(const NonmonotoneBBStepState<Scalar, Memory>& memory) {
    const unsigned n = memory.num_recorded < Memory ? memory.num_recorded : static_cast<unsigned>(Memory);
    Scalar largest = memory.recent_objective[0];
    for (unsigned i = 1; i < n; ++i) {
      largest = memory.recent_objective[i] > largest ? memory.recent_objective[i] : largest;
    }
    return largest;
  }
};

namespace impl {
// Entries of MPRGP's gradients over x >= lower_bound. A row is free where x lies above the bound.

/// \brief phi(x, g): g on free rows, 0 on the bound.
template <class Scalar>
struct FreeGradient {
  Scalar lower_bound;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g) const {
    return x > lower_bound ? g : Scalar(0);
  }
};

/// \brief beta(x, g): min(g, 0) on the bound, 0 on free rows.
template <class Scalar>
struct ChoppedGradient {
  Scalar lower_bound;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g) const {
    return x > lower_bound ? Scalar(0) : min(g, Scalar(0));
  }
};

/// \brief beta_i^2.
template <class Scalar>
struct ChoppedGradientSquared {
  Scalar lower_bound;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g) const {
    const Scalar beta = ChoppedGradient<Scalar>{lower_bound}(x, g);
    return beta * beta;
  }
};

/// \brief phi~_i phi_i, for phi~ = min((x - lower_bound) / step, g) on free rows: the part of phi a step of length step
/// can follow before the bound.
template <class Scalar>
struct ReducedTimesFreeGradient {
  Scalar lower_bound;
  Scalar step;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g) const {
    return x > lower_bound ? min((x - lower_bound) / step, g) * g : Scalar(0);
  }
};

/// \brief The step t at which x - t p reaches the bound, or the largest finite Scalar where p does not approach it.
template <class Scalar>
struct StepToBound {
  Scalar lower_bound;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar p) const {
    return p > Scalar(0) ? (x - lower_bound) / p : NumTraits<Scalar>::highest();
  }
};

/// \brief phi_i(x, g - step Ap) Ap_i: the free gradient after a step of length step along p, against A p.
template <class Scalar>
struct FreeGradientAfterStepTimes {
  Scalar lower_bound;
  Scalar step;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g, Scalar Ap) const {
    return x > lower_bound ? (g - step * Ap) * Ap : Scalar(0);
  }
};

/// \brief The next conjugate direction, phi(x, g) - gamma p.
template <class Scalar>
struct ConjugateDirection {
  Scalar lower_bound;
  Scalar gamma;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g, Scalar p) const {
    return FreeGradient<Scalar>{lower_bound}(x, g) - gamma * p;
  }
};

/// \brief Proj(x - step phi(x, g)).
template <class Scalar>
struct ProjectedFreeGradientStep {
  Scalar lower_bound;
  Scalar step;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar x, Scalar g) const {
    return x > lower_bound ? max(x - step * g, lower_bound) : x;
  }
};

/// \brief value, whatever the entry.
template <class Scalar>
struct Constant {
  Scalar value;
  KOKKOS_INLINE_FUNCTION Scalar operator()(Scalar) const {
    return value;
  }
};
}  // namespace impl

/// \brief The expansion step an MPRGP strategy fixes at its start.
template <class Scalar>
struct MPRGPStepState {
  Scalar expansion_step{0};
};

/// \brief Modified proportioning with reduced gradient projections (MPRGP; Dostal & Schoberl 2005), for x at or above a
/// lower bound.
///
/// Rows above the bound are free. The free gradient phi is grad on the free rows and 0 elsewhere; the chopped gradient
/// beta is min(grad, 0) on rows at the bound, the pull that would lift them off it, and 0 elsewhere. While beta is
/// small against phi, ||beta||^2 <= proportioning^2 phi~^T phi with phi~ = min((x - bound) / step, phi) on free rows,
/// each step is a conjugate gradient step over the free rows along a direction p, unless that step would cross the
/// bound; it then goes to the bound along p and takes the projected step x = Proj(x - step phi). Otherwise the step is
/// the exact minimizer along -beta, which lifts rows off the bound. The projected step is expansion_step_fraction /
/// lambda, for lambda the largest eigenvalue of A estimated by power iterations to a relative change of
/// norm_estimate.tol, and the fraction must lie in (0, 2).
///
/// Conjugate gradient steps find which rows end at the bound in few steps where projected gradient steps can take many,
/// as when rows nearly duplicate each other. grad is accumulated by recurrence.
///
/// p holds the direction from one step to the next: it must not alias any vector of the PGD state, the problem's
/// workspace, or anything else in use during the solve.
template <class PVector>
class MPRGPStepStrategy {
 public:
  static constexpr bool accumulates_gradient = true;

  double proportioning = 1.0;
  double expansion_step_fraction = 1.8;
  PowerConfig<double> norm_estimate{.max_iters = 50, .tol = 1e-2};

  template <class Scalar>
  using step_state_t = MPRGPStepState<Scalar>;

  KOKKOS_INLINE_FUNCTION
  explicit MPRGPStepStrategy(PVector&& p) : p_(std::forward<PVector>(p)) {
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    using value_type = typename State::value_type;
    using backend_t = decltype(prob.backend());
    static_assert(std::same_as<std::remove_cvref_t<decltype(prob.space())>, LowerBoundSpace<value_type>>,
                  "MPRGPStepStrategy: the problem's space must be a LowerBoundSpace.");
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);
    auto& p = p_.get();

    // lambda_max(A) by power iterations from the all-ones vector, through p and grad; x holds their residual
    backend_t::wrapped_axpbyz(zero, state.x_tmp(), zero, state.x_tmp(), p, impl::Constant<value_type>{one});
    const auto eigen_problem = EigenProblem(prob.backend(), prob.A(), prob.workspace());
    auto power_state = make_power_state(p, state.grad(), state.x());
    const auto largest = solve_eigen_problem(
        eigen_problem, make_power_strategy(RelativeChangeResidual{}, norm_estimate), power_state);
    MUNDY_THROW_ASSERT(largest.eigenvalue > zero, std::invalid_argument,
                       "MPRGPStepStrategy: A must have a positive eigenvalue.");
    state.step_state().expansion_step = static_cast<value_type>(expansion_step_fraction) / largest.eigenvalue;
    backend_t::deep_copy(state.x(), state.x_tmp());

    const value_type lower_bound = prob.space().lower_bound;
    backend_t::elementwise_transform(state.x_tmp(), state.grad_tmp(), p, impl::FreeGradient<value_type>{lower_bound});
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void step(const Problem& prob, State& state) const {
    using value_type = typename State::value_type;
    using backend_t = decltype(prob.backend());
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);
    const value_type lower_bound = prob.space().lower_bound;
    const value_type expansion_step = state.step_state().expansion_step;
    auto& p = p_.get();
    auto& x = state.x();
    auto& grad = state.grad();
    const auto& x_tmp = state.x_tmp();
    const auto& grad_tmp = state.grad_tmp();

    const value_type chopped = backend_t::template transform_reduce_sum<value_type>(
        x_tmp, grad_tmp, impl::ChoppedGradientSquared<value_type>{lower_bound});
    const value_type free = backend_t::template transform_reduce_sum<value_type>(
        x_tmp, grad_tmp, impl::ReducedTimesFreeGradient<value_type>{lower_bound, expansion_step});
    const value_type proportioning_sq = static_cast<value_type>(proportioning * proportioning);

    if (chopped > proportioning_sq * free) {
      // Proportioning: the exact minimizer along -beta, with beta in x and A beta in grad
      backend_t::elementwise_transform(x_tmp, grad_tmp, x, impl::ChoppedGradient<value_type>{lower_bound});
      backend_t::apply(prob.A(), x, grad, prob.workspace());
      const value_type beta_A_beta = backend_t::template dot<value_type>(x, grad);
      MUNDY_THROW_ASSERT(beta_A_beta > zero, std::invalid_argument,
                         "MPRGPStepStrategy: the objective is unbounded below along the chopped gradient.");
      const value_type step = chopped / beta_A_beta;  // grad^T beta = ||beta||^2
      backend_t::axpby(one, x_tmp, -step, x);
      backend_t::axpby(one, grad_tmp, -step, grad);
      backend_t::elementwise_transform(x, grad, p, impl::FreeGradient<value_type>{lower_bound});
      return;
    }

    // A p in grad, until grad is formed from it
    backend_t::apply(prob.A(), p, grad, prob.workspace());
    const value_type p_A_p = backend_t::template dot<value_type>(p, grad);
    const value_type grad_p = backend_t::template dot<value_type>(grad_tmp, p);
    const value_type to_bound =
        backend_t::template transform_reduce_min<value_type>(x_tmp, p, impl::StepToBound<value_type>{lower_bound});
    if (p_A_p > zero && grad_p / p_A_p <= to_bound) {
      // Conjugate gradient step
      const value_type step = grad_p / p_A_p;
      backend_t::wrapped_axpbyz(one, x_tmp, -step, p, x, prob.space());
      const value_type gamma = backend_t::template transform_reduce_sum<value_type>(
                                   x, grad_tmp, grad, impl::FreeGradientAfterStepTimes<value_type>{lower_bound, step}) /
                               p_A_p;
      backend_t::axpby(one, grad_tmp, -step, grad);
      backend_t::elementwise_transform(x, grad, p, p, impl::ConjugateDirection<value_type>{lower_bound, gamma});
      return;
    }

    // Expansion: to the bound along p, where p descends, then the projected step along the free gradient there
    const value_type to_bound_step = (p_A_p > zero || grad_p > zero) ? to_bound : zero;
    MUNDY_THROW_ASSERT(to_bound_step < NumTraits<value_type>::highest(), std::invalid_argument,
                       "MPRGPStepStrategy: the objective is unbounded below along p.");
    backend_t::wrapped_axpbyz(one, x_tmp, -to_bound_step, p, x, prob.space());
    backend_t::axpby(one, grad_tmp, -to_bound_step, grad);
    backend_t::elementwise_transform(x, grad, x,
                                     impl::ProjectedFreeGradientStep<value_type>{lower_bound, expansion_step});
    backend_t::apply(prob.A(), x, grad, prob.workspace());
    backend_t::axpby(one, prob.q(), one, grad);
    backend_t::elementwise_transform(x, grad, p, impl::FreeGradient<value_type>{lower_bound});
  }

 private:
  mutable ::mundy::storage<PVector> p_;
};

/// \brief The PGD strategy: initialize/iterate/done/result over (Problem, State). StepStrategy is a PGDStepStrategy and
/// ResidualPolicy a ProjectedResidualPolicy.
template <class StepStrategy, class ResidualPolicy, class Config>
class PGDStrategy {
 public:
  using value_type = typename Config::value_type;

  using step_strategy_t = StepStrategy;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = PGDResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(step_strategy_t step, residual_policy_t resid, config_t cfg = {})
      : step_(step), resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  MUNDY_REQUIRES(ProjectedProblem<Problem>)
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    static_assert(PGDStepStrategy<step_strategy_t, Problem, State>,
                  "PGDStrategy: StepStrategy must be a PGDStepStrategy.");
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    // x_tmp = x
    backend_t::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    backend_t::apply(prob.A(), state.x_tmp(), state.grad_tmp(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad_tmp());

    state.residual() = measure(backend, state.x_tmp(), state.grad_tmp(), prob.space());

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<value_type>(cfg_.tol));
    if (state.converged()) {
      // If already converged, copy grad_tmp to grad
      backend_t::deep_copy(state.grad(), state.grad_tmp());
      impl::workspace_commit(workspace);
    } else {
      step_.initialize(prob, state);
    }
  }

  template <class Problem, class State>
  MUNDY_REQUIRES(ProjectedProblem<Problem>)
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x, grad = the next iterate and its gradient
    step_.step(prob, state);
    ++state.iter();

    // residual & test, on A x + q itself before an iterate whose gradient was accumulated is accepted
    state.residual() = measure(backend, state.x(), state.grad(), prob.space());
    if constexpr (impl::accumulates_gradient_v<step_strategy_t>) {
      if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
        constexpr value_type one = static_cast<value_type>(1);
        backend_t::apply(prob.A(), state.x(), state.grad(), workspace);
        backend_t::axpby(one, prob.q(), one, state.grad());
        state.residual() = measure(backend, state.x(), state.grad(), prob.space());
      }
    }
    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    // roll x_tmp/grad_tmp forward
    backend_t::deep_copy(state.x_tmp(), state.x());
    backend_t::deep_copy(state.grad_tmp(), state.grad());
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
  template <class Backend, class XVector, class GradVector, class ConvexSpace>
  KOKKOS_FUNCTION value_type measure(const Backend& backend, const XVector& x, const GradVector& grad,
                                     const ConvexSpace& space) const {
    static_assert(ProjectedResidualPolicy<residual_policy_t, Backend, XVector, GradVector, ConvexSpace>,
                  "PGDStrategy: ResidualPolicy must be a ProjectedResidualPolicy.");
    return resid_(backend, x, grad, space);
  }

  step_strategy_t step_;
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

/// \brief Deduction guide for PGDStrategy
template <class StepStrategy, class ResidualPolicy, class Config>
PGDStrategy(StepStrategy, ResidualPolicy, Config = {}) -> PGDStrategy<StepStrategy, ResidualPolicy, Config>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <class StepStrategy, class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(StepStrategy&& step_strategy,      //
                                                       ResidualPolicy&& residual_policy,  //
                                                       const PGDConfig<Scalar>& cfg = {}) {
  return PGDStrategy(std::forward<StepStrategy>(step_strategy), std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const PGDConfig<Scalar>& cfg = {}) {
  using DefaultStepStrategy = BBStepStrategy;
  using DefaultResidualPolicy = LinfNormProjectedDiffResidual;
  return PGDStrategy(DefaultStepStrategy{}, DefaultResidualPolicy{}, cfg);
}
//
/// \brief MPRGP steps, carrying their direction in p.
template <class PVector>
KOKKOS_INLINE_FUNCTION auto make_mprgp_step_strategy(PVector&& p) {
  return MPRGPStepStrategy<PVector>(std::forward<PVector>(p));
}
//
/// \brief The state pgd iterates, over x, grad and their previous values, with the memory of pgd's step strategy.
template <class StepStrategy, class ResidualPolicy, class Config, class XVector, class GradVector, class XTmpVector,
          class GradTmpVector>
KOKKOS_INLINE_FUNCTION auto make_pgd_state([[maybe_unused]] const PGDStrategy<StepStrategy, ResidualPolicy, Config>& pgd,
                                           XVector&& x,         //
                                           GradVector&& grad,   //
                                           XTmpVector&& x_tmp,  //
                                           GradTmpVector&& grad_tmp) {
  using scalar_t = impl::vector_value_type<XVector>;
  using step_state_t = typename StepStrategy::template step_state_t<scalar_t>;
  return PGDState<scalar_t, XVector, GradVector, XTmpVector, GradTmpVector, step_state_t>(
      std::forward<XVector>(x), std::forward<GradVector>(grad), std::forward<XTmpVector>(x_tmp),
      std::forward<GradTmpVector>(grad_tmp));
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_PGD_HPP_
