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

#ifndef MUNDY_MATH_CONVEX_HPP_
#define MUNDY_MATH_CONVEX_HPP_

// Mundy core:
#include <mundy_core/throw_assert.hpp>

// Mundy math:
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::math::Vector

namespace mundy {

namespace math {

namespace convex {

namespace space {

// These are 1d convex spaces, which will be applied to each element of a vector assuming a separable convex space
template <typename Scalar>
struct Unconstrained {
  using scalar_t = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return x;
  }
};

template <typename Scalar>
struct LowerBound {
  using scalar_t = Scalar;

  scalar_t lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::max(x, lower_bound);
  }
};

template <typename Scalar>
struct UpperBound {
  using scalar_t = Scalar;

  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(x, upper_bound);
  }
};

template <typename Scalar>
struct Bounded {
  using scalar_t = Scalar;

  scalar_t lower_bound;
  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(Kokkos::max(x, lower_bound), upper_bound);
  }
};

}  // namespace space

//! \name Backends
//@{

/// \brief Backend for Kokkos single process execution
template <typename Scalar, typename Layout, typename MemorySpace, typename ExecSpace>
struct KokkosBackend {
 public:
  using scalar_t = Scalar;
  using vector_t = Kokkos::View<scalar_t*, Layout, MemorySpace>;
  using exec_space = ExecSpace;
  using memory_space = MemorySpace;

  static size_t vector_size(const vector_t& x) {
    return x.extent(0);
  }

  static const scalar_t& vector_data(const vector_t& x, size_t i) {
    return x(i);
  }

  static scalar_t& vector_data(vector_t& x, size_t i) {
    return x(i);
  }

  static void deep_copy(vector_t& dest, const vector_t& src) {
    Kokkos::deep_copy(dest, src);
  }

  template <typename LinearOp>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const vector_t& x, vector_t& y) {
    // How should we apply?
    MUNDY_THROW_REQUIRE(false, std::logic_error, "apply not implemented for KokkosBackend.");
  }

  static void axpby(const scalar_t alpha, const vector_t& x, const scalar_t beta, const vector_t& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    Kokkos::parallel_for(
        "axpbyg", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i) { y(i) = alpha * x(i) + beta * y(i); });
  }

  template <typename Wrapper>
  static void wrapped_axpbyz(const scalar_t alpha, const vector_t& x, const scalar_t beta, const vector_t& y,
                             vector_t& z, const Wrapper& wrapper) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && x.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    Kokkos::parallel_for(
        "axpbyg", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i) { z(i) = wrapper(alpha * x(i) + beta * y(i)); });
  }

  static scalar_t diff_dot(const vector_t& x, const vector_t& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    scalar_t result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i, scalar_t& sum) {
          scalar_t diff = x(i) - y(i);
          sum += diff * diff;
        },
        result);
    return result;
  }

  static scalar_t diff_dot(const vector_t& x1, const vector_t& x2,  //
                           const vector_t& y1, const vector_t& y2) {
    MUNDY_THROW_ASSERT(x1.extent(0) == x2.extent(0) && x1.extent(0) == y1.extent(0) &&
                       x1.extent(0) == y2.extent(0),
                       std::invalid_argument, "x1, x2, y1, and y2 must have the same size.");
    scalar_t result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x1.extent(0)),
        KOKKOS_LAMBDA(const int i, scalar_t& sum) {
          scalar_t x_diff = x1(i) - x2(i);
          scalar_t y_diff = y1(i) - y2(i);
          sum += x_diff * y_diff;
        },
        result);
    return result;
  }

  template <typename Functor>
  static void reduce_max(size_t n, const Functor& func, scalar_t& result) {
    Kokkos::parallel_reduce(
        "reduce_max", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) { func(i, max_val); }, Kokkos::Max<scalar_t>(result));
  }
};  // KokkosBackend

/// \brief Backend for Mundy math within a kernel
template <typename Scalar, size_t N>
struct MundyMathBackend {
  using scalar_t = Scalar;
  using vector_t = Vector<scalar_t, N>;

  KOKKOS_INLINE_FUNCTION static size_t vector_size(const vector_t& /*x*/) {
    return vector_t::size;
  }

  KOKKOS_INLINE_FUNCTION static const scalar_t& vector_data(const vector_t& x, size_t i) {
    return x[i];
  }

  KOKKOS_INLINE_FUNCTION static scalar_t& vector_data(vector_t& x, size_t i) {
    return x[i];
  }

  KOKKOS_INLINE_FUNCTION static void deep_copy(vector_t& dest, const vector_t& src) {
    dest = src;
  }

  template <typename LinearOp>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const vector_t& x, vector_t& y) {
    y = op * x;
  }

  KOKKOS_INLINE_FUNCTION static void axpby(const scalar_t alpha, const vector_t& x, const scalar_t beta, vector_t& y) {
    y = alpha * x + beta * y;
  }

  template <typename Wrapper>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const scalar_t alpha, const vector_t& x, const scalar_t beta,
                                                    const vector_t& y, vector_t& z, const Wrapper& wrapper) {
    z = ::mundy::math::apply(wrapper, alpha * x + beta * y);
  }

  KOKKOS_INLINE_FUNCTION static scalar_t diff_dot(const vector_t& x, const vector_t& y) {
    auto diff = x - y;
    return dot(diff, diff);
  }

  KOKKOS_INLINE_FUNCTION static scalar_t diff_dot(const vector_t& x1, const vector_t& x2,  //
                                                  const vector_t& y1, const vector_t& y2) {
    auto x_diff = x1 - x2;
    auto y_diff = y1 - y2;
    return dot(x_diff, y_diff);
  }

  template <class Functor>
  KOKKOS_INLINE_FUNCTION static void reduce_max(size_t n, const Functor& func, scalar_t& result) {
    MUNDY_THROW_ASSERT(n == N, std::invalid_argument, "reduce_max: n must match the size of the vector.");
    reduce_max_impl(std::make_index_sequence<N>{}, func, result);
  }

 private:
  template <size_t... Is, class Functor>
  KOKKOS_INLINE_FUNCTION static void reduce_max_impl(std::index_sequence<Is...>, const Functor& func,
                                                     scalar_t& result) {
    scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
    ((func(Is, max_val)), ...);
    result = max_val;
  }
};  // MundyMathBackend
//@}

//! \name Problems + state
//@{

/// \brief Constrained quadratic programming problem (CQPP) formulation
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOp, typename ConvexSpace>
class CQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_t = LinearOp;
  using space_t = ConvexSpace;
  using scalar_t = typename Backend::scalar_t;
  using vector_t = typename Backend::vector_t;

  KOKKOS_INLINE_FUNCTION
  CQPPProblem(Backend, const linear_op_t& A, const vector_t& q, const space_t& space) : A_(A), q_(q), space_(space) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const linear_op_t& A() const { return A_; }
  KOKKOS_INLINE_FUNCTION const vector_t& q() const { return q_; }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  // clang-format on

 private:
  const linear_op_t& A_;
  const vector_t& q_;
  const space_t& space_;
};  // CQPPProblem

/// \brief Linear complementarity problem (LCP) formulation
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOp>
class LCPProblem {
 public:
  using backend_t = Backend;
  using linear_op_t = LinearOp;
  using scalar_t = typename Backend::scalar_t;
  using vector_t = typename Backend::vector_t;

  KOKKOS_INLINE_FUNCTION
  LCPProblem(Backend, const linear_op_t& A, const vector_t& q) : A_(A), q_(q) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const linear_op_t& A() const { return A_; }
  KOKKOS_INLINE_FUNCTION const vector_t& q() const { return q_; }
  // clang-format on

 private:
  const linear_op_t& A_;
  const vector_t& q_;
};  // LCPProblem

template <class Backend, class LinearOp>
auto to_cqpp(const LCPProblem<Backend, LinearOp>& P) {
  static constexpr space::LowerBound Rn_plus{static_cast<typename Backend::scalar_t>(0)};
  return CQPPProblem(P.backend(), P.A(), P.q(), Rn_plus);
}
//@}

//! \name Policies
//@{

struct LinfNormProjectedGradientResidual {  // LCP only
  template <typename Backend, typename ConvexSpace>
  KOKKOS_INLINE_FUNCTION typename Backend::scalar_t operator()([[maybe_unused]] const Backend& backend,  //
                                                               const typename Backend::vector_t& x,      //
                                                               const typename Backend::vector_t& grad,   //
                                                               const ConvexSpace& convex_space) const {
    using scalar_t = typename Backend::scalar_t;

    size_t n = Backend::vector_size(x);
    scalar_t largest_abs_gradient;
    Backend::reduce_max(
        n,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          // perform the projection EQ 2.2 of Dai & Fletcher 2005
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);

          scalar_t abs_projected_grad;
          if (x_i < get_zero_tolerance<scalar_t>()) {
            abs_projected_grad = Kokkos::max(0.0, grad_i);
          } else {
            abs_projected_grad = Kokkos::abs(grad_i);
          }

          if (abs_projected_grad > max_val) {
            max_val = abs_projected_grad;
          }
        },
        largest_abs_gradient);

    return largest_abs_gradient;
  }
};

struct LinfNormProjectedDiffResidual {
  template <typename Backend, typename ConvexSpace>
  KOKKOS_INLINE_FUNCTION typename Backend::scalar_t operator()([[maybe_unused]] const Backend& backend,  //
                                                               const typename Backend::vector_t& x,      //
                                                               const typename Backend::vector_t& grad,   //
                                                               const ConvexSpace& convex_space) const {
    using scalar_t = typename Backend::scalar_t;

    // This res comes from line 17 and Eq 25 of Mazhar 2015
    // res =  1.0 / (3 * num_unknowns * gd) * norm_inf(xk - proj(xk - gd * gk))
    size_t num_unknowns = Backend::vector_size(x);
    constexpr scalar_t small_step_size = 1e-6;
    scalar_t largest_abs_diff;
    Backend::reduce_max(
        num_unknowns,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);
          scalar_t x_i_proj = convex_space.project(x_i - small_step_size * grad_i);
          scalar_t abs_diff = Kokkos::abs(x_i - x_i_proj);
          if (abs_diff > max_val) {
            max_val = abs_diff;
          }
        },
        largest_abs_diff);

    return largest_abs_diff / small_step_size;
  }
};

struct BBStepStrategy {
  template <typename Backend>
  KOKKOS_INLINE_FUNCTION typename Backend::scalar_t operator()([[maybe_unused]] const Backend& backend,  //
                                                               const typename Backend::vector_t& x_old,
                                                               const typename Backend::vector_t& grad_old,  //
                                                               const typename Backend::vector_t& x,
                                                               const typename Backend::vector_t& grad) const {
    using scalar_t = typename Backend::scalar_t;

    scalar_t num = Backend::diff_dot(x, x_old);                    // (x - x_old) dot (x - x_old)
    scalar_t denom = Backend::diff_dot(x, x_old, grad, grad_old);  // (x - x_old) dot (grad - grad_old)

    // Avoid division by zero
    constexpr scalar_t eps = get_zero_tolerance<scalar_t>() * 10;
    denom += eps * (Kokkos::abs(denom) < eps);

    return num / denom;
  }
};  // BBStepStrategy
//@}

template <typename Scalar>
struct PGDConfig {
  using scalar_t = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

template <class Scalar>
struct SolveResult {
  using scalar_t = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  bool converged{false};
};

/// \brief Write SolveResult to an ostream
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const SolveResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}

template <class Backend>
class PGDState {
 public:
  using backend_t = Backend;
  using vector_t = typename Backend::vector_t;
  using scalar_t = typename Backend::scalar_t;

  KOKKOS_INLINE_FUNCTION
  PGDState(const Backend&, vector_t& x, vector_t& g, vector_t& x_tmp, vector_t& g_tmp)
      : x_(x), g_(g), x_tmp_(x_tmp), g_tmp_(g_tmp) {
  }

  // Accessors (const/non-const as needed)
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION       vector_t& x()      { return x_; }
  KOKKOS_INLINE_FUNCTION const vector_t& x() const{ return x_; }
  KOKKOS_INLINE_FUNCTION       vector_t& grad()      { return g_; }
  KOKKOS_INLINE_FUNCTION const vector_t& grad() const{ return g_; }
  KOKKOS_INLINE_FUNCTION       vector_t& x_tmp()      { return x_tmp_; }
  KOKKOS_INLINE_FUNCTION const vector_t& x_tmp() const{ return x_tmp_; }
  KOKKOS_INLINE_FUNCTION       vector_t& grad_tmp()      { return g_tmp_; }
  KOKKOS_INLINE_FUNCTION const vector_t& grad_tmp() const{ return g_tmp_; }
  // clang-format on

  // Iteration locals with accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION unsigned& iter()         { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()    { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t& residual()     { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t& step_size()        { return step_size_; }

  KOKKOS_INLINE_FUNCTION unsigned  iter()    const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool      converged() const { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t  residual() const  { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t  step_size()    const  { return step_size_; }
  // clang-format on

 private:
  vector_t& x_;
  vector_t& g_;
  vector_t& x_tmp_;
  vector_t& g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  scalar_t residual_{0};
  scalar_t step_size_{1};
};

template <class Backend, class StepPolicy, class ResidualPolicy>
class PGDStrategy {
 public:
  using backend_t = Backend;
  using scalar_t = typename Backend::scalar_t;
  using vector_t = typename Backend::vector_t;

  using step_policy_t = StepPolicy;
  using residual_policy_t = ResidualPolicy;
  using config_t = PGDConfig<scalar_t>;
  using state_t = PGDState<Backend>;
  using result_t = SolveResult<scalar_t>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(backend_t, step_policy_t step, residual_policy_t resid, config_t cfg = {})
      : step_(step), resid_(resid), cfg_(cfg) {
  }

  KOKKOS_INLINE_FUNCTION Backend backend() const {
    return Backend{};
  }

  template <class Problem>
  KOKKOS_INLINE_FUNCTION void initialize(const Problem& prob, state_t& state) const {
    constexpr scalar_t one = static_cast<scalar_t>(1);

    // x_tmp = x
    Backend::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    Backend::apply(prob.A(), state.x_tmp(), state.grad_tmp());
    Backend::axpby(one, prob.q(), one, state.grad_tmp());

    // Dai-Fletcher Sec. 5 initial step
    state.residual() = resid_(Backend{}, state.x_tmp(), state.grad_tmp(), prob.space());
    state.step_size() = one / state.residual();

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<scalar_t>(cfg_.tol));
    if (state.converged()) {
      // If already converged, copy grad_tmp to grad
      Backend::deep_copy(state.grad(), state.grad_tmp());
    }
  }

  template <class Problem>
  KOKKOS_INLINE_FUNCTION bool iterate(const Problem& prob, state_t& state) const {
    constexpr scalar_t one = static_cast<scalar_t>(1);

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x = Proj(x_tmp - step_size * grad_tmp)
    Backend::wrapped_axpbyz(one, state.x_tmp(), -state.step_size(), state.grad_tmp(), state.x(), prob.space());

    // grad = A x + q
    Backend::apply(prob.A(), state.x(), state.grad());
    Backend::axpby(one, prob.q(), one, state.grad());

    // residual & test
    state.residual() = resid_(Backend{}, state.x(), state.grad(), prob.space());
    if (state.residual() <= static_cast<scalar_t>(cfg_.tol)) {
      state.converged() = true;
      return true;
    }

    // update step size and roll x_tmp/grad_tmp forward
    state.step_size() = step_(Backend{}, state.x_tmp(), state.grad_tmp(), state.x(), state.grad());
    Backend::deep_copy(state.x_tmp(), state.x());
    Backend::deep_copy(state.grad_tmp(), state.grad());
    ++state.iter();
    return false;
  }

  KOKKOS_INLINE_FUNCTION
  bool done(const state_t& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  KOKKOS_INLINE_FUNCTION result_t result(const state_t& state) const {
    return {state.iter(), state.residual(), state.converged()};
  }

 private:
  step_policy_t step_;
  residual_policy_t resid_;
  config_t cfg_;
};

template <class Strategy, class Problem>
concept CQPPSolverStrategy = requires(Strategy& s, const Problem& prob) {
  typename Strategy::state_t;
  typename Strategy::result_t;
  { s.initialize(prob) } -> std::same_as<void>;
  { s.iterate(prob) } -> std::same_as<bool>;
  { s.done() } -> std::same_as<bool>;
  { s.result() } -> std::same_as<typename Strategy::result_t>;
};

//! \name Deduction guides
//@{

/// \brief Deduction guide for CQPPProblem
template <typename Backend, typename LinearOp, typename ConvexSpace>
CQPPProblem(Backend, const LinearOp&, const typename Backend::vector_t&, const ConvexSpace&)
    -> CQPPProblem<Backend, LinearOp, ConvexSpace>;

/// \brief Deduction guide for LCPProblem
template <typename Backend, typename LinearOp>
LCPProblem(Backend, const LinearOp&, const typename Backend::vector_t&) -> LCPProblem<Backend, LinearOp>;

/// \brief Deduction guide for PGDConfig
template <typename Scalar>
PGDConfig(unsigned, Scalar) -> PGDConfig<Scalar>;

/// \brief Deduction guide for PGDState
template <class Backend>
PGDState(const Backend&, typename Backend::vector_t&, typename Backend::vector_t&, typename Backend::vector_t&,
         typename Backend::vector_t&) -> PGDState<Backend>;

/// \brief Deduction guide for PGDStrategy
template <class Backend, class StepPolicy, class ResidualPolicy>
PGDStrategy(Backend, StepPolicy, ResidualPolicy, PGDConfig<typename Backend::scalar_t> = {})
    -> PGDStrategy<Backend, StepPolicy, ResidualPolicy>;
//@}

}  // namespace convex

template <typename LinearOp, typename ConvexSpace, typename Scalar, size_t N>
KOKKOS_INLINE_FUNCTION auto make_mundy_math_cqpp(const LinearOp& A, const Vector<Scalar, N>& q,
                                                 const ConvexSpace& space) {
  using backend_t = convex::MundyMathBackend<Scalar, N>;
  return convex::CQPPProblem(backend_t{}, A, q, space);
}

template <typename LinearOp, typename Scalar, size_t N>
KOKKOS_INLINE_FUNCTION auto make_mundy_math_lcp(const LinearOp& A, const Vector<Scalar, N>& q) {
  using backend_t = convex::MundyMathBackend<Scalar, N>;
  return convex::LCPProblem(backend_t{}, A, q);
}

template <class Backend, class StepPolicy, class ResidualPolicy>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const Backend& backend,                 //
                                                       const StepPolicy& step_policy,          //
                                                       const ResidualPolicy& residual_policy,  //
                                                       const convex::PGDConfig<typename Backend::scalar_t>& cfg = {}) {
  return convex::PGDStrategy(backend, step_policy, residual_policy, cfg);
}

template <class Backend>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const Backend& backend,  //
                                                       const convex::PGDConfig<typename Backend::scalar_t>& cfg = {}) {
  using DefaultStepPolicy = convex::BBStepStrategy;
  using DefaultResidualPolicy = convex::LinfNormProjectedDiffResidual;
  return convex::PGDStrategy(backend, DefaultStepPolicy{}, DefaultResidualPolicy{}, cfg);
}

template <class Backend>
KOKKOS_INLINE_FUNCTION auto make_pgd_state(const Backend& backend,             //
                                           typename Backend::vector_t& x,      //
                                           typename Backend::vector_t& grad,   //
                                           typename Backend::vector_t& x_tmp,  //
                                           typename Backend::vector_t& grad_tmp) {
  return convex::PGDState(backend, x, grad, x_tmp, grad_tmp);
}

/// \brief Solve a constrained quadratic programming problem (CQPP)
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \param prob The constrained quadratic programming problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy>
KOKKOS_INLINE_FUNCTION auto solve_cqpp(const Problem& prob, const Strategy& strat, typename Strategy::state_t& state) ->
    typename Strategy::result_t {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  return strat.result(state);
}

/// \brief Solve a linear complementarity problem (LCP) using a constrained quadratic programming solver
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// Example usage:
/// \code{.cpp}
///    // Problem setup
///    Matrix3d A = {/*...*/};
///    Vector3d q = {/*...*/};
///    Vector3d x{/* initial_guess */}, grad{}, x_tmp{}, grad_tmp{};
///
///    // Build the problem (no template args at callsite)
///    const auto lcp = make_mundy_math_lcp(A, q);
///
///    // Reuse the backend token from the problem
///    const auto backend = lcp.backend();
///
///    // Strategy + state
///    PGDConfig cfg{1000, 1e-6};
///
///    auto pgd = make_pgd_solution_strategy(                //
///        backend, cfg);                                    // Use default step/residual strategies
///    // auto pgd = make_pgd_solution_strategy(             //
///        backend, MyStepStrat{}, MyResidualStrat{}, cfg);  // Custom step/residual strategies
///    auto pgd_state = make_pgd_state(backend, x, grad, x_tmp, grad_tmp);
///
///    // Solve (can reuse "lcp" and "pgd" across many states)
///    auto result = solve_lcp(lcp, pgd, pgd_state);
/// \endcode
///
/// \param prob The linear complementarity problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy>
KOKKOS_INLINE_FUNCTION auto solve_lcp(const Problem& prob, const Strategy& strat, typename Strategy::state_t& state) ->
    typename Strategy::result_t {
  // Convert LCP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_HPP_