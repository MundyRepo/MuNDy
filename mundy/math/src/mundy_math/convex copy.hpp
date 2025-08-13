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

/// \brief Constrained quadratic programming problem (CQPP) formulation
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOperator, typename ConvexSpace>
struct CQPPProblem {
  using scalar_t = typename Backend::scalar_t;
  using vector_t = typename Backend::vector_t;

  const vector_t& q;
  const LinearOperator& A;
  const ConvexSpace& convex_space;
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
template <typename Backend, typename LinearOperator>
struct LCPProblem {
  using scalar_t = typename Backend::scalar_t;
  using vector_t = typename Backend::vector_t;

  const vector_t& q;
  const LinearOperator& A;
};  // LCPProblem

struct LinfNormProjectedGradient {
  template <typename Backend, typename ConvexSpace>
  KOKKOS_INLINE_FUNCTION typename Backend::scalar_t operator()([[maybe_unused]] const Backend& backend,  //
                                                               const typename Backend::vector_t& x,      //
                                                               const typename Backend::vector_t& grad,   //
                                                               const ConvexSpace& convex_space) const {
    size_t n = Backend::vector_size(x);
    scalar_t largest_abs_gradient;
    Backend::reduce_max(
        n,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          // perform the projection EQ 2.2 of Dai & Fletcher 2005
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i)

              scalar_t abs_projected_grad;
          if (x_i < get_zero_tolerance<scalar_t>()) {
            abs_projected_grad = convex_space.project(grad_i);
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

struct BBStepStrategy {
  template <typename Backend>
  KOKKOS_INLINE_FUNCTION
  typename Backend::scalar_t operator()([[maybe_unused]] const Backend& backend,  //
                                        const typename Backend::vector_t& x_old,
                                        const typename Backend::vector_t& grad_old,  //
                                        const typename Backend::vector_t& x,
                                        const typename Backend::vector_t& grad) const {
    using scalar_t = typename Backend::scalar_t;

    scalar_t num = Backend::diff_dot(x - x_old, grad - grad_old);
    scalar_t denom = Backend::diff_dot(grad - grad_old, grad - grad_old);
    constexpr scalar_t eps = get_zero_tolerance<scalar_t>() * 10;
    denom += eps * (Kokkos::abs(denom) < eps);
    return num / denom;
  }
};  // BBStepStrategy

template <typename Backend, typename StepStrategy>
struct PGDSolutionStrategy {
  using vector_t = typename Backend::vector_t;
  using scalar_t = typename Backend::scalar_t;

  StepStrategy step_size;
  vector_t& x_tmp;
  vector_t& grad_tmp;
  vector_t& x;
  vector_t& grad;
  unsigned max_iter;
  scalar_t tol;
};  // PGDSolutionStrategy

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

  static scalar_t vector_data(const vector_t& x, size_t i) {
    return x(i);
  }

  static void deep_copy(vector_t& dest, const vector_t& src) {
    Kokkos::deep_copy(dest, src);
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

  template <typename Functor>
  static void reduce_max(size_t n, const Functor& func, scalar_t& result) {
    Kokkos::parallel_reduce(
        "reduce_max", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) { func(i, max_val); }, Kokkos::Max<scalar_t>(result));
  }
};  // KokkosBackend

template <typename Scalar, size_t N>
struct MundyMathBackend {
  using scalar_t = Scalar;
  using vector_t = Vector<scalar_t, N>;

  KOKKOS_INLINE_FUNCTION static size_t vector_size(const vector_t& x) {
    return vector_t::size;
  }

  KOKKOS_INLINE_FUNCTION static scalar_t vector_data(const vector_t& x, size_t i) {
    return x[i];
  }

  KOKKOS_INLINE_FUNCTION static void deep_copy(vector_t& dest, const vector_t& src) {
    dest = src;
  }

  KOKKOS_INLINE_FUNCTION static void axpby(const scalar_t alpha, const vector_t& x, const scalar_t beta, vector_t& y) {
    y = alpha * x + beta * y;
  }

  template <typename Wrapper>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const scalar_t alpha, const vector_t& x, const scalar_t beta,
                                                    const vector_t& y, vector_t& z, const Wrapper& wrapper) {
    z = apply(wrapper, alpha * x + beta * y);
  }

  KOKKOS_INLINE_FUNCTION static scalar_t diff_dot(const vector_t& x, const vector_t& y) {
    return dot(x - y, x - y);
  }

  template <class Functor, class T>
  KOKKOS_INLINE_FUNCTION static void reduce_max(size_t n, const Functor& func, scalar_t& result) {
    MUNDY_THROW_ASSERT(n == N, std::invalid_argument, "reduce_max: n must match the size of the vector.");
    reduce_max_impl(std::make_index_sequence<N>{}, func, result);
  }

 private:
  template <size_t Is..., class Functor>
  KOKKOS_INLINE_FUNCTION static void reduce_max_impl(std::index_sequence<Is...>, const Functor& func,
                                                     scalar_t& result) {
    scalar_t max_val = std::numeric_limits<T>::lowest();
    ((func(Is, max_val)), ...);
    result = max_val;
  }
};  // MundyMathBackend

namespace space {

// These are 1d convex spaces, which will be applied to each element of a vector assuming a separable convex space
template <typename Scalar>
struct Unconstrained {
  using scalar_t = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return x;
  }
};

template <typename Backend>
struct LowerBound {
  using scalar_t = Scalar;

  scalar_t lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::max(x, lower_bound);
  }
};

template <typename Backend>
struct UpperBound {
  using scalar_t = Scalar;

  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(x, upper_bound);
  }
};

template <typename Backend>
struct Bounded {
  using scalar_t = Scalar;

  scalar_t lower_bound;
  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(Kokkos::max(x, lower_bound), upper_bound);
  }
};

}  // namespace space

/// \brief Solve a constrained quadratic programming problem
/// The form of the problem is:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in Omega subset of R^n
/// where Omega is the convex feasible set with the property that proj(y) in Omega for all y in R^n.
template <typename Backend, typename ConvexSpace, typename ResidueStrategy, typename SolutionStrategy,
          typename LinearOperator>
KOKKOS_INLINE_FUNCTION int abstract_solve_cqpp(
    const Backend& backend,                     //
    const ResidueStrategy& residue_strategy,    //
    const SolutionStrategy& solution_strategy,  //
    const convex::CQPPProblem<Backend, LinearOperator, ConvexSpace>& problem) {
  unsigned iteration_count = 0;

  // Compute the initial gradient
  auto& x = solution_strategy.x;
  auto& grad = solution_strategy.grad;
  auto& x_tmp = solution_strategy.x_tmp;
  auto& grad_tmp = solution_strategy.grad_tmp;

  problem.A.apply(x_tmp, grad_tmp);
  Backend::axpby(1.0, x_tmp, 1.0, problem.q);  // grad_tmp = A x_tmp + q

  // Compute the initial residue
  Scalar residue = residue_strategy(x_tmp, grad_tmp);
  MUNDY_THROW_ASSERT(residue > 0, std::runtime_error, "solve_cqpp: Residue may not be negative.");
  if (residue < tol) {
    // The initial guess was correct, nothing more is needed
    Backend::deep_copy(x, x_tmp);
    Backend::deep_copy(grad, grad_tmp);
    return 0;
  } else {
    // Initial guess insufficient, iterate
    // First step, Dai&Fletcher2005 Section 5.
    Scalar step_size = static_cast<Scalar>(1.0) / residue;
    while (iteration_count < solution_strategy.max_iter) {
      Backend::wrapped_axpbyz(step_size, x_tmp, static_cast<Scalar>(1.0), grad_tmp, x, problem.convex_space);

      problem.A.apply(x, grad);
      Backend::axpby(1.0, x, 1.0, problem.q);  // grad = A x + q

      residue = residue_strategy(x, grad);
      if (residue < tol) {
        return iteration_count;
      }
      step_size = solution_strategy.step_size(x_tmp, grad_tmp, x, grad);

      Backend::deep_copy(x_tmp, x);
      Backend::deep_copy(grad_tmp, grad);
      ++iteration_count;
    }
  }

  // If we get here, we failed to converge
  return -1;
}

/// \brief Solve a linear complementarity problem
/// The form of the problem is:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
template <typename Backend, typename ResidueStrategy, typename SolutionStrategy, typename LinearOperator>
KOKKOS_INLINE_FUNCTION int abstract_solve_lcp(const Backend& backend,                     //
                                              const ResidueStrategy& residue_strategy,    //
                                              const SolutionStrategy& solution_strategy,  //
                                              const convex::LCPProblem<Backend, LinearOperator>& problem) {
  convex::space::LowerBound<Backend> lower_bound(0.0);
  convex::CCPPProblem<Backend, LinearOperator, convex::space::LowerBound<Backend>> cqpp_problem(problem.q, problem.A,
                                                                                                lower_bound);
  return abstract_solve_cqpp(backend, lower_bound, residue_strategy, solution_strategy, cqpp_problem);
}

}  // namespace convex

/// \brief Solve a constrained quadratic programming problem (with MundyMath as the backend)
///
/// This relies on the default implementation, which uses projected gradient descent to solve the problem.
///
/// The form of the problem is:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in Omega subset of R^n
/// where Omega is the convex feasible set with the property that proj(y) in Omega for all y in R^n.
template <typename Scalar, size_t N,  // N is the size of the vector space
          typename ConvexSpace, typename ResidueStrategy, typename SolutionStrategy, typename LinearOperator>
KOKKOS_INLINE_FUNCTION int solve_ccqp(const Vector<Scalar, N>& q, const LinearOperator& A,  //
                                      const ConvexSpace& convex_space,                      //
                                      Vector<Scalar, N>& x, unsigned max_iter = 1000,       //
                                      Scalar tol = mundy::math::get_relaxed_zero_tolerance<Scalar>()) {
  using Backend = mundy::math::convex::MundyMathBackend<Scalar, N>;
  using Problem = mundy::math::convex::CQPPProblem<Backend, LinearOperator, ConvexSpace>;
  using ResidueStrategyType = mundy::math::convex::LinfNormProjectedGradient;
  using SolutionStrategyType = mundy::math::convex::PGDSolutionStrategy<Backend, mundy::math::convex::BBStepStrategy>;

  // Prepare the problem
  Backend backend;
  Problem problem{q, A, convex_space};
  ResidueStrategyType residue_strategy;
  Vector<Scalar, N> x_tmp, grad_tmp, grad;
  SolutionStrategyType solution_strategy{
      mundy::math::convex::BBStepStrategy{}, x_tmp, grad_tmp, x, grad, max_iter, tol};

  // Solve the problem
  return mundy::math::convex::abstract_solve_cqpp(Backend{}, residue_strategy, solution_strategy, problem);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_HPP_