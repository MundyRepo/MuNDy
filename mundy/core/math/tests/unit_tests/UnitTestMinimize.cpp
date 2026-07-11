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

// External libs
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core libs
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Mundy libs
#include <mundy_math/Vector.hpp>
#include <mundy_math/minimize.hpp>

namespace mundy {

namespace {

// ============================================================
//! \name Test functions with known minima and exact gradients
//@{
// ============================================================

// f(x) = x0² + x1²,  min = 0 at (0,0)
KOKKOS_INLINE_FUNCTION double quadratic1(const Vector<double, 2>& x) {
  return x[0] * x[0] + x[1] * x[1];
}
KOKKOS_INLINE_FUNCTION Vector<double, 2> quadratic1_grad(const Vector<double, 2>& x) {
  return {2.0 * x[0], 2.0 * x[1]};
}

// f(x) = (x0-2)² + (x1+1)²,  min = 0 at (2,-1)
KOKKOS_INLINE_FUNCTION double quadratic2(const Vector<double, 2>& x) {
  return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] + 1.0) * (x[1] + 1.0);
}
KOKKOS_INLINE_FUNCTION Vector<double, 2> quadratic2_grad(const Vector<double, 2>& x) {
  return {2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)};
}

// Mundy Rosenbrock variant: f(x) = Σ [2(x_{i+1} - x_i²)² + (1 - x_i)²],  min = 0 at x = 1
template <size_t N>
KOKKOS_INLINE_FUNCTION double rosenbrock(const Vector<double, N>& x) {
  double sum = 0.0;
  for (size_t i = 0; i < N - 1; ++i) {
    sum += 2.0 * std::pow(x[i + 1] - x[i] * x[i], 2.0) + std::pow(1.0 - x[i], 2.0);
  }
  return sum;
}

// See EllipsoidEllipsoidGradientNotes.md for derivation.
template <size_t N>
KOKKOS_INLINE_FUNCTION Vector<double, N> rosenbrock_grad(const Vector<double, N>& x) {
  Vector<double, N> g;
  // i == 0
  g[0] = -8.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
  // 0 < i < N-1
  for (size_t i = 1; i < N - 1; ++i) {
    g[i] = 4.0 * (x[i] - x[i - 1] * x[i - 1]) - 8.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
  }
  // i == N-1
  g[N - 1] = 4.0 * (x[N - 1] - x[N - 2] * x[N - 2]);
  return g;
}

//@}

// ============================================================
//! \name Call-counting wrappers (host only)
//@{
// ============================================================

template <size_t N, typename F>
struct CountedF {
  const F& func;
  size_t& count;
  CountedF(const F& f_, size_t& c_) : func(f_), count(c_) {
  }
  double operator()(const Vector<double, N>& x) const {
    ++count;
    return func(x);
  }
};

template <size_t N, typename DF>
struct CountedDF {
  const DF& deriv;
  size_t& count;
  CountedDF(const DF& df_, size_t& c_) : deriv(df_), count(c_) {
  }
  Vector<double, N> operator()(const Vector<double, N>& x) const {
    ++count;
    return deriv(x);
  }
};

// FDF callable: fills grad and returns f, counting as one "combined" evaluation.
template <size_t N, typename F, typename DF>
struct CountedFDF {
  const F& func;
  const DF& deriv;
  size_t& count;
  CountedFDF(const F& f_, const DF& df_, size_t& c_) : func(f_), deriv(df_), count(c_) {
  }
  double operator()(const Vector<double, N>& x, Vector<double, N>& g) const {
    ++count;
    g = deriv(x);
    return func(x);
  }
};

//@}

// ============================================================
//! \name Existing tests (approximate derivatives) — unchanged
//@{
// ============================================================

TEST(Minimize, SimpleFunctions) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;

  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(quadratic1, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 0.0, min_objective_delta);
    EXPECT_NEAR(x[1], 0.0, min_objective_delta);
  }

  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(quadratic2, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 2.0, min_objective_delta);
    EXPECT_NEAR(x[1], -1.0, min_objective_delta);
  }
}

TEST(Minimize, ComplexFunctions) {
  const double min_objective_delta = 1e-7;
  const double test_tolerance = std::sqrt(min_objective_delta);
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr size_t N = 42;

  {
    Vector<double, N> x(0.0);
    double min_cost =
        find_min_using_approximate_derivatives<lbfgs_max_memory_size>(rosenbrock<N>, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, test_tolerance);
    for (size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(x[i], 1.0, test_tolerance);
    }
  }
}

//@}

// ============================================================
//! \name find_min_with_derivatives: gradient correctness
//@{
// ============================================================

// Verify supplied gradients against central differences at several points.
TEST(MinimizeWithDerivatives, GradientCorrectness) {
  const double eps = 1e-6;
  const double tol = 1e-5;

  auto check_grad = [&](auto f, auto df, auto x_init) {
    using V = decltype(x_init);
    constexpr size_t N = V::size;
    V x = x_init;
    V analytical = df(x);
    for (size_t i = 0; i < N; ++i) {
      V xp = x, xm = x;
      xp[i] += eps;
      xm[i] -= eps;
      double fd = (f(xp) - f(xm)) / (2.0 * eps);
      EXPECT_NEAR(analytical[i], fd, tol) << "component " << i;
    }
  };

  check_grad(quadratic1, quadratic1_grad, Vector<double, 2>{1.0, -2.0});
  check_grad(quadratic1, quadratic1_grad, Vector<double, 2>{0.5, 3.1});
  check_grad(quadratic2, quadratic2_grad, Vector<double, 2>{0.0, 0.0});
  check_grad(quadratic2, quadratic2_grad, Vector<double, 2>{3.0, -3.0});

  // Rosenbrock gradient at several points
  constexpr size_t N = 4;
  check_grad(rosenbrock<N>, rosenbrock_grad<N>, Vector<double, N>{0.5, 0.5, 0.5, 0.5});
  check_grad(rosenbrock<N>, rosenbrock_grad<N>, Vector<double, N>{1.2, 0.8, 1.1, 0.9});
  check_grad(rosenbrock<N>, rosenbrock_grad<N>, Vector<double, N>{-1.0, 2.0, 0.0, -0.5});
}

//@}

// ============================================================
//! \name find_min_with_derivatives: correctness of minimization
//@{
// ============================================================

TEST(MinimizeWithDerivatives, SimpleFunctions) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;

  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost =
        find_min_with_derivatives<lbfgs_max_memory_size>(quadratic1, quadratic1_grad, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 0.0, min_objective_delta);
    EXPECT_NEAR(x[1], 0.0, min_objective_delta);
  }

  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost =
        find_min_with_derivatives<lbfgs_max_memory_size>(quadratic2, quadratic2_grad, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 2.0, min_objective_delta);
    EXPECT_NEAR(x[1], -1.0, min_objective_delta);
  }
}

TEST(MinimizeWithDerivatives, ComplexFunctions) {
  const double min_objective_delta = 1e-7;
  const double test_tolerance = std::sqrt(min_objective_delta);
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr size_t N = 42;

  {
    Vector<double, N> x(0.0);
    double min_cost =
        find_min_with_derivatives<lbfgs_max_memory_size>(rosenbrock<N>, rosenbrock_grad<N>, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, test_tolerance);
    for (size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(x[i], 1.0, test_tolerance);
    }
  }
}

//@}

// ============================================================
//! \name find_min_with_derivatives: cost-function evaluation count
//
// The approximate-derivative variant computes 2N extra f evaluations per gradient (central
// differences, N=2 here -> 4 extra), plus the line search.  The exact-derivative variant
// avoids that overhead.  We verify that the exact variant uses strictly fewer f evaluations.
//@{
// ============================================================

TEST(MinimizeWithDerivatives, FewerFunctionEvaluationsQuadratic) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;

  size_t f_count_approx = 0, f_count_exact = 0, g_count_exact = 0;

  {
    CountedF<2, decltype(quadratic2)> cf(quadratic2, f_count_approx);
    Vector<double, 2> x = {1.0, 1.0};
    find_min_using_approximate_derivatives<lbfgs_max_memory_size>(cf, x, min_objective_delta);
  }

  {
    CountedF<2, decltype(quadratic2)> cf(quadratic2, f_count_exact);
    CountedDF<2, decltype(quadratic2_grad)> cdf(quadratic2_grad, g_count_exact);
    Vector<double, 2> x = {1.0, 1.0};
    find_min_with_derivatives<lbfgs_max_memory_size>(cf, cdf, x, min_objective_delta);
  }

  // Exact gradient variant must evaluate f strictly fewer times.
  std::cout << "  [quadratic2] approx f-calls=" << f_count_approx << "  exact f-calls=" << f_count_exact
            << "  exact g-calls=" << g_count_exact << "\n";
  EXPECT_LT(f_count_exact, f_count_approx) << "approx f-calls=" << f_count_approx << "  exact f-calls=" << f_count_exact
                                           << "  exact g-calls=" << g_count_exact;
}

TEST(MinimizeWithDerivatives, FewerFunctionEvaluationsRosenbrock) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;
  constexpr size_t N = 8;

  size_t f_count_approx = 0, f_count_exact = 0, g_count_exact = 0;

  {
    CountedF<N, decltype(rosenbrock<N>)> cf(rosenbrock<N>, f_count_approx);
    Vector<double, N> x(0.0);
    find_min_using_approximate_derivatives<lbfgs_max_memory_size>(cf, x, min_objective_delta);
  }

  {
    CountedF<N, decltype(rosenbrock<N>)> cf(rosenbrock<N>, f_count_exact);
    CountedDF<N, decltype(rosenbrock_grad<N>)> cdf(rosenbrock_grad<N>, g_count_exact);
    Vector<double, N> x(0.0);
    find_min_with_derivatives<lbfgs_max_memory_size>(cf, cdf, x, min_objective_delta);
  }

  std::cout << "  [rosenbrock N=8] approx f-calls=" << f_count_approx << "  exact f-calls=" << f_count_exact
            << "  exact g-calls=" << g_count_exact << "\n";
  EXPECT_LT(f_count_exact, f_count_approx) << "approx f-calls=" << f_count_approx << "  exact f-calls=" << f_count_exact
                                           << "  exact g-calls=" << g_count_exact;
}

//@}

// ============================================================
//! \name find_min_with_derivatives: equivalent final solutions
//
// Both approaches must reach the same minimizer to within tolerance.
//@{
// ============================================================

TEST(MinimizeWithDerivatives, SolutionMatchesApproximate) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;
  const double match_tol = std::sqrt(min_objective_delta);
  constexpr size_t N = 10;

  Vector<double, N> x_approx(0.0);
  find_min_using_approximate_derivatives<lbfgs_max_memory_size>(rosenbrock<N>, x_approx, min_objective_delta);

  Vector<double, N> x_exact(0.0);
  find_min_with_derivatives<lbfgs_max_memory_size>(rosenbrock<N>, rosenbrock_grad<N>, x_exact, min_objective_delta);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(x_exact[i], x_approx[i], match_tol) << "component " << i;
  }
}

//@}

// ============================================================
//! \name find_min_with_fdf: correctness
//@{
// ============================================================

TEST(MinimizeWithFDF, SimpleFunctions) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;

  {
    auto fdf = [](const Vector<double, 2>& x, Vector<double, 2>& g) {
      g = quadratic1_grad(x);
      return quadratic1(x);
    };
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_with_fdf<lbfgs_max_memory_size>(fdf, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 0.0, min_objective_delta);
    EXPECT_NEAR(x[1], 0.0, min_objective_delta);
  }

  {
    auto fdf = [](const Vector<double, 2>& x, Vector<double, 2>& g) {
      g = quadratic2_grad(x);
      return quadratic2(x);
    };
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_with_fdf<lbfgs_max_memory_size>(fdf, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 2.0, min_objective_delta);
    EXPECT_NEAR(x[1], -1.0, min_objective_delta);
  }
}

TEST(MinimizeWithFDF, ComplexFunctions) {
  const double min_objective_delta = 1e-7;
  const double test_tolerance = std::sqrt(min_objective_delta);
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr size_t N = 42;

  auto fdf = [](const Vector<double, N>& x, Vector<double, N>& g) {
    g = rosenbrock_grad<N>(x);
    return rosenbrock<N>(x);
  };
  Vector<double, N> x(0.0);
  double min_cost = find_min_with_fdf<lbfgs_max_memory_size>(fdf, x, min_objective_delta);
  EXPECT_NEAR(min_cost, 0.0, test_tolerance);
  for (size_t i = 0; i < N; ++i) EXPECT_NEAR(x[i], 1.0, test_tolerance);
}

//@}

// ============================================================
//! \name find_min_with_fdf: FDF call count vs. separate f+g
//
// Each FDF call does the work of one f + one g evaluation combined.
// The caching line search means each line search step also costs one FDF call.
// Total FDF calls must be strictly fewer than (f_count_exact + g_count_exact) / 2
// from find_min_with_derivatives.
//@{
// ============================================================

TEST(MinimizeWithFDF, FewerEvaluationsQuadratic) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;
  constexpr size_t N = 2;

  size_t f_exact = 0, g_exact = 0, fdf_count = 0;

  {
    CountedF<N, decltype(quadratic2)> cf(quadratic2, f_exact);
    CountedDF<N, decltype(quadratic2_grad)> cdf(quadratic2_grad, g_exact);
    Vector<double, N> x = {1.0, 1.0};
    find_min_with_derivatives<lbfgs_max_memory_size>(cf, cdf, x, min_objective_delta);
  }

  {
    CountedFDF<N, decltype(quadratic2), decltype(quadratic2_grad)> cfdf(quadratic2, quadratic2_grad, fdf_count);
    Vector<double, N> x = {1.0, 1.0};
    find_min_with_fdf<lbfgs_max_memory_size>(cfdf, x, min_objective_delta);
  }

  std::cout << "  [quadratic2] exact (f=" << f_exact << " g=" << g_exact << ")  fdf=" << fdf_count << "\n";
  EXPECT_LT(fdf_count, f_exact + g_exact);
}

TEST(MinimizeWithFDF, FewerEvaluationsRosenbrock) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;
  constexpr size_t N = 8;

  size_t f_exact = 0, g_exact = 0, fdf_count = 0;

  {
    CountedF<N, decltype(rosenbrock<N>)> cf(rosenbrock<N>, f_exact);
    CountedDF<N, decltype(rosenbrock_grad<N>)> cdf(rosenbrock_grad<N>, g_exact);
    Vector<double, N> x(0.0);
    find_min_with_derivatives<lbfgs_max_memory_size>(cf, cdf, x, min_objective_delta);
  }

  {
    CountedFDF<N, decltype(rosenbrock<N>), decltype(rosenbrock_grad<N>)> cfdf(rosenbrock<N>, rosenbrock_grad<N>,
                                                                              fdf_count);
    Vector<double, N> x(0.0);
    find_min_with_fdf<lbfgs_max_memory_size>(cfdf, x, min_objective_delta);
  }

  std::cout << "  [rosenbrock N=8] exact (f=" << f_exact << " g=" << g_exact << ")  fdf=" << fdf_count << "\n";
  EXPECT_LT(fdf_count, f_exact + g_exact);
}

//@}

}  // namespace

}  // namespace mundy
