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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

#include <Kokkos_Core.hpp>  // for Kokkos::Array

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_math/Vector.hpp>    // for mundy::math::Vector
#include <mundy_math/minimize.hpp>  // for mundy::math::find_min_using_approximate_derivatives

namespace mundy {

namespace math {

namespace {

KOKKOS_INLINE_FUNCTION double quadratic1(const Vector<double, 2>& x) {
  return x[0] * x[0] + x[1] * x[1];
};

KOKKOS_INLINE_FUNCTION double quadratic2(const Vector<double, 2>& x) {
  return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] + 1.0) * (x[1] + 1.0);
};

template <size_t N>
KOKKOS_INLINE_FUNCTION double rosenbrock(const Vector<double, N>& x) {
  double sum = 0.0;
  for (size_t i = 0; i < N - 1; ++i) {
    sum += 2.0 * std::pow(x[i + 1] - x[i] * x[i], 2.0) + std::pow(1.0 - x[i], 2.0);
  }
  return sum;
};

TEST(Minimize, SimpleFunctions) {
  constexpr size_t lbfgs_max_memory_size = 10;
  const double min_objective_delta = 1e-7;

  // Simple quadratic function
  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(quadratic1, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 0.0, min_objective_delta);
    EXPECT_NEAR(x[1], 0.0, min_objective_delta);
  }

  // Simple shifted quadratic function
  {
    Vector<double, 2> x = {1.0, 1.0};
    double min_cost = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(quadratic2, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, min_objective_delta);
    EXPECT_NEAR(x[0], 2.0, min_objective_delta);
    EXPECT_NEAR(x[1], -1.0, min_objective_delta);
  }
}

TEST(Minimize, ComplexFunctions) {
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const double min_objective_delta = 1e-7;
  const double test_tolerance = std::sqrt(min_objective_delta);
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr size_t N = 42;

  // N-dimensional Rosenbrock function
  {
    Vector<double, N> x = {0.0};
    double min_cost =
        find_min_using_approximate_derivatives<lbfgs_max_memory_size>(rosenbrock<N>, x, min_objective_delta);
    EXPECT_NEAR(min_cost, 0.0, test_tolerance);
    for (size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(x[i], 1.0, test_tolerance);
    }
  }
}

}  // namespace

}  // namespace math

}  // namespace mundy
