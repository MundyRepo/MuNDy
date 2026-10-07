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

/// \file UnitTestInvert.cpp
/// \brief The dense inverse of a Kokkos view (invert.hpp): exact small inverses, pivoting, LU factors, and failures.
///
/// Every case runs on host matrices and on matrices in the default execution space's memory. A case that factors a
/// matrix is skipped where this build cannot invert matrices in that memory (invert_is_available_v). invert needs the
/// KokkosKernels TPL; without it this is an empty (trivially passing) translation unit.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// Mundy
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS

#if defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

// C++ core
#include <initializer_list>  // for std::initializer_list
#include <limits>            // for std::numeric_limits
#include <stdexcept>         // for std::invalid_argument, std::runtime_error
#include <type_traits>       // for std::conditional_t, std::is_same_v

// Mundy
#include <mundy_math/invert.hpp>  // for mundy::{invert, invert_is_available_v}
#include <mundy_utils/rng.hpp>    // for mundy::make_philox

namespace mundy {

namespace {

/// \brief An n x n matrix in Space's memory with the given entries, listed row by row.
template <class Space, class Scalar>
Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename Space::memory_space> make_matrix(
    size_t n, std::initializer_list<Scalar> rows) {
  Kokkos::View<Scalar**, Kokkos::LayoutLeft, typename Space::memory_space> matrix("matrix", n, n);
  const auto matrix_h = Kokkos::create_mirror_view(matrix);
  size_t k = 0;
  for (const Scalar entry : rows) {
    matrix_h(k / n, k % n) = entry;
    ++k;
  }
  Kokkos::deep_copy(matrix, matrix_h);
  return matrix;
}

/// \brief A host copy of a view, for reading its entries.
template <class View>
auto on_host(const View& view) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}

/// \brief The host execution space, and the default one where its memory is not the host's.
using TestSpaces =
    std::conditional_t<std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>,
                       ::testing::Types<Kokkos::DefaultHostExecutionSpace>,
                       ::testing::Types<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>>;

template <class Space>
class Invert : public ::testing::Test {};
TYPED_TEST_SUITE(Invert, TestSpaces);

TYPED_TEST(Invert, RandomMatrixTimesItsInverseIsTheIdentity) {
  using Space = TypeParam;
  if constexpr (!invert_is_available_v<typename Space::memory_space>) {
    GTEST_SKIP() << "this build cannot invert matrices in " << Space::memory_space::name() << ".";
  }

  // Gaussian random matrices are invertible with probability one. original is its own allocation, never an alias of a
  // host matrix, since invert overwrites matrix with its LU factors.
  const size_t n = 10;
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> matrix("matrix", n, n);
  const Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> original("original", n, n);
  openrand::Philox rng = make_philox(1234, 0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      original(i, j) = rng.randn<double>();
    }
  }
  Kokkos::deep_copy(matrix, original);
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", n, n);
  invert(Space(), matrix, inverse);

  const auto inverse_h = on_host(inverse);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double product = 0.0;
      for (size_t k = 0; k < n; ++k) {
        product += original(i, k) * inverse_h(k, j);
      }
      EXPECT_NEAR(product, i == j ? 1.0 : 0.0, 1e-12) << "(A A^{-1})(" << i << ", " << j << ")";
    }
  }
}

TYPED_TEST(Invert, SmallIntegerMatricesInvertExactly) {
  using Space = TypeParam;
  if constexpr (!invert_is_available_v<typename Space::memory_space>) {
    GTEST_SKIP() << "this build cannot invert matrices in " << Space::memory_space::name() << ".";
  }

  // [[2, 1], [1, 1]] has inverse [[1, -1], [-1, 2]], and every step of its factorization is exact.
  const auto matrix = make_matrix<Space, double>(2, {2.0, 1.0, 1.0, 1.0});
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", 2, 2);
  invert(Space(), matrix, inverse);
  const auto inverse_h = on_host(inverse);
  EXPECT_EQ(inverse_h(0, 0), 1.0);
  EXPECT_EQ(inverse_h(0, 1), -1.0);
  EXPECT_EQ(inverse_h(1, 0), -1.0);
  EXPECT_EQ(inverse_h(1, 1), 2.0);

  // The matrix is left holding its LU factors: U on and above the diagonal, L's multiplier 1/2 below it.
  const auto matrix_h = on_host(matrix);
  EXPECT_EQ(matrix_h(0, 0), 2.0);
  EXPECT_EQ(matrix_h(0, 1), 1.0);
  EXPECT_EQ(matrix_h(1, 0), 0.5);
  EXPECT_EQ(matrix_h(1, 1), 0.5);
}

TYPED_TEST(Invert, PivotsPastAZeroLeadingEntry) {
  using Space = TypeParam;
  if constexpr (!invert_is_available_v<typename Space::memory_space>) {
    GTEST_SKIP() << "this build cannot invert matrices in " << Space::memory_space::name() << ".";
  }

  // The swap [[0, 1], [1, 0]] cannot be factored without pivoting; it is its own inverse.
  const auto matrix = make_matrix<Space, double>(2, {0.0, 1.0, 1.0, 0.0});
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", 2, 2);
  invert(Space(), matrix, inverse);
  const auto inverse_h = on_host(inverse);
  EXPECT_EQ(inverse_h(0, 0), 0.0);
  EXPECT_EQ(inverse_h(0, 1), 1.0);
  EXPECT_EQ(inverse_h(1, 0), 1.0);
  EXPECT_EQ(inverse_h(1, 1), 0.0);
}

TYPED_TEST(Invert, InvertsFloatMatrices) {
  using Space = TypeParam;
  if constexpr (!invert_is_available_v<typename Space::memory_space>) {
    GTEST_SKIP() << "this build cannot invert matrices in " << Space::memory_space::name() << ".";
  }

  // [[4, 3], [6, 3]] has inverse [[-1/2, 1/2], [1, -2/3]].
  const auto matrix = make_matrix<Space, float>(2, {4.0f, 3.0f, 6.0f, 3.0f});
  Kokkos::View<float**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", 2, 2);
  invert(Space(), matrix, inverse);
  const auto inverse_h = on_host(inverse);
  const float tolerance = 4 * std::numeric_limits<float>::epsilon();
  EXPECT_NEAR(inverse_h(0, 0), -0.5f, tolerance);
  EXPECT_NEAR(inverse_h(0, 1), 0.5f, tolerance);
  EXPECT_NEAR(inverse_h(1, 0), 1.0f, tolerance);
  EXPECT_NEAR(inverse_h(1, 1), -2.0f / 3.0f, tolerance);
}

TYPED_TEST(Invert, ThrowsOnSingularMatrices) {
  using Space = TypeParam;
  if constexpr (!invert_is_available_v<typename Space::memory_space>) {
    GTEST_SKIP() << "this build cannot invert matrices in " << Space::memory_space::name() << ".";
  }

  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", 2, 2);
  EXPECT_THROW(invert(Space(), make_matrix<Space, double>(2, {1.0, 2.0, 2.0, 4.0}), inverse), std::runtime_error)
      << "dependent rows";
  EXPECT_THROW(invert(Space(), make_matrix<Space, double>(2, {0.0, 0.0, 0.0, 0.0}), inverse), std::runtime_error)
      << "the zero matrix";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(invert(Space(), make_matrix<Space, double>(2, {1.0, nan, 0.0, 1.0}), inverse), std::runtime_error)
      << "a NaN entry";
}

TYPED_TEST(Invert, ThrowsOnMismatchedExtents) {
  using Space = TypeParam;
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> square("square", 2, 2);
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> wide("wide", 2, 3);
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> larger("larger", 3, 3);
  EXPECT_THROW(invert(Space(), wide, larger), std::invalid_argument) << "a non-square matrix";
  EXPECT_THROW(invert(Space(), square, larger), std::invalid_argument) << "an inverse of the wrong size";
}

TYPED_TEST(Invert, EmptyMatrixHasAnEmptyInverse) {
  using Space = TypeParam;
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> empty("empty", 0, 0);
  Kokkos::View<double**, Kokkos::LayoutLeft, typename Space::memory_space> inverse("inverse", 0, 0);
  EXPECT_NO_THROW(invert(Space(), empty, inverse));
}

#if defined(KOKKOS_ENABLE_SERIAL)
TEST(InvertOnSerial, RunsOnTheGivenExecutionSpace) {
  if constexpr (!invert_is_available_v<Kokkos::HostSpace>) {
    GTEST_SKIP() << "this build cannot invert matrices in HostSpace.";
  }

  // Serial need not be the host views' default execution space; invert must still solve there.
  const auto matrix = make_matrix<Kokkos::Serial, double>(2, {2.0, 1.0, 1.0, 1.0});
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> inverse("inverse", 2, 2);
  invert(Kokkos::Serial(), matrix, inverse);
  EXPECT_EQ(inverse(0, 0), 1.0);
  EXPECT_EQ(inverse(0, 1), -1.0);
  EXPECT_EQ(inverse(1, 0), -1.0);
  EXPECT_EQ(inverse(1, 1), 2.0);
}
#endif

}  // namespace

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
