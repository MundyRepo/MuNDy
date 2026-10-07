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
/// invert needs the KokkosKernels TPL; without it this is an empty (trivially passing) translation unit.

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

// Mundy
#include <mundy_math/invert.hpp>  // for mundy::invert
#include <mundy_utils/rng.hpp>    // for mundy::make_philox

namespace mundy {

namespace {

using host_space = Kokkos::DefaultHostExecutionSpace;

template <class Scalar>
using HostMatrix = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

/// \brief A host matrix with the given n x n entries, listed row by row.
template <class Scalar>
HostMatrix<Scalar> make_matrix(size_t n, std::initializer_list<Scalar> rows) {
  HostMatrix<Scalar> matrix("matrix", n, n);
  size_t k = 0;
  for (const Scalar entry : rows) {
    matrix(k / n, k % n) = entry;
    ++k;
  }
  return matrix;
}

TEST(Invert, RandomMatrixTimesItsInverseIsTheIdentity) {
  // Gaussian random matrices are invertible with probability one.
  const size_t n = 10;
  HostMatrix<double> matrix("matrix", n, n);
  HostMatrix<double> original("original", n, n);
  openrand::Philox rng = make_philox(1234, 0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      original(i, j) = matrix(i, j) = rng.randn<double>();
    }
  }
  HostMatrix<double> inverse("inverse", n, n);
  invert(host_space(), matrix, inverse);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double product = 0.0;
      for (size_t k = 0; k < n; ++k) {
        product += original(i, k) * inverse(k, j);
      }
      EXPECT_NEAR(product, i == j ? 1.0 : 0.0, 1e-12) << "(A A^{-1})(" << i << ", " << j << ")";
    }
  }
}

TEST(Invert, SmallIntegerMatricesInvertExactly) {
  // [[2, 1], [1, 1]] has inverse [[1, -1], [-1, 2]], and every step of its factorization is exact.
  const HostMatrix<double> matrix = make_matrix<double>(2, {2.0, 1.0, 1.0, 1.0});
  HostMatrix<double> inverse("inverse", 2, 2);
  invert(host_space(), matrix, inverse);
  EXPECT_EQ(inverse(0, 0), 1.0);
  EXPECT_EQ(inverse(0, 1), -1.0);
  EXPECT_EQ(inverse(1, 0), -1.0);
  EXPECT_EQ(inverse(1, 1), 2.0);

  // The matrix is left holding its LU factors: U on and above the diagonal, L's multiplier 1/2 below it.
  EXPECT_EQ(matrix(0, 0), 2.0);
  EXPECT_EQ(matrix(0, 1), 1.0);
  EXPECT_EQ(matrix(1, 0), 0.5);
  EXPECT_EQ(matrix(1, 1), 0.5);
}

TEST(Invert, PivotsPastAZeroLeadingEntry) {
  // The swap [[0, 1], [1, 0]] cannot be factored without pivoting; it is its own inverse.
  const HostMatrix<double> matrix = make_matrix<double>(2, {0.0, 1.0, 1.0, 0.0});
  HostMatrix<double> inverse("inverse", 2, 2);
  invert(host_space(), matrix, inverse);
  EXPECT_EQ(inverse(0, 0), 0.0);
  EXPECT_EQ(inverse(0, 1), 1.0);
  EXPECT_EQ(inverse(1, 0), 1.0);
  EXPECT_EQ(inverse(1, 1), 0.0);
}

#if defined(KOKKOS_ENABLE_SERIAL)
TEST(Invert, RunsOnTheGivenExecutionSpace) {
  // Serial need not be the host views' default execution space; invert must still solve there.
  const HostMatrix<double> matrix = make_matrix<double>(2, {2.0, 1.0, 1.0, 1.0});
  HostMatrix<double> inverse("inverse", 2, 2);
  invert(Kokkos::Serial(), matrix, inverse);
  EXPECT_EQ(inverse(0, 0), 1.0);
  EXPECT_EQ(inverse(0, 1), -1.0);
  EXPECT_EQ(inverse(1, 0), -1.0);
  EXPECT_EQ(inverse(1, 1), 2.0);
}
#endif

TEST(Invert, InvertsFloatMatrices) {
  // [[4, 3], [6, 3]] has inverse [[-1/2, 1/2], [1, -2/3]].
  const HostMatrix<float> matrix = make_matrix<float>(2, {4.0f, 3.0f, 6.0f, 3.0f});
  HostMatrix<float> inverse("inverse", 2, 2);
  invert(host_space(), matrix, inverse);
  const float tolerance = 4 * std::numeric_limits<float>::epsilon();
  EXPECT_NEAR(inverse(0, 0), -0.5f, tolerance);
  EXPECT_NEAR(inverse(0, 1), 0.5f, tolerance);
  EXPECT_NEAR(inverse(1, 0), 1.0f, tolerance);
  EXPECT_NEAR(inverse(1, 1), -2.0f / 3.0f, tolerance);
}

TEST(Invert, ThrowsOnSingularMatrices) {
  HostMatrix<double> inverse("inverse", 2, 2);
  EXPECT_THROW(invert(host_space(), make_matrix<double>(2, {1.0, 2.0, 2.0, 4.0}), inverse), std::runtime_error)
      << "dependent rows";
  EXPECT_THROW(invert(host_space(), make_matrix<double>(2, {0.0, 0.0, 0.0, 0.0}), inverse), std::runtime_error)
      << "the zero matrix";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(invert(host_space(), make_matrix<double>(2, {1.0, nan, 0.0, 1.0}), inverse), std::runtime_error)
      << "a NaN entry";
}

TEST(Invert, ThrowsOnMismatchedExtents) {
  HostMatrix<double> square("square", 2, 2);
  HostMatrix<double> wide("wide", 2, 3);
  HostMatrix<double> larger("larger", 3, 3);
  EXPECT_THROW(invert(host_space(), wide, larger), std::invalid_argument) << "a non-square matrix";
  EXPECT_THROW(invert(host_space(), square, larger), std::invalid_argument) << "an inverse of the wrong size";
}

TEST(Invert, EmptyMatrixHasAnEmptyInverse) {
  HostMatrix<double> empty("empty", 0, 0);
  HostMatrix<double> inverse("inverse", 0, 0);
  EXPECT_NO_THROW(invert(host_space(), empty, inverse));
}

}  // namespace

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
