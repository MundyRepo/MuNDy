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

/// \file UnitTestCmath.cpp
/// \brief The cmath.hpp dispatch: sqrt in constant expressions agrees bit for bit with the hardware square root.
///
/// mundy::sqrt takes impl::constexpr_sqrt in a constant expression and Kokkos::sqrt at run time. The run-time tests
/// call impl::constexpr_sqrt directly, so they check the compile-time branch against the hardware.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>    // for std::isnan, std::nextafter
#include <cstdint>  // for std::uint32_t, std::uint64_t
#include <limits>   // for std::numeric_limits
#include <random>   // for std::mt19937_64

// Mundy
#include <mundy_math/AutoDiffScalar.hpp>  // for mundy::AutoDiffScalar
#include <mundy_math/Vector.hpp>          // for mundy::Vector, mundy::norm
#include <mundy_math/cmath.hpp>           // for mundy::sqrt, mundy::bit_cast, mundy::impl::constexpr_sqrt

namespace mundy {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();

//! \name Group 0: compile-time checks
//@{

// sqrt of float and double is a constant expression, correctly rounded over the whole range.
static_assert(sqrt(4.0) == 2.0);
static_assert(sqrt(2.0) == 0x1.6a09e667f3bcdp+0);
static_assert(sqrt(2.0f) == 0x1.6a09e6p+0f);
static_assert(sqrt(0x1p-1074) == 0x1p-537, "The smallest subnormal.");
static_assert(sqrt(0x1p-1073) == 0x1.6a09e667f3bcdp-537);
static_assert(sqrt(kMax) == 0x1.fffffffffffffp+511);

// IEEE special values: +-0 and +infinity are their own roots, and negative numbers and NaN give NaN.
static_assert(bit_cast<std::uint64_t>(sqrt(-0.0)) == bit_cast<std::uint64_t>(-0.0));
static_assert(sqrt(kInfinity) == kInfinity);
static_assert(sqrt(-1.0) != sqrt(-1.0) && sqrt(-kInfinity) != sqrt(-kInfinity));

// So are the MundyMath functions built on it: norms, and the chain rule of autodiff scalars.
static_assert(norm(Vector<double, 3>{3.0, 4.0, 12.0}) == 13.0);
static_assert(sqrt(AutoDiffScalar<double, 1>(4.0, 0)).value() == 2.0);
static_assert(sqrt(AutoDiffScalar<double, 1>(4.0, 0)).derivatives()[0] == 0.25);
//@}

/// \brief Whether impl::constexpr_sqrt(a) and the hardware square root have the same bits (any NaN matches any NaN).
bool same_bits_as_hardware(double a) {
  const double expected = Kokkos::sqrt(a);
  const double actual = impl::constexpr_sqrt(a);
  return bit_cast<std::uint64_t>(expected) == bit_cast<std::uint64_t>(actual) ||
         (std::isnan(expected) && std::isnan(actual));
}

TEST(Cmath, ConstexprSqrtMatchesTheHardwareForDoubles) {
  for (const double a : {0.0, -0.0, 0x1p-1074, 0x1p-1022, 0x1.fffffffffffffp-1023, 1.0, 0x1.0000000000001p+0,
                         0x1.fffffffffffffp+1, 4.0, kMax, kInfinity, -1.0, -kInfinity,
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_TRUE(same_bits_as_hardware(a)) << "a = " << a;
  }

  // Uniformly random bit patterns cover every exponent, subnormals included.
  std::mt19937_64 rng(7);
  int mismatches = 0;
  for (int i = 0; i < 1000000; ++i) {
    mismatches += !same_bits_as_hardware(bit_cast<double>(rng() & 0x7FFF'FFFF'FFFF'FFFFULL));
  }
  for (int i = 0; i < 100000; ++i) {
    mismatches += !same_bits_as_hardware(bit_cast<double>(rng() & 0x000F'FFFF'FFFF'FFFFULL));
  }

  // Squares of doubles and their neighbours have roots nearest a rounding boundary: the hardest cases.
  for (int m = 1; m < 100000; ++m) {
    const double q = 1.0 + m * 0x1p-40;
    const double square = q * q;
    mismatches += !same_bits_as_hardware(square);
    mismatches += !same_bits_as_hardware(std::nextafter(square, 0.0));
    mismatches += !same_bits_as_hardware(std::nextafter(square, kInfinity));
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(Cmath, ConstexprSqrtMatchesTheHardwareForEveryFloatInTwoBinades) {
  // Every float in [1, 4), so every mantissa with both exponent parities.
  int mismatches = 0;
  for (std::uint32_t bits = bit_cast<std::uint32_t>(1.0f); bits < bit_cast<std::uint32_t>(4.0f); ++bits) {
    const float a = bit_cast<float>(bits);
    mismatches += bit_cast<std::uint32_t>(impl::constexpr_sqrt(a)) != bit_cast<std::uint32_t>(Kokkos::sqrt(a));
  }
  EXPECT_EQ(mismatches, 0);
}

/// \brief How many of 2^20 hashed doubles get different mundy::sqrt and impl::constexpr_sqrt bits inside a kernel.
///
/// A free function, since CUDA forbids KOKKOS_LAMBDA in a test body (a private member function).
int count_sqrt_mismatches_in_kernel() {
  int mismatches = 0;
  Kokkos::parallel_reduce(
      "UnitTestCmath::SqrtAgreesWithConstexprSqrtInKernels", Kokkos::RangePolicy<>(0, 1 << 20),
      KOKKOS_LAMBDA(const int i, int& count) {
        const std::uint64_t bits = (static_cast<std::uint64_t>(i) * 0x9E37'79B9'7F4A'7C15ULL) >> 1;
        const double a = bit_cast<double>(bits);
        const double expected = mundy::sqrt(a);
        const double actual = impl::constexpr_sqrt(a);
        count += bit_cast<std::uint64_t>(expected) != bit_cast<std::uint64_t>(actual) && expected == expected;
      },
      mismatches);
  return mismatches;
}

TEST(Cmath, SqrtAgreesWithConstexprSqrtInKernels) {
  // mundy::sqrt and impl::constexpr_sqrt, both called inside a kernel on the default execution space.
  EXPECT_EQ(count_sqrt_mismatches_in_kernel(), 0);
}

}  // namespace

}  // namespace mundy
