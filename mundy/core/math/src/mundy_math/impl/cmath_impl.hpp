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

#ifndef MUNDY_MATH_IMPL_CMATH_IMPL_HPP_
#define MUNDY_MATH_IMPL_CMATH_IMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for KOKKOS_INLINE_FUNCTION, Kokkos::Experimental::quiet_NaN_v, ...

// C++ core
#include <cstdint>      // for std::int64_t, std::uint32_t, std::uint64_t
#include <type_traits>  // for std::conditional_t, std::is_same_v, std::is_trivially_copyable_v

namespace mundy {

namespace impl {

/// \brief Reinterprets the bits of From as To, in constant expressions and on the device (see mundy::bit_cast).
template <typename To, typename From>
KOKKOS_INLINE_FUNCTION constexpr To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast requires equal-size types.");
  static_assert(std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>,
                "bit_cast requires trivially copyable types.");
#if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else
  return Kokkos::bit_cast<To>(from);
#endif
}

//! \name Constant-expression versions of hardware math functions
//@{

/// \brief Correctly rounded square root of a double, usable in constant expressions.
///
/// IEEE 754 requires the hardware square root to be correctly rounded, so this returns the same bits: callers may
/// switch between the two without changing a result. The method writes a = r s^2 with r in [1, 4), runs Newton's
/// iteration on r, rounds the root with exact integer arithmetic, and scales it by s.
KOKKOS_INLINE_FUNCTION constexpr double constexpr_sqrt(double a) {
  // +0, -0, +infinity, and NaN are their own square roots; negative numbers have none. NaN is caught with != first,
  // because ordering a NaN raises an invalid-operation exception, which nvcc does not allow in a constant expression.
  if (a != a) {
    return a;
  }
  if (!(a > 0.0 && a <= Kokkos::Experimental::finite_max_v<double>)) {
    return a < 0.0 ? Kokkos::Experimental::quiet_NaN_v<double> : a;
  }

  // Reduce: a = r s^2 with r in [1, 4), by a binary search on the exponent. Every step scales by a power of 2, so the
  // reduction is exact. The search spans 4^-511 to 4^511, so a subnormal a is first lifted by 2^108.
  constexpr double powers_of_two[] = {0x1p256, 0x1p128, 0x1p64, 0x1p32, 0x1p16, 0x1p8, 0x1p4, 0x1p2, 0x1p1};
  double r = a;
  double s = 1.0;
  if (r < 0x1p-1022) {
    r *= 0x1p108;
    s = 0x1p-54;
  }
  for (const double p : powers_of_two) {
    if (r >= p * p) {
      r /= p * p;
      s *= p;
    } else if (r * p * p < 4.0) {
      r *= p * p;
      s /= p;
    }
  }

  // Newton's iteration from (r + 1) / 2, which lies above sqrt(r), decreases to sqrt(r) in [1, 2) and is within an ulp
  // of it after 6 steps.
  double x = 0.5 * (r + 1.0);
  for (int i = 0; i < 6; ++i) {
    x = 0.5 * (x + r / x);
  }
  x = x < 1.0 ? 1.0 : (x > 2.0 ? 2.0 : x);

  // Round exactly. In units of 2^-52 the root is the integer X and r the integer R. sqrt(r) lies above the midpoint
  // X + 1/2 iff R 2^52 - X^2 > X, and below X - 1/2 iff R 2^52 - X^2 <= -X. That residual is far below 2^63, so
  // unsigned 64-bit arithmetic, which wraps modulo 2^64, computes it exactly. Integers also keep FMA contraction out.
  std::uint64_t root = static_cast<std::uint64_t>(x * 0x1p52);
  std::int64_t residual = static_cast<std::int64_t>((static_cast<std::uint64_t>(r * 0x1p52) << 52) - root * root);
  while (residual > static_cast<std::int64_t>(root)) {
    residual -= static_cast<std::int64_t>(2 * root + 1);
    ++root;
  }
  while (residual <= -static_cast<std::int64_t>(root)) {
    residual += static_cast<std::int64_t>(2 * root - 1);
    --root;
  }
  return static_cast<double>(root) * 0x1p-52 * s;
}

/// \brief Correctly rounded square root of a float, usable in constant expressions.
///
/// Rounding the correctly rounded double root to float gives the correctly rounded float root: double's 53 bits are at
/// least 2 p + 2 = 50 for float's p = 24, the condition under which rounding twice cannot change a square root
/// (Figueroa, 1995).
KOKKOS_INLINE_FUNCTION constexpr float constexpr_sqrt(float a) {
  return static_cast<float>(constexpr_sqrt(static_cast<double>(a)));
}

/// \brief 1 / sqrt(a) of a float or double, usable in constant expressions: the host's 1 / sqrt, bit for bit.
///
/// Both round 1 / r for the correctly rounded root r. Dividing by zero, making a NaN, and ordering a NaN are not
/// constant expressions (the last for nvcc), so the special values are returned directly, as the hardware gives them:
/// 1 / sqrt(+-0) = +-infinity, a NaN gives itself, and a negative number gives NaN. 1 / sqrt(+infinity) = +0 needs no
/// special case.
template <typename Real>
KOKKOS_INLINE_FUNCTION constexpr Real constexpr_rsqrt(const Real a) {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "constexpr_rsqrt: Real must be float or double.");
  if (a != a) {
    return a;
  }
  if (a == Real(0)) {
    // Only the sign bit tells -0 from +0.
    using Bits = std::conditional_t<std::is_same_v<Real, double>, std::uint64_t, std::uint32_t>;
    const bool negative = (bit_cast<Bits>(a) >> (8 * sizeof(Real) - 1)) != 0;
    return negative ? -Kokkos::Experimental::infinity_v<Real> : Kokkos::Experimental::infinity_v<Real>;
  }
  if (a < Real(0)) {
    return Kokkos::Experimental::quiet_NaN_v<Real>;
  }
  return Real(1) / constexpr_sqrt(a);
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_CMATH_IMPL_HPP_
