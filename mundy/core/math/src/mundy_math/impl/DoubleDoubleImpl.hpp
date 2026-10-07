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

#ifndef MUNDY_MATH_IMPL_DOUBLEDOUBLEIMPL_HPP_
#define MUNDY_MATH_IMPL_DOUBLEDOUBLEIMPL_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cstdint>  // for std::uint64_t

// Mundy
#include <mundy_math/cmath.hpp>  // for mundy::abs, mundy::bit_cast

namespace mundy {

namespace impl {

//! \name Error-free transformations of doubles, the building blocks of DoubleDouble
//@{

/// \brief The unevaluated sum hi + lo returned by an error-free transformation.
struct UnevaluatedSum {
  double hi;
  double lo;
};

/// \brief Whether x is finite, in a form usable in constant expressions.
KOKKOS_INLINE_FUNCTION constexpr bool is_finite_double(double x) {
  return x - x == 0.0;
}

/// \brief The sign bit of x, in a form usable in constant expressions.
KOKKOS_INLINE_FUNCTION constexpr bool sign_bit_double(double x) {
  return (mundy::bit_cast<std::uint64_t>(x) >> 63) != 0;
}

/// \brief hi + lo == a + b exactly, given |a| >= |b| (Dekker's fast two-sum).
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum quick_two_sum(double a, double b) {
  const double s = a + b;
  return is_finite_double(s) ? UnevaluatedSum{s, b - (s - a)} : UnevaluatedSum{s, 0.0};
}

/// \brief hi + lo == a + b exactly (Knuth's two-sum).
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum two_sum(double a, double b) {
  const double s = a + b;
  if (!is_finite_double(s)) {
    return {s, 0.0};
  }
  const double b_part = s - a;
  return {s, (a - (s - b_part)) + (b - b_part)};
}

/// \brief hi + lo == a with each half at most 26 significant bits, so the product of two halves is exact (Dekker).
///
/// Values beyond 2^996 are scaled down first, since splitting them would overflow.
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum dekker_split(double a) {
  constexpr double splitter = 0x1p27 + 1.0;
  if (a > 0x1p996 || a < -0x1p996) {
    const double scaled = a * 0x1p-28;
    const double t = splitter * scaled;
    const double hi = t - (t - scaled);
    return {hi * 0x1p28, (scaled - hi) * 0x1p28};
  }
  const double t = splitter * a;
  const double hi = t - (t - a);
  return {hi, a - hi};
}

/// \brief hi + lo == a * b exactly (Dekker's two-product).
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum two_prod(double a, double b) {
  const double p = a * b;
  if (!is_finite_double(p)) {
    return {p, 0.0};
  }
  const UnevaluatedSum as = dekker_split(a);
  const UnevaluatedSum bs = dekker_split(b);
  return {p, ((as.hi * bs.hi - p) + as.hi * bs.lo + as.lo * bs.hi) + as.lo * bs.lo};
}

/// \brief 2^e for -1022 <= e <= 1023, built from its bit pattern so that it is exact.
KOKKOS_INLINE_FUNCTION constexpr double exact_pow2(int e) {
  return mundy::bit_cast<double>(static_cast<std::uint64_t>(e + 1023) << 52);
}

/// \brief (hi + lo) * 2^e, exact while the result stays normal. The factor is applied in two halves so it never
/// overflows.
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum mul_pow2(double hi, double lo, int e) {
  const double f1 = exact_pow2(e / 2);
  const double f2 = exact_pow2(e - e / 2);
  const double scaled_hi = hi * f1 * f2;
  return is_finite_double(scaled_hi) ? UnevaluatedSum{scaled_hi, lo * f1 * f2} : UnevaluatedSum{scaled_hi, 0.0};
}

/// \brief pi / 2 as a double-double (hi, lo).
KOKKOS_INLINE_FUNCTION constexpr UnevaluatedSum pi_over_2() {
  return {0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54};
}

/// \brief 2^-106: a series term below this times the sum no longer changes a double-double.
inline constexpr double double_double_negligible = 0x1p-106;
//@}

//! \name Kernels of the DoubleDouble math functions
//
// Templated on the double-double type only so that this file can precede its definition, as VectorImpl.hpp does for
// AVector.
//@{

/// \brief A sine and cosine of the same argument.
template <typename DoubleDoubleType>
struct SinCos {
  DoubleDoubleType sin;
  DoubleDoubleType cos;
};

/// \brief sin(t) and cos(t) for |t| <= pi/4, by their Taylor series.
template <typename DoubleDoubleType>
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDoubleType> sin_cos_taylor(const DoubleDoubleType& t) {
  const DoubleDoubleType t2 = t * t;
  DoubleDoubleType s = t;
  DoubleDoubleType c = 1.0;
  DoubleDoubleType s_term = t;
  DoubleDoubleType c_term = 1.0;
  for (int k = 1; k <= 20 && (mundy::abs(c_term.hi()) > double_double_negligible ||
                              mundy::abs(s_term.hi()) > double_double_negligible * mundy::abs(s.hi()));
       ++k) {
    s_term = -s_term * t2 / static_cast<double>((2 * k) * (2 * k + 1));
    c_term = -c_term * t2 / static_cast<double>((2 * k - 1) * (2 * k));
    s += s_term;
    c += c_term;
  }
  return {s, c};
}

/// \brief The sine and cosine of t + quadrant * pi/2, from those of t (quadrant in [0, 4)).
template <typename DoubleDoubleType>
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDoubleType> rotate_by_quadrant(const SinCos<DoubleDoubleType>& v,
                                                                             int quadrant) {
  switch (quadrant) {
    case 0:
      return v;
    case 1:
      return {v.cos, -v.sin};
    case 2:
      return {-v.sin, -v.cos};
    default:
      return {-v.cos, v.sin};
  }
}

/// \brief The sine and cosine of 2 pi k / m, the fraction k / m of a full turn.
///
/// 2 pi k / m = (pi/2) (4k / m). Integer division splits 4k = q m + r exactly; moving r into (-m/2, m/2] leaves
/// t = (pi/2) r / m with |t| <= pi/4 for the Taylor series, and the quadrant q rotates the result.
template <typename DoubleDoubleType>
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDoubleType> sin_cos_of_turn_fraction(unsigned k, unsigned m) {
  unsigned quadrant = (4 * k) / m;
  long r = static_cast<long>(4 * k) - static_cast<long>(quadrant * m);
  if (2 * r > static_cast<long>(m)) {
    r -= static_cast<long>(m);
    ++quadrant;
  }
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDoubleType t =
      DoubleDoubleType(half_pi.hi, half_pi.lo) * static_cast<double>(r) / static_cast<double>(m);
  return rotate_by_quadrant(sin_cos_taylor(t), static_cast<int>(quadrant % 4));
}

/// \brief sin(a) and cos(a).
///
/// Reduce a = j pi/2 + t with |t| <= pi/4, sum both Taylor series in t, and rotate the result by the quadrant j. The
/// reduction subtracts j pi/2 with pi/2 split into three doubles (about 160 bits, Cody-Waite), each product j p_i
/// formed exactly, so it stays accurate for |j| < 2^50.
template <typename DoubleDoubleType>
KOKKOS_INLINE_FUNCTION SinCos<DoubleDoubleType> sin_cos_impl(const DoubleDoubleType& a) {
  if (!(Kokkos::abs(a.hi()) < 0x1p50)) {
    return {Kokkos::sin(a.hi()), Kokkos::cos(a.hi())};  // beyond the reduction's range, infinity, NaN
  }
  constexpr double pi_2_part1 = pi_over_2().hi;
  constexpr double pi_2_part2 = pi_over_2().lo;
  constexpr double pi_2_part3 = -0x1.f1976b7ed8fbcp-110;
  const double j = Kokkos::round(a.hi() / pi_2_part1);
  const auto exact_product = [j](double p) {
    const UnevaluatedSum jp = two_prod(j, p);
    return DoubleDoubleType(jp.hi, jp.lo);
  };
  const DoubleDoubleType t = ((a - exact_product(pi_2_part1)) - exact_product(pi_2_part2)) - exact_product(pi_2_part3);
  const int quadrant = static_cast<int>(j - 4.0 * Kokkos::floor(j / 4.0));  // j mod 4, in [0, 4)
  return rotate_by_quadrant(sin_cos_taylor(t), quadrant);
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_DOUBLEDOUBLEIMPL_HPP_
