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

#ifndef MUNDY_MATH_DOUBLEDOUBLE_HPP_
#define MUNDY_MATH_DOUBLEDOUBLE_HPP_

/// \file DoubleDouble.hpp
/// \brief A double-double scalar: the unevaluated sum hi + lo of two doubles, carrying about 32 significant digits.
///
/// DoubleDouble satisfies the MundyMath custom-scalar contract (NumTraits, ScalarBinaryOpTraits, and an ADL overload
/// for every function in cmath.hpp's dispatch set), so it stands in for double in Scalar, Vector, Matrix, Quaternion,
/// AutoDiffScalar, and Kokkos reductions, on the host and the device. Its arithmetic is constexpr.
///
/// Accuracy: arithmetic, the roots, exp, log, and the trigonometric functions are within a few units of 2^-104
/// relative; pow's error grows like |b log a| such units. sin, cos, and tan fall back to double accuracy beyond
/// |x| = 2^50, where their argument reduction ends. Values must stay above NumTraits::norm_min() (about 2e-292) for
/// lo to be normal.
///
/// zmort.hpp and the KokkosBlas/Belos solver backends do not support DoubleDouble.
///
/// The algorithms follow Hida, Li & Bailey's QD library.

// External
#include <Kokkos_Core.hpp>  // for KOKKOS_INLINE_FUNCTION, Kokkos::cbrt, Kokkos::reduction_identity, ...

// C++ core
#include <concepts>     // for std::integral
#include <ostream>      // for std::ostream
#include <type_traits>  // for std::is_arithmetic_v, std::true_type

// Mundy
#include <mundy_math/NumTraits.hpp>              // for mundy::NumTraits, mundy::is_passive_scalar
#include <mundy_math/ScalarBinaryOpTraits.hpp>   // for mundy::ScalarBinaryOpTraits
#include <mundy_math/cmath.hpp>                  // for mundy::abs, mundy::sqrt
#include <mundy_math/impl/DoubleDoubleImpl.hpp>  // for the error-free transformations and math kernels

namespace mundy {

/// \brief A double-double scalar hi + lo, normalized so that |lo| <= ulp(hi) / 2.
class DoubleDouble {
 public:
  //! \name Constructors
  //@{

  /// \brief Zero.
  KOKKOS_DEFAULTED_FUNCTION constexpr DoubleDouble() = default;

  /// \brief The given double, exactly.
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble(double value) : hi_(value) {  // NOLINT(runtime/explicit)
  }

  /// \brief The given integer, exactly (64-bit integers beyond 2^53 included).
  template <std::integral Integer>
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble(Integer value) {  // NOLINT(runtime/explicit)
    static_assert(sizeof(Integer) <= 8, "DoubleDouble: integers wider than 64 bits are not supported.");
    if constexpr (sizeof(Integer) <= 4) {
      hi_ = static_cast<double>(value);  // at most 32 bits: exact
    } else {
      // value = high * 2^32 + low, with both parts exact in a double.
      const impl::UnevaluatedSum sum = impl::two_sum(static_cast<double>(value >> 32) * 0x1p32,
                                                     static_cast<double>(value & static_cast<Integer>(0xFFFFFFFF)));
      hi_ = sum.hi;
      lo_ = sum.lo;
    }
  }

  /// \brief The pair hi + lo, which must already be normalized (|lo| <= ulp(hi) / 2).
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble(double hi, double lo) : hi_(hi), lo_(lo) {
  }
  //@}

  //! \name Accessors and conversion
  //@{

  /// \brief The leading double, which is the value rounded to double.
  KOKKOS_INLINE_FUNCTION constexpr double hi() const {
    return hi_;
  }

  /// \brief The trailing double: the rounding error of hi().
  KOKKOS_INLINE_FUNCTION constexpr double lo() const {
    return lo_;
  }

  /// \brief The value rounded to double. Explicit, so a DoubleDouble never silently loses its low part.
  KOKKOS_INLINE_FUNCTION explicit constexpr operator double() const {
    return hi_;
  }
  //@}

  //! \name Compound assignment
  //@{
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& operator+=(const DoubleDouble& other);
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& operator-=(const DoubleDouble& other);
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& operator*=(const DoubleDouble& other);
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& operator/=(const DoubleDouble& other);
  //@}

 private:
  double hi_ = 0.0;
  double lo_ = 0.0;
};

//! \name Arithmetic
//@{

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator+(const DoubleDouble& a) {
  return a;
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator-(const DoubleDouble& a) {
  return {-a.hi(), -a.lo()};
}

/// \brief a + b, within 2 units of 2^-104 relative even under cancellation (QD's IEEE-style addition).
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator+(const DoubleDouble& a, const DoubleDouble& b) {
  const impl::UnevaluatedSum s = impl::two_sum(a.hi(), b.hi());
  if (!impl::is_finite_double(s.hi)) {
    return s.hi;
  }
  const impl::UnevaluatedSum t = impl::two_sum(a.lo(), b.lo());
  const impl::UnevaluatedSum u = impl::quick_two_sum(s.hi, s.lo + t.hi);
  const impl::UnevaluatedSum sum = impl::quick_two_sum(u.hi, u.lo + t.lo);
  return {sum.hi, sum.lo};
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator-(const DoubleDouble& a, const DoubleDouble& b) {
  return a + (-b);
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator*(const DoubleDouble& a, const DoubleDouble& b) {
  const impl::UnevaluatedSum p = impl::two_prod(a.hi(), b.hi());
  if (!impl::is_finite_double(p.hi)) {
    return p.hi;
  }
  const impl::UnevaluatedSum product = impl::quick_two_sum(p.hi, p.lo + (a.hi() * b.lo() + a.lo() * b.hi()));
  return {product.hi, product.lo};
}

/// \brief a / b: the double quotient, corrected twice by the remainder it leaves (QD's accurate division).
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator/(const DoubleDouble& a, const DoubleDouble& b) {
  const double q1 = a.hi() / b.hi();
  if (!impl::is_finite_double(q1) || !impl::is_finite_double(b.hi())) {
    return q1;  // infinities, NaN, division by zero, and division by infinity
  }
  DoubleDouble r = a - q1 * b;
  const double q2 = r.hi() / b.hi();
  r = r - q2 * b;
  const double q3 = r.hi() / b.hi();
  const impl::UnevaluatedSum q = impl::quick_two_sum(q1, q2);
  return DoubleDouble(q.hi, q.lo) + q3;
}

/// \brief a / b for a double divisor: one correction of the double quotient, cheaper than the general division.
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble operator/(const DoubleDouble& a, double b) {
  const double q1 = a.hi() / b;
  if (!impl::is_finite_double(q1) || !impl::is_finite_double(b)) {
    return q1;  // infinities, NaN, division by zero, and division by infinity
  }
  const impl::UnevaluatedSum q1_b = impl::two_prod(q1, b);
  const impl::UnevaluatedSum remainder = impl::two_sum(a.hi(), -q1_b.hi);
  const double q2 = (remainder.hi + ((remainder.lo - q1_b.lo) + a.lo())) / b;
  const impl::UnevaluatedSum q = impl::quick_two_sum(q1, q2);
  return {q.hi, q.lo};
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& DoubleDouble::operator+=(const DoubleDouble& other) {
  return *this = *this + other;
}
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& DoubleDouble::operator-=(const DoubleDouble& other) {
  return *this = *this - other;
}
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& DoubleDouble::operator*=(const DoubleDouble& other) {
  return *this = *this * other;
}
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble& DoubleDouble::operator/=(const DoubleDouble& other) {
  return *this = *this / other;
}
//@}

//! \name Comparison (lexicographic on (hi, lo), which orders normalized values; NaN compares false)
//@{
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const DoubleDouble& a, const DoubleDouble& b) {
  return a.hi() == b.hi() && a.lo() == b.lo();
}
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const DoubleDouble& a, const DoubleDouble& b) {
  return !(a == b);
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<(const DoubleDouble& a, const DoubleDouble& b) {
  return a.hi() < b.hi() || (a.hi() == b.hi() && a.lo() < b.lo());
}
KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const DoubleDouble& a, const DoubleDouble& b) {
  return a.hi() < b.hi() || (a.hi() == b.hi() && a.lo() <= b.lo());
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>(const DoubleDouble& a, const DoubleDouble& b) {
  return b < a;
}
KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const DoubleDouble& a, const DoubleDouble& b) {
  return b <= a;
}
//@}

//! \name Sign, rounding, and selection
//@{

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble abs(const DoubleDouble& a) {
  return a.hi() < 0.0 ? -a : DoubleDouble(mundy::abs(a.hi()), a.lo());
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble copysign(const DoubleDouble& a, const DoubleDouble& sign) {
  return impl::sign_bit_double(a.hi()) == impl::sign_bit_double(sign.hi()) ? a : -a;
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble min(const DoubleDouble& a, const DoubleDouble& b) {
  return b < a ? b : a;
}

KOKKOS_INLINE_FUNCTION constexpr DoubleDouble max(const DoubleDouble& a, const DoubleDouble& b) {
  return a < b ? b : a;
}

/// \brief Round down. If hi is already an integer the fraction lives in lo, so lo is rounded instead.
KOKKOS_INLINE_FUNCTION DoubleDouble floor(const DoubleDouble& a) {
  const double hi = Kokkos::floor(a.hi());
  if (hi != a.hi()) {
    return hi;
  }
  const impl::UnevaluatedSum rounded = impl::quick_two_sum(hi, Kokkos::floor(a.lo()));
  return {rounded.hi, rounded.lo};
}

/// \brief Round up. If hi is already an integer the fraction lives in lo, so lo is rounded instead.
KOKKOS_INLINE_FUNCTION DoubleDouble ceil(const DoubleDouble& a) {
  const double hi = Kokkos::ceil(a.hi());
  if (hi != a.hi()) {
    return hi;
  }
  const impl::UnevaluatedSum rounded = impl::quick_two_sum(hi, Kokkos::ceil(a.lo()));
  return {rounded.hi, rounded.lo};
}

/// \brief Round to nearest, halfway cases away from zero (like std::round).
KOKKOS_INLINE_FUNCTION DoubleDouble round(const DoubleDouble& a) {
  const double hi = Kokkos::round(a.hi());
  if (hi == a.hi()) {
    // hi is an integer, so a.lo holds the fraction. A half that points toward zero rounds away from it, to hi.
    const bool half_toward_zero = Kokkos::abs(a.lo()) == 0.5 && (a.lo() < 0.0) != (a.hi() < 0.0);
    const impl::UnevaluatedSum rounded = impl::quick_two_sum(hi, half_toward_zero ? 0.0 : Kokkos::round(a.lo()));
    return {rounded.hi, rounded.lo};
  }
  // hi is exactly halfway and lo tips the value toward zero: the nearest integer is the one toward zero.
  const bool tie_broken_toward_zero =
      Kokkos::abs(hi - a.hi()) == 0.5 && a.lo() != 0.0 && (a.lo() < 0.0) != (a.hi() < 0.0);
  return tie_broken_toward_zero ? hi - Kokkos::copysign(1.0, a.hi()) : hi;
}
//@}

//! \name Roots
//@{

/// \brief Constexpr square root: one Newton step x + (a - x^2) / (2x) from the double root x.
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble sqrt(const DoubleDouble& a) {
  const double x = mundy::sqrt(a.hi());
  if (a.hi() <= 0.0 || !impl::is_finite_double(x)) {
    return x;  // +-0, negative (NaN), infinity, NaN
  }
  const impl::UnevaluatedSum x_squared = impl::two_prod(x, x);
  const impl::UnevaluatedSum root =
      impl::quick_two_sum(x, (a - DoubleDouble(x_squared.hi, x_squared.lo)).hi() / (2.0 * x));
  return {root.hi, root.lo};
}

/// \brief Cube root: one Newton step x + (a - x^3) / (3x^2) from the double root x.
KOKKOS_INLINE_FUNCTION DoubleDouble cbrt(const DoubleDouble& a) {
  const double x = Kokkos::cbrt(a.hi());
  if (a.hi() == 0.0 || !impl::is_finite_double(x)) {
    return x;
  }
  const impl::UnevaluatedSum x_squared = impl::two_prod(x, x);
  const DoubleDouble x_cubed = DoubleDouble(x_squared.hi, x_squared.lo) * x;
  const impl::UnevaluatedSum root = impl::quick_two_sum(x, (a - x_cubed).hi() / (3.0 * x * x));
  return {root.hi, root.lo};
}
//@}

//! \name Exponentials and logarithms
//@{

/// \brief e^a.
///
/// Write a = m ln2 + r, sum the Taylor series of s = e^(r/512) - 1, square back up through s <- 2s + s^2 (which keeps
/// the relative accuracy of the small s), and scale 1 + s by 2^m. The reduction subtracts m ln2 with ln2 split into
/// three doubles (about 160 bits, Cody-Waite), each product m l_i formed exactly.
KOKKOS_INLINE_FUNCTION DoubleDouble exp(const DoubleDouble& a) {
  if (!(a.hi() > -708.0 && a.hi() < 709.79)) {
    return Kokkos::exp(a.hi());  // overflow, NaN, or so small that lo would be subnormal
  }
  constexpr double ln2_part1 = 0x1.62e42fefa39efp-1;
  constexpr double ln2_part2 = 0x1.abc9e3b39803fp-56;
  constexpr double ln2_part3 = 0x1.7b57a079a1934p-111;
  const double m = Kokkos::round(a.hi() / ln2_part1);
  const auto exact_product = [m](double p) {
    const impl::UnevaluatedSum mp = impl::two_prod(m, p);
    return DoubleDouble(mp.hi, mp.lo);
  };
  const DoubleDouble r =
      (((a - exact_product(ln2_part1)) - exact_product(ln2_part2)) - exact_product(ln2_part3)) * (1.0 / 512.0);

  DoubleDouble s = r;
  DoubleDouble term = r;
  for (int k = 2; k <= 20 && mundy::abs(term.hi()) > impl::double_double_negligible * mundy::abs(s.hi()); ++k) {
    term = term * r / static_cast<double>(k);
    s += term;
  }
  for (int i = 0; i < 9; ++i) {
    s = 2.0 * s + s * s;
  }
  const DoubleDouble one_plus_s = s + 1.0;
  const impl::UnevaluatedSum result = impl::mul_pow2(one_plus_s.hi(), one_plus_s.lo(), static_cast<int>(m));
  return {result.hi, result.lo};
}

/// \brief Natural logarithm.
///
/// Near 1, the series log a = 2 (u + u^3/3 + u^5/5 + ...) in u = (a - 1) / (a + 1) keeps the relative accuracy of a
/// small result. Elsewhere, one Newton step x + a e^-x - 1 from the double logarithm x.
KOKKOS_INLINE_FUNCTION DoubleDouble log(const DoubleDouble& a) {
  const double x = Kokkos::log(a.hi());
  if (a.hi() <= 0.0 || !impl::is_finite_double(x)) {
    return x;  // 0 (-infinity), negative (NaN), infinity, NaN
  }
  if (Kokkos::abs(a.hi() - 1.0) < 0.25) {
    const DoubleDouble u = (a - 1.0) / (a + 1.0);  // |u| < 1/7
    const DoubleDouble u2 = u * u;
    DoubleDouble sum = u;
    DoubleDouble power = u;
    for (int k = 1; k <= 40 && mundy::abs(power.hi()) > impl::double_double_negligible * mundy::abs(sum.hi()); ++k) {
      power *= u2;
      sum += power / static_cast<double>(2 * k + 1);
    }
    return 2.0 * sum;
  }
  return x + (a * exp(DoubleDouble(-x)) - 1.0);
}

KOKKOS_INLINE_FUNCTION DoubleDouble log10(const DoubleDouble& a) {
  constexpr DoubleDouble ln10(0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53);
  return log(a) / ln10;
}

/// \brief a^b. Integer exponents use repeated squaring (any sign of a); others use exp(b log a), which needs a > 0
/// and whose relative error grows like |b log a| units of 2^-104.
KOKKOS_INLINE_FUNCTION DoubleDouble pow(const DoubleDouble& a, const DoubleDouble& b) {
  if (b == floor(b) && Kokkos::abs(b.hi()) < 0x1p31) {
    const long long n = static_cast<long long>(b.hi());
    DoubleDouble result = 1.0;
    DoubleDouble base = a;
    for (unsigned long long m = n < 0 ? -n : n; m != 0; m >>= 1) {
      if (m & 1) {
        result *= base;
      }
      base *= base;
    }
    return n < 0 ? 1.0 / result : result;
  }
  return exp(b * log(a));
}
//@}

//! \name Trigonometric functions
//@{

KOKKOS_INLINE_FUNCTION DoubleDouble sin(const DoubleDouble& a) {
  return a.hi() == 0.0 ? a : impl::sin_cos_impl(a).sin;  // keeps the sign of zero
}

KOKKOS_INLINE_FUNCTION DoubleDouble cos(const DoubleDouble& a) {
  return impl::sin_cos_impl(a).cos;
}

KOKKOS_INLINE_FUNCTION DoubleDouble tan(const DoubleDouble& a) {
  if (a.hi() == 0.0) {
    return a;
  }
  const impl::SinCos<DoubleDouble> v = impl::sin_cos_impl(a);
  return v.sin / v.cos;
}

/// \brief The angle of the point (x, y), in (-pi, pi].
///
/// One Newton step from the double angle z: on sin z = y / r when |x| > |y|, else on cos z = x / r. When the answer
/// is a multiple of pi/4 (a zero or infinite coordinate, or |x| == |y|), it is that multiple, read off the double
/// atan2 so that signed zeros and NaN follow std::atan2.
KOKKOS_INLINE_FUNCTION DoubleDouble atan2(const DoubleDouble& y, const DoubleDouble& x) {
  if (x.hi() == 0.0 || y.hi() == 0.0 || abs(x) == abs(y) || !impl::is_finite_double(x.hi()) ||
      !impl::is_finite_double(y.hi())) {
    constexpr DoubleDouble pi_4(0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55);
    const double angle = Kokkos::atan2(y.hi(), x.hi());
    if (angle == 0.0 || angle != angle) {
      return angle;  // +-0 or NaN
    }
    return pi_4 * Kokkos::round(angle / pi_4.hi());
  }
  const DoubleDouble scale = max(abs(x), abs(y));
  const DoubleDouble xs = x / scale;
  const DoubleDouble ys = y / scale;
  const DoubleDouble r = sqrt(xs * xs + ys * ys);

  const DoubleDouble z = Kokkos::atan2(y.hi(), x.hi());
  const impl::SinCos<DoubleDouble> v = impl::sin_cos_impl(z);
  if (Kokkos::abs(xs.hi()) > Kokkos::abs(ys.hi())) {
    return z + (ys / r - v.sin) / v.cos;
  }
  return z - (xs / r - v.cos) / v.sin;
}

KOKKOS_INLINE_FUNCTION DoubleDouble atan(const DoubleDouble& a) {
  return atan2(a, DoubleDouble(1.0));
}

/// \brief Arcsine, through atan2(a, sqrt((1 - a)(1 + a))); NaN for |a| > 1.
KOKKOS_INLINE_FUNCTION DoubleDouble asin(const DoubleDouble& a) {
  return atan2(a, sqrt((1.0 - a) * (1.0 + a)));
}

/// \brief Arccosine, through atan2(sqrt((1 - a)(1 + a)), a); NaN for |a| > 1.
KOKKOS_INLINE_FUNCTION DoubleDouble acos(const DoubleDouble& a) {
  return atan2(sqrt((1.0 - a) * (1.0 + a)), a);
}
//@}

//! \name The MundyMath custom-scalar contract
//@{

template <>
struct is_passive_scalar<DoubleDouble> : std::true_type {};

/// \brief Numeric traits of DoubleDouble, in DoubleDouble.
template <>
struct NumTraits<DoubleDouble> {
  using Real = DoubleDouble;
  using NonInteger = DoubleDouble;
  using Literal = double;
  static constexpr bool IsInteger = false;
  static constexpr bool IsSigned = true;
  static constexpr bool IsComplex = false;
  static constexpr bool RequireInitialization = false;

  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble epsilon() {
    return 0x1p-104;
  }
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble dummy_precision() {
    return 1e-28;
  }
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble highest() {
    return {0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+969};
  }
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble lowest() {
    return -highest();
  }
  /// \brief The smallest value whose lo part is still a normal double.
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble norm_min() {
    return 0x1p-969;
  }
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble infinity() {
    return Kokkos::Experimental::infinity_v<double>;
  }
  KOKKOS_INLINE_FUNCTION static constexpr DoubleDouble quiet_NaN() {
    return Kokkos::Experimental::quiet_NaN_v<double>;
  }
};

/// \brief DoubleDouble with an arithmetic primitive yields DoubleDouble.
template <typename A, typename Op>
  requires(std::is_arithmetic_v<A>)
struct ScalarBinaryOpTraits<DoubleDouble, A, Op> {
  using ReturnType = DoubleDouble;
};

/// \brief An arithmetic primitive with DoubleDouble yields DoubleDouble.
template <typename A, typename Op>
  requires(std::is_arithmetic_v<A>)
struct ScalarBinaryOpTraits<A, DoubleDouble, Op> {
  using ReturnType = DoubleDouble;
};

/// \brief Write the exact pair as "hi + lo", at the stream's precision.
inline std::ostream& operator<<(std::ostream& os, const DoubleDouble& a) {
  return os << a.hi() << " + " << a.lo();
}
//@}

}  // namespace mundy

/// \brief Identities for Kokkos reductions (Sum, Prod, Max, Min) over DoubleDouble.
template <>
struct Kokkos::reduction_identity<mundy::DoubleDouble> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static mundy::DoubleDouble sum() {
    return 0.0;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static mundy::DoubleDouble prod() {
    return 1.0;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static mundy::DoubleDouble max() {
    return -Kokkos::Experimental::infinity_v<double>;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static mundy::DoubleDouble min() {
    return Kokkos::Experimental::infinity_v<double>;
  }
};

#endif  // MUNDY_MATH_DOUBLEDOUBLE_HPP_
