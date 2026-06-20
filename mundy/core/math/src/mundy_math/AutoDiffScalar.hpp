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

#ifndef MUNDY_MATH_AUTODIFFSCALAR_HPP_
#define MUNDY_MATH_AUTODIFFSCALAR_HPP_

/// \file AutoDiffScalar.hpp
/// \brief A forward-mode automatic-differentiation scalar: carries a value and its derivatives.
///
/// AutoDiffScalar is a drop-in scalar that propagates first derivatives by the chain rule. It satisfies
/// the MundyMath custom-scalar contract — \ref NumTraits, \ref ScalarBinaryOpTraits, and ADL math
/// overloads — so it can stand in for a primitive scalar throughout the library.
///
/// \code{.cpp}
///   AutoDiffScalar<double, 2> x(3.0, 0);   // independent variable, d/dx seeded at slot 0
///   AutoDiffScalar<double, 2> y(4.0, 1);   // independent variable, d/dy seeded at slot 1
///   auto f = x * y + sqrt(x);              // f.value() == 12 + sqrt(3)
///   f.derivatives()[0];                    // df/dx == y + 0.5/sqrt(x)
///   f.derivatives()[1];                    // df/dy == x
/// \endcode

// External
#include <Kokkos_Core.hpp>  // for KOKKOS_INLINE_FUNCTION

// C++ core
#include <cstddef>      // for size_t
#include <iostream>     // for std::ostream
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_arithmetic_v

// Mundy
#include <mundy_math/NumTraits.hpp>             // for mundy::NumTraits
#include <mundy_math/ScalarBinaryOpTraits.hpp>  // for mundy::ScalarBinaryOpTraits
#include <mundy_math/Vector.hpp>                // for mundy::Vector (derivative storage)
#include <mundy_math/cmath.hpp>                 // for mundy:: math dispatch used on the underlying value
#include <mundy_utils/requires.hpp>             // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>         // for MUNDY_THROW_ASSERT

namespace mundy {

/// \brief A forward-mode autodiff scalar over passive real type T with N tracked derivatives.
///
/// The value is a T; the derivatives are a \c Vector<T,N>, one slot per independent variable. Arithmetic
/// and math operations carry both forward by the chain rule.
///
/// \tparam T The passive (non-differentiated) real type, e.g. double.
/// \tparam N The number of independent variables whose derivatives are tracked.
template <typename T, size_t N>
MUNDY_REQUIRES(std::is_arithmetic_v<T>)
class AutoDiffScalar {
 public:
  //! \name Type aliases
  //@{

  /// \brief The passive value type.
  using value_type = T;

  /// \brief The derivative storage type.
  using derivative_type = Vector<T, N>;

  /// \brief The number of tracked derivatives.
  static constexpr size_t num_derivatives = N;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Value and derivatives are uninitialized.
  KOKKOS_DEFAULTED_FUNCTION constexpr AutoDiffScalar() = default;

  /// \brief Construct a constant: the given value with zero derivatives.
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar(const T& value)  // NOLINT(runtime/explicit) intentional promotion
      : value_(value), derivatives_(T(0)) {
  }

  /// \brief Construct from a value and an explicit derivative vector.
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar(const T& value, const Vector<T, N>& derivatives)
      : value_(value), derivatives_(derivatives) {
  }

  /// \brief Construct an independent variable: the given value with a unit derivative in slot \c index.
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar(const T& value, size_t index) : value_(value), derivatives_(T(0)) {
    MUNDY_THROW_ASSERT(index < N, std::invalid_argument, "AutoDiffScalar: derivative index out of range.");
    derivatives_[index] = T(1);
  }

  KOKKOS_DEFAULTED_FUNCTION constexpr ~AutoDiffScalar() = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr AutoDiffScalar(const AutoDiffScalar&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr AutoDiffScalar(AutoDiffScalar&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr AutoDiffScalar& operator=(const AutoDiffScalar&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr AutoDiffScalar& operator=(AutoDiffScalar&&) = default;
  //@}

  //! \name Accessors
  //@{

  /// \brief The underlying value.
  KOKKOS_INLINE_FUNCTION constexpr const T& value() const {
    return value_;
  }
  KOKKOS_INLINE_FUNCTION constexpr T& value() {
    return value_;
  }

  /// \brief The derivative vector.
  KOKKOS_INLINE_FUNCTION constexpr const Vector<T, N>& derivatives() const {
    return derivatives_;
  }
  KOKKOS_INLINE_FUNCTION constexpr Vector<T, N>& derivatives() {
    return derivatives_;
  }
  //@}

  //! \name Arithmetic with another AutoDiffScalar
  //@{

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator-() const {
    return {-value_, -derivatives_};
  }

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator+(const AutoDiffScalar& o) const {
    return {value_ + o.value_, derivatives_ + o.derivatives_};
  }

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator-(const AutoDiffScalar& o) const {
    return {value_ - o.value_, derivatives_ - o.derivatives_};
  }

  /// \brief Product rule: (u v)' = u' v + u v'.
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator*(const AutoDiffScalar& o) const {
    return {value_ * o.value_, derivatives_ * o.value_ + o.derivatives_ * value_};
  }

  /// \brief Quotient rule: (u / v)' = (u' - (u/v) v') / v.
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator/(const AutoDiffScalar& o) const {
    const T inv = T(1) / o.value_;
    const T quotient = value_ * inv;
    return {quotient, (derivatives_ - o.derivatives_ * quotient) * inv};
  }

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator+=(const AutoDiffScalar& o) {
    value_ += o.value_;
    derivatives_ += o.derivatives_;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator-=(const AutoDiffScalar& o) {
    value_ -= o.value_;
    derivatives_ -= o.derivatives_;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator*=(const AutoDiffScalar& o) {
    derivatives_ = derivatives_ * o.value_ + o.derivatives_ * value_;  // uses old value_, so update value_ after
    value_ *= o.value_;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator/=(const AutoDiffScalar& o) {
    return *this = *this / o;
  }
  //@}

  //! \name Arithmetic with a passive value
  //@{

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator+(const T& s) const {
    return {value_ + s, derivatives_};
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator-(const T& s) const {
    return {value_ - s, derivatives_};
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator*(const T& s) const {
    return {value_ * s, derivatives_ * s};
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar operator/(const T& s) const {
    const T inv = T(1) / s;
    return {value_ * inv, derivatives_ * inv};
  }

  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator+=(const T& s) {
    value_ += s;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator-=(const T& s) {
    value_ -= s;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator*=(const T& s) {
    value_ *= s;
    derivatives_ *= s;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar& operator/=(const T& s) {
    const T inv = T(1) / s;
    value_ *= inv;
    derivatives_ *= inv;
    return *this;
  }
  //@}

 private:
  //! \name Data
  //@{

  /// \brief The underlying value.
  T value_;

  /// \brief The derivative with respect to each tracked variable.
  Vector<T, N> derivatives_;
  //@}
};  // class AutoDiffScalar

//! \name Passive-value arithmetic (passive op AutoDiffScalar)
//@{

template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> operator+(const T& s, const AutoDiffScalar<T, N>& a) {
  return a + s;
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> operator-(const T& s, const AutoDiffScalar<T, N>& a) {
  return {s - a.value(), -a.derivatives()};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> operator*(const T& s, const AutoDiffScalar<T, N>& a) {
  return a * s;
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> operator/(const T& s, const AutoDiffScalar<T, N>& a) {
  const T inv = T(1) / a.value();
  const T quotient = s * inv;                              // s / a
  return {quotient, a.derivatives() * (-quotient * inv)};  // d(s/a) = -s/a^2 * a'
}
//@}

//! \name Comparisons (by value)
//@{

// clang-format off
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator==(const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() == b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() != b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator< (const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() <  b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() <= b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator> (const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() >  b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const AutoDiffScalar<T, N>& a, const AutoDiffScalar<T, N>& b) { return a.value() >= b.value(); }

template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator==(const AutoDiffScalar<T, N>& a, const T& b) { return a.value() == b; }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const AutoDiffScalar<T, N>& a, const T& b) { return a.value() != b; }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator< (const AutoDiffScalar<T, N>& a, const T& b) { return a.value() <  b; }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const AutoDiffScalar<T, N>& a, const T& b) { return a.value() <= b; }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator> (const AutoDiffScalar<T, N>& a, const T& b) { return a.value() >  b; }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const AutoDiffScalar<T, N>& a, const T& b) { return a.value() >= b; }

template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator==(const T& a, const AutoDiffScalar<T, N>& b) { return a == b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const T& a, const AutoDiffScalar<T, N>& b) { return a != b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator< (const T& a, const AutoDiffScalar<T, N>& b) { return a <  b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const T& a, const AutoDiffScalar<T, N>& b) { return a <= b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator> (const T& a, const AutoDiffScalar<T, N>& b) { return a >  b.value(); }
template <typename T, size_t N> KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const T& a, const AutoDiffScalar<T, N>& b) { return a >= b.value(); }
// clang-format on
//@}

//! \name Math overloads (value via the mundy:: dispatch, derivative via the chain rule)
//@{

template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> sqrt(const AutoDiffScalar<T, N>& x) {
  const T v = sqrt(x.value());
  return {v, x.derivatives() * (T(0.5) / v)};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> exp(const AutoDiffScalar<T, N>& x) {
  const T v = exp(x.value());
  return {v, x.derivatives() * v};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> log(const AutoDiffScalar<T, N>& x) {
  return {log(x.value()), x.derivatives() * (T(1) / x.value())};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> sin(const AutoDiffScalar<T, N>& x) {
  return {sin(x.value()), x.derivatives() * cos(x.value())};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> cos(const AutoDiffScalar<T, N>& x) {
  return {cos(x.value()), x.derivatives() * (-sin(x.value()))};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> acos(const AutoDiffScalar<T, N>& x) {
  return {acos(x.value()), x.derivatives() * (T(-1) / sqrt(T(1) - x.value() * x.value()))};
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> abs(const AutoDiffScalar<T, N>& x) {
  return {abs(x.value()), x.derivatives() * (x.value() < T(0) ? T(-1) : T(1))};
}

/// \brief Power with a passive exponent: d/dx x^p = p x^(p-1).
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> pow(const AutoDiffScalar<T, N>& x, const T& p) {
  return {pow(x.value(), p), x.derivatives() * (p * pow(x.value(), p - T(1)))};
}

/// \brief Two-argument arctangent: d atan2(y,x) = (x dy - y dx) / (x^2 + y^2).
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> atan2(const AutoDiffScalar<T, N>& y,
                                                            const AutoDiffScalar<T, N>& x) {
  const T inv = T(1) / (x.value() * x.value() + y.value() * y.value());
  return {atan2(y.value(), x.value()), (y.derivatives() * x.value() - x.derivatives() * y.value()) * inv};
}

/// \brief Copy the sign of a passive value onto x; scales the derivative by the resulting sign flip.
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> copysign(const AutoDiffScalar<T, N>& x, const T& s) {
  const T sign_s = (s < T(0)) ? T(-1) : T(1);
  const T sign_x = (x.value() < T(0)) ? T(-1) : T(1);
  return {copysign(x.value(), s), x.derivatives() * (sign_s * sign_x)};
}

// min/max return the selected operand, preserving its derivatives; a passive operand is a constant.
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> min(const AutoDiffScalar<T, N>& a,
                                                          const AutoDiffScalar<T, N>& b) {
  return (a.value() < b.value()) ? a : b;
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> max(const AutoDiffScalar<T, N>& a,
                                                          const AutoDiffScalar<T, N>& b) {
  return (a.value() > b.value()) ? a : b;
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> min(const AutoDiffScalar<T, N>& a, const T& b) {
  return (a.value() < b) ? a : AutoDiffScalar<T, N>(b);
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> max(const AutoDiffScalar<T, N>& a, const T& b) {
  return (a.value() > b) ? a : AutoDiffScalar<T, N>(b);
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> min(const T& a, const AutoDiffScalar<T, N>& b) {
  return (a < b.value()) ? AutoDiffScalar<T, N>(a) : b;
}
template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr AutoDiffScalar<T, N> max(const T& a, const AutoDiffScalar<T, N>& b) {
  return (a > b.value()) ? AutoDiffScalar<T, N>(a) : b;
}
//@}

/// \brief Stream as "value [d0, d1, ...]".
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const AutoDiffScalar<T, N>& a) {
  return os << a.value() << " " << a.derivatives();
}

//! \name Trait specializations for AutoDiffScalar
//@{

/// \brief Numeric traits for AutoDiffScalar. Numeric values come from the passive type; the related
/// types remain differentiable.
template <typename T, size_t N>
struct NumTraits<AutoDiffScalar<T, N>> : NumTraits<typename NumTraits<T>::Real> {
  using Real = AutoDiffScalar<typename NumTraits<T>::Real, N>;
  using NonInteger = AutoDiffScalar<typename NumTraits<T>::NonInteger, N>;
  using Literal = typename NumTraits<T>::Literal;
  static constexpr bool IsInteger = false;
  static constexpr bool IsComplex = false;
  static constexpr bool RequireInitialization = true;
};

/// \brief An AutoDiffScalar combined with its passive type yields an AutoDiffScalar (either order).
template <typename T, size_t N, typename Op>
struct ScalarBinaryOpTraits<AutoDiffScalar<T, N>, T, Op> {
  using ReturnType = AutoDiffScalar<T, N>;
};
template <typename T, size_t N, typename Op>
struct ScalarBinaryOpTraits<T, AutoDiffScalar<T, N>, Op> {
  using ReturnType = AutoDiffScalar<T, N>;
};
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_AUTODIFFSCALAR_HPP_
