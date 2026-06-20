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

#ifndef MUNDY_MATH_NUMTRAITS_HPP_
#define MUNDY_MATH_NUMTRAITS_HPP_

/// \file NumTraits.hpp
/// \brief Scalar-type concepts and per-scalar numeric traits (limits, precision, related types).
///
/// A pure traits header: it describes a scalar type and performs no arithmetic. A custom scalar (e.g.
/// an autodiff dual) becomes usable across MundyMath by satisfying \ref ValidScalarType and
/// specializing \c NumTraits<T> with the member set shown in \ref GenericNumTraits.

// External
#include <Kokkos_Core.hpp>           // for KOKKOS_INLINE_FUNCTION
#include <Kokkos_NumericTraits.hpp>  // for Kokkos::Experimental::{epsilon,infinity,finite_*,quiet_NaN,...}_v

// C++ core
#include <concepts>     // for std::convertible_to
#include <type_traits>  // for std::is_arithmetic_v, std::is_integral_v, std::is_signed_v, std::conditional_t

// Mundy
#include <mundy_math/ScalarBinaryOpTraits.hpp>

namespace mundy {

/// \brief Numeric traits for an arithmetic scalar type; the reusable base \c NumTraits forwards to.
///
/// Custom-scalar specializations must expose this same member set.
///
/// \tparam T An arithmetic scalar type.
template <typename T>
struct GenericNumTraits {
  /// \brief The real-valued counterpart of T (T itself for real scalars).
  using Real = T;

  /// \brief A type closed under division and roots; integers widen to double, others are unchanged.
  using NonInteger = std::conditional_t<std::is_integral_v<T>, double, T>;

  /// \brief The type of a numeric literal paired with a value of type T.
  using Literal = T;

  /// \brief Whether T represents integers.
  static constexpr bool IsInteger = std::is_integral_v<T>;

  /// \brief Whether T is a signed type.
  static constexpr bool IsSigned = std::is_signed_v<T>;

  /// \brief Whether T is complex; always false (complex scalars are not supported).
  static constexpr bool IsComplex = false;

  /// \brief Whether T must be constructed before use (true for non-trivial custom scalars).
  static constexpr bool RequireInitialization = !std::is_arithmetic_v<T>;

  /// \brief Machine epsilon: the gap between 1 and the next representable value.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real epsilon() {
    return Kokkos::Experimental::epsilon_v<Real>;
  }

  /// \brief Loose tolerance for fuzzy comparisons; 0 for exact types.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real dummy_precision() {
    return Real(0);
  }

  /// \brief Largest finite representable value.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real highest() {
    return Kokkos::Experimental::finite_max_v<Real>;
  }

  /// \brief Most negative finite representable value (min() for integers, -max() for floats).
  KOKKOS_INLINE_FUNCTION
  static constexpr Real lowest() {
    return Kokkos::Experimental::finite_min_v<Real>;
  }

  /// \brief Smallest positive normalized value.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real norm_min() {
    return Kokkos::Experimental::norm_min_v<Real>;
  }

  /// \brief Positive infinity (or highest() for types without an infinity).
  KOKKOS_INLINE_FUNCTION
  static constexpr Real infinity() {
    return Kokkos::Experimental::infinity_v<Real>;
  }

  /// \brief A quiet NaN.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real quiet_NaN() {
    return Kokkos::Experimental::quiet_NaN_v<Real>;
  }
};

/// \brief Numeric traits for a scalar-like type T.
///
/// The default forwards to \ref GenericNumTraits (valid for arithmetic primitives). Custom scalar
/// types opt into MundyMath by specializing this template with the same member set.
///
/// \tparam T The scalar type to describe.
template <typename T>
struct NumTraits : GenericNumTraits<T> {};

/// \brief float traits; nonzero fuzzy-comparison precision.
template <>
struct NumTraits<float> : GenericNumTraits<float> {
  KOKKOS_INLINE_FUNCTION
  static constexpr float dummy_precision() {
    return 1e-5f;
  }
};

/// \brief double traits; nonzero fuzzy-comparison precision.
template <>
struct NumTraits<double> : GenericNumTraits<double> {
  KOKKOS_INLINE_FUNCTION
  static constexpr double dummy_precision() {
    return 1e-12;
  }
};

/// \brief long double traits; nonzero fuzzy-comparison precision.
template <>
struct NumTraits<long double> : GenericNumTraits<long double> {
  KOKKOS_INLINE_FUNCTION
  static constexpr long double dummy_precision() {
    return 1e-15;
  }
};

//! \name NumTraits sanity checks
//@{
static_assert(NumTraits<double>::IsInteger == false);
static_assert(NumTraits<int>::IsInteger == true);
static_assert(NumTraits<unsigned>::IsSigned == false);
static_assert(std::is_same_v<NumTraits<double>::Real, double>);
static_assert(std::is_same_v<NumTraits<int>::NonInteger, double>);
static_assert(std::is_same_v<NumTraits<float>::NonInteger, float>);
static_assert(NumTraits<double>::RequireInitialization == false);
//@}

/// \brief Concept satisfied by any type that is a mathematical scalar.
/// It is a single value closed under +, -, *, /, unary - and constructible from a numeric literal.
template <typename T>
concept ValidScalarType = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
  { T(static_cast<typename NumTraits<T>::Literal>(1.0)) } -> std::convertible_to<T>;
};

static_assert(ValidScalarType<float>);
static_assert(ValidScalarType<double>);
static_assert(ValidScalarType<int>);

}  // namespace mundy

#endif  // MUNDY_MATH_NUMTRAITS_HPP_
