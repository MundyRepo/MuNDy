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

#ifndef MUNDY_MATH_SCALARBINARYOPTRAITS_HPP_
#define MUNDY_MATH_SCALARBINARYOPTRAITS_HPP_

/// \file ScalarBinaryOpTraits.hpp
/// \brief Result type and legality of a binary operation between two scalar types.
///
/// An operation is permitted iff the trait exposes a \c ReturnType; an unsupported mix has none and is
/// a compile error. Arithmetic primitives may mix (e.g. \c float and \c double); any other mix —
/// including a custom scalar with its passive type — is opted in by specialization.

// C++ core
#include <type_traits>  // for std::is_arithmetic_v, std::is_integral_v, std::common_type_t, std::conditional_t

namespace mundy {

/// \brief Operation tags selecting which binary operation a \ref ScalarBinaryOpTraits describes.
///
/// Per-operation tags let a single type pair yield different result types per operation — e.g. integer
/// division promotes to double while integer multiplication does not.
namespace scalar_binary_op {
struct sum {};
struct difference {};
struct product {};
struct quotient {};
struct unary_minus {};
struct min {};
struct max {};
}  // namespace scalar_binary_op

namespace impl {

/// \brief Result scalar for two arithmetic primitives under operation \c Op (default: their common type).
template <typename A, typename B, typename Op>
struct arithmetic_binary_result {
  using type = std::common_type_t<A, B>;
};

/// \brief Quotient promotes integer/integer to double, since integer division is lossy.
template <typename A, typename B>
struct arithmetic_binary_result<A, B, scalar_binary_op::quotient> {
  using type = std::conditional_t<std::is_integral_v<A> && std::is_integral_v<B>, double, std::common_type_t<A, B>>;
};

}  // namespace impl

/// \brief Determines whether a binary operation between scalar types A and B is allowed, and its result.
///
/// The primary template exposes no \c ReturnType: an unspecified mix is rejected at compile time.
/// Specializations supply a \c ReturnType to permit and shape the result of an operation.
///
/// \tparam A Left-hand scalar type.
/// \tparam B Right-hand scalar type.
/// \tparam Op An operation tag from \ref scalar_binary_op (defaults to product).
template <typename A, typename B, typename Op = scalar_binary_op::product>
struct ScalarBinaryOpTraits {};

/// \brief Same non-arithmetic type on both sides yields that type (e.g. autodiff dual op dual).
template <typename T, typename Op>
  requires(!std::is_arithmetic_v<T>)
struct ScalarBinaryOpTraits<T, T, Op> {
  using ReturnType = T;
};

/// \brief Two arithmetic primitives (same or mixed) yield their promoted common type.
template <typename A, typename B, typename Op>
  requires(std::is_arithmetic_v<A> && std::is_arithmetic_v<B>)
struct ScalarBinaryOpTraits<A, B, Op> {
  using ReturnType = typename impl::arithmetic_binary_result<A, B, Op>::type;
};

//! \name Result-type aliases
//@{

/// \brief The result scalar type of operation Op between A and B (compile error if unsupported).
template <typename A, typename B, typename Op = scalar_binary_op::product>
using scalar_binary_op_result_t = typename ScalarBinaryOpTraits<A, B, Op>::ReturnType;

template <typename A, typename B>
using scalar_sum_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::sum>;
template <typename A, typename B>
using scalar_difference_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::difference>;
template <typename A, typename B>
using scalar_product_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::product>;
template <typename A, typename B>
using scalar_quotient_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::quotient>;
template <typename A, typename B>
using scalar_unary_minus_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::unary_minus>;
template <typename A, typename B>
using scalar_min_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::min>;
template <typename A, typename B>
using scalar_max_result_t = scalar_binary_op_result_t<A, B, scalar_binary_op::max>;
//@}

/// \brief Whether operation \c Op between scalar types A and B is supported (the trait exposes a
/// \c ReturnType). Use to constrain mixed-type operators.
template <typename A, typename B, typename Op>
concept ScalarBinaryOpSupported = requires { typename ScalarBinaryOpTraits<A, B, Op>::ReturnType; };

//! \name ScalarBinaryOpTraits sanity checks
//@{
static_assert(std::is_same_v<scalar_product_result_t<float, double>, double>);
static_assert(std::is_same_v<scalar_product_result_t<int, int>, int>);
static_assert(std::is_same_v<scalar_sum_result_t<float, float>, float>);
static_assert(std::is_same_v<scalar_quotient_result_t<int, int>, double>);     // integer division promotes
static_assert(std::is_same_v<scalar_quotient_result_t<float, float>, float>);  // float division does not
static_assert(ScalarBinaryOpSupported<double, float, scalar_binary_op::product>);
static_assert(!ScalarBinaryOpSupported<int*, float, scalar_binary_op::product>);  // unspecified mixes are rejected
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_SCALARBINARYOPTRAITS_HPP_
