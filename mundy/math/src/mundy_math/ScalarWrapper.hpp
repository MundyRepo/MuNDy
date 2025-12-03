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

#ifndef MUNDY_MATH_SCALARWRAPPER_HPP_
#define MUNDY_MATH_SCALARWRAPPER_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>  // for std::decay_t
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector.hpp>        // for mundy::math::Vector

namespace mundy {

namespace math {

/// \brief An owning/viewing scalar type
///
/// This scalar type is just a 1D vector with a single entry.
template <typename T, ValidAccessor<T> Accessor = Array<T, 1>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
using AScalarWrapper = AVector<T, 1, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
  requires std::is_arithmetic_v<T>
using ScalarView = AVector<T, 1, Accessor, Ownership::Views>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
  requires std::is_arithmetic_v<T>
using OwningScalar = AVector<T, 1, Accessor, Ownership::Owns>;

/// \brief (Implementation) Type trait to determine if a type is a AScalarWrapper
template <typename TypeToCheck>
struct is_scalar_wrapper_impl : std::false_type {};
//
template <typename T, typename Accessor, typename OwnershipType>
struct is_scalar_wrapper_impl<AScalarWrapper<T, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a AScalarWrapper
template <typename TypeToCheck>
struct is_scalar_wrapper : public is_scalar_wrapper_impl<std::decay_t<TypeToCheck>> {};
//
template <typename TypeToCheck>
constexpr bool is_scalar_wrapper_v = is_scalar_wrapper<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid AScalarWrapper type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our vectors.
template <typename ScalarWrapperType>
concept ValidScalarWrapperType = is_scalar_wrapper_v<std::decay_t<ScalarWrapperType>> &&
                                 requires(std::decay_t<ScalarWrapperType> scalar_wrapper,
                                          const std::decay_t<ScalarWrapperType> const_scalar_wrapper) {
                                   typename std::decay_t<ScalarWrapperType>::scalar_t;
                                   {
                                     scalar_wrapper[0]
                                   } -> std::convertible_to<typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     scalar_wrapper(0)
                                   } -> std::convertible_to<typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     const_scalar_wrapper[0]
                                   } -> std::convertible_to<const typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     const_scalar_wrapper(0)
                                   } -> std::convertible_to<const typename std::decay_t<ScalarWrapperType>::scalar_t>;
                                 };  // ValidScalarWrapperType

//! \name Special scalar operations
//@{

/// \brief Scalar-scalar multiplication (not otherwise inherited by the math of AVector)
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AScalarWrapper<U, Accessor1, Ownership1>& a,
                                                const AScalarWrapper<T, Accessor2, Ownership2>& b)
    -> AScalarWrapper<std::common_type_t<T, U>> {
  return AScalarWrapper<std::common_type_t<T, U>>{a[0] * b[0]};
}
//@}

//! \name atomic_load/store. Atomic memory management operations.
//
// \note Atomics are covered by Vector naturally, so we're using this space to make our atomic operations on
// scalars forward to Kokkos atomics. This way, we can always call mundy::math::atomic_add regardless of whether we're
// dealing with a scalar or a vector/matrix.
//
//@{

/// \brief Atomic s_copy = s.
template <typename T>
KOKKOS_INLINE_FUNCTION T atomic_load(T* const s) {
  return Kokkos::atomic_load(s);
}

/// \brief Atomic s = value.
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_store(T* const s, const U& value) {
  Kokkos::atomic_store(s, static_cast<T>(value));
}
//@}

//! \name atomic_[op] Atomic operation which don’t return anything. [op] might be add, sub, mul, div.
//@{

/// \brief Atomic s += value.
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_add(T* const s, const U& value) {
  Kokkos::atomic_add(s, static_cast<T>(value));
}

/// \brief Atomic s -= value.
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_sub(T* const s, const U& value) {
  Kokkos::atomic_sub(s, static_cast<T>(value));
}

/// \brief Atomic s *= value.
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_mul(T* const s, const U& value) {
  Kokkos::atomic_mul(s, static_cast<T>(value));
}

/// \brief Atomic s /= value.
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_div(T* const s, const U& value) {
  Kokkos::atomic_div(s, static_cast<T>(value));
}
//@}

//! \name atomic_fetch_[op] Various atomic operations which return the old value. [op] might be add, sub, mul, div.
//@{

/// \brief Atomic s += value (returns old s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_add(T* const s, const U& value) {
  return Kokkos::atomic_fetch_add(s, static_cast<T>(value));
}

/// \brief Atomic s -= value (returns old s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(T* const s, const U& value) {
  return Kokkos::atomic_fetch_sub(s, static_cast<T>(value));
}

/// \brief Atomic s *= value (returns old s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_mul(T* const s, const U& value) {
  return Kokkos::atomic_fetch_mul(s, static_cast<T>(value));
}

/// \brief Atomic s /= value (returns old s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_div(T* const s, const U& value) {
  return Kokkos::atomic_fetch_div(s, static_cast<T>(value));
}
//@}

//! \name atomic_[op]_fetch Various atomic operations which return the new value. [op] might be add, sub, mul, div.
//@{

/// \brief Atomic s += value (returns new s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_add_fetch(T* const s, const U& value) {
  return Kokkos::atomic_add_fetch(s, static_cast<T>(value));
}

/// \brief Atomic s -= value (returns new s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_sub_fetch(T* const s, const U& value) {
  return Kokkos::atomic_sub_fetch(s, static_cast<T>(value));
}

/// \brief Atomic s *= value (returns new s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_mul_fetch(T* const s, const U& value) {
  return Kokkos::atomic_mul_fetch(s, static_cast<T>(value));
}

/// \brief Atomic s /= value (returns new s)
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_div_fetch(T* const s, const U& value) {
  return Kokkos::atomic_div_fetch(s, static_cast<T>(value));
}
//@}

//! \name AScalarWrapper<T, Accessor> views
//@{

/// \brief A helper function to create a AScalarWrapper<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a AScalarWrapper<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   ScalarView<T, Accessor> s(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_scalar_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_scalar_view(Accessor& data) {
  return ScalarView<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_scalar_view(Accessor&& data) {
  return ScalarView<T, Accessor>(std::forward<Accessor>(data));
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_scalar(Accessor& data) {
  return OwningScalar<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_scalar(Accessor&& data) {
  return OwningScalar<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_SCALARWRAPPER_HPP_
