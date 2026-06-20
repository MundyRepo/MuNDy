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

#ifndef MUNDY_MATH_VECTOR3_HPP_
#define MUNDY_MATH_VECTOR3_HPP_

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
#include <mundy_math/Accessor.hpp>              // for mundy::ValidAccessor
#include <mundy_math/Array.hpp>                 // for mundy::Array
#include <mundy_math/Matrix3.hpp>               // for mundy::Matrix3
#include <mundy_math/ScalarBinaryOpTraits.hpp>  // for mundy::scalar_product_result_t
#include <mundy_math/Tolerance.hpp>             // for mundy::get_zero_tolerance
#include <mundy_math/Vector.hpp>                // for mundy::Vector
#include <mundy_utils/throw_assert.hpp>         // for MUNDY_THROW_ASSERT

namespace mundy {

/// \brief A temporary concept to check if a type is a valid AVector3 type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our vectors.
template <typename Vector3Type>
concept ValidVector3Type = is_vector3_v<std::decay_t<Vector3Type>> &&
                           requires(std::decay_t<Vector3Type> vector3, const std::decay_t<Vector3Type> const_vector3) {
                             typename std::decay_t<Vector3Type>::value_type;
                             { vector3[0] } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;
                             { vector3[1] } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;
                             { vector3[2] } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;

                             { vector3(0) } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;
                             { vector3(1) } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;
                             { vector3(2) } -> std::convertible_to<typename std::decay_t<Vector3Type>::value_type>;

                             {
                               const_vector3[0]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;
                             {
                               const_vector3[1]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;
                             {
                               const_vector3[2]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;

                             {
                               const_vector3(0)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;
                             {
                               const_vector3(1)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;
                             {
                               const_vector3(2)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::value_type>;
                           };  // ValidVector3Type

//! \name Non-member functions
//@{

//! \name Special vector3 operations
//@{

/// \brief Cross product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto cross(const AVector3<U, Accessor1>& a, const AVector3<T, Accessor2>& b)
    -> AVector3<scalar_product_result_t<U, T>> {
  using R = scalar_product_result_t<U, T>;
  AVector3<R> result;
  result[0] = static_cast<R>(a[1] * b[2] - a[2] * b[1]);
  result[1] = static_cast<R>(a[2] * b[0] - a[0] * b[2]);
  result[2] = static_cast<R>(a[0] * b[1] - a[1] * b[0]);
  return result;
}
//@}

//! \name AVector3<T, Accessor> views
//@{

/// \brief A helper function to create a AVector3<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a AVector3<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   AVector3<T, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_vector3_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_vector3_view(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::forward<Accessor>(data)));
  return AVector3<T, decltype(data_storage)>(data_storage);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_vector3(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::move(data)));
  return AVector3<T, decltype(data_storage)>(data_storage);
}
//@}

//@}

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR3_HPP_
