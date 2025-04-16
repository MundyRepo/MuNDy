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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector.hpp>        // for mundy::math::Vector
#include <mundy_math/Matrix3.hpp>       // for mundy::math::Matrix3

namespace mundy {

namespace math {

/// \brief Class for a 3x1 vector with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
template <typename T, ValidAccessor<T> Accessor = Array<T, 3>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
using Vector3 = Vector<T, 3, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 3>>
  requires std::is_arithmetic_v<T>
using Vector3View = Vector<T, 3, Accessor, Ownership::Views>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 3>>
  requires std::is_arithmetic_v<T>
using OwningVector3 = Vector<T, 3, Accessor, Ownership::Owns>;

/// \brief (Implementation) Type trait to determine if a type is a Vector3
template <typename TypeToCheck>
struct is_vector3_impl : std::false_type {};
//
template <typename T, typename Accessor, typename OwnershipType>
struct is_vector3_impl<Vector3<T, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Vector3
template <typename TypeToCheck>
struct is_vector3 : public is_vector3_impl<std::decay_t<TypeToCheck>> {};
//
template <typename TypeToCheck>
constexpr bool is_vector3_v = is_vector3<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid Vector3 type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our vectors.
template <typename Vector3Type>
concept ValidVector3Type = is_vector3_v<std::decay_t<Vector3Type>> &&
                           requires(std::decay_t<Vector3Type> vector3, const std::decay_t<Vector3Type> const_vector3) {
                             typename std::decay_t<Vector3Type>::scalar_t;
                             { vector3[0] } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;
                             { vector3[1] } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;
                             { vector3[2] } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;

                             { vector3(0) } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;
                             { vector3(1) } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;
                             { vector3(2) } -> std::convertible_to<typename std::decay_t<Vector3Type>::scalar_t>;

                             {
                               const_vector3[0]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;
                             {
                               const_vector3[1]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;
                             {
                               const_vector3[2]
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;

                             {
                               const_vector3(0)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;
                             {
                               const_vector3(1)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;
                             {
                               const_vector3(2)
                             } -> std::convertible_to<const typename std::decay_t<Vector3Type>::scalar_t>;
                           };  // ValidVector3Type

//! \name Non-member functions
//@{

//! \name Special vector3 operations
//@{

/// \brief Cross product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto cross(const Vector3<U, Accessor1, Ownership1>& a,
                                            const Vector3<T, Accessor2, Ownership2>& b)
    -> Vector3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Vector3<CommonType> result;
  result[0] = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[2]) -
              static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[1]);
  result[1] = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[0]) -
              static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[2]);
  result[2] = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[1]) -
              static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[0]);
  return result;
}

/// \brief Element-wise product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto element_multiply(const Vector3<U, Accessor1, Ownership1>& a,
                                                       const Vector3<T, Accessor2, Ownership2>& b)
    -> Vector3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Vector3<CommonType> result;
  result[0] = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[0]);
  result[1] = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[1]);
  result[2] = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[2]);
  return result;
}
//@}

//! \name Vector3<T, Accessor> views
//@{

/// \brief A helper function to create a Vector3<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Vector3<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Vector3<T, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_vector3_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_vector3_view(Accessor& data) {
  return Vector3View<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_vector3_view(Accessor&& data) {
  return Vector3View<T, Accessor>(std::forward<Accessor>(data));
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_vector3(Accessor& data) {
  return OwningVector3<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_vector3(Accessor&& data) {
  return OwningVector3<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR3_HPP_
