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
#include <stdexcept>    // for std::invalid_argument
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

/// \brief A unit vector perpendicular to v, chosen deterministically.
///
/// Every nonzero v admits infinitely many perpendiculars; this returns the same one every time for a
/// given v, and never a badly-conditioned one. It crosses v with whichever coordinate axis v is
/// *least* aligned with: that axis obeys |v . e| <= |v|/sqrt(3), so |v x e| >= sqrt(2/3)|v| and the
/// normalization below can never divide by a near-cancelled cross product. Crossing with a fixed
/// axis instead would collapse whenever v approached it.
///
/// v need not be unit; only its direction matters. The result is unit whenever v is nonzero.
///
/// \param[in] v The vector to find a perpendicular of.
/// \pre v is nonzero.
template <typename T, ValidAccessor<T> Accessor, typename OutputType = typename NumTraits<T>::NonInteger>
KOKKOS_INLINE_FUNCTION AVector3<OutputType> perp(const AVector3<T, Accessor>& v) {
  const AVector3<OutputType> v_out{static_cast<OutputType>(v[0]), static_cast<OutputType>(v[1]),
                                   static_cast<OutputType>(v[2])};
  const OutputType abs_x = abs(v_out[0]);
  const OutputType abs_y = abs(v_out[1]);
  const OutputType abs_z = abs(v_out[2]);

  const AVector3<OutputType> least_aligned_axis =
      (abs_x <= abs_y && abs_x <= abs_z)
          ? AVector3<OutputType>{static_cast<OutputType>(1), static_cast<OutputType>(0), static_cast<OutputType>(0)}
      : (abs_y <= abs_z)
          ? AVector3<OutputType>{static_cast<OutputType>(0), static_cast<OutputType>(1), static_cast<OutputType>(0)}
          : AVector3<OutputType>{static_cast<OutputType>(0), static_cast<OutputType>(0), static_cast<OutputType>(1)};

  const AVector3<OutputType> result = cross(v_out, least_aligned_axis);
  const OutputType result_norm = norm(result);
  MUNDY_THROW_ASSERT(result_norm > get_zero_tolerance<OutputType>(), std::invalid_argument,
                     "perp: v has zero length, so it has no well-defined perpendicular.");
  return result / result_norm;
}
//@}

//@}

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR3_HPP_
