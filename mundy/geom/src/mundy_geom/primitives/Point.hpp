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

#ifndef MUNDY_GEOM_PRIMITIVES_POINT_HPP_
#define MUNDY_GEOM_PRIMITIVES_POINT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace geom {

/// @brief A point in 3D space
/// @tparam Scalar
///
/// The following is a methodological choice to use the Vector3 class as the underlying data structure for the Point
/// class. This is done to allow points to access the same mathematical operations as vectors (dot product, cross
/// product, etc.). Had we created our own interface, we would have hidden the mathematical operations from the user.
template <typename Scalar, mundy::math::ValidAccessor<Scalar> Accessor = mundy::math::Array<Scalar, 3>,
          typename OwnershipType = mundy::math::Ownership::Owns>
using Point = mundy::math::AVector3<Scalar, Accessor, OwnershipType>;
//
template <typename Scalar, mundy::math::ValidAccessor<Scalar> Accessor = mundy::math::Array<Scalar, 3>>
using OwningPoint = Point<Scalar, Accessor, mundy::math::Ownership::Owns>;
//
template <typename Scalar, mundy::math::ValidAccessor<Scalar> Accessor = mundy::math::Array<Scalar, 3>>
using PointView = Point<Scalar, Accessor, mundy::math::Ownership::Views>;

/// @brief Type trait to determine if a type is a Point
template <typename T>
using is_point = mundy::math::is_vector3<T>;
//
template <typename T>
constexpr bool is_point_v = is_point<T>::value;

/// @brief Concept to check if a type has the necessary properties to be a valid Point type
/// As a predicate to creating a new point type, specialize is_point for the new type.
template <typename PointType>
concept ValidPointType = requires(std::decay_t<PointType> point, const std::decay_t<PointType> const_point) {
  is_point_v<std::decay_t<PointType>>;
  typename std::decay_t<PointType>::scalar_t;
  { point[0] } -> std::convertible_to<typename std::decay_t<PointType>::scalar_t>;
  { point[1] } -> std::convertible_to<typename std::decay_t<PointType>::scalar_t>;
  { point[2] } -> std::convertible_to<typename std::decay_t<PointType>::scalar_t>;

  { const_point[0] } -> std::convertible_to<const typename std::decay_t<PointType>::scalar_t>;
  { const_point[1] } -> std::convertible_to<const typename std::decay_t<PointType>::scalar_t>;
  { const_point[2] } -> std::convertible_to<const typename std::decay_t<PointType>::scalar_t>;
};  // ValidPointType

static_assert(ValidPointType<Point<float>> && ValidPointType<const Point<float>> && ValidPointType<Point<double>> &&
                  ValidPointType<const Point<double>>,
              "Point must satisfy the ValidPointType concept.");

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_POINT_HPP_
