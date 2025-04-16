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

#ifndef MUNDY_GEOM_COMPUTE_BOUNDING_RADIUS_HPP_
#define MUNDY_GEOM_COMPUTE_BOUNDING_RADIUS_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>

namespace mundy {

namespace geom {

/// @brief Compute the bounding radius of a point
template <ValidPointType PointType>
KOKKOS_FUNCTION typename PointType::scalar_t compute_bounding_radius([[maybe_unused]] const PointType& point) {
  return static_cast<typename PointType::scalar_t>(0);
}

/// @brief Compute the bounding radius of a line segment
template <ValidLineSegmentType LineSegmentType>
KOKKOS_FUNCTION typename LineSegmentType::scalar_t compute_bounding_radius(const LineSegmentType& line_segment) {
  using scalar_t = typename LineSegmentType::scalar_t;
  const auto& start = line_segment.start();
  const auto& end = line_segment.end();
  const scalar_t length = mundy::math::norm(end - start);
  return static_cast<scalar_t>(0.5) * length;
}

/// @brief Compute the bounding radius of a sphere
template <ValidSphereType SphereType>
KOKKOS_FUNCTION typename SphereType::scalar_t compute_bounding_radius(const SphereType& sphere) {
  return sphere.radius();
}

/// @brief Compute the bounding radius of an ellipsoid
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION EllipsoidType::scalar_t compute_bounding_radius(const EllipsoidType& ellipsoid) {
  return mundy::math::max(ellipsoid.radii());
}

/// @brief Compute the bounding radius of a spherocylinder
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION typename SpherocylinderType::scalar_t compute_bounding_radius(
    const SpherocylinderType& spherocylinder) {
  using scalar_t = typename SpherocylinderType::scalar_t;
  const auto& radius = spherocylinder.radius();
  const auto& length = spherocylinder.length();
  return static_cast<scalar_t>(0.5) * length + radius;
}

/// @brief Compute the bounding radius of a spherocylinder segment
template <ValidSpherocylinderSegmentType SegmentType>
KOKKOS_FUNCTION typename SegmentType::scalar_t compute_bounding_radius(const SegmentType& segment) {
  using scalar_t = typename SegmentType::scalar_t;
  const auto& start = segment.start();
  const auto& end = segment.end();
  const auto& radius = segment.radius();
  const scalar_t length = mundy::math::norm(end - start);
  return static_cast<scalar_t>(0.5) * length + radius;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_BOUNDING_RADIUS_HPP_
