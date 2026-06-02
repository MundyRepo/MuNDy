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
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// @brief Compute the bounding radius of a point
template <ValidPointType PointType>
KOKKOS_FUNCTION typename PointType::value_type compute_bounding_radius([[maybe_unused]] const PointType& point) {
  return static_cast<typename PointType::value_type>(0);
}

/// @brief Compute the bounding radius of a line segment
template <ValidLineSegmentType LineSegmentType>
KOKKOS_FUNCTION typename LineSegmentType::value_type compute_bounding_radius(const LineSegmentType& line_segment) {
  using value_type = typename LineSegmentType::value_type;
  const auto& start = line_segment.start();
  const auto& end = line_segment.end();
  const value_type length = mundy::norm(end - start);
  return static_cast<value_type>(0.5) * length;
}

/// @brief Compute the bounding radius of a sphere
template <ValidSphereType SphereType>
KOKKOS_FUNCTION typename SphereType::value_type compute_bounding_radius(const SphereType& sphere) {
  return sphere.radius();
}

/// @brief Compute the bounding radius of an ellipsoid
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION EllipsoidType::value_type compute_bounding_radius(const EllipsoidType& ellipsoid) {
  return mundy::max(ellipsoid.radii());
}

/// @brief Compute the bounding radius of a spherocylinder
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION typename SpherocylinderType::value_type compute_bounding_radius(
    const SpherocylinderType& spherocylinder) {
  using value_type = typename SpherocylinderType::value_type;
  const auto& radius = spherocylinder.radius();
  const auto& length = spherocylinder.length();
  return static_cast<value_type>(0.5) * length + radius;
}

/// @brief Compute the bounding radius of a spherocylinder segment
template <ValidSpherocylinderSegmentType SegmentType>
KOKKOS_FUNCTION typename SegmentType::value_type compute_bounding_radius(const SegmentType& segment) {
  using value_type = typename SegmentType::value_type;
  const auto& start = segment.start();
  const auto& end = segment.end();
  const auto& radius = segment.radius();
  const value_type length = mundy::norm(end - start);
  return static_cast<value_type>(0.5) * length + radius;
}

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_BOUNDING_RADIUS_HPP_
