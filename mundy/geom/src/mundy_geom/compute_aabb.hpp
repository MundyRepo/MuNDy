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

#ifndef MUNDY_GEOM_COMPUTE_AABB_HPP_
#define MUNDY_GEOM_COMPUTE_AABB_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/AABB.hpp>
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>

namespace mundy {

namespace geom {

/// @brief Compute the axis-aligned bounding box of a point
template <ValidPointType PointType>
KOKKOS_FUNCTION AABB<typename PointType::scalar_t> compute_aabb(const PointType& point) {
  using scalar_t = typename PointType::scalar_t;
  const scalar_t x = point[0];
  const scalar_t y = point[1];
  const scalar_t z = point[2];
  return AABB<scalar_t>{x, y, z, x, y, z};
}

/// @brief Compute the axis-aligned bounding box of a line segment
template <ValidLineSegmentType LineSegmentType>
KOKKOS_FUNCTION AABB<typename LineSegmentType::scalar_t> compute_aabb(const LineSegmentType& line_segment) {
  using scalar_t = typename LineSegmentType::scalar_t;
  const auto& start = line_segment.start();
  const auto& end = line_segment.end();
  const scalar_t min_x = Kokkos::min(start[0], end[0]);
  const scalar_t min_y = Kokkos::min(start[1], end[1]);
  const scalar_t min_z = Kokkos::min(start[2], end[2]);
  const scalar_t max_x = Kokkos::max(start[0], end[0]);
  const scalar_t max_y = Kokkos::max(start[1], end[1]);
  const scalar_t max_z = Kokkos::max(start[2], end[2]);
  return AABB<scalar_t>{min_x, min_y, min_z, max_x, max_y, max_z};
}

/// @brief Compute the axis-aligned bounding box of a sphere
template <ValidSphereType SphereType>
KOKKOS_FUNCTION AABB<typename SphereType::scalar_t> compute_aabb(const SphereType& sphere) {
  using scalar_t = typename SphereType::scalar_t;
  constexpr mundy::math::Vector3<scalar_t> ones{static_cast<scalar_t>(1), static_cast<scalar_t>(1),
                                                static_cast<scalar_t>(1)};
  const mundy::math::Vector3<scalar_t> min_corner = sphere.center() - ones * sphere.radius();
  const mundy::math::Vector3<scalar_t> max_corner = sphere.center() + ones * sphere.radius();
  return AABB<scalar_t>{min_corner, max_corner};
}

/// @brief Compute the axis-aligned bounding box of an ellipsoid
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION AABB<typename EllipsoidType::scalar_t> compute_aabb(const EllipsoidType& ellipsoid) {
  using scalar_t = typename EllipsoidType::scalar_t;
  using point_t = Point<scalar_t>;
  const auto& center = ellipsoid.center();
  const auto& radii = ellipsoid.radii();
  const auto& orient = ellipsoid.orientation();

  // The body frame min/max corners are -/+ the radii
  // Rotate/translate these into the lab frame and find their min/max in each direction.
  const point_t rotated_radii = orient * radii;
  const point_t obb_min_corner = center - rotated_radii;
  const point_t obb_max_corner = center + rotated_radii;
  const scalar_t min_x = Kokkos::min(obb_min_corner[0], obb_max_corner[0]);
  const scalar_t min_y = Kokkos::min(obb_min_corner[1], obb_max_corner[1]);
  const scalar_t min_z = Kokkos::min(obb_min_corner[2], obb_max_corner[2]);
  const scalar_t max_x = Kokkos::max(obb_min_corner[0], obb_max_corner[0]);
  const scalar_t max_y = Kokkos::max(obb_min_corner[1], obb_max_corner[1]);
  const scalar_t max_z = Kokkos::max(obb_min_corner[2], obb_max_corner[2]);
  return AABB<scalar_t>{min_x, min_y, min_z, max_x, max_y, max_z};
}

/// @brief Compute the axis-aligned bounding box of a spherocylinder
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION AABB<typename SpherocylinderType::scalar_t> compute_aabb(const SpherocylinderType& spherocylinder) {
  using scalar_t = typename SpherocylinderType::scalar_t;
  using point_t = Point<scalar_t>;
  const auto& center = spherocylinder.center();
  const auto& orientation = spherocylinder.orientation();
  const auto& radius = spherocylinder.radius();
  const auto& length = spherocylinder.length();

  constexpr mundy::math::Vector3<scalar_t> z_axis = {static_cast<scalar_t>(0), static_cast<scalar_t>(0),
                                                     static_cast<scalar_t>(1)};
  const point_t scaled_dir = static_cast<scalar_t>(0.5) * length * (orientation * z_axis);
  const point_t obb_centerline_min_corner = center - scaled_dir;
  const point_t obb_centerline_max_corner = center + scaled_dir;
  const scalar_t min_x = Kokkos::min(obb_centerline_min_corner[0], obb_centerline_max_corner[0]) - radius;
  const scalar_t min_y = Kokkos::min(obb_centerline_min_corner[1], obb_centerline_max_corner[1]) - radius;
  const scalar_t min_z = Kokkos::min(obb_centerline_min_corner[2], obb_centerline_max_corner[2]) - radius;
  const scalar_t max_x = Kokkos::max(obb_centerline_min_corner[0], obb_centerline_max_corner[0]) + radius;
  const scalar_t max_y = Kokkos::max(obb_centerline_min_corner[1], obb_centerline_max_corner[1]) + radius;
  const scalar_t max_z = Kokkos::max(obb_centerline_min_corner[2], obb_centerline_max_corner[2]) + radius;
  return AABB<scalar_t>{min_x, min_y, min_z, max_x, max_y, max_z};
}

/// @brief Compute the axis-aligned bounding box of a spherocylinder segment
template <ValidSpherocylinderSegmentType SegmentType>
KOKKOS_FUNCTION AABB<typename SegmentType::scalar_t> compute_aabb(const SegmentType& segment) {
  using scalar_t = typename SegmentType::scalar_t;
  const auto& start = segment.start();
  const auto& end = segment.end();
  const auto& radius = segment.radius();
  const scalar_t min_x = Kokkos::min(start[0], end[0]) - radius;
  const scalar_t min_y = Kokkos::min(start[1], end[1]) - radius;
  const scalar_t min_z = Kokkos::min(start[2], end[2]) - radius;
  const scalar_t max_x = Kokkos::max(start[0], end[0]) + radius;
  const scalar_t max_y = Kokkos::max(start[1], end[1]) + radius;
  const scalar_t max_z = Kokkos::max(start[2], end[2]) + radius;
  return AABB<scalar_t>{min_x, min_y, min_z, max_x, max_y, max_z};
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_AABB_HPP_
