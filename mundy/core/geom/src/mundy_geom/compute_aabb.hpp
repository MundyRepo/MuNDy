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
#include <mundy_geom/primitives/AABB.hpp>
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// @brief Compute the axis-aligned bounding box of a point
template <ValidPointType PointType>
KOKKOS_FUNCTION AABB<typename PointType::scalar_t> compute_aabb(const PointType& point) {
  using scalar_t = typename PointType::scalar_t;
  const scalar_t x = point[0];
  const scalar_t y = point[1];
  const scalar_t z = point[2];
  return AABB<scalar_t>{x, y, z, x, y, z};
}
template <ValidPointType PointType, typename Metric>
KOKKOS_FUNCTION AABB<typename PointType::scalar_t> compute_aabb(const PointType& point, const Metric& /*metric*/) {
  return compute_aabb(point);
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
template <ValidLineSegmentType LineSegmentType, typename Metric>
KOKKOS_FUNCTION AABB<typename LineSegmentType::scalar_t> compute_aabb(const LineSegmentType& line_segment,
                                                                      const Metric& metric) {
  return compute_aabb(unwrap_points_to_ref(line_segment, metric, reference_point(line_segment)));
}

/// @brief Compute the axis-aligned bounding box of a sphere
template <ValidSphereType SphereType>
KOKKOS_FUNCTION AABB<typename SphereType::scalar_t> compute_aabb(const SphereType& sphere) {
  using scalar_t = typename SphereType::scalar_t;
  constexpr mundy::Vector3<scalar_t> ones{static_cast<scalar_t>(1), static_cast<scalar_t>(1), static_cast<scalar_t>(1)};
  const mundy::Vector3<scalar_t> min_corner = sphere.center() - ones * sphere.radius();
  const mundy::Vector3<scalar_t> max_corner = sphere.center() + ones * sphere.radius();
  return AABB<scalar_t>{min_corner, max_corner};
}
template <ValidSphereType SphereType, typename Metric>
KOKKOS_FUNCTION AABB<typename SphereType::scalar_t> compute_aabb(const SphereType& sphere, const Metric& /*metric*/) {
  return compute_aabb(sphere);
}

/// @brief Compute the axis-aligned bounding box of an ellipsoid
///
/// An oriented ellipsoid with center `c`, radii `r_i`, and lab-frame principal axes `u_i = Q e_i` can be written as
/// `x = c + sum_i alpha_i r_i u_i`, where `||alpha||_2 <= 1`. The half-width in lab coordinate direction `e_j` is
/// therefore
/// `max_{||alpha|| <= 1} |sum_i alpha_i r_i (e_j dot u_i)| = sqrt(sum_i (r_i u_i[j])^2)`.
///
/// In code, we form the three scaled lab-frame axes `r_i u_i`, square them component-wise, add the squared
/// contributions, and take a component-wise square root. The AABB is `center +/- extents`.
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION AABB<typename EllipsoidType::scalar_t> compute_aabb(const EllipsoidType& ellipsoid) {
  using scalar_t = typename EllipsoidType::scalar_t;
  using point_t = Point<scalar_t>;
  const auto& center = ellipsoid.center();
  const auto& radii = ellipsoid.radii();
  const auto& orient = ellipsoid.orientation();

  constexpr point_t body_x{static_cast<scalar_t>(1), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
  constexpr point_t body_y{static_cast<scalar_t>(0), static_cast<scalar_t>(1), static_cast<scalar_t>(0)};
  constexpr point_t body_z{static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(1)};
  const point_t scaled_axis0 = radii[0] * (orient * body_x);
  const point_t scaled_axis1 = radii[1] * (orient * body_y);
  const point_t scaled_axis2 = radii[2] * (orient * body_z);

  const point_t squared_extents = elementwise_mul(scaled_axis0, scaled_axis0) +
                                  elementwise_mul(scaled_axis1, scaled_axis1) +
                                  elementwise_mul(scaled_axis2, scaled_axis2);
  const point_t extents = apply([](scalar_t value) { return Kokkos::sqrt(value); }, squared_extents);
  return AABB<scalar_t>{center - extents, center + extents};
}
template <ValidEllipsoidType EllipsoidType, typename Metric>
KOKKOS_FUNCTION AABB<typename EllipsoidType::scalar_t> compute_aabb(const EllipsoidType& ellipsoid,
                                                                    const Metric& /*metric*/) {
  return compute_aabb(ellipsoid);
}

/// @brief Compute the axis-aligned bounding box of a spherocylinder
///
/// The oriented spherocylinder is a line segment swept by a sphere of radius `radius`. Its centerline runs along the
/// lab-frame image of the body z-axis with half-length `length / 2`, so the AABB is the centerline endpoint AABB padded
/// by `radius` in every coordinate direction.
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION AABB<typename SpherocylinderType::scalar_t> compute_aabb(const SpherocylinderType& spherocylinder) {
  using scalar_t = typename SpherocylinderType::scalar_t;
  using point_t = Point<scalar_t>;
  const auto& center = spherocylinder.center();
  const auto& orientation = spherocylinder.orientation();
  const auto& radius = spherocylinder.radius();
  const auto& length = spherocylinder.length();

  constexpr mundy::Vector3<scalar_t> z_axis = {static_cast<scalar_t>(0), static_cast<scalar_t>(0),
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
template <ValidSpherocylinderType SpherocylinderType, typename Metric>
KOKKOS_FUNCTION AABB<typename SpherocylinderType::scalar_t> compute_aabb(const SpherocylinderType& spherocylinder,
                                                                         const Metric& /*metric*/) {
  return compute_aabb(spherocylinder);
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
template <ValidSpherocylinderSegmentType SegmentType, typename Metric>
KOKKOS_FUNCTION AABB<typename SegmentType::scalar_t> compute_aabb(const SegmentType& segment, const Metric& metric) {
  return compute_aabb(unwrap_points_to_ref(segment, metric, reference_point(segment)));
}

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_AABB_HPP_
