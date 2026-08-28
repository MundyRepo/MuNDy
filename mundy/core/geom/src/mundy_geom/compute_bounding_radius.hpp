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

/// \file compute_bounding_radius.hpp
/// \brief `compute_bounding_radius` overloads for the standard MundyGeom primitive types.
///
/// Each overload returns the radius of the smallest sphere centered at the shape's reference
/// point (usually its centroid) that fully encloses the shape.

// External
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/periodicity.hpp>  // for unwrap_points_to_ref, reference_point
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>

namespace mundy {

// =============================================================================
// Point
// =============================================================================

/// @brief Bounding radius of a point: zero.
template <ValidPointType PointType>
KOKKOS_FUNCTION typename PointType::value_type compute_bounding_radius([[maybe_unused]] const PointType& point) {
  return static_cast<typename PointType::value_type>(0);
}
template <ValidPointType PointType, typename Metric>
KOKKOS_FUNCTION typename PointType::value_type compute_bounding_radius([[maybe_unused]] const PointType& point,
                                                                       const Metric& /*metric*/) {
  return static_cast<typename PointType::value_type>(0);
}

// =============================================================================
// LineSegment
// =============================================================================

/// @brief Bounding radius of a line segment: half the segment length.
template <ValidLineSegmentType LineSegmentType>
KOKKOS_FUNCTION typename LineSegmentType::value_type compute_bounding_radius(const LineSegmentType& line_segment) {
  using value_type = typename LineSegmentType::value_type;
  const value_type length = mundy::norm(line_segment.end() - line_segment.start());
  return static_cast<value_type>(0.5) * length;
}
template <ValidLineSegmentType LineSegmentType, typename Metric>
KOKKOS_FUNCTION typename LineSegmentType::value_type compute_bounding_radius(const LineSegmentType& line_segment,
                                                                             const Metric& metric) {
  return compute_bounding_radius(unwrap_points_to_ref(line_segment, metric, reference_point(line_segment)));
}

// =============================================================================
// Sphere
// =============================================================================

/// @brief Bounding radius of a sphere: its radius.
template <ValidSphereType SphereType>
KOKKOS_FUNCTION typename SphereType::value_type compute_bounding_radius(const SphereType& sphere) {
  return sphere.radius();
}
template <ValidSphereType SphereType, typename Metric>
KOKKOS_FUNCTION typename SphereType::value_type compute_bounding_radius(const SphereType& sphere,
                                                                        const Metric& /*metric*/) {
  return compute_bounding_radius(sphere);
}

// =============================================================================
// Ellipsoid
// =============================================================================

/// @brief Bounding radius of an ellipsoid: the largest of the three semi-axis radii.
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION EllipsoidType::value_type compute_bounding_radius(const EllipsoidType& ellipsoid) {
  return mundy::max(ellipsoid.radii());
}
template <ValidEllipsoidType EllipsoidType, typename Metric>
KOKKOS_FUNCTION EllipsoidType::value_type compute_bounding_radius(const EllipsoidType& ellipsoid,
                                                                  const Metric& /*metric*/) {
  return compute_bounding_radius(ellipsoid);
}

// =============================================================================
// Spherocylinder
// =============================================================================

/// @brief Bounding radius of a spherocylinder (capsule): half the centerline length plus radius.
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION typename SpherocylinderType::value_type compute_bounding_radius(
    const SpherocylinderType& spherocylinder) {
  using value_type = typename SpherocylinderType::value_type;
  return static_cast<value_type>(0.5) * spherocylinder.length() + spherocylinder.radius();
}
template <ValidSpherocylinderType SpherocylinderType, typename Metric>
KOKKOS_FUNCTION typename SpherocylinderType::value_type compute_bounding_radius(
    const SpherocylinderType& spherocylinder, const Metric& /*metric*/) {
  return compute_bounding_radius(spherocylinder);
}

// =============================================================================
// SpherocylinderSegment
// =============================================================================

/// @brief Bounding radius of a spherocylinder segment: half the segment length plus radius.
template <ValidSpherocylinderSegmentType SegmentType>
KOKKOS_FUNCTION typename SegmentType::value_type compute_bounding_radius(const SegmentType& segment) {
  using value_type = typename SegmentType::value_type;
  const value_type length = mundy::norm(segment.end() - segment.start());
  return static_cast<value_type>(0.5) * length + segment.radius();
}
template <ValidSpherocylinderSegmentType SegmentType, typename Metric>
KOKKOS_FUNCTION typename SegmentType::value_type compute_bounding_radius(const SegmentType& segment,
                                                                         const Metric& metric) {
  return compute_bounding_radius(unwrap_points_to_ref(segment, metric, reference_point(segment)));
}

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_BOUNDING_RADIUS_HPP_
