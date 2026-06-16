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

#ifndef MUNDY_GEOM_COMPUTE_OBB_HPP_
#define MUNDY_GEOM_COMPUTE_OBB_HPP_

/// \file compute_obb.hpp
/// \brief `compute_obb` overloads for the standard MundyGeom primitive types.
///
/// Each overload returns the tightest OBB (as defined by the `OBB` primitive) for the given
/// input shape.  The OBB orientation is expressed as a unit quaternion that maps local box axes
/// to world space; the half-extents are the half-lengths of the box along those axes.
///
/// Where an explicit metric argument is accepted it is used to wrap the shape into the primary
/// periodic cell before computing the OBB — the same semantics as `compute_aabb`.

// External
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/primitives/OBB.hpp>
#include <mundy_geom/primitives/AABB.hpp>
#include <mundy_geom/primitives/Ellipsoid.hpp>
#include <mundy_geom/primitives/LineSegment.hpp>
#include <mundy_geom/primitives/Point.hpp>
#include <mundy_geom/primitives/Sphere.hpp>
#include <mundy_geom/primitives/Spherocylinder.hpp>
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>
#include <mundy_geom/periodicity.hpp>   // for unwrap_points_to_ref, reference_point
#include <mundy_math/Quaternion.hpp>    // for mundy::Quaternion, mundy::rotation_between
#include <mundy_math/Vector3.hpp>       // for mundy::Vector3
#include <mundy_math/cmath.hpp>         // for mundy::sqrt

namespace mundy {

// =============================================================================
// Point
// =============================================================================

/// @brief OBB of a point: identity orientation, zero half-extents.
template <ValidPointType PointType>
KOKKOS_FUNCTION OBB<typename PointType::value_type> compute_obb(const PointType& point) {
  using value_type = typename PointType::value_type;
  return OBB<value_type>{point, Quaternion<value_type>::identity(),
                         static_cast<value_type>(0),
                         static_cast<value_type>(0),
                         static_cast<value_type>(0)};
}
template <ValidPointType PointType, typename Metric>
KOKKOS_FUNCTION OBB<typename PointType::value_type> compute_obb(const PointType& point,
                                                                 const Metric& /*metric*/) {
  return compute_obb(point);
}

// =============================================================================
// Sphere
// =============================================================================

/// @brief OBB of a sphere: identity orientation, uniform half-extents equal to the radius.
template <ValidSphereType SphereType>
KOKKOS_FUNCTION OBB<typename SphereType::value_type> compute_obb(const SphereType& sphere) {
  using value_type = typename SphereType::value_type;
  return OBB<value_type>{sphere.center(), Quaternion<value_type>::identity(),
                         sphere.radius(), sphere.radius(), sphere.radius()};
}
template <ValidSphereType SphereType, typename Metric>
KOKKOS_FUNCTION OBB<typename SphereType::value_type> compute_obb(const SphereType& sphere,
                                                                  const Metric& /*metric*/) {
  return compute_obb(sphere);
}

// =============================================================================
// AABB
// =============================================================================

/// @brief OBB of an AABB: identity orientation, half-extents = (max - min) / 2.
template <ValidAABBType AABBType>
KOKKOS_FUNCTION OBB<typename AABBType::value_type> compute_obb(const AABBType& aabb) {
  using value_type = typename AABBType::value_type;
  const auto center = static_cast<value_type>(0.5) * (aabb.min_corner() + aabb.max_corner());
  const auto he     = static_cast<value_type>(0.5) * (aabb.max_corner() - aabb.min_corner());
  return OBB<value_type>{center, Quaternion<value_type>::identity(), he[0], he[1], he[2]};
}
template <ValidAABBType AABBType, typename Metric>
KOKKOS_FUNCTION OBB<typename AABBType::value_type> compute_obb(const AABBType& aabb,
                                                                const Metric& /*metric*/) {
  return compute_obb(aabb);
}

// =============================================================================
// Ellipsoid
// =============================================================================

/// @brief OBB of an ellipsoid: orientation from the stored quaternion, half-extents from the radii.
///
/// The ellipsoid's principal semi-axes ARE the local box axes.  The tightest box in the
/// principal frame is the axis-aligned box with half-extents equal to the three radii.
/// The quaternion that maps body axes to world is exactly the ellipsoid's stored orientation.
template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION OBB<typename EllipsoidType::value_type> compute_obb(const EllipsoidType& ellipsoid) {
  using value_type = typename EllipsoidType::value_type;
  const auto& r = ellipsoid.radii();
  return OBB<value_type>{ellipsoid.center(), ellipsoid.orientation(), r[0], r[1], r[2]};
}
template <ValidEllipsoidType EllipsoidType, typename Metric>
KOKKOS_FUNCTION OBB<typename EllipsoidType::value_type> compute_obb(const EllipsoidType& ellipsoid,
                                                                     const Metric& /*metric*/) {
  return compute_obb(ellipsoid);
}

// =============================================================================
// Spherocylinder
// =============================================================================

/// @brief OBB of a spherocylinder (capsule).
///
/// The capsule axis is the body z-axis rotated to world space.  The tightest box has:
///   - local z half-extent = length/2 + radius  (hemispherical caps along the long axis)
///   - local x/y half-extents = radius           (radial directions)
///
/// The box orientation matches the capsule orientation.
template <ValidSpherocylinderType SpherocylinderType>
KOKKOS_FUNCTION OBB<typename SpherocylinderType::value_type>
compute_obb(const SpherocylinderType& sc) {
  using value_type = typename SpherocylinderType::value_type;
  const value_type hz = static_cast<value_type>(0.5) * sc.length() + sc.radius();
  return OBB<value_type>{sc.center(), sc.orientation(), sc.radius(), sc.radius(), hz};
}
template <ValidSpherocylinderType SpherocylinderType, typename Metric>
KOKKOS_FUNCTION OBB<typename SpherocylinderType::value_type>
compute_obb(const SpherocylinderType& sc, const Metric& /*metric*/) {
  return compute_obb(sc);
}

// =============================================================================
// SpherocylinderSegment
// =============================================================================

/// @brief OBB of a spherocylinder segment.
///
/// The segment's long axis runs from start to end.  The OBB is centered at the segment midpoint,
/// oriented so that local z points from start to end, with:
///   - local z half-extent = segment_length/2 + radius
///   - local x/y half-extents = radius
template <ValidSpherocylinderSegmentType SegmentType>
KOKKOS_FUNCTION OBB<typename SegmentType::value_type>
compute_obb(const SegmentType& seg) {
  using value_type = typename SegmentType::value_type;
  using vec3_t = Vector3<value_type>;

  const auto& s = seg.start();
  const auto& e = seg.end();

  // Center at segment midpoint.
  const auto center = static_cast<value_type>(0.5) * (s + e);

  // Compute a quaternion that rotates world-z to the segment direction.
  const vec3_t seg_vec = e - s;
  const value_type seg_len = sqrt(seg_vec[0]*seg_vec[0] + seg_vec[1]*seg_vec[1] + seg_vec[2]*seg_vec[2]);
  const value_type hz = static_cast<value_type>(0.5) * seg_len + seg.radius();

  Quaternion<value_type> orient;
  if (seg_len < get_zero_tolerance<value_type>()) {
    // Degenerate segment: use identity orientation.
    orient = Quaternion<value_type>::identity();
  } else {
    // Rotate world-z = (0,0,1) to the normalized segment direction.
    const vec3_t axis_z{static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(1)};
    const vec3_t dir = seg_vec * (static_cast<value_type>(1) / seg_len);
    // cross(z, dir)
    const vec3_t cross{axis_z[1]*dir[2] - axis_z[2]*dir[1],
                       axis_z[2]*dir[0] - axis_z[0]*dir[2],
                       axis_z[0]*dir[1] - axis_z[1]*dir[0]};
    const value_type sin_angle = sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    const value_type cos_angle = axis_z[0]*dir[0] + axis_z[1]*dir[1] + axis_z[2]*dir[2];

    if (sin_angle < get_zero_tolerance<value_type>()) {
      // Parallel (same or opposite direction).
      if (cos_angle > static_cast<value_type>(0)) {
        orient = Quaternion<value_type>::identity();
      } else {
        // 180° rotation about x-axis.
        orient = Quaternion<value_type>{static_cast<value_type>(0),
                                        static_cast<value_type>(1),
                                        static_cast<value_type>(0),
                                        static_cast<value_type>(0)};
      }
    } else {
      const value_type half_sin = sqrt(static_cast<value_type>(0.5) * (static_cast<value_type>(1) - cos_angle));
      const value_type half_cos = sqrt(static_cast<value_type>(0.5) * (static_cast<value_type>(1) + cos_angle));
      const value_type inv_sin  = static_cast<value_type>(1) / sin_angle;
      orient = Quaternion<value_type>{half_cos,
                                      half_sin * cross[0] * inv_sin,
                                      half_sin * cross[1] * inv_sin,
                                      half_sin * cross[2] * inv_sin};
    }
  }

  return OBB<value_type>{center, orient, seg.radius(), seg.radius(), hz};
}
template <ValidSpherocylinderSegmentType SegmentType, typename Metric>
KOKKOS_FUNCTION OBB<typename SegmentType::value_type>
compute_obb(const SegmentType& seg, const Metric& metric) {
  return compute_obb(unwrap_points_to_ref(seg, metric, reference_point(seg)));
}

// =============================================================================
// LineSegment
// =============================================================================

/// @brief OBB of a line segment (zero radius): oriented along the segment with zero radial extents.
template <ValidLineSegmentType LineSegmentType>
KOKKOS_FUNCTION OBB<typename LineSegmentType::value_type>
compute_obb(const LineSegmentType& ls) {
  using value_type = typename LineSegmentType::value_type;
  using vec3_t = Vector3<value_type>;

  const auto& s = ls.start();
  const auto& e = ls.end();
  const auto center = static_cast<value_type>(0.5) * (s + e);
  const vec3_t seg_vec = e - s;
  const value_type seg_len = sqrt(seg_vec[0]*seg_vec[0] + seg_vec[1]*seg_vec[1] + seg_vec[2]*seg_vec[2]);
  const value_type hz = static_cast<value_type>(0.5) * seg_len;

  Quaternion<value_type> orient;
  if (seg_len < get_zero_tolerance<value_type>()) {
    orient = Quaternion<value_type>::identity();
  } else {
    const vec3_t axis_z{static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(1)};
    const vec3_t dir = seg_vec * (static_cast<value_type>(1) / seg_len);
    const vec3_t cross{axis_z[1]*dir[2] - axis_z[2]*dir[1],
                       axis_z[2]*dir[0] - axis_z[0]*dir[2],
                       axis_z[0]*dir[1] - axis_z[1]*dir[0]};
    const value_type sin_angle = sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    const value_type cos_angle = axis_z[0]*dir[0] + axis_z[1]*dir[1] + axis_z[2]*dir[2];
    if (sin_angle < get_zero_tolerance<value_type>()) {
      if (cos_angle > static_cast<value_type>(0)) {
        orient = Quaternion<value_type>::identity();
      } else {
        orient = Quaternion<value_type>{static_cast<value_type>(0),
                                        static_cast<value_type>(1),
                                        static_cast<value_type>(0),
                                        static_cast<value_type>(0)};
      }
    } else {
      const value_type half_sin = sqrt(static_cast<value_type>(0.5) * (static_cast<value_type>(1) - cos_angle));
      const value_type half_cos = sqrt(static_cast<value_type>(0.5) * (static_cast<value_type>(1) + cos_angle));
      const value_type inv_sin  = static_cast<value_type>(1) / sin_angle;
      orient = Quaternion<value_type>{half_cos,
                                      half_sin * cross[0] * inv_sin,
                                      half_sin * cross[1] * inv_sin,
                                      half_sin * cross[2] * inv_sin};
    }
  }

  return OBB<value_type>{center, orient,
                         static_cast<value_type>(0), static_cast<value_type>(0), hz};
}
template <ValidLineSegmentType LineSegmentType, typename Metric>
KOKKOS_FUNCTION OBB<typename LineSegmentType::value_type>
compute_obb(const LineSegmentType& ls, const Metric& metric) {
  return compute_obb(unwrap_points_to_ref(ls, metric, reference_point(ls)));
}

}  // namespace mundy

#endif  // MUNDY_GEOM_COMPUTE_OBB_HPP_
