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

#ifndef MUNDY_GEOM_TRANSFORM_HPP_
#define MUNDY_GEOM_TRANSFORM_HPP_

/// \file transform.hpp
/// \defgroup MundyGeomTransform mundy::transform
/// \brief Translation and rotation helpers for Mundy geometric primitives.

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_geom/primitives.hpp>  // for mundy::Point, mundy::Line, ...
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomTransform
/// @{

/*
Supported shapes:
  - Point
  - Line
  - LineSegment
  - VSegment
  - Circle3D
  - AABB
  - Sphere
  - Spherocylinder
  - SpherocylinderSegment
  - Ring
  - Ellipsoid
*/

//! \name Translate
//@{

/// \brief Translate a point by a given displacement vector
template <ValidPointType PointT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const PointT& point, const Vector3T& disp) {
  return point + disp;
}

/// \brief Translate a point by a given displacement vector (inplace)
template <ValidPointType PointT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(PointT& point, const Vector3T& disp) {
  point += disp;
}

/// \brief Translate a line by a given displacement vector
template <ValidLineType LineT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename LineT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const LineT& line, const Vector3T& disp) {
  return Line<typename Vector3T::value_type>(line.center() + disp, line.direction());
}

/// \brief Translate a line by a given displacement vector (inplace)
template <ValidLineType LineT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename LineT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(LineT& line, const Vector3T& disp) {
  line.center() += disp;
}

/// \brief Translate a line segment by a given displacement vector
template <ValidLineSegmentType LineSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const LineSegmentT& line_segment, const Vector3T& disp) {
  return LineSegment<typename Vector3T::value_type>(line_segment.start() + disp, line_segment.end() + disp);
}

/// \brief Translate a line segment by a given displacement vector (inplace)
template <ValidLineSegmentType LineSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(LineSegmentT& line_segment, const Vector3T& disp) {
  line_segment.start() += disp;
  line_segment.end() += disp;
}

/// \brief Translate a v-segment by a given displacement vector
template <ValidVSegmentType VSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename VSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const VSegmentT& v_segment, const Vector3T& disp) {
  return VSegment<typename Vector3T::value_type>(v_segment.start() + disp, v_segment.middle() + disp,
                                                 v_segment.end() + disp);
}

/// \brief Translate a v-segment by a given displacement vector (inplace)
template <ValidVSegmentType VSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename VSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(VSegmentT& v_segment, const Vector3T& disp) {
  v_segment.start() += disp;
  v_segment.middle() += disp;
  v_segment.end() += disp;
}

/// \brief Translate a circle3D by a given displacement vector
template <ValidCircle3DType Circle3DT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const Circle3DT& circle, const Vector3T& disp) {
  return Circle3D<typename Vector3T::value_type>(circle.center() + disp, circle.orientation(), circle.radius());
}

/// \brief Translate a circle3D by a given displacement vector (inplace)
template <ValidCircle3DType Circle3DT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(Circle3DT& circle, const Vector3T& disp) {
  circle.center() += disp;
}

/// \brief Translate an AABB by a given displacement vector
template <ValidAABBType AABBT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename AABBT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const AABBT& aabb, const Vector3T& disp) {
  return AABB<typename Vector3T::value_type>(aabb.min_corner() + disp, aabb.max_corner() + disp);
}

/// \brief Translate an AABB by a given displacement vector (inplace)
template <ValidAABBType AABBT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename AABBT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(AABBT& aabb, const Vector3T& disp) {
  aabb.min_corner() += disp;
  aabb.max_corner() += disp;
}

/// \brief Translate a sphere by a given displacement vector
template <ValidSphereType SphereT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SphereT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const SphereT& sphere, const Vector3T& disp) {
  return Sphere<typename Vector3T::value_type>(sphere.center() + disp, sphere.radius());
}

/// \brief Translate a sphere by a given displacement vector (inplace)
template <ValidSphereType SphereT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SphereT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(SphereT& sphere, const Vector3T& disp) {
  sphere.center() += disp;
}

/// \brief Translate a spherocylinder by a given displacement vector
template <ValidSpherocylinderType SpherocylinderT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const SpherocylinderT& spherocylinder, const Vector3T& disp) {
  return Spherocylinder<typename Vector3T::value_type>(spherocylinder.center() + disp, spherocylinder.orientation(),
                                                       spherocylinder.radius(), spherocylinder.length());
}

/// \brief Translate a spherocylinder by a given displacement vector (inplace)
template <ValidSpherocylinderType SpherocylinderT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(SpherocylinderT& spherocylinder, const Vector3T& disp) {
  spherocylinder.center() += disp;
}

/// \brief Translate a spherocylinder segment by a given displacement vector
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const SpherocylinderSegmentT& spherocylinder_segment, const Vector3T& disp) {
  return SpherocylinderSegment<typename Vector3T::value_type>(spherocylinder_segment.start() + disp,
                                                              spherocylinder_segment.end() + disp);
}

/// \brief Translate a spherocylinder segment by a given displacement vector (inplace)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderSegmentT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(SpherocylinderSegmentT& spherocylinder_segment, const Vector3T& disp) {
  spherocylinder_segment.start() += disp;
  spherocylinder_segment.end() += disp;
}

/// \brief Translate a ring by a given displacement vector
template <ValidRingType RingT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename RingT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const RingT& ring, const Vector3T& disp) {
  return Ring<typename Vector3T::value_type>(ring.center() + disp, ring.orientation(), ring.major_radius(),
                                             ring.minor_radius());
}

/// \brief Translate a ring by a given displacement vector (inplace)
template <ValidRingType RingT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename RingT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(RingT& ring, const Vector3T& disp) {
  ring.center() += disp;
}

/// \brief Translate an ellipsoid by a given displacement vector
template <ValidEllipsoidType EllipsoidT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION auto translate(const EllipsoidT& ellipsoid, const Vector3T& disp) {
  return Ellipsoid<typename Vector3T::value_type>(ellipsoid.center() + disp, ellipsoid.orientation(),
                                                  ellipsoid.radii());
}

/// \brief Translate an ellipsoid by a given displacement vector (inplace)
template <ValidEllipsoidType EllipsoidT, ValidVector3Type Vector3T>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidT::value_type, typename Vector3T::value_type>)
KOKKOS_INLINE_FUNCTION void translate_inplace(EllipsoidT& ellipsoid, const Vector3T& disp) {
  ellipsoid.center() += disp;
}
//@}

//! \name Rotate (about the origin)
//@{

/*
Supported shapes:
  - Point
  - Line
  - LineSegment
  - VSegment
  - Circle3D
  - AABB (not valid)
  - Sphere
  - Spherocylinder
  - SpherocylinderSegment
  - Ring
  - Ellipsoid
*/

/// \brief Rotate a point about the origin by a given quaternion
template <ValidPointType PointT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Point<typename QuaternionT::value_type> rotate(const PointT& point, const QuaternionT& q) {
  return q * point;
}

/// \brief Rotate a point about the origin by a given quaternion (inplace)
template <ValidPointType PointT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(PointT& point, const QuaternionT& q) {
  point = q * point;
}

/// \brief Rotate a line about the origin by a given quaternion
template <ValidLineType LineT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename LineT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Line<typename QuaternionT::value_type> rotate(const LineT& line, const QuaternionT& q) {
  return Line<typename QuaternionT::value_type>(q * line.center(), q * line.direction());
}

/// \brief Rotate a line about the origin by a given quaternion (inplace)
template <ValidLineType LineT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename LineT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(LineT& line, const QuaternionT& q) {
  line.center() = q * line.center();
  line.direction() = q * line.direction();
}

/// \brief Rotate a line segment about the origin by a given quaternion
template <ValidLineSegmentType LineSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION LineSegment<typename QuaternionT::value_type> rotate(const LineSegmentT& line_segment,
                                                                            const QuaternionT& q) {
  return LineSegment<typename QuaternionT::value_type>(q * line_segment.start(), q * line_segment.end());
}

/// \brief Rotate a line segment about the origin by a given quaternion (inplace)
template <ValidLineSegmentType LineSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(LineSegmentT& line_segment, const QuaternionT& q) {
  line_segment.start() = q * line_segment.start();
  line_segment.end() = q * line_segment.end();
}

/// \brief Rotate a v-segment about the origin by a given quaternion
template <ValidVSegmentType VSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename VSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION VSegment<typename QuaternionT::value_type> rotate(const VSegmentT& v_segment,
                                                                         const QuaternionT& q) {
  return VSegment<typename QuaternionT::value_type>(q * v_segment.start(), q * v_segment.middle(), q * v_segment.end());
}

/// \brief Rotate a v-segment about the origin by a given quaternion (inplace)
template <ValidVSegmentType VSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename VSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(VSegmentT& v_segment, const QuaternionT& q) {
  v_segment.start() = q * v_segment.start();
  v_segment.middle() = q * v_segment.middle();
  v_segment.end() = q * v_segment.end();
}

/// \brief Rotate a circle3D about the origin by a given quaternion
template <ValidCircle3DType Circle3DT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Circle3D<typename QuaternionT::value_type> rotate(const Circle3DT& circle,
                                                                         const QuaternionT& q) {
  return Circle3D<typename QuaternionT::value_type>(q * circle.center(), circle.orientation() * q, circle.radius());
}

/// \brief Rotate a circle3D about the origin by a given quaternion (inplace)
template <ValidCircle3DType Circle3DT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(Circle3DT& circle, const QuaternionT& q) {
  circle.center() = q * circle.center();
  circle.orientation() = circle.orientation() * q;
}

/// \brief Rotate an AABB about the origin by a given quaternion
template <ValidAABBType AABBT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename AABBT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION AABB<typename QuaternionT::value_type> rotate(const AABBT& aabb, const QuaternionT& q) {
  MUNDY_THROW_ASSERT(
      false, std::runtime_error,
      "AABB rotation is not well-defined. Should it becomes a generalized Box or should it be the aabb of said box?");
}

/// \brief Rotate an AABB about the origin by a given quaternion (inplace)
template <ValidAABBType AABBT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename AABBT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(AABBT& aabb, const QuaternionT& q) {
  MUNDY_THROW_ASSERT(
      false, std::runtime_error,
      "AABB rotation is not well-defined. Should it becomes a generalized Box or should it be the aabb of said box?");
}

/// \brief Rotate a sphere about the origin by a given quaternion
template <ValidSphereType SphereT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SphereT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Sphere<typename QuaternionT::value_type> rotate(const SphereT& sphere, const QuaternionT& q) {
  return Sphere<typename QuaternionT::value_type>(q * sphere.center(), sphere.radius());
}

/// \brief Rotate a sphere about the origin by a given quaternion (inplace)
template <ValidSphereType SphereT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SphereT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(SphereT& sphere, const QuaternionT& q) {
  sphere.center() = q * sphere.center();
}

/// \brief Rotate a spherocylinder about the origin by a given quaternion
template <ValidSpherocylinderType SpherocylinderT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Spherocylinder<typename QuaternionT::value_type> rotate(const SpherocylinderT& spherocylinder,
                                                                               const QuaternionT& q) {
  return Spherocylinder<typename QuaternionT::value_type>(q * spherocylinder.center(), spherocylinder.orientation() * q,
                                                          spherocylinder.radius(), spherocylinder.length());
}

/// \brief Rotate a spherocylinder about the origin by a given quaternion (inplace)
template <ValidSpherocylinderType SpherocylinderT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(SpherocylinderT& spherocylinder, const QuaternionT& q) {
  spherocylinder.center() = q * spherocylinder.center();
  spherocylinder.orientation() = spherocylinder.orientation() * q;
}

/// \brief Rotate a spherocylinder segment about the origin by a given quaternion
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<typename QuaternionT::value_type> rotate(
    const SpherocylinderSegmentT& spherocylinder_segment, const QuaternionT& q) {
  return SpherocylinderSegment<typename QuaternionT::value_type>(q * spherocylinder_segment.start(),
                                                                 q * spherocylinder_segment.end());
}

/// \brief Rotate a spherocylinder segment about the origin by a given quaternion (inplace)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename SpherocylinderSegmentT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(SpherocylinderSegmentT& spherocylinder_segment, const QuaternionT& q) {
  spherocylinder_segment.start() = q * spherocylinder_segment.start();
  spherocylinder_segment.end() = q * spherocylinder_segment.end();
}

/// \brief Rotate a ring about the origin by a given quaternion
template <ValidRingType RingT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename RingT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Ring<typename QuaternionT::value_type> rotate(const RingT& ring, const QuaternionT& q) {
  return Ring<typename QuaternionT::value_type>(q * ring.center(), ring.orientation() * q, ring.major_radius(),
                                                ring.minor_radius());
}

/// \brief Rotate a ring about the origin by a given quaternion (inplace)
template <ValidRingType RingT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename RingT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(RingT& ring, const QuaternionT& q) {
  ring.center() = q * ring.center();
  ring.orientation() = ring.orientation() * q;
}

/// \brief Rotate an ellipsoid about the origin by a given quaternion
template <ValidEllipsoidType EllipsoidT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION Ellipsoid<typename QuaternionT::value_type> rotate(const EllipsoidT& ellipsoid,
                                                                          const QuaternionT& q) {
  return Ellipsoid<typename QuaternionT::value_type>(q * ellipsoid.center(), ellipsoid.orientation() * q,
                                                     ellipsoid.radii());
}

/// \brief Rotate an ellipsoid about the origin by a given quaternion (inplace)
template <ValidEllipsoidType EllipsoidT, ValidQuaternionType QuaternionT>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidT::value_type, typename QuaternionT::value_type>)
KOKKOS_INLINE_FUNCTION void rotate_inplace(EllipsoidT& ellipsoid, const QuaternionT& q) {
  ellipsoid.center() = q * ellipsoid.center();
  ellipsoid.orientation() = ellipsoid.orientation() * q;
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_TRANSFORM_HPP_
