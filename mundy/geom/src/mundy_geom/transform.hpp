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

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives.hpp>    // for mundy::geom::Point, mundy::geom::Line, ...

namespace mundy {

namespace geom {
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
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> translate(const Point<Scalar>& point, const math::Vector3<Scalar>& disp) {
  return point + disp;
}

/// \brief Translate a point by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Point<Scalar>& point, const math::Vector3<Scalar>& disp) {
  point += disp;
}

/// \brief Translate a line by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Line<Scalar> translate(const Line<Scalar>& line, const math::Vector3<Scalar>& disp) {
  return Line<Scalar>(line.center() + disp, line.direction());
}

/// \brief Translate a line by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Line<Scalar>& line, const math::Vector3<Scalar>& disp) {
  line.center() += disp;
}

/// \brief Translate a line segment by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> translate(const LineSegment<Scalar>& line_segment,
                                                     const math::Vector3<Scalar>& disp) {
  return LineSegment<Scalar>(line_segment.start() + disp, line_segment.end() + disp);
}

/// \brief Translate a line segment by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(LineSegment<Scalar>& line_segment, const math::Vector3<Scalar>& disp) {
  line_segment.start() += disp;
  line_segment.end() += disp;
}

/// \brief Translate a v-segment by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> translate(const VSegment<Scalar>& v_segment,
                                                  const math::Vector3<Scalar>& disp) {
  return VSegment<Scalar>(v_segment.start() + disp, v_segment.middle() + disp, v_segment.end() + disp);
}

/// \brief Translate a v-segment by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(VSegment<Scalar>& v_segment, const math::Vector3<Scalar>& disp) {
  v_segment.start() += disp;
  v_segment.middle() += disp;
  v_segment.end() += disp;
}

/// \brief Translate a circle3D by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> translate(const Circle3D<Scalar>& circle, const math::Vector3<Scalar>& disp) {
  return Circle3D<Scalar>(circle.center() + disp, circle.orientation(), circle.radius());
}

/// \brief Translate a circle3D by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Circle3D<Scalar>& circle, const math::Vector3<Scalar>& disp) {
  circle.center() += disp;
}

/// \brief Translate an AABB by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION AABB<Scalar> translate(const AABB<Scalar>& aabb, const math::Vector3<Scalar>& disp) {
  return AABB<Scalar>(aabb.min_corner() + disp, aabb.max_corner() + disp);
}

/// \brief Translate an AABB by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(AABB<Scalar>& aabb, const math::Vector3<Scalar>& disp) {
  aabb.min_corner() += disp;
  aabb.max_corner() += disp;
}

/// \brief Translate a sphere by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> translate(const Sphere<Scalar>& sphere, const math::Vector3<Scalar>& disp) {
  return Sphere<Scalar>(sphere.center() + disp, sphere.radius());
}

/// \brief Translate a sphere by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Sphere<Scalar>& sphere, const math::Vector3<Scalar>& disp) {
  sphere.center() += disp;
}

/// \brief Translate a spherocylinder by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> translate(const Spherocylinder<Scalar>& spherocylinder,
                                                        const math::Vector3<Scalar>& disp) {
  return Spherocylinder<Scalar>(spherocylinder.center() + disp, spherocylinder.orientation(), spherocylinder.radius(),
                                spherocylinder.length());
}

/// \brief Translate a spherocylinder by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Spherocylinder<Scalar>& spherocylinder,
                                              const math::Vector3<Scalar>& disp) {
  spherocylinder.center() += disp;
}

/// \brief Translate a spherocylinder segment by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> translate(
    const SpherocylinderSegment<Scalar>& spherocylinder_segment, const math::Vector3<Scalar>& disp) {
  return SpherocylinderSegment<Scalar>(spherocylinder_segment.start() + disp, spherocylinder_segment.end() + disp);
}

/// \brief Translate a spherocylinder segment by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                              const math::Vector3<Scalar>& disp) {
  spherocylinder_segment.start() += disp;
  spherocylinder_segment.end() += disp;
}

/// \brief Translate a ring by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Ring<Scalar> translate(const Ring<Scalar>& ring, const math::Vector3<Scalar>& disp) {
  return Ring<Scalar>(ring.center() + disp, ring.orientation(), ring.major_radius(), ring.minor_radius());
}

/// \brief Translate a ring by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Ring<Scalar>& ring, const math::Vector3<Scalar>& disp) {
  ring.center() += disp;
}

/// \brief Translate an ellipsoid by a given displacement vector
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Ellipsoid<Scalar> translate(const Ellipsoid<Scalar>& ellipsoid,
                                                   const math::Vector3<Scalar>& disp) {
  return Ellipsoid<Scalar>(ellipsoid.center() + disp, ellipsoid.orientation(), ellipsoid.radii());
}

/// \brief Translate an ellipsoid by a given displacement vector (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void translate_inplace(Ellipsoid<Scalar>& ellipsoid, const math::Vector3<Scalar>& disp) {
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
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> rotate(const Point<Scalar>& point, const math::Quaternion<Scalar>& q) {
  return q * point;
}

/// \brief Rotate a point about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Point<Scalar>& point, const math::Quaternion<Scalar>& q) {
  point = q * point;
}

/// \brief Rotate a line about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Line<Scalar> rotate(const Line<Scalar>& line, const math::Quaternion<Scalar>& q) {
  return Line<Scalar>(q * line.center(), q * line.direction());
}

/// \brief Rotate a line about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Line<Scalar>& line, const math::Quaternion<Scalar>& q) {
  line.center() = q * line.center();
  line.direction() = q * line.direction();
}

/// \brief Rotate a line segment about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> rotate(const LineSegment<Scalar>& line_segment,
                                                  const math::Quaternion<Scalar>& q) {
  return LineSegment<Scalar>(q * line_segment.start(), q * line_segment.end());
}

/// \brief Rotate a line segment about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(LineSegment<Scalar>& line_segment, const math::Quaternion<Scalar>& q) {
  line_segment.start() = q * line_segment.start();
  line_segment.end() = q * line_segment.end();
}

/// \brief Rotate a v-segment about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> rotate(const VSegment<Scalar>& v_segment, const math::Quaternion<Scalar>& q) {
  return VSegment<Scalar>(q * v_segment.start(), q * v_segment.middle(), q * v_segment.end());
}

/// \brief Rotate a v-segment about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(VSegment<Scalar>& v_segment, const math::Quaternion<Scalar>& q) {
  v_segment.start() = q * v_segment.start();
  v_segment.middle() = q * v_segment.middle();
  v_segment.end() = q * v_segment.end();
}

/// \brief Rotate a circle3D about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> rotate(const Circle3D<Scalar>& circle, const math::Quaternion<Scalar>& q) {
  return Circle3D<Scalar>(q * circle.center(), circle.orientation() * q, circle.radius());
}

/// \brief Rotate a circle3D about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Circle3D<Scalar>& circle, const math::Quaternion<Scalar>& q) {
  circle.center() = q * circle.center();
  circle.orientation() = circle.orientation() * q;
}

/// \brief Rotate an AABB about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION AABB<Scalar> rotate(const AABB<Scalar>& aabb, const math::Quaternion<Scalar>& q) {
  MUNDY_THROW_ASSERT(
      false, std::runtime_error,
      "AABB rotation is not well-defined. Should it becomes a generalized Box or should it be the aabb of said box?");
}

/// \brief Rotate an AABB about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(AABB<Scalar>& aabb, const math::Quaternion<Scalar>& q) {
  MUNDY_THROW_ASSERT(
      false, std::runtime_error,
      "AABB rotation is not well-defined. Should it becomes a generalized Box or should it be the aabb of said box?");
}

/// \brief Rotate a sphere about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> rotate(const Sphere<Scalar>& sphere, const math::Quaternion<Scalar>& q) {
  return Sphere<Scalar>(q * sphere.center(), sphere.radius());
}

/// \brief Rotate a sphere about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Sphere<Scalar>& sphere, const math::Quaternion<Scalar>& q) {
  sphere.center() = q * sphere.center();
}

/// \brief Rotate a spherocylinder about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> rotate(const Spherocylinder<Scalar>& spherocylinder,
                                                     const math::Quaternion<Scalar>& q) {
  return Spherocylinder<Scalar>(q * spherocylinder.center(), spherocylinder.orientation() * q, spherocylinder.radius(),
                                spherocylinder.length());
}

/// \brief Rotate a spherocylinder about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Spherocylinder<Scalar>& spherocylinder, const math::Quaternion<Scalar>& q) {
  spherocylinder.center() = q * spherocylinder.center();
  spherocylinder.orientation() = spherocylinder.orientation() * q;
}

/// \brief Rotate a spherocylinder segment about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> rotate(const SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                                            const math::Quaternion<Scalar>& q) {
  return SpherocylinderSegment<Scalar>(q * spherocylinder_segment.start(), q * spherocylinder_segment.end());
}

/// \brief Rotate a spherocylinder segment about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                           const math::Quaternion<Scalar>& q) {
  spherocylinder_segment.start() = q * spherocylinder_segment.start();
  spherocylinder_segment.end() = q * spherocylinder_segment.end();
}

/// \brief Rotate a ring about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Ring<Scalar> rotate(const Ring<Scalar>& ring, const math::Quaternion<Scalar>& q) {
  return Ring<Scalar>(q * ring.center(), ring.orientation() * q, ring.major_radius(), ring.minor_radius());
}

/// \brief Rotate a ring about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Ring<Scalar>& ring, const math::Quaternion<Scalar>& q) {
  ring.center() = q * ring.center();
  ring.orientation() = ring.orientation() * q;
}

/// \brief Rotate an ellipsoid about the origin by a given quaternion
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Ellipsoid<Scalar> rotate(const Ellipsoid<Scalar>& ellipsoid, const math::Quaternion<Scalar>& q) {
  return Ellipsoid<Scalar>(q * ellipsoid.center(), ellipsoid.orientation() * q, ellipsoid.radii());
}

/// \brief Rotate an ellipsoid about the origin by a given quaternion (inplace)
template <typename Scalar>
KOKKOS_INLINE_FUNCTION void rotate_inplace(Ellipsoid<Scalar>& ellipsoid, const math::Quaternion<Scalar>& q) {
  ellipsoid.center() = q * ellipsoid.center();
  ellipsoid.orientation() = ellipsoid.orientation() * q;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_TRANSFORM_HPP_
