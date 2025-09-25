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

#ifndef MUNDY_GEOM_RANDOMIZE_HPP_
#define MUNDY_GEOM_RANDOMIZE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives.hpp>    // for mundy::geom::Point, mundy::geom::Line, ...
#include <mundy_math/Quaternion.hpp>    // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

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

/// \brief Generate a random point within a given bounding box
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Point<Scalar> generate_random_point(const AABB<Scalar>& box, RNG& rng) {
  // Generate a random point within the bounding box
  Scalar x = rng.uniform(box.x_min(), box.x_max());
  Scalar y = rng.uniform(box.y_min(), box.y_max());
  Scalar z = rng.uniform(box.z_min(), box.z_max());
  return Point<Scalar>(x, y, z);
}

/// \brief Generate a random unit vector (uniformly distributed on the unit sphere)
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Point<Scalar> generate_random_unit_vector(RNG& rng) {
  constexpr Scalar two_pi = 2.0 * Kokkos::numbers::pi_v<Scalar>;

  const Scalar zrand = static_cast<Scalar>(2) * rng.template rand<Scalar>() - static_cast<Scalar>(1);
  const Scalar wrand = Kokkos::sqrt(static_cast<Scalar>(1) - zrand * zrand);
  const Scalar trand = two_pi * rng.template rand<Scalar>();

  return math::Vector3<Scalar>{wrand * Kokkos::cos(trand), wrand * Kokkos::sin(trand), zrand};
}

/// \brief Generate a random unit quaternion mapping the z-axis to a random unit vector via parallel transport
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION math::Quaternion<Scalar> generate_random_unit_quaternion(RNG& rng) {
  constexpr math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
  math::Vector3<Scalar> u_hat = generate_random_unit_vector<Scalar>(rng);
  return math::quat_from_parallel_transport(z_hat, u_hat);
}

/// \brief Generate a random line with a center point within a given bounding box and a random direction
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Line<Scalar> generate_random_line(const AABB<Scalar>& box, RNG& rng) {
  // Generate random point in the domain and a random direction
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  math::Vector3<Scalar> direction = generate_random_unit_vector<Scalar>(rng);
  return Line<Scalar>(center, direction);
}

/// \brief Generate a random line segment with endpoints within a given bounding box
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> generate_random_line_segment(const AABB<Scalar>& box, RNG& rng) {
  // Generate two random points within the bounding box
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
  return LineSegment<Scalar>(p1, p2);
}

/// \brief Generate a random line segment with center in the given bounding box, random length within the given bounds,
/// and random orientation
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> generate_random_line_segment(const AABB<Scalar>& box, Scalar min_length,
                                                                        Scalar max_length, RNG& rng) {
  // Generate random center point in the domain
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);

  // Generate random tangent
  math::Vector3<Scalar> tangent = generate_random_unit_vector<Scalar>(rng);

  // Generate random length
  Scalar length = rng.uniform(min_length, max_length);

  // Compute endpoints
  Scalar half_length = 0.5 * length;
  Point<Scalar> p1 = center - half_length * tangent;
  Point<Scalar> p2 = center + half_length * tangent;
  return LineSegment<Scalar>(p1, p2);
}

/// \brief Generate a random v-segment with endpoints and middle point within a given bounding box
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> generate_random_vsegment(const AABB<Scalar>& box, RNG& rng) {
  // Generate three random points within the bounding box
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p3 = generate_random_point<Scalar>(box, rng);
  return VSegment<Scalar>(p1, p2, p3);
}

/// \brief Generate a random v-segment with center in the given bounding box, random edge lengths within the given
/// bounds, and random edge orientations
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> generate_random_vsegment(const AABB<Scalar>& box, Scalar min_length,
                                                                 Scalar max_length, RNG& rng) {
  // Generate random center point in the domain
  Point<Scalar> middle = generate_random_point<Scalar>(box, rng);

  // Generate two random tangents
  math::Vector3<Scalar> tangent1 = generate_random_unit_vector<Scalar>(rng);
  math::Vector3<Scalar> tangent2 = generate_random_unit_vector<Scalar>(rng);

  // Generate two random lengths
  Scalar length1 = rng.uniform(min_length, max_length);
  Scalar length2 = rng.uniform(min_length, max_length);

  // Compute endpoints and middle point
  Point<Scalar> start = middle - 0.5 * length1 * tangent1;
  Point<Scalar> end = middle + 0.5 * length2 * tangent2;
  return VSegment<Scalar>(start, middle, end);
}

/// \brief Generate a random circle3D with center point within a given bounding box, random radius, and random
/// orientation
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> generate_random_circle3D(const AABB<Scalar>& box, Scalar min_radius,
                                                                 Scalar max_radius, RNG& rng) {
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  Scalar radius = rng.uniform(min_radius, max_radius);
  auto random_quaternion = generate_random_unit_quaternion<Scalar>(rng);

  return Circle3D<Scalar>(center, random_quaternion, radius);
}

/// \brief Generate a random AABB within a given bounding box with both points inside the box
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION AABB<Scalar> generate_random_aabb(const AABB<Scalar>& box, RNG& rng) {
  // Generate two random points within the bounding box
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);

  // Compute min and max corners
  Scalar min_x = Kokkos::min(p1[0], p2[0]);
  Scalar max_x = Kokkos::max(p1[0], p2[0]);
  Scalar min_y = Kokkos::min(p1[1], p2[1]);
  Scalar max_y = Kokkos::max(p1[1], p2[1]);
  Scalar min_z = Kokkos::min(p1[2], p2[2]);
  Scalar max_z = Kokkos::max(p1[2], p2[2]);
  return AABB<Scalar>(Point<Scalar>(min_x, min_y, min_z), Point<Scalar>(max_x, max_y, max_z));
}

/// \brief Generate a random AABB with random size and center point inside the box
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION AABB<Scalar> generate_random_aabb(const AABB<Scalar>& box,
                                                         const math::Vector3<Scalar>& min_sizes,
                                                         const math::Vector3<Scalar>& max_sizes, RNG& rng) {
  // Generate a random center point within the bounding box
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);

  // Generate random sizes within the given bounds
  Scalar size_x = rng.uniform(min_sizes[0], max_sizes[0]);
  Scalar size_y = rng.uniform(min_sizes[1], max_sizes[1]);
  Scalar size_z = rng.uniform(min_sizes[2], max_sizes[2]);

  // Compute min and max corners
  Point<Scalar> min_corner = center - 0.5 * math::Vector3<Scalar>(size_x, size_y, size_z);
  Point<Scalar> max_corner = center + 0.5 * math::Vector3<Scalar>(size_x, size_y, size_z);

  return AABB<Scalar>(min_corner, max_corner);
}

/// \brief Generate a random sphere with center point within a given bounding box and random radius
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> generate_random_sphere(const AABB<Scalar>& box, Scalar min_radius,
                                                             Scalar max_radius, RNG& rng) {
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  Scalar radius = rng.uniform(min_radius, max_radius);
  return Sphere<Scalar>(center, radius);
}

/// \brief Generate a random spherocylinder with endpoints within a given bounding box and random radius
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> generate_random_spherocylinder(const AABB<Scalar>& box, Scalar min_radius,
                                                                             Scalar max_radius, RNG& rng) {
  // Generate two random points within the bounding box
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);

  // Convert them into a unit tangent and use that to get a quaternion
  auto tangent = p2 - p1;
  Scalar length = math::norm(tangent);
  tangent /= length;

  constexpr math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
  auto quaternion = math::quat_from_parallel_transport(z_hat, tangent);

  // Generate a random radius (between min_radius and max_radius)
  Scalar radius = rng.uniform(min_radius, max_radius);

  return Spherocylinder<Scalar>(0.5 * (p1 + p2), quaternion, radius, length);
}

/// \brief Generate a random spherocylinder with center point within a given bounding box, random radius, random length,
/// and random orientation
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> generate_random_spherocylinder(const AABB<Scalar>& box, Scalar min_radius,
                                                                             Scalar max_radius, Scalar min_length,
                                                                             Scalar max_length, RNG& rng) {
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  Scalar radius = rng.uniform(min_radius, max_radius);
  Scalar length = rng.uniform(min_length, max_length);
  auto random_quaternion = generate_random_unit_quaternion<Scalar>(rng);

  return Spherocylinder<Scalar>(center, random_quaternion, radius, length);
}

/// \brief Generate a random spherocylinder segment with endpoints within a given bounding box and random radius
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> generate_random_spherocylinder_segment(const AABB<Scalar>& box,
                                                                                            Scalar min_radius,
                                                                                            Scalar max_radius,
                                                                                            RNG& rng) {
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
  Scalar radius = rng.uniform(min_radius, max_radius);
  return SpherocylinderSegment<Scalar>(p1, p2, radius);
}

/// \brief Generate a random spherocylinder segment with center point within a given bounding box, random radius, random
/// length, and random orientation
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> generate_random_spherocylinder_segment(
    const AABB<Scalar>& box, Scalar min_radius, Scalar max_radius,  //
    Scalar min_length, Scalar max_length, RNG& rng) {
  auto line_segment = generate_random_line_segment(box, min_length, max_length, rng);
  Scalar radius = rng.uniform(min_radius, max_radius);
  return SpherocylinderSegment<Scalar>(line_segment.start(), line_segment.end(), radius);
}

/// \brief Generate a random ring with center point within a given bounding box, random major and minor radii, and
/// random orientation
template <typename Scalar, typename RNG>
KOKKOS_INLINE_FUNCTION Ring<Scalar> generate_random_ring(const AABB<Scalar>& box, Scalar min_major_radius,
                                                         Scalar max_major_radius, Scalar min_minor_radius,
                                                         Scalar max_minor_radius, RNG& rng) {
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  Scalar major_radius = rng.uniform(min_major_radius, max_major_radius);
  Scalar minor_radius = rng.uniform(min_minor_radius, max_minor_radius);
  auto random_quaternion = generate_random_unit_quaternion<Scalar>(rng);

  return Ring<Scalar>(center, random_quaternion, major_radius, minor_radius);
}

/// \brief Generate a random ellipsoid with center point within a given bounding box, random semi-axis radii, and random
/// orientation
template <typename Scalar, typename RNG>
Ellipsoid<Scalar> generate_random_ellipsoid(const AABB<Scalar>& box, const math::Vector3<Scalar>& min_radii,
                                            const math::Vector3<Scalar>& max_radii, RNG& rng) {
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  Scalar r0 = rng.uniform(min_radii[0], max_radii[0]);
  Scalar r1 = rng.uniform(min_radii[1], max_radii[1]);
  Scalar r2 = rng.uniform(min_radii[2], max_radii[2]);
  auto random_quaternion = generate_random_unit_quaternion<Scalar>(rng);

  return Ellipsoid<Scalar>(center, random_quaternion, r0, r1, r2);
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_RANDOMIZE_HPP_
