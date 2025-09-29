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

#ifndef MUNDY_MATH_DISTANCE_SEGMENTSEGMENT_HPP_
#define MUNDY_MATH_DISTANCE_SEGMENTSEGMENT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Matrix3.hpp>       // for mundy::math::Matrix3
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace math {

namespace distance {

/* Notes:

The purpose of the segment-segment distance function is to compute the distance between two line segments in 3D space.
How one performs this calculation varies widely from software to software and is not always well-documented. One of the
core difficulties in this calculation is addressing edge cases (parallel rods, perpendicular rods, degenerate rods,
etc). Importantly, some implementations do not account for all edge cases and lack of documentation makes it hard to
tell a-priori if a given implementation is correct or covers all possibilities.

We looked at existing implementations in VTK, NVIDIA Flex/PhysX, HOOMD-blue, and DCPquery from Geometric Tools. They all
have benefits and drawbacks. By far, VTK was the best documented and most thoroughly tested. Their code is located here
  VTK/Common/DataModel/vtkLine.cxx
and their tests here
  VTK/Common/DataModel/Testing/Cxx/UnitTestLine.cxx

However, they do perform some questionable operations, such as exact comparisons to 0.0 and 1.0 and using 1e-6 as their
zero tolerance.

To start, we'll copy VTK's algorithm, replace their use of Scalar[3] arrays with our Vector3, and attempt to replicate
their validation tests. We may also make minor modifications to the algorithm to improve performance, readability, and
accuracy.

A note about our Vector3s: mundy::math::Vector3 is templated by T and AccessorType. This allows Vector3 with non-owning
memory access. Usually, users will interact with Vector3<Scalar>, which defaults to owning a Scalar[3], but for function
calls that can accept Vector3's with non-owning memory, you should write Vector3<T> instead.
*/

//
// Determine the distance of the current vertex to the edge defined by the vertices provided. Returns distance squared.
// Note: line is assumed infinite in extent.
//
// Here, p1 and p2 are two non-coincident points on the line.
template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Scalar distance_from_point_to_line(const AVector3<Scalar, Accessor1, OwnershipType1>& x,
                                                   const AVector3<Scalar, Accessor2, OwnershipType2>& p1,
                                                   const AVector3<Scalar, Accessor3, OwnershipType3>& p2) {
  const auto np1 = x - p1;
  auto p1p2 = p1 - p2;
  const Scalar p1p2_norm_sq = mundy::math::dot(p1p2, p1p2);
  const Scalar np1_norm_sq = mundy::math::dot(np1, np1);
  if (p1p2_norm_sq != 0.0) {  // TODO(palmerb4): Exact comparison to 0.0 seems bad. Shouldn't we use a tolerance?
    p1p2 /= Kokkos::sqrt(p1p2_norm_sq);
  } else {
    return np1_norm_sq;
  }

  const Scalar proj = mundy::math::dot(np1, p1p2);
  return np1_norm_sq - proj * proj;
}

/**
 * Compute the squared distance of a point x to a finite line (p1,p2). The method
 * computes the parametric coordinate t and the point location on the
 * line. Note that t is constrained to the range and the closest point must lie within the finite line [p1,p2].
 *
 * We return the parameter of the closest point on the line segment in t and the closest point on the line segment in
 * closest_point. The distance squared between the point and the line segment is returned. By default, neither the
 * parameter nor the closest point are returned. If you want them, pass in a pointer to a Scalar and a Vector3
 * respectively.
 *
 */
template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename OwnershipType1, typename OwnershipType2, typename OwnershipType3, typename OwnershipType4>
KOKKOS_FUNCTION Scalar distance_sq_from_point_to_line_segment(
    const AVector3<Scalar, Accessor1, OwnershipType1>& x, const AVector3<Scalar, Accessor2, OwnershipType2>& p1,
    const AVector3<Scalar, Accessor3, OwnershipType3>& p2, AVector3<Scalar, Accessor4, OwnershipType4>& closest_point,
    Scalar& t) {
  // Determine appropriate vectors
  const auto p21 = p2 - p1;

  // Get parametric location
  const Scalar num = mundy::math::dot(p21, x - p1);
  if ((num < get_zero_tolerance<Scalar>()) & (num > -get_zero_tolerance<Scalar>())) {
    // CASE 1: The vector from p1 to x is orthogonal to the line.
    // In this case, the closest point is p1 and the parametric coordinate is 0.
    t = 0.0;
    closest_point = p1;
  } else {
    const Scalar denom = mundy::math::dot(p21, p21);

    if (denom < get_zero_tolerance<Scalar>()) {
      // CASE 2: The line is degenerate (i.e., p1 and p2 are numerically the same point).
      // In this case, either point could really be the closest point. We'll arbitrarily pick p1 and set t to 0.
      closest_point = p1;
      t = 0.0;
    } else {
      // CASE 3: The line is well-defined and we can compute the closest point.
      t = num / denom;

      if (t < 0.0) {
        // CASE 3.1: The parameter for the infinite line is less than 0. Therefore, the closest point is p1.
        t = 0.0;
        closest_point = p1;
      } else if (t > 1.0) {
        // CASE 3.2: The parameter for the infinite line is greater than 1. Therefore, the closest point is p2.
        t = 1.0;
        closest_point = p2;
      } else {
        // CASE 3.3: The closest point is falls within the line segment.
        closest_point = p1 + t * p21;
      }
    }
  }

  const Scalar distance_sq = mundy::math::two_norm_squared(closest_point - x);
  return distance_sq;
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Scalar distance_sq_from_point_to_line_segment(const AVector3<Scalar, Accessor1, OwnershipType1>& x,
                                                              const AVector3<Scalar, Accessor2, OwnershipType2>& p1,
                                                              const AVector3<Scalar, Accessor3, OwnershipType3>& p2) {
  // Define some temporary variables
  Vector3<Scalar> closest_point_tmp;
  Scalar t_tmp;

  // Determine appropriate vectors
  const auto p21 = p2 - p1;

  // Get parametric location
  const Scalar num = mundy::math::dot(p21, x - p1);
  if ((num < get_zero_tolerance<Scalar>()) & (num > -get_zero_tolerance<Scalar>())) {
    // CASE 1: The vector from p1 to x is orthogonal to the line.
    // In this case, the closest point is p1 and the parametric coordinate is 0.
    t_tmp = 0.0;
    closest_point_tmp = p1;
  } else {
    const Scalar denom = mundy::math::dot(p21, p21);

    if (denom < get_zero_tolerance<Scalar>()) {
      // CASE 2: The line is degenerate (i.e., p1 and p2 are numerically the same point).
      // In this case, either point could really be the closest point. We'll arbitrarily pick p1 and set t to 0.
      closest_point_tmp = p1;
      t_tmp = 0.0;
    } else {
      // CASE 3: The line is well-defined and we can compute the closest point.
      t_tmp = num / denom;

      if (t_tmp < 0.0) {
        // CASE 3.1: The parameter for the infinite line is less than 0. Therefore, the closest point is p1.
        t_tmp = 0.0;
        closest_point_tmp = p1;
      } else if (t_tmp > 1.0) {
        // CASE 3.2: The parameter for the infinite line is greater than 1. Therefore, the closest point is p2.
        t_tmp = 1.0;
        closest_point_tmp = p2;
      } else {
        // CASE 3.3: The closest point is falls within the line segment.
        closest_point_tmp = p1 + t_tmp * p21;
      }
    }
  }

  const Scalar distance_sq = mundy::math::two_norm_squared(closest_point_tmp - x);
  return distance_sq;
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename Accessor5, typename Accessor6, typename OwnershipType1, typename OwnershipType2,
          typename OwnershipType3, typename OwnershipType4, typename OwnershipType5, typename OwnershipType6>
KOKKOS_FUNCTION Scalar distance_sq_between_lines(const AVector3<Scalar, Accessor1, OwnershipType1>& l0,
                                                 const AVector3<Scalar, Accessor2, OwnershipType2>& l1,  // line 1
                                                 const AVector3<Scalar, Accessor3, OwnershipType3>& m0,
                                                 const AVector3<Scalar, Accessor4, OwnershipType4>& m1,  // line 2
                                                 AVector3<Scalar, Accessor5, OwnershipType5>& closest_point1,
                                                 AVector3<Scalar, Accessor6, OwnershipType6>& closest_point2,
                                                 Scalar& t1,
                                                 Scalar& t2)  // parametric coords of the closest points
{
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto u = l1 - l0;
  const auto v = m1 - m0;
  const auto w = l0 - m0;
  const Scalar a = mundy::math::dot(u, u);
  const Scalar b = mundy::math::dot(u, v);
  const Scalar c = mundy::math::dot(v, v);
  const Scalar d = mundy::math::dot(u, w);
  const Scalar e = mundy::math::dot(v, w);
  const Scalar D = a * c - b * b;  // always >= 0

  // Compute the line parameters of the two closest points
  if (D < get_zero_tolerance<Scalar>()) {
    // CASE 1: The lines are colinear.
    // TODO(palmerb4): The following confuses me. If the lines are colinear and the lines are actually infinite, then
    // all points on the lines are equally close to each other. In this case, I see no reason to prefer one point over
    // another. Arbitrarily, I would pick t = 0.5 for both lines. However, in the following, VTK chose t1 = 0.0 and
    // t2 = (b > c ? d / b : e / c). I don't understand why this is the case.
    t1 = 0.0;
    t2 = (b > c ? d / b : e / c);  // use the largest denominator
  } else {
    t1 = (b * e - c * d) / D;
    t2 = (a * e - b * d) / D;
  }

  // Compute the closest points on the lines
  closest_point1 = l0 + t1 * u;
  closest_point2 = m0 + t2 * v;
  const Scalar distance_sq = mundy::math::two_norm_squared(closest_point1 - closest_point2);
  return distance_sq;
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename OwnershipType1, typename OwnershipType2, typename OwnershipType3, typename OwnershipType4>
KOKKOS_FUNCTION Scalar distance_sq_between_lines(const AVector3<Scalar, Accessor1, OwnershipType1>& l0,
                                                 const AVector3<Scalar, Accessor2, OwnershipType2>& l1,
                                                 const AVector3<Scalar, Accessor3, OwnershipType3>& m0,
                                                 const AVector3<Scalar, Accessor4, OwnershipType4>& m1) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto u = l1 - l0;
  const auto v = m1 - m0;
  const auto w = l0 - m0;
  const Scalar a = mundy::math::dot(u, u);
  const Scalar b = mundy::math::dot(u, v);
  const Scalar c = mundy::math::dot(v, v);
  const Scalar d = mundy::math::dot(u, w);
  const Scalar e = mundy::math::dot(v, w);
  const Scalar D = a * c - b * b;  // always >= 0

  // Set up some temporary variables
  Scalar t1_tmp;
  Scalar t2_tmp;

  // Compute the line parameters of the two closest points
  if (D < get_zero_tolerance<Scalar>()) {
    // CASE 1: The lines are colinear.
    // TODO(palmerb4): The following confuses me. If the lines are colinear and the lines are actually infinite, then
    // all points on the lines are equally close to each other. In this case, I see no reason to prefer one point over
    // another. Arbitrarily, I would pick t = 0.5 for both lines. However, in the following, VTK chose t1 = 0.0 and
    // t2 = (b > c ? d / b : e / c). I don't understand why this is the case.
    t1_tmp = 0.0;
    t2_tmp = (b > c ? d / b : e / c);  // use the largest denominator
  } else {
    t1_tmp = (b * e - c * d) / D;
    t2_tmp = (a * e - b * d) / D;
  }

  // Compute the closest points on the lines
  const Vector3<Scalar> closest_point1_tmp = l0 + t1_tmp * u;
  const Vector3<Scalar> closest_point2_tmp = m0 + t2_tmp * v;
  const Scalar distance_sq = mundy::math::two_norm_squared(closest_point1_tmp - closest_point2_tmp);
  return distance_sq;
}

/**
 * Computes the squared shortest distance squared between two finite line segments
 * defined by their end points (l0,l1) and (m0,m1).
 * Upon return, the closest points on the two line segments will be stored
 * in closest_point1 and closest_point2. Their parametric coords (0 <= t0, t1 <= 1)
 * will be stored in t0 and t1. The return value is the shortest distance
 * squared between the two line-segments.
 */
template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename Accessor5, typename Accessor6, typename OwnershipType1, typename OwnershipType2,
          typename OwnershipType3, typename OwnershipType4, typename OwnershipType5, typename OwnershipType6>
KOKKOS_FUNCTION Scalar
distance_sq_between_line_segments(const AVector3<Scalar, Accessor1, OwnershipType1>& l0,
                                  const AVector3<Scalar, Accessor2, OwnershipType2>& l1,  // line segment 1
                                  const AVector3<Scalar, Accessor3, OwnershipType3>& m0,
                                  const AVector3<Scalar, Accessor4, OwnershipType4>& m1,  // line segment 2
                                  AVector3<Scalar, Accessor5, OwnershipType5>& closest_point1,
                                  AVector3<Scalar, Accessor6, OwnershipType6>& closest_point2, Scalar& t1, Scalar& t2) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto u = l1 - l0;
  const auto v = m1 - m0;
  const auto w = l0 - m0;
  const Scalar a = mundy::math::dot(u, u);
  const Scalar b = mundy::math::dot(u, v);
  const Scalar c = mundy::math::dot(v, v);
  const Scalar d = mundy::math::dot(u, w);
  const Scalar e = mundy::math::dot(v, w);
  const Scalar D = a * c - b * b;  // always >= 0

  // Compute the line parameters of the two closest points
  if (D < get_zero_tolerance<Scalar>()) {
    // CASE 1: The lines are colinear. Therefore, one of the four endpoints is the
    // point of closest approach. We'll directly compute the 4 distances and the closest point on the line.
    Scalar t_tmp1;
    Scalar t_tmp2;
    Scalar t_tmp3;
    Scalar t_tmp4;
    mundy::math::Vector3<Scalar> pn_tmp1;
    mundy::math::Vector3<Scalar> pn_tmp2;
    mundy::math::Vector3<Scalar> pn_tmp3;
    mundy::math::Vector3<Scalar> pn_tmp4;
    const Scalar dist_sq1 = distance_sq_from_point_to_line_segment(l0, m0, m1, pn_tmp1, t_tmp1);
    const Scalar dist_sq2 = distance_sq_from_point_to_line_segment(l1, m0, m1, pn_tmp2, t_tmp2);
    const Scalar dist_sq3 = distance_sq_from_point_to_line_segment(m0, l0, l1, pn_tmp3, t_tmp3);
    const Scalar dist_sq4 = distance_sq_from_point_to_line_segment(m1, l0, l1, pn_tmp4, t_tmp4);

    // Determine which of the 4 distances is the minimum using exact Scalar comparison.
    Scalar min_distance_sq = Kokkos::min(Kokkos::min(dist_sq1, dist_sq2), Kokkos::min(dist_sq3, dist_sq4));
    if (min_distance_sq == dist_sq1) {
      // CASE 1.1: l0 is closest to the line segment m0-m1.
      t1 = 0.0;
      t2 = t_tmp1;
      closest_point1 = l0;
      closest_point2 = pn_tmp1;
    } else if (min_distance_sq == dist_sq2) {
      // CASE 1.2: l1 is closest to the line segment m0-m1.
      t1 = 1.0;
      t2 = t_tmp2;
      closest_point1 = l1;
      closest_point2 = pn_tmp2;
    } else if (min_distance_sq == dist_sq3) {
      // CASE 1.3: m0 is closest to the line segment l0-l1.
      t1 = t_tmp3;
      t2 = 0.0;
      closest_point1 = pn_tmp3;
      closest_point2 = m0;
    } else {
      // CASE 1.4: m1 is closest to the line segment l0-l1.
      t1 = t_tmp4;
      t2 = 1.0;
      closest_point1 = pn_tmp4;
      closest_point2 = m1;
    }

    return min_distance_sq;
  }

  // CASE 2: The lines aren't parallel. Get the closest points on the infinite lines.
  // TODO(palmerb4): Comment the various if-else cases in terms of their physical meaning.
  Scalar sN = b * e - c * d;
  Scalar tN = a * e - b * d;
  Scalar sD = D;   // sc = sN / sD, default sD = D >= 0
  Scalar tD = D;   // tc = tN / tD, default tD = D >= 0
  if (sN < 0.0) {  // sc < 0 => the s=0 edge is visible
    sN = 0.0;
    tN = e;
    tD = c;
  } else if (sN > sD) {  // sc > 1 => the s=1 edge is visible
    sN = sD;
    tN = e + b;
    tD = c;
  }

  if (tN < 0.0) {  // tc < 0 => the t=0 edge is visible
    tN = 0.0;

    // Recompute sc for this edge
    if (-d < 0.0) {
      sN = 0.0;
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {  // tc > 1 => the t=1 edge is visible
    tN = tD;

    // Recompute sc for this edge
    if ((-d + b) < 0.0) {
      sN = 0;
    } else if ((-d + b) > a) {
      sN = sD;
    } else {
      sN = (-d + b);
      sD = a;
    }
  }

  // Finally, get the arch-length parameters, the corresponding closest points, and their distance squared.
  t1 = (fabs(sN) < get_zero_tolerance<Scalar>()) ? 0.0 : sN / sD;
  t2 = (fabs(tN) < get_zero_tolerance<Scalar>()) ? 0.0 : tN / tD;
  closest_point1 = l0 + t1 * u;
  closest_point2 = m0 + t2 * v;
  const Scalar distance_sq = mundy::math::two_norm_squared(closest_point1 - closest_point2);

  return distance_sq;
}

}  // namespace distance

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_SEGMENTSEGMENT_HPP_
