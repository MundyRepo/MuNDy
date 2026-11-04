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

#ifndef MUNDY_GEOM_DISTANCE_LINESEGMENTLINESEGMENT_HPP_
#define MUNDY_GEOM_DISTANCE_LINESEGMENTLINESEGMENT_HPP_

// C++ core
#include <type_traits>

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance/Types.hpp>                            // for mundy::SharedNormalSigned
#include <mundy_geom/distance/impl/DistanceScalingImpl.hpp>         // for mundy::impl::{centred_scale_exponent, scaled_norm}
#include <mundy_geom/primitives/LineSegment.hpp>                    // for mundy::LineSegment
#include <mundy_geom/primitives/Point.hpp>                          // for mundy::Point
#include <mundy_math/Vector3.hpp>                                   // for mundy::Vector3, mundy::dot, mundy::cross
#include <mundy_math/cmath.hpp>                                     // for mundy::passive_scalar_t
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Compute the distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type distance(const LineSegmentType1& line_segment1,  //
                                                               const LineSegmentType2& line_segment2) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2);
}

/// \brief Compute the euclidean separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type distance([[maybe_unused]] const Euclidean distance_type,  //
                                                               const LineSegmentType1& line_segment1,           //
                                                               const LineSegmentType2& line_segment2) {
  // no difference between distance types for line segments
  return distance(SharedNormalSigned{}, line_segment1, line_segment2);
}

/// \brief Compute the euclidean separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type
    distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
             const LineSegmentType1& line_segment1,                    //
             const LineSegmentType2& line_segment2) {
  using Scalar = typename LineSegmentType1::value_type;
  Point<Scalar> closest_point1;
  Point<Scalar> closest_point2;
  Scalar arch_length1;
  Scalar arch_length2;
  mundy::Vector3<Scalar> sep;
  return distance(distance_type, line_segment1, line_segment2, closest_point1, closest_point2, arch_length1,
                  arch_length2, sep);
}

/// \brief Compute the distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type
    distance(const LineSegmentType1& line_segment1,                         //
             const LineSegmentType2& line_segment2,                         //
             Point<typename LineSegmentType1::value_type>& closest_point1,  //
             Point<typename LineSegmentType1::value_type>& closest_point2,  //
             typename LineSegmentType1::value_type& arch_length1,           //
             typename LineSegmentType1::value_type& arch_length2,           //
             mundy::Vector3<typename LineSegmentType1::value_type>& sep) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2,  //
                  closest_point1, closest_point2, arch_length1, arch_length2, sep);
}

/// \brief Compute the shared normal signed separation distance between two line segments
///
/// The closest points minimize |closest_point2 - closest_point1| over the two segments. Where that minimizer is not
/// unique (parallel segments whose projections overlap), any minimizing pair is returned. Each closest point equals
/// start + arch_length (end - start) with arch_length in [0, 1], sep = closest_point2 - closest_point1, and the return
/// value is |sep|. A segment shorter than about 2^-1020 times the other may be treated as the point at its start.
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type
    distance([[maybe_unused]] const SharedNormalSigned distance_type,       //
             const LineSegmentType1& line_segment1,                         //
             const LineSegmentType2& line_segment2,                         //
             Point<typename LineSegmentType1::value_type>& closest_point1,  //
             Point<typename LineSegmentType1::value_type>& closest_point2,  //
             typename LineSegmentType1::value_type& arch_length1,           //
             typename LineSegmentType1::value_type& arch_length2,           //
             mundy::Vector3<typename LineSegmentType1::value_type>& sep) {
  using Scalar = typename LineSegmentType1::value_type;
  using T = passive_scalar_t<Scalar>;

  // Minimize R(s, t) = |l(s) - m(t)|^2 over [0, 1]^2 without classifying the segments as parallel. Each parameter
  // after the first minimizes R exactly along one coordinate direction, so rounding in the first moves the result along
  // the valley of R, where R barely changes.
  const auto& l0 = line_segment1.start();
  const auto& l1 = line_segment1.end();
  const auto& m0 = line_segment2.start();
  const auto& m1 = line_segment2.end();

  // Quarter the endpoints when a difference of them would overflow.
  auto u = l1 - l0;
  auto v = m1 - m0;
  auto w = l0 - m0;
  const bool prescaled = !(impl::max_abs_component<T>(u, v, w) < mundy_infinity_v<T>);
  const T q = prescaled ? static_cast<T>(0.25) : static_cast<T>(1.0);
  if (prescaled) {
    u = l1 * q - l0 * q;
    v = m1 * q - m0 * q;
    w = l0 * q - m0 * q;
  }

  // a c - b^2 = |u x v|^2 and b e - c d = (u x v) . (v x w), formed on u, v, and w scaled alike by a power of two.
  // Every quotient below is invariant to that scale. A quotient by a subnormal is avoided: the parameter falls back to
  // the case of a zero denominator, which costs at most a relative 2^-511 of the segment lengths.
  const T mu = impl::max_abs_component<T>(u, u, u);
  const T mv = impl::max_abs_component<T>(v, v, v);
  const T mw = impl::max_abs_component<T>(w, w, w);
  const T norm_min = mundy_norm_min_v<T>;
  if (mu > static_cast<T>(0.0) || mv > static_cast<T>(0.0)) {
    // CASE 1: At least one segment has length. Minimize over both parameters.
    const int k = impl::centred_scale_exponent(mu, mv, mw);
    const auto us = impl::scale_by_pow2<T>(u, k);
    const auto vs = impl::scale_by_pow2<T>(v, k);
    const auto ws = impl::scale_by_pow2<T>(w, k);
    const Scalar a = mundy::dot(us, us);
    const Scalar b = mundy::dot(us, vs);
    const Scalar c = mundy::dot(vs, vs);
    const Scalar d = mundy::dot(us, ws);
    const Scalar e = mundy::dot(vs, ws);
    const auto n = mundy::cross(us, vs);
    const Scalar D = mundy::dot(n, n);
    const Scalar sN = mundy::dot(n, mundy::cross(vs, ws));

    // First, get s minimizing R on the infinite lines, or s = 0 when the lines are parallel and every s does.
    arch_length1 = (D >= norm_min) ? impl::clamp_to_unit_interval(sN / D) : static_cast<Scalar>(0.0);

    // Second, get t minimizing R for that s, or t = 0 when the second segment is a point.
    arch_length2 =
        (c >= norm_min) ? impl::clamp_to_unit_interval((b * arch_length1 + e) / c) : static_cast<Scalar>(0.0);

    // Third, get s minimizing R for that t, or s = 0 when the first segment is a point.
    arch_length1 =
        (a >= norm_min) ? impl::clamp_to_unit_interval((b * arch_length2 - d) / a) : static_cast<Scalar>(0.0);
  } else {
    // CASE 2: Both segments are points.
    arch_length1 = static_cast<Scalar>(0.0);
    arch_length2 = static_cast<Scalar>(0.0);
  }

  // Finally, get the corresponding closest points and their distance, undoing the quartering.
  if (!prescaled) {
    closest_point1 = l0 + arch_length1 * u;
    closest_point2 = m0 + arch_length2 * v;
    sep = closest_point2 - closest_point1;
  } else {
    const T inv_q = static_cast<T>(4.0);
    const auto cq1 = l0 * q + arch_length1 * u;
    const auto cq2 = m0 * q + arch_length2 * v;
    closest_point1 = cq1 * inv_q;
    closest_point2 = cq2 * inv_q;
    sep = (cq2 - cq1) * inv_q;
  }
  return impl::scaled_norm(sep);
}

/// \brief Compute the euclidean distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
MUNDY_REQUIRES(std::is_same_v<typename LineSegmentType1::value_type, typename LineSegmentType2::value_type>)
KOKKOS_FUNCTION typename LineSegmentType1::value_type
    distance([[maybe_unused]] const Euclidean distance_type,                //
             const LineSegmentType1& line_segment1,                         //
             const LineSegmentType2& line_segment2,                         //
             Point<typename LineSegmentType1::value_type>& closest_point1,  //
             Point<typename LineSegmentType1::value_type>& closest_point2,  //
             typename LineSegmentType1::value_type& arch_length1,           //
             typename LineSegmentType1::value_type& arch_length2,           //
             mundy::Vector3<typename LineSegmentType1::value_type>& sep) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2,  //
                  closest_point1, closest_point2, arch_length1, arch_length2, sep);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_LINESEGMENTLINESEGMENT_HPP_
