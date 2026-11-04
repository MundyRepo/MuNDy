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

#ifndef MUNDY_GEOM_DISTANCE_LINELINE_HPP_
#define MUNDY_GEOM_DISTANCE_LINELINE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/Types.hpp>                            // for mundy::SharedNormalSigned
#include <mundy_geom/distance/impl/DistanceScalingImpl.hpp>         // for mundy::impl::{centred_scale_exponent, scaled_norm}
#include <mundy_geom/primitives/Line.hpp>                           // for mundy::Line
#include <mundy_geom/primitives/Point.hpp>                          // for mundy::Point
#include <mundy_math/Vector3.hpp>                                   // for mundy::Vector3, mundy::dot, mundy::cross
#include <mundy_math/cmath.hpp>                                     // for mundy::passive_scalar_t
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Compute the distance between two lines (defaults to SharedNormalSigned distance)
/// \tparam Scalar The scalar type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <ValidLineType LineType1, ValidLineType LineType2>
MUNDY_REQUIRES(std::is_same_v<typename LineType1::value_type, typename LineType2::value_type>)
KOKKOS_FUNCTION typename LineType1::value_type distance(const LineType1& line1,  //
                                                        const LineType2& line2) {
  return distance(SharedNormalSigned{}, line1, line2);
}

/// \brief Compute the distance between two lines
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <ValidLineType LineType1, ValidLineType LineType2>
MUNDY_REQUIRES(std::is_same_v<typename LineType1::value_type, typename LineType2::value_type>)
KOKKOS_FUNCTION typename LineType1::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const LineType1& line1,                                   //
                                                        const LineType2& line2) {
  using Scalar = typename LineType1::value_type;
  Point<Scalar> closest_point1;
  Point<Scalar> closest_point2;
  Scalar arch_length1;
  Scalar arch_length2;
  mundy::Vector3<Scalar> sep;
  return distance(distance_type, line1, line2, closest_point1, closest_point2, arch_length1, arch_length2, sep);
}

/// \brief Compute the distance between two lines
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <ValidLineType LineType1, ValidLineType LineType2>
MUNDY_REQUIRES(std::is_same_v<typename LineType1::value_type, typename LineType2::value_type>)
KOKKOS_FUNCTION typename LineType1::value_type distance([[maybe_unused]] const Euclidean distance_type,  //
                                                        const LineType1& line1,                          //
                                                        const LineType2& line2) {
  return distance(SharedNormalSigned{}, line1, line2);  // no difference between distance types for lines
}

/// \brief Compute the distance between two lines (defaults to SharedNormalSigned distance)
/// \tparam Scalar The scalar type
/// \param[in] line1 One line
/// \param[in] line2 The other line
/// \param[out] closest_point1 The closest point on line1
/// \param[out] closest_point2 The closest point on line2
/// \param[out] arch_length1 The arch-length parameter of the closest point on line1
/// \param[out] arch_length2 The arch-length parameter of the closest point on line2
/// \param[out] sep The separation vector (from line1 to line2)
template <ValidLineType LineType1, ValidLineType LineType2>
MUNDY_REQUIRES(std::is_same_v<typename LineType1::value_type, typename LineType2::value_type>)
KOKKOS_FUNCTION typename LineType1::value_type distance(const LineType1& line1,                                 //
                                                        const LineType2& line2,                                 //
                                                        Point<typename LineType1::value_type>& closest_point1,  //
                                                        Point<typename LineType1::value_type>& closest_point2,  //
                                                        typename LineType1::value_type& arch_length1,           //
                                                        typename LineType1::value_type& arch_length2,           //
                                                        mundy::Vector3<typename LineType1::value_type>& sep) {
  return distance(SharedNormalSigned{}, line1, line2,  //
                  closest_point1, closest_point2, arch_length1, arch_length2, sep);
}

/// \brief Compute the distance between two lines
///
/// The closest points minimize |closest_point2 - closest_point1| over the two lines. Where that minimizer is not
/// unique (parallel lines), any minimizing pair is returned. Each closest point equals center + arch_length direction,
/// sep = closest_point2 - closest_point1, and the return value is |sep|. A zero direction makes its line the point at
/// its center.
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
/// \param[out] closest_point1 The closest point on line1
/// \param[out] closest_point2 The closest point on line2
/// \param[out] arch_length1 The arch-length parameter of the closest point on line1
/// \param[out] arch_length2 The arch-length parameter of the closest point on line2
/// \param[out] sep The separation vector (from line1 to line2)
template <ValidLineType LineType1, ValidLineType LineType2>
MUNDY_REQUIRES(std::is_same_v<typename LineType1::value_type, typename LineType2::value_type>)
KOKKOS_FUNCTION typename LineType1::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const LineType1& line1,                                   //
                                                        const LineType2& line2,                                   //
                                                        Point<typename LineType1::value_type>& closest_point1,    //
                                                        Point<typename LineType1::value_type>& closest_point2,    //
                                                        typename LineType1::value_type& arch_length1,             //
                                                        typename LineType1::value_type& arch_length2,             //
                                                        mundy::Vector3<typename LineType1::value_type>& sep) {
  using Scalar = typename LineType1::value_type;
  using T = passive_scalar_t<Scalar>;

  // Minimize R(s, t) = |l(s) - m(t)|^2 without classifying the lines as parallel. Each parameter after the first
  // minimizes R exactly along one coordinate direction, so rounding in the first moves the result along the valley of
  // R, where R barely changes, and parallel lines get a minimizing pair.
  const auto& c1 = line1.center();
  const auto& c2 = line2.center();
  const auto& d1 = line1.direction();
  const auto& d2 = line2.direction();

  // Quarter the centers and directions alike, to which s and t are invariant, when the centers' difference would
  // overflow.
  auto u = d1 * static_cast<T>(1.0);
  auto v = d2 * static_cast<T>(1.0);
  auto w = c1 - c2;
  const bool prescaled = !(impl::max_abs_component<T>(u, v, w) < mundy_infinity_v<T>);
  const T q = prescaled ? static_cast<T>(0.25) : static_cast<T>(1.0);
  if (prescaled) {
    u = d1 * q;
    v = d2 * q;
    w = c1 * q - c2 * q;
  }

  // a c - b^2 = |u x v|^2 and b e - c d = (u x v) . (v x w), formed on u, v, and w scaled alike by a power of two,
  // so |u x v|^2 is about the squared sine of the angle between the lines. Every quotient below is invariant to that
  // scale. A quotient by a subnormal is avoided: the parameter falls back to the case of a zero denominator.
  const T mu = impl::max_abs_component<T>(u, u, u);
  const T mv = impl::max_abs_component<T>(v, v, v);
  const T mw = impl::max_abs_component<T>(w, w, w);
  const T norm_min = mundy_norm_min_v<T>;
  if (mu > static_cast<T>(0.0) || mv > static_cast<T>(0.0)) {
    // CASE 1: At least one line has a direction. Minimize over both parameters.
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

    // First, get s minimizing R, or s = 0 when the lines are parallel and every s does. Unlike a segment's, a line's s
    // is unbounded, so |u x v| within the rounding of its own products (4 eps |u| |v|) counts as parallel: s = 0 there
    // moves the pair along the valley of R by at most that relative amount of the exact closest points' distance from
    // the centers, which their own rounding already allows.
    const T eps = mundy_epsilon_v<T>;
    const bool parallel = !(D > static_cast<T>(16.0) * eps * eps * a * c) || !(D >= norm_min);
    arch_length1 = parallel ? static_cast<Scalar>(0.0) : sN / D;

    // Second, get t minimizing R for that s, or t = 0 when the second line is a point.
    arch_length2 = (c >= norm_min) ? (b * arch_length1 + e) / c : static_cast<Scalar>(0.0);

    // Third, get s minimizing R for that t, or s = 0 when the first line is a point.
    arch_length1 = (a >= norm_min) ? (b * arch_length2 - d) / a : static_cast<Scalar>(0.0);
  } else {
    // CASE 2: Both lines are points.
    arch_length1 = static_cast<Scalar>(0.0);
    arch_length2 = static_cast<Scalar>(0.0);
  }

  // Finally, get the corresponding closest points and their distance, undoing the quartering.
  if (!prescaled) {
    closest_point1 = c1 + arch_length1 * d1;
    closest_point2 = c2 + arch_length2 * d2;
    sep = closest_point2 - closest_point1;
  } else {
    const T inv_q = static_cast<T>(4.0);
    const auto cq1 = c1 * q + arch_length1 * u;
    const auto cq2 = c2 * q + arch_length2 * v;
    closest_point1 = cq1 * inv_q;
    closest_point2 = cq2 * inv_q;
    sep = (cq2 - cq1) * inv_q;
  }
  return impl::scaled_norm(sep);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_LINELINE_HPP_
