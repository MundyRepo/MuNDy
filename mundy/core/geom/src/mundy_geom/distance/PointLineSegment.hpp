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

#ifndef MUNDY_GEOM_DISTANCE_POINTLINESEGMENT_HPP_
#define MUNDY_GEOM_DISTANCE_POINTLINESEGMENT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/Types.hpp>                            // for mundy::SharedNormalSigned
#include <mundy_geom/distance/impl/DistanceScalingImpl.hpp>         // for mundy::impl::{centred_scale_exponent, scaled_norm}
#include <mundy_geom/primitives/LineSegment.hpp>                    // for mundy::LineSegment
#include <mundy_geom/primitives/Point.hpp>                          // for mundy::Point
#include <mundy_math/Vector3.hpp>                                   // for mundy::Vector3, mundy::dot
#include <mundy_math/cmath.hpp>                                     // for mundy::passive_scalar_t
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineSegmentType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance(const PointType& point,  //
                                                        const LineSegmentType& line_segment) {
  return distance(SharedNormalSigned{}, point, line_segment);
}

/// \brief Compute the shared normal signed separation distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineSegmentType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const PointType& point,                                   //
                                                        const LineSegmentType& line_segment) {
  using Scalar = typename PointType::value_type;
  Point<Scalar> closest_point;
  Scalar arch_length;
  mundy::Vector3<Scalar> sep;
  return distance(distance_type, point, line_segment, closest_point, arch_length, sep);
}

/// \brief Compute the euclidean distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from point to line segment)
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineSegmentType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance(const PointType& point,                                //
                                                        const LineSegmentType& line_segment,                   //
                                                        Point<typename PointType::value_type>& closest_point,  //
                                                        typename PointType::value_type& arch_length,           //
                                                        mundy::Vector3<typename PointType::value_type>& sep) {
  return distance(SharedNormalSigned{}, point, line_segment,  //
                  closest_point, arch_length, sep);
}

/// \brief Compute the shared normal signed separation distance between a point and a line segment
///
/// The closest point minimizes |closest_point - point| over the segment. It equals start + arch_length (end - start)
/// with arch_length in [0, 1], sep = closest_point - point, and the return value is |sep|. A segment of zero length
/// gives arch_length = 0.
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from point to line segment)
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineSegmentType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const PointType& point,                                   //
                                                        const LineSegmentType& line_segment,                      //
                                                        Point<typename PointType::value_type>& closest_point,     //
                                                        typename PointType::value_type& arch_length,              //
                                                        mundy::Vector3<typename PointType::value_type>& sep) {
  using Scalar = typename PointType::value_type;
  using T = passive_scalar_t<Scalar>;
  const auto& p1 = line_segment.start();
  const auto& p2 = line_segment.end();

  // Quarter the points when a difference of them would overflow.
  auto p21 = p2 - p1;
  auto p01 = point - p1;
  const bool prescaled = !(impl::max_abs_component<T>(p21, p01, p01) < mundy_infinity_v<T>);
  const T q = prescaled ? static_cast<T>(0.25) : static_cast<T>(1.0);
  if (prescaled) {
    p21 = p2 * q - p1 * q;
    p01 = point * q - p1 * q;
  }

  // The parametric location p21 . p01 / p21 . p21 is formed on p21 and p01 scaled alike by a power of two, to which it
  // is invariant, so neither product overflows or underflows.
  const T m21 = impl::max_abs_component<T>(p21, p21, p21);
  const T m01 = impl::max_abs_component<T>(p01, p01, p01);
  if (m21 > static_cast<T>(0.0)) {
    // CASE 1: The segment has length. Project the point onto it and clamp to the segment.
    const int k = impl::centred_scale_exponent(m21, static_cast<T>(0.0), m01);
    const auto p21s = impl::scale_by_pow2<T>(p21, k);
    const auto p01s = impl::scale_by_pow2<T>(p01, k);
    const Scalar num = mundy::dot(p21s, p01s);
    const Scalar denom = mundy::dot(p21s, p21s);
    arch_length = (denom > static_cast<T>(0.0)) ? impl::clamp_to_unit_interval(num / denom) : static_cast<Scalar>(0.0);
  } else {
    // CASE 2: The segment is a point.
    arch_length = static_cast<Scalar>(0.0);
  }

  // Finally, get the closest point and the separation, undoing the quartering.
  // TODO(palmerb4): A parameter clamped to 1 gives p1 + p21, which can differ from p2 by an ulp. Decide whether callers
  // need the endpoint exactly there, and if so return p2 when the parameter is 1.
  if (!prescaled) {
    closest_point = p1 + arch_length * p21;
    sep = closest_point - point;
  } else {
    const T inv_q = static_cast<T>(4.0);
    const auto cq = p1 * q + arch_length * p21;
    closest_point = cq * inv_q;
    sep = (cq - point * q) * inv_q;
  }
  return impl::scaled_norm(sep);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTLINESEGMENT_HPP_
