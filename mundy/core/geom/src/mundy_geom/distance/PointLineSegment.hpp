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
#include <mundy_geom/distance/PointPoint.hpp>     // for mundy::distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>          // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/LineSegment.hpp>  // for mundy::LineSegment
#include <mundy_geom/primitives/Point.hpp>        // for mundy::Point
#include <mundy_math/Tolerance.hpp>               // for mundy::get_zero_tolerance
#include <mundy_utils/requires.hpp>

namespace mundy {

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename LineSegmentType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance(const PointType& point,  //
                                                      const LineSegmentType& line_segment) {
  return distance(SharedNormalSigned{}, point, line_segment);
}

/// \brief Compute the shared normal signed separation distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename LineSegmentType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                      const PointType& point,                                   //
                                                      const LineSegmentType& line_segment) {
  using Scalar = typename PointType::scalar_t;
  const auto& p1 = line_segment.start();
  const auto& p2 = line_segment.end();
  const auto p21 = p2 - p1;

  // Define some temporary variables
  mundy::Vector3<Scalar> closest_point_tmp;
  Scalar t_tmp;

  // Get parametric location
  const Scalar num = mundy::dot(p21, point - p1);
  if ((num < mundy::get_zero_tolerance<Scalar>()) & (num > -mundy::get_zero_tolerance<Scalar>())) {
    // CASE 1: The vector from p1 to x is orthogonal to the line.
    // In this case, the closest point is p1 and the parametric coordinate is 0.
    closest_point_tmp = p1;
    t_tmp = static_cast<Scalar>(0.0);
  } else {
    const Scalar denom = dot(p21, p21);

    if (denom < mundy::get_zero_tolerance<Scalar>()) {
      // CASE 2: The line is degenerate (i.e., p1 and p2 are numerically the same point).
      // In this case, either point could really be the closest point. We'll arbitrarily pick p1 and set t to 0.
      closest_point_tmp = p1;
      t_tmp = static_cast<Scalar>(0.0);
    } else {
      // CASE 3: The line is well-defined and we can compute the closest point.
      t_tmp = num / denom;

      if (t_tmp < static_cast<Scalar>(0.0)) {
        // CASE 3.1: The parameter for the infinite line is less than 0. Therefore, the closest point is p1.
        closest_point_tmp = p1;
      } else if (t_tmp > static_cast<Scalar>(1.0)) {
        // CASE 3.2: The parameter for the infinite line is greater than 1. Therefore, the closest point is p2.
        closest_point_tmp = p2;
      } else {
        // CASE 3.3: The closest point is falls within the line segment.
        closest_point_tmp = p1 + t_tmp * p21;
      }
    }
  }

  return distance(point, closest_point_tmp);
}

/// \brief Compute the euclidean distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from point to line segment)
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename LineSegmentType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance(const PointType& point,                              //
                                                      const LineSegmentType& line_segment,                 //
                                                      Point<typename PointType::scalar_t>& closest_point,  //
                                                      typename PointType::scalar_t& arch_length,           //
                                                      mundy::Vector3<typename PointType::scalar_t>& sep) {
  return distance(SharedNormalSigned{}, point, line_segment,  //
                  closest_point, arch_length, sep);
}

/// \brief Compute the shared normal signed separation distance between a point and a line segment
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line_segment The line segment
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from point to line segment)
template <ValidPointType PointType, ValidLineSegmentType LineSegmentType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename LineSegmentType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                      const PointType& point,                                   //
                                                      const LineSegmentType& line_segment,                      //
                                                      Point<typename PointType::scalar_t>& closest_point,       //
                                                      typename PointType::scalar_t& arch_length,                //
                                                      mundy::Vector3<typename PointType::scalar_t>& sep) {
  using Scalar = typename PointType::scalar_t;
  const auto& p1 = line_segment.start();
  const auto& p2 = line_segment.end();
  const auto p21 = p2 - p1;

  // Get parametric location
  const Scalar num = dot(p21, point - p1);
  if ((num < mundy::get_zero_tolerance<Scalar>()) & (num > -mundy::get_zero_tolerance<Scalar>())) {
    // CASE 1: The vector from p1 to x is orthogonal to the line.
    // In this case, the closest point is p1 and the parametric coordinate is 0.
    closest_point = p1;
    arch_length = static_cast<Scalar>(0.0);
  } else {
    const Scalar denom = dot(p21, p21);

    if (denom < mundy::get_zero_tolerance<Scalar>()) {
      // CASE 2: The line is degenerate (i.e., p1 and p2 are numerically the same point).
      // In this case, either point could really be the closest point. We'll arbitrarily pick p1 and set t to 0.
      closest_point = p1;
      arch_length = static_cast<Scalar>(0.0);
    } else {
      // CASE 3: The line is well-defined and we can compute the closest point.
      arch_length = num / denom;

      if (arch_length < static_cast<Scalar>(0.0)) {
        // CASE 3.1: The parameter for the infinite line is less than 0. Therefore, the closest point is p1.
        closest_point = p1;
      } else if (arch_length > static_cast<Scalar>(1.0)) {
        // CASE 3.2: The parameter for the infinite line is greater than 1. Therefore, the closest point is p2.
        closest_point = p2;
      } else {
        // CASE 3.3: The closest point is falls within the line segment.
        closest_point = p1 + arch_length * p21;
      }
    }
  }

  return distance(point, closest_point, sep);
}
//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTLINESEGMENT_HPP_
