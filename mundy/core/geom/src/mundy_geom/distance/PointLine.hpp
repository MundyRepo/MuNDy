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

#ifndef MUNDY_GEOM_DISTANCE_POINTLINE_HPP_
#define MUNDY_GEOM_DISTANCE_POINTLINE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/Types.hpp>    // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/Line.hpp>   // for mundy::Line
#include <mundy_geom/primitives/Point.hpp>  // for mundy::Point
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
template <ValidPointType PointType, ValidLineType LineType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance(const PointType& point,  //
                                                        const LineType& line) {
  return distance(SharedNormalSigned{}, point, line);
}

/// \brief Compute the shared normal signed separation distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
template <ValidPointType PointType, ValidLineType LineType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const PointType& point,                                   //
                                                        const LineType& line) {
  using Scalar = typename PointType::value_type;

  // Compute the projection of the vector onto the line's direction
  auto line_to_point = point - line.center();
  Scalar projection = mundy::dot(line_to_point, line.direction());

  // Compute the magnitude of the component of the vector perpendicular to the line
  return distance(projection * line.direction(), line_to_point);
}

/// \brief Compute the euclidean distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
/// \param[out] closest_point The closest point on the line
/// \param[out] arch_length The arch-length parameter of the closest point on the line
/// \param[out] sep The separation vector (from point to line)
template <ValidPointType PointType, ValidLineType LineType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance(const PointType& point,                                //
                                                        const LineType& line,                                  //
                                                        Point<typename PointType::value_type>& closest_point,  //
                                                        typename PointType::value_type& arch_length,           //
                                                        mundy::Vector3<typename PointType::value_type>& sep) {
  // No difference between distance types for points and lines
  return distance(SharedNormalSigned{}, point, line, closest_point, arch_length, sep);
}

/// \brief Compute the shared normal signed separation distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
/// \param[out] closest_point The closest point on the line
/// \param[out] arch_length The arch-length parameter of the closest point on the line
/// \param[out] sep The separation vector (from point to line)
template <ValidPointType PointType, ValidLineType LineType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename LineType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const PointType& point,                                   //
                                                        const LineType& line,                                     //
                                                        Point<typename PointType::value_type>& closest_point,     //
                                                        typename PointType::value_type& arch_length,              //
                                                        mundy::Vector3<typename PointType::value_type>& sep) {
  // Compute the projection of the vector onto the line's direction
  auto line_to_point = point - line.center();
  arch_length = mundy::dot(line_to_point, line.direction());
  closest_point = line.center() + arch_length * line.direction();

  // Compute the magnitude of the component of the vector perpendicular to the line
  return distance(arch_length * line.direction(), line_to_point);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTLINE_HPP_
