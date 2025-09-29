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

// Mundy
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::geom::FreeSpaceMetric
#include <mundy_geom/distance/Types.hpp>            // for mundy::geom::SharedNormalSigned
#include <mundy_geom/primitives/Line.hpp>           // for mundy::geom::Line
#include <mundy_geom/primitives/Point.hpp>          // for mundy::geom::Point

namespace mundy {

namespace geom {

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point,  //
                                const Line<Scalar>& line) {
  return distance(SharedNormalSigned{}, point, line);
}

/// \brief Compute the shared normal signed separation distance between a point and a line
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] line The line
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                const Point<Scalar>& point,                               //
                                const Line<Scalar>& line) {
  // Compute the projection of the vector onto the line's direction
  auto line_to_point = point - line.center();
  Scalar projection = mundy::math::dot(line_to_point, line.direction());

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
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point,    //
                                const Line<Scalar>& line,      //
                                Point<Scalar>& closest_point,  //
                                Scalar& arch_length,           //
                                mundy::math::Vector3<Scalar>& sep) {
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
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                const Point<Scalar>& point,                               //
                                const Line<Scalar>& line,                                 //
                                Point<Scalar>& closest_point,                             //
                                Scalar& arch_length,                                      //
                                mundy::math::Vector3<Scalar>& sep) {
  // Compute the projection of the vector onto the line's direction
  auto line_to_point = point - line.center();
  arch_length = mundy::math::dot(line_to_point, line.direction());
  closest_point = line.center() + arch_length * line.direction();

  // Compute the magnitude of the component of the vector perpendicular to the line
  return distance(arch_length * line.direction(), line_to_point);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTLINE_HPP_
