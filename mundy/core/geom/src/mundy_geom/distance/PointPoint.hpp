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

#ifndef MUNDY_GEOM_DISTANCE_POINTPOINT_HPP_
#define MUNDY_GEOM_DISTANCE_POINTPOINT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::FreeSpaceMetric
#include <mundy_geom/distance/Types.hpp>            // for mundy::SharedNormalSigned, Euclidean
#include <mundy_geom/primitives/Point.hpp>          // for mundy::Point

namespace mundy {

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                       const PointType1& point1,                                //
                                                       const PointType2& point2) {
  return mundy::norm(point2 - point1);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                       const PointType1& point1,                                //
                                                       const PointType2& point2,                                //
                                                       mundy::Vector3<typename PointType1::scalar_t>& sep) {
  sep = point2 - point1;
  return mundy::norm(sep);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance([[maybe_unused]] const Euclidean distance_type,  //
                                                       const PointType1& point1,                       //
                                                       const PointType2& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance([[maybe_unused]] const Euclidean distance_type,  //
                                                       const PointType1& point1,                       //
                                                       const PointType2& point2,                       //
                                                       mundy::Vector3<typename PointType1::scalar_t>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance(const PointType1& point1,  //
                                                       const PointType2& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <ValidPointType PointType1, ValidPointType PointType2>
  requires std::is_same_v<typename PointType1::scalar_t, typename PointType2::scalar_t>
KOKKOS_FUNCTION typename PointType1::scalar_t distance(const PointType1& point1,  //
                                                       const PointType2& point2,  //
                                                       mundy::Vector3<typename PointType1::scalar_t>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}
//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTPOINT_HPP_
