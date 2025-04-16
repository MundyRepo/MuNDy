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

// Mundy
#include <mundy_geom/distance/Types.hpp>    // for mundy::geom::SharedNormalSigned, Euclidean
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point

namespace mundy {

namespace geom {

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1, const Point<Scalar>& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2) {
  Scalar dx = point2[0] - point1[0];
  Scalar dy = point2[1] - point1[1];
  Scalar dz = point2[2] - point1[2];
  return Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1, const Point<Scalar>& point2,
                                mundy::math::Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2, mundy::math::Vector3<Scalar>& sep) {
  sep = point2 - point1;
  return mundy::math::norm(sep);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2, mundy::math::Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTPOINT_HPP_
