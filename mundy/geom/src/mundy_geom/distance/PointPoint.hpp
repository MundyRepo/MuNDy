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
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::geom::FreeSpaceMetric
#include <mundy_geom/distance/Types.hpp>            // for mundy::geom::SharedNormalSigned, Euclidean
#include <mundy_geom/primitives/Point.hpp>          // for mundy::geom::Point

namespace mundy {

namespace geom {

//! \name Periodic space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                    const Point<Scalar>& point1,                              //
                                    const Point<Scalar>& point2, const Metric& metric) {
  return mundy::math::norm(metric(point1, point2));
}

/// \brief Compute the shared normal signed separation distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                    const Point<Scalar>& point1,                              //
                                    const Point<Scalar>& point2,                              //
                                    const Metric& metric, mundy::math::Vector3<Scalar>& sep) {
  sep = metric(point1, point2);
  return mundy::math::norm(sep);
}

/// \brief Compute the euclidean distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const Euclidean distance_type,  //
                                    const Point<Scalar>& point1,                     //
                                    const Point<Scalar>& point2,                     //
                                    const Metric& metric) {
  return distance_pbc(SharedNormalSigned{}, point1, point2, metric);
}

/// \brief Compute the euclidean distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const Euclidean distance_type,  //
                                    const Point<Scalar>& point1,                     //
                                    const Point<Scalar>& point2,                     //
                                    const Metric& metric,                            //
                                    mundy::math::Vector3<Scalar>& sep) {
  return distance_pbc(SharedNormalSigned{}, point1, point2, metric, sep);
}

/// \brief Compute the shared normal signed separation distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc(const Point<Scalar>& point1,  //
                                    const Point<Scalar>& point2,  //
                                    const Metric& metric) {
  return distance_pbc(SharedNormalSigned{}, point1, point2, metric);
}

/// \brief Compute the euclidean distance between two points according to a metric
/// \tparam Scalar The scalar type
/// \tparam Metric The metric type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[in] metric The metric to use for the distance calculation
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc(const Point<Scalar>& point1,  //
                                    const Point<Scalar>& point2,  //
                                    const Metric& metric,         //
                                    mundy::math::Vector3<Scalar>& sep) {
  return distance_pbc(SharedNormalSigned{}, point1, point2, metric, sep);
}
//@}

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1,  //
                                const Point<Scalar>& point2) {
  return distance_pbc(point1, point2, FreeSpaceMetric<Scalar>{});
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar, typename DistanceType>
KOKKOS_FUNCTION Scalar distance(const DistanceType distance_type,  //
                                const Point<Scalar>& point1,       //
                                const Point<Scalar>& point2) {
  return distance_pbc(distance_type, point1, point2, FreeSpaceMetric<Scalar>{});
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1,  //
                                const Point<Scalar>& point2,  //
                                mundy::math::Vector3<Scalar>& sep) {
  return distance_pbc(point1, point2, FreeSpaceMetric<Scalar>{}, sep);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar, typename DistanceType>
KOKKOS_FUNCTION Scalar distance(const DistanceType distance_type,  //
                                const Point<Scalar>& point1,       //
                                const Point<Scalar>& point2,       //
                                mundy::math::Vector3<Scalar>& sep) {
  return distance_pbc(distance_type, point1, point2, FreeSpaceMetric<Scalar>{}, sep);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTPOINT_HPP_
