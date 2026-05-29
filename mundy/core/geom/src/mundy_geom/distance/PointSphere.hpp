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

#ifndef MUNDY_GEOM_DISTANCE_POINTSPHERE_HPP_
#define MUNDY_GEOM_DISTANCE_POINTSPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/PointPoint.hpp>  // for distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>       // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/Point.hpp>     // for mundy::Point
#include <mundy_geom/primitives/Sphere.hpp>    // for mundy::Sphere
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
template <ValidPointType PointType, ValidSphereType SphereType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename SphereType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance(const PointType& point,  //
                                                      const SphereType& sphere) {
  return distance(SharedNormalSigned{}, point, sphere);
}

/// \brief Compute the shared normal signed separation distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
template <ValidPointType PointType, ValidSphereType SphereType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename SphereType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                      const PointType& point,                                   //
                                                      const SphereType& sphere) {
  return distance(point, sphere.center()) - sphere.radius();
}

/// \brief Compute the distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
/// \param[out] sep The separation vector (from point to sphere)
template <ValidPointType PointType, ValidSphereType SphereType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::scalar_t, typename SphereType::scalar_t>)
KOKKOS_FUNCTION typename PointType::scalar_t
    distance(const PointType& point,  //
             const SphereType& sphere, mundy::Vector3<typename PointType::scalar_t>& sep) {
  using Scalar = typename PointType::scalar_t;
  const Scalar center_point_distance = distance(point, sphere.center(), sep);

  // Rescale the separation vector to the surface of the sphere
  const Scalar surface_distance = center_point_distance - sphere.radius();
  sep *= surface_distance / center_point_distance;
  return surface_distance;
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTSPHERE_HPP_
