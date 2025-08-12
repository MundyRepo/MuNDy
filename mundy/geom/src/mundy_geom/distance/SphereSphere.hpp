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

#ifndef MUNDY_GEOM_DISTANCE_SPHERESPHERE_HPP_
#define MUNDY_GEOM_DISTANCE_SPHERESPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::geom::FreeSpaceMetric
#include <mundy_geom/distance/PointPoint.hpp>       // for distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>            // for mundy::geom::SharedNormalSigned
#include <mundy_geom/primitives/Sphere.hpp>         // for mundy::geom::Sphere

namespace mundy {

namespace geom {

//! \name Free space distance calculations
//@{

/// \brief Compute the shared normal signed separation distance between two spheres
/// \tparam Scalar The scalar type
/// \param[in] sphere1 One sphere
/// \param[in] sphere2 The other sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Sphere<Scalar>& sphere1,  //
                                const Sphere<Scalar>& sphere2) {
  return distance(SharedNormalSigned{}, sphere1, sphere2);
}

/// \brief Compute the shared normal signed separation distance between two spheres
/// \tparam Scalar The scalar type
/// \param[in] sphere1 One sphere
/// \param[in] sphere2 The other sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                const Sphere<Scalar>& sphere1,                            //
                                const Sphere<Scalar>& sphere2) {
  return distance(sphere1.center(), sphere2.center()) - sphere1.radius() - sphere2.radius();
}

/// \brief Compute the distance between two spheres
/// \tparam Scalar The scalar type
/// \param[in] sphere1 One sphere
/// \param[in] sphere2 The other sphere
/// \param[out] sep The separation vector (from the surface of sphere1 to the surface of sphere2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Sphere<Scalar>& sphere1,  //
                                const Sphere<Scalar>& sphere2,  //
                                mundy::math::Vector3<Scalar>& sep) {
  const Scalar center_center_distance = distance(sphere1.center(), sphere2.center(), sep);

  // Rescale the separation vector to the surface of the sphere
  const Scalar surface_distance = center_center_distance - sphere1.radius() - sphere2.radius();
  sep *= surface_distance / center_center_distance;
  return surface_distance;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_SPHERESPHERE_HPP_
