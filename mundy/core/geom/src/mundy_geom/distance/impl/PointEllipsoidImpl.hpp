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

#ifndef MUNDY_GEOM_DISTANCE_IMPL_POINTELLIPSOIDIMPL_HPP_
#define MUNDY_GEOM_DISTANCE_IMPL_POINTELLIPSOIDIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/PointPoint.hpp>   // for mundy::distance(Point, Point)
#include <mundy_geom/primitives/Ellipsoid.hpp>  // for mundy::Ellipsoid, map_surface_normal_to_foot_point_on_ellipsoid
#include <mundy_geom/primitives/Point.hpp>      // for mundy::Point
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3
#include <mundy_math/cmath.hpp>                 // for mundy::sin, mundy::cos
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace impl {

/// \brief Cost-only functor for point–ellipsoid distance: the shared-normal Euclidean separation
/// between the point and the ellipsoid foot point parameterised by the surface-normal angles.
/// \internal
template <ValidPointType PointType, ValidEllipsoidType EllipsoidType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename EllipsoidType::value_type>)
class PointEllipsoidObjective {
 public:
  using Scalar = typename PointType::value_type;

  KOKKOS_FUNCTION
  PointEllipsoidObjective(const PointType& point,                 //
                          const EllipsoidType& ellipsoid,         //
                          mundy::Vector3<Scalar>& shared_normal,  //
                          Point<Scalar>& foot_point)
      : point_(point), ellipsoid_(ellipsoid), shared_normal_(shared_normal), foot_point_(foot_point) {
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& theta_phi) const {
    // Map theta and phi to the lab frame normal vector
    const Scalar sin_theta = sin(theta_phi[0]);
    const Scalar cos_theta = cos(theta_phi[0]);
    const Scalar sin_phi = sin(theta_phi[1]);
    const Scalar cos_phi = cos(theta_phi[1]);
    shared_normal_.set(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    // Map the normal vector to the foot point on the ellipsoid
    foot_point_ = map_surface_normal_to_foot_point_on_ellipsoid(shared_normal_, ellipsoid_);

    // The objective is the shared normal euclidean separation distance. NOT the signed separation distance.
    return distance(foot_point_, point_);
  }

 private:
  const PointType& point_;
  const EllipsoidType& ellipsoid_;
  mundy::Vector3<Scalar>& shared_normal_;
  Point<Scalar>& foot_point_;
};

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_IMPL_POINTELLIPSOIDIMPL_HPP_
