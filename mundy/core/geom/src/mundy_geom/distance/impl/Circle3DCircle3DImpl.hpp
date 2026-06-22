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

#ifndef MUNDY_GEOM_DISTANCE_IMPL_CIRCLE3DCIRCLE3DIMPL_HPP_
#define MUNDY_GEOM_DISTANCE_IMPL_CIRCLE3DCIRCLE3DIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/primitives/Circle3D.hpp>  // for mundy::Circle3D
#include <mundy_geom/primitives/Point.hpp>     // for mundy::Point
#include <mundy_math/Quaternion.hpp>           // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>            // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>              // for mundy::Vector3, mundy::norm
#include <mundy_math/cmath.hpp>                // for mundy::sin, mundy::cos
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace impl {

/// \brief Cost-only functor for circle–circle distance: the Euclidean separation between the foot
/// points parameterised by the angular position on each circle.
/// \internal
template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DType1::value_type, typename Circle3DType2::value_type>)
class Circle3DCircle3DObjective {
 public:
  using Scalar = typename Circle3DType1::value_type;

  KOKKOS_FUNCTION
  Circle3DCircle3DObjective(const Circle3DType1& circle3d0,          //
                            const Circle3DType2& circle3d1,          //
                            mundy::Vector3<Scalar>& shared_normal0,  //
                            mundy::Vector3<Scalar>& shared_normal1,  //
                            Point<Scalar>& foot_point0,              //
                            Point<Scalar>& foot_point1)
      : circle3d0_(circle3d0),
        circle3d1_(circle3d1),
        shared_normal0_(shared_normal0),
        shared_normal1_(shared_normal1),
        foot_point0_(foot_point0),
        foot_point1_(foot_point1) {
  }

  template <ValidCircle3DType Circle3DType>
  KOKKOS_INLINE_FUNCTION Point<Scalar> theta_to_foot_point_on_circle3d(const Scalar theta,
                                                                       const Circle3DType& circle3d) const {
    Point<Scalar> p_local{circle3d.radius() * cos(theta), circle3d.radius() * sin(theta), 0.0};
    auto p_global = circle3d.orientation() * p_local + circle3d.center();
    return p_global;
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& theta1_theta2) const {
    foot_point0_ = theta_to_foot_point_on_circle3d(theta1_theta2[0], circle3d0_);
    foot_point1_ = theta_to_foot_point_on_circle3d(theta1_theta2[1], circle3d1_);

    shared_normal0_ = foot_point1_ - foot_point0_;

    const Scalar norm = mundy::norm(shared_normal0_);
    shared_normal0_ /= (norm > mundy::get_zero_tolerance<Scalar>() ? norm : static_cast<Scalar>(1.0));
    shared_normal1_ = -shared_normal0_;

    return norm;
  }

 private:
  const Circle3DType1& circle3d0_;
  const Circle3DType2& circle3d1_;
  mundy::Vector3<Scalar>& shared_normal0_;
  mundy::Vector3<Scalar>& shared_normal1_;
  Point<Scalar>& foot_point0_;
  Point<Scalar>& foot_point1_;
};

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_IMPL_CIRCLE3DCIRCLE3DIMPL_HPP_
