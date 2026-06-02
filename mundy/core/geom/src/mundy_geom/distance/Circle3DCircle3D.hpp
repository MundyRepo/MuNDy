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

#ifndef MUNDY_GEOM_DISTANCE_CIRCLE3DCIRCLE3D_HPP_
#define MUNDY_GEOM_DISTANCE_CIRCLE3DCIRCLE3D_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy math
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3
#include <mundy_math/minimize.hpp>    // for mundy::find_min_using_approximate_derivatives

// Mundy geom
#include <mundy_geom/distance/PointPoint.hpp>  // for mundy::distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>       // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/Circle3D.hpp>  // for mundy::Circle3D
#include <mundy_geom/primitives/Point.hpp>     // for mundy::Point
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DType1::value_type, typename Circle3DType2::value_type>)
KOKKOS_FUNCTION typename Circle3DType1::value_type distance(const Circle3DType1& circle3d1,  //
                                                          const Circle3DType2& circle3d2) {
  return distance(Euclidean{}, circle3d1, circle3d2);
}

template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DType1::value_type, typename Circle3DType2::value_type>)
KOKKOS_FUNCTION typename Circle3DType1::value_type distance([[maybe_unused]] const Euclidean distance_type,  //
                                                          const Circle3DType1& circle3d1,                  //
                                                          const Circle3DType2& circle3d2) {
  using Scalar = typename Circle3DType1::value_type;
  Point<Scalar> closest_point1;
  Point<Scalar> closest_point2;
  mundy::Vector3<Scalar> shared_normal1;
  mundy::Vector3<Scalar> shared_normal2;
  return distance(distance_type, circle3d1, circle3d2,  //
                  closest_point1, closest_point2, shared_normal1, shared_normal2);
}

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
    Point<Scalar> p_local{circle3d.radius() * Kokkos::cos(theta), circle3d.radius() * Kokkos::sin(theta), 0.0};
    auto p_global = circle3d.orientation() * p_local + circle3d.center();
    return p_global;
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& theta1_theta2) const {
    foot_point0_ = theta_to_foot_point_on_circle3d(theta1_theta2[0], circle3d0_);
    foot_point1_ = theta_to_foot_point_on_circle3d(theta1_theta2[1], circle3d1_);

    shared_normal0_ = foot_point1_ - foot_point0_;

    const double norm = mundy::norm(shared_normal0_);
    shared_normal0_ /= (norm > mundy::get_zero_tolerance<Scalar>() ? norm : 1.0);
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

template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
MUNDY_REQUIRES(std::is_same_v<typename Circle3DType1::value_type, typename Circle3DType2::value_type>)
KOKKOS_FUNCTION typename Circle3DType1::value_type
    distance([[maybe_unused]] const Euclidean distance_type,                    //
             const Circle3DType1& circle3d1,                                    //
             const Circle3DType2& circle3d2,                                    //
             Point<typename Circle3DType1::value_type>& closest_point1,           //
             Point<typename Circle3DType1::value_type>& closest_point2,           //
             mundy::Vector3<typename Circle3DType1::value_type>& shared_normal1,  //
             mundy::Vector3<typename Circle3DType1::value_type>& shared_normal2) {
  using Scalar = typename Circle3DType1::value_type;

  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  constexpr Scalar min_objective_delta = mundy::get_relaxed_zero_tolerance<Scalar>();
  constexpr size_t lbfgs_max_memory_size = 10;

  // Reuse the solution space rather than re-allocating it each time
  Circle3DCircle3DObjective<Circle3DType1, Circle3DType2> minimize_euclidean_distance(
      circle3d1, circle3d2, shared_normal1, shared_normal2, closest_point1, closest_point2);

  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  constexpr Scalar zero = static_cast<Scalar>(0.0);
  constexpr Scalar one_third_pi = pi / static_cast<Scalar>(3.0);
  constexpr Scalar five_third_pi = static_cast<Scalar>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Scalar, 3> theta_guesses{one_third_pi, pi, five_third_pi};

  Scalar global_dist = Kokkos::Experimental::infinity_v<Scalar>;
  mundy::Vector<Scalar, 2> theta1_theta2_sol{zero, zero};
  mundy::Vector<Scalar, 2> global_theta1_theta2_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta1_theta2_sol = {theta_guesses[t_idx], theta_guesses[p_idx]};
      const Scalar dist = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          minimize_euclidean_distance, theta1_theta2_sol, min_objective_delta);
      if (dist < global_dist) {
        global_dist = dist;
        global_theta1_theta2_sol = theta1_theta2_sol;
      }
    }
  }

  // Evaluating the objective updates the shared normal and foot points
  minimize_euclidean_distance(global_theta1_theta2_sol);
  return global_dist;
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_CIRCLE3DCIRCLE3D_HPP_
