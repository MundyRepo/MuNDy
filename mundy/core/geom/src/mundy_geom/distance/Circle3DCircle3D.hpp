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

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3
#include <mundy_math/minimize.hpp>    // for mundy::find_min_using_approximate_derivatives
#include <mundy_geom/distance/PointPoint.hpp>  // for mundy::distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>       // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/Circle3D.hpp>  // for mundy::Circle3D
#include <mundy_geom/primitives/Point.hpp>     // for mundy::Point
#include <mundy_utils/requires.hpp>
#include <mundy_math/cmath.hpp>
#include <mundy_geom/distance/impl/Circle3DCircle3DImpl.hpp>  // for the cost-only objective functor

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
KOKKOS_FUNCTION typename Circle3DType1::value_type
    distance([[maybe_unused]] const Euclidean distance_type,                    //
             const Circle3DType1& circle3d1,                                    //
             const Circle3DType2& circle3d2,                                    //
             Point<typename Circle3DType1::value_type>& closest_point1,           //
             Point<typename Circle3DType1::value_type>& closest_point2,           //
             mundy::Vector3<typename Circle3DType1::value_type>& shared_normal1,  //
             mundy::Vector3<typename Circle3DType1::value_type>& shared_normal2) {
  using Scalar = typename Circle3DType1::value_type;
  using Passive = passive_scalar_t<Scalar>;  // double for an AD scalar; Scalar itself otherwise
  using PassiveCircle3D = Circle3D<Passive>;

  // Envelope theorem: solve for the optimal angles theta* in the passive type (the L-BFGS minimiser is
  // double-only), then re-evaluate the closest points on the original — possibly AD — circles with
  // theta* frozen. Since d(objective)/d(theta) = 0 at the optimum, the returned distance carries the
  // exact derivative w.r.t. the circle parameters without differentiating through the solve.
  //
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of
  // approximate derivatives. The error is typically less than the square root of min_objective_delta.
  constexpr Passive min_allowable_cost  = -Kokkos::Experimental::infinity_v<Passive>;    // no early-exit on cost
  constexpr Passive min_objective_delta = mundy::get_relaxed_zero_tolerance<Passive>();  // L-BFGS convergence tolerance
  constexpr size_t lbfgs_max_memory_size = 10;

  const PassiveCircle3D c0p = impl::passive_copy(circle3d1);
  const PassiveCircle3D c1p = impl::passive_copy(circle3d2);
  Point<Passive> cp0p, cp1p;
  mundy::Vector3<Passive> sn0p, sn1p;
  impl::Circle3DCircle3DObjective<PassiveCircle3D, PassiveCircle3D> objective_p(c0p, c1p, sn0p, sn1p, cp0p, cp1p);

  constexpr Passive pi = Kokkos::numbers::pi_v<Passive>;
  constexpr Passive zero = static_cast<Passive>(0.0);
  constexpr Passive one_third_pi = pi / static_cast<Passive>(3.0);
  constexpr Passive five_third_pi = static_cast<Passive>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Passive, 3> theta_guesses{one_third_pi, pi, five_third_pi};

  Passive global_dist = Kokkos::Experimental::infinity_v<Passive>;
  mundy::Vector<Passive, 2> theta1_theta2_sol{zero, zero};
  mundy::Vector<Passive, 2> global_theta1_theta2_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta1_theta2_sol = {theta_guesses[t_idx], theta_guesses[p_idx]};
      const Passive dist = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          objective_p, theta1_theta2_sol, min_allowable_cost, min_objective_delta);
      if (dist < global_dist) {
        global_dist = dist;
        global_theta1_theta2_sol = theta1_theta2_sol;
      }
    }
  }

  // Re-evaluate on the original circles with theta* frozen as a zero-derivative constant; the objective
  // fills the closest points / shared normals and returns the (AD) separation distance.
  impl::Circle3DCircle3DObjective<Circle3DType1, Circle3DType2> objective_ad(
      circle3d1, circle3d2, shared_normal1, shared_normal2, closest_point1, closest_point2);
  const mundy::Vector<Scalar, 2> theta_star{static_cast<Scalar>(global_theta1_theta2_sol[0]),
                                            static_cast<Scalar>(global_theta1_theta2_sol[1])};
  return objective_ad(theta_star);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_CIRCLE3DCIRCLE3D_HPP_
