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

#ifndef MUNDY_GEOM_DISTANCE_POINTELLIPSOID_HPP_
#define MUNDY_GEOM_DISTANCE_POINTELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/PointPoint.hpp>               // for mundy::distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>                    // for mundy::SharedNormalSigned
#include <mundy_geom/distance/impl/PointEllipsoidImpl.hpp>  // for the cost-only objective functor
#include <mundy_geom/primitives/Ellipsoid.hpp>              // for mundy::Ellipsoid
#include <mundy_geom/primitives/Point.hpp>                  // for mundy::Point
#include <mundy_math/Quaternion.hpp>                        // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>                         // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>                           // for mundy::Vector3
#include <mundy_math/cmath.hpp>
#include <mundy_math/minimize.hpp>  // for mundy::find_min_using_approximate_derivatives
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

template <ValidPointType PointType, ValidEllipsoidType EllipsoidType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename EllipsoidType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance(const PointType& point,  //
                                                        const EllipsoidType& ellipsoid) {
  return distance(SharedNormalSigned{}, point, ellipsoid);
}

template <ValidPointType PointType, ValidEllipsoidType EllipsoidType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename EllipsoidType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                        const PointType& point,                                   //
                                                        const EllipsoidType& ellipsoid) {
  using Scalar = typename PointType::value_type;
  Point<Scalar> closest_point;
  mundy::Vector3<Scalar> ellipsoid_normal;
  return distance(distance_type, point, ellipsoid,  //
                  closest_point, ellipsoid_normal);
}

template <ValidPointType PointType, ValidEllipsoidType EllipsoidType>
MUNDY_REQUIRES(std::is_same_v<typename PointType::value_type, typename EllipsoidType::value_type>)
KOKKOS_FUNCTION typename PointType::value_type
    distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
             const PointType& point,                                   //
             const EllipsoidType& ellipsoid,                           //
             Point<typename PointType::value_type>& closest_point,     //
             mundy::Vector3<typename PointType::value_type>& ellipsoid_normal) {
  using Scalar = typename PointType::value_type;
  using Passive = passive_scalar_t<Scalar>;  // double for an AD scalar; Scalar itself otherwise
  using PassiveEllipsoid = Ellipsoid<Passive>;

  // Envelope theorem: solve for the optimal angles theta* in the passive type (the L-BFGS minimiser is
  // double-only), then re-evaluate the closest point on the original — possibly AD — ellipsoid with
  // theta* frozen. Since d(objective)/d(theta) = 0 at the optimum, the returned distance carries the
  // exact derivative w.r.t. the point/ellipsoid parameters without differentiating through the solve.
  //
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of
  // approximate derivatives. The error is typically less than the square root of min_objective_delta.
  constexpr Passive min_allowable_cost = -Kokkos::Experimental::infinity_v<Passive>;     // no early-exit on cost
  constexpr Passive min_objective_delta = mundy::get_relaxed_zero_tolerance<Passive>();  // L-BFGS convergence tolerance
  constexpr size_t lbfgs_max_memory_size = 10;

  const Point<Passive> point_p(impl::passive_value(point[0]), impl::passive_value(point[1]),
                               impl::passive_value(point[2]));
  const PassiveEllipsoid ellipsoid_p = impl::passive_copy(ellipsoid);
  Point<Passive> foot_p;
  mundy::Vector3<Passive> normal_p;
  impl::PointEllipsoidObjective<Point<Passive>, PassiveEllipsoid> objective_p(point_p, ellipsoid_p, normal_p, foot_p);

  constexpr Passive pi = Kokkos::numbers::pi_v<Passive>;
  constexpr Passive zero = static_cast<Passive>(0.0);
  constexpr Passive half_pi = static_cast<Passive>(0.5) * pi;
  constexpr Passive one_third_pi = pi / static_cast<Passive>(3.0);
  constexpr Passive five_third_pi = static_cast<Passive>(5.0) * one_third_pi;
  constexpr mundy::Vector<Passive, 3> theta_guesses{zero, half_pi, pi};
  constexpr mundy::Vector<Passive, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Passive global_dist = Kokkos::Experimental::infinity_v<Passive>;
  mundy::Vector<Passive, 2> theta_phi_sol{zero, zero};
  mundy::Vector<Passive, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Passive dist = mundy::find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          objective_p, theta_phi_sol, min_allowable_cost, min_objective_delta);
      if (dist < global_dist) {
        global_dist = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // Re-evaluate on the original point/ellipsoid with theta* frozen as a zero-derivative constant.
  impl::PointEllipsoidObjective<PointType, EllipsoidType> objective_ad(point, ellipsoid, ellipsoid_normal,
                                                                       closest_point);
  const mundy::Vector<Scalar, 2> theta_star{static_cast<Scalar>(global_theta_phi_sol[0]),
                                            static_cast<Scalar>(global_theta_phi_sol[1])};
  objective_ad(theta_star);
  return mundy::dot(point - closest_point, ellipsoid_normal);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTELLIPSOID_HPP_
