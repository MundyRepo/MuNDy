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

#ifndef MUNDY_GEOM_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
#define MUNDY_GEOM_DISTANCE_ELLIPSOIDELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/PointPoint.hpp>   // for mundy::distance(Point, Point)
#include <mundy_geom/distance/Types.hpp>        // for mundy::SharedNormalSigned
#include <mundy_geom/primitives/Ellipsoid.hpp>  // for mundy::Ellipsoid
#include <mundy_geom/primitives/Point.hpp>      // for mundy::Point
#include <mundy_math/Quaternion.hpp>            // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>             // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3
#include <mundy_math/minimize.hpp>              // for mundy::find_min_using_approximate_derivatives
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::scalar_t, typename EllipsoidType2::scalar_t>)
KOKKOS_FUNCTION typename EllipsoidType1::scalar_t distance(const EllipsoidType1& ellipsoid1,  //
                                                           const EllipsoidType2& ellipsoid2) {
  return distance(SharedNormalSigned{}, ellipsoid1, ellipsoid2);
}

template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::scalar_t, typename EllipsoidType2::scalar_t>)
KOKKOS_FUNCTION typename EllipsoidType1::scalar_t distance([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                                           const EllipsoidType1& ellipsoid1,                         //
                                                           const EllipsoidType2& ellipsoid2) {
  using Scalar = typename EllipsoidType1::scalar_t;
  Point<Scalar> closest_point1;
  Point<Scalar> closest_point2;
  mundy::Vector3<Scalar> shared_normal1;
  mundy::Vector3<Scalar> shared_normal2;
  return distance(distance_type, ellipsoid1, ellipsoid2,  //
                  closest_point1, closest_point2, shared_normal1, shared_normal2);
}

template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::scalar_t, typename EllipsoidType2::scalar_t>)
class EllipsoidEllipsoidObjective {
 public:
  using Scalar = typename EllipsoidType1::scalar_t;

  KOKKOS_FUNCTION
  EllipsoidEllipsoidObjective(const EllipsoidType1& ellipsoid0,        //
                              const EllipsoidType2& ellipsoid1,        //
                              mundy::Vector3<Scalar>& shared_normal0,  //
                              mundy::Vector3<Scalar>& shared_normal1,  //
                              Point<Scalar>& foot_point0,              //
                              Point<Scalar>& foot_point1)
      : ellipsoid0_(ellipsoid0),
        ellipsoid1_(ellipsoid1),
        shared_normal0_(shared_normal0),
        shared_normal1_(shared_normal1),
        foot_point0_(foot_point0),
        foot_point1_(foot_point1) {
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& theta_phi) const {
    // Map theta and phi to the lab frame normal vector
    const Scalar sin_theta = std::sin(theta_phi[0]);
    const Scalar cos_theta = std::cos(theta_phi[0]);
    const Scalar sin_phi = std::sin(theta_phi[1]);
    const Scalar cos_phi = std::cos(theta_phi[1]);
    shared_normal0_.set(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    shared_normal1_ = -shared_normal0_;

    // Map the shared normal to the lab frame foot points on the ellipsoids
    foot_point0_ = map_surface_normal_to_foot_point_on_ellipsoid(shared_normal0_, ellipsoid0_);
    foot_point1_ = map_surface_normal_to_foot_point_on_ellipsoid(shared_normal1_, ellipsoid1_);

    // The objective is the shared normal euclidean separation distance. NOT the signed separation distance.
    return distance(foot_point0_, foot_point1_);
  }

 private:
  const EllipsoidType1& ellipsoid0_;
  const EllipsoidType2& ellipsoid1_;
  mundy::Vector3<Scalar>& shared_normal0_;
  mundy::Vector3<Scalar>& shared_normal1_;
  Point<Scalar>& foot_point0_;
  Point<Scalar>& foot_point1_;
};

template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::scalar_t, typename EllipsoidType2::scalar_t>)
KOKKOS_FUNCTION typename EllipsoidType1::scalar_t
    distance([[maybe_unused]] const SharedNormalSigned distance_type,            //
             const EllipsoidType1& ellipsoid1,                                   //
             const EllipsoidType2& ellipsoid2,                                   //
             Point<typename EllipsoidType1::scalar_t>& closest_point1,           //
             Point<typename EllipsoidType1::scalar_t>& closest_point2,           //
             mundy::Vector3<typename EllipsoidType1::scalar_t>& shared_normal1,  //
             mundy::Vector3<typename EllipsoidType1::scalar_t>& shared_normal2) {
  using Scalar = typename EllipsoidType1::scalar_t;

  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  constexpr Scalar min_objective_delta = mundy::get_relaxed_zero_tolerance<Scalar>();
  constexpr size_t lbfgs_max_memory_size = 10;

  // Reuse the solution space rather than re-allocating it each time
  EllipsoidEllipsoidObjective<EllipsoidType1, EllipsoidType2> shared_normal_objective(
      ellipsoid1, ellipsoid2, shared_normal1, shared_normal2, closest_point1, closest_point2);

  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  constexpr Scalar zero = static_cast<Scalar>(0.0);
  constexpr Scalar half_pi = static_cast<Scalar>(0.5) * pi;
  constexpr Scalar one_third_pi = pi / static_cast<Scalar>(3.0);
  constexpr Scalar five_third_pi = static_cast<Scalar>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Scalar, 3> theta_guesses{zero, half_pi, pi};
  constexpr Kokkos::Array<Scalar, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Scalar global_dist = Kokkos::Experimental::infinity_v<Scalar>;
  mundy::Vector<Scalar, 2> theta_phi_sol{zero, zero};
  mundy::Vector<Scalar, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar dist = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective, theta_phi_sol, min_objective_delta);
      if (dist < global_dist) {
        global_dist = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // Evaluating the objective updates the shared normal and foot points
  shared_normal_objective(global_theta_phi_sol);
  return mundy::dot(closest_point2 - closest_point1, shared_normal1);
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
