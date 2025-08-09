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

// Mundy
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::geom::FreeSpaceMetric
#include <mundy_geom/distance/Types.hpp>            // for mundy::geom::SharedNormalSigned
#include <mundy_geom/primitives/Ellipsoid.hpp>      // for mundy::geom::Ellipsoid
#include <mundy_geom/primitives/Point.hpp>          // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>                // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>                 // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>                   // for mundy::math::Vector3
#include <mundy_math/minimize.hpp>                  // for mundy::math::find_min_using_approximate_derivatives

namespace mundy {

namespace geom {

//! \name Periodic space distance calculations
//@{

template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc(const Point<Scalar>& point,  //
                                    const Ellipsoid<Scalar>& ellipsoid, //
                                    const Metric& metric) {
  return distance_pbc(SharedNormalSigned{}, point, ellipsoid, metric);
}

template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                    const Point<Scalar>& point,                               //
                                    const Ellipsoid<Scalar>& ellipsoid,                       //
                                    const Metric& metric) {
  Point<Scalar> closest_point;
  mundy::math::Vector3<Scalar> ellipsoid_normal;
  return distance_pbc(distance_type, point, ellipsoid, metric,  //
                      closest_point, ellipsoid_normal);
}

template <typename Scalar, typename Metric>
class PointEllipsoidObjective {
 public:
  KOKKOS_FUNCTION
  PointEllipsoidObjective(const Point<Scalar>& point,                   //
                          const Ellipsoid<Scalar>& ellipsoid,           //
                          const Metric& metric,                         //
                          mundy::math::Vector3<Scalar>& shared_normal,  //
                          Point<Scalar>& foot_point)
      : point_(point), ellipsoid_(ellipsoid), metric_(metric), shared_normal_(shared_normal), foot_point_(foot_point) {
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::math::Vector<Scalar, 2>& theta_phi) const {
    // Map theta and phi to the lab frame normal vector
    const Scalar sin_theta = std::sin(theta_phi[0]);
    const Scalar cos_theta = std::cos(theta_phi[0]);
    const Scalar sin_phi = std::sin(theta_phi[1]);
    const Scalar cos_phi = std::cos(theta_phi[1]);
    shared_normal_.set(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    // Map the normal vector to the foot point on the ellipsoid
    foot_point_ = map_surface_normal_to_foot_point_on_ellipsoid(shared_normal_, ellipsoid_);

    // The objective is the shared normal euclidean separation distance. NOT the signed separation distance.
    return distance_pbc(foot_point_, point_, metric_);
  }

 private:
  const Point<Scalar>& point_;
  const Ellipsoid<Scalar>& ellipsoid_;
  const Metric& metric_;
  mundy::math::Vector3<Scalar>& shared_normal_;
  Point<Scalar>& foot_point_;
};

template <typename Scalar, typename Metric>
KOKKOS_FUNCTION Scalar distance_pbc([[maybe_unused]] const SharedNormalSigned distance_type,  //
                                    const Point<Scalar>& point,                               //
                                    const Ellipsoid<Scalar>& ellipsoid,                       //
                                    const Metric& metric,                                     //
                                    Point<Scalar>& closest_point,                             //
                                    mundy::math::Vector3<Scalar>& ellipsoid_normal) {
  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  constexpr Scalar min_objective_delta = mundy::math::get_relaxed_zero_tolerance<Scalar>();
  constexpr size_t lbfgs_max_memory_size = 10;

  // Reuse the solution space rather than re-allocating it each time
  PointEllipsoidObjective shared_normal_objective(point, ellipsoid, metric, ellipsoid_normal, closest_point);

  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  constexpr Scalar zero = static_cast<Scalar>(0.0);
  constexpr Scalar half_pi = static_cast<Scalar>(0.5) * pi;
  constexpr Scalar one_third_pi = pi / static_cast<Scalar>(3.0);
  constexpr Scalar five_third_pi = static_cast<Scalar>(5.0) * one_third_pi;
  constexpr mundy::math::Vector<Scalar, 3> theta_guesses{zero, half_pi, pi};
  constexpr mundy::math::Vector<Scalar, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Scalar global_dist = Kokkos::Experimental::infinity_v<Scalar>;
  mundy::math::Vector<Scalar, 2> theta_phi_sol{zero, zero};
  mundy::math::Vector<Scalar, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar dist = mundy::math::find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective, theta_phi_sol, min_objective_delta);
      if (dist < global_dist) {
        global_dist = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // Evaluating the objective updates the shared normal and foot points
  shared_normal_objective(global_theta_phi_sol);
  return mundy::math::dot(metric(closest_point, point), ellipsoid_normal);
}
//@}

//! \name Free space distance calculations
//@{

template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point,  //
                                const Ellipsoid<Scalar>& ellipsoid) {
  return distance_pbc(point, ellipsoid, FreeSpaceMetric<Scalar>{});
}

template <typename Scalar, typename DistanceType>
KOKKOS_FUNCTION Scalar distance(const DistanceType distance_type,  //
                                const Point<Scalar>& point,        //
                                const Ellipsoid<Scalar>& ellipsoid) {
  return distance_pbc(distance_type, point, ellipsoid, FreeSpaceMetric<Scalar>{});
}

template <typename Scalar, typename DistanceType>
KOKKOS_FUNCTION Scalar distance(const DistanceType distance_type,    //
                                const Point<Scalar>& point,          //
                                const Ellipsoid<Scalar>& ellipsoid,  //
                                Point<Scalar>& closest_point,        //
                                mundy::math::Vector3<Scalar>& ellipsoid_normal) {
  return distance_pbc(distance_type, point, ellipsoid, FreeSpaceMetric<Scalar>{},  //
                  closest_point, ellipsoid_normal);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_POINTELLIPSOID_HPP_
