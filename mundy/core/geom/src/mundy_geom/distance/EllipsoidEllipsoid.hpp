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
#include <mundy_math/minimize.hpp>              // for mundy::find_min_with_fdf, find_min_using_approximate_derivatives
#include <mundy_utils/requires.hpp>
#include <mundy_math/cmath.hpp>
#include <mundy_geom/distance/impl/EllipsoidEllipsoidImpl.hpp>  // for the cost-only and FDF objective functors

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Ellipsoid–ellipsoid shared-normal signed separation distance.
///
/// For an autodiff scalar the result is differentiable with respect to the ellipsoid parameters.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
KOKKOS_FUNCTION typename EllipsoidType1::value_type distance(const EllipsoidType1& ellipsoid1,
                                                            const EllipsoidType2& ellipsoid2) {
  return distance(SharedNormalSigned{}, ellipsoid1, ellipsoid2);
}

/// \brief Convenience overload returning only the distance (discards the closest points and normals).
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
KOKKOS_FUNCTION typename EllipsoidType1::value_type distance([[maybe_unused]] const SharedNormalSigned tag,
                                                            const EllipsoidType1& ellipsoid1,
                                                            const EllipsoidType2& ellipsoid2) {
  using Scalar = typename EllipsoidType1::value_type;
  Point<Scalar> cp1, cp2;
  mundy::Vector3<Scalar> sn1, sn2;
  return distance(tag, ellipsoid1, ellipsoid2, cp1, cp2, sn1, sn2);
}

/// \brief Shared-normal signed separation distance, also returning the closest points and shared
/// normals. For an autodiff scalar the returned distance is differentiable with respect to the
/// ellipsoid parameters.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
KOKKOS_FUNCTION typename EllipsoidType1::value_type
    distance([[maybe_unused]] const SharedNormalSigned,
             const EllipsoidType1& ellipsoid1,
             const EllipsoidType2& ellipsoid2,
             Point<typename EllipsoidType1::value_type>& closest_point1,
             Point<typename EllipsoidType1::value_type>& closest_point2,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal1,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal2) {
  using Scalar          = typename EllipsoidType1::value_type;
  using Passive         = passive_scalar_t<Scalar>;  // double for an AD scalar; Scalar itself otherwise
  using PassiveEllipsoid = Ellipsoid<Passive>;
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr Passive min_allowable_cost  = -Kokkos::Experimental::infinity_v<Passive>;    // no minimum cost early-exit
  constexpr Passive min_objective_delta = mundy::get_relaxed_zero_tolerance<Passive>();  // L-BFGS convergence tolerance

  // Envelope theorem: solve for the optimal angles theta* in the passive type (the L-BFGS solver is
  // double-only), then recompute the closest points on the original ellipsoids with theta* frozen.
  // Since d(objective)/d(theta) = 0 at the optimum, freezing theta* yields the exact derivative of
  // the distance w.r.t. the ellipsoid parameters without differentiating through the solve.
  const PassiveEllipsoid e0p = impl::passive_copy(ellipsoid1);
  const PassiveEllipsoid e1p = impl::passive_copy(ellipsoid2);

  Point<Passive> cp0p, cp1p;
  mundy::Vector3<Passive> sn0p, sn1p;
  impl::EllipsoidEllipsoidObjectiveFDF<PassiveEllipsoid, PassiveEllipsoid> fdf(
      e0p, e1p, sn0p, sn1p, cp0p, cp1p);

  constexpr Passive pi            = Kokkos::numbers::pi_v<Passive>;
  constexpr Passive zero          = static_cast<Passive>(0.0);
  constexpr Passive half_pi       = static_cast<Passive>(0.5) * pi;
  constexpr Passive one_third_pi  = pi / static_cast<Passive>(3.0);
  constexpr Passive five_third_pi = static_cast<Passive>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Passive, 3> theta_guesses{zero, half_pi, pi};
  constexpr Kokkos::Array<Passive, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Passive global_dist = Kokkos::Experimental::infinity_v<Passive>;
  mundy::Vector<Passive, 2> theta_phi_sol{zero, zero};
  mundy::Vector<Passive, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol      = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Passive dist = find_min_with_fdf<lbfgs_max_memory_size>(
          fdf, theta_phi_sol, min_allowable_cost, min_objective_delta);
      if (dist < global_dist) {
        global_dist          = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // theta* frozen as a zero-derivative constant: the shared normal n is constant; the foot points
  // carry the ellipsoids' derivatives through map_surface_normal_to_foot_point_on_ellipsoid.
  const Scalar theta = static_cast<Scalar>(global_theta_phi_sol[0]);
  const Scalar phi   = static_cast<Scalar>(global_theta_phi_sol[1]);
  const Scalar sth = sin(theta), cth = cos(theta);
  const Scalar sph = sin(phi), cph = cos(phi);
  const mundy::Vector3<Scalar> n{sth * cph, sth * sph, cth};

  shared_normal1 = n;
  shared_normal2 = -n;
  closest_point1 = map_surface_normal_to_foot_point_on_ellipsoid(n, ellipsoid1);
  closest_point2 = map_surface_normal_to_foot_point_on_ellipsoid(-n, ellipsoid2);
  return mundy::dot(closest_point2 - closest_point1, shared_normal1);
}

/// \brief Finite-difference-gradient variant — retained for benchmarking only.
///
/// Uses central-difference approximations for the L-BFGS gradient instead of the analytical
/// gradient.  Kept so that timing and accuracy comparisons with the default FDF implementation
/// remain reproducible.  Not intended for production use.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
KOKKOS_FUNCTION typename EllipsoidType1::value_type distance(
    [[maybe_unused]] const SharedNormalSignedFiniteDiff tag,
    const EllipsoidType1& ellipsoid1,
    const EllipsoidType2& ellipsoid2) {
  using Scalar = typename EllipsoidType1::value_type;
  Point<Scalar> cp1, cp2;
  mundy::Vector3<Scalar> sn1, sn2;
  return distance(tag, ellipsoid1, ellipsoid2, cp1, cp2, sn1, sn2);
}

template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
KOKKOS_FUNCTION typename EllipsoidType1::value_type
    distance([[maybe_unused]] const SharedNormalSignedFiniteDiff,
             const EllipsoidType1& ellipsoid1,
             const EllipsoidType2& ellipsoid2,
             Point<typename EllipsoidType1::value_type>& closest_point1,
             Point<typename EllipsoidType1::value_type>& closest_point2,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal1,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal2) {
  using Scalar          = typename EllipsoidType1::value_type;
  using Passive         = passive_scalar_t<Scalar>;  // double for an AD scalar; Scalar itself otherwise
  using PassiveEllipsoid = Ellipsoid<Passive>;
  constexpr size_t lbfgs_max_memory_size = 10;
  constexpr Passive min_allowable_cost  = -Kokkos::Experimental::infinity_v<Passive>;    // no minimum cost early-exit
  constexpr Passive min_objective_delta = mundy::get_relaxed_zero_tolerance<Passive>();  // L-BFGS convergence tolerance

  // Same envelope as the default path (solve theta* in the passive type, re-evaluate at frozen theta*),
  // but with the finite-difference-gradient minimiser rather than the analytical FDF one.
  const PassiveEllipsoid e0p = impl::passive_copy(ellipsoid1);
  const PassiveEllipsoid e1p = impl::passive_copy(ellipsoid2);
  Point<Passive> cp0p, cp1p;
  mundy::Vector3<Passive> sn0p, sn1p;
  impl::EllipsoidEllipsoidObjective<PassiveEllipsoid, PassiveEllipsoid> objective_p(e0p, e1p, sn0p, sn1p, cp0p, cp1p);

  constexpr Passive pi            = Kokkos::numbers::pi_v<Passive>;
  constexpr Passive zero          = static_cast<Passive>(0.0);
  constexpr Passive half_pi       = static_cast<Passive>(0.5) * pi;
  constexpr Passive one_third_pi  = pi / static_cast<Passive>(3.0);
  constexpr Passive five_third_pi = static_cast<Passive>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Passive, 3> theta_guesses{zero, half_pi, pi};
  constexpr Kokkos::Array<Passive, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Passive global_dist = Kokkos::Experimental::infinity_v<Passive>;
  mundy::Vector<Passive, 2> theta_phi_sol{zero, zero};
  mundy::Vector<Passive, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol      = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Passive dist = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          objective_p, theta_phi_sol, min_allowable_cost, min_objective_delta);
      if (dist < global_dist) {
        global_dist          = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // Re-evaluate on the original ellipsoids with theta* frozen as a zero-derivative constant.
  impl::EllipsoidEllipsoidObjective<EllipsoidType1, EllipsoidType2> objective_ad(
      ellipsoid1, ellipsoid2, shared_normal1, shared_normal2, closest_point1, closest_point2);
  const mundy::Vector<Scalar, 2> theta_star{static_cast<Scalar>(global_theta_phi_sol[0]),
                                            static_cast<Scalar>(global_theta_phi_sol[1])};
  objective_ad(theta_star);
  return mundy::dot(closest_point2 - closest_point1, shared_normal1);
}

//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
