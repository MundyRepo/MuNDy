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

namespace mundy {

// ============================================================
// Implementation details — not public API
// ============================================================

namespace impl {

/// \brief Cost-only functor for ellipsoid–ellipsoid distance, used by SharedNormalSignedFiniteDiff.
/// \internal
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
class EllipsoidEllipsoidObjective {
 public:
  using Scalar = typename EllipsoidType1::value_type;

  KOKKOS_FUNCTION
  EllipsoidEllipsoidObjective(const EllipsoidType1& e0, const EllipsoidType2& e1,
                               mundy::Vector3<Scalar>& sn0, mundy::Vector3<Scalar>& sn1,
                               Point<Scalar>& fp0, Point<Scalar>& fp1)
      : e0_(e0), e1_(e1), sn0_(sn0), sn1_(sn1), fp0_(fp0), fp1_(fp1) {
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& tp) const {
    const Scalar sth = Kokkos::sin(tp[0]), cth = Kokkos::cos(tp[0]);
    const Scalar sph = Kokkos::sin(tp[1]), cph = Kokkos::cos(tp[1]);
    sn0_.set(sth * cph, sth * sph, cth);
    sn1_ = -sn0_;
    fp0_ = map_surface_normal_to_foot_point_on_ellipsoid(sn0_, e0_);
    fp1_ = map_surface_normal_to_foot_point_on_ellipsoid(sn1_, e1_);
    return distance(fp0_, fp1_);
  }

 private:
  const EllipsoidType1& e0_;
  const EllipsoidType2& e1_;
  mundy::Vector3<Scalar>& sn0_;
  mundy::Vector3<Scalar>& sn1_;
  Point<Scalar>& fp0_;
  Point<Scalar>& fp1_;
};

/// \brief Combined FDF functor: computes cost and gradient in one pass, sharing foot-point
/// evaluations between the objective value and the Jacobian.
///
/// This is the implementation backing the default \c distance(SharedNormalSigned{}, ...) call.
/// It exploits the identity D = dot(p_body, n_body) to obtain the normalisation factor without
/// a sqrt.  See EllipsoidEllipsoidGradientNotes.md for the full derivation.
///
/// \internal
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
class EllipsoidEllipsoidObjectiveFDF {
 public:
  using Scalar = typename EllipsoidType1::value_type;

  KOKKOS_FUNCTION
  EllipsoidEllipsoidObjectiveFDF(const EllipsoidType1& e0, const EllipsoidType2& e1,
                                  mundy::Vector3<Scalar>& sn0, mundy::Vector3<Scalar>& sn1,
                                  Point<Scalar>& fp0, Point<Scalar>& fp1)
      : e0_(e0), e1_(e1), sn0_(sn0), sn1_(sn1), fp0_(fp0), fp1_(fp1) {
  }

  KOKKOS_FUNCTION Scalar operator()(const mundy::Vector<Scalar, 2>& tp,
                                     mundy::Vector<Scalar, 2>& g) const {
    const Scalar sth = Kokkos::sin(tp[0]), cth = Kokkos::cos(tp[0]);
    const Scalar sph = Kokkos::sin(tp[1]), cph = Kokkos::cos(tp[1]);

    const mundy::Vector3<Scalar> n{sth * cph, sth * sph, cth};
    sn0_.set(n[0], n[1], n[2]);
    sn1_ = -sn0_;

    // Foot points computed once — shared between objective value and gradient.
    fp0_ = map_surface_normal_to_foot_point_on_ellipsoid( n, e0_);
    fp1_ = map_surface_normal_to_foot_point_on_ellipsoid(-n, e1_);

    const auto d    = fp0_ - fp1_;
    const Scalar dist = mundy::norm(d);

    if (dist < get_zero_tolerance<Scalar>()) {
      g = {Scalar(0), Scalar(0)};
      return dist;
    }

    const mundy::Vector3<Scalar> unit_d = d / dist;

    // Jacobian contributions — reuse fp0_/fp1_ to avoid recomputing p_body.
    const mundy::Vector3<Scalar> grad_n = jac(e0_,  n, fp0_, unit_d)
                                        + jac(e1_, -n, fp1_, unit_d);

    g = {mundy::dot(grad_n, mundy::Vector3<Scalar>{cth * cph, cth * sph, -sth}),
         mundy::dot(grad_n, mundy::Vector3<Scalar>{-sth * sph, sth * cph, Scalar(0)})};
    return dist;
  }

 private:
  // Computes R * J_body * R^T * unit_d using p_lab from the forward pass.
  // D = dot(p_body, n_body)  (identity, no sqrt).
  template <ValidEllipsoidType E>
  KOKKOS_FUNCTION static mundy::Vector3<Scalar> jac(const E& e, const mundy::Vector3<Scalar>& n_lab,
                                                     const Point<Scalar>& p_lab,
                                                     const mundy::Vector3<Scalar>& unit_d) {
    const auto qc                     = conjugate(e.orientation());
    const mundy::Vector3<Scalar> p_body = qc * (p_lab - e.center());
    const mundy::Vector3<Scalar> n_body = qc * n_lab;
    const Scalar D                    = mundy::dot(p_body, n_body);
    if (D < get_zero_tolerance<Scalar>()) return {Scalar(0), Scalar(0), Scalar(0)};
    const mundy::Vector3<Scalar> u  = qc * unit_d;
    const Scalar r1 = e.radius_1(), r2 = e.radius_2(), r3 = e.radius_3();
    const mundy::Vector3<Scalar> Au{r1 * r1 * u[0], r2 * r2 * u[1], r3 * r3 * u[2]};
    return e.orientation() * ((Au - p_body * mundy::dot(p_body, u)) / D);
  }

  const EllipsoidType1& e0_;
  const EllipsoidType2& e1_;
  mundy::Vector3<Scalar>& sn0_;
  mundy::Vector3<Scalar>& sn1_;
  Point<Scalar>& fp0_;
  Point<Scalar>& fp1_;
};

}  // namespace impl

// ============================================================
// Public API
// ============================================================

/// \addtogroup MundyGeomDistance
/// @{

//! \name Free space distance calculations
//@{

/// \brief Ellipsoid–ellipsoid shared-normal signed separation distance.
///
/// Dispatches to \c distance(SharedNormalSigned{}, ...).  The implementation uses a combined
/// cost-and-gradient (FDF) L-BFGS minimiser over a 3×3 grid of initial guesses.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
typename EllipsoidType1::value_type distance(const EllipsoidType1& ellipsoid1,
                                           const EllipsoidType2& ellipsoid2) {
  return distance(SharedNormalSigned{}, ellipsoid1, ellipsoid2);
}

/// \brief 2-arg overload — convenience wrapper for the 6-arg FDF implementation.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
typename EllipsoidType1::value_type distance([[maybe_unused]] const SharedNormalSigned tag,
                                           const EllipsoidType1& ellipsoid1,
                                           const EllipsoidType2& ellipsoid2) {
  using Scalar = typename EllipsoidType1::value_type;
  Point<Scalar> cp1, cp2;
  mundy::Vector3<Scalar> sn1, sn2;
  return distance(tag, ellipsoid1, ellipsoid2, cp1, cp2, sn1, sn2);
}

/// \brief Full 6-arg distance using the combined FDF L-BFGS minimiser.
///
/// Each evaluation point in the minimiser calls \c map_surface_normal_to_foot_point_on_ellipsoid
/// exactly once for each ellipsoid, with the result shared between the objective value and the
/// analytical gradient.
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
MUNDY_REQUIRES(std::is_same_v<typename EllipsoidType1::value_type, typename EllipsoidType2::value_type>)
typename EllipsoidType1::value_type
    distance([[maybe_unused]] const SharedNormalSigned,
             const EllipsoidType1& ellipsoid1,
             const EllipsoidType2& ellipsoid2,
             Point<typename EllipsoidType1::value_type>& closest_point1,
             Point<typename EllipsoidType1::value_type>& closest_point2,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal1,
             mundy::Vector3<typename EllipsoidType1::value_type>& shared_normal2) {
  using Scalar = typename EllipsoidType1::value_type;
  constexpr size_t lbfgs_max_memory_size = 10;

  impl::EllipsoidEllipsoidObjectiveFDF<EllipsoidType1, EllipsoidType2> fdf(
      ellipsoid1, ellipsoid2, shared_normal1, shared_normal2, closest_point1, closest_point2);

  constexpr Scalar pi           = Kokkos::numbers::pi_v<Scalar>;
  constexpr Scalar zero         = static_cast<Scalar>(0.0);
  constexpr Scalar half_pi      = static_cast<Scalar>(0.5) * pi;
  constexpr Scalar one_third_pi  = pi / static_cast<Scalar>(3.0);
  constexpr Scalar five_third_pi = static_cast<Scalar>(5.0) * one_third_pi;
  constexpr Kokkos::Array<Scalar, 3> theta_guesses{zero, half_pi, pi};
  constexpr Kokkos::Array<Scalar, 3> phi_guesses{one_third_pi, pi, five_third_pi};

  Scalar global_dist = Kokkos::Experimental::infinity_v<Scalar>;
  mundy::Vector<Scalar, 2> theta_phi_sol{zero, zero};
  mundy::Vector<Scalar, 2> global_theta_phi_sol{zero, zero};
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_sol    = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar dist = find_min_with_fdf<lbfgs_max_memory_size>(
          fdf, theta_phi_sol, mundy::get_relaxed_zero_tolerance<Scalar>());
      if (dist < global_dist) {
        global_dist = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  // Final FDF call updates closest_point1/2 and shared_normal1/2 for the global minimiser.
  mundy::Vector<Scalar, 2> dummy_g;
  fdf(global_theta_phi_sol, dummy_g);
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
  using Scalar = typename EllipsoidType1::value_type;
  constexpr size_t lbfgs_max_memory_size = 10;

  impl::EllipsoidEllipsoidObjective<EllipsoidType1, EllipsoidType2> objective(
      ellipsoid1, ellipsoid2, shared_normal1, shared_normal2, closest_point1, closest_point2);

  constexpr Scalar pi           = Kokkos::numbers::pi_v<Scalar>;
  constexpr Scalar zero         = static_cast<Scalar>(0.0);
  constexpr Scalar half_pi      = static_cast<Scalar>(0.5) * pi;
  constexpr Scalar one_third_pi  = pi / static_cast<Scalar>(3.0);
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
          objective, theta_phi_sol, mundy::get_relaxed_zero_tolerance<Scalar>());
      if (dist < global_dist) {
        global_dist = dist;
        global_theta_phi_sol = theta_phi_sol;
      }
    }
  }

  objective(global_theta_phi_sol);
  return mundy::dot(closest_point2 - closest_point1, shared_normal1);
}

//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
