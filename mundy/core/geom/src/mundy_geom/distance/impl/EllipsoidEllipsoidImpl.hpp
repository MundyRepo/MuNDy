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

#ifndef MUNDY_GEOM_DISTANCE_IMPL_ELLIPSOIDELLIPSOIDIMPL_HPP_
#define MUNDY_GEOM_DISTANCE_IMPL_ELLIPSOIDELLIPSOIDIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_geom/distance/PointPoint.hpp>    // for mundy::distance(Point, Point)
#include <mundy_geom/primitives/Ellipsoid.hpp>   // for mundy::Ellipsoid, map_surface_normal_to_foot_point_on_ellipsoid
#include <mundy_geom/primitives/Point.hpp>       // for mundy::Point
#include <mundy_math/Quaternion.hpp>             // for mundy::conjugate
#include <mundy_math/Tolerance.hpp>              // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>                // for mundy::Vector3, mundy::norm, mundy::dot
#include <mundy_math/cmath.hpp>                  // for mundy::sin, mundy::cos
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace impl {

/// \brief Cost-only functor for ellipsoid–ellipsoid distance.
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
    const Scalar sth = sin(tp[0]), cth = cos(tp[0]);
    const Scalar sph = sin(tp[1]), cph = cos(tp[1]);
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
    const Scalar sth = sin(tp[0]), cth = cos(tp[0]);
    const Scalar sph = sin(tp[1]), cph = cos(tp[1]);

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

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_IMPL_ELLIPSOIDELLIPSOIDIMPL_HPP_
