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

#ifndef MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
#define MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

// Our libs
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector.hpp>      // for mundy::math::Vector
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_math/minimize.hpp>    // for mundy::math::find_min_using_approximate_derivatives

namespace mundy {

namespace math {

namespace distance {

/* Notes:

The purpose of the shared normal ellipsoid-ellipsoid distance function is to compute the minimum distance between two
ellipsoids in 3D space under the constraint that they share a normal at the corresponding points of closest approach.
The benefit of the shared normal approach is that it takes the optimization problem from 4D to 2D. These two dimensions
are the spherical polar coordinates for the angle of the shared normal vector. We convert these coordinates to the
normal vector in R3 and then use a special map which takes the normal vector and maps it its foot point on the
ellipsoid.
*/

namespace impl {

template <typename Scalar>
KOKKOS_FUNCTION Scalar sign(Scalar x) {
  return (x > 0) - (x < 0);
}

}  // namespace impl

template <typename Scalar>
KOKKOS_FUNCTION Vector3<Scalar> map_spherical_to_unit_vector(Scalar theta, Scalar phi) {
  const Scalar sin_theta = std::sin(theta);
  const Scalar cos_theta = std::cos(theta);
  const Scalar sin_phi = std::sin(phi);
  const Scalar cos_phi = std::cos(phi);
  return Vector3<Scalar>(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

template <typename Scalar, typename Accessor, typename OwnershipType>
KOKKOS_FUNCTION Vector3<Scalar> map_body_frame_normal_to_ellipsoid(
    const Vector3<Scalar, Accessor, OwnershipType>& body_frame_nhat, Scalar r1, Scalar r2, Scalar r3) {
  constexpr Scalar half = static_cast<Scalar>(0.5);
  constexpr Scalar one = static_cast<Scalar>(1.0);
  constexpr Scalar zero = static_cast<Scalar>(0.0);

  const Scalar sign0 = Kokkos::copysign(one, body_frame_nhat[0]);
  const Scalar sign1 = Kokkos::copysign(one, body_frame_nhat[1]);
  const Scalar sign2 = Kokkos::copysign(one, body_frame_nhat[2]);

  Scalar alpha1, alpha2;
  if (sign0 * body_frame_nhat[0] > get_zero_tolerance<Scalar>()) {
    const Scalar tmp0 = one / (r1 * body_frame_nhat[0]);
    const Scalar tmp1 = tmp0 * r2 * body_frame_nhat[1];
    const Scalar tmp2 = tmp0 * r3 * body_frame_nhat[2];
    alpha1 = one / (one + tmp1 * tmp1);
    alpha2 = one / (one + tmp2 * tmp2 * alpha1);
  } else if (sign1 * body_frame_nhat[1] > get_zero_tolerance<Scalar>()) {
    const Scalar tmp = r3 * body_frame_nhat[2] / (r2 * body_frame_nhat[1]);
    alpha1 = zero;
    alpha2 = one / (one + tmp * tmp);
  } else {
    alpha1 = zero;
    alpha2 = zero;
  }

  const Scalar sqrt_alpha1 = Kokkos::sqrt(alpha1);
  const Scalar sqrt_alpha2 = Kokkos::sqrt(alpha2);

  const Scalar x = half * sign0 * ((one + sign0) * r1 + (one - sign0) * r1) * sqrt_alpha1 * sqrt_alpha2;
  const Scalar y = half * sign1 * ((one + sign1) * r2 + (one - sign1) * r2) * Kokkos::sqrt(one - alpha1) * sqrt_alpha2;
  const Scalar z = half * sign2 * ((one + sign2) * r3 + (one - sign2) * r3) * Kokkos::sqrt(one - alpha2);

  return Vector3<Scalar>(x, y, z);
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Vector3<Scalar> map_surface_normal_to_foot_point_on_ellipsoid(
    const Vector3<Scalar, Accessor1, OwnershipType1>& lab_frame_ellipsoid_nhat,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation,
    const Vector3<Scalar, Accessor3, OwnershipType3>& center, Scalar r1, Scalar r2, Scalar r3) {
  const auto body_frame_nhat = conjugate(orientation) * lab_frame_ellipsoid_nhat;
  const auto body_frame_foot_point = map_body_frame_normal_to_ellipsoid(body_frame_nhat, r1, r2, r3);
  return orientation * body_frame_foot_point + center;
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Vector3<Scalar> map_surface_normal_to_foot_point_on_ellipsoid(
    const Vector<Scalar, 2, Accessor1, OwnershipType1>& theta_phi,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation,
    const Vector3<Scalar, Accessor3, OwnershipType3>& center, Scalar r1, Scalar r2, Scalar r3) {
  // Step 1: Map theta and phi to the lab frame normal vector
  const Vector3<Scalar> lab_frame_ellipsoid_nhat = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);
  return map_surface_normal_to_foot_point_on_ellipsoid(lab_frame_ellipsoid_nhat, orientation, center, r1, r2, r3);
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
class EllipsoidPointObjective {
 public:
  KOKKOS_FUNCTION
  EllipsoidPointObjective(const Vector3<Scalar, Accessor1, OwnershipType1>& center,
                          const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation, const Scalar r1,
                          const Scalar r2, const Scalar r3, const Vector3<Scalar, Accessor3, OwnershipType3>& point)
      : center_(center), orientation_(orientation), r1_(r1), r2_(r2), r3_(r3), point_(point) {
  }

  KOKKOS_FUNCTION Scalar operator()(const Vector<Scalar, 2>& theta_phi) const {
    // Map theta and phi to the lab frame normal vector
    const Scalar sin_theta = std::sin(theta_phi[0]);
    const Scalar cos_theta = std::cos(theta_phi[0]);
    const Scalar sin_phi = std::sin(theta_phi[1]);
    const Scalar cos_phi = std::cos(theta_phi[1]);
    const Vector3<Scalar> lab_frame_nhat(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

    // Map the normal vector to the foot point on the ellipsoid
    const Vector3<Scalar> lab_frame_foot_point =
        map_surface_normal_to_foot_point_on_ellipsoid(lab_frame_nhat, orientation_, center_, r1_, r2_, r3_);

    // The objective is the shared normal euclidean separation distance. NOT the signed separation distance.
    const Scalar euclidean_separation_distance = mundy::math::norm(point_ - lab_frame_foot_point);
    return euclidean_separation_distance;
  }

 private:
  const Vector3<Scalar, Accessor1, OwnershipType1>& center_;
  const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation_;
  const Scalar r1_;
  const Scalar r2_;
  const Scalar r3_;
  const Vector3<Scalar, Accessor3, OwnershipType3>& point_;
};

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Scalar shared_normal_ssd_between_ellipsoid_and_point(
    const Vector3<Scalar, Accessor1, OwnershipType1>& center,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation, const Scalar r1, const Scalar r2, const Scalar r3,
    const Vector3<Scalar, Accessor3, OwnershipType3>& point, Vector3<Scalar>& closest_point,
    Vector3<Scalar>& ellipsoid_normal) {
  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const Scalar min_objective_delta = 1e-7;
  constexpr size_t lbfgs_max_memory_size = 10;

  EllipsoidPointObjective shared_normal_objective(center, orientation, r1, r2, r3, point);

  Scalar global_separation_distance = Kokkos::Experimental::infinity_v<Scalar>;
  Vector<Scalar, 2> global_theta_phi_guess_and_solution{0, 0};
  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  Kokkos::Array<Scalar, 3> theta_guesses = {0, 0.5 * pi, pi};
  Kokkos::Array<Scalar, 3> phi_guesses = {pi / 3.0, pi, 5.0 * pi / 3.0};

  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      Vector<Scalar, 2> theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar separation_distance = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_distance < global_separation_distance) {
        global_separation_distance = separation_distance;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  ellipsoid_normal =
      map_spherical_to_unit_vector(global_theta_phi_guess_and_solution[0], global_theta_phi_guess_and_solution[1]);
  closest_point = map_surface_normal_to_foot_point_on_ellipsoid(ellipsoid_normal, orientation, center, r1, r2, r3);
  const Scalar global_signed_separation_distance = mundy::math::dot(point - closest_point, ellipsoid_normal);

  return global_signed_separation_distance;
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3>
KOKKOS_FUNCTION Scalar shared_normal_ssd_between_ellipsoid_and_point(
    const Vector3<Scalar, Accessor1, OwnershipType1>& center,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation, const Scalar r1, const Scalar r2, const Scalar r3,
    const Vector3<Scalar, Accessor3, OwnershipType3>& point) {
  Vector3<Scalar> closest_point;
  Vector3<Scalar> ellipsoid_normal;
  return shared_normal_ssd_between_ellipsoid_and_point(center, orientation, r1, r2, r3, point, closest_point,
                                                       ellipsoid_normal);
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename OwnershipType1, typename OwnershipType2, typename OwnershipType3, typename OwnershipType4>
class EllipsoidEllipsoidObjective {
 public:
  KOKKOS_FUNCTION
  EllipsoidEllipsoidObjective(const Vector3<Scalar, Accessor1, OwnershipType1>& center0,
                              const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation0, const Scalar r1_0,
                              const Scalar r2_0, const Scalar r3_0,
                              const Vector3<Scalar, Accessor3, OwnershipType3>& center1,
                              const Quaternion<Scalar, Accessor4, OwnershipType4>& orientation1, const Scalar r1_1,
                              const Scalar r2_1, const Scalar r3_1)
      : center0_(center0),
        orientation0_(orientation0),
        r1_0_(r1_0),
        r2_0_(r2_0),
        r3_0_(r3_0),
        center1_(center1),
        orientation1_(orientation1),
        r1_1_(r1_1),
        r2_1_(r2_1),
        r3_1_(r3_1) {
  }

  KOKKOS_FUNCTION Scalar operator()(const Vector<Scalar, 2>& theta_phi) const {
    const Vector3<Scalar> lab_frame_nhat0 = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);
    const Vector3<Scalar> lab_frame_foot_point0 =
        map_surface_normal_to_foot_point_on_ellipsoid(lab_frame_nhat0, orientation0_, center0_, r1_0_, r2_0_, r3_0_);
    const Vector3<Scalar> lab_frame_foot_point1 =
        map_surface_normal_to_foot_point_on_ellipsoid(-lab_frame_nhat0, orientation1_, center1_, r1_1_, r2_1_, r3_1_);
    const Scalar euclidean_separation_distance = mundy::math::norm(lab_frame_foot_point1 - lab_frame_foot_point0);
    return euclidean_separation_distance;
  }

 private:
  const Vector3<Scalar>& center0_;
  const Quaternion<Scalar>& orientation0_;
  const Scalar r1_0_;
  const Scalar r2_0_;
  const Scalar r3_0_;
  const Vector3<Scalar>& center1_;
  const Quaternion<Scalar>& orientation1_;
  const Scalar r1_1_;
  const Scalar r2_1_;
  const Scalar r3_1_;
};

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename Accessor5, typename Accessor6, typename Accessor7, typename Accessor8, typename OwnershipType1,
          typename OwnershipType2, typename OwnershipType3, typename OwnershipType4, typename OwnershipType5,
          typename OwnershipType6, typename OwnershipType7, typename OwnershipType8>
KOKKOS_FUNCTION Scalar shared_normal_ssd_between_ellipsoids(
    const Vector3<Scalar, Accessor1, OwnershipType1>& center0,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation0, const Scalar r1_0, const Scalar r2_0,
    const Scalar r3_0, const Vector3<Scalar, Accessor3, OwnershipType3>& center1,
    const Quaternion<Scalar, Accessor4, OwnershipType4>& orientation1, const Scalar r1_1, const Scalar r2_1,
    const Scalar r3_1, Vector3<Scalar, Accessor5, OwnershipType5>& closest_point0,
    Vector3<Scalar, Accessor6, OwnershipType6>& closest_point1,
    Vector3<Scalar, Accessor7, OwnershipType7>& shared_normal0,
    Vector3<Scalar, Accessor8, OwnershipType8>& shared_normal1) {
  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const Scalar min_objective_delta = 1e-7;
  constexpr size_t lbfgs_max_memory_size = 10;

  EllipsoidEllipsoidObjective shared_normal_objective(center0, orientation0, r1_0, r2_0, r3_0, center1, orientation1,
                                                      r1_1, r2_1, r3_1);

  Scalar global_separation_distance = Kokkos::Experimental::infinity_v<Scalar>;
  Vector<Scalar, 2> global_theta_phi_guess_and_solution;
  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  Kokkos::Array<Scalar, 3> theta_guesses = {0, 0.5 * pi, pi};
  Kokkos::Array<Scalar, 3> phi_guesses = {pi / 3.0, pi, 5.0 * pi / 3.0};

  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      Vector<Scalar, 2> theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar separation_distance = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_distance < global_separation_distance) {
        global_separation_distance = separation_distance;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  shared_normal0 =
      map_spherical_to_unit_vector(global_theta_phi_guess_and_solution[0], global_theta_phi_guess_and_solution[1]);
  shared_normal1 = -shared_normal0;
  closest_point0 =
      map_surface_normal_to_foot_point_on_ellipsoid(shared_normal0, orientation0, center0, r1_0, r2_0, r3_0);
  closest_point1 =
      map_surface_normal_to_foot_point_on_ellipsoid(shared_normal1, orientation1, center1, r1_1, r2_1, r3_1);

  return mundy::math::dot(closest_point1 - closest_point0, shared_normal0);
}

template <typename Scalar, typename Accessor1, typename Accessor2, typename Accessor3, typename Accessor4,
          typename OwnershipType1, typename OwnershipType2, typename OwnershipType3, typename OwnershipType4>
KOKKOS_FUNCTION Scalar shared_normal_ssd_between_ellipsoids(
    const Vector3<Scalar, Accessor1, OwnershipType1>& center0,
    const Quaternion<Scalar, Accessor2, OwnershipType2>& orientation0, const Scalar r1_0, const Scalar r2_0,
    const Scalar r3_0, const Vector3<Scalar, Accessor3, OwnershipType3>& center1,
    const Quaternion<Scalar, Accessor4, OwnershipType4>& orientation1, const Scalar r1_1, const Scalar r2_1,
    const Scalar r3_1) {
  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const Scalar min_objective_delta = 1e-7;
  constexpr size_t lbfgs_max_memory_size = 10;

  EllipsoidEllipsoidObjective shared_normal_objective(center0, orientation0, r1_0, r2_0, r3_0, center1, orientation1,
                                                      r1_1, r2_1, r3_1);

  Scalar global_separation_distance = Kokkos::Experimental::infinity_v<Scalar>;
  Vector<Scalar, 2> global_theta_phi_guess_and_solution{0, 0};
  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  Kokkos::Array<Scalar, 3> theta_guesses = {0, 0.5 * pi, pi};
  Kokkos::Array<Scalar, 3> phi_guesses = {pi / 3.0, pi, 5.0 * pi / 3.0};

  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      Vector<Scalar, 2> theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar separation_distance = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_distance < global_separation_distance) {
        global_separation_distance = separation_distance;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  const auto lab_frame_shared_normal0 =
      map_spherical_to_unit_vector(global_theta_phi_guess_and_solution[0], global_theta_phi_guess_and_solution[1]);
  const auto lab_frame_shared_normal1 = -lab_frame_shared_normal0;
  const auto foot_point0 =
      map_surface_normal_to_foot_point_on_ellipsoid(lab_frame_shared_normal0, orientation0, center0, r1_0, r2_0, r3_0);
  const auto foot_point1 =
      map_surface_normal_to_foot_point_on_ellipsoid(lab_frame_shared_normal1, orientation1, center1, r1_1, r2_1, r3_1);
  const Scalar global_signed_separation_distance =
      mundy::math::dot(foot_point1 - foot_point0, lab_frame_shared_normal0);

  return global_signed_separation_distance;
}

}  // namespace distance

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
