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

#ifndef MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_
#define MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_

/// \file QuasiUniformSpherocylinder.hpp
/// \brief Quasi-uniform surface quadrature on a unit-radius spherocylinder.
///
/// The cylindrical surface is discretized into Nz axial rings, each containing Ntheta equally spaced circumferential
/// nodes, with successive rings azimuthally staggered. Each hemispherical endcap is discretized using a Fibonacci
/// lattice of Ncap approximately uniformly distributed nodes, constructed by uniformly spacing points in the cosine
/// of the polar angle and incrementing the azimuth by the golden angle.
///
/// The number of axial rings and endcap nodes is determined by matching their nominal surface spacing to the
/// circumferential spacing, h = 2 pi / Ntheta, giving Nz = round(AspectRatio / h) and Ncap = round(2 pi / h^2).
/// For AspectRatio = 0, Nz = 0; otherwise at least one cylindrical ring is used.
///
/// The reference spherocylinder has radius 1 and cylindrical length AspectRatio, aligned with the z-axis.
/// Points are (x, y, z) triples ordered cylinder, north cap, and south cap. For a physical radius a and cylindrical
/// length L = a * AspectRatio, scale the points by a and the weights by a^2.
///
/// Unlike Gaussian quadrature, this rule prioritizes approximately uniform surface spacing rather than polynomial
/// exactness. Points and weights are computed at compile time using double-double arithmetic before rounding to
/// Scalar. Construction requires C++20 floating-point non-type template parameters.

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/impl/QuasiUniformSpherocylinderImpl.hpp>  // for rule construction
#include <mundy_utils/requires.hpp>                            // for MUNDY_REQUIRES

namespace mundy {

/// \brief Quasi-uniform surface rule for a unit-radius spherocylinder of cylindrical length AspectRatio.
///
/// Ntheta controls the circumferential resolution; the corresponding cylinder and cap counts are determined by
/// the aspect ratio. Nodes and weights are computed at compile time and rounded to Scalar.
template <class Scalar, unsigned Ntheta, double AspectRatio>
class QuasiUniformSpherocylinder {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "QuasiUniformSpherocylinder: Scalar must be float or double.");
  static_assert(Ntheta >= 3, "QuasiUniformSpherocylinder: a rule needs at least three circumferential nodes.");
  static_assert(AspectRatio >= 0.0, "QuasiUniformSpherocylinder: AspectRatio must be nonnegative.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_points_per_ring = Ntheta;
  static constexpr unsigned num_cylinder_rings =  //
      impl::QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio>::num_cylinder_rings;
  static constexpr unsigned num_cap_points =  //
      impl::QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio>::num_cap_points;
  static constexpr unsigned num_points =  //
      impl::QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio>::num_points;

  /// \brief The points as (x, y, z) triples, ordered by cylinder, north cap, then south cap.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> points() {
    return rule().points;
  }

  /// \brief The num_points surface-area weights, in the order of points().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, num_points> weights() {
    return rule().weights;
  }

  /// \brief sum_i w_i f(x_i, y_i, z_i), approximating the unit-radius spherocylinder surface integral.
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(const Function& f) {
    constexpr impl::QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio> r = rule();
    auto sum = r.weights[0] * f(r.points[0], r.points[1], r.points[2]);
    for (unsigned i = 1; i < num_points; ++i) {
      sum += r.weights[i] * f(r.points[3 * i], r.points[3 * i + 1], r.points[3 * i + 2]);
    }
    return sum;
  }

 private:
  /// \brief The reference rule, built once per (Scalar, Ntheta, AspectRatio) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr impl::QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio> rule() {
    return impl::make_quasi_uniform_spherocylinder_rule<Scalar, Ntheta, AspectRatio>();
  }
};

/// \brief The quasi-uniform spherocylinder rule with run-time resolution and aspect ratio, into std::vectors.
///
/// Resizes points to 3 * (ntheta * nz + 2 * ncap) and weights to ntheta * nz + 2 * ncap.
template <class Scalar>
void quasi_uniform_spherocylinder_rule(unsigned ntheta, double aspect_ratio, std::vector<Scalar>& points,
                                       std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "quasi_uniform_spherocylinder_rule: Scalar must be float or double.");
  MUNDY_THROW_REQUIRE(ntheta >= 3, std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: at least three circumferential nodes are required.");
  MUNDY_THROW_REQUIRE(aspect_ratio >= 0.0, std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: aspect_ratio must be nonnegative.");

  const unsigned nz = impl::compute_num_cylinder_rings(ntheta, aspect_ratio);
  const unsigned ncap = impl::compute_num_cap_points(ntheta);
  const unsigned total = nz * ntheta + 2u * ncap;

  points.resize(3u * total);
  weights.resize(total);
  for (unsigned i = 0; i < nz; ++i) {
    impl::write_quasi_uniform_spherocylinder_cylinder_ring<Scalar>(ntheta, nz, aspect_ratio, i, points, weights);
  }
  for (unsigned i = 0; i < ncap; ++i) {
    impl::write_quasi_uniform_spherocylinder_cap_pair<Scalar>(ntheta, nz, ncap, aspect_ratio, i, points, weights);
  }
}

/// \brief The quasi-uniform spherocylinder rule with run-time resolution and aspect ratio, into Kokkos::View
///
/// Resizes points to 3 * (ntheta * nz + 2 * ncap) and weights to ntheta * nz + 2 * ncap.
template <class PointsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<WeightsView>)
void quasi_uniform_spherocylinder_rule(unsigned ntheta, double aspect_ratio, PointsView& points, WeightsView& weights) {
  using scalar_t = typename PointsView::non_const_value_type;
  using execution_space = typename PointsView::execution_space;
  static_assert(PointsView::rank() == 1 && WeightsView::rank() == 1,
                "quasi_uniform_spherocylinder_rule: views must have rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "quasi_uniform_spherocylinder_rule: Scalar must be float or double.");
  static_assert(std::is_same_v<scalar_t, typename PointsView::value_type> &&
                    std::is_same_v<scalar_t, typename WeightsView::value_type>,
                "quasi_uniform_spherocylinder_rule: points and weights must be writable views of the same type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "quasi_uniform_spherocylinder_rule: weights must be accessible from the points' execution space.");
  MUNDY_THROW_REQUIRE(ntheta >= 3, std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: at least three circumferential nodes are required.");
  MUNDY_THROW_REQUIRE(aspect_ratio >= 0.0, std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: aspect_ratio must be nonnegative.");

  const unsigned nz = impl::compute_num_cylinder_rings(ntheta, aspect_ratio);
  const unsigned ncap = impl::compute_num_cap_points(ntheta);
  const unsigned total = nz * ntheta + 2u * ncap;

  Kokkos::resize(points, 3u * total);
  Kokkos::resize(weights, total);

  Kokkos::parallel_for(
      "mundy::quasi_uniform_spherocylinder_rule::cylinder", Kokkos::RangePolicy<execution_space>(0, nz),
      KOKKOS_LAMBDA(const unsigned i) {
        impl::write_quasi_uniform_spherocylinder_cylinder_ring<scalar_t>(ntheta, nz, aspect_ratio, i, points, weights);
      });

  Kokkos::parallel_for(
      "mundy::quasi_uniform_spherocylinder_rule::caps", Kokkos::RangePolicy<execution_space>(0, ncap),
      KOKKOS_LAMBDA(const unsigned i) {
        impl::write_quasi_uniform_spherocylinder_cap_pair<scalar_t>(ntheta, nz, ncap, aspect_ratio, i, points, weights);
      });
  execution_space().fence();
}

}  // namespace mundy

#endif  // MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_
