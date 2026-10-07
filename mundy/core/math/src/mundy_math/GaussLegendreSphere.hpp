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

#ifndef MUNDY_MATH_GAUSSLEGENDRESPHERE_HPP_
#define MUNDY_MATH_GAUSSLEGENDRESPHERE_HPP_

/// \file GaussLegendreSphere.hpp
/// \brief Gauss-Legendre product quadrature on the unit sphere, with the ring count fixed at compile time or run time.
///
/// The rule is Gauss-Legendre in z = cos(theta) on N rings times the trapezoidal rule in the longitude phi on 2N points
/// per ring, 2N^2 points in all. Like GaussLegendre<Scalar, N>, it is exact through degree 2N - 1: for every
/// polynomial in (x, y, z) of that degree, and so for every spherical harmonic of degree at most 2N - 1.
///
/// Points are (x, y, z) triples on the unit sphere, ring by ring from the north pole (+z) southward, and by increasing
/// phi = 2 pi k / (2N) from +x within a ring. For a sphere of radius r, scale the points by r and the weights by r^2;
/// the outward unit normals are the points themselves. Points and weights are correctly rounded to Scalar.
///
/// GaussLegendreSphere uses compile-time work that grows like N^2. At the time of writing, GCC 13.3.0 compiles for
/// N <= 142 whereas clang 18.1.8 compiles for N <= 41. Larger N need -fconstexpr-ops-limit (GCC) or -fconstexpr-steps
/// (clang). Its table holds 8N^2 scalars, so for large N or on the device prefer gauss_legendre_sphere_rule(n, points,
/// weights), which builds the same rule, bit for bit, at run time.

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <stdexcept>    // for std::invalid_argument, std::runtime_error
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/DoubleDouble.hpp>                  // for mundy::DoubleDouble
#include <mundy_math/impl/GaussLegendreSphereImpl.hpp>  // for mundy::impl::make_gauss_legendre_sphere_rule, ...
#include <mundy_utils/requires.hpp>                     // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>                 // for MUNDY_THROW_REQUIRE

namespace mundy {

/// \brief The Gauss-Legendre product rule on the unit sphere with N rings, exact through degree 2N - 1.
///
/// The points and weights are correctly rounded to Scalar and computed by the compiler, so every member is a constant
/// expression. Like GaussLegendre's rule, it integrates on the unit sphere; scale for other radii.
template <class Scalar, unsigned N>
class GaussLegendreSphere {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "GaussLegendreSphere: the tables are computed to double precision, so Scalar must be float or double.");
  static_assert(N >= 1, "GaussLegendreSphere: a rule needs at least one ring.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_rings = N;
  static constexpr unsigned num_points = 2 * N * N;

  /// \brief The points as (x, y, z) triples: ring by ring from the north pole, by increasing longitude within a ring.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> points() {
    return rule().points;
  }

  /// \brief The num_points weights, in the order of points().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, num_points> weights() {
    return rule().weights;
  }

  /// \brief sum_i w_i f(x_i, y_i, z_i), approximating the integral of f over the unit sphere.
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(const Function& f) {
    constexpr impl::GaussLegendreSphereRule<Scalar, N> r = rule();
    auto sum = r.weights[0] * f(r.points[0], r.points[1], r.points[2]);
    for (unsigned i = 1; i < num_points; ++i) {
      sum += r.weights[i] * f(r.points[3 * i], r.points[3 * i + 1], r.points[3 * i + 2]);
    }
    return sum;
  }

 private:
  /// \brief The rule, built once per (Scalar, N) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr impl::GaussLegendreSphereRule<Scalar, N> rule() {
    constexpr impl::GaussLegendreSphereRule<Scalar, N> r = impl::make_gauss_legendre_sphere_rule<Scalar, N>();
    static_assert(r.converged, "GaussLegendreSphere: Newton's method did not converge to every root of P_N.");
    return r;
  }
};

/// \brief The Gauss-Legendre product rule on the unit sphere with a run-time number of rings n, into std::vectors.
///
/// Resizes points to 3 * 2n^2 and weights to 2n^2 and fills them with the values GaussLegendreSphere<Scalar, n> would
/// hold.
template <class Scalar>
void gauss_legendre_sphere_rule(unsigned n, std::vector<Scalar>& points, std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "gauss_legendre_sphere_rule: rules are computed to double precision, so Scalar must be float or "
                "double.");
  MUNDY_THROW_REQUIRE(n >= 1, std::invalid_argument, "gauss_legendre_sphere_rule: a rule needs at least one ring.");
  const unsigned m = 2 * n;  // longitudes per ring
  std::vector<impl::SinCos<DoubleDouble>> longitudes(m);
  for (unsigned k = 0; k < m; ++k) {
    longitudes[k] = impl::sin_cos_of_turn_fraction<DoubleDouble>(k, m);
  }
  points.resize(3 * n * m);
  weights.resize(n * m);
  bool converged = true;
  for (unsigned j = 0; j < (n + 1) / 2; ++j) {
    converged = impl::write_gauss_legendre_sphere_ring_pair<Scalar>(n, j, longitudes, points, weights) && converged;
  }
  MUNDY_THROW_REQUIRE(converged, std::runtime_error,
                      "gauss_legendre_sphere_rule: Newton's method did not converge to every root of P_n.");
}

/// \brief The Gauss-Legendre product rule on the unit sphere with a run-time number of rings n, into rank-1 views.
///
/// Resizes points to 3 * 2n^2 and weights to 2n^2 and fills them with the values GaussLegendreSphere<Scalar, n> would
/// hold, one ring pair per thread on the views' execution space. Returns once the rule is complete.
template <class PointsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<WeightsView>)
void gauss_legendre_sphere_rule(unsigned n, PointsView& points, WeightsView& weights) {
  using scalar_t = typename PointsView::non_const_value_type;
  using execution_space = typename PointsView::execution_space;
  using memory_space = typename PointsView::memory_space;
  static_assert(PointsView::rank() == 1 && WeightsView::rank() == 1,
                "gauss_legendre_sphere_rule: views must be rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "gauss_legendre_sphere_rule: rules are computed to double precision, so Scalar must be float or "
                "double.");
  static_assert(std::is_same_v<scalar_t, typename WeightsView::value_type> &&
                    std::is_same_v<scalar_t, typename PointsView::value_type>,
                "gauss_legendre_sphere_rule: points and weights must be writable views of the same scalar type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "gauss_legendre_sphere_rule: weights must be accessible from the points' execution space.");
  MUNDY_THROW_REQUIRE(n >= 1, std::invalid_argument, "gauss_legendre_sphere_rule: a rule needs at least one ring.");

  const unsigned m = 2 * n;  // longitudes per ring
  Kokkos::resize(points, 3 * n * m);
  Kokkos::resize(weights, n * m);
  Kokkos::View<impl::SinCos<DoubleDouble>*, memory_space> longitudes("mundy::gauss_legendre_sphere_rule::longitudes",
                                                                      m);
  Kokkos::parallel_for(
      "mundy::gauss_legendre_sphere_rule::longitudes", Kokkos::RangePolicy<execution_space>(0, m),
      KOKKOS_LAMBDA(const unsigned k) { longitudes(k) = impl::sin_cos_of_turn_fraction<DoubleDouble>(k, m); });
  unsigned num_unconverged = 0;
  Kokkos::parallel_reduce(
      "mundy::gauss_legendre_sphere_rule", Kokkos::RangePolicy<execution_space>(0, (n + 1) / 2),
      KOKKOS_LAMBDA(const unsigned j, unsigned& count) {
        count += impl::write_gauss_legendre_sphere_ring_pair<scalar_t>(n, j, longitudes, points, weights) ? 0 : 1;
      },
      num_unconverged);
  MUNDY_THROW_REQUIRE(num_unconverged == 0, std::runtime_error,
                      "gauss_legendre_sphere_rule: Newton's method did not converge to every root of P_n.");
}

}  // namespace mundy

#endif  // MUNDY_MATH_GAUSSLEGENDRESPHERE_HPP_
