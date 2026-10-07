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

#ifndef MUNDY_MATH_TRAPEZOIDALTORUS_HPP_
#define MUNDY_MATH_TRAPEZOIDALTORUS_HPP_

/// \file TrapezoidalTorus.hpp
/// \brief Trapezoidal product quadrature on a torus, with the point counts fixed at compile time or run time.
///
/// The torus has minor radius 1 and major radius a > 1, its aspect ratio: the points ((a + cos theta) cos phi,
/// (a + cos theta) sin phi, sin theta) for the poloidal angle theta around the tube and the toroidal angle phi around
/// the z axis, with area element (a + cos theta) dtheta dphi. The rule is the trapezoidal rule in both angles: N
/// equispaced theta times M equispaced phi, N M points in all. Both angles are periodic, so the rule converges
/// spectrally for smooth integrands, and it is exact for every polynomial in (x, y, z) of degree at most
/// min(N - 2, M - 1).
///
/// Points are (x, y, z) triples, ring by ring in theta = 2 pi i / N from the outer equator (theta = 0) over the top,
/// and by increasing phi = 2 pi j / M from +x within a ring. For a torus with minor radius r, scale the points by r and
/// the weights by r^2; the outward unit normal at a point is (cos theta cos phi, cos theta sin phi, sin theta). Points
/// and weights are correctly rounded to Scalar.
///
/// TrapezoidalTorus uses compile-time work that grows like N M. At the time of writing, GCC 13.3.0 compiles for
/// N = M <= 238 whereas clang 18.1.8 compiles for N = M <= 64. Larger rules need -fconstexpr-ops-limit (GCC) or
/// -fconstexpr-steps (clang). Its table holds 4 N M scalars, so for large rules or on the device prefer
/// trapezoidal_torus_rule(n, m, aspect_ratio, points, weights), which builds the same rule, bit for bit, at run time.

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <cmath>        // for std::isfinite
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/DoubleDouble.hpp>               // for mundy::DoubleDouble
#include <mundy_math/impl/TrapezoidalTorusImpl.hpp>  // for mundy::impl::make_trapezoidal_torus_rule, ...
#include <mundy_utils/requires.hpp>                  // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_REQUIRE

namespace mundy {

/// \brief The N x M trapezoidal rule on the torus with minor radius 1 and major radius AspectRatio.
///
/// The points and weights are correctly rounded to Scalar and computed by the compiler, so every member is a constant
/// expression. Like GaussLegendreSphere's rule, it integrates over a unit-scale surface; scale for other minor radii.
template <class Scalar, unsigned N, unsigned M, double AspectRatio>
class TrapezoidalTorus {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "TrapezoidalTorus: the tables are computed to double precision, so Scalar must be float or double.");
  static_assert(N >= 1 && M >= 1, "TrapezoidalTorus: a rule needs at least one point in each angle.");
  static_assert(AspectRatio > 1.0, "TrapezoidalTorus: the major radius must exceed the minor radius 1.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_poloidal = N;
  static constexpr unsigned num_toroidal = M;
  static constexpr unsigned num_points = N * M;
  static constexpr double aspect_ratio = AspectRatio;

  /// \brief The points as (x, y, z) triples, ring by ring in theta from the outer equator, by increasing phi in a ring.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> points() {
    return rule().points;
  }

  /// \brief The num_points weights, in the order of points().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, num_points> weights() {
    return rule().weights;
  }

  /// \brief sum_i w_i f(x_i, y_i, z_i), approximating the integral of f over the torus.
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(const Function& f) {
    constexpr impl::TrapezoidalTorusRule<Scalar, N, M> r = rule();
    auto sum = r.weights[0] * f(r.points[0], r.points[1], r.points[2]);
    for (unsigned i = 1; i < num_points; ++i) {
      sum += r.weights[i] * f(r.points[3 * i], r.points[3 * i + 1], r.points[3 * i + 2]);
    }
    return sum;
  }

 private:
  /// \brief The rule, built once per (Scalar, N, M, AspectRatio) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr impl::TrapezoidalTorusRule<Scalar, N, M> rule() {
    constexpr impl::TrapezoidalTorusRule<Scalar, N, M> r =
        impl::make_trapezoidal_torus_rule<Scalar, N, M, AspectRatio>();
    return r;
  }
};

/// \brief The n x m trapezoidal rule on the torus with minor radius 1 and major radius aspect_ratio, into std::vectors.
///
/// Resizes points to 3 n m and weights to n m and fills them with the values TrapezoidalTorus<Scalar, n, m,
/// aspect_ratio> would hold.
template <class Scalar>
void trapezoidal_torus_rule(unsigned n, unsigned m, double aspect_ratio, std::vector<Scalar>& points,
                            std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "trapezoidal_torus_rule: rules are computed to double precision, so Scalar must be float or double.");
  MUNDY_THROW_REQUIRE(n >= 1 && m >= 1, std::invalid_argument,
                      "trapezoidal_torus_rule: a rule needs at least one point in each angle.");
  MUNDY_THROW_REQUIRE(std::isfinite(aspect_ratio) && aspect_ratio > 1.0, std::invalid_argument,
                      "trapezoidal_torus_rule: the major radius must be finite and exceed the minor radius 1.");
  std::vector<impl::SinCos<DoubleDouble>> toroidal(m);
  for (unsigned j = 0; j < m; ++j) {
    toroidal[j] = impl::sin_cos_of_turn_fraction<DoubleDouble>(j, m);
  }
  points.resize(3 * n * m);
  weights.resize(n * m);
  for (unsigned i = 0; i < n; ++i) {
    impl::write_trapezoidal_torus_ring<Scalar>(n, m, aspect_ratio, i, toroidal, points, weights);
  }
}

/// \brief The n x m trapezoidal rule on the torus with minor radius 1 and major radius aspect_ratio, into rank-1 views.
///
/// Resizes points to 3 n m and weights to n m and fills them with the values TrapezoidalTorus<Scalar, n, m,
/// aspect_ratio> would hold, one ring per thread on the views' execution space. Returns once the rule is complete.
template <class PointsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<WeightsView>)
void trapezoidal_torus_rule(unsigned n, unsigned m, double aspect_ratio, PointsView& points, WeightsView& weights) {
  using scalar_t = typename PointsView::non_const_value_type;
  using execution_space = typename PointsView::execution_space;
  using memory_space = typename PointsView::memory_space;
  static_assert(PointsView::rank() == 1 && WeightsView::rank() == 1, "trapezoidal_torus_rule: views must be rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "trapezoidal_torus_rule: rules are computed to double precision, so Scalar must be float or double.");
  static_assert(std::is_same_v<scalar_t, typename WeightsView::value_type> &&
                    std::is_same_v<scalar_t, typename PointsView::value_type>,
                "trapezoidal_torus_rule: points and weights must be writable views of the same scalar type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "trapezoidal_torus_rule: weights must be accessible from the points' execution space.");
  MUNDY_THROW_REQUIRE(n >= 1 && m >= 1, std::invalid_argument,
                      "trapezoidal_torus_rule: a rule needs at least one point in each angle.");
  MUNDY_THROW_REQUIRE(std::isfinite(aspect_ratio) && aspect_ratio > 1.0, std::invalid_argument,
                      "trapezoidal_torus_rule: the major radius must be finite and exceed the minor radius 1.");

  Kokkos::resize(points, 3 * n * m);
  Kokkos::resize(weights, n * m);
  const execution_space space;
  Kokkos::View<impl::SinCos<DoubleDouble>*, memory_space> toroidal("mundy::trapezoidal_torus_rule::toroidal", m);
  Kokkos::parallel_for(
      "mundy::trapezoidal_torus_rule::toroidal", Kokkos::RangePolicy<execution_space>(space, 0, m),
      KOKKOS_LAMBDA(const unsigned j) { toroidal(j) = impl::sin_cos_of_turn_fraction<DoubleDouble>(j, m); });
  Kokkos::parallel_for(
      "mundy::trapezoidal_torus_rule", Kokkos::RangePolicy<execution_space>(space, 0, n),
      KOKKOS_LAMBDA(const unsigned i) {
        impl::write_trapezoidal_torus_ring<scalar_t>(n, m, aspect_ratio, i, toroidal, points, weights);
      });
  space.fence("mundy::trapezoidal_torus_rule");
}

}  // namespace mundy

#endif  // MUNDY_MATH_TRAPEZOIDALTORUS_HPP_
