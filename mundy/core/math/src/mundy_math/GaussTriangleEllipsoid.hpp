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

#ifndef MUNDY_MATH_GAUSSTRIANGLEELLIPSOID_HPP_
#define MUNDY_MATH_GAUSSTRIANGLEELLIPSOID_HPP_

/// \file GaussTriangleEllipsoid.hpp
/// \brief Three-point Gauss triangle quadrature on an ellipsoid meshed from a refined octahedron or icosahedron, with
/// the refinement level fixed at compile time or run time.
///
/// The ellipsoid is meshed as in C. Pozrikidis's BEMLIB (trgl6_octa.f, trgl6_icos.f). Each face of the polyhedron is
/// split into four L times with every new edge midpoint projected onto the unit sphere, and the resulting F 4^L
/// six-node triangles (F = 8 or 20 faces) are stretched by (x, y, z) -> (x, (b/a) y, (c/a) z) onto the ellipsoid with
/// semi-axes 1, b/a, c/a. Each element is the quadratic patch through its six nodes, with each edge node placed along
/// its reference edge in proportion to its distances from the edge's ends (abc.f). The rule is the three-point Gauss
/// rule on every patch (gauss_trgl.f), 3 F 4^L points in all. It integrates over the patches rather than the ellipsoid
/// and converges at fourth order in the mesh spacing for smooth integrands. With b/a = c/a = 1 the ellipsoid is the
/// unit sphere.
///
/// Points are (x, y, z) triples, element by element in BEMLIB's order, and within an element nearest its vertices 0,
/// 1, 2 in turn. The points lie on the patches, within O(h^3) of the ellipsoid (O(h^4) of the sphere), so the normals
/// are the patches' outward unit normals. For semi-axes a, b, c, scale the points by a and the weights by a^2; BEMLIB's
/// volume-equivalent radius r gives a = r / (b/a c/a)^(1/3). Points, normals, and weights are correctly rounded to
/// Scalar.
///
/// GaussTriangleEllipsoid uses compile-time work that grows like 4^L. At the time of writing, GCC 13.3.0 compiles for
/// L <= 2 (octahedron) and L <= 1 (icosahedron) whereas clang 18.1.8 compiles for L = 0. Larger L need
/// -fconstexpr-ops-limit (GCC) or -fconstexpr-steps (clang). Its tables hold 21 F 4^L scalars, so for large L or on
/// the device prefer gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points, normals, weights), which
/// builds the same rule, bit for bit, at run time.

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <cmath>        // for std::isfinite
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/impl/GaussTriangleEllipsoidImpl.hpp>  // for mundy::impl::make_gauss_triangle_ellipsoid_rule, ...
#include <mundy_utils/requires.hpp>                        // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>                    // for MUNDY_THROW_REQUIRE

namespace mundy {

/// \brief The polyhedron whose refined faces mesh the sphere. Each enumerator's value is the polyhedron's face count.
enum class SpherePolyhedron : unsigned { octahedron = 8, icosahedron = 20 };

/// \brief The three-point Gauss triangle rule on the Polyhedron refined L times and stretched onto the ellipsoid with
/// semi-axes 1, BOverA, COverA, with outward unit normals.
///
/// The points, normals, and weights are correctly rounded to Scalar and computed by the compiler, so every member is a
/// constant expression. Like TrapezoidalTorus's rule, it integrates over a unit-scale surface; scale for other sizes.
template <class Scalar, SpherePolyhedron Polyhedron, unsigned L, double BOverA = 1.0, double COverA = 1.0>
class GaussTriangleEllipsoid {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "GaussTriangleEllipsoid: the tables are computed to double precision, so Scalar must be float or "
                "double.");
  static_assert(Polyhedron == SpherePolyhedron::octahedron || Polyhedron == SpherePolyhedron::icosahedron,
                "GaussTriangleEllipsoid: the polyhedron must be the octahedron or the icosahedron.");
  static_assert(L <= 12, "GaussTriangleEllipsoid: more than 12 refinements overflow the unsigned point indices.");
  static_assert(BOverA > 0.0 && COverA > 0.0, "GaussTriangleEllipsoid: the semi-axis ratios must be positive.");

 public:
  using value_type = Scalar;
  static constexpr SpherePolyhedron polyhedron = Polyhedron;
  static constexpr unsigned num_refinements = L;
  static constexpr double b_over_a = BOverA;
  static constexpr double c_over_a = COverA;
  static constexpr unsigned num_elements = static_cast<unsigned>(Polyhedron) << (2 * L);
  static constexpr unsigned num_points = 3 * num_elements;

  /// \brief The points as (x, y, z) triples: element by element in BEMLIB's order, three per element.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> points() {
    return rule().points;
  }

  /// \brief The outward unit normals of the patches at the points, as (x, y, z) triples in the order of points().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> normals() {
    return rule().normals;
  }

  /// \brief The num_points weights, in the order of points().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, num_points> weights() {
    return rule().weights;
  }

  /// \brief sum_i w_i f(x_i, y_i, z_i), approximating the integral of f over the ellipsoid.
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(const Function& f) {
    constexpr rule_t r = rule();
    auto sum = r.weights[0] * f(r.points[0], r.points[1], r.points[2]);
    for (unsigned i = 1; i < num_points; ++i) {
      sum += r.weights[i] * f(r.points[3 * i], r.points[3 * i + 1], r.points[3 * i + 2]);
    }
    return sum;
  }

 private:
  using rule_t = impl::GaussTriangleEllipsoidRule<Scalar, static_cast<unsigned>(Polyhedron), L>;

  /// \brief The rule, built once per (Scalar, Polyhedron, L, BOverA, COverA) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr rule_t rule() {
    constexpr rule_t r =
        impl::make_gauss_triangle_ellipsoid_rule<Scalar, static_cast<unsigned>(Polyhedron), L, BOverA, COverA>();
    return r;
  }
};

/// \brief The three-point Gauss triangle rule on the polyhedron refined l times and stretched onto the ellipsoid with
/// semi-axes 1, b_over_a, c_over_a, into std::vectors.
///
/// Resizes points and normals to 3 * 3 F 4^l and weights to 3 F 4^l, F the polyhedron's face count, and fills them with
/// the values GaussTriangleEllipsoid<Scalar, polyhedron, l, b_over_a, c_over_a> would hold.
template <class Scalar>
void gauss_triangle_ellipsoid_rule(SpherePolyhedron polyhedron, unsigned l, double b_over_a, double c_over_a,
                                   std::vector<Scalar>& points, std::vector<Scalar>& normals,
                                   std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "gauss_triangle_ellipsoid_rule: rules are computed to double precision, so Scalar must be float or "
                "double.");
  MUNDY_THROW_REQUIRE(polyhedron == SpherePolyhedron::octahedron || polyhedron == SpherePolyhedron::icosahedron,
                      std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: the polyhedron must be the octahedron or the icosahedron.");
  MUNDY_THROW_REQUIRE(l <= 12, std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: more than 12 refinements overflow the unsigned point indices.");
  MUNDY_THROW_REQUIRE(std::isfinite(b_over_a) && std::isfinite(c_over_a) && b_over_a > 0.0 && c_over_a > 0.0,
                      std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: the semi-axis ratios must be finite and positive.");
  const unsigned num_faces = static_cast<unsigned>(polyhedron);
  const unsigned num_elements = num_faces << (2 * l);
  points.resize(9 * num_elements);
  normals.resize(9 * num_elements);
  weights.resize(3 * num_elements);
  for (unsigned e = 0; e < num_elements; ++e) {
    impl::write_gauss_triangle_ellipsoid_element<Scalar>(num_faces, l, b_over_a, c_over_a, e, points, normals, weights);
  }
}

/// \brief The three-point Gauss triangle rule on the polyhedron refined l times and stretched onto the ellipsoid with
/// semi-axes 1, b_over_a, c_over_a, into rank-1 views.
///
/// Resizes points and normals to 3 * 3 F 4^l and weights to 3 F 4^l, F the polyhedron's face count, and fills them with
/// the values GaussTriangleEllipsoid<Scalar, polyhedron, l, b_over_a, c_over_a> would hold, one element per thread on
/// the views' execution space. Returns once the rule is complete.
template <class PointsView, class NormalsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<NormalsView>&& Kokkos::is_view_v<WeightsView>)
void gauss_triangle_ellipsoid_rule(SpherePolyhedron polyhedron, unsigned l, double b_over_a, double c_over_a,
                                   PointsView& points, NormalsView& normals, WeightsView& weights) {
  using scalar_t = typename PointsView::non_const_value_type;
  using execution_space = typename PointsView::execution_space;
  static_assert(PointsView::rank() == 1 && NormalsView::rank() == 1 && WeightsView::rank() == 1,
                "gauss_triangle_ellipsoid_rule: views must be rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "gauss_triangle_ellipsoid_rule: rules are computed to double precision, so Scalar must be float or "
                "double.");
  static_assert(std::is_same_v<scalar_t, typename PointsView::value_type> &&
                    std::is_same_v<scalar_t, typename NormalsView::value_type> &&
                    std::is_same_v<scalar_t, typename WeightsView::value_type>,
                "gauss_triangle_ellipsoid_rule: points, normals, and weights must be writable views of the same scalar "
                "type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename NormalsView::memory_space>::accessible &&
                    Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "gauss_triangle_ellipsoid_rule: normals and weights must be accessible from the points' execution "
                "space.");
  MUNDY_THROW_REQUIRE(polyhedron == SpherePolyhedron::octahedron || polyhedron == SpherePolyhedron::icosahedron,
                      std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: the polyhedron must be the octahedron or the icosahedron.");
  MUNDY_THROW_REQUIRE(l <= 12, std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: more than 12 refinements overflow the unsigned point indices.");
  MUNDY_THROW_REQUIRE(std::isfinite(b_over_a) && std::isfinite(c_over_a) && b_over_a > 0.0 && c_over_a > 0.0,
                      std::invalid_argument,
                      "gauss_triangle_ellipsoid_rule: the semi-axis ratios must be finite and positive.");

  const unsigned num_faces = static_cast<unsigned>(polyhedron);
  const unsigned num_elements = num_faces << (2 * l);
  Kokkos::resize(points, 9 * num_elements);
  Kokkos::resize(normals, 9 * num_elements);
  Kokkos::resize(weights, 3 * num_elements);
  const execution_space space;
  Kokkos::parallel_for(
      "mundy::gauss_triangle_ellipsoid_rule", Kokkos::RangePolicy<execution_space>(space, 0, num_elements),
      KOKKOS_LAMBDA(const unsigned e) {
        impl::write_gauss_triangle_ellipsoid_element<scalar_t>(num_faces, l, b_over_a, c_over_a, e, points, normals,
                                                               weights);
      });
  space.fence("mundy::gauss_triangle_ellipsoid_rule");
}

}  // namespace mundy

#endif  // MUNDY_MATH_GAUSSTRIANGLEELLIPSOID_HPP_
