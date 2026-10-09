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

/// \file UnitTestGaussTriangleEllipsoid.cpp
/// \brief GaussTriangleEllipsoid and gauss_triangle_ellipsoid_rule (GaussTriangleEllipsoid.hpp) against exact
/// identities, high-precision references, fourth-order convergence, and the rule's layout.
///
/// Anchors:
///   - Closure: sum_i w_i n_i = 0, since the patches close up and the three-point rule is exact for each patch's area
///     vector x_xi cross x_eta, a quadratic in (xi, eta).
///   - Symmetry: every mesh and ellipsoid is symmetric under x -> -x and the half turn about the x axis, so
///     sum_i w_i x_i = 0 and sum_i w_i x y = sum_i w_i x z = 0. On the sphere the polyhedron's rotations, which fix no
///     anisotropic second moment, also give sum_i w_i x_i x_i^T = (1/3) sum_i w_i |x_i|^2 I.
///   - 30-digit values of a point, normal, and weight for (octahedron, L = 0, b/a = 0.5, c/a = 2) and (icosahedron,
///     L = 6, b/a = 0.75, c/a = 1.5), from BEMLIB's trgl6, abc.f, and gauss_trgl.f formulas in 50-digit arithmetic.
///   - Fourth-order convergence to the area (Legendre's elliptic-integral formula) and the volume 4 pi (b/a) (c/a) / 3.
///
/// Every compile-time L stays small, so the test builds within any compiler's default constexpr limits.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/DoubleDouble.hpp>
#include <mundy_math/GaussTriangleEllipsoid.hpp>

namespace mundy {

namespace {

constexpr double pi = Kokkos::numbers::pi_v<double>;
constexpr double eps = std::numeric_limits<double>::epsilon();

//! \name Compile-time contracts
//@{

using ellipsoid_t = GaussTriangleEllipsoid<double, SpherePolyhedron::octahedron, 0, 0.5, 2.0>;
static_assert(ellipsoid_t::num_points == 24);
static_assert(GaussTriangleEllipsoid<double, SpherePolyhedron::icosahedron, 0>::num_points == 60);
static_assert(abs(ellipsoid_t::integrate([](double, double, double z) { return z; })) < 1e-14,
              "integrate is a constant expression, and the mesh's symmetry makes int z dA vanish.");
//@}

//! \name Helpers
//@{

/// \brief The vertices of face f of BEMLIB's polyhedron, transcribed from trgl6_octa.f and trgl6_icos.f (unnormalized),
/// stretched by (x, y, z) -> (x, b_over_a y, c_over_a z).
std::array<std::array<double, 3>, 3> polyhedron_face(SpherePolyhedron polyhedron, unsigned f, double b_over_a,
                                                     double c_over_a) {
  const double ru = 0.25 * std::sqrt(10.0 + 2.0 * std::sqrt(5.0));
  const double rm = 0.25 * (1.0 + std::sqrt(5.0));
  const double c2 = 2.0 * ru / std::sqrt(5.0);
  const double c4 = ru / std::sqrt(5.0);
  const double c5 = std::sqrt(ru * ru - rm * rm - c4 * c4);
  const double c7 = std::sqrt(ru * ru - c4 * c4 - 0.25);
  // clang-format off
  const double octahedron_vertices[6][3] = {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
  constexpr unsigned octahedron_faces[8][3] = {{0, 1, 2}, {3, 0, 2}, {3, 4, 0}, {0, 4, 1},
                                               {1, 5, 2}, {5, 3, 2}, {5, 4, 3}, {1, 4, 5}};
  const double icosahedron_vertices[12][3] = {{  0,   0,  ru}, {  0,  c2,  c4}, { rm,  c5,  c4},
                                              {0.5, -c7,  c4}, {-0.5, -c7, c4}, {-rm,  c5,  c4},
                                              {-0.5, c7, -c4}, {0.5,  c7, -c4}, { rm, -c5, -c4},
                                              {  0, -c2, -c4}, {-rm, -c5, -c4}, {  0,   0, -ru}};
  constexpr unsigned icosahedron_faces[20][3] = {{0, 2,  1}, {0, 3,  2}, {0, 4,  3}, {0,  5,  4}, { 0, 1,  5},
                                                 {1, 2,  7}, {2, 3,  8}, {3, 4,  9}, {4,  5, 10}, { 5, 1,  6},
                                                 {1, 7,  6}, {2, 8,  7}, {3, 9,  8}, {4, 10,  9}, { 5, 6, 10},
                                                 {6, 7, 11}, {7, 8, 11}, {8, 9, 11}, {9, 10, 11}, {10, 6, 11}};
  // clang-format on
  const bool octahedron = polyhedron == SpherePolyhedron::octahedron;
  std::array<std::array<double, 3>, 3> face;
  for (unsigned k = 0; k < 3; ++k) {
    const double* vertex =
        octahedron ? octahedron_vertices[octahedron_faces[f][k]] : icosahedron_vertices[icosahedron_faces[f][k]];
    face[k] = {vertex[0], b_over_a * vertex[1], c_over_a * vertex[2]};
  }
  return face;
}

std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// \brief The area of the ellipsoid with semi-axes a > b > c (Legendre's formula).
double ellipsoid_area(double a, double b, double c) {
  const double phi = std::acos(c / a);
  const double k = std::sqrt(a * a * (b * b - c * c) / (b * b * (a * a - c * c)));
  return 2.0 * pi * c * c + 2.0 * pi * a * b / std::sin(phi) *
                                (std::ellint_2(k, phi) * std::sin(phi) * std::sin(phi) +
                                 std::ellint_1(k, phi) * std::cos(phi) * std::cos(phi));
}

/// \brief sum_i term(i), formed and summed in double-double, and the scale sum_i |term(i)|.
///
/// Each term is then off only by the rounding of its rule entries (at most three, each within eps/2 relatively), so an
/// identity that holds for the exact rule holds here to within 2 eps times the scale.
template <class Term>
std::pair<double, double> exact_sum(unsigned num_points, const Term& term) {
  DoubleDouble sum = 0.0;
  double scale = 0.0;
  for (unsigned i = 0; i < num_points; ++i) {
    const DoubleDouble t = term(i);
    sum += t;
    scale += std::abs(t.hi());
  }
  return {sum.hi(), scale};
}

/// \brief Expect sum_i w_i n_i = 0.
void expect_closed(const double* normals, const double* weights, unsigned num_points) {
  for (unsigned d = 0; d < 3; ++d) {
    const auto [sum, scale] =
        exact_sum(num_points, [&](unsigned i) { return DoubleDouble(weights[i]) * normals[3 * i + d]; });
    EXPECT_NEAR(sum, 0.0, 2.0 * eps * scale) << "num_points = " << num_points << ", d = " << d;
  }
}

/// \brief Expect sum_i w_i x_i = 0 and sum_i w_i x y = sum_i w_i x z = 0.
void expect_symmetric(const double* points, const double* weights, unsigned num_points) {
  const auto x = [&](unsigned i, unsigned d) { return DoubleDouble(points[3 * i + d]); };
  for (unsigned d = 0; d < 3; ++d) {
    const auto [first, first_scale] = exact_sum(num_points, [&](unsigned i) { return weights[i] * x(i, d); });
    EXPECT_NEAR(first, 0.0, 2.0 * eps * first_scale) << "num_points = " << num_points << ", x_" << d;
  }
  for (unsigned e = 1; e < 3; ++e) {
    const auto [second, second_scale] =
        exact_sum(num_points, [&](unsigned i) { return weights[i] * x(i, 0) * x(i, e); });
    EXPECT_NEAR(second, 0.0, 2.0 * eps * second_scale) << "num_points = " << num_points << ", x_0 x_" << e;
  }
}

/// \brief Expect sum_i w_i x_i x_i^T = (1/3) sum_i w_i |x_i|^2 I.
void expect_isotropic(const double* points, const double* weights, unsigned num_points) {
  const auto x = [&](unsigned i, unsigned d) { return DoubleDouble(points[3 * i + d]); };
  const auto [trace, trace_scale] = exact_sum(
      num_points, [&](unsigned i) { return weights[i] * (x(i, 0) * x(i, 0) + x(i, 1) * x(i, 1) + x(i, 2) * x(i, 2)); });
  for (unsigned d = 0; d < 3; ++d) {
    for (unsigned e = d; e < 3; ++e) {
      const auto [second, second_scale] =
          exact_sum(num_points, [&](unsigned i) { return weights[i] * x(i, d) * x(i, e); });
      EXPECT_NEAR(second, d == e ? trace / 3.0 : 0.0, 2.0 * eps * trace_scale)
          << "num_points = " << num_points << ", x_" << d << " x_" << e;
    }
  }
}

/// \brief Expect outward unit normals and positive weights, each element's three points counterclockwise about its
/// normals, and the 3 * 4^l points of face f inside the stretched face's cone.
void expect_layout(const double* points, const double* normals, const double* weights, SpherePolyhedron polyhedron,
                   unsigned l, double b_over_a, double c_over_a) {
  const unsigned num_faces = static_cast<unsigned>(polyhedron);
  const unsigned points_per_face = 3u << (2 * l);
  const auto point = [&](unsigned i) { return std::array{points[3 * i], points[3 * i + 1], points[3 * i + 2]}; };
  const auto normal = [&](unsigned i) { return std::array{normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]}; };
  for (unsigned f = 0; f < num_faces; ++f) {
    const auto v = polyhedron_face(polyhedron, f, b_over_a, c_over_a);
    for (unsigned i = f * points_per_face; i < (f + 1) * points_per_face; ++i) {
      EXPECT_NEAR(dot(normal(i), normal(i)), 1.0, 4e-16) << "l = " << l << ", i = " << i;
      EXPECT_GT(dot(normal(i), point(i)), 0.0) << "The normals point outward; l = " << l << ", i = " << i;
      EXPECT_GT(weights[i], 0.0);
      for (unsigned k = 0; k < 3; ++k) {
        EXPECT_GT(dot(point(i), cross(v[k], v[(k + 1) % 3])), 0.0)
            << "Point " << i << " lies on face " << f << "; l = " << l;
      }
    }
  }
  for (unsigned i = 0; i < 3 * (num_faces << (2 * l)); i += 3) {
    std::array<double, 3> edge1;
    std::array<double, 3> edge2;
    std::array<double, 3> n;
    for (unsigned d = 0; d < 3; ++d) {
      edge1[d] = points[3 * (i + 1) + d] - points[3 * i + d];
      edge2[d] = points[3 * (i + 2) + d] - points[3 * i + d];
      n[d] = normals[3 * i + d] + normals[3 * (i + 1) + d] + normals[3 * (i + 2) + d];
    }
    EXPECT_GT(dot(cross(edge1, edge2), n), 0.0)
        << "An element's points run counterclockwise about its normals; l = " << l << ", element " << i / 3;
  }
}

template <SpherePolyhedron Polyhedron, unsigned L, double B, double C>
void expect_closed() {
  using rule_t = GaussTriangleEllipsoid<double, Polyhedron, L, B, C>;
  constexpr auto normals = rule_t::normals();
  constexpr auto weights = rule_t::weights();
  expect_closed(normals.data(), weights.data(), rule_t::num_points);
}

/// \brief Expect the symmetry of GaussTriangleEllipsoid<double, Polyhedron, L, B, C>, and its isotropy on the sphere.
template <SpherePolyhedron Polyhedron, unsigned L, double B, double C>
void expect_symmetric() {
  using rule_t = GaussTriangleEllipsoid<double, Polyhedron, L, B, C>;
  constexpr auto points = rule_t::points();
  constexpr auto weights = rule_t::weights();
  expect_symmetric(points.data(), weights.data(), rule_t::num_points);
  if (B == 1.0 && C == 1.0) {
    expect_isotropic(points.data(), weights.data(), rule_t::num_points);
  }
}

template <SpherePolyhedron Polyhedron, unsigned L, double B, double C>
void expect_layout() {
  using rule_t = GaussTriangleEllipsoid<double, Polyhedron, L, B, C>;
  constexpr auto points = rule_t::points();
  constexpr auto normals = rule_t::normals();
  constexpr auto weights = rule_t::weights();
  expect_layout(points.data(), normals.data(), weights.data(), Polyhedron, L, B, C);
}

/// \brief Expect gauss_triangle_ellipsoid_rule(Polyhedron, L, B, C, ...) to reproduce GaussTriangleEllipsoid<double,
/// Polyhedron, L, B, C> bit for bit.
template <SpherePolyhedron Polyhedron, unsigned L, double B, double C>
void expect_runtime_rule_matches_compile_time_rule() {
  using rule_t = GaussTriangleEllipsoid<double, Polyhedron, L, B, C>;
  std::vector<double> points;
  std::vector<double> normals;
  std::vector<double> weights;
  gauss_triangle_ellipsoid_rule(Polyhedron, L, B, C, points, normals, weights);
  constexpr auto points_compile_time = rule_t::points();
  constexpr auto normals_compile_time = rule_t::normals();
  constexpr auto weights_compile_time = rule_t::weights();
  ASSERT_EQ(weights.size(), rule_t::num_points);
  for (unsigned i = 0; i < 3 * rule_t::num_points; ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]) << "L = " << L << ", i = " << i;
    EXPECT_EQ(std::signbit(points[i]), std::signbit(points_compile_time[i])) << "L = " << L << ", i = " << i;
    EXPECT_EQ(normals[i], normals_compile_time[i]) << "L = " << L << ", i = " << i;
    EXPECT_EQ(std::signbit(normals[i]), std::signbit(normals_compile_time[i])) << "L = " << L << ", i = " << i;
  }
  for (unsigned i = 0; i < rule_t::num_points; ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]) << "L = " << L << ", i = " << i;
  }
}
//@}

TEST(GaussTriangleEllipsoid, WeightedNormalsSumToZero) {
  expect_closed<SpherePolyhedron::octahedron, 0, 1.0, 1.0>();
  expect_closed<SpherePolyhedron::icosahedron, 0, 1.0, 1.0>();
  expect_closed<SpherePolyhedron::octahedron, 0, 0.5, 2.0>();
  expect_closed<SpherePolyhedron::icosahedron, 0, 0.5, 2.0>();
}

TEST(GaussTriangleEllipsoid, HasTheMeshsSymmetry) {
  expect_symmetric<SpherePolyhedron::octahedron, 0, 1.0, 1.0>();
  expect_symmetric<SpherePolyhedron::icosahedron, 0, 1.0, 1.0>();
  expect_symmetric<SpherePolyhedron::octahedron, 0, 0.5, 2.0>();
  expect_symmetric<SpherePolyhedron::icosahedron, 0, 0.5, 2.0>();
}

TEST(GaussTriangleEllipsoid, IsLaidOutElementByElement) {
  expect_layout<SpherePolyhedron::octahedron, 0, 0.5, 2.0>();
  expect_layout<SpherePolyhedron::icosahedron, 0, 0.5, 2.0>();
}

TEST(GaussTriangleEllipsoid, IsCorrectlyRounded) {
  constexpr auto points = ellipsoid_t::points();
  constexpr auto normals = ellipsoid_t::normals();
  constexpr auto weights = ellipsoid_t::weights();
  EXPECT_EQ(points[39], 3.267706516541918305168061159154e-1);
  EXPECT_EQ(points[40], 1.987584906275703368447103042109e-1);
  EXPECT_EQ(points[41], -1.455330950746658361918808025741e+0);
  EXPECT_EQ(normals[39], 3.975872442238303115750297974374e-1);
  EXPECT_EQ(normals[40], 8.243490706066969210735834408850e-1);
  EXPECT_EQ(normals[41], -4.029553238516341092399957782356e-1);
  EXPECT_EQ(weights[13], 7.503961783324879533544694987352e-1);
}

TEST(GaussTriangleEllipsoidRule, MatchesTheCompileTimeRule) {
  expect_runtime_rule_matches_compile_time_rule<SpherePolyhedron::octahedron, 0, 1.0, 1.0>();
  expect_runtime_rule_matches_compile_time_rule<SpherePolyhedron::icosahedron, 0, 1.0, 1.0>();
  expect_runtime_rule_matches_compile_time_rule<SpherePolyhedron::octahedron, 0, 0.5, 2.0>();
  expect_runtime_rule_matches_compile_time_rule<SpherePolyhedron::icosahedron, 0, 0.5, 2.0>();
}

TEST(GaussTriangleEllipsoidRule, BuildsLargeRules) {
  std::vector<double> points;
  std::vector<double> normals;
  std::vector<double> weights;
  gauss_triangle_ellipsoid_rule(SpherePolyhedron::icosahedron, 6, 0.75, 1.5, points, normals, weights);
  ASSERT_EQ(weights.size(), 3u * 20 * 4096);
  EXPECT_EQ(points[501783], -4.165821381010303098362832516473e-1);
  EXPECT_EQ(points[501784], -6.045297114533678140384424675421e-1);
  EXPECT_EQ(points[501785], -6.306417744783993459622318184994e-1);
  EXPECT_EQ(normals[501783], -3.511895639664882397203751007855e-1);
  EXPECT_EQ(normals[501784], -9.059999358914880330046051870094e-1);
  EXPECT_EQ(normals[501785], -2.362837411368954926048706365539e-1);
  EXPECT_EQ(weights[167261], 6.472009524108669063835616792928e-5);
  expect_layout(points.data(), normals.data(), weights.data(), SpherePolyhedron::icosahedron, 6, 0.75, 1.5);
  expect_closed(normals.data(), weights.data(), weights.size());
  expect_symmetric(points.data(), weights.data(), weights.size());

  gauss_triangle_ellipsoid_rule(SpherePolyhedron::octahedron, 4, 1.0, 1.0, points, normals, weights);
  expect_layout(points.data(), normals.data(), weights.data(), SpherePolyhedron::octahedron, 4, 1.0, 1.0);
  expect_closed(normals.data(), weights.data(), weights.size());
  expect_isotropic(points.data(), weights.data(), weights.size());
}

TEST(GaussTriangleEllipsoidRule, ConvergesAtFourthOrder) {
  // Halving the mesh spacing cuts the error 16-fold, so from L = 2 the observed order rounds to 4; the relative error
  // falls below 1e-6 by L = 5 on the octahedron and L = 4 on the icosahedron.
  for (const auto& [polyhedron, finest, b_over_a, c_over_a, area_exact] :
       {std::tuple{SpherePolyhedron::octahedron, 5u, 1.0, 1.0, 4.0 * pi},
        std::tuple{SpherePolyhedron::icosahedron, 4u, 1.0, 1.0, 4.0 * pi},
        std::tuple{SpherePolyhedron::octahedron, 5u, 0.5, 2.0, ellipsoid_area(2.0, 1.0, 0.5)},
        std::tuple{SpherePolyhedron::icosahedron, 4u, 0.5, 2.0, ellipsoid_area(2.0, 1.0, 0.5)}}) {
    const double volume_exact = 4.0 * pi * b_over_a * c_over_a / 3.0;
    double area_error = 0.0;
    double volume_error = 0.0;
    for (unsigned l = 2; l <= finest; ++l) {
      std::vector<double> points;
      std::vector<double> normals;
      std::vector<double> weights;
      gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points, normals, weights);
      double area = 0.0;
      double volume = 0.0;
      for (size_t i = 0; i < weights.size(); ++i) {
        area += weights[i];
        volume += weights[i] *
                  (points[3 * i] * normals[3 * i] + points[3 * i + 1] * normals[3 * i + 1] +
                   points[3 * i + 2] * normals[3 * i + 2]) /
                  3.0;
      }
      const double previous_area_error = area_error;
      const double previous_volume_error = volume_error;
      area_error = std::abs(area - area_exact) / area_exact;
      volume_error = std::abs(volume - volume_exact) / volume_exact;
      if (l > 2) {
        EXPECT_EQ(std::lround(std::log2(previous_area_error / area_error)), 4)
            << "Area; b/a = " << b_over_a << ", c/a = " << c_over_a << ", l = " << l;
        EXPECT_EQ(std::lround(std::log2(previous_volume_error / volume_error)), 4)
            << "Volume; b/a = " << b_over_a << ", c/a = " << c_over_a << ", l = " << l;
      }
    }
    EXPECT_LT(area_error, 1e-6) << "Area; b/a = " << b_over_a << ", c/a = " << c_over_a;
    EXPECT_LT(volume_error, 1e-6) << "Volume; b/a = " << b_over_a << ", c/a = " << c_over_a;
  }
}

TEST(GaussTriangleEllipsoidRule, ViewsMatchVectors) {
  for (const auto polyhedron : {SpherePolyhedron::octahedron, SpherePolyhedron::icosahedron}) {
    for (const unsigned l : {0u, 1u, 3u}) {
      for (const auto& [b_over_a, c_over_a] : {std::pair{1.0, 1.0}, std::pair{0.5, 2.0}}) {
        std::vector<double> points;
        std::vector<double> normals;
        std::vector<double> weights;
        gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points, normals, weights);

        Kokkos::View<double*> points_view("points", 0);
        Kokkos::View<double*> normals_view("normals", 0);
        Kokkos::View<double*> weights_view("weights", 0);
        gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points_view, normals_view, weights_view);
        const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
        const auto normals_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, normals_view);
        const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
        ASSERT_EQ(points_host.extent(0), points.size());
        ASSERT_EQ(normals_host.extent(0), normals.size());
        ASSERT_EQ(weights_host.extent(0), weights.size());
        for (size_t i = 0; i < points.size(); ++i) {
          EXPECT_EQ(points_host(i), points[i]) << "l = " << l << ", i = " << i;
          EXPECT_EQ(normals_host(i), normals[i]) << "l = " << l << ", i = " << i;
        }
        for (size_t i = 0; i < weights.size(); ++i) {
          EXPECT_EQ(weights_host(i), weights[i]) << "l = " << l << ", i = " << i;
        }
      }
    }
  }

  // Single precision rounds the same double-double values.
  using rule_t = GaussTriangleEllipsoid<float, SpherePolyhedron::octahedron, 0, 0.5, 2.0>;
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<float> weights;
  gauss_triangle_ellipsoid_rule(SpherePolyhedron::octahedron, 0, 0.5, 2.0, points, normals, weights);
  Kokkos::View<float*> points_view("points", 0);
  Kokkos::View<float*> normals_view("normals", 0);
  Kokkos::View<float*> weights_view("weights", 0);
  gauss_triangle_ellipsoid_rule(SpherePolyhedron::octahedron, 0, 0.5, 2.0, points_view, normals_view, weights_view);
  const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
  const auto normals_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, normals_view);
  const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
  constexpr auto points_compile_time = rule_t::points();
  constexpr auto normals_compile_time = rule_t::normals();
  constexpr auto weights_compile_time = rule_t::weights();
  for (unsigned i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]);
    EXPECT_EQ(points_host(i), points_compile_time[i]);
    EXPECT_EQ(normals[i], normals_compile_time[i]);
    EXPECT_EQ(normals_host(i), normals_compile_time[i]);
  }
  for (unsigned i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]);
    EXPECT_EQ(weights_host(i), weights_compile_time[i]);
  }
}

TEST(GaussTriangleEllipsoidRule, RejectsInvalidRules) {
  std::vector<double> points;
  std::vector<double> normals;
  std::vector<double> weights;
  Kokkos::View<double*> points_view("points", 0);
  Kokkos::View<double*> normals_view("normals", 0);
  Kokkos::View<double*> weights_view("weights", 0);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [polyhedron, l, b_over_a, c_over_a] :
       {std::tuple{SpherePolyhedron::octahedron, 13u, 1.0, 1.0},
        std::tuple{SpherePolyhedron::icosahedron, 13u, 1.0, 1.0},
        std::tuple{static_cast<SpherePolyhedron>(12), 0u, 1.0, 1.0},
        std::tuple{SpherePolyhedron::octahedron, 0u, 0.0, 1.0}, std::tuple{SpherePolyhedron::octahedron, 0u, 1.0, -1.0},
        std::tuple{SpherePolyhedron::octahedron, 0u, inf, 1.0},
        std::tuple{SpherePolyhedron::octahedron, 0u, 1.0, nan}}) {
    EXPECT_THROW(gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points, normals, weights),
                 std::invalid_argument)
        << "faces = " << static_cast<unsigned>(polyhedron) << ", l = " << l << ", b/a = " << b_over_a
        << ", c/a = " << c_over_a;
    EXPECT_THROW(
        gauss_triangle_ellipsoid_rule(polyhedron, l, b_over_a, c_over_a, points_view, normals_view, weights_view),
        std::invalid_argument)
        << "faces = " << static_cast<unsigned>(polyhedron) << ", l = " << l << ", b/a = " << b_over_a
        << ", c/a = " << c_over_a;
  }
}

}  // namespace

}  // namespace mundy
