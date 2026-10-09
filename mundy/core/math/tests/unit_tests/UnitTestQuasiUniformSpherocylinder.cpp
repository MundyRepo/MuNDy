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

/// \file UnitTestQuasiUniformSpherocylinder.cpp
/// \brief QuasiUniformSpherocylinder and quasi_uniform_spherocylinder_rule (QuasiUniformSpherocylinder.hpp) against exact
/// spherocylinder integrals, high-precision references, and the rule's layout.
///
/// Anchors:
///   - Exact surface area 2 pi (L + 2), and analytic monomial moments through degree two for unit radius and
///     cylindrical length L. The quasi-uniform rule converges algebraically rather than integrating polynomials exactly.
///   - 30-digit values of a cap point and weight for Ntheta = 8 and 37, L = 4 (the latter crosses a Fibonacci block).
///   - Non-polynomial integrands: int exp(z) dA = 4 pi sinh(1 + L/2) (smooth) and
///     int max(0, x, x cos t + y sin t) dA = (2L + pi) (1 + sin(t/2)) for 0 <= t <= pi (a kink).
///
/// Every compile-time Ntheta stays small, so the test builds within any compiler's default constexpr limits.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/QuasiUniformSpherocylinder.hpp>

namespace mundy {

namespace {

//! \name Compile-time contracts
//@{

static_assert(impl::compute_num_cylinder_rings(32, 4.0) == 20);
static_assert(impl::compute_num_cap_points(32) == 163);
constexpr auto step = impl::golden_angle_sin_cos();

static_assert(QuasiUniformSpherocylinder<double, 8, 4.0>::num_points == 60);
static_assert(QuasiUniformSpherocylinder<double, 8, 0.0>::num_cylinder_rings == 0);
static_assert(QuasiUniformSpherocylinder<double, 8, 4.0>::points()[1] == 0.0,
              "The first point of an unstaggered cylinder ring lies at theta = 0.");
static_assert(QuasiUniformSpherocylinder<double, 8, 4.0>::integrate([](double, double, double) { return 1.0; }) >
                  12.0 * Kokkos::numbers::pi_v<double> - 1e-13 &&
                  QuasiUniformSpherocylinder<double, 8, 4.0>::integrate([](double, double, double) { return 1.0; }) <
                      12.0 * Kokkos::numbers::pi_v<double> + 1e-13,
              "integrate is a constant expression and the surface area is exact to rounding.");
//@}

//! \name Helpers
//@{

/// \brief int_{spherocylinder} x^a y^b z^c dA for a + b + c <= 2, with unit radius and cylinder length L.
double monomial_spherocylinder_integral(int a, int b, int c, double L) {
  const double pi = Kokkos::numbers::pi_v<double>;
  if (a == 0 && b == 0 && c == 0) return 2.0 * pi * (L + 2.0);
  if (a == 2 && b == 0 && c == 0) return pi * L + 4.0 * pi / 3.0;
  if (a == 0 && b == 2 && c == 0) return pi * L + 4.0 * pi / 3.0;
  if (a == 0 && b == 0 && c == 2) return pi * L * L * L / 6.0 + pi * L * L + 2.0 * pi * L + 4.0 * pi / 3.0;
  return 0.0;  // The remaining monomials through degree two vanish by symmetry.
}

/// \brief Expect the Ntheta-resolution rule to integrate low-degree monomials with second-order accuracy.
template <class Scalar>
void expect_accurate_for_polynomials(const Scalar* points, const Scalar* weights, unsigned ntheta, double L,
                                    double tolerance) {
  const int max_degree = 2;
  const unsigned nz = impl::compute_num_cylinder_rings(ntheta, L);
  const unsigned ncap = impl::compute_num_cap_points(ntheta);
  const unsigned num_points = ntheta * nz + 2 * ncap;
  for (int a = 0; a <= max_degree; ++a) {
    for (int b = 0; a + b <= max_degree; ++b) {
      for (int c = 0; a + b + c <= max_degree; ++c) {
        double sum = 0.0;
        for (unsigned i = 0; i < num_points; ++i) {
          sum += weights[i] * std::pow(points[3 * i], a) * std::pow(points[3 * i + 1], b) *
                 std::pow(points[3 * i + 2], c);
        }
        EXPECT_NEAR(sum, monomial_spherocylinder_integral(a, b, c, L), tolerance)
            << "Ntheta = " << ntheta << ", L = " << L << ", x^" << a << " y^" << b << " z^" << c;
      }
    }
  }
}

template <unsigned Ntheta>
void expect_accurate_for_polynomials(double tolerance) {
  constexpr auto points = QuasiUniformSpherocylinder<double, Ntheta, 4.0>::points();
  constexpr auto weights = QuasiUniformSpherocylinder<double, Ntheta, 4.0>::weights();
  expect_accurate_for_polynomials(points.data(), weights.data(), Ntheta, 4.0, tolerance);
}

/// \brief Expect unit cylinder rings ordered from -z to +z, followed by mirrored north/south Fibonacci endcaps.
void expect_layout(const double* points, const double* weights, unsigned ntheta, double L) {
  const unsigned nz = impl::compute_num_cylinder_rings(ntheta, L);
  const unsigned ncap = impl::compute_num_cap_points(ntheta);
  const unsigned offset = nz * ntheta;
  const double pi = Kokkos::numbers::pi_v<double>;
  for (unsigned j = 0; j < nz; ++j) {
    for (unsigned k = 0; k < ntheta; ++k) {
      const unsigned i = j * ntheta + k;
      const double x = points[3 * i];
      const double y = points[3 * i + 1];
      const double z = points[3 * i + 2];
      const double theta = 2.0 * pi * (k + 0.5 * (j & 1u)) / ntheta;
      EXPECT_NEAR(x * x + y * y, 1.0, 4e-15) << "Ntheta = " << ntheta << ", i = " << i;
      EXPECT_NEAR(x, std::cos(theta), 4e-15) << "Ntheta = " << ntheta << ", i = " << i;
      EXPECT_NEAR(y, std::sin(theta), 4e-15) << "Ntheta = " << ntheta << ", i = " << i;
      EXPECT_EQ(z, points[3 * j * ntheta + 2]) << "z is constant on a ring; i = " << i;
      EXPECT_NEAR(z, L * ((j + 0.5) / nz - 0.5), 1e-14) << "i = " << i;
      EXPECT_EQ(weights[i], weights[j * ntheta]) << "weights are constant on a ring; i = " << i;
      EXPECT_NEAR(weights[i], 2.0 * pi * L / (ntheta * nz), 1e-14) << "i = " << i;
      EXPECT_GT(weights[i], 0.0);
    }
    if (j > 0) {
      EXPECT_GT(points[3 * j * ntheta + 2], points[3 * (j - 1) * ntheta + 2])
          << "Cylinder rings run from -z to +z.";
    }
  }
  for (unsigned k = 0; k < ncap; ++k) {
    const unsigned north = offset + k;
    const unsigned south = offset + ncap + k;
    const double x = points[3 * north];
    const double y = points[3 * north + 1];
    const double z = points[3 * north + 2];
    const double mu = (k + 0.5) / ncap;
    EXPECT_NEAR(x * x + y * y + (z - L / 2.0) * (z - L / 2.0), 1.0, 4e-15)
        << "Ntheta = " << ntheta << ", cap point = " << k;
    EXPECT_NEAR(z, L / 2.0 + mu, 1e-14) << "cap point = " << k;
    const double theta = k * pi * (3.0 - std::sqrt(5.0));
    const double rho = std::sqrt(1.0 - mu * mu);
    EXPECT_NEAR(x, rho * std::cos(theta), 1e-12) << "golden-angle longitude; cap point = " << k;
    EXPECT_NEAR(y, rho * std::sin(theta), 1e-12) << "golden-angle longitude; cap point = " << k;
    EXPECT_NEAR(weights[north], 2.0 * pi / ncap, 2e-15) << "equal-area cap weight; cap point = " << k;
    EXPECT_EQ(weights[north], weights[offset]) << "north cap weights are equal";
    EXPECT_EQ(weights[south], weights[offset]) << "both caps have the same weights";
    EXPECT_GT(weights[north], 0.0);
    EXPECT_EQ(x, points[3 * south]);
    EXPECT_EQ(y, points[3 * south + 1]);
    EXPECT_EQ(z, -points[3 * south + 2]);
    if (k > 0) {
      EXPECT_GT(z, points[3 * (north - 1) + 2]) << "Cap nodes run from equator to pole.";
    }
  }
}

/// \brief Expect quasi_uniform_spherocylinder_rule(Ntheta, L, ...) to reproduce the compile-time rule bit for bit.
template <unsigned Ntheta, double L = 4.0>
void expect_runtime_rule_matches_compile_time_rule() {
  std::vector<double> points;
  std::vector<double> weights;
  quasi_uniform_spherocylinder_rule(Ntheta, L, points, weights);
  constexpr auto points_compile_time = QuasiUniformSpherocylinder<double, Ntheta, L>::points();
  constexpr auto weights_compile_time = QuasiUniformSpherocylinder<double, Ntheta, L>::weights();
  ASSERT_EQ(weights.size(), weights_compile_time.size());
  ASSERT_EQ(points.size(), points_compile_time.size());
  for (size_t i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]) << "Ntheta = " << Ntheta << ", i = " << i;
    EXPECT_EQ(std::signbit(points[i]), std::signbit(points_compile_time[i])) << "Ntheta = " << Ntheta << ", i = " << i;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]) << "Ntheta = " << Ntheta << ", i = " << i;
  }
}
//@}

TEST(QuasiUniformSpherocylinder, IntegratesLowDegreePolynomialsWithSecondOrderAccuracy) {
  // For L = 4, the largest quadratic-moment error is the axial midpoint error in int z^2 dA.
  expect_accurate_for_polynomials<3>(95.0 / (3 * 3));
  expect_accurate_for_polynomials<8>(95.0 / (8 * 8));
  expect_accurate_for_polynomials<16>(95.0 / (16 * 16));
}

TEST(QuasiUniformSpherocylinder, IsLaidOutCylinderThenNorthAndSouthCaps) {
  constexpr auto points7 = QuasiUniformSpherocylinder<double, 7, 4.0>::points();
  constexpr auto weights7 = QuasiUniformSpherocylinder<double, 7, 4.0>::weights();
  expect_layout(points7.data(), weights7.data(), 7, 4.0);
  constexpr auto points8 = QuasiUniformSpherocylinder<double, 8, 4.0>::points();
  constexpr auto weights8 = QuasiUniformSpherocylinder<double, 8, 4.0>::weights();
  expect_layout(points8.data(), weights8.data(), 8, 4.0);
}

TEST(QuasiUniformSpherocylinder, IsCorrectlyRounded) {
  using rule8_t = QuasiUniformSpherocylinder<double, 8, 4.0>;
  constexpr unsigned north8 = rule8_t::num_cylinder_rings * 8 + 3;
  EXPECT_EQ(rule8_t::points()[3 * north8], 5.699549203441188303205705190952e-1);
  EXPECT_EQ(rule8_t::points()[3 * north8 + 1], 7.434052655016166611479771176675e-1);
  EXPECT_EQ(rule8_t::points()[3 * north8 + 2], 2.350000000000000000000000000000e+0);
  EXPECT_EQ(rule8_t::weights()[north8], 6.283185307179586476925286766559e-1);
}

TEST(QuasiUniformSpherocylinderRule, MatchesTheCompileTimeRule) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_runtime_rule_matches_compile_time_rule<I + 3>(), ...);
  }(std::make_integer_sequence<unsigned, 14>{});
  expect_runtime_rule_matches_compile_time_rule<8, 0.0>();
}

TEST(QuasiUniformSpherocylinderRule, BuildsLargeRules) {
  std::vector<double> points;
  std::vector<double> weights;
  quasi_uniform_spherocylinder_rule(37, 4.0, points, weights);
  ASSERT_EQ(weights.size(), 1324u);
  const unsigned north = 24u * 37u + 33u;  // Crosses the first 32-point Fibonacci block.
  EXPECT_EQ(points[3 * north], -7.812323644263196173943745326662e-1);
  EXPECT_EQ(points[3 * north + 1], -6.050302541706300601330350147927e-1);
  EXPECT_EQ(points[3 * north + 2], 2.153669724770642201834862385321e+0);
  EXPECT_EQ(weights[north], 2.882195095036507558222608608513e-2);
  expect_layout(points.data(), weights.data(), 37, 4.0);

  quasi_uniform_spherocylinder_rule(32, 4.0, points, weights);
  expect_accurate_for_polynomials(points.data(), weights.data(), 32, 4.0, 95.0 / (32 * 32));
}

TEST(QuasiUniformSpherocylinderRule, ViewsMatchVectors) {
  for (const unsigned ntheta : {3u, 4u, 7u, 20u}) {
    for (const double L : {0.0, 4.0}) {
      std::vector<double> points;
      std::vector<double> weights;
      quasi_uniform_spherocylinder_rule(ntheta, L, points, weights);

      Kokkos::View<double*> points_view("points", 0);
      Kokkos::View<double*> weights_view("weights", 0);
      quasi_uniform_spherocylinder_rule(ntheta, L, points_view, weights_view);
      const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
      const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
      ASSERT_EQ(points_host.extent(0), points.size());
      ASSERT_EQ(weights_host.extent(0), weights.size());
      for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_EQ(points_host(i), points[i]) << "Ntheta = " << ntheta << ", L = " << L << ", i = " << i;
      }
      for (size_t i = 0; i < weights.size(); ++i) {
        EXPECT_EQ(weights_host(i), weights[i]) << "Ntheta = " << ntheta << ", L = " << L << ", i = " << i;
      }
    }
  }

  // Single precision rounds the same double-double values.
  std::vector<float> points;
  std::vector<float> weights;
  quasi_uniform_spherocylinder_rule(8, 4.0, points, weights);
  Kokkos::View<float*> points_view("points", 0);
  Kokkos::View<float*> weights_view("weights", 0);
  quasi_uniform_spherocylinder_rule(8, 4.0, points_view, weights_view);
  const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
  const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
  constexpr auto points_compile_time = QuasiUniformSpherocylinder<float, 8, 4.0>::points();
  constexpr auto weights_compile_time = QuasiUniformSpherocylinder<float, 8, 4.0>::weights();
  for (size_t i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]);
    EXPECT_EQ(points_host(i), points_compile_time[i]);
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]);
    EXPECT_EQ(weights_host(i), weights_compile_time[i]);
  }
}

TEST(QuasiUniformSpherocylinderRule, ConvergesForNonPolynomialIntegrands) {
  const double L = 4.0;
  const auto integrate = [L](unsigned ntheta, const auto& f) {
    std::vector<double> points;
    std::vector<double> weights;
    quasi_uniform_spherocylinder_rule(ntheta, L, points, weights);
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
      sum += weights[i] * f(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    }
    return sum;
  };
  const double pi = Kokkos::numbers::pi_v<double>;

  const double smooth_exact = 4.0 * pi * std::sinh(1.0 + L / 2.0);
  const auto smooth = [](double, double, double z) { return std::exp(z); };
  const double smooth_error32 = std::abs(integrate(32, smooth) - smooth_exact);
  const double smooth_error64 = std::abs(integrate(64, smooth) - smooth_exact);
  EXPECT_LT(smooth_error32, 0.085);
  EXPECT_LT(smooth_error64, 0.30 * smooth_error32);

  const double t = pi / 4.0;
  const double kink_exact = (2.0 * L + pi) * (1.0 + std::sin(t / 2.0));
  const auto kink = [t](double x, double y, double) { return std::max({0.0, x, x * std::cos(t) + y * std::sin(t)}); };
  EXPECT_NEAR(integrate(37, kink), kink_exact, 1.2e-2);
}

TEST(QuasiUniformSpherocylinderRule, InvalidRules) {
  std::vector<double> points;
  std::vector<double> weights;
  Kokkos::View<double*> points_view("points", 0);
  Kokkos::View<double*> weights_view("weights", 0);
  for (const unsigned ntheta : {0u, 1u, 2u}) {
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(ntheta, 4.0, points, weights), std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(ntheta, 4.0, points_view, weights_view), std::invalid_argument);
  }
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, -1, points, weights), std::invalid_argument);
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, -1, points_view, weights_view), std::invalid_argument);
}

TEST(QuasiUniformSpherocylinderRule, ZeroCylinderLengthProducesSphere) {
  std::vector<double> points;
  std::vector<double> weights;
  quasi_uniform_spherocylinder_rule(20, 0.0, points, weights);
  constexpr unsigned ncap = 64;  // round(20^2 / (2 pi)), independent of the count helper.
  ASSERT_EQ(weights.size(), 2u * ncap);
  EXPECT_EQ(points.size(), 6u * ncap);
  expect_layout(points.data(), weights.data(), 20, 0.0);
  double area = 0.0;
  for (double weight : weights) area += weight;
  EXPECT_NEAR(area, 4.0 * Kokkos::numbers::pi_v<double>, 1e-13);
}

TEST(QuasiUniformSpherocylinderRule, StaggersAdjacentCylinderRings) {
  std::vector<double> points;
  std::vector<double> weights;
  quasi_uniform_spherocylinder_rule(16, 4.0, points, weights);
  ASSERT_EQ(weights.size(), 242u);  // 10 cylinder rings * 16 nodes + 2 caps * 41 nodes.
  const double theta = Kokkos::numbers::pi_v<double> / 16.0;
  EXPECT_EQ(points[0], 1.0);
  EXPECT_EQ(points[1], 0.0);
  EXPECT_NEAR(points[3 * 16], std::cos(theta), 4e-16);
  EXPECT_NEAR(points[3 * 16 + 1], std::sin(theta), 4e-16);
  EXPECT_NEAR(points[3 * 32], 1.0, 4e-16);
  EXPECT_NEAR(points[3 * 32 + 1], 0.0, 4e-16);
  EXPECT_EQ(weights[0], weights[16]);
  EXPECT_EQ(weights[16], weights[32]);
}

}  // namespace

}  // namespace mundy