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

/// \file UnitTestGaussLegendreSphere.cpp
/// \brief GaussLegendreSphere and gauss_legendre_sphere_rule (GaussLegendreSphere.hpp) against exact sphere integrals,
/// high-precision references, and the rule's layout.
///
/// Anchors:
///   - Exactness: int_{S^2} x^a y^b z^c dA = 2 G((a+1)/2) G((b+1)/2) G((c+1)/2) / G((a+b+c+3)/2) when a, b, c are all
///     even (G the gamma function) and 0 otherwise, for every a + b + c <= 2N - 1.
///   - 30-digit values of a point and weight on the northernmost ring for N = 8 and 37.
///   - Non-polynomial integrands: int exp(x - y) dA = 2 sqrt(2) pi sinh(sqrt(2)) (smooth, so spectral convergence) and
///     int max(0, x, x cos t + y sin t) dA = pi + (pi/2) sqrt((1 - cos t)^2 + sin^2 t) (a kink, so algebraic).
///
/// Every compile-time N stays small, so the test builds within any compiler's default constexpr limits.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/GaussLegendreSphere.hpp>

namespace mundy {

namespace {

//! \name Compile-time contracts
//@{

static_assert(GaussLegendreSphere<double, 3>::num_points == 18);
static_assert(GaussLegendreSphere<double, 3>::points()[1] == 0.0, "The first point of a ring lies at phi = 0.");
static_assert(abs(GaussLegendreSphere<double, 2>::integrate([](double, double, double z) { return z * z; }) -
                  4.0 * Kokkos::numbers::pi_v<double> / 3.0) < 1e-14,
              "integrate is a constant expression, and the 2-ring rule is exact for z^2.");
//@}

//! \name Helpers
//@{

/// \brief int_{S^2} x^a y^b z^c dA.
double monomial_sphere_integral(int a, int b, int c) {
  if (a % 2 == 1 || b % 2 == 1 || c % 2 == 1) {
    return 0.0;
  }
  return 2.0 * std::tgamma(0.5 * (a + 1)) * std::tgamma(0.5 * (b + 1)) * std::tgamma(0.5 * (c + 1)) /
         std::tgamma(0.5 * (a + b + c + 3));
}

/// \brief Expect the n-ring rule to integrate every monomial of degree <= 2n - 1 exactly.
template <class Scalar>
void expect_exact_for_polynomials(const Scalar* points, const Scalar* weights, unsigned n, double tolerance) {
  const int max_degree = 2 * static_cast<int>(n) - 1;
  const unsigned num_points = 2 * n * n;
  for (int a = 0; a <= max_degree; ++a) {
    for (int b = 0; a + b <= max_degree; ++b) {
      for (int c = 0; a + b + c <= max_degree; ++c) {
        double sum = 0.0;
        for (unsigned i = 0; i < num_points; ++i) {
          sum += weights[i] * std::pow(points[3 * i], a) * std::pow(points[3 * i + 1], b) *
                 std::pow(points[3 * i + 2], c);
        }
        EXPECT_NEAR(sum, monomial_sphere_integral(a, b, c), tolerance)
            << "n = " << n << ", x^" << a << " y^" << b << " z^" << c;
      }
    }
  }
}

template <unsigned N>
void expect_exact_for_polynomials(double tolerance) {
  constexpr auto points = GaussLegendreSphere<double, N>::points();
  constexpr auto weights = GaussLegendreSphere<double, N>::weights();
  expect_exact_for_polynomials(points.data(), weights.data(), N, tolerance);
}

/// \brief Expect unit points laid out ring by ring from the north, phi = 0 first, and mirror rings with z negated.
void expect_layout(const double* points, const double* weights, unsigned n) {
  const unsigned m = 2 * n;
  for (unsigned j = 0; j < n; ++j) {
    const unsigned mirror = n - 1 - j;
    for (unsigned k = 0; k < m; ++k) {
      const unsigned i = j * m + k;
      const unsigned i_mirror = mirror * m + k;
      const double x = points[3 * i];
      const double y = points[3 * i + 1];
      const double z = points[3 * i + 2];
      EXPECT_NEAR(x * x + y * y + z * z, 1.0, 4e-16) << "n = " << n << ", i = " << i;
      EXPECT_EQ(z, points[3 * j * m + 2]) << "z is constant on a ring; n = " << n << ", i = " << i;
      EXPECT_EQ(weights[i], weights[j * m]) << "weights are constant on a ring; n = " << n << ", i = " << i;
      EXPECT_GT(weights[i], 0.0);
      EXPECT_EQ(x, points[3 * i_mirror]);
      EXPECT_EQ(y, points[3 * i_mirror + 1]);
      if (j != mirror) {
        EXPECT_EQ(z, -points[3 * i_mirror + 2]);
      }
    }
    EXPECT_EQ(points[3 * j * m + 1], 0.0) << "Each ring starts at phi = 0.";
    EXPECT_GT(points[3 * j * m], 0.0) << "Each ring starts on the +x side.";
    if (j > 0) {
      EXPECT_LT(points[3 * j * m + 2], points[3 * (j - 1) * m + 2]) << "Rings run from north to south.";
    }
  }
  if (n % 2 == 1) {
    EXPECT_FALSE(std::signbit(points[3 * (n / 2) * m + 2])) << "The equator of an odd rule has z = +0.";
  }
}

/// \brief Expect gauss_legendre_sphere_rule(N, ...) to reproduce GaussLegendreSphere<double, N> bit for bit.
template <unsigned N>
void expect_runtime_rule_matches_compile_time_rule() {
  std::vector<double> points;
  std::vector<double> weights;
  gauss_legendre_sphere_rule(N, points, weights);
  constexpr auto points_compile_time = GaussLegendreSphere<double, N>::points();
  constexpr auto weights_compile_time = GaussLegendreSphere<double, N>::weights();
  ASSERT_EQ(weights.size(), 2 * N * N);
  for (unsigned i = 0; i < 6 * N * N; ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]) << "N = " << N << ", i = " << i;
    EXPECT_EQ(std::signbit(points[i]), std::signbit(points_compile_time[i])) << "N = " << N << ", i = " << i;
  }
  for (unsigned i = 0; i < 2 * N * N; ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]) << "N = " << N << ", i = " << i;
  }
}
//@}

TEST(GaussLegendreSphere, IntegratesPolynomialsOfDegreeUpTo2NMinus1Exactly) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_exact_for_polynomials<I + 1>(1e-13), ...);
  }(std::make_integer_sequence<unsigned, 8>{});
}

TEST(GaussLegendreSphere, IsLaidOutRingByRingFromTheNorth) {
  constexpr auto points7 = GaussLegendreSphere<double, 7>::points();
  constexpr auto weights7 = GaussLegendreSphere<double, 7>::weights();
  expect_layout(points7.data(), weights7.data(), 7);
  constexpr auto points8 = GaussLegendreSphere<double, 8>::points();
  constexpr auto weights8 = GaussLegendreSphere<double, 8>::weights();
  expect_layout(points8.data(), weights8.data(), 8);
}

TEST(GaussLegendreSphere, IsCorrectlyRounded) {
  using rule8_t = GaussLegendreSphere<double, 8>;
  EXPECT_EQ(rule8_t::points()[3], 2.577663491553599315412243101860e-1);
  EXPECT_EQ(rule8_t::points()[4], 1.067703177435486772533188200773e-1);
  EXPECT_EQ(rule8_t::points()[5], 9.602898564975362316835608685695e-1);
  EXPECT_EQ(rule8_t::weights()[1], 3.975235324293672957534384296125e-2);
}

TEST(GaussLegendreSphereRule, MatchesTheCompileTimeRule) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_runtime_rule_matches_compile_time_rule<I + 1>(), ...);
  }(std::make_integer_sequence<unsigned, 16>{});
}

TEST(GaussLegendreSphereRule, BuildsLargeRules) {
  std::vector<double> points;
  std::vector<double> weights;
  gauss_legendre_sphere_rule(37, points, weights);
  ASSERT_EQ(weights.size(), 2u * 37 * 37);
  EXPECT_EQ(points[9], 6.201507500092037086301266899280e-2);
  EXPECT_EQ(points[10], 1.614746963498689434069564753865e-2);
  EXPECT_EQ(points[11], 9.979445824779136489408030743174e-1);
  EXPECT_EQ(weights[3], 4.477242705737542520400998433168e-4);
  expect_layout(points.data(), weights.data(), 37);

  gauss_legendre_sphere_rule(20, points, weights);
  expect_exact_for_polynomials(points.data(), weights.data(), 20, 1e-13);
}

TEST(GaussLegendreSphereRule, ViewsMatchVectors) {
  for (const unsigned n : {1u, 2u, 7u, 20u}) {
    std::vector<double> points;
    std::vector<double> weights;
    gauss_legendre_sphere_rule(n, points, weights);

    Kokkos::View<double*> points_view("points", 0);
    Kokkos::View<double*> weights_view("weights", 0);
    gauss_legendre_sphere_rule(n, points_view, weights_view);
    const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
    const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
    ASSERT_EQ(points_host.extent(0), points.size());
    ASSERT_EQ(weights_host.extent(0), weights.size());
    for (size_t i = 0; i < points.size(); ++i) {
      EXPECT_EQ(points_host(i), points[i]) << "n = " << n << ", i = " << i;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
      EXPECT_EQ(weights_host(i), weights[i]) << "n = " << n << ", i = " << i;
    }
  }

  // Single precision rounds the same double-double values.
  std::vector<float> points;
  std::vector<float> weights;
  gauss_legendre_sphere_rule(4, points, weights);
  Kokkos::View<float*> points_view("points", 0);
  Kokkos::View<float*> weights_view("weights", 0);
  gauss_legendre_sphere_rule(4, points_view, weights_view);
  const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
  const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
  constexpr auto points_compile_time = GaussLegendreSphere<float, 4>::points();
  constexpr auto weights_compile_time = GaussLegendreSphere<float, 4>::weights();
  for (unsigned i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]);
    EXPECT_EQ(points_host(i), points_compile_time[i]);
  }
  for (unsigned i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]);
    EXPECT_EQ(weights_host(i), weights_compile_time[i]);
  }
}

TEST(GaussLegendreSphereRule, ConvergesForNonPolynomialIntegrands) {
  const auto integrate = [](unsigned n, const auto& f) {
    std::vector<double> points;
    std::vector<double> weights;
    gauss_legendre_sphere_rule(n, points, weights);
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
      sum += weights[i] * f(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    }
    return sum;
  };
  const double pi = Kokkos::numbers::pi_v<double>;

  const double smooth_exact = 2.0 * std::sqrt(2.0) * pi * std::sinh(std::sqrt(2.0));
  const auto smooth = [](double x, double y, double) { return std::exp(x - y); };
  EXPECT_NEAR(integrate(13, smooth), smooth_exact, 1e-12);

  const double t = pi / 4.0;
  const double kink_exact =
      pi + pi / 2.0 * std::sqrt((1.0 - std::cos(t)) * (1.0 - std::cos(t)) + std::sin(t) * std::sin(t));
  const auto kink = [t](double x, double y, double) { return std::max({0.0, x, x * std::cos(t) + y * std::sin(t)}); };
  EXPECT_NEAR(integrate(17, kink), kink_exact, 5e-3);
}

TEST(GaussLegendreSphereRule, RejectsEmptyRules) {
  std::vector<double> points;
  std::vector<double> weights;
  EXPECT_THROW(gauss_legendre_sphere_rule(0, points, weights), std::invalid_argument);
  Kokkos::View<double*> points_view("points", 0);
  Kokkos::View<double*> weights_view("weights", 0);
  EXPECT_THROW(gauss_legendre_sphere_rule(0, points_view, weights_view), std::invalid_argument);
}

}  // namespace

}  // namespace mundy
