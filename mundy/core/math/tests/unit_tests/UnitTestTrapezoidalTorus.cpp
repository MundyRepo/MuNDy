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

/// \file UnitTestTrapezoidalTorus.cpp
/// \brief TrapezoidalTorus and trapezoidal_torus_rule (TrapezoidalTorus.hpp) against exact torus integrals,
/// high-precision references, and the rule's layout.
///
/// Anchors:
///   - Exactness: with C(p, q) = int_0^{2 pi} cos^p t sin^q t dt (2 G((p+1)/2) G((q+1)/2) / G((p+q+2)/2) when p, q are
///     both even and 0 otherwise, G the gamma function), the integral of x^p y^q z^s over the torus is
///     C(p, q) sum_k binom(K, k) a^(K-k) C(k, s) with K = p + q + 1, for every p + q + s <= min(N - 2, M - 1).
///   - 30-digit values of a point and weight for (N, M, a) = (5, 7, 2.5) and (24, 40, 3).
///   - Smooth integrands: exp(z) and exp(x / sqrt(x^2 + y^2)) both integrate to 4 pi^2 a I_0(1).

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

// Mundy
#include <mundy_math/TrapezoidalTorus.hpp>

namespace mundy {

namespace {

constexpr double pi = Kokkos::numbers::pi_v<double>;

//! \name Group 0: compile-time checks
//@{

static_assert(TrapezoidalTorus<double, 5, 7, 2.5>::num_points == 35);
static_assert(TrapezoidalTorus<double, 5, 7, 2.5>::points()[0] == 3.5 &&
                  TrapezoidalTorus<double, 5, 7, 2.5>::points()[1] == 0.0 &&
                  TrapezoidalTorus<double, 5, 7, 2.5>::points()[2] == 0.0,
              "The first point is on the outer equator at phi = 0: (a + 1, 0, 0).");
static_assert(abs(TrapezoidalTorus<double, 2, 1, 2.5>::integrate([](double, double, double) { return 1.0; }) -
                  4.0 * pi * pi * 2.5) < 1e-13,
              "integrate is a constant expression, and the rule integrates 1 to the area 4 pi^2 a.");
//@}

//! \name Helpers
//@{

/// \brief int_0^{2 pi} cos^p t sin^q t dt.
double circle_moment(int p, int q) {
  if (p % 2 == 1 || q % 2 == 1) {
    return 0.0;
  }
  return 2.0 * std::tgamma(0.5 * (p + 1)) * std::tgamma(0.5 * (q + 1)) / std::tgamma(0.5 * (p + q + 2));
}

/// \brief The integral of x^p y^q z^s over the torus with minor radius 1 and major radius a.
double torus_monomial_integral(int p, int q, int s, double a) {
  const int power = p + q + 1;  // (a + cos theta)^(p + q) from x^p y^q, times the area element
  double theta_integral = 0.0;
  double binomial = 1.0;
  for (int k = 0; k <= power; ++k) {
    theta_integral += binomial * std::pow(a, power - k) * circle_moment(k, s);
    binomial = binomial * (power - k) / (k + 1);
  }
  return circle_moment(p, q) * theta_integral;
}

/// \brief Expect the n x m rule to integrate every monomial of degree <= min(n - 2, m - 1) exactly, to within
/// tolerance times sum_i |w_i f_i| (the scale of the rounding in the sum).
template <class Scalar>
void expect_exact_for_polynomials(const Scalar* points, const Scalar* weights, unsigned n, unsigned m, double a,
                                  double tolerance) {
  const int max_degree = std::min(static_cast<int>(n) - 2, static_cast<int>(m) - 1);
  for (int p = 0; p <= max_degree; ++p) {
    for (int q = 0; p + q <= max_degree; ++q) {
      for (int s = 0; p + q + s <= max_degree; ++s) {
        double sum = 0.0;
        double scale = 0.0;
        for (unsigned i = 0; i < n * m; ++i) {
          const double term =
              weights[i] * std::pow(points[3 * i], p) * std::pow(points[3 * i + 1], q) * std::pow(points[3 * i + 2], s);
          sum += term;
          scale += std::abs(term);
        }
        EXPECT_NEAR(sum, torus_monomial_integral(p, q, s, a), tolerance * scale)
            << "n = " << n << ", m = " << m << ", x^" << p << " y^" << q << " z^" << s;
      }
    }
  }
}

template <unsigned N, unsigned M, double A>
void expect_exact_for_polynomials(double tolerance) {
  constexpr auto points = TrapezoidalTorus<double, N, M, A>::points();
  constexpr auto weights = TrapezoidalTorus<double, N, M, A>::weights();
  expect_exact_for_polynomials(points.data(), weights.data(), N, M, A, tolerance);
}

/// \brief Expect points on the torus, laid out ring by ring in theta from the outer equator, phi = 0 first.
void expect_layout(const double* points, const double* weights, unsigned n, unsigned m, double a) {
  for (unsigned i = 0; i < n; ++i) {
    const double theta = 2.0 * pi * i / n;
    for (unsigned j = 0; j < m; ++j) {
      const unsigned index = i * m + j;
      const double x = points[3 * index];
      const double y = points[3 * index + 1];
      const double z = points[3 * index + 2];
      const double tube_distance = std::hypot(std::hypot(x, y) - a, z);
      EXPECT_NEAR(tube_distance, 1.0, 4e-15) << "n = " << n << ", m = " << m << ", index = " << index;
      EXPECT_EQ(z, points[3 * i * m + 2]) << "z is constant on a ring; index = " << index;
      EXPECT_EQ(weights[index], weights[i * m]) << "weights are constant on a ring; index = " << index;
    }
    EXPECT_NEAR(points[3 * i * m + 2], std::sin(theta), 1e-15) << "Ring " << i << " is at theta = 2 pi i / n.";
    EXPECT_EQ(points[3 * i * m + 1], 0.0) << "Each ring starts at phi = 0.";
    EXPECT_GT(points[3 * i * m], 0.0) << "Each ring starts on the +x side.";
    EXPECT_NEAR(weights[i * m], 4.0 * pi * pi / (n * m) * (a + std::cos(theta)), 1e-15 * a)
        << "The weight is the trapezoidal weight times the area element.";
  }
}

/// \brief Expect trapezoidal_torus_rule(N, M, A, ...) to reproduce TrapezoidalTorus<double, N, M, A> bit for bit.
template <unsigned N, unsigned M, double A>
void expect_runtime_rule_matches_compile_time_rule() {
  std::vector<double> points;
  std::vector<double> weights;
  trapezoidal_torus_rule(N, M, A, points, weights);
  constexpr auto points_compile_time = TrapezoidalTorus<double, N, M, A>::points();
  constexpr auto weights_compile_time = TrapezoidalTorus<double, N, M, A>::weights();
  ASSERT_EQ(weights.size(), N * M);
  for (unsigned i = 0; i < 3 * N * M; ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]) << "N = " << N << ", M = " << M << ", i = " << i;
    EXPECT_EQ(std::signbit(points[i]), std::signbit(points_compile_time[i])) << "N = " << N << ", i = " << i;
  }
  for (unsigned i = 0; i < N * M; ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]) << "N = " << N << ", M = " << M << ", i = " << i;
  }
}
//@}

TEST(TrapezoidalTorus, IntegratesLowDegreePolynomialsExactly) {
  expect_exact_for_polynomials<3, 2, 2.5>(1e-14);
  expect_exact_for_polynomials<6, 5, 2.5>(1e-14);
  expect_exact_for_polynomials<8, 9, 1.5>(1e-14);
  expect_exact_for_polynomials<12, 12, 4.0>(1e-14);
}

TEST(TrapezoidalTorus, IsLaidOutRingByRing) {
  constexpr auto points = TrapezoidalTorus<double, 7, 9, 2.5>::points();
  constexpr auto weights = TrapezoidalTorus<double, 7, 9, 2.5>::weights();
  expect_layout(points.data(), weights.data(), 7, 9, 2.5);
}

TEST(TrapezoidalTorus, IsCorrectlyRounded) {
  using rule_t = TrapezoidalTorus<double, 5, 7, 2.5>;
  EXPECT_EQ(rule_t::points()[27], -6.250650850874724662502704304330e-1);
  EXPECT_EQ(rule_t::points()[28], 2.738589073609228839382311252378e+0);
  EXPECT_EQ(rule_t::points()[29], 9.510565162951535721164393333794e-1);
  EXPECT_EQ(rule_t::weights()[9], 3.168444170333460939424407664042e+0);
}

TEST(TrapezoidalTorusRule, MatchesTheCompileTimeRule) {
  expect_runtime_rule_matches_compile_time_rule<1, 1, 2.5>();
  expect_runtime_rule_matches_compile_time_rule<2, 3, 2.5>();
  expect_runtime_rule_matches_compile_time_rule<5, 7, 2.5>();
  expect_runtime_rule_matches_compile_time_rule<8, 8, 1.25>();
  expect_runtime_rule_matches_compile_time_rule<16, 12, 3.0>();
}

TEST(TrapezoidalTorusRule, BuildsLargeRules) {
  std::vector<double> points;
  std::vector<double> weights;
  trapezoidal_torus_rule(24, 40, 3.0, points, weights);
  ASSERT_EQ(weights.size(), 24u * 40);
  EXPECT_EQ(points[621], 1.479472886846846076138159948521e+0);
  EXPECT_EQ(points[622], 2.903629030335653025441394237382e+0);
  EXPECT_EQ(points[623], 9.659258262890682867497431997289e-1);
  EXPECT_EQ(weights[207], 1.340135616245735832548935964680e-1);
  expect_layout(points.data(), weights.data(), 24, 40, 3.0);

  trapezoidal_torus_rule(20, 15, 3.0, points, weights);
  expect_exact_for_polynomials(points.data(), weights.data(), 20, 15, 3.0, 1e-14);
}

TEST(TrapezoidalTorusRule, ConvergesSpectrallyForSmoothIntegrands) {
  const double a = 3.0;
  std::vector<double> points;
  std::vector<double> weights;
  trapezoidal_torus_rule(16, 16, a, points, weights);
  const auto integrate = [&](const auto& f) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
      sum += weights[i] * f(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    }
    return sum;
  };
  const double bessel_i0_of_1 = 1.26606587775200833559824462521472;
  const double exact = 4.0 * pi * pi * a * bessel_i0_of_1;
  EXPECT_NEAR(integrate([](double, double, double z) { return std::exp(z); }), exact, 1e-12 * exact)
      << "Spectral in theta.";
  EXPECT_NEAR(integrate([](double x, double y, double) { return std::exp(x / std::hypot(x, y)); }), exact,
              1e-12 * exact)
      << "Spectral in phi.";
}

TEST(TrapezoidalTorusRule, ViewsMatchVectors) {
  for (const unsigned n : {1u, 3u, 8u}) {
    for (const unsigned m : {1u, 5u, 12u}) {
      std::vector<double> points;
      std::vector<double> weights;
      trapezoidal_torus_rule(n, m, 2.5, points, weights);

      Kokkos::View<double*> points_view("points", 0);
      Kokkos::View<double*> weights_view("weights", 0);
      trapezoidal_torus_rule(n, m, 2.5, points_view, weights_view);
      const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
      const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
      ASSERT_EQ(points_host.extent(0), points.size());
      ASSERT_EQ(weights_host.extent(0), weights.size());
      for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_EQ(points_host(i), points[i]) << "n = " << n << ", m = " << m << ", i = " << i;
      }
      for (size_t i = 0; i < weights.size(); ++i) {
        EXPECT_EQ(weights_host(i), weights[i]) << "n = " << n << ", m = " << m << ", i = " << i;
      }
    }
  }

  // Single precision rounds the same double-double values.
  std::vector<float> points;
  std::vector<float> weights;
  trapezoidal_torus_rule(5, 7, 2.5, points, weights);
  Kokkos::View<float*> points_view("points", 0);
  Kokkos::View<float*> weights_view("weights", 0);
  trapezoidal_torus_rule(5, 7, 2.5, points_view, weights_view);
  const auto points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_view);
  const auto weights_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weights_view);
  constexpr auto points_compile_time = TrapezoidalTorus<float, 5, 7, 2.5>::points();
  constexpr auto weights_compile_time = TrapezoidalTorus<float, 5, 7, 2.5>::weights();
  for (unsigned i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], points_compile_time[i]);
    EXPECT_EQ(points_host(i), points_compile_time[i]);
  }
  for (unsigned i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], weights_compile_time[i]);
    EXPECT_EQ(weights_host(i), weights_compile_time[i]);
  }
}

TEST(TrapezoidalTorusRule, RejectsInvalidRules) {
  std::vector<double> points;
  std::vector<double> weights;
  Kokkos::View<double*> points_view("points", 0);
  Kokkos::View<double*> weights_view("weights", 0);
  for (const auto& [n, m, a] : {std::tuple{0u, 4u, 2.5}, std::tuple{4u, 0u, 2.5}, std::tuple{4u, 4u, 1.0},
                                std::tuple{4u, 4u, 0.5}, std::tuple{4u, 4u, std::numeric_limits<double>::infinity()},
                                std::tuple{4u, 4u, std::numeric_limits<double>::quiet_NaN()}}) {
    EXPECT_THROW(trapezoidal_torus_rule(n, m, a, points, weights), std::invalid_argument)
        << "n = " << n << ", m = " << m << ", a = " << a;
    EXPECT_THROW(trapezoidal_torus_rule(n, m, a, points_view, weights_view), std::invalid_argument)
        << "n = " << n << ", m = " << m << ", a = " << a;
  }
}

}  // namespace

}  // namespace mundy
