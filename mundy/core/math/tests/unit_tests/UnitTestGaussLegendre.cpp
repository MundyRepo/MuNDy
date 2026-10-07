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

/// \file UnitTestGaussLegendre.cpp
/// \brief GaussLegendre and gauss_legendre_rule (GaussLegendre.hpp) against closed forms, high-precision references,
/// and polynomial exactness.
///
/// Anchors:
///   - Closed forms for N = 1..4, e.g. N = 3 has nodes 0, +-sqrt(3/5) and weights 8/9, 5/9.
///   - 30-digit values of the outermost node and weight for N = 16, 48, 62, 128, where double arithmetic loses the
///     most.
///   - Exactness: sum_i w_i P_k(x_i) = int_{-1}^{1} P_k(x) dx = 2 delta_{k0} for every k <= 2N - 1.
///
/// Every compile-time N stays within clang's default constexpr limit (N <= 62), so the test builds with any compiler.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/GaussLegendre.hpp>

namespace mundy {

namespace {

//! \name Group 0: compile-time checks
//@{

static_assert(GaussLegendre<double, 1>::nodes()[0] == 0.0 && GaussLegendre<double, 1>::weights()[0] == 2.0);
static_assert(GaussLegendre<double, 2>::weights()[0] == 1.0 && GaussLegendre<double, 2>::weights()[1] == 1.0);
static_assert(abs(GaussLegendre<double, 3>::integrate([](double x) { return x * x * x * x; }) - 0.4) < 1e-15,
              "integrate is a constant expression, and the 3-point rule is exact for x^4.");
//@}

//! \name Helpers
//@{

/// \brief Expect sum_i w_i P_k(x_i) = 2 delta_{k0} for every k <= 2n - 1, with P_k from the three-term recurrence.
template <class Scalar>
void expect_exact_for_legendre_polynomials(const Scalar* x, const Scalar* w, unsigned n, double tolerance) {
  std::vector<double> sums(2 * n, 0.0);
  for (unsigned i = 0; i < n; ++i) {
    double p_prev = 0.0;  // P_{k-1}(x_i)
    double p = 1.0;       // P_k(x_i)
    for (unsigned k = 0; k < 2 * n; ++k) {
      sums[k] += w[i] * p;
      const double p_next = ((2 * k + 1) * x[i] * p - k * p_prev) / (k + 1);
      p_prev = p;
      p = p_next;
    }
  }
  for (unsigned k = 0; k < 2 * n; ++k) {
    EXPECT_NEAR(sums[k], k == 0 ? 2.0 : 0.0, tolerance) << "n = " << n << ", k = " << k;
  }
}

template <class Scalar, unsigned N>
void expect_exact_for_legendre_polynomials(double tolerance) {
  constexpr auto x = GaussLegendre<Scalar, N>::nodes();
  constexpr auto w = GaussLegendre<Scalar, N>::weights();
  expect_exact_for_legendre_polynomials(x.data(), w.data(), N, tolerance);
}

void expect_symmetric_and_ascending(const double* x, const double* w, unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    EXPECT_EQ(x[i], -x[n - 1 - i]) << "n = " << n << ", i = " << i;
    EXPECT_EQ(w[i], w[n - 1 - i]) << "n = " << n << ", i = " << i;
    EXPECT_GT(w[i], 0.0) << "n = " << n << ", i = " << i;
    if (i > 0) {
      EXPECT_LT(x[i - 1], x[i]) << "n = " << n << ", i = " << i;
    }
  }
  if (n % 2 == 1) {
    EXPECT_FALSE(std::signbit(x[n / 2])) << "The middle node of an odd rule is +0.";
  }
}

template <unsigned N>
void expect_symmetric_and_ascending() {
  constexpr auto x = GaussLegendre<double, N>::nodes();
  constexpr auto w = GaussLegendre<double, N>::weights();
  expect_symmetric_and_ascending(x.data(), w.data(), N);
}

/// \brief Expect gauss_legendre_rule(N, ...) to reproduce GaussLegendre<double, N> bit for bit.
template <unsigned N>
void expect_runtime_rule_matches_compile_time_rule() {
  std::vector<double> x;
  std::vector<double> w;
  gauss_legendre_rule(N, x, w);
  constexpr auto x_compile_time = GaussLegendre<double, N>::nodes();
  constexpr auto w_compile_time = GaussLegendre<double, N>::weights();
  ASSERT_EQ(x.size(), N);
  for (unsigned i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], x_compile_time[i]) << "N = " << N << ", i = " << i;
    EXPECT_EQ(std::signbit(x[i]), std::signbit(x_compile_time[i])) << "N = " << N << ", i = " << i;
    EXPECT_EQ(w[i], w_compile_time[i]) << "N = " << N << ", i = " << i;
  }
}
//@}

TEST(GaussLegendre, MatchesClosedForms) {
  // The closed forms are themselves rounded, so compare to EXPECT_DOUBLE_EQ's 4 ulps.
  constexpr auto x2 = GaussLegendre<double, 2>::nodes();
  EXPECT_DOUBLE_EQ(x2[1], 1.0 / std::sqrt(3.0));

  constexpr auto x3 = GaussLegendre<double, 3>::nodes();
  constexpr auto w3 = GaussLegendre<double, 3>::weights();
  EXPECT_EQ(x3[1], 0.0);
  EXPECT_DOUBLE_EQ(x3[2], std::sqrt(3.0 / 5.0));
  EXPECT_DOUBLE_EQ(w3[1], 8.0 / 9.0);
  EXPECT_DOUBLE_EQ(w3[2], 5.0 / 9.0);

  constexpr auto x4 = GaussLegendre<double, 4>::nodes();
  constexpr auto w4 = GaussLegendre<double, 4>::weights();
  EXPECT_DOUBLE_EQ(x4[2], std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0)));
  EXPECT_DOUBLE_EQ(x4[3], std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0)));
  EXPECT_DOUBLE_EQ(w4[2], (18.0 + std::sqrt(30.0)) / 36.0);
  EXPECT_DOUBLE_EQ(w4[3], (18.0 - std::sqrt(30.0)) / 36.0);
}

TEST(GaussLegendre, IsCorrectlyRoundedAtTheEndpoints) {
  using rule16_t = GaussLegendre<double, 16>;
  using rule48_t = GaussLegendre<double, 48>;
  using rule62_t = GaussLegendre<double, 62>;
  EXPECT_EQ(rule16_t::nodes()[15], 0.989400934991649932596154173450);
  EXPECT_EQ(rule16_t::weights()[15], 0.02715245941175409485178057245601);
  EXPECT_EQ(rule48_t::nodes()[47], 0.998771007252426118600541491563);
  EXPECT_EQ(rule48_t::weights()[47], 0.00315334605230583863267731154389);
  EXPECT_EQ(rule62_t::nodes()[61], 0.999259859308777029698408465035);
  EXPECT_EQ(rule62_t::weights()[61], 0.00189920567951369048039734386093);
}

TEST(GaussLegendre, IntegratesPolynomialsOfDegreeUpTo2NMinus1Exactly) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_exact_for_legendre_polynomials<double, I + 1>(1e-14), ...);
  }(std::make_integer_sequence<unsigned, 32>{});
  expect_exact_for_legendre_polynomials<double, 62>(1e-13);
  expect_exact_for_legendre_polynomials<float, 8>(1e-6);
}

TEST(GaussLegendre, IsSymmetricAndAscending) {
  expect_symmetric_and_ascending<7>();
  expect_symmetric_and_ascending<8>();
  expect_symmetric_and_ascending<61>();
}

TEST(GaussLegendre, IntegratesOnTheDevice) {
  // int_{-1}^{1} (x + c)^5 dx = ((1 + c)^6 - (c - 1)^6) / 6, a degree-5 polynomial that the 3-point rule integrates
  // exactly, summed over c = 0, 0.01, ..., 0.99.
  constexpr int num_shifts = 100;
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "GaussLegendre::integrate", Kokkos::RangePolicy<>(0, num_shifts),
      KOKKOS_LAMBDA(const int s, double& local_sum) {
        const double c = 0.01 * s;
        local_sum += GaussLegendre<double, 3>::integrate([c](double x) {
          const double y = x + c;
          return y * y * y * y * y;
        });
      },
      sum);

  double expected = 0.0;
  for (int s = 0; s < num_shifts; ++s) {
    const double c = 0.01 * s;
    expected += (std::pow(1.0 + c, 6) - std::pow(c - 1.0, 6)) / 6.0;
  }
  EXPECT_NEAR(sum, expected, 1e-12);
}

TEST(GaussLegendreRule, MatchesTheCompileTimeRule) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_runtime_rule_matches_compile_time_rule<I + 1>(), ...);
  }(std::make_integer_sequence<unsigned, 32>{});
  expect_runtime_rule_matches_compile_time_rule<62>();
}

TEST(GaussLegendreRule, BuildsRulesBeyondTheCompileTimeLimits) {
  std::vector<double> x;
  std::vector<double> w;
  gauss_legendre_rule(128, x, w);
  EXPECT_EQ(x[127], 0.999824887947131914473608082981);
  EXPECT_EQ(w[127], 0.00044938096029209037639429223998);

  gauss_legendre_rule(500, x, w);
  ASSERT_EQ(x.size(), 500u);
  expect_symmetric_and_ascending(x.data(), w.data(), 500);
  expect_exact_for_legendre_polynomials(x.data(), w.data(), 500, 1e-12);
}

TEST(GaussLegendreRule, ViewsMatchVectors) {
  for (const unsigned n : {1u, 2u, 7u, 64u, 129u}) {
    std::vector<double> x;
    std::vector<double> w;
    gauss_legendre_rule(n, x, w);

    Kokkos::View<double*> x_view("nodes", 0);
    Kokkos::View<double*> w_view("weights", 0);
    gauss_legendre_rule(n, x_view, w_view);
    const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_view);
    const auto w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w_view);
    ASSERT_EQ(x_host.extent(0), n);
    ASSERT_EQ(w_host.extent(0), n);
    for (unsigned i = 0; i < n; ++i) {
      EXPECT_EQ(x_host(i), x[i]) << "n = " << n << ", i = " << i;
      EXPECT_EQ(w_host(i), w[i]) << "n = " << n << ", i = " << i;
    }
  }

  // Single precision rounds the same double-double values.
  std::vector<float> x;
  std::vector<float> w;
  gauss_legendre_rule(8, x, w);
  Kokkos::View<float*> x_view("nodes", 0);
  Kokkos::View<float*> w_view("weights", 0);
  gauss_legendre_rule(8, x_view, w_view);
  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_view);
  const auto w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w_view);
  constexpr auto x_compile_time = GaussLegendre<float, 8>::nodes();
  constexpr auto w_compile_time = GaussLegendre<float, 8>::weights();
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_EQ(x[i], x_compile_time[i]);
    EXPECT_EQ(w[i], w_compile_time[i]);
    EXPECT_EQ(x_host(i), x_compile_time[i]);
    EXPECT_EQ(w_host(i), w_compile_time[i]);
  }
}

TEST(GaussLegendreRule, RejectsEmptyRules) {
  std::vector<double> x;
  std::vector<double> w;
  EXPECT_THROW(gauss_legendre_rule(0, x, w), std::invalid_argument);
  Kokkos::View<double*> x_view("nodes", 0);
  Kokkos::View<double*> w_view("weights", 0);
  EXPECT_THROW(gauss_legendre_rule(0, x_view, w_view), std::invalid_argument);
}

}  // namespace

}  // namespace mundy
