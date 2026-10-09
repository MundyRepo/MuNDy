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

/// \file UnitTestClenshawCurtis.cpp
/// \brief ClenshawCurtis and clenshaw_curtis_rule (ClenshawCurtis.hpp) against closed forms and polynomial exactness.
///
/// Anchors:
///   - Closed forms for N = 2..5, e.g. N = 3 is Simpson's rule and N = 5 has weights 1/15, 8/15, 4/5. Odd N is where
///     the last cosine term of the weight sum must be halved.
///   - Exactness: sum_i w_i P_k(x_i) = int_{-1}^{1} P_k(x) dx = 2 delta_{k0} for every k <= N - 1 (k <= N for odd N).

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/ClenshawCurtis.hpp>

namespace mundy {

namespace {

//! \name Compile-time checks
//@{

static_assert(ClenshawCurtis<double, 2>::nodes()[0] == -1.0 && ClenshawCurtis<double, 2>::nodes()[1] == 1.0);
static_assert(ClenshawCurtis<double, 2>::weights()[0] == 1.0 && ClenshawCurtis<double, 2>::weights()[1] == 1.0);
static_assert(ClenshawCurtis<double, 3>::integrate([](double x) { return x * x; }) - 2.0 / 3.0 < 1e-15 &&
                  ClenshawCurtis<double, 3>::integrate([](double x) { return x * x; }) - 2.0 / 3.0 > -1e-15,
              "integrate is a constant expression, and the 3-point rule (Simpson's) is exact for x^2.");
//@}

//! \name Helpers
//@{

/// \brief The highest polynomial degree the n-point Clenshaw-Curtis rule integrates exactly.
constexpr unsigned exact_degree(const unsigned n) {
  return n % 2 == 1 ? n : n - 1;
}

/// \brief Expect sum_i w_i P_k(x_i) = 2 delta_{k0} for every k <= exact_degree(n), with P_k from the three-term
/// recurrence.
template <class Scalar>
void expect_exact_for_legendre_polynomials(const Scalar* x, const Scalar* w, unsigned n, double tolerance) {
  const unsigned num_degrees = exact_degree(n) + 1;
  std::vector<double> sums(num_degrees, 0.0);
  for (unsigned i = 0; i < n; ++i) {
    double p_prev = 0.0;  // P_{k-1}(x_i)
    double p = 1.0;       // P_k(x_i)
    for (unsigned k = 0; k < num_degrees; ++k) {
      sums[k] += w[i] * p;
      const double p_next = ((2 * k + 1) * x[i] * p - k * p_prev) / (k + 1);
      p_prev = p;
      p = p_next;
    }
  }
  for (unsigned k = 0; k < num_degrees; ++k) {
    EXPECT_NEAR(sums[k], k == 0 ? 2.0 : 0.0, tolerance) << "n = " << n << ", k = " << k;
  }
}

template <class Scalar, unsigned N>
void expect_exact_for_legendre_polynomials(double tolerance) {
  constexpr auto x = ClenshawCurtis<Scalar, N>::nodes();
  constexpr auto w = ClenshawCurtis<Scalar, N>::weights();
  expect_exact_for_legendre_polynomials(x.data(), w.data(), N, tolerance);
}

void expect_symmetric_and_ascending(const double* x, const double* w, unsigned n) {
  EXPECT_EQ(x[0], -1.0) << "n = " << n;
  EXPECT_EQ(x[n - 1], 1.0) << "n = " << n;
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
  constexpr auto x = ClenshawCurtis<double, N>::nodes();
  constexpr auto w = ClenshawCurtis<double, N>::weights();
  expect_symmetric_and_ascending(x.data(), w.data(), N);
}

/// \brief Expect clenshaw_curtis_rule(N, ...) to reproduce ClenshawCurtis<double, N> bit for bit.
template <unsigned N>
void expect_runtime_rule_matches_compile_time_rule() {
  std::vector<double> x;
  std::vector<double> w;
  clenshaw_curtis_rule(N, x, w);
  constexpr auto x_compile_time = ClenshawCurtis<double, N>::nodes();
  constexpr auto w_compile_time = ClenshawCurtis<double, N>::weights();
  ASSERT_EQ(x.size(), N);
  for (unsigned i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], x_compile_time[i]) << "N = " << N << ", i = " << i;
    EXPECT_EQ(std::signbit(x[i]), std::signbit(x_compile_time[i])) << "N = " << N << ", i = " << i;
    EXPECT_EQ(w[i], w_compile_time[i]) << "N = " << N << ", i = " << i;
  }
}
//@}

TEST(ClenshawCurtis, MatchesClosedForms) {
  // The closed forms are themselves rounded, so compare to EXPECT_DOUBLE_EQ's 4 ulps.
  constexpr auto x3 = ClenshawCurtis<double, 3>::nodes();
  constexpr auto w3 = ClenshawCurtis<double, 3>::weights();
  EXPECT_EQ(x3[1], 0.0);
  EXPECT_DOUBLE_EQ(w3[0], 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(w3[1], 4.0 / 3.0);

  constexpr auto x4 = ClenshawCurtis<double, 4>::nodes();
  constexpr auto w4 = ClenshawCurtis<double, 4>::weights();
  EXPECT_DOUBLE_EQ(x4[2], 0.5);
  EXPECT_DOUBLE_EQ(w4[0], 1.0 / 9.0);
  EXPECT_DOUBLE_EQ(w4[2], 8.0 / 9.0);

  constexpr auto x5 = ClenshawCurtis<double, 5>::nodes();
  constexpr auto w5 = ClenshawCurtis<double, 5>::weights();
  EXPECT_DOUBLE_EQ(x5[3], std::sqrt(0.5));
  EXPECT_DOUBLE_EQ(w5[0], 1.0 / 15.0);
  EXPECT_DOUBLE_EQ(w5[1], 8.0 / 15.0);
  EXPECT_DOUBLE_EQ(w5[2], 4.0 / 5.0);
}

TEST(ClenshawCurtis, IntegratesPolynomialsUpToItsDegreeExactly) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_exact_for_legendre_polynomials<double, I + 2>(1e-14), ...);
  }(std::make_integer_sequence<unsigned, 32>{});
  expect_exact_for_legendre_polynomials<float, 8>(1e-6);
}

TEST(ClenshawCurtis, IsSymmetricAndAscending) {
  expect_symmetric_and_ascending<2>();
  expect_symmetric_and_ascending<7>();
  expect_symmetric_and_ascending<8>();
  expect_symmetric_and_ascending<33>();
}

/// \brief Sum over c = 0.01 s, s < num_shifts, of the 5-point rule for int_{-1}^{1} (x + c)^5 dx, on the device.
///
/// A free function, since CUDA forbids KOKKOS_LAMBDA in a test body (a private member function).
double integrate_shifted_quintics_on_device(const int num_shifts) {
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "ClenshawCurtis::integrate", Kokkos::RangePolicy<>(0, num_shifts),
      KOKKOS_LAMBDA(const int s, double& local_sum) {
        const double c = 0.01 * s;
        local_sum += ClenshawCurtis<double, 5>::integrate([c](double x) {
          const double y = x + c;
          return y * y * y * y * y;
        });
      },
      sum);
  return sum;
}

TEST(ClenshawCurtis, IntegratesOnTheDevice) {
  // int_{-1}^{1} (x + c)^5 dx = ((1 + c)^6 - (c - 1)^6) / 6, a degree-5 polynomial that the 5-point rule integrates
  // exactly, summed over c = 0, 0.01, ..., 0.99.
  constexpr int num_shifts = 100;
  const double sum = integrate_shifted_quintics_on_device(num_shifts);

  double expected = 0.0;
  for (int s = 0; s < num_shifts; ++s) {
    const double c = 0.01 * s;
    expected += (std::pow(1.0 + c, 6) - std::pow(c - 1.0, 6)) / 6.0;
  }
  EXPECT_NEAR(sum, expected, 1e-12);
}

TEST(ClenshawCurtisRule, MatchesTheCompileTimeRule) {
  []<unsigned... I>(std::integer_sequence<unsigned, I...>) {
    (expect_runtime_rule_matches_compile_time_rule<I + 2>(), ...);
  }(std::make_integer_sequence<unsigned, 32>{});
}

TEST(ClenshawCurtisRule, BuildsLargeRules) {
  std::vector<double> x;
  std::vector<double> w;
  clenshaw_curtis_rule(501, x, w);
  ASSERT_EQ(x.size(), 501u);
  expect_symmetric_and_ascending(x.data(), w.data(), 501);
  expect_exact_for_legendre_polynomials(x.data(), w.data(), 501, 1e-12);
}

TEST(ClenshawCurtisRule, ViewsMatchVectors) {
  for (const unsigned n : {2u, 3u, 7u, 64u, 129u}) {
    std::vector<double> x;
    std::vector<double> w;
    clenshaw_curtis_rule(n, x, w);

    Kokkos::View<double*> x_view("nodes", 0);
    Kokkos::View<double*> w_view("weights", 0);
    clenshaw_curtis_rule(n, x_view, w_view);
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
  clenshaw_curtis_rule(8, x, w);
  Kokkos::View<float*> x_view("nodes", 0);
  Kokkos::View<float*> w_view("weights", 0);
  clenshaw_curtis_rule(8, x_view, w_view);
  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_view);
  const auto w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, w_view);
  constexpr auto x_compile_time = ClenshawCurtis<float, 8>::nodes();
  constexpr auto w_compile_time = ClenshawCurtis<float, 8>::weights();
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_EQ(x[i], x_compile_time[i]);
    EXPECT_EQ(w[i], w_compile_time[i]);
    EXPECT_EQ(x_host(i), x_compile_time[i]);
    EXPECT_EQ(w_host(i), w_compile_time[i]);
  }
}

TEST(ClenshawCurtisRule, RejectsRulesWithoutBothEndpoints) {
  for (const unsigned n : {0u, 1u}) {
    std::vector<double> x;
    std::vector<double> w;
    EXPECT_THROW(clenshaw_curtis_rule(n, x, w), std::invalid_argument) << "n = " << n;
    Kokkos::View<double*> x_view("nodes", 0);
    Kokkos::View<double*> w_view("weights", 0);
    EXPECT_THROW(clenshaw_curtis_rule(n, x_view, w_view), std::invalid_argument) << "n = " << n;
  }
}

}  // namespace

}  // namespace mundy
