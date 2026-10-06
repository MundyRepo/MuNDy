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

#ifndef MUNDY_MATH_IMPL_GAUSSLEGENDREIMPL_HPP_
#define MUNDY_MATH_IMPL_GAUSSLEGENDREIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, Kokkos::numbers::pi_v, KOKKOS_INLINE_FUNCTION

// Mundy
#include <mundy_math/DoubleDouble.hpp>  // for mundy::DoubleDouble
#include <mundy_math/cmath.hpp>         // for mundy::abs

namespace mundy {

namespace impl {

//! \name Compile-time construction of Gauss-Legendre rules
//@{

/// \brief cos(t) for 0 <= t <= pi/2 by its Taylor series, which is accurate to double precision there.
KOKKOS_INLINE_FUNCTION constexpr double cos_taylor(double t) {
  double term = 1.0;
  double sum = 1.0;
  for (int k = 1; k <= 12; ++k) {
    term *= -t * t / ((2 * k - 1) * (2 * k));
    sum += term;
  }
  return sum;
}

/// \brief The value and derivative of a Legendre polynomial at one point.
template <typename Real>
struct LegendreValue {
  Real value;
  Real derivative;
};

/// \brief P_n(x) and P_n'(x), for |x| < 1, in double or DoubleDouble.
///
/// P_n comes from the recurrence (k + 1) P_{k+1} = (2k + 1) x P_k - k P_{k-1}, and its derivative from
/// (1 - x^2) P_n' = n (P_{n-1} - x P_n).
template <typename Real>
KOKKOS_INLINE_FUNCTION constexpr LegendreValue<Real> legendre_p(unsigned n, Real x) {
  Real p_prev = 0.0;  // P_{k-1}
  Real p = 1.0;       // P_k
  for (unsigned k = 0; k < n; ++k) {
    const Real p_next = (Real(2 * k + 1) * x * p - Real(k) * p_prev) / static_cast<double>(k + 1);
    p_prev = p;
    p = p_next;
  }
  return {p, Real(n) * (p_prev - x * p) / (1.0 - x * x)};
}

/// \brief The nodes (ascending) and weights of an N-point Gauss-Legendre rule.
template <class Scalar, unsigned N>
struct GaussLegendreRule {
  Kokkos::Array<Scalar, N> nodes;
  Kokkos::Array<Scalar, N> weights;
  bool converged = true;  //!< whether Newton's method reached every root
};

/// \brief Build the N-point Gauss-Legendre rule.
///
/// The nodes are the roots of P_N and come in pairs +-x, so only the nonnegative ones are computed. Each one is
///   1. estimated by Tricomi's formula x ~ (1 - 1/(8N^2) + 1/(8N^3)) cos(pi (4j + 3) / (4N + 2)) for the j-th largest,
///   2. refined by Newton's method x <- x - P_N(x) / P_N'(x): in double until it converges, then one step in
///      double-double, which squares the remaining error of about 1e-16,
///   3. given the weight w = 2 / ((1 - x^2) P_N'(x)^2), in double-double.
/// These are the textbook formulas. Evaluated in double they lose several digits of the weights near x = +-1, so the
/// last step and the weight are evaluated in double-double and rounded once at the end.
template <class Scalar, unsigned N>
KOKKOS_INLINE_FUNCTION constexpr GaussLegendreRule<Scalar, N> make_gauss_legendre_rule() {
  // Newton's error is about the square of the previous step, so once a step falls below 1e-12 the root is accurate to
  // double precision. From Tricomi's estimate that takes at most 4 steps for N <= 256; the limit is only a safeguard.
  constexpr int max_newton_iterations = 16;
  constexpr double converged_step = 1e-12;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  GaussLegendreRule<Scalar, N> rule{};
  for (unsigned j = 0; j < (N + 1) / 2; ++j) {
    DoubleDouble x = 0.0;  // the middle root of an odd rule is exactly 0
    if (2 * j + 1 < N) {
      double x_double =
          (1.0 - 1.0 / (8.0 * N * N) + 1.0 / (8.0 * N * N * N)) * cos_taylor(pi * (4 * j + 3) / (4 * N + 2));
      bool converged = false;
      for (int iter = 0; iter < max_newton_iterations && !converged; ++iter) {
        const LegendreValue<double> p = legendre_p(N, x_double);
        const double step = p.value / p.derivative;
        x_double -= step;
        converged = mundy::abs(step) < converged_step;
      }
      rule.converged = rule.converged && converged;

      const LegendreValue<DoubleDouble> p = legendre_p(N, DoubleDouble(x_double));
      x = x_double - p.value / p.derivative;
    }

    const DoubleDouble dp = legendre_p(N, x).derivative;
    const DoubleDouble w = 2.0 / ((1.0 - x * x) * dp * dp);
    rule.nodes[j] = -static_cast<Scalar>(x.hi());
    rule.nodes[N - 1 - j] = static_cast<Scalar>(x.hi());  // written second, so the middle node of an odd rule is +0
    rule.weights[j] = static_cast<Scalar>(w.hi());
    rule.weights[N - 1 - j] = static_cast<Scalar>(w.hi());
  }
  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_GAUSSLEGENDREIMPL_HPP_
