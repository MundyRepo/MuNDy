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

#ifndef MUNDY_MATH_GAUSSLEGENDRE_HPP_
#define MUNDY_MATH_GAUSSLEGENDRE_HPP_

/// \file GaussLegendre.hpp
/// \brief Gauss-Legendre quadrature with the number of points fixed at compile time.
///
/// GaussLegendre<Scalar, N> follows boost::math::quadrature::gauss<Real, N>, with one difference: Boost hard-codes its
/// tables for a few N, while here the compiler builds the table for whichever N is used. The resulting table is
/// constexpr and usable on the host or device.
///
/// GaussLegendre uses compile-time work that grows like N^2. At the time of writing, GCC 13.3.0 compiles for N <= 204
/// whereas clang 18.1.8 compiles for N <= 62. Larger N need -fconstexpr-ops-limit (GCC) or -fconstexpr-steps (clang).

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <type_traits>  // for std::is_same_v

// Mundy
#include <mundy_math/impl/GaussLegendreImpl.hpp>  // for mundy::impl::make_gauss_legendre_rule

namespace mundy {

/// \brief The N-point Gauss-Legendre rule on [-1, 1], exact for polynomials of degree at most 2N - 1.
///
/// The nodes and weights are correctly rounded to Scalar and computed by the compiler, so every member is a constant
/// expression and works on the device. Like Boost's rule, it integrates on [-1, 1]; map other intervals affinely.
template <class Scalar, unsigned N>
class GaussLegendre {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "GaussLegendre: the tables are computed to double precision, so Scalar must be float or double.");
  static_assert(N >= 1, "GaussLegendre: a rule needs at least one point.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_points = N;

  /// \brief The N nodes in ascending order. Node N - 1 - i is minus node i.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, N> nodes() {
    return rule().nodes;
  }

  /// \brief The N weights, in the order of nodes().
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, N> weights() {
    return rule().weights;
  }

  /// \brief sum_i w_i f(x_i), approximating the integral of f over [-1, 1].
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(const Function& f) {
    constexpr impl::GaussLegendreRule<Scalar, N> r = rule();
    auto sum = r.weights[0] * f(r.nodes[0]);
    for (unsigned i = 1; i < N; ++i) {
      sum += r.weights[i] * f(r.nodes[i]);
    }
    return sum;
  }

 private:
  /// \brief The rule, built once per (Scalar, N) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr impl::GaussLegendreRule<Scalar, N> rule() {
    constexpr impl::GaussLegendreRule<Scalar, N> r = impl::make_gauss_legendre_rule<Scalar, N>();
    static_assert(r.converged, "GaussLegendre: Newton's method did not converge to every root of P_N.");
    return r;
  }
};

}  // namespace mundy

#endif  // MUNDY_MATH_GAUSSLEGENDRE_HPP_
