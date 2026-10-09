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

#ifndef MUNDY_MATH_CLENSHAWCURTIS_HPP_
#define MUNDY_MATH_CLENSHAWCURTIS_HPP_

/// \file ClenshawCurtis.hpp
/// \brief Clenshaw-Curtis quadrature, with the number of points fixed at compile time or chosen at run time.
///
/// The n-point rule integrates over [-1, 1] at the Chebyshev extreme points -cos(pi i / (n - 1)), endpoints included,
/// with the weights of Trefethen's clencurt (Spectral Methods in MATLAB, SIAM 2000; see also Waldvogel, BIT 46, 2006).
/// It is exact for polynomials of degree at most n - 1, and n for odd n.
///
/// ClenshawCurtis<Scalar, N> mirrors GaussLegendre<Scalar, N>: the compiler builds the table, which is constexpr and
/// usable on the host or device. clenshaw_curtis_rule(n, nodes, weights) builds the same rule, bit for bit, for a
/// run-time n. Both cost O(n^2) work.

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/impl/ClenshawCurtisImpl.hpp>  // for mundy::impl::make_clenshaw_curtis_rule, ...
#include <mundy_utils/requires.hpp>                // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>            // for MUNDY_THROW_REQUIRE

namespace mundy {

/// \brief The N-point Clenshaw-Curtis rule on [-1, 1], exact for polynomials of degree at most N - 1 (N for odd N).
///
/// The nodes and weights are correctly rounded to Scalar and computed by the compiler, so every member is a constant
/// expression and works on the device. Like GaussLegendre, it integrates on [-1, 1]; map other intervals affinely.
template <class Scalar, unsigned N>
class ClenshawCurtis {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "ClenshawCurtis: the tables are computed to double precision, so Scalar must be float or double.");
  static_assert(N >= 2, "ClenshawCurtis: a rule needs both endpoints, so at least two points.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_points = N;

  /// \brief The N nodes in ascending order, from -1 to 1. Node N - 1 - i is minus node i.
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
    constexpr impl::ClenshawCurtisRule<Scalar, N> r = rule();
    auto sum = r.weights[0] * f(r.nodes[0]);
    for (unsigned i = 1; i < N; ++i) {
      sum += r.weights[i] * f(r.nodes[i]);
    }
    return sum;
  }

 private:
  /// \brief The rule, built once per (Scalar, N) by the compiler.
  KOKKOS_INLINE_FUNCTION static constexpr impl::ClenshawCurtisRule<Scalar, N> rule() {
    constexpr impl::ClenshawCurtisRule<Scalar, N> r = impl::make_clenshaw_curtis_rule<Scalar, N>();
    return r;
  }
};

/// \brief The n-point Clenshaw-Curtis rule on [-1, 1] for a run-time n, into std::vectors.
///
/// Resizes nodes and weights to n and fills them in ascending node order with the values ClenshawCurtis<Scalar, n>
/// would hold.
template <class Scalar>
void clenshaw_curtis_rule(unsigned n, std::vector<Scalar>& nodes, std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "clenshaw_curtis_rule: rules are computed to double precision, so Scalar must be float or double.");
  MUNDY_THROW_REQUIRE(n >= 2, std::invalid_argument,
                      "clenshaw_curtis_rule: a rule needs both endpoints, so at least two points.");
  nodes.resize(n);
  weights.resize(n);
  for (unsigned i = 0; i < (n + 1) / 2; ++i) {
    impl::write_clenshaw_curtis_pair<Scalar>(n, i, nodes, weights);
  }
}

/// \brief The n-point Clenshaw-Curtis rule on [-1, 1] for a run-time n, into rank-1 Kokkos views.
///
/// Resizes nodes and weights to n and fills them in ascending node order with the values ClenshawCurtis<Scalar, n>
/// would hold, one node pair per thread on the views' execution space. Returns once the rule is complete.
template <class NodesView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<NodesView>&& Kokkos::is_view_v<WeightsView>)
void clenshaw_curtis_rule(unsigned n, NodesView& nodes, WeightsView& weights) {
  using scalar_t = typename NodesView::non_const_value_type;
  using execution_space = typename NodesView::execution_space;
  static_assert(NodesView::rank() == 1 && WeightsView::rank() == 1, "clenshaw_curtis_rule: views must be rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "clenshaw_curtis_rule: rules are computed to double precision, so Scalar must be float or double.");
  static_assert(std::is_same_v<scalar_t, typename WeightsView::value_type> &&
                    std::is_same_v<scalar_t, typename NodesView::value_type>,
                "clenshaw_curtis_rule: nodes and weights must be writable views of the same scalar type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "clenshaw_curtis_rule: weights must be accessible from the nodes' execution space.");
  MUNDY_THROW_REQUIRE(n >= 2, std::invalid_argument,
                      "clenshaw_curtis_rule: a rule needs both endpoints, so at least two points.");

  Kokkos::resize(nodes, n);
  Kokkos::resize(weights, n);
  const execution_space space;
  Kokkos::parallel_for(
      "mundy::clenshaw_curtis_rule", Kokkos::RangePolicy<execution_space>(space, 0, (n + 1) / 2),
      KOKKOS_LAMBDA(const unsigned i) { impl::write_clenshaw_curtis_pair<scalar_t>(n, i, nodes, weights); });
  space.fence("mundy::clenshaw_curtis_rule");
}

}  // namespace mundy

#endif  // MUNDY_MATH_CLENSHAWCURTIS_HPP_
