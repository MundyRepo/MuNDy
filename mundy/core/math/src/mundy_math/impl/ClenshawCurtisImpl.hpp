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

#ifndef MUNDY_MATH_IMPL_CLENSHAWCURTISIMPL_HPP_
#define MUNDY_MATH_IMPL_CLENSHAWCURTISIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// Mundy
#include <mundy_math/DoubleDouble.hpp>           // for mundy::DoubleDouble
#include <mundy_math/impl/DoubleDoubleImpl.hpp>  // for mundy::impl::sin_cos_of_turn_fraction

namespace mundy {

namespace impl {

//! \name Construction of Clenshaw-Curtis rules, at compile time or run time
//@{

/// \brief Write node i of the n-point Clenshaw-Curtis rule and its mirror n - 1 - i, rounded to Scalar (i <= m / 2).
///
/// With m = n - 1, node i is -cos(pi i / m) and its weight is
///   w_i = c_i / m (1 - sum_{j = 1}^{floor(m / 2)} b_j cos(2 pi i j / m) / (4 j^2 - 1)),
/// where c_i is 1 at the endpoints and 2 inside, and b_j is 1 for j = m / 2 and 2 otherwise. The mirror is node i
/// negated, with the same weight.
template <class Scalar, class Nodes, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_clenshaw_curtis_pair(unsigned n, unsigned i, Nodes& nodes,
                                                                 Weights& weights) {
  const unsigned m = n - 1;
  DoubleDouble sum = 0.0;
  for (unsigned j = 1; 2 * j <= m; ++j) {
    const double b = 2 * j == m ? 1.0 : 2.0;
    sum += sin_cos_of_turn_fraction<DoubleDouble>((i * j) % m, m).cos * b / static_cast<double>(4 * j * j - 1);
  }
  const double c = i == 0 ? 1.0 : 2.0;
  const Scalar weight = static_cast<Scalar>(((1.0 - sum) * c / static_cast<double>(m)).hi());
  const Scalar node = static_cast<Scalar>((-sin_cos_of_turn_fraction<DoubleDouble>(i, 2 * m).cos).hi());
  // The mirror is written first, so the middle node of an odd rule keeps +0.
  nodes[m - i] = -node;
  weights[m - i] = weight;
  nodes[i] = node;
  weights[i] = weight;
}

/// \brief The nodes and weights of the N-point Clenshaw-Curtis rule.
template <class Scalar, unsigned N>
struct ClenshawCurtisRule {
  Kokkos::Array<Scalar, N> nodes;
  Kokkos::Array<Scalar, N> weights;
};

/// \brief Build the N-point Clenshaw-Curtis rule on [-1, 1].
template <class Scalar, unsigned N>
KOKKOS_INLINE_FUNCTION constexpr ClenshawCurtisRule<Scalar, N> make_clenshaw_curtis_rule() {
  ClenshawCurtisRule<Scalar, N> rule{};
  for (unsigned i = 0; i < (N + 1) / 2; ++i) {
    write_clenshaw_curtis_pair<Scalar>(N, i, rule.nodes, rule.weights);
  }
  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_CLENSHAWCURTISIMPL_HPP_
