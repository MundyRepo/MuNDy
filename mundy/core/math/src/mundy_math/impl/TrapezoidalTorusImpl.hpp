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

#ifndef MUNDY_MATH_IMPL_TRAPEZOIDALTORUSIMPL_HPP_
#define MUNDY_MATH_IMPL_TRAPEZOIDALTORUSIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// Mundy
#include <mundy_math/DoubleDouble.hpp>           // for mundy::DoubleDouble
#include <mundy_math/impl/DoubleDoubleImpl.hpp>  // for mundy::impl::{SinCos, sin_cos_of_turn_fraction, pi_over_2}

namespace mundy {

namespace impl {

//! \name Construction of trapezoidal torus rules, at compile time or run time
//@{

/// \brief Write ring i of the n x m trapezoidal rule on the torus with minor radius 1 and major radius a, as Scalar.
///
/// Ring i is the circle at the poloidal angle theta_i = 2 pi i / n: the points ((a + cos theta_i) cos phi_j,
/// (a + cos theta_i) sin phi_j, sin theta_i) at the toroidal angles phi_j = 2 pi j / m, whose sines and cosines
/// toroidal[j] holds. Each has weight (2 pi / n)(2 pi / m)(a + cos theta_i): the trapezoidal weight times the area
/// element.
template <class Scalar, class Toroidal, class Points, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_trapezoidal_torus_ring(unsigned n, unsigned m, double aspect_ratio,
                                                                 unsigned i, const Toroidal& toroidal, Points& points,
                                                                 Weights& weights) {
  const SinCos<DoubleDouble> poloidal = sin_cos_of_turn_fraction<DoubleDouble>(i, n);
  const DoubleDouble distance_from_axis = aspect_ratio + poloidal.cos;
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDouble two_pi = DoubleDouble(half_pi.hi, half_pi.lo) * 4.0;
  const DoubleDouble weight = two_pi / static_cast<double>(n) * (two_pi / static_cast<double>(m)) * distance_from_axis;
  const Scalar z = static_cast<Scalar>(poloidal.sin.hi());
  for (unsigned j = 0; j < m; ++j) {
    const unsigned index = i * m + j;
    points[3 * index + 0] = static_cast<Scalar>((distance_from_axis * toroidal[j].cos).hi());
    points[3 * index + 1] = static_cast<Scalar>((distance_from_axis * toroidal[j].sin).hi());
    points[3 * index + 2] = z;
    weights[index] = static_cast<Scalar>(weight.hi());
  }
}

/// \brief The points and weights of the N x M trapezoidal torus rule.
template <class Scalar, unsigned N, unsigned M>
struct TrapezoidalTorusRule {
  Kokkos::Array<Scalar, 3 * N * M> points;  //!< (x, y, z) triples
  Kokkos::Array<Scalar, N * M> weights;
};

/// \brief Build the N x M trapezoidal rule on the torus with minor radius 1 and major radius AspectRatio.
template <class Scalar, unsigned N, unsigned M, double AspectRatio>
KOKKOS_INLINE_FUNCTION constexpr TrapezoidalTorusRule<Scalar, N, M> make_trapezoidal_torus_rule() {
  Kokkos::Array<SinCos<DoubleDouble>, M> toroidal{};
  for (unsigned j = 0; j < M; ++j) {
    toroidal[j] = sin_cos_of_turn_fraction<DoubleDouble>(j, M);
  }
  TrapezoidalTorusRule<Scalar, N, M> rule{};
  for (unsigned i = 0; i < N; ++i) {
    write_trapezoidal_torus_ring<Scalar>(N, M, AspectRatio, i, toroidal, rule.points, rule.weights);
  }
  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_TRAPEZOIDALTORUSIMPL_HPP_
