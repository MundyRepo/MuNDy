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

#ifndef MUNDY_MATH_IMPL_GAUSSLEGENDRESPHEREIMPL_HPP_
#define MUNDY_MATH_IMPL_GAUSSLEGENDRESPHEREIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// Mundy
#include <mundy_math/DoubleDouble.hpp>             // for mundy::DoubleDouble, mundy::sqrt
#include <mundy_math/impl/DoubleDoubleImpl.hpp>   // for mundy::impl::{SinCos, sin_cos_of_turn_fraction}
#include <mundy_math/impl/GaussLegendreImpl.hpp>  // for mundy::impl::gauss_legendre_point

namespace mundy {

namespace impl {

//! \name Construction of Gauss-Legendre sphere rules, at compile time or run time
//@{

/// \brief One latitude ring of a Gauss-Legendre sphere rule: z = cos(theta), sin(theta), and the Gauss-Legendre weight.
struct SphereRing {
  DoubleDouble z;
  DoubleDouble sin_theta;
  DoubleDouble weight;
  bool converged;  //!< whether Newton's method reached the Gauss-Legendre root
};

/// \brief Ring j of the n-ring rule, counted from the north pole, for j in the northern half (j <= n - 1 - j).
KOKKOS_INLINE_FUNCTION constexpr SphereRing gauss_legendre_sphere_ring(unsigned n, unsigned j) {
  const GaussLegendrePoint root = gauss_legendre_point(n, j);  // the j-th largest root of P_n
  return {root.node, sqrt((1.0 - root.node) * (1.0 + root.node)), root.weight, root.converged};
}

/// \brief Write ring j of the n-ring rule and its mirror image n - 1 - j, rounded to Scalar, into points and weights.
///
/// Points are (x, y, z) triples, ring by ring from the north, 2n longitudes per ring with longitudes[k] holding the
/// sine and cosine of longitude k. The mirror ring is ring j with z negated, and every point of a ring has weight
/// (2 pi / 2n) times the ring's Gauss-Legendre weight. Returns whether Newton's method converged.
template <class Scalar, class Longitudes, class Points, class Weights>
KOKKOS_INLINE_FUNCTION constexpr bool write_gauss_legendre_sphere_ring_pair(unsigned n, unsigned j,
                                                                            const Longitudes& longitudes,
                                                                            Points& points, Weights& weights) {
  const unsigned m = 2 * n;  // longitudes per ring
  const SphereRing ring = gauss_legendre_sphere_ring(n, j);
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDouble two_pi_over_m = DoubleDouble(half_pi.hi, half_pi.lo) * 4.0 / static_cast<double>(m);
  const Scalar weight = static_cast<Scalar>((two_pi_over_m * ring.weight).hi());
  const Scalar z = static_cast<Scalar>(ring.z.hi());
  const unsigned north = j * m;
  const unsigned south = (n - 1 - j) * m;
  for (unsigned k = 0; k < m; ++k) {
    const Scalar x = static_cast<Scalar>((ring.sin_theta * longitudes[k].cos).hi());
    const Scalar y = static_cast<Scalar>((ring.sin_theta * longitudes[k].sin).hi());
    // The south ring is written first, so the middle ring of an odd rule keeps z = +0.
    points[3 * (south + k) + 0] = x;
    points[3 * (south + k) + 1] = y;
    points[3 * (south + k) + 2] = -z;
    weights[south + k] = weight;
    points[3 * (north + k) + 0] = x;
    points[3 * (north + k) + 1] = y;
    points[3 * (north + k) + 2] = z;
    weights[north + k] = weight;
  }
  return ring.converged;
}

/// \brief The points and weights of the N-ring Gauss-Legendre sphere rule.
template <class Scalar, unsigned N>
struct GaussLegendreSphereRule {
  Kokkos::Array<Scalar, 6 * N * N> points;  //!< (x, y, z) triples
  Kokkos::Array<Scalar, 2 * N * N> weights;
  bool converged = true;  //!< whether Newton's method reached every Gauss-Legendre root
};

/// \brief Build the N-ring Gauss-Legendre sphere rule. Rings come in mirror pairs, so only the northern half is built.
template <class Scalar, unsigned N>
KOKKOS_INLINE_FUNCTION constexpr GaussLegendreSphereRule<Scalar, N> make_gauss_legendre_sphere_rule() {
  Kokkos::Array<SinCos<DoubleDouble>, 2 * N> longitudes{};
  for (unsigned k = 0; k < 2 * N; ++k) {
    longitudes[k] = sin_cos_of_turn_fraction<DoubleDouble>(k, 2 * N);
  }
  GaussLegendreSphereRule<Scalar, N> rule{};
  for (unsigned j = 0; j < (N + 1) / 2; ++j) {
    rule.converged =
        write_gauss_legendre_sphere_ring_pair<Scalar>(N, j, longitudes, rule.points, rule.weights) && rule.converged;
  }
  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_GAUSSLEGENDRESPHEREIMPL_HPP_
