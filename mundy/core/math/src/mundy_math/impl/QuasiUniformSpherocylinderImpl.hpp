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

#ifndef MUNDY_MATH_IMPL_QUASIUNIFORMSPHEROCYLINDERIMPL_HPP_
#define MUNDY_MATH_IMPL_QUASIUNIFORMSPHEROCYLINDERIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <cstdint>    // for uint64_t
#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::invalid_argument

// Mundy
#include <mundy_math/DoubleDouble.hpp>           // for mundy::DoubleDouble, mundy::sqrt
#include <mundy_math/impl/DoubleDoubleImpl.hpp>  // for SinCos, sin_cos_taylor, sin_cos_of_turn_fraction, pi_over_2
#include <mundy_utils/throw_assert.hpp>          // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace impl {

//! \name Construction of quasi-uniform spherocylinder rules, at compile time or run time
//@{

/// \brief Sine and cosine of the golden angle pi * (3 - sqrt(5)), evaluated in double-double.
///
/// alpha = pi * (sqrt(5) - 2) is small and gamma = pi - alpha, so sin(gamma) = sin(alpha) and
/// cos(gamma) = -cos(alpha). Sixteen Taylor steps resolve both functions beyond double-double precision.
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDouble> golden_angle_sin_cos() {
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDouble pi = DoubleDouble(half_pi.hi, half_pi.lo) * 2.0;
  const DoubleDouble alpha = pi * (sqrt(DoubleDouble(5.0)) - 2.0);
  const DoubleDouble alpha2 = alpha * alpha;

  DoubleDouble sin_term = alpha;
  DoubleDouble cos_term = 1.0;
  DoubleDouble sin_alpha = sin_term;
  DoubleDouble cos_alpha = cos_term;

  for (unsigned n = 1; n <= 16; ++n) {
    sin_term = sin_term * (-alpha2 / double((2 * n) * (2 * n + 1)));
    cos_term = cos_term * (-alpha2 / double((2 * n - 1) * (2 * n)));
    sin_alpha = sin_alpha + sin_term;
    cos_alpha = cos_alpha + cos_term;
  }

  return {sin_alpha, -cos_alpha};
}

/// \brief Number of axial rings needed to match the circumferential node spacing.
KOKKOS_INLINE_FUNCTION constexpr unsigned compute_num_cylinder_rings(unsigned ntheta, double aspect_ratio) {
  if (aspect_ratio == 0.0) return 0u;
  const unsigned nz = static_cast<unsigned>(aspect_ratio * ntheta / (2.0 * Kokkos::numbers::pi_v<double>)+0.5);
  return nz > 0 ? nz : 1u;
}

/// \brief Number of equal-area Fibonacci nodes needed per hemispherical endcap.
KOKKOS_INLINE_FUNCTION constexpr unsigned compute_num_cap_points(unsigned ntheta) {
  const unsigned nc = static_cast<unsigned>(ntheta * double(ntheta) / (2.0 * Kokkos::numbers::pi_v<double>)+0.5);
  return nc > 0 ? nc : 1u;
}

/// \brief Multiply sine/cosine pairs corresponding to the sum of two angles.
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDouble> multiply_sin_cos(const SinCos<DoubleDouble>& lhs,
                                                                       const SinCos<DoubleDouble>& rhs) {
  return {lhs.sin * rhs.cos + lhs.cos * rhs.sin, lhs.cos * rhs.cos - lhs.sin * rhs.sin};
}

/// \brief Sine and cosine of k times the golden angle, using double-double binary exponentiation.
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDouble> golden_angle_multiple(unsigned k, SinCos<DoubleDouble> step) {
  SinCos<DoubleDouble> angle{0.0, 1.0};
  while (k != 0) {
    if (k & 1u) angle = multiply_sin_cos(angle, step);
    k >>= 1u;
    if (k != 0) step = multiply_sin_cos(step, step);
  }
  return angle;
}

/// \brief Write the i-th cylinder ring of the quasi-uniform spherocylinder rule, into points and weights.
template <class Scalar, class Points, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_quasi_uniform_spherocylinder_cylinder_ring(unsigned ntheta, unsigned nz,
                                                                                       double aspect_ratio, unsigned i,
                                                                                       Points& points,
                                                                                       Weights& weights) {
  const DoubleDouble z = aspect_ratio * ((DoubleDouble(i) + 0.5) / double(nz) - 0.5);
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDouble two_pi_over_ntheta = DoubleDouble(half_pi.hi, half_pi.lo) * 4.0 / double(ntheta);
  const Scalar weight = static_cast<Scalar>((two_pi_over_ntheta * aspect_ratio / double(nz)).hi());

  for (unsigned j = 0; j < ntheta; ++j) {
    const unsigned k = i * ntheta + j;
    const auto angle = sin_cos_of_turn_fraction<DoubleDouble>(2u * j + (i & 1u), 2u * ntheta);

    points[3u * k + 0u] = static_cast<Scalar>(angle.cos.hi());
    points[3u * k + 1u] = static_cast<Scalar>(angle.sin.hi());
    points[3u * k + 2u] = static_cast<Scalar>(z.hi());
    weights[k] = weight;
  }
}

/// \brief Write the i-th mirrored Fibonacci point pair on the north and south caps, into points and weights.
///
/// Both points have the same x and y coordinates, opposite z coordinates, and equal weights. The azimuth
/// is i times the golden angle, evaluated independently so distinct pairs can be generated in parallel.
template <class Scalar, class Points, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_quasi_uniform_spherocylinder_cap_pair(unsigned ntheta, unsigned nz,
                                                                                  unsigned ncap, double aspect_ratio,
                                                                                  unsigned i, Points& points,
                                                                                  Weights& weights) {
  constexpr auto step = golden_angle_sin_cos();
  const auto angle = golden_angle_multiple(i, step);
  const DoubleDouble mu = (DoubleDouble(i) + 0.5) / double(ncap);
  const DoubleDouble rho = sqrt((1.0 - mu) * (1.0 + mu));
  const UnevaluatedSum half_pi = pi_over_2();
  const DoubleDouble two_pi = DoubleDouble(half_pi.hi, half_pi.lo) * 4.0;
  const Scalar cap_weight = static_cast<Scalar>((two_pi / double(ncap)).hi());

  const Scalar x = static_cast<Scalar>((rho * angle.cos).hi());
  const Scalar y = static_cast<Scalar>((rho * angle.sin).hi());
  const Scalar z = static_cast<Scalar>((DoubleDouble(aspect_ratio) / 2.0 + mu).hi());

  const unsigned north = nz * ntheta + i;
  const unsigned south = nz * ntheta + ncap + i;
  points[3u * north + 0u] = x;
  points[3u * north + 1u] = y;
  points[3u * north + 2u] = z;
  weights[north] = cap_weight;
  points[3u * south + 0u] = x;
  points[3u * south + 1u] = y;
  points[3u * south + 2u] = -z;
  weights[south] = cap_weight;
}

/// \brief Unit-radius spherocylinder quadrature with Ntheta circumferential nodes per cylinder ring.
///
/// The cylinder has length AspectRatio; the number of axial rings and the number of Fibonacci nodes on each
/// hemisphere are determined by matching nominal spacing to h = 2 pi / Ntheta.
///   Nz = round(AspectRatio / h),
///   Ncap = round(2 pi / h^2).
/// Points are (x,y,z) triples; all weights are surface weights for this unit-radius geometry.
template <class Scalar, unsigned Ntheta, double AspectRatio>
struct QuasiUniformSpherocylinderRule {
  static_assert(Ntheta >= 3);
  static_assert(AspectRatio >= 0.0);

  static constexpr unsigned num_points_per_ring = Ntheta;
  static constexpr unsigned num_cylinder_rings = compute_num_cylinder_rings(Ntheta, AspectRatio);
  static constexpr unsigned num_cap_points = compute_num_cap_points(Ntheta);
  static constexpr unsigned num_points = num_cylinder_rings * num_points_per_ring + 2 * num_cap_points;

  Kokkos::Array<Scalar, 3u * num_points> points;  //!< (x,y,z) triples
  Kokkos::Array<Scalar, num_points> weights;
};

/// \brief Build the quasi-uniform spherocylinder rule at compile time, for the given circumferential node count
template <class Scalar, unsigned Ntheta, double AspectRatio>
KOKKOS_INLINE_FUNCTION constexpr QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio>
make_quasi_uniform_spherocylinder_rule() {
  using Rule = QuasiUniformSpherocylinderRule<Scalar, Ntheta, AspectRatio>;
  constexpr unsigned Nz = Rule::num_cylinder_rings;
  constexpr unsigned Ncap = Rule::num_cap_points;

  Rule rule{};

  // Cylinder: uniform axial midpoints and staggered azimuthal nodes.
  for (unsigned i = 0; i < Nz; ++i) {
    write_quasi_uniform_spherocylinder_cylinder_ring<Scalar>(Ntheta, Nz, AspectRatio, i, rule.points, rule.weights);
  }

  // Caps: uniform in mu = |cos(polar angle)| with golden-angle azimuthal increments.
  for (unsigned i = 0; i < Ncap; ++i) {
    write_quasi_uniform_spherocylinder_cap_pair<Scalar>(Ntheta, Nz, Ncap, AspectRatio, i, rule.points, rule.weights);
  }

  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_QUASIUNIFORMSPHEROCYLINDERIMPL_HPP_
