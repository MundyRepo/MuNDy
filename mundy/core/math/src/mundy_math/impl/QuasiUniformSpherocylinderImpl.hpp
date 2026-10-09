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

//! \name Resolutions and geometry
//@{

struct QuasiUniformSpherocylinderCounts {
  unsigned num_longitudes;
  unsigned num_cylinder_rings;
  unsigned num_cap_points;
};

/// \brief Whether the counts describe a rule whose flattened indices and longitude arithmetic fit in unsigned.
KOKKOS_INLINE_FUNCTION constexpr bool valid_spherocylinder_counts(unsigned n_theta, unsigned n_z, unsigned n_cap) {
  constexpr unsigned limit = std::numeric_limits<unsigned>::max();
  const uint64_t num_points = uint64_t(n_theta) * n_z + 2 * uint64_t(n_cap);
  return n_theta >= 3 && n_theta <= limit / 8 && n_cap >= 1 && num_points <= limit / 3;
}

KOKKOS_INLINE_FUNCTION constexpr void validate_spherocylinder_counts(unsigned n_theta, unsigned n_z, unsigned n_cap) {
  MUNDY_THROW_REQUIRE(valid_spherocylinder_counts(n_theta, n_z, n_cap), std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: requires at least 3 longitudes and 1 point per cap, "
                      "with counts small enough for unsigned indexing.");
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr void validate_spherocylinder_geometry(unsigned n_z, Scalar radius, Scalar length) {
  constexpr Scalar largest = std::numeric_limits<Scalar>::max();
  MUNDY_THROW_REQUIRE(radius > Scalar(0) && radius <= largest && length >= Scalar(0) && length <= largest,
                      std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: radius must be positive, length nonnegative, "
                      "and both finite.");
  MUNDY_THROW_REQUIRE(n_z > 0 || length == Scalar(0), std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: a positive cylinder length needs at least one ring.");
}

/// \brief Choose approximately equal area per node and comparable axial and circumferential spacings.
///
/// If c is the number of nodes per cap, the desired spacings give n_theta ~ sqrt(2 pi c),
/// n_z ~ (length / radius) sqrt(c / (2 pi)), and c ~ target / (length / radius + 2).
template <class Scalar>
KOKKOS_INLINE_FUNCTION QuasiUniformSpherocylinderCounts quasi_uniform_spherocylinder_counts(unsigned target,
                                                                                            Scalar radius,
                                                                                            Scalar length) {
  validate_spherocylinder_geometry(1, radius, length);
  MUNDY_THROW_REQUIRE(target > 0, std::invalid_argument, "quasi_uniform_spherocylinder_rule: target must be positive.");
  const double aspect = static_cast<double>(length) / static_cast<double>(radius);
  const double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;
  const double cap_count = static_cast<double>(target) / (aspect + 2.0);
  const double per_length = Kokkos::sqrt(cap_count / two_pi);
  const double theta_count = Kokkos::floor(two_pi * per_length + 0.5);
  const double ring_count = Kokkos::floor(aspect * per_length + 0.5);
  const double rounded_cap_count = Kokkos::floor(cap_count + 0.5);
  constexpr double limit = std::numeric_limits<unsigned>::max();
  // Check before floating-to-integer conversion; these comparisons also reject infinities and NaNs.
  MUNDY_THROW_REQUIRE(theta_count <= limit && ring_count <= limit && rounded_cap_count <= limit, std::invalid_argument,
                      "quasi_uniform_spherocylinder_rule: requested resolution is too large.");
  const unsigned n_theta = theta_count < 8.0 ? 8u : static_cast<unsigned>(theta_count);
  const unsigned n_z = length == Scalar(0) ? 0u : (ring_count < 1.0 ? 1u : static_cast<unsigned>(ring_count));
  const unsigned n_cap = rounded_cap_count < 8.0 ? 8u : static_cast<unsigned>(rounded_cap_count);
  validate_spherocylinder_counts(n_theta, n_z, n_cap);
  return {n_theta, n_z, n_cap};
}
//@}

//! \name Reference nodes and physical geometry
//@{

/// \brief Fibonacci azimuth k pi (3 - sqrt(5)), without a growing trigonometric argument.
///
/// Binary exponentiation permits independent cap nodes. Double-double arithmetic keeps the accumulated rotation
/// error below the output precision even for large k; the base rotation is computed rather than rounded in advance.
KOKKOS_INLINE_FUNCTION constexpr SinCos<DoubleDouble> fibonacci_sin_cos(unsigned k) {
  constexpr auto half_pi = pi_over_2();
  constexpr auto small_angle =
      sin_cos_taylor((sqrt(DoubleDouble(5.0)) - 2.0) * (DoubleDouble(half_pi.hi, half_pi.lo) * 2.0));
  SinCos<DoubleDouble> power{small_angle.sin, -small_angle.cos};
  SinCos<DoubleDouble> result{DoubleDouble(0.0), DoubleDouble(1.0)};
  while (k != 0) {
    if (k & 1u) {
      result = {result.sin * power.cos + result.cos * power.sin,  //
                result.cos * power.cos - result.sin * power.sin};
    }
    k >>= 1u;
    if (k != 0) {
      power = {2.0 * power.sin * power.cos, power.cos * power.cos - power.sin * power.sin};
    }
  }
  return result;
}

/// \brief Reference node i: cylinder radius 1 and length 1; unit hemispheres centered at the origin.
template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Array<Scalar, 3> spherocylinder_reference_point(unsigned n_theta, unsigned n_z,
                                                                                         unsigned n_cap, unsigned i) {
  const unsigned cylinder_points = n_theta * n_z;
  if (i < cylinder_points) {
    const unsigned ring = i / n_theta;
    const unsigned longitude = i % n_theta;
    const auto angle = sin_cos_of_turn_fraction<DoubleDouble>(2 * longitude + 1 + (ring & 1u), 2 * n_theta);
    const DoubleDouble z = (DoubleDouble(static_cast<double>(ring)) + 0.5) / static_cast<double>(n_z) - 0.5;
    return {static_cast<Scalar>(angle.cos.hi()), static_cast<Scalar>(angle.sin.hi()), static_cast<Scalar>(z.hi())};
  }
  const unsigned cap_index = i - cylinder_points;
  const unsigned k = cap_index % n_cap;
  const DoubleDouble z = (DoubleDouble(static_cast<double>(k)) + 0.5) / static_cast<double>(n_cap);
  const DoubleDouble rho = sqrt((1.0 - z) * (1.0 + z));
  const auto angle = fibonacci_sin_cos(k);
  return {static_cast<Scalar>((rho * angle.cos).hi()),  //
          static_cast<Scalar>((rho * angle.sin).hi()),  //
          cap_index < n_cap ? static_cast<Scalar>(z.hi()) : -static_cast<Scalar>(z.hi())};
}

/// \brief Map a reference point to a radius-r, cylindrical-length-L surface. Caps shift by +/-L/2.
template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Array<Scalar, 3> spherocylinder_physical_point(bool cylinder, Scalar radius,
                                                                                        Scalar length, Scalar x,
                                                                                        Scalar y, Scalar z) {
  // Round the cap's multiply-and-add once, consistently in constexpr, host, and device evaluation.
  DoubleDouble physical_z = DoubleDouble(static_cast<double>(z)) * static_cast<double>(cylinder ? length : radius);
  if (!cylinder) {
    physical_z += DoubleDouble(static_cast<double>(length)) * (z > Scalar(0) ? 0.5 : -0.5);
  }
  return {radius * x, radius * y, static_cast<Scalar>(physical_z.hi())};
}

/// \brief Surface area per node on one patch. Cylinder weights vanish in the sphere limit.
template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar spherocylinder_patch_weight(unsigned count, bool cylinder, Scalar radius,
                                                                    Scalar length) {
  const auto half_pi = pi_over_2();
  const DoubleDouble two_pi = DoubleDouble(half_pi.hi, half_pi.lo) * 4.0;
  return static_cast<Scalar>((two_pi / static_cast<double>(count) * static_cast<double>(radius) *
                              static_cast<double>(cylinder ? length : radius))
                                 .hi());
}

/// \brief Write one physical point and its patch weight, for host or device storage.
template <class Scalar, class Points, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_quasi_uniform_spherocylinder_point(unsigned n_theta, unsigned n_z,
                                                                               unsigned n_cap, unsigned i,
                                                                               Scalar radius, Scalar length,
                                                                               Points& points, Weights& weights) {
  const unsigned cylinder_points = n_theta * n_z;
  const auto reference = spherocylinder_reference_point<Scalar>(n_theta, n_z, n_cap, i);
  const auto point = spherocylinder_physical_point(i < cylinder_points, radius, length,  //
                                                   reference[0], reference[1], reference[2]);
  for (unsigned d = 0; d < 3; ++d) {
    points[3 * i + d] = point[d];
  }
  weights[i] = spherocylinder_patch_weight(i < cylinder_points ? cylinder_points : n_cap,  //
                                           i < cylinder_points, radius, length);
}

/// \brief Compile-time reference points, ordered cylinder, north cap, south cap.
template <class Scalar, unsigned NTheta, unsigned NZ, unsigned NCap>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Array<Scalar, 3 * (NTheta * NZ + 2 * NCap)>
make_quasi_uniform_spherocylinder_points() {
  Kokkos::Array<Scalar, 3 * (NTheta * NZ + 2 * NCap)> points{};
  for (unsigned i = 0; i < NTheta * NZ + 2 * NCap; ++i) {
    const auto point = spherocylinder_reference_point<Scalar>(NTheta, NZ, NCap, i);
    for (unsigned d = 0; d < 3; ++d) {
      points[3 * i + d] = point[d];
    }
  }
  return points;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_QUASIUNIFORMSPHEROCYLINDERIMPL_HPP_
