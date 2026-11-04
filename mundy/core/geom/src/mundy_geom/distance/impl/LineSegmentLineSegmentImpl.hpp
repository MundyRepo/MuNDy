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

#ifndef MUNDY_GEOM_DISTANCE_IMPL_LINESEGMENTLINESEGMENTIMPL_HPP_
#define MUNDY_GEOM_DISTANCE_IMPL_LINESEGMENTLINESEGMENTIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3, mundy::norm
#include <mundy_math/cmath.hpp>    // for mundy::abs, mundy::passive_scalar_t, mundy::impl::passive_value

namespace mundy {

namespace impl {

/// \brief Clamp an arch-length parameter to the segment, [0, 1]
/// \tparam Scalar The scalar type
/// \param[in] x The parameter
/// \internal
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar clamp_to_unit_interval(const Scalar& x) {
  return x < static_cast<Scalar>(0.0) ? static_cast<Scalar>(0.0)
                                      : (x > static_cast<Scalar>(1.0) ? static_cast<Scalar>(1.0) : x);
}

/// \brief The power of two 2^j, exactly, for |j| small enough that 2^j is a normal number
/// \tparam T The passive scalar type
/// \param[in] j The exponent
/// \internal
template <typename T>
KOKKOS_INLINE_FUNCTION T pow2(const int j) {
  T base = j >= 0 ? static_cast<T>(2.0) : static_cast<T>(0.5);
  T result = static_cast<T>(1.0);
  for (int n = j >= 0 ? j : -j; n != 0;) {
    if (n & 1) {
      result *= base;
    }
    n >>= 1;
    if (n != 0) {
      base *= base;
    }
  }
  return result;
}

/// \brief The exponent k with x 2^-k in [1, 2), for finite x > 0 (subnormal x included)
/// \tparam T The passive scalar type
/// \param[in] x The value
/// \internal
template <typename T>
KOKKOS_INLINE_FUNCTION int binary_exponent(T x) {
  constexpr int steps[4] = {512, 64, 8, 1};
  int k = 0;
  for (const int step : steps) {
    const T up = pow2<T>(step);
    const T down = pow2<T>(-step);
    while (x >= up) {
      x *= down;
      k += step;
    }
  }
  for (const int step : steps) {
    const T up = pow2<T>(step);
    while (x * up < static_cast<T>(2.0)) {
      x *= up;
      k -= step;
    }
  }
  return k;
}

/// \brief x 2^-k, exact unless the result is subnormal
/// \tparam T The passive scalar type
/// \tparam X The (possibly-AD) scalar or vector type
/// \param[in] x The value
/// \param[in] k The exponent, |k| < 1100
/// \internal
template <typename T, typename X>
KOKKOS_INLINE_FUNCTION X scale_by_pow2(const X& x, const int k) {
  // 2^-k itself need not be representable (k = -1074), so it is applied as two normal factors.
  const int half = -k / 2;
  return (x * pow2<T>(half)) * pow2<T>(-k - half);
}

/// \brief The largest magnitude among the passive values of three vectors' components
/// \tparam T The passive scalar type
/// \param[in] x, y, z The vectors
/// \internal
template <typename T, typename V1, typename V2, typename V3>
KOKKOS_INLINE_FUNCTION T max_abs_component(const V1& x, const V2& y, const V3& z) {
  T m = static_cast<T>(0.0);
  for (int i = 0; i < 3; ++i) {
    const T xi = mundy::abs(passive_value(x[i]));
    const T yi = mundy::abs(passive_value(y[i]));
    const T zi = mundy::abs(passive_value(z[i]));
    m = xi > m ? xi : m;
    m = yi > m ? yi : m;
    m = zi > m ? zi : m;
  }
  return m;
}

/// \brief |x|, without overflow or underflow in the sum of squares
/// \tparam Scalar The (possibly-AD) scalar type
/// \param[in] x The vector
/// \internal
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar scaled_norm(const mundy::Vector3<Scalar>& x) {
  using T = passive_scalar_t<Scalar>;
  const T m = max_abs_component<T>(x, x, x);
  if (!(m > static_cast<T>(0.0)) || !(m < mundy_infinity_v<T>)) {
    return mundy::norm(x);
  }
  const int k = binary_exponent(m);
  return scale_by_pow2<T>(mundy::norm(scale_by_pow2<T>(x, k)), -k);
}

/// \brief floor(p / q) for q > 0
/// \internal
KOKKOS_INLINE_FUNCTION int floor_div(const int p, const int q) {
  return p >= 0 ? p / q : -((-p + q - 1) / q);
}

/// \brief ceil(p / q) for q > 0
/// \internal
KOKKOS_INLINE_FUNCTION int ceil_div(const int p, const int q) {
  return -floor_div(-p, q);
}

/// \brief The exponent k for which u 2^-k, v 2^-k, and w 2^-k give a = u.u, b = u.v, c = v.v, d = u.w, e = v.w,
/// u x v, |u x v|^2, and (u x v) . (v x w) without overflow, with |u| 2^-k and |v| 2^-k centred about 1
///
/// Let 2^eu, 2^ev, 2^ew bound the largest components of u, v, w from below (to a factor 2). The centred choice is
/// k = floor((eu + ev) / 2), which gives |u x v|^2 ~ sin^2 of the angle between u and v and keeps a and c normal while
/// |eu - ev| <= 1018. It is raised only as far as needed to keep every product finite:
///   a, c:             2 (max(eu, ev) - k) + 4 <= 1023
///   d, e, v x w:      max(eu, ev) + ew - 2 k + 4 <= 1023
///   w:                ew - k + 1 <= 1023
///   u x v, D:         2 (eu + ev) - 4 k + 8 <= 1023
///   sN:               eu + 2 ev + ew - 4 k + 8 <= 1023
/// A zero vector imposes no bound. When u or v is zero, the other one centres the scale.
/// \tparam T The passive scalar type
/// \param[in] mu, mv, mw The largest component magnitudes of u, v, w (finite, >= 0, mu + mv > 0)
/// \internal
template <typename T>
KOKKOS_INLINE_FUNCTION int centred_scale_exponent(const T mu, const T mv, const T mw) {
  const bool has_u = mu > static_cast<T>(0.0);
  const bool has_v = mv > static_cast<T>(0.0);
  const bool has_w = mw > static_cast<T>(0.0);
  const int eu = has_u ? binary_exponent(mu) : binary_exponent(mv);
  const int ev = has_v ? binary_exponent(mv) : eu;
  const int ehi = eu > ev ? eu : ev;
  int k = floor_div(eu + ev, 2);
  k = Kokkos::max(k, ehi - 509);
  k = Kokkos::max(k, ceil_div(2 * (eu + ev) - 1015, 4));
  if (has_w) {
    const int ew = binary_exponent(mw);
    k = Kokkos::max(k, ceil_div(ehi + ew - 1019, 2));
    k = Kokkos::max(k, ew - 1022);
    if (has_u && has_v) {
      k = Kokkos::max(k, ceil_div(eu + 2 * ev + ew - 1015, 4));
    }
  }
  return k;
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_IMPL_LINESEGMENTLINESEGMENTIMPL_HPP_
