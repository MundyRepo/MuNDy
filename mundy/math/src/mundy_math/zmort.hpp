// @HEADER
// **********************************************************************************************************************
//
// This file is part of zorder_knn. (We gleefully stole this code, making minor modifications to improve portability.
// The original code is located here: https://github.com/sebastianlipponer/zorder_knn)
// It is licensed under the following license:
//
// Copyright(c) 2010, 2021 Sebastian Lipponer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_MATH_ZMORT_HPP_
#define MUNDY_MATH_ZMORT_HPP_

// C++ core
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

// External
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>  // MUNDY_THROW_ASSERT

namespace zorder_knn {

namespace detail {

template <typename To, typename From>
KOKKOS_INLINE_FUNCTION constexpr To constexpr_bit_cast(const From& from) {
  static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
  static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");
  static_assert(sizeof(To) == sizeof(From), "To and From must have the same size");
  static_assert(alignof(To) <= alignof(From), "To must be aligned at least as much as From");

  const union {
    const From* src;
    const To* dst;
  } result = {&from};
  return *result.dst;
}

KOKKOS_INLINE_FUNCTION uint32_t FloatToUInt(float x) {
  static_assert(sizeof(float) == sizeof(uint32_t), "sizeof(float) != sizeof(uint32_t)");
  return constexpr_bit_cast<uint32_t>(x);  // TODO(palmerb4): Swap to Kokkos::bit_cast
}

KOKKOS_INLINE_FUNCTION uint64_t FloatToUInt(double x) {
  static_assert(sizeof(double) == sizeof(uint64_t), "sizeof(double) != sizeof(uint64_t)");
  return constexpr_bit_cast<uint64_t>(x);
}

KOKKOS_INLINE_FUNCTION int FloatExp(uint32_t xi) {
  // ignore sign bit
  auto uxi = xi & 0x7fffffffu;

  // 0, inf, nan
  if (uxi == 0 || uxi >= 0x7f800000u) return 0;

  // ignore significand
  uxi = uxi >> 23;

  int exp = (uxi == 0) ? -126 : static_cast<int>(uxi) - 127;
  return exp;
}

KOKKOS_INLINE_FUNCTION int FloatExp(uint64_t xi) {
  // ignore sign bit
  auto uxi = xi & 0x7fffffffffffffffll;

  // 0, inf, nan
  if (uxi == 0 || uxi >= 0x7ff0000000000000ll) return 0;

  // ignore significand
  uxi = uxi >> 52;

  int exp = (uxi == 0) ? -1022 : static_cast<int>(uxi) - 1023;
  return exp;
}

KOKKOS_INLINE_FUNCTION auto FloatSig(uint32_t xi) -> decltype(xi) {
  return xi & 0x007fffff;
}

KOKKOS_INLINE_FUNCTION auto FloatSig(uint64_t xi) -> decltype(xi) {
  return xi & 0x000fffffffffffffll;
}

constexpr int8_t log0_nan = Kokkos::Experimental::finite_min_v<int8_t>;

static constexpr decltype(log0_nan) log_base2_table[256] = {
    log0_nan, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5,        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6,        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6,        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7,        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

KOKKOS_INLINE_FUNCTION auto UIntLogBase2(uint32_t x) -> decltype(log0_nan) {
  MUNDY_THROW_ASSERT(x != 0, std::invalid_argument, "x must not be zero");

  auto log_base2 = log0_nan;
  auto x16 = x >> 16;

  if (x16) {
    auto x8 = x16 >> 8;
    log_base2 = (x8) ? log_base2_table[x8] + 24 : log_base2_table[x16] + 16;
  } else {
    auto x8 = x >> 8;
    log_base2 = (x8) ? log_base2_table[x8] + 8 : log_base2_table[x];
  }

  return log_base2;
}

KOKKOS_INLINE_FUNCTION auto UIntLogBase2(uint64_t x) -> decltype(log0_nan) {
  auto x32 = static_cast<uint32_t>(x >> 32);
  auto log_base2 = (x32) ? UIntLogBase2(x32) + 32 : UIntLogBase2(static_cast<uint32_t>(x));

  return log_base2;
}

template <typename T>
struct significand;

template <>
struct significand<float> {
  static constexpr uint8_t nbits = 23;
};

template <>
struct significand<double> {
  static constexpr uint8_t nbits = 52;
};

template <typename Scalar>
KOKKOS_INLINE_FUNCTION auto FloatXorMsb(Scalar p, Scalar q) -> decltype(FloatExp(FloatToUInt(p))) {
  if (p == q || p == -q) {
    return Kokkos::Experimental::finite_min_v<int>;
  }

  auto pui = FloatToUInt(p);
  auto qui = FloatToUInt(q);

  auto p_exp = FloatExp(pui);
  auto q_exp = FloatExp(qui);

  if (p_exp == q_exp) {
    auto xor_psig_qsig = FloatSig(pui) ^ FloatSig(qui);

    if (xor_psig_qsig > 0)
      return p_exp + UIntLogBase2(xor_psig_qsig) - significand<Scalar>::nbits;
    else
      return p_exp;
  }

  return Kokkos::max(p_exp, q_exp);
}

}  // namespace detail

// The relative z-order of two points is determined by the pair of
// coordinates who have the first differing bit with the highest
// exponent.
template <typename Point, std::size_t d>
struct Less {
  bool operator()(Point const& p, Point const& q) const {
    using Scalar = decltype(p[0]);
    constexpr auto zero = Scalar(0.0);

    auto x = Kokkos::Experimental::finite_min_v<int>;
    std::size_t k{0};

    // Starting from j = 0 generates a N- instead of a Z-curve.
    for (std::size_t j{d}; j-- > 0;) {
      if ((p[j] < zero) != (q[j] < zero)) {
        return p[j] < q[j];
      }

      auto y = detail::FloatXorMsb(p[j], q[j]);

      if (x < y) {
        x = y;
        k = j;
      }
    }

    return p[k] < q[k];
  }
};

}  // namespace zorder_knn

namespace mundy {

namespace math {

template <ValidVector3Type Vector3Type1, ValidVector3Type Vector3Type2>
bool zmorton_less(const Vector3Type1& p, const Vector3Type2& q) {
  using Scalar = typename Vector3Type1::scalar_t;
  constexpr auto zero = Scalar(0.0);

  // Signed less then
  // If the signs are different, return if the first is less than the second.
  Vector3<int> signed_less_than{((p[0] < zero) != (q[0] < zero)) ? (p[0] < q[0]) : -1,
                                ((p[1] < zero) != (q[1] < zero)) ? (p[1] < q[1]) : -1,
                                ((p[2] < zero) != (q[2] < zero)) ? (p[2] < q[2]) : -1};

  // Determine the most significant bit (only valid if signs are the same)
  int x = Kokkos::Experimental::finite_min_v<int>;
  std::size_t k{0};
  auto y0 = zorder_knn::detail::FloatXorMsb(p[0], q[0]);
  auto y1 = zorder_knn::detail::FloatXorMsb(p[1], q[1]);
  auto y2 = zorder_knn::detail::FloatXorMsb(p[2], q[2]);

  if (x < y0) {
    x = y0;
    k = 0;
  }

  if (x < y1) {
    x = y1;
    k = 1;
  }

  if (x < y2) {
    x = y2;
    k = 2;
  }

  return (signed_less_than[2] != -1)
             ? signed_less_than[2]
             : ((signed_less_than[1] != -1) ? signed_less_than[1]
                                            : ((signed_less_than[0] != -1) ? signed_less_than[0] : (p[k] < q[k])));
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_ZMORT_HPP_
