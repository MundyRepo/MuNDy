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

#ifndef MUNDY_MATH_TOLERANCE_HPP_
#define MUNDY_MATH_TOLERANCE_HPP_

// External
#include <Kokkos_Core.hpp>  // for KOKKOS_INLINE_FUNCTION

// C++ core
#include <type_traits>  // for std::is_same_v, std::is_floating_point_v, std::common_type_t

// Mundy
#include <mundy_math/cmath.hpp>      // for mundy::passive_scalar_t
#include <mundy_utils/requires.hpp>  // for MUNDY_REQUIRES

namespace mundy {

/// \brief Function to get the zero tolerance for a type. That is, the smallest value that we will consider non-zero.
/// We use approximately 10 * std::numeric_limits<T>::epsilon() as the default tolerance for floats and doubles and 0
/// for integer types. To make this code GPU-compatable, we'll directly evaluate the epsilon instead of using
/// std::numeric_limits.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
MUNDY_REQUIRES(std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_zero_tolerance() {
  using cT = passive_scalar_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<cT, float>) {
    return 1e-6f;
  } else if constexpr (std::is_same_v<cT, double>) {
    return 1e-15;
  } else {
    // For integral types, tolerance doesn't make sense, return 0
    return cT(0);
  }
}

/// \brief Function to get the relaxed zero tolerance for a type. That is, the smallest value that we will consider
/// non-zero. Our choice of relaxed tolerance is based on personal preference, not on a hard standard and is mostly used
/// during testing. To make this code GPU-compatable, we'll directly evaluate the epsilon instead of using
/// std::numeric_limits.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
MUNDY_REQUIRES(std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_relaxed_zero_tolerance() {
  using cT = passive_scalar_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<cT, float>) {
    return 1e-3f;
  } else if constexpr (std::is_same_v<cT, double>) {
    return 1e-8;
  } else {
    // For integral types, tolerance doesn't make sense, return 0
    return cT(0);
  }
}

/// \brief The tolerance to use when comparing two scalar types. The choice is made on the passive
/// (underlying) type, so a custom scalar such as an autodiff dual resolves to its passive tolerance.
template <typename T1, typename T2>
MUNDY_REQUIRES(std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T1>>>&&
                   std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T2>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_comparison_tolerance() {
  // Both floating point: the smaller type. One floating point and one integer: the floating point type.
  // Both integers: their common type.
  using cT1 = passive_scalar_t<std::remove_reference_t<T1>>;
  using cT2 = passive_scalar_t<std::remove_reference_t<T2>>;

  if constexpr (std::is_floating_point_v<cT1> && std::is_floating_point_v<cT2>) {
    using T = std::conditional_t<(sizeof(cT1) < sizeof(cT2)), cT1, cT2>;
    return get_zero_tolerance<T>();
  } else if constexpr (std::is_floating_point_v<cT1> && std::is_integral_v<cT2>) {
    return get_zero_tolerance<cT1>();
  } else if constexpr (std::is_integral_v<cT1> && std::is_floating_point_v<cT2>) {
    return get_zero_tolerance<cT2>();
  } else {
    using T = std::common_type_t<cT1, cT2>;
    return get_zero_tolerance<T>();
  }
}

/// \brief The relaxed tolerance to use when comparing two scalar types. Like \ref get_comparison_tolerance,
/// the choice is made on the passive (underlying) type.
template <typename T1, typename T2>
MUNDY_REQUIRES(std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T1>>>&&
                   std::is_arithmetic_v<passive_scalar_t<std::remove_reference_t<T2>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_relaxed_comparison_tolerance() {
  using T = decltype(get_comparison_tolerance<T1, T2>());
  return get_relaxed_zero_tolerance<T>();
}

}  // namespace mundy

#endif  // MUNDY_MATH_TOLERANCE_HPP_
