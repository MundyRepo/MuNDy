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
#include <type_traits>  // for std::is_same_v, std::conditional_t, std::common_type_t

// Mundy
#include <mundy_math/NumTraits.hpp>  // for mundy::NumTraits, mundy::is_passive_scalar_v
#include <mundy_math/cmath.hpp>      // for mundy::passive_scalar_t
#include <mundy_utils/requires.hpp>  // for MUNDY_REQUIRES

namespace mundy {

/// \brief Function to get the zero tolerance for a type. That is, the smallest value that we will consider non-zero.
/// We use approximately 10 * epsilon: fixed values for float and double, 10 * NumTraits<T>::epsilon() for any other
/// non-integer passive scalar, and 0 for integer types. To make this code GPU-compatable, we'll directly evaluate the
/// epsilon instead of using std::numeric_limits.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
MUNDY_REQUIRES(is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_zero_tolerance() {
  using cT = passive_scalar_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<cT, float>) {
    return 1e-6f;
  } else if constexpr (std::is_same_v<cT, double>) {
    return 1e-15;
  } else if constexpr (NumTraits<cT>::IsInteger) {
    return cT(0);  // for integral types, tolerance doesn't make sense
  } else {
    return cT(10) * NumTraits<cT>::epsilon();
  }
}

/// \brief Function to get the relaxed zero tolerance for a type. That is, the smallest value that we will consider
/// non-zero. Our choice of relaxed tolerance is based on personal preference, not on a hard standard and is mostly used
/// during testing: fixed values for float and double, NumTraits<T>::dummy_precision() for any other non-integer
/// passive scalar, and 0 for integer types.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
MUNDY_REQUIRES(is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_relaxed_zero_tolerance() {
  using cT = passive_scalar_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<cT, float>) {
    return 1e-3f;
  } else if constexpr (std::is_same_v<cT, double>) {
    return 1e-8;
  } else if constexpr (NumTraits<cT>::IsInteger) {
    return cT(0);  // for integral types, tolerance doesn't make sense
  } else {
    return NumTraits<cT>::dummy_precision();
  }
}

/// \brief The tolerance to use when comparing two scalar types. The choice is made on the passive
/// (underlying) type, so a custom scalar such as an autodiff dual resolves to its passive tolerance.
template <typename T1, typename T2>
MUNDY_REQUIRES(is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T1>>>&&
                   is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T2>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_comparison_tolerance() {
  // Both non-integer: the coarser type (larger epsilon). One integer: the other type. Both integers: their common type.
  using cT1 = passive_scalar_t<std::remove_reference_t<T1>>;
  using cT2 = passive_scalar_t<std::remove_reference_t<T2>>;
  constexpr bool integer1 = NumTraits<cT1>::IsInteger;
  constexpr bool integer2 = NumTraits<cT2>::IsInteger;

  if constexpr (!integer1 && !integer2) {
    constexpr bool cT1_is_coarser =
        static_cast<double>(NumTraits<cT1>::epsilon()) >= static_cast<double>(NumTraits<cT2>::epsilon());
    return get_zero_tolerance<std::conditional_t<cT1_is_coarser, cT1, cT2>>();
  } else if constexpr (!integer1) {
    return get_zero_tolerance<cT1>();
  } else if constexpr (!integer2) {
    return get_zero_tolerance<cT2>();
  } else {
    return get_zero_tolerance<std::common_type_t<cT1, cT2>>();
  }
}

/// \brief The relaxed tolerance to use when comparing two scalar types. Like \ref get_comparison_tolerance,
/// the choice is made on the passive (underlying) type.
template <typename T1, typename T2>
MUNDY_REQUIRES(is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T1>>>&&
                   is_passive_scalar_v<passive_scalar_t<std::remove_reference_t<T2>>>)
KOKKOS_INLINE_FUNCTION constexpr auto get_relaxed_comparison_tolerance() {
  using T = decltype(get_comparison_tolerance<T1, T2>());
  return get_relaxed_zero_tolerance<T>();
}

}  // namespace mundy

#endif  // MUNDY_MATH_TOLERANCE_HPP_
