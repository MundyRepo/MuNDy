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

#ifndef MUNDY_MATH_IMPL_ARRAYIMPL_HPP_
#define MUNDY_MATH_IMPL_ARRAYIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

namespace mundy {

namespace math {

/// \brief A simplistic array type with a fixed size and type
template <typename T, size_t N>
class Array;

namespace impl {

/// \brief Deep copy implementation for Array
template <size_t... Is, typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr void deep_copy_impl(std::index_sequence<Is...>, Array<T, N>& array,
                                                     const Array<T, N>& other) {
  ((array[Is] = other[Is]), ...);
}

/// \brief Fill implementation for Array
template <size_t... Is, typename T, size_t N>
KOKKOS_INLINE_FUNCTION constexpr void fill_impl(std::index_sequence<Is...>, Array<T, N>& array, const T& value) {
  ((array[Is] = value), ...);
}

}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_ARRAYIMPL_HPP_
