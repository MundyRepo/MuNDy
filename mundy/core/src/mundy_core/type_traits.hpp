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

#ifndef MUNDY_CORE_TYPE_TRAITS_HPP_
#define MUNDY_CORE_TYPE_TRAITS_HPP_

// C++ core
#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

namespace mundy {

namespace core {

/* What all do we offer

  - contains_type<T, Types...>   : check if T is in Types...
  - count_type<T, Types...>      : count how many times T appears in Types...
  - index_finder<T, Types...>    : find the index of T in Types... (only valid if all types are unique and the type is
  present)
  - type_at_index_t<I, Types...> : get the I'th type in Types...

*/

// **********************************************************************************************************************
/// \brief Check if a type is in a variadic list of types
template <class T, class... Ts>
struct contains_type {
  static constexpr bool value = (std::is_same_v<T, Ts> || ...);
};
//
template <class T, class... Ts>
static constexpr bool contains_type_v = contains_type<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Count how many times a type appears in a variadic list of types
template <class T, class... Types>
struct count_type {
  static constexpr size_t value = (0 + ... + (std::is_same_v<T, Types> ? 1 : 0));
};
template <class T, class... Types>
static constexpr size_t count_type_v = count_type<T, Types...>::value;

// **********************************************************************************************************************
/// \brief Find the index in the variadic list of types that matches the given type
template <class T, class... Ts>
struct index_finder;
//
template <class T, class First, class... Rest>
struct index_finder<T, First, Rest...> {
  static_assert(count_type_v<T, First, Rest...> == 1, "Type must appear exactly once in list");
  static_assert(sizeof...(Rest) + 1 > 0, "Type not found in list");
  static constexpr size_t value = std::is_same_v<T, First> ? 0 : 1 + index_finder<T, Rest...>::value;
};
//
template <class T>
struct index_finder<T> {
  static constexpr size_t value = 0;
};
//
template <class T, class... Ts>
static constexpr size_t index_finder_v = index_finder<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Get the I'th type in a variadic list of types
template <size_t I, class... Ts>
struct type_at_index;
//
// Primary template for getting the I-th type
template <std::size_t I, typename Head, typename... Tail>
struct type_at_index {
    static_assert(I < 1 + sizeof...(Tail), "Index out of bounds in type_at_index");
    using type = typename type_at_index<I - 1, Tail...>::type;
};
//
// Specialization for the base case (I = 0)
template <typename Head, typename... Tail>
struct type_at_index<0, Head, Tail...> {
    using type = Head;
};
//
template <size_t I, class... Ts>
using type_at_index_t = typename type_at_index<I, Ts...>::type;

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_TYPE_TRAITS_HPP_
