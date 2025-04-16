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

#ifndef MUNDY_MATH_IMPL_ACCESSORIMPL_HPP_
#define MUNDY_MATH_IMPL_ACCESSORIMPL_HPP_

// C++ core libs
#include <concepts>

namespace mundy {

namespace math {

namespace impl {

// Helper for generating a tuple with T repeated N times
template <typename T, size_t... Is>
auto generate_tuple_with_t_repeated_impl(std::index_sequence<Is...>) {
  return std::tuple<std::conditional_t<true, T, decltype(Is)>...>{};
}

// Main template to generate a tuple with T repeated N times
template <typename T, size_t N>
auto generate_tuple_with_t_repeated_n_times() {
  return generate_tuple_with_t_repeated_impl<T>(std::make_index_sequence<N>{});
}

// Helper function to check if Accessor is constructible from unpacked tuple
template <typename Accessor, typename Tuple, size_t... Is>
constexpr bool can_construct_from_unpacked_tuple_impl(std::index_sequence<Is...>) {
  return std::is_constructible_v<Accessor, std::tuple_element_t<Is, Tuple>...>;
}

template <typename Accessor, typename Tuple>
constexpr bool can_construct_from_unpacked_tuple() {
  constexpr auto size = std::tuple_size_v<std::remove_reference_t<Tuple>>;
  return can_construct_from_unpacked_tuple_impl<Accessor, Tuple>(std::make_index_sequence<size>{});
}

}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_ACCESSORIMPL_HPP_
