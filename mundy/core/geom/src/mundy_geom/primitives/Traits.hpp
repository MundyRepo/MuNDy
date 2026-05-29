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

#ifndef MUNDY_GEOM_PRIMITIVES_TRAITS_HPP_
#define MUNDY_GEOM_PRIMITIVES_TRAITS_HPP_

// C++ core
#include <type_traits>

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

/// @brief Trait: does a primitive type have finite spatial extent?
///
/// Opt in by declaring `static constexpr bool is_finite = true/false;` on the class,
/// or by providing an explicit specialization.
///
/// Unknown types default to false.
template <typename T, typename = void>
struct is_finite : std::false_type {};

template <typename T>
struct is_finite<T, std::void_t<decltype(std::remove_cv_t<T>::is_finite)>>
    : std::bool_constant<std::remove_cv_t<T>::is_finite> {};

template <typename T>
inline constexpr bool is_finite_v = is_finite<T>::value;

/// @brief Concept: satisfied by primitives with finite spatial extent (is_finite == true).
template <typename T>
concept FinitePrimitive = is_finite_v<T>;

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_TRAITS_HPP_
