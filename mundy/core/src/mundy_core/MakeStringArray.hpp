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

#ifndef MUNDY_CORE_MAKESTRINGARRAY_HPP_
#define MUNDY_CORE_MAKESTRINGARRAY_HPP_

/// \file MakeStringArray.hpp
/// \brief Declaration of the make_string_array helper function.

// Mundy
#include <MundyCore_config.hpp>  // for HAVE_MUNDYCORE_*

#ifdef HAVE_MUNDYCORE_TEUCHOS
// C++ core
#include <concepts>  // for std::convertible_to
#include <string>    // for std::string

// Trilinos
#include <Teuchos_Array.hpp>  // for Teuchos::Array and Teuchos::tuple

namespace mundy {

namespace core {

template <typename T>
concept StringConvertible = requires(T a) {
  { std::string(a) } -> std::convertible_to<std::string>;
};

/// \brief Helper function to create a Teuchos::Array of strings from any number of raw strings or string convertible
/// types.
///
/// \param args The raw strings or string convertible types to be converted to strings.
template <StringConvertible... Args>
Teuchos::Array<std::string> make_string_array(Args&&... args) {
  return Teuchos::tuple<std::string>(std::string(std::forward<Args>(args))...);
}

}  // namespace core

}  // namespace mundy

#endif

#endif  // MUNDY_CORE_MAKESTRINGARRAY_HPP_
