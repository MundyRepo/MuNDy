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

#ifndef MUNDY_UTILS_REQUIRES_HPP_
#define MUNDY_UTILS_REQUIRES_HPP_

/// \brief A tiny helper macro that allows us to (optionally) hide the `requires` keyword from Doxygen/ReadTheDocs.
#ifndef MUNDY_REQUIRES
#define MUNDY_REQUIRES(...) requires(__VA_ARGS__)
#endif

#endif  // MUNDY_UTILS_REQUIRES_HPP_
