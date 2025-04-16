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

#ifndef MUNDY_MESH_STRINGTOSELECTOR_HPP_
#define MUNDY_MESH_STRINGTOSELECTOR_HPP_

/// \file StringToSelector.hpp
/// \brief Declaration of the string to selector helper function

// C++ core
#include <string>  // for std::string

// Trilinos
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData

namespace mundy {

namespace mesh {

/// \brief Map a string with a valid set of selector math to the corresponding selector.
///
/// Selectors are allowed to be combined using the following operators:
///  - Subtraction:    -
///  - Arythmetic and: &
///  - Arythmetic or:  |
///  - Unary not:      !
///  - Parentheses:   ( )
///
/// For example, to select all elements of parts A and B that are not in part C, the selector string would be:
///   "(partA | partB) & !partC"
/// To instead, get all elements in both parts A and B that are not in part C, the selector string would be:
///   "(partA & partB) & !partC"
///
/// Spaces are allowed in the selector string, but are not required. Names may contain any combination of letters,
/// numbers, underscores, and periods, so long as they start with a letter.
///
/// The names of the parts are fetched from the BulkData object. If a part name is not found, an exception is thrown.
/// We offer the following special part names:
///  - "UNIVERSAL"        -> The universal part, which contains all entities.
///  - "LOCALLY_OWNED"    -> The locally owned part, which contains all entities owned by the current process.
///  - "GLOBALLY_SHARED"  -> The globally shared part, which contains all entities shared from another processes.
///  - "AURA"             -> The automatically generated auto part, which contains all entities ghosted from another
///  process.
///
/// \param bulk_data [in] BulkData object used to fetch the part names.
/// \param selector_string [in] String containing valid selector math.
stk::mesh::Selector string_to_selector(const BulkData &bulk_data, const std::string &selector_string);

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_STRINGTOSELECTOR_HPP_
