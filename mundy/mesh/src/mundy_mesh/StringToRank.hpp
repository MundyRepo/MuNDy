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

#ifndef MUNDY_MESH_STRINGTORANK_HPP_
#define MUNDY_MESH_STRINGTORANK_HPP_

/// \file StringToRank.hpp
/// \brief Helpers for mapping string names to STK entity ranks.

// C++ core libs
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

namespace mundy {

namespace mesh {

/// \brief Map a string with a valid rank name to the corresponding rank.
///
/// The set of valid rank names and their corresponding type is
///  - NODE_RANK        -> stk::topology::NODE_RANK
///  - EDGE_RANK        -> stk::topology::EDGE_RANK
///  - FACE_RANK        -> stk::topology::FACE_RANK
///  - ELEMENT_RANK     -> stk::topology::ELEMENT_RANK
///  - CONSTRAINT_RANK  -> stk::topology::CONSTRAINT_RANK
///  - INVALID_RANK     -> stk::topology::INVALID_RANK
///
/// \param rank_string [in] String containing a valid rank name.
stk::topology::rank_t string_to_rank(const std::string& rank_string);

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_STRINGTORANK_HPP_
