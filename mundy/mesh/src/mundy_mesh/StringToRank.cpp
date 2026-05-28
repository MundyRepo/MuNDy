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

/// \file StringToRank.cpp
/// \brief Definition of the string-to-rank helper.

// C++ core libs
#include <stdexcept>  // for std::invalid_argument
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/StringToRank.hpp>   // for mundy::mesh::string_to_rank
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace mesh {

stk::topology::rank_t string_to_rank(const std::string& rank_string) {
  if (rank_string == "NODE_RANK") {
    return stk::topology::NODE_RANK;
  } else if (rank_string == "EDGE_RANK") {
    return stk::topology::EDGE_RANK;
  } else if (rank_string == "FACE_RANK") {
    return stk::topology::FACE_RANK;
  } else if (rank_string == "ELEMENT_RANK") {
    return stk::topology::ELEMENT_RANK;
  } else if (rank_string == "CONSTRAINT_RANK") {
    return stk::topology::CONSTRAINT_RANK;
  } else if (rank_string == "INVALID_RANK") {
    return stk::topology::INVALID_RANK;
  } else {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        std::string("The provided rank string ") + rank_string + " is not valid.");
  }

  return stk::topology::INVALID_RANK;  // Should never be reached.
}

}  // namespace mesh

}  // namespace mundy
