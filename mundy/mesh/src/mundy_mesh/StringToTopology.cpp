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

/// \file StringToTopology.cpp
/// \brief Definition of the StringToTopology class

// C++ core libs
#include <regex>   // for std::regex
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/StringToTopology.hpp>  // for mundy::mesh::string_to_rank and mundy::mesh::string_to_topology

namespace mundy {

namespace mesh {

stk::topology::rank_t string_to_rank(const std::string &rank_string) {
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

stk::topology string_to_topology(const std::string &topology_string) {
  if (topology_string == "INVALID_TOPOLOGY") {
    return stk::topology::INVALID_TOPOLOGY;
  } else if (topology_string == "NODE") {
    return stk::topology::NODE;
  } else if (topology_string == "LINE_2") {
    return stk::topology::LINE_2;
  } else if (topology_string == "LINE_3") {
    return stk::topology::LINE_3;
  } else if (topology_string == "TRI_3") {
    return stk::topology::TRI_3;
  } else if (topology_string == "TRI_4") {
    return stk::topology::TRI_4;
  } else if (topology_string == "TRI_6") {
    return stk::topology::TRI_6;
  } else if (topology_string == "QUAD_4") {
    return stk::topology::QUAD_4;
  } else if (topology_string == "QUAD_6") {
    return stk::topology::QUAD_6;
  } else if (topology_string == "QUAD_8") {
    return stk::topology::QUAD_8;
  } else if (topology_string == "QUAD_9") {
    return stk::topology::QUAD_9;
  } else if (topology_string == "PARTICLE") {
    return stk::topology::PARTICLE;
  } else if (topology_string == "LINE_2_1D") {
    return stk::topology::LINE_2_1D;
  } else if (topology_string == "LINE_3_1D") {
    return stk::topology::LINE_3_1D;
  } else if (topology_string == "BEAM_2") {
    return stk::topology::BEAM_2;
  } else if (topology_string == "BEAM_3") {
    return stk::topology::BEAM_3;
  } else if (topology_string == "SHELL_LINE_2") {
    return stk::topology::SHELL_LINE_2;
  } else if (topology_string == "SHELL_LINE_3") {
    return stk::topology::SHELL_LINE_3;
  } else if (topology_string == "SPRING_2") {
    return stk::topology::SPRING_2;
  } else if (topology_string == "SPRING_3") {
    return stk::topology::SPRING_3;
  } else if (topology_string == "TRI_3_2D") {
    return stk::topology::TRI_3_2D;
  } else if (topology_string == "TRI_4_2D") {
    return stk::topology::TRI_4_2D;
  } else if (topology_string == "TRI_6_2D") {
    return stk::topology::TRI_6_2D;
  } else if (topology_string == "QUAD_4_2D") {
    return stk::topology::QUAD_4_2D;
  } else if (topology_string == "QUAD_8_2D") {
    return stk::topology::QUAD_8_2D;
  } else if (topology_string == "QUAD_9_2D") {
    return stk::topology::QUAD_9_2D;
  } else if (topology_string == "SHELL_TRI_3") {
    return stk::topology::SHELL_TRI_3;
  } else if (topology_string == "SHELL_TRI_4") {
    return stk::topology::SHELL_TRI_4;
  } else if (topology_string == "SHELL_TRI_6") {
    return stk::topology::SHELL_TRI_6;
  } else if (topology_string == "SHELL_QUAD_4") {
    return stk::topology::SHELL_QUAD_4;
  } else if (topology_string == "SHELL_QUAD_8") {
    return stk::topology::SHELL_QUAD_8;
  } else if (topology_string == "SHELL_QUAD_9") {
    return stk::topology::SHELL_QUAD_9;
  } else if (topology_string == "TET_4") {
    return stk::topology::TET_4;
  } else if (topology_string == "TET_8") {
    return stk::topology::TET_8;
  } else if (topology_string == "TET_10") {
    return stk::topology::TET_10;
  } else if (topology_string == "TET_11") {
    return stk::topology::TET_11;
  } else if (topology_string == "PYRAMID_5") {
    return stk::topology::PYRAMID_5;
  } else if (topology_string == "PYRAMID_13") {
    return stk::topology::PYRAMID_13;
  } else if (topology_string == "PYRAMID_14") {
    return stk::topology::PYRAMID_14;
  } else if (topology_string == "WEDGE_6") {
    return stk::topology::WEDGE_6;
  } else if (topology_string == "WEDGE_12") {
    return stk::topology::WEDGE_12;
  } else if (topology_string == "WEDGE_15") {
    return stk::topology::WEDGE_15;
  } else if (topology_string == "WEDGE_18") {
    return stk::topology::WEDGE_18;
  } else if (topology_string == "HEX_8") {
    return stk::topology::HEX_8;
  } else if (topology_string == "HEX_20") {
    return stk::topology::HEX_20;
  } else if (topology_string == "HEX_27") {
    return stk::topology::HEX_27;
  } else if (std::regex_match(topology_string, std::regex("SUPEREDGE<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superedge_topology(num_nodes);
  } else if (std::regex_match(topology_string, std::regex("SUPERFACE<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superface_topology(num_nodes);
  } else if (std::regex_match(topology_string, std::regex("SUPERELEMENT<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superelement_topology(num_nodes);
  } else {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        std::string("PartReqs: The provided topology string ") + topology_string + " is not valid.");
  }

  return stk::topology::INVALID_TOPOLOGY;  // Should never be reached.
}

}  // namespace mesh

}  // namespace mundy
