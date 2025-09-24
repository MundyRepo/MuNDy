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

#ifndef MUNDY_MESH_STRINGTOTOPOLOGY_HPP_
#define MUNDY_MESH_STRINGTOTOPOLOGY_HPP_

/// \file StringToTopology.hpp
/// \brief Declaration of the StringToTopology class

// C++ core libs
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

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
stk::topology::rank_t string_to_rank(const std::string &rank_string);

/// \brief Map a string with a valid topology name to the corresponding topology.
///
/// The set of valid topology names and their corresponding return type is
///  - No rank topologies
///     - INVALID_TOPOLOGY                       -> stk::topology::INVALID_TOPOLOGY
///  - Node rank topologies
///     - NODE                                   -> stk::topology::NODE
///  - Edge rank topologies
///     - LINE_2                                 -> stk::topology::LINE_2
///     - LINE_3                                 -> stk::topology::LINE_3
///  - Face rank topologies
///     - TRI_3 or TRIANGLE_3                    -> stk::topology::TRI_3
///     - TRI_4 or TRIANGLE_4                    -> stk::topology::TRI_4
///     - TRI_6 or TRIANGLE_6                    -> stk::topology::TRI_6
///     - QUAD_4 or QUADRILATERAL_4              -> stk::topology::QUAD_4
///     - QUAD_6 or QUADRILATERAL_6              -> stk::topology::QUAD_6
///     - QUAD_8 or QUADRILATERAL_8              -> stk::topology::QUAD_8
///     - QUAD_9 or QUADRILATERAL_9              -> stk::topology::QUAD_9
///   - Element rank topologies
///     - PARTICLE                               -> stk::topology::PARTICLE
///     - LINE_2_1D                              -> stk::topology::LINE_2_1D
///     - LINE_3_1D                              -> stk::topology::LINE_3_1D
///     - BEAM_2                                 -> stk::topology::BEAM_2
///     - BEAM_3                                 -> stk::topology::BEAM_3
///     - SHELL_LINE_2                           -> stk::topology::SHELL_LINE_2
///     - SHELL_LINE_3                           -> stk::topology::SHELL_LINE_3
///     - SPRING_2                               -> stk::topology::SPRING_2
///     - SPRING_3                               -> stk::topology::SPRING_3
///     - TRI_3_2D or TRIANGLE_3_2D              -> stk::topology::TRI_3_2D
///     - TRI_4_2D or TRIANGLE_4_2D              -> stk::topology::TRI_4_2D
///     - TRI_6_2D or TRIANGLE_6_2D              -> stk::topology::TRI_6_2D
///     - QUAD_4_2D or QUADRILATERAL_4_2D        -> stk::topology::QUAD_4_2D
///     - QUAD_8_2D or QUADRILATERAL_8_2D        -> stk::topology::QUAD_8_2D
///     - QUAD_9_2D or QUADRILATERAL_9_2D        -> stk::topology::QUAD_9_2D
///     - SHELL_TRI_3 or SHELL_TRIANGLE_3        -> stk::topology::SHELL_TRI_3
///     - SHELL_TRI_4 or SHELL_TRIANGLE_4        -> stk::topology::SHELL_TRI_4
///     - SHELL_TRI_6 or SHELL_TRIANGLE_6        -> stk::topology::SHELL_TRI_6
///     - SHELL_QUAD_4 or SHELL_QUADRILATERAL_4  -> stk::topology::SHELL_QUAD_4
///     - SHELL_QUAD_8 or SHELL_QUADRILATERAL_8  -> stk::topology::SHELL_QUAD_8
///     - SHELL_QUAD_9 or SHELL_QUADRILATERAL_9  -> stk::topology::SHELL_QUAD_9
///     - TET_4 or TETRAHEDRON_4                 -> stk::topology::TET_4
///     - TET_8 or TETRAHEDRON_8                 -> stk::topology::TET_8
///     - TET_10 or TETRAHEDRON_10               -> stk::topology::TET_10
///     - TET_11 or TETRAHEDRON_11               -> stk::topology::TET_11
///     - PYRAMID_5                              -> stk::topology::PYRAMID_5
///     - PYRAMID_13                             -> stk::topology::PYRAMID_13
///     - PYRAMID_14                             -> stk::topology::PYRAMID_14
///     - WEDGE_6                                -> stk::topology::WEDGE_6
///     - WEDGE_12                               -> stk::topology::WEDGE_12
///     - WEDGE_15                               -> stk::topology::WEDGE_15
///     - WEDGE_18                               -> stk::topology::WEDGE_18
///     - HEX_8 or HEXAHEDRON_8                  -> stk::topology::HEX_8
///     - HEX_20 or HEXAHEDRON_20                -> stk::topology::HEX_20
///     - HEX_27 or HEXAHEDRON_27                -> stk::topology::HEX_27
///   - Super topologies
///     - SUPEREDGE<N>                           -> create_superedge_topology(N)
///     - SUPERFACE<N>                           -> create_superface_topology(N)
///     - SUPERELEMENT<N>                        -> create_superelement_topology(N)
/// \param rank_string [in] String containing a valid rank name.
stk::topology string_to_topology(const std::string &topology_string);

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_STRINGTOTOPOLOGY_HPP_
