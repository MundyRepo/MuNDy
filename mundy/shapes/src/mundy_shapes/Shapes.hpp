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

#ifndef MUNDY_SHAPES_SHAPES_HPP_
#define MUNDY_SHAPES_SHAPES_HPP_

/// \file Shapes.hpp
/// \brief Declaration of the Shapes class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agents/Agents.hpp>          // for mundy::agents::Agents
#include <mundy_agents/RankedAssembly.hpp>  // for mundy::agents::RankedAssembly
#include <mundy_core/StringLiteral.hpp>     // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_meta/FieldReqs.hpp>         // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>          // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>          // for mundy::meta::PartReqs

namespace mundy {

namespace shapes {

/// \class Shapes
/// In the current design, a "shape" is an element-rank Part with some set of requirements that endow the entities of
/// that part with some shape. For example,
///   - a point particle can be represented as having a PARTICLE topology with one node at its center (it need not be
///   orientable)
///   - a line particle can be represented as having a LINE_3 topology with three nodes (one at each end and one at its
///   center).
///   - a sphere is a point particle with an element radius.
///   - an ellipsoid is a point particle with three element axis lengths and an element orientation.
///   - a spherocylinder is a point particle with an element radius, length, and orientation.
///   - a spherocylidner_segment is a line segment with element radius and length.
///   - a NURBS is a SUPERTOPOLOGY<N> with N nodes corresponding to the control points.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is an element rank assembly part containing all shapes. It is a subset of the Agents part.
class Shapes : public mundy::agents::RankedAssembly<mundy::core::make_string_literal("SHAPES"),
                                                    stk::topology::ELEMENT_RANK, mundy::agents::Agents> {};  // Shapes

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_SHAPES_HPP_
