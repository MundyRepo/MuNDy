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

#ifndef MUNDY_CONSTRAINTS_CONSTRAINTS_HPP_
#define MUNDY_CONSTRAINTS_CONSTRAINTS_HPP_

/// \file Constraints.hpp
/// \brief Declaration of the Constraints class

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

namespace constraints {

/// \class Constraints
/// In the current design, a "constraint" is an element-rank Part with some set of requirements that endow the entities
/// of that part with some constraint.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is an element rank assembly part containing all Constraints. It is a subset of the Agents part.
class Constraints : public mundy::agents::RankedAssembly<mundy::core::make_string_literal("CONSTRAINTS"),
                                                         stk::topology::ELEMENT_RANK, mundy::agents::Agents> {
};  // Constraints

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_CONSTRAINTS_HPP_
