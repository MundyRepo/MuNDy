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

#ifndef MUNDY_AGENTS_RANKEDASSEMBLY_HPP_
#define MUNDY_AGENTS_RANKEDASSEMBLY_HPP_

/// \file RankedAssembly.hpp
/// \brief Declaration of the RankedAssembly agent helper class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>       // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>       // for mundy::meta::PartReqs

namespace mundy {

namespace agents {

/// \class RankedAssembly
/// \brief A helper class for defining Agents whose only purpose is to assemble other agents of a given rank.
///
/// An AssemblyAgent is a special type of Agent that is used to assemble other agents. It is associated with a ranked
/// assembly part but no topology.
///
/// The design of this class is in accordance with the static interface requirements of
/// mundy::agents::AgentFactory.
///
/// \tparam name The name of our part (as a compile-time StringLiteral).
/// \tparam rank The rank of our part.
/// \tparam ParentAgentTypes Any number of agents that are parents of this agent. Can be empty.
template <mundy::core::StringLiteral name, stk::topology::rank_t rank, typename... ParentAgentTypes>
class RankedAssembly {
 public:
  //! \name Getters
  //@{

  /// \brief Get the name of our part.
  static inline std::string get_name() {
    return name.to_string();
  }

  /// \brief Get the names of our parent parts.
  static inline std::vector<std::string> get_parent_names() {
    return {ParentAgentTypes::get_name()...};
  }

  /// \brief Get the topology of our part.
  static constexpr inline stk::topology::topology_t get_topology() {
    return stk::topology::INVALID_TOPOLOGY;
  }

  /// \brief Get the rank of our part.
  static constexpr inline stk::topology::rank_t get_rank() {
    return rank;
  }

  /// \brief Get if our part has a topology or not.
  static constexpr inline bool has_topology() {
    return false;
  }

  /// \brief Get if our part has a rank or not.
  static constexpr inline bool has_rank() {
    return true;
  }

  /// \brief Add new part requirements to ALL members of this agent part.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr) {
    part_reqs_ptr_->sync(part_reqs_ptr);
  }

  /// \brief Add sub-part requirements.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr) {
    part_reqs_ptr_->add_and_sync_subpart_reqs(subpart_reqs_ptr);
  }

  /// \brief Get our mesh requirements.
  static inline std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements() {
    MUNDY_THROW_REQUIRE(part_reqs_ptr_ != nullptr, std::logic_error, "get_mesh_requirements: part_reqs_ptr_ is null.");

    // Declare our part as a subpart of our parent parts.
    (ParentAgentTypes::add_and_sync_subpart_reqs(part_reqs_ptr_), ...);

    // Because we passed our part requirements up the chain, we can now fetch and sync all of our parent's
    // requirements. If done correctly, this call will result in a upward tree traversal. Our part is declared as a
    // subpart of our parent, which is declared as a subpart of its parent. This process repeated until we reach a root
    // node. The combined requirements for all parts touched in this traversal are then returned here.
    //
    // We add our part requirements directly to the mesh to account for the case where we are the root node.
    static auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs_ptr_);
    (mesh_reqs_ptr->sync(ParentAgentTypes::get_mesh_requirements()), ...);
    return mesh_reqs_ptr;
  }
  //@}

 private:
  //! \name Member variable definitions
  //@{

  /// \brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr_ =
      std::make_shared<mundy::meta::PartReqs>(name.to_string(), rank);
  //@}
};  // RankedAssembly

}  // namespace agents

}  // namespace mundy

#endif  // MUNDY_AGENTS_RANKEDASSEMBLY_HPP_
