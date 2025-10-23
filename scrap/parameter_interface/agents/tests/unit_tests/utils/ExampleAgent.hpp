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

#ifndef UNIT_TESTS_MUNDY_AGENTS_UTILS_EXAMPLEAGENT_HPP_
#define UNIT_TESTS_MUNDY_AGENTS_UTILS_EXAMPLEAGENT_HPP_

/// \file ExampleAgent.hpp
/// \brief Declaration of the ExampleAgent class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agents/Agents.hpp>  // for mundy::agents::Agents
#include <mundy_meta/MeshReqs.hpp>  // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs

namespace mundy {

namespace agents {

namespace utils {

/// \class ExampleAgent
/// \brief The static interface for all of Mundy's ExampleAgent objects.
template <int I>
class ExampleAgent {
 public:
  //! \name Getters
  //@{

  /// \brief Get the name of our part.
  static inline std::string get_name() {
    get_name_counter_++;
    return std::string(name_);
  }

  /// \brief Get the names of our parent parts.
  static inline std::vector<std::string> get_parent_names() {
    get_parent_names_counter_++;
    return {mundy::agents::Agents::get_name()};
  }

  /// \brief Get the topology of our part.
  static constexpr inline stk::topology::topology_t get_topology() {
    get_topology_counter_++;
    return topology_;
  }

  /// \brief Get the rank of our part.
  static constexpr inline stk::topology::rank_t get_rank() {
    get_rank_counter_++;
    return rank_;
  }

  /// \brief Get if our part has a topology or not.
  static constexpr inline bool has_topology() {
    has_topology_counter_++;
    return has_topology_;
  }

  /// \brief Get if our part has a rank or not.
  static constexpr inline bool has_rank() {
    has_rank_counter_++;
    return has_rank_;
  }

  /// \brief Add new part requirements to ALL members of this agent part.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr) {
    add_part_reqs_counter_++;
    part_reqs_ptr_->sync(part_reqs_ptr);
  }

  /// \brief Add sub-part requirements.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr) {
    add_subpart_reqs_counter_++;
    part_reqs_ptr_->add_and_sync_subpart_reqs(subpart_reqs_ptr);
  }

  /// \brief Get our mesh requirements.
  static inline std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements() {
    get_mesh_requirements_counter_++;
    // By default, we assume that the ExampleAgents part is an agent with an INVALID_TOPOLOGY.

    // Declare our part as a subpart of our parent parts.
    mundy::agents::Agents::add_and_sync_subpart_reqs(part_reqs_ptr_);

    // Because we passed our part requirements up the chain, we can now fetch and sync all of our parent's
    // requirements. If done correctly, this call will result in a upward tree traversal. Our part is declared as a
    // subpart of our parent, which is declared as a subpart of its parent. This process repeated until we reach a root
    // node. The combined requirements for all parts touched in this traversal are then returned here.
    //
    // We add our part requirements directly to the mesh to account for the case where we are the root node.
    static auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs_ptr_);
    mesh_reqs_ptr->sync(mundy::agents::Agents::get_mesh_requirements());
    return mesh_reqs_ptr;
  }

  /// \brief Get the number of times get_name() has been called.
  static inline int get_get_name_counter() {
    return get_name_counter_;
  }

  /// \brief Get the number of times get_parent_names() has been called.
  static inline int get_get_parent_names_counter() {
    return get_parent_names_counter_;
  }

  /// \brief Get the number of times get_topology() has been called.
  static inline int get_get_topology_counter() {
    return get_topology_counter_;
  }

  /// \brief Get the number of times get_rank() has been called.
  static inline int get_get_rank_counter() {
    return get_rank_counter_;
  }

  /// \brief Get the number of times has_topology() has been called.
  static inline int get_has_topology_counter() {
    return has_topology_counter_;
  }

  /// \brief Get the number of times has_rank() has been called.
  static inline int get_has_rank_counter() {
    return has_rank_counter_;
  }

  /// \brief Get the number of times add_and_sync_part_reqs() has been called.
  static inline int get_add_part_reqs_counter() {
    return add_part_reqs_counter_;
  }

  /// \brief Get the number of times add_and_sync_subpart_reqs() has been called.
  static inline int get_add_subpart_reqs_counter() {
    return add_subpart_reqs_counter_;
  }

  /// \brief Get the number of times get_mesh_requirements() has been called.
  static inline int get_get_mesh_requirements_counter() {
    return get_mesh_requirements_counter_;
  }
  //@}

 private:
  //! \name Debug member variables
  //@{

  /// \brief The number of times get_name() has been called.
  static inline int get_name_counter_ = 0;

  /// \brief The number of times get_parent_names() has been called.
  static inline int get_parent_names_counter_ = 0;

  /// \brief The number of times get_topology() has been called.
  static inline int get_topology_counter_ = 0;

  /// \brief The number of times get_rank() has been called.
  static inline int get_rank_counter_ = 0;

  /// \brief The number of times has_topology() has been called.
  static inline int has_topology_counter_ = 0;

  /// \brief The number of times has_rank() has been called.
  static inline int has_rank_counter_ = 0;

  /// \brief The number of times add_and_sync_part_reqs() has been called.
  static inline int add_part_reqs_counter_ = 0;

  /// \brief The number of times add_and_sync_subpart_reqs() has been called.
  static inline int add_subpart_reqs_counter_ = 0;

  /// \brief The number of times get_mesh_requirements() has been called.
  static inline int get_mesh_requirements_counter_ = 0;
  //@}

  //! \name Standard member variables
  //@{

  /// \brief The name of the our part.
  static constexpr std::string_view name_ = []() constexpr -> std::string_view {
    if constexpr (I == 0) {
      return "FAKE_NAME_0";
    } else if constexpr (I == 1) {
      return "FAKE_NAME_1";
    } else if constexpr (I == 2) {
      return "FAKE_NAME_2";
    } else if constexpr (I == 3) {
      return "FAKE_NAME_3";
    } else {
      static_assert(I < 4, "Invalid ExampleAgent index.");
      return "FAKE_NAME_N";
    }
  }();

  /// \brief Our topology
  static constexpr stk::topology::topology_t topology_ = stk::topology::INVALID_TOPOLOGY;

  /// \brief Our rank (we have a rank, so this is never used).
  static constexpr inline stk::topology::rank_t rank_ = stk::topology::INVALID_RANK;

  /// \brief If our part has a topology or not.
  static constexpr inline bool has_topology_ = true;

  /// \brief If our part has a rank or not.
  static constexpr inline bool has_rank_ = false;

  /// \brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr_ = []() {
    auto part_reqs_ptr = std::make_shared<mundy::meta::PartReqs>();
    part_reqs_ptr->set_part_name(std::string(name_));
    part_reqs_ptr->set_part_topology(topology_);
    return part_reqs_ptr;
  }();
  //@}
};  // ExampleAgent

}  // namespace utils

}  // namespace agents

}  // namespace mundy

#endif  // UNIT_TESTS_MUNDY_AGENTS_UTILS_EXAMPLEAGENT_HPP_
