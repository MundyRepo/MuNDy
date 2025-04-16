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

#ifndef MUNDY_CONSTRAINTS_FENESPRINGS_HPP_
#define MUNDY_CONSTRAINTS_FENESPRINGS_HPP_

/// \file FENESprings.hpp
/// \brief Declaration of the FENESprings part class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_constraints/Constraints.hpp>  // for mundy::constraints::Constraints
#include <mundy_meta/FieldReqs.hpp>           // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>            // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>            // for mundy::meta::PartReqs

namespace mundy {

namespace constraints {

/// \class FENESprings
/// \brief The static interface for all of Mundy's FENESprings constraint.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
class FENESprings {
 public:
  //! \name Getters
  //@{

  /// \brief Get the name of our part.
  static inline std::string get_name() {
    return std::string(name_);
  }

  /// \brief Get the names of our parent parts.
  static inline std::vector<std::string> get_parent_names() {
    return {mundy::constraints::Constraints::get_name()};
  }

  /// \brief Get the topology of our part.
  static constexpr inline stk::topology::topology_t get_topology() {
    return topology_;
  }

  /// \brief Get the rank of our part.
  static constexpr inline stk::topology::rank_t get_rank() {
    return rank_;
  }

  /// \brief Get if our part has a topology or not.
  static constexpr inline bool has_topology() {
    return has_topology_;
  }

  /// \brief Get if our part has a rank or not.
  static constexpr inline bool has_rank() {
    return has_rank_;
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
    // By default, we assume that the FENESprings part is a beam 2 particle (an element that forms a line between two
    // nodes) with a spring constant and a maximum length. All FENESprings are Constraints.

    // Declare our part as a subpart of our parent parts.
    mundy::constraints::Constraints::add_and_sync_subpart_reqs(part_reqs_ptr_);

    // Because we passed our part requirements up the chain, we can now fetch and sync all of our parent's
    // requirements. If done correctly, this call will result in a upward tree traversal. Our part is declared as a
    // subpart of our parent, which is declared as a subpart of its parent. This process repeated until we reach a root
    // node. The combined requirements for all parts touched in this traversal are then returned here.
    //
    // We add our part requirements directly to the mesh to account for the case where we are the root node.
    static auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs_ptr_);
    mesh_reqs_ptr->sync(mundy::constraints::Constraints::get_mesh_requirements());
    return mesh_reqs_ptr;
  }

  /// \brief Get the default node coordinate field name for the FENESprings part.
  static inline std::string get_node_coord_field_name() {
    return std::string(node_coord_field_name_);
  }

  /// \brief Get the default element FENE spring constant field name for the FENESprings part.
  static inline std::string get_element_spring_constant_field_name() {
    return std::string(element_fene_spring_constant_field_name_);
  }

  /// \brief Get the default element FENE spring maximum extension field name for the FENESprings part.
  static inline std::string get_element_rmax_field_name() {
    return std::string(element_fene_spring_rmax_field_name_);
  }

 private:
  //! \name Member variable definitions
  //@{

  /// \brief The name of the our part.
  static constexpr std::string_view name_ = "FENE_SPRINGS";

  /// \brief Our topology
  static constexpr stk::topology::topology_t topology_ = stk::topology::BEAM_2;

  /// \brief Our rank (we have a rank, so this is never used).
  static constexpr inline stk::topology::rank_t rank_ = stk::topology::INVALID_RANK;

  /// \brief If our part has a topology or not.
  static constexpr inline bool has_topology_ = true;

  /// \brief If our part has a rank or not.
  static constexpr inline bool has_rank_ = false;

  /// \brief The name of our node coordinate field.
  static constexpr std::string_view node_coord_field_name_ = "NODE_COORDS";

  /// \brief The name of our element FENE spring constant field.
  static constexpr std::string_view element_fene_spring_constant_field_name_ = "ELEMENT_SPRING_CONSTANT";

  /// \brief The name of our element FENE spring maximum extension field.
  static constexpr std::string_view element_fene_spring_rmax_field_name_ = "ELEMENT_SPRING_R0";

  /// \brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr_ = []() {
    auto part_reqs_ptr = std::make_shared<mundy::meta::PartReqs>();
    part_reqs_ptr->set_part_name(std::string(name_));
    part_reqs_ptr->set_part_topology(topology_);
    part_reqs_ptr->add_field_reqs<double>(std::string(node_coord_field_name_), stk::topology::NODE_RANK, 3, 1);
    part_reqs_ptr->add_field_reqs<double>(std::string(element_fene_spring_constant_field_name_),
                                          stk::topology::ELEMENT_RANK, 1, 1);
    part_reqs_ptr->add_field_reqs<double>(std::string(element_fene_spring_rmax_field_name_),
                                          stk::topology::ELEMENT_RANK, 1, 1);
    return part_reqs_ptr;
  }();
  //@}
};  // FENESprings

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_FENESPRINGS_HPP_
