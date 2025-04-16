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

#ifndef MUNDY_LINKERS_NEIGHBORLINKERS_HPP_
#define MUNDY_LINKERS_NEIGHBORLINKERS_HPP_

/// \file NeighborLinkers.hpp
/// \brief Declaration of the NeighborLinkers part class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_linkers/Linkers.hpp>  // for mundy::linkers::Linkers
#include <mundy_meta/FieldReqs.hpp>   // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>    // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>    // for mundy::meta::PartReqs

namespace mundy {

namespace linkers {

/// \class NeighborLinkers
/// \brief The static interface for all of Mundy's NeighborLinkers.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing all neighbor linkers. It is a subset of the Linkers
/// agent.
class NeighborLinkers : public mundy::agents::RankedAssembly<mundy::core::make_string_literal("NEIGHBOR_LINKERS"),
                                                             stk::topology::CONSTRAINT_RANK, mundy::linkers::Linkers> {
 public:
  static inline std::string get_linked_entities_field_name() {
    return std::string("LINKED_NEIGHBOR_ENTITIES");
  }

  static inline std::string get_linked_entity_owners_field_name() {
    return std::string("LINKED_NEIGHBOR_ENTITY_OWNERS");
  }

  /// \brief Get our mesh requirements.
  static inline std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements() {
    add_and_sync_part_reqs(additional_part_reqs_ptr_);
    return mundy::agents::RankedAssembly<mundy::core::make_string_literal("NEIGHBOR_LINKERS"),
                                         stk::topology::CONSTRAINT_RANK,
                                         mundy::linkers::Linkers>::get_mesh_requirements();
  }
  //@}

 private:
  //! \name Member variable definitions
  //@{

  /// \brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartReqs> additional_part_reqs_ptr_ = []() {
    auto part_reqs_ptr = std::make_shared<mundy::meta::PartReqs>("NEIGHBOR_LINKERS", stk::topology::CONSTRAINT_RANK);
    part_reqs_ptr
        ->add_field_reqs<LinkedEntitiesFieldType::value_type>("LINKED_NEIGHBOR_ENTITIES",
                                                              stk::topology::CONSTRAINT_RANK, 2, 1)
        .add_field_reqs<int>("LINKED_NEIGHBOR_ENTITY_OWNERS", stk::topology::CONSTRAINT_RANK, 2, 1);
    return part_reqs_ptr;
  }();
  //@}
};  // NeighborLinkers

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_NEIGHBORLINKERS_HPP_
