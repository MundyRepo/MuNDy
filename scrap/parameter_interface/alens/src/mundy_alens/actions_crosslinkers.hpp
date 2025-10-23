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

#ifndef MUNDY_ALENS_ACTIONS_CROSSLINKERS_HPP_
#define MUNDY_ALENS_ACTIONS_CROSSLINKERS_HPP_

// Trilinos
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>    // for mundy::mesh::vector3_field_data
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData

namespace mundy {

namespace alens {

namespace crosslinkers {

/// \brief An enum specying the current state of a crosslinker.
/// The main purpose of this enum is to reduce redundant state-checking code.
enum CROSSLINKER_STATE : unsigned { UNBOUND = 0u, LEFT_BOUND, RIGHT_BOUND, DOUBLY_BOUND, NUM_STATES };

/// \brief An enum specifying the possible (single operation) state changes for a crosslinker.
/// We explicitly forbid state changes suc as unbound to doubly bound, as it would require two operations.
enum BINDING_STATE_CHANGE : unsigned {
  NONE = 0u,
  UNBOUND_TO_LEFTBOUND,
  UNBOUND_TO_RIGHTBOUND,
  LEFTBOUND_TO_UNBOUND,
  LEFTBOUND_TO_DOUBLYBOUND,
  RIGHTBOUND_TO_UNBOUND,
  RIGHTBOUND_TO_DOUBLYBOUND,
  DOUBLYBOUND_TO_LEFTBOUND,
  DOUBLYBOUND_TO_RIGHTBOUND,
  NUM_ADMISSIBLE_STATE_CHANGES
};

/// \brief Unbind a crosslinker from a node.
///
/// If the crosslinker isn't connected to the node on the given ordinal. This is a no-op if the crosslinker isn't
/// connected to the node on the given ordinal. We'll return a bool indicating whether the operation was successful.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the node will be unbound.
inline bool unbind_crosslinker_from_node(mundy::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker,
                                         const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "unbind_crosslinker_from_node: The mesh must be in a modification cycle.");
  // Maybe unsafe?
  // MUNDY_THROW_ASSERT(bulk_data.bucket(crosslinker).topology().base() == stk::topology::BEAM_2,
  // std::logic_error, "bind_crosslinker_to_node: The crosslinker must have BEAM_2 as a base topology.");

  // If a node already exists at the ordinal, we'll destroy that relation.
  const int num_nodes = bulk_data.num_nodes(crosslinker);
  stk::mesh::Entity const *nodes = bulk_data.begin_nodes(crosslinker);
  stk::mesh::ConnectivityOrdinal const *node_ords = bulk_data.begin_node_ordinals(crosslinker);
  for (int i = 0; i < num_nodes; ++i) {
    if (node_ords[i] == conn_ordinal) {
      // We found the node in the ordinal that we're trying to bind to. We'll attempt to destroy this relation.
      // This doesn't mean that it was sucessfully destroyed. That's up to the bulk data object and will be returned by
      // destroy_relation.
      return bulk_data.destroy_relation(crosslinker, nodes[i], conn_ordinal);
    }
  }

  // If we didn't find the node, this is a no-op.
  return false;
}

/// \brief Connect a crosslinker to a new node.
///
/// If the crosslinker is already connected to to a node on the given ordinal, the operation will fail.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param new_node The new node entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the new node will be bound.
inline bool bind_crosslinker_to_node(mundy::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker,
                                     const stk::mesh::Entity &new_node, const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "bind_crosslinker_to_node: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(new_node) == stk::topology::NODE_RANK, std::logic_error,
                     "bind_crosslinker_to_node: The node must have NODE_RANK.");
  // Maybe unsafe?
  // MUNDY_THROW_ASSERT(bulk_data.bucket(crosslinker).topology().base() == stk::topology::BEAM_2,
  // std::logic_error, "bind_crosslinker_to_node: The crosslinker must have BEAM_2 as a base topology.");

  // Check a node already exists at the ordinal
  const int num_nodes = bulk_data.num_nodes(crosslinker);
  stk::mesh::ConnectivityOrdinal const *node_ords = bulk_data.begin_node_ordinals(crosslinker);
  for (int i = 0; i < num_nodes; ++i) {
    if (node_ords[i] == conn_ordinal) {
      // We found the node in the ordinal that we're trying to bind to. Fail the operation.
      return false;
    }
  }

  // Declare the new relation.
  bulk_data.declare_relation(crosslinker, new_node, conn_ordinal);

  return true;
}

/// \brief Connect a crosslinker to a new node and unbind the existing node.
///
/// If the crosslinker is already connected to to a node on the given ordinal, we'll destroy that relation and replace
/// it with the new one. We'll return a bool indicating whether the operation was successful.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param new_node The new node entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the new node will be bound.
inline bool bind_crosslinker_to_node_unbind_existing(mundy::mesh::BulkData &bulk_data,
                                                     const stk::mesh::Entity &crosslinker,
                                                     const stk::mesh::Entity &new_node, const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "bind_crosslinker_to_node: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(new_node) == stk::topology::NODE_RANK, std::logic_error,
                     "bind_crosslinker_to_node: The node must have NODE_RANK.");
  // Maybe unsafe?
  // MUNDY_THROW_ASSERT(bulk_data.bucket(crosslinker).topology().base() == stk::topology::BEAM_2,
  // std::logic_error, "bind_crosslinker_to_node: The crosslinker must have BEAM_2 as a base topology.");

  // If a node already exists at the ordinal, we'll destroy that relation.
  unbind_crosslinker_from_node(bulk_data, crosslinker, conn_ordinal);

  // Declare the new relation.
  bulk_data.declare_relation(crosslinker, new_node, conn_ordinal);

  return true;
}

}  // namespace crosslinkers

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_ACTIONS_CROSSLINKERS_HPP_
