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

/// \file DeclareEntities.cpp

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <iostream>       // for std::ostream
#include <memory>         // for std::shared_ptr
#include <stdexcept>      // for std::runtime_error
#include <tuple>          // for std::tuple, std::make_tuple
#include <typeindex>      // for std::type_index
#include <unordered_map>  // for std::unordered_map
#include <unordered_set>  // for std::unordered_set
#include <utility>        // for std::pair, std::make_pair
#include <vector>         // for std::vector

// Trilinos
#include <stk_mesh/base/BulkData.hpp>    // for stk::mesh::BulkData
#include <stk_mesh/base/FEMHelpers.hpp>  // for stk::mesh::declare_element

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_REQUIRE
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/fmt_stk_types.hpp>    // adds fmt::format for stk types

namespace mundy {

namespace mesh {

void DeclareEntitiesHelper::check_consistency(const stk::mesh::BulkData& bulk_data) const {
  // Get the set of unique node and element ids to be added
  // In doing so, if we find a duplicate, we can return early
  std::unordered_set<stk::mesh::EntityId> unique_node_ids;
  std::unordered_set<stk::mesh::EntityId> unique_element_ids;
  for (const auto& node_info : node_info_vec_) {
    if (!unique_node_ids.insert(node_info.id).second) {
      MUNDY_THROW_REQUIRE(false, std::runtime_error, fmt::format("Duplicate node id found: {}", node_info.id));
    }
  }
  for (const auto& element_info : element_info_vec_) {
    if (!unique_element_ids.insert(element_info.id).second) {
      MUNDY_THROW_REQUIRE(false, std::runtime_error, fmt::format("Duplicate element id found: {}", element_info.id));
    }
  }

  // Check for elements connected to non-existent nodes that aren't marked as Invalid
  for (const auto& element_info : element_info_vec_) {
    for (const stk::mesh::EntityId node_id : element_info.node_ids) {
      const bool is_node_marked_as_invalid = node_id == stk::mesh::InvalidEntityId;
      if (!is_node_marked_as_invalid) {
        const bool is_node_in_unique_node_ids = unique_node_ids.find(node_id) != unique_node_ids.end();
        if (!is_node_in_unique_node_ids) {
          const bool node_in_bulk_data = bulk_data.is_valid(bulk_data.get_entity(stk::topology::NODE_RANK, node_id));
          MUNDY_THROW_REQUIRE(node_in_bulk_data, std::runtime_error,
                              fmt::format("Element {} connects to non-existent node {}", element_info.id, node_id));
        }
      }
    }
  }

  // Check that the given set of parts will create an entity of the given topology
  for (const auto& element_info : element_info_vec_) {
    const stk::topology given_topo = element_info.topology;
    const stk::topology actual_topo =
        get_topology(bulk_data.mesh_meta_data(), stk::topology::ELEMENT_RANK, element_info.parts);
    MUNDY_THROW_REQUIRE(given_topo == actual_topo, std::runtime_error,
                        fmt::format("Element {} has parts that do not match its topology\n"
                                    "Given Topology: {}\n"
                                    "Actual Topology: {}\n"
                                    "Dumping the element info:\n{}",
                                    element_info.id, given_topo.name(), actual_topo.name(), element_info));
  }

  // Check that the number of nodes that an element connects to matches its topology
  for (const auto& element_info : element_info_vec_) {
    const stk::topology topo = element_info.topology;
    const size_t num_nodes = element_info.node_ids.size();
    const size_t num_nodes_expected = topo.num_nodes();
    MUNDY_THROW_REQUIRE(num_nodes == num_nodes_expected, std::runtime_error,
                        fmt::format("Element {} has {} nodes but its topology {} expects {} nodes", element_info.id,
                                    num_nodes, topo.name(), num_nodes_expected));
  }

  // Check that the field rank matches the rank of the entity.
  for (const auto& node_info : node_info_vec_) {
    for (const auto& field_data : node_info.field_data) {
      const stk::mesh::FieldBase* field = field_data->field();
      const stk::mesh::EntityRank field_rank = field->entity_rank();
      MUNDY_THROW_REQUIRE(
          field_rank == stk::topology::NODE_RANK, std::runtime_error,
          fmt::format("Field {} is not a node-rank field and yet is set on node {}", field->name(), node_info.id));
    }
  }

  for (const auto& element_info : element_info_vec_) {
    for (const auto& field_data : element_info.field_data) {
      const stk::mesh::FieldBase* field = field_data->field();
      const stk::mesh::EntityRank field_rank = field->entity_rank();
      MUNDY_THROW_REQUIRE(field_rank == stk::topology::ELEMENT_RANK, std::runtime_error,
                          fmt::format("Field {} is not an element-rank field and yet is set on element {}",
                                      field->name(), element_info.id));
    }
  }
}

DeclareEntitiesHelper& DeclareEntitiesHelper::declare_entities(stk::mesh::BulkData& bulk_data) {
#ifndef NDEBUG
  check_consistency(bulk_data);
#endif

  const int our_rank = bulk_data.parallel_rank();

  // Construct maps for quick lookup of node and element info
  std::unordered_map<stk::mesh::EntityId, DeclareNodeInfo> node_info_map;
  std::unordered_map<stk::mesh::EntityId, DeclareElementInfo> element_info_map;
  for (const auto& node_info : node_info_vec_) {
    MUNDY_THROW_REQUIRE(node_info_map.find(node_info.id) == node_info_map.end(), std::runtime_error,
                        fmt::format("Duplicate node id found: {}", node_info.id));
    node_info_map[node_info.id] = node_info;
  }

  for (const auto& element_info : element_info_vec_) {
    MUNDY_THROW_REQUIRE(element_info_map.find(element_info.id) == element_info_map.end(), std::runtime_error,
                        fmt::format("Duplicate element id found: {}", element_info.id));
    element_info_map[element_info.id] = element_info;
  }

  // Declare all elements. Declare their nodes as we go. Mark sharing as necessary.
  for (const auto& element_info : element_info_vec_) {
    if (element_info.owning_proc == our_rank) {
      stk::mesh::Entity element = bulk_data.declare_element(element_info.id, element_info.parts);
      if (element_info.node_ids.size() != 0) {
        stk::mesh::Permutation perm = stk::mesh::Permutation::INVALID_PERMUTATION;
        stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
        for (size_t i = 0; i < element_info.node_ids.size(); ++i) {
          // Skip intentionally invalid nodes
          stk::mesh::EntityId node_id = element_info.node_ids[i];
          if (node_id == stk::mesh::InvalidEntityId) {
            continue;
          }

          stk::mesh::Entity node = bulk_data.get_entity(stk::topology::NODE_RANK, node_id);
          if (!bulk_data.is_valid(node)) {
            const auto& node_info = node_info_map[node_id];
            if (node_info.parts.size() != 0) {
              node = bulk_data.declare_node(node_info.id, node_info.parts);
            } else {
              node = bulk_data.declare_node(node_info.id);
            }
          }

          bulk_data.declare_relation(element, node, i, perm, scratch1, scratch2, scratch3);

          // If the current element is owned by a different processor than the node, we (the element's owning process)
          // need to share the node with the node's owning processor.
          if (node_info_map[node_id].owning_proc != our_rank) {
            bulk_data.add_node_sharing(node, node_info_map[node_id].owning_proc);
            node_info_map[node_id].non_owning_shared_procs.push_back(element_info.owning_proc);
          }
        }
      }
      for (const auto& field_data : element_info.field_data) {
        field_data->set_field_data(element);
      }
    } else {
      // We don't own the element, but if it connects to a node we own, we need to share that node with the element's
      // owning processor.
      for (const auto& node_id : element_info.node_ids) {
        if (node_id == stk::mesh::InvalidEntityId) {
          continue;
        }
        const auto& node_info = node_info_map[node_id];
        if (node_info.owning_proc == our_rank) {
          stk::mesh::Entity node = bulk_data.get_entity(stk::topology::NODE_RANK, node_id);
          if (!bulk_data.is_valid(node)) {
            const auto& node_info = node_info_map[node_id];
            if (node_info.parts.size() != 0) {
              node = bulk_data.declare_node(node_info.id, node_info.parts);
            } else {
              node = bulk_data.declare_node(node_info.id);
            }
          }

          bulk_data.add_node_sharing(node, element_info.owning_proc);
        }
      }
    }
  }

  // Declare all nodes not already declared. Set all node field data.
  for (const auto& node_info : node_info_vec_) {
    const bool we_own_node = node_info.owning_proc == our_rank;
    const bool we_share_node =
        std::find(node_info.non_owning_shared_procs.begin(), node_info.non_owning_shared_procs.end(), our_rank) !=
        node_info.non_owning_shared_procs.end();
    if (we_own_node) {
      stk::mesh::Entity node = bulk_data.get_entity(stk::topology::NODE_RANK, node_info.id);
      if (!bulk_data.is_valid(node)) {
        if (node_info.parts.size() != 0) {
          node = bulk_data.declare_node(node_info.id, node_info.parts);
        } else {
          node = bulk_data.declare_node(node_info.id);
        }
      }
      for (const auto& field_data : node_info.field_data) {
        field_data->set_field_data(node);
      }
    } else if (we_share_node) {
      stk::mesh::Entity node = bulk_data.get_entity(stk::topology::NODE_RANK, node_info.id);
      MUNDY_THROW_REQUIRE(bulk_data.is_valid(node), std::runtime_error,
                          fmt::format("Node {} is not valid and yet we are sharing it.", node_info.id));
      for (const auto& field_data : node_info.field_data) {
        field_data->set_field_data(node);
      }
    }
  }

  return *this;
}

}  // namespace mesh

}  // namespace mundy
