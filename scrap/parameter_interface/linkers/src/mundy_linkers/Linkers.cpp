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

/// \file Linkers.cpp
/// \brief Defintion of the Linkers helper functions

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <memory>         // for std::shared_ptr, std::unique_ptr
#include <string>         // for std::string
#include <type_traits>    // for std::enable_if, std::is_base_of
#include <unordered_set>  // for std::unordered_set
#include <vector>         // for std::vector

// Trilinos
#include <stk_mesh/base/EntityLess.hpp>                         // for stk::mesh::EntityLess
#include <stk_mesh/base/FieldParallel.hpp>                      // for stk::mesh::communicate_field_data
#include <stk_mesh/baseImpl/ForEachEntityLoopAbstractions.hpp>  // for stk::mesh::impl::for_each_selected_entity_run_no_threads
#include <stk_topology/topology.hpp>                            // for stk::topology
#include <stk_util/parallel/CommSparse.hpp>                     // for stk::CommSparse

// Mundy
#include <mundy_agents/Agents.hpp>          // for mundy::agents::Agents
#include <mundy_agents/RankedAssembly.hpp>  // for mundy::agents::RankedAssembly
#include <mundy_core/StringLiteral.hpp>     // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_linkers/Linkers.hpp>        // for mundy::linkers::fixup_linker_node_sharing
#include <mundy_mesh/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/fmt_stk_types.hpp>     // adds fmt::format for stk types
#include <mundy_meta/FieldReqs.hpp>         // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>          // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>          // for mundy::meta::PartReqs

namespace mundy {

namespace linkers {

void fixup_linker_entity_ghosting(stk::mesh::BulkData& bulk_data, const LinkedEntitiesFieldType& linked_entities_field,
                                  stk::mesh::Field<int>& linked_entity_owners_field,
                                  const stk::mesh::Selector& linker_selector) {
  MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error,
                      "fixup_linker_entity_sharing: The mesh must be in a modification cycle.");

  // If the mesh is serial, then there's nothing to do.
  const int parallel_size = bulk_data.parallel_size();
  const int parallel_rank = bulk_data.parallel_rank();
  if (parallel_size == 1) {
    return;
  }

  // Stage 1:
  // Each locally owned linker has access to the entities it links either through ownership or the aura. It is the
  // only process guarenteed to know the ownership of the linked entities. We propagate this information to all other
  // processes that ghost the linker via the linked_entity_owners_field.
  mundy::mesh::for_each_entity_run(
      bulk_data, stk::topology::CONSTRAINT_RANK, linker_selector & bulk_data.mesh_meta_data().locally_owned_part(),
      [&linked_entities_field, &linked_entity_owners_field, &parallel_rank](const stk::mesh::BulkData& bulk_data,
                                                                            const stk::mesh::Entity& linker) {
        const unsigned bytes_per_entity =
            stk::mesh::field_bytes_per_entity(linked_entities_field, bulk_data.bucket(linker));
        const size_t max_num_linked_entities = bytes_per_entity / linked_entities_field.data_traits().size_of;
        const stk::mesh::EntityKey::entity_key_t* key_t_ptr =
            reinterpret_cast<stk::mesh::EntityKey::entity_key_t*>(stk::mesh::field_data(linked_entities_field, linker));
        int* owner_ptr = stk::mesh::field_data(linked_entity_owners_field, linker);
        for (unsigned i = 0; i < max_num_linked_entities; ++i) {
          const bool entity_exist = key_t_ptr[i] != stk::mesh::EntityKey::entity_key_t::INVALID;
          if (entity_exist) {
            const stk::mesh::Entity linked_entity = bulk_data.get_entity(key_t_ptr[i]);
            MUNDY_THROW_ASSERT(bulk_data.is_valid(linked_entity), std::logic_error,
                               "fixup_linker_entity_sharing: The linked entity is somehow invalid.");
            owner_ptr[i] = bulk_data.parallel_owner_rank(linked_entity);
          }
        }
      });

  // Communicate the linked entities and their owners to all processes.
  stk::mesh::communicate_field_data(bulk_data, {&linked_entities_field, &linked_entity_owners_field});

  // Stage 2:
  // Each linker connected to a shared node requests the owning process of any linked entity (that we do not have access
  // to) to ghost said entity to the current process.
  using EntityKeyOwnerRequester = std::tuple<stk::mesh::EntityKey, int, int>;
  std::vector<EntityKeyOwnerRequester> requested_ghosts_from_procs;
  stk::mesh::Selector shared = bulk_data.mesh_meta_data().globally_shared_part();
  stk::mesh::impl::for_each_selected_entity_run_no_threads(
      bulk_data, stk::topology::NODE_RANK, shared,
      [&linker_selector, &linked_entities_field, &linked_entity_owners_field, &requested_ghosts_from_procs,
       &parallel_rank](const stk::mesh::BulkData& bulk_data, const stk::mesh::MeshIndex& mesh_index) {
        const stk::mesh::Bucket& bucket = *mesh_index.bucket;
        const unsigned bucket_ord = mesh_index.bucket_ordinal;
        const stk::mesh::Entity& node = bucket[bucket_ord];
        const stk::mesh::Entity* node_constraints = bulk_data.begin(node, stk::topology::CONSTRAINT_RANK);
        const unsigned num_constraints = bulk_data.num_connectivity(node, stk::topology::CONSTRAINT_RANK);
        for (unsigned i = 0; i < num_constraints; ++i) {
          const stk::mesh::Entity& constraint = node_constraints[i];
          stk::mesh::Bucket& constraint_bucket = bulk_data.bucket(constraint);
          const bool is_linker = linker_selector(constraint_bucket);

          if (is_linker) {
            const unsigned bytes_per_entity =
                stk::mesh::field_bytes_per_entity(linked_entities_field, constraint_bucket);
            const size_t max_num_linked_entities = bytes_per_entity / linked_entities_field.data_traits().size_of;
            const stk::mesh::EntityKey::entity_key_t* key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t*>(
                stk::mesh::field_data(linked_entities_field, constraint));
            const int* owner_ptr = stk::mesh::field_data(linked_entity_owners_field, constraint);

            for (unsigned j = 0; j < max_num_linked_entities; ++j) {
              const bool entity_exists = key_t_ptr[j] != stk::mesh::EntityKey::entity_key_t::INVALID;
              if (entity_exists) {
                const bool we_do_not_have_access_to_linked_entity =
                    !bulk_data.is_valid(bulk_data.get_entity(key_t_ptr[j]));
                if (we_do_not_have_access_to_linked_entity) {
                  // Request the owning process of the linked entity to ghost said entity to us.
                  requested_ghosts_from_procs.emplace_back(key_t_ptr[j], owner_ptr[j], parallel_rank);
                }
              }
            }
          }
        }
      });

  // Sort and uniquify the list of requested ghosts.
  std::sort(requested_ghosts_from_procs.begin(), requested_ghosts_from_procs.end(),
            [](const EntityKeyOwnerRequester& a, const EntityKeyOwnerRequester& b) {
              return std::get<0>(a) < std::get<0>(b);
            });
  requested_ghosts_from_procs.erase(std::unique(requested_ghosts_from_procs.begin(), requested_ghosts_from_procs.end()),
                                    requested_ghosts_from_procs.end());

  // Allocate and pack the requests to send to the owning processes.
  stk::CommSparse comm_sparse(bulk_data.parallel());
  for (const EntityKeyOwnerRequester& ekor : requested_ghosts_from_procs) {
    const int owner = std::get<1>(ekor);
    comm_sparse.send_buffer(owner).skip<stk::mesh::EntityKey>(1).skip<int>(1);
  }

  comm_sparse.allocate_buffers();

  for (const EntityKeyOwnerRequester& ekor : requested_ghosts_from_procs) {
    const stk::mesh::EntityKey entity_key = std::get<0>(ekor);
    const int owner = std::get<1>(ekor);
    const int requester = std::get<2>(ekor);
    comm_sparse.send_buffer(owner).pack<stk::mesh::EntityKey>(entity_key).pack<int>(requester);
  }

  comm_sparse.communicate();

  std::vector<stk::mesh::EntityProc> send_ghosts;
  for (int p = 0; p < parallel_size; ++p) {
    stk::CommBuffer& buf = comm_sparse.recv_buffer(p);
    while (buf.remaining()) {
      stk::mesh::EntityKey entity_key;
      int requester_proc = 0;
      buf.unpack(entity_key).unpack(requester_proc);

      const stk::mesh::Entity entity = bulk_data.get_entity(entity_key);
      MUNDY_THROW_ASSERT(
          bulk_data.is_valid(entity), std::logic_error,
          fmt::format("fixup_linker_entity_sharing: Rank {} received request for {} from {} but the entity is invalid.",
                      parallel_rank, entity_key, requester_proc));
      MUNDY_THROW_ASSERT(
          bulk_data.parallel_owner_rank(entity) == parallel_rank, std::logic_error,
          fmt::format("fixup_linker_entity_sharing: Rank {} received request for {} from {} but we do not own it.",
                      parallel_rank, entity_key, requester_proc));

      // Receiving a request to ghost an entity we own.
      send_ghosts.emplace_back(entity, requester_proc);
    }
  }

  // Stage 3: Perform the ghosting.
  stk::mesh::Ghosting& ghosting = bulk_data.create_ghosting("MUNDY_LINKED_ENTITY_GHOSTING");
  bulk_data.change_ghosting(ghosting, send_ghosts);
}

}  // namespace linkers

}  // namespace mundy
