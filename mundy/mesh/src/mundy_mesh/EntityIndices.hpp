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

#ifndef MUNDY_MESH_ENTITYINDICES_HPP_
#define MUNDY_MESH_ENTITYINDICES_HPP_

/// \file EntityIndices.hpp
/// \brief Helpers that enumerate the entities of a (rank, selector) chunk into device-usable views.
///
/// These pair a selector with a dense, deterministically-ordered enumeration of its entities:
///   - `get_local_entity_indices` returns the per-entity `FastMeshIndex` (for NGP field / component access),
///   - `get_local_entities`       returns the matching `stk::mesh::Entity` (for entity keys, ghosting, etc.).
///
/// Both use `stk::mesh::get_entities(bulk, rank, selector, ...)` ordering, so the i-th `FastMeshIndex`
/// and the i-th `Entity` refer to the same entity.
///
/// \warning `FastMeshIndex` values are bucket-relative and are invalidated by mesh modification
/// (`modification_end()`). Re-enumerate after any ghosting/modification rather than reusing a stale view.

// C++ core
#include <cstddef>  // for size_t
#include <vector>   // for std::vector

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Bucket.hpp>       // for stk::mesh::MeshIndex
#include <stk_mesh/base/BulkData.hpp>     // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::get_entities
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::FastMeshIndex
#include <stk_mesh/base/Selector.hpp>     // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>        // for stk::mesh::EntityRank
#include <stk_util/ngp/NgpSpaces.hpp>     // for stk::ngp::HostRangePolicy

// Mundy
#include <mundy_utils/NgpView.hpp>  // for mundy::NgpViewT

namespace mundy {

namespace mesh {

/// \brief Get the local fast mesh indices for the entities of a (rank, selector) chunk as an NgpView.
///
/// The returned view is ordered by `stk::mesh::get_entities` and is host-modified (call `.sync_to_device()`
/// or pass through an NGP path before device use). See the file note on `FastMeshIndex` invalidation.
template <typename OurExecSpace>
NgpViewT<stk::mesh::FastMeshIndex*, OurExecSpace> get_local_entity_indices(const stk::mesh::BulkData& bulk_data,
                                                                           stk::mesh::EntityRank rank,
                                                                           const stk::mesh::Selector& selector,
                                                                           const OurExecSpace& /*exec_space*/) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities, /*sort by global id*/ true);

  NgpViewT<stk::mesh::FastMeshIndex*, OurExecSpace> ngp_local_entity_indices("local_entity_indices",
                                                                             local_entities.size());

  Kokkos::parallel_for(stk::ngp::HostRangePolicy(0, local_entities.size()),
                       [&bulk_data, &local_entities, &ngp_local_entity_indices](const int i) {
                         const stk::mesh::MeshIndex& mesh_index = bulk_data.mesh_index(local_entities[i]);
                         ngp_local_entity_indices.view_host()(i) =
                             stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
                       });

  ngp_local_entity_indices.modify_on_host();
  return ngp_local_entity_indices;
}

/// \brief Get the entities of a (rank, selector) chunk as an NgpView, in the same order as
/// `get_local_entity_indices`.
///
/// The i-th entry of this view and the i-th entry of `get_local_entity_indices(...)` refer to the same entity,
/// so the two can be used together: the index view to read NGP component/field data, the entity view to form
/// `EntityKey`s (for STK coarse search, ghosting, etc.).
template <typename OurExecSpace>
NgpViewT<stk::mesh::Entity*, OurExecSpace> get_local_entities(const stk::mesh::BulkData& bulk_data,
                                                              stk::mesh::EntityRank rank,
                                                              const stk::mesh::Selector& selector,
                                                              const OurExecSpace& /*exec_space*/) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities, /*sort by global id*/ true);

  NgpViewT<stk::mesh::Entity*, OurExecSpace> ngp_local_entities("local_entities", local_entities.size());

  Kokkos::parallel_for(stk::ngp::HostRangePolicy(0, local_entities.size()),
                       [&local_entities, &ngp_local_entities](const int i) {
                         ngp_local_entities.view_host()(i) = local_entities[i];
                       });

  ngp_local_entities.modify_on_host();
  return ngp_local_entities;
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_ENTITYINDICES_HPP_
