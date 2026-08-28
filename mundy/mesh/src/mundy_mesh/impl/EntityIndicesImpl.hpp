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

#ifndef MUNDY_MESH_IMPL_ENTITYINDICESIMPL_HPP_
#define MUNDY_MESH_IMPL_ENTITYINDICESIMPL_HPP_

/// \file impl/EntityIndicesImpl.hpp
/// \brief Implementation details for EntityIndices.hpp: selector enumeration + the per-MetaData memoization cache.

// C++ core
#include <cstddef>  // for size_t
#include <vector>   // for std::vector

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Bucket.hpp>       // for stk::mesh::MeshIndex
#include <stk_mesh/base/BulkData.hpp>     // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::get_entities
#include <stk_mesh/base/MetaData.hpp>     // for stk::mesh::MetaData attribute store
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::FastMeshIndex
#include <stk_mesh/base/Selector.hpp>     // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>        // for stk::mesh::EntityRank
#include <stk_util/ngp/NgpSpaces.hpp>     // for stk::ngp::HostRangePolicy

// Mundy
#include <mundy_utils/NgpView.hpp>  // for mundy::NgpViewT

namespace mundy {

namespace mesh {

namespace impl {

/// \brief Enumerate a (rank, selector) chunk into a host-modified `FastMeshIndex` view (no caching).
template <typename OurExecSpace>
NgpViewT<stk::mesh::FastMeshIndex*, OurExecSpace> build_local_entity_indices(const stk::mesh::BulkData& bulk_data,
                                                                             stk::mesh::EntityRank rank,
                                                                             const stk::mesh::Selector& selector) {
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

/// \brief Enumerate a (rank, selector) chunk into a host-modified `Entity` view (no caching).
template <typename OurExecSpace>
NgpViewT<stk::mesh::Entity*, OurExecSpace> build_local_entities(const stk::mesh::BulkData& bulk_data,
                                                                stk::mesh::EntityRank rank,
                                                                const stk::mesh::Selector& selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities, /*sort by global id*/ true);

  NgpViewT<stk::mesh::Entity*, OurExecSpace> ngp_local_entities("local_entities", local_entities.size());
  Kokkos::parallel_for(
      stk::ngp::HostRangePolicy(0, local_entities.size()),
      [&local_entities, &ngp_local_entities](const int i) { ngp_local_entities.view_host()(i) = local_entities[i]; });
  ngp_local_entities.modify_on_host();
  return ngp_local_entities;
}

/// \brief One cached enumeration: the `(rank, selector)` it was built for, the view, and the build-time sync count.
template <typename ViewType>
struct CachedSelectorView {
  stk::mesh::EntityRank rank;
  stk::mesh::Selector selector;
  ViewType view;
  size_t sync_count;
};

/// \brief Per-`MetaData`, per-execution-space cache holding the two memoized maps (entities and indices).
///
/// Stored as a `MetaData` attribute (like `get_or_create_class_map`). Each map is scanned linearly by
/// `(rank, selector)` equality — a selector set is small, so the scan is cheap.
template <typename OurExecSpace>
struct LocalEntityIndexCache {
  std::vector<CachedSelectorView<NgpViewT<stk::mesh::Entity*, OurExecSpace>>> entities;
  std::vector<CachedSelectorView<NgpViewT<stk::mesh::FastMeshIndex*, OurExecSpace>>> indices;
};

/// \brief Fetch (creating if needed) the per-execution-space enumeration cache on a mesh's `MetaData`.
///
/// The cache is logically-const memoization, so a `const BulkData` is fine; the `MetaData` attribute store is
/// `const_cast` exactly as `get_or_create_class_map` does.
template <typename OurExecSpace>
LocalEntityIndexCache<OurExecSpace>& get_or_create_local_entity_index_cache(const stk::mesh::BulkData& bulk_data) {
  using cache_t = LocalEntityIndexCache<OurExecSpace>;
  auto& meta_data = const_cast<stk::mesh::MetaData&>(bulk_data.mesh_meta_data());
  auto* cache = const_cast<cache_t*>(meta_data.get_attribute<cache_t>());
  if (cache == nullptr) {
    const cache_t* fresh_cache = new cache_t();
    cache = const_cast<cache_t*>(meta_data.declare_attribute_with_delete(fresh_cache));
  }
  return *cache;
}

/// \brief Return the cached view for `(rank, selector)` when its sync count matches; otherwise (re)build and update.
template <typename ViewType, typename BuildFn>
ViewType get_or_refresh_cached_view(std::vector<CachedSelectorView<ViewType>>& map, stk::mesh::EntityRank rank,
                                    const stk::mesh::Selector& selector, size_t sync_count, BuildFn&& build) {
  for (auto& entry : map) {
    if (entry.rank == rank && entry.selector == selector) {
      if (entry.sync_count != sync_count) {
        entry.view = build();
        entry.sync_count = sync_count;
      }
      return entry.view;
    }
  }
  ViewType view = build();
  map.push_back(CachedSelectorView<ViewType>{rank, selector, view, sync_count});
  return view;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_ENTITYINDICESIMPL_HPP_
