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
/// \brief Memoized helpers that enumerate the entities of a (rank, selector) chunk into device-usable views.
///
/// These pair a selector with a dense, deterministically-ordered enumeration of its entities:
///   - `get_local_entity_indices` returns the per-entity `FastMeshIndex` (for NGP field / component access),
///   - `get_local_entities`       returns the matching `stk::mesh::Entity` (for entity keys, ghosting, etc.).
///
/// Both use `stk::mesh::get_entities(bulk, rank, selector, ...)` ordering, so the i-th `FastMeshIndex`
/// and the i-th `Entity` refer to the same entity.
///
/// \par Memoization
/// Enumerating a selector is host work (`get_entities` + a host fill + a host→device sync) that recurs across phases
/// and consumers (the neighbor-list build, excluders, rebuilders, …). To avoid redundant re-enumeration, the
/// results are cached on the `BulkData`'s `MetaData` as an attribute (one cache per execution space), keyed by
/// `(rank, selector)` using `Selector::operator==`, and tagged with the `BulkData::synchronized_count()` at which
/// they were built. A lookup returns the cached view when the synchronized count is unchanged; otherwise the view
/// is rebuilt and the cached count updated. Because `synchronized_count()` advances on every mesh modification (the
/// only thing that can change a selector's entity sequence, or invalidate a bucket-relative `FastMeshIndex`), the
/// cache is self-invalidating — callers never see a stale enumeration. The enumeration and cache machinery live in
/// impl/EntityIndicesImpl.hpp.

// Trilinos
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData, synchronized_count
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/NgpMesh.hpp>   // for stk::mesh::FastMeshIndex
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank

// Mundy
#include <mundy_mesh/impl/EntityIndicesImpl.hpp>  // for impl enumeration + memoization cache
#include <mundy_utils/NgpView.hpp>                // for mundy::NgpViewT

namespace mundy {

namespace mesh {

/// \brief Get the local fast mesh indices for the entities of a (rank, selector) chunk as an NgpView (memoized).
///
/// The returned view is ordered by `stk::mesh::get_entities` and is host-modified (call `.sync_to_device()`
/// or pass through an NGP path before device use). The result is cached per the file note on memoization, so
/// repeated calls within one mesh-modification epoch reuse the same view.
template <typename OurExecSpace>
NgpViewT<stk::mesh::FastMeshIndex*, OurExecSpace> get_local_entity_indices(const stk::mesh::BulkData& bulk_data,
                                                                           stk::mesh::EntityRank rank,
                                                                           const stk::mesh::Selector& selector,
                                                                           const OurExecSpace& /*exec_space*/) {
  auto& cache = impl::get_or_create_local_entity_index_cache<OurExecSpace>(bulk_data);
  return impl::get_or_refresh_cached_view(
      cache.indices, rank, selector, bulk_data.synchronized_count(),
      [&]() { return impl::build_local_entity_indices<OurExecSpace>(bulk_data, rank, selector); });
}

/// \brief Get the entities of a (rank, selector) chunk as an NgpView, in the same order as
/// `get_local_entity_indices` (memoized).
///
/// The i-th entry of this view and the i-th entry of `get_local_entity_indices(...)` refer to the same entity,
/// so the two can be used together: the index view to read NGP component/field data, the entity view to form
/// `EntityKey`s (for STK coarse search, ghosting, etc.). The result is cached per the file note on memoization.
template <typename OurExecSpace>
NgpViewT<stk::mesh::Entity*, OurExecSpace> get_local_entities(const stk::mesh::BulkData& bulk_data,
                                                              stk::mesh::EntityRank rank,
                                                              const stk::mesh::Selector& selector,
                                                              const OurExecSpace& /*exec_space*/) {
  auto& cache = impl::get_or_create_local_entity_index_cache<OurExecSpace>(bulk_data);
  return impl::get_or_refresh_cached_view(
      cache.entities, rank, selector, bulk_data.synchronized_count(),
      [&]() { return impl::build_local_entities<OurExecSpace>(bulk_data, rank, selector); });
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_ENTITYINDICES_HPP_
