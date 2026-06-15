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

#ifndef MUNDY_SEARCH_IMPL_STKSEARCHBOXES_HPP_
#define MUNDY_SEARCH_IMPL_STKSEARCHBOXES_HPP_

/// \file impl/STKSearchBoxes.hpp
/// \brief STK instantiations of the unified search boxes plus their component-driven generators.

// C++ core
#include <cstddef>  // for size_t
#include <utility>  // for std::pair, std::make_pair

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntityKey.hpp>   // for stk::mesh::EntityKey (periodic identity owner)
#include <stk_mesh/base/GetNgpMesh.hpp>  // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpMesh.hpp>     // for stk::mesh::NgpMesh::fast_mesh_index
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>  // for stk::mesh::EntityRank
#include <stk_search/BoundingBox.hpp>

// Mundy
#include <mundy_geom/primitives/AABB.hpp>            // for mundy::AABB (periodic prune bbox)
#include <mundy_mesh/EntityIndices.hpp>              // for mundy::mesh::get_local_entities
#include <mundy_mesh/FieldComponent.hpp>             // for mundy::mesh::get_updated_ngp_component
#include <mundy_search/impl/PeriodicImageBoxes.hpp>  // for impl::PeriodicImages
#include <mundy_search/impl/SearchBoxes.hpp>         // for impl::SearchBoxes, impl::PeriodicImageIdentity

namespace mundy {

namespace search {

namespace impl {

/// \brief STK non-periodic search boxes: one box per entity, identity is the entity's global key.
template <typename MemorySpace>
using STKSearchBoxesT = SearchBoxes<MemorySpace, stk::search::Box<float>, stk::mesh::EntityKey>;

/// \brief STK periodic image search boxes: one box per image, identity is the imaged owner key + lattice shift.
///
/// The owner is the global `EntityKey` (not a local `Entity`) so the identity survives `coarse_search`'s distributed
/// communication and drives owner ghosting on the receiving rank.
template <typename MemorySpace, typename ImageShiftScalar = float>
using PeriodicSTKSearchBoxesT =
    SearchBoxes<MemorySpace, stk::search::Box<float>, PeriodicImageIdentity<stk::mesh::EntityKey, ImageShiftScalar>>;

/// \brief Pack a MundyGeom AABB (the component's broad-phase volume) into an `stk::search::Box`, casting corners
/// to the search scalar.
template <typename BoxScalar, typename AABBType>
KOKKOS_INLINE_FUNCTION stk::search::Box<BoxScalar> pack_search_box(const AABBType& aabb) {
  const auto& lo = aabb.min_corner();
  const auto& hi = aabb.max_corner();
  return stk::search::Box<BoxScalar>(static_cast<BoxScalar>(lo[0]), static_cast<BoxScalar>(lo[1]),
                                     static_cast<BoxScalar>(lo[2]), static_cast<BoxScalar>(hi[0]),
                                     static_cast<BoxScalar>(hi[1]), static_cast<BoxScalar>(hi[2]));
}

/// \brief Enumerate a `(rank, selector)` chunk and build its broad-phase STK search boxes from a component.
///
/// \return `{entities, boxes}` — a device entity view (dense ordinal order) and matching device box view. Both
///         alias reference-counted device storage (no deep copies). The build derives each box's `EntityKey`
///         identity from its entity when it assembles the `BoxIdentProc` views.
template <typename BoxScalar, typename ExecSpace, typename Component>
std::pair<Kokkos::View<stk::mesh::Entity*, typename ExecSpace::memory_space>,
          Kokkos::View<stk::search::Box<BoxScalar>*, typename ExecSpace::memory_space>>
make_stk_search_boxes(const stk::mesh::BulkData& bulk_data, const ExecSpace& exec, stk::mesh::EntityRank rank,
                      const stk::mesh::Selector& selector, Component& component) {
  using memory_space = typename ExecSpace::memory_space;
  using box_type = stk::search::Box<BoxScalar>;

  // Enumerate entities in a deterministic order; bring the (caller-populated) geometry field to device. Each box is
  // read through the NGP component, resolving the entity to its FastMeshIndex on device (no separate index pass).
  auto entities_ngp = mundy::mesh::get_local_entities(bulk_data, rank, selector, exec);
  entities_ngp.sync_to_device();
  component.sync_to_device();
  Kokkos::View<stk::mesh::Entity*, memory_space> entities = entities_ngp.view_device();
  auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  const size_t num_entities = entities.extent(0);

  Kokkos::View<box_type*, memory_space> boxes(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mundy_stk_search_boxes"),
                                              num_entities);
  Kokkos::parallel_for(
      "mundy_make_stk_search_boxes", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_entities),
      KOKKOS_LAMBDA(const size_t i) {
        boxes(i) = pack_search_box<BoxScalar>(ngp_component(ngp_mesh.fast_mesh_index(entities(i))));
      });

  return std::make_pair(entities, boxes);
}

/// \brief Pack enumerated entities + boxes into STK non-periodic search boxes (boxes + per-entity key identities).
///
/// Projects each box's local owner `Entity` to the global `EntityKey` identity that `coarse_search` carries across
/// ranks; reuses the input box view (no copy). The caller keeps the `Entity` view for final-list storage and ghosting.
template <typename ExecSpace, typename NgpMesh, typename EntityView, typename BoxView>
STKSearchBoxesT<typename ExecSpace::memory_space> pack_stk_search_boxes(const ExecSpace& exec, const NgpMesh& ngp_mesh,
                                                                        const stk::mesh::Selector& selector,
                                                                        const EntityView& entities,
                                                                        const BoxView& boxes) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t n = entities.extent(0);
  Kokkos::View<stk::mesh::EntityKey*, memory_space> identities(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "stk_search_ids"), n);
  Kokkos::parallel_for(
      "mundy_pack_stk_search_boxes", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
      KOKKOS_LAMBDA(const size_t i) { identities(i) = ngp_mesh.entity_key(entities(i)); });
  return STKSearchBoxesT<memory_space>(selector, boxes, identities);
}

/// \brief Pack backend-neutral periodic images into STK periodic search boxes (boxes + per-image identities).
///
/// Each image's identity is the imaged owner's global key (`ngp_mesh.entity_key(owner_entities(owner_indices(i)))`)
/// and its lattice shift — the projection from the neutral images' local owner `Entity` to the global `EntityKey`
/// that `coarse_search` carries across ranks. The dense per-owner `owner_entities` from `images` remain the build's
/// input for final-list storage and owner-ordinal recovery.
template <typename ExecSpace, typename NgpMesh, typename MScalar, typename ShiftScalar, typename MemSpace>
PeriodicSTKSearchBoxesT<MemSpace, ShiftScalar> pack_periodic_stk_search_boxes(
    const ExecSpace& exec, const NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
    const PeriodicImages<MScalar, ShiftScalar, MemSpace>& images) {
  using identity_t = PeriodicImageIdentity<stk::mesh::EntityKey, ShiftScalar>;
  using box_type = stk::search::Box<float>;
  const size_t n = images.aabbs.extent(0);
  Kokkos::View<box_type*, MemSpace> boxes(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_stk_boxes"), n);
  Kokkos::View<identity_t*, MemSpace> identities(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_stk_ids"), n);
  auto aabbs = images.aabbs;
  auto owner_entities = images.owner_entities;
  auto owner_indices = images.owner_indices;
  auto shifts = images.shifts;
  Kokkos::parallel_for(
      "mundy_pack_periodic_stk_search_boxes", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
      KOKKOS_LAMBDA(const size_t i) {
        boxes(i) = pack_search_box<float>(aabbs(i));
        identities(i) = identity_t{ngp_mesh.entity_key(owner_entities(owner_indices(i))), shifts(i)};
      });
  return PeriodicSTKSearchBoxesT<MemSpace, ShiftScalar>(selector, boxes, identities);
}

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_STKSEARCHBOXES_HPP_
