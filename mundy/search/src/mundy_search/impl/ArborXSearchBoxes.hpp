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

#ifndef MUNDY_SEARCH_IMPL_ARBORXSEARCHBOXES_HPP_
#define MUNDY_SEARCH_IMPL_ARBORXSEARCHBOXES_HPP_

/// \file impl/ArborXSearchBoxes.hpp
/// \brief ArborX instantiations of the unified search boxes plus their ArborX::AccessTraits specializations.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// C++ core
#include <cstddef>  // for size_t

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>  // for stk::mesh::EntityRank

// Mundy
#include <mundy_geom/primitives/AABB.hpp>            // for mundy::AABB
#include <mundy_geom/primitives/Point.hpp>           // for mundy::Point (ArborX box corners)
#include <mundy_mesh/EntityIndices.hpp>              // for mundy::mesh::get_local_entities, get_local_entity_indices
#include <mundy_mesh/FieldComponent.hpp>             // for mundy::mesh::get_updated_ngp_component
#include <mundy_search/impl/PeriodicImageBoxes.hpp>  // for impl::PeriodicImages
#include <mundy_search/impl/SearchBoxes.hpp>         // for impl::SearchBoxes, impl::PeriodicImageIdentity

namespace mundy {

namespace search {

namespace impl {

/// \brief ArborX non-periodic search boxes: one box per entity, identity is the entity itself.
/// \tparam MemorySpace Kokkos memory space in which the boxes and identities live.
template <typename MemorySpace>
using ArborXSearchBoxesT = SearchBoxes<MemorySpace, ArborX::Box, stk::mesh::Entity>;

/// \brief ArborX periodic image search boxes: one box per image, identity is the imaged owner entity + lattice shift.
/// \tparam MemorySpace Kokkos memory space in which the boxes and identities live.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar = float>
using PeriodicArborXSearchBoxesT =
    SearchBoxes<MemorySpace, ArborX::Box, PeriodicImageIdentity<stk::mesh::Entity, ImageShiftScalar>>;

/// \brief Pack a MundyGeom AABB (the component's broad-phase volume) into an `ArborX::Box` (always float).
template <typename AABBType>
KOKKOS_INLINE_FUNCTION ArborX::Box pack_arborx_box(const AABBType& aabb) {
  const auto& lo = aabb.min_corner();
  const auto& hi = aabb.max_corner();
  return ArborX::Box{ArborX::Point{static_cast<float>(lo[0]), static_cast<float>(lo[1]), static_cast<float>(lo[2])},
                     ArborX::Point{static_cast<float>(hi[0]), static_cast<float>(hi[1]), static_cast<float>(hi[2])}};
}

/// \brief Enumerate a `(rank, selector)` chunk and build its broad-phase ArborX boxes from a component.
///
/// The component must yield an AABB (the ArborX list's broad-phase volume). Returned views alias reference-counted
/// device storage (no deep copies).
/// \return `{entities, boxes}` — device entity view (dense ordinal order) and matching device `ArborX::Box` view. The
///         entity view doubles as the non-periodic identity view.
template <typename ExecSpace, typename Component>
std::pair<Kokkos::View<stk::mesh::Entity*, typename ExecSpace::memory_space>,
          Kokkos::View<ArborX::Box*, typename ExecSpace::memory_space>>
make_arborx_search_boxes(const stk::mesh::BulkData& bulk_data, const ExecSpace& exec, stk::mesh::EntityRank rank,
                         const stk::mesh::Selector& selector, Component& component) {
  using memory_space = typename ExecSpace::memory_space;

  // Enumerate entities + local indices in a common, deterministic order; bring the (caller-populated) geometry
  // field to device and read it through the NGP component.
  auto entities_ngp = mundy::mesh::get_local_entities(bulk_data, rank, selector, exec);
  auto indices_ngp = mundy::mesh::get_local_entity_indices(bulk_data, rank, selector, exec);
  entities_ngp.sync_to_device();
  indices_ngp.sync_to_device();
  component.sync_to_device();
  Kokkos::View<stk::mesh::Entity*, memory_space> entities = entities_ngp.view_device();
  auto indices = indices_ngp.view_device();
  auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
  const size_t num_entities = entities.extent(0);

  Kokkos::View<ArborX::Box*, memory_space> boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "mundy_arborx_search_boxes"), num_entities);
  Kokkos::parallel_for(
      "mundy_make_arborx_search_boxes", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_entities),
      KOKKOS_LAMBDA(const size_t i) { boxes(i) = pack_arborx_box(ngp_component(indices(i))); });

  return std::make_pair(entities, boxes);
}

/// \brief Pack backend-neutral periodic images into ArborX periodic search boxes (boxes + per-image identities).
///
/// Each image's identity is the imaged owner entity (`owner_entities(owner_indices(i))`) and its lattice shift. The
/// dense per-owner `owner_entities`/`owner_indices` from `images` remain the build's inputs for final-list storage and
/// owner-ordinal recovery; only the encapsulated identity travels into the search.
template <typename ExecSpace, typename MScalar, typename ShiftScalar, typename MemSpace>
PeriodicArborXSearchBoxesT<MemSpace, ShiftScalar> pack_periodic_arborx_search_boxes(
    const ExecSpace& exec, const stk::mesh::Selector& selector,
    const PeriodicImages<MScalar, ShiftScalar, MemSpace>& images) {
  using identity_t = PeriodicImageIdentity<stk::mesh::Entity, ShiftScalar>;
  const size_t n = images.aabbs.extent(0);
  Kokkos::View<ArborX::Box*, MemSpace> boxes(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_arborx_boxes"), n);
  Kokkos::View<identity_t*, MemSpace> identities(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_arborx_ids"), n);
  auto aabbs = images.aabbs;
  auto owner_entities = images.owner_entities;
  auto owner_indices = images.owner_indices;
  auto shifts = images.shifts;
  Kokkos::parallel_for(
      "mundy_pack_periodic_arborx_search_boxes", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
      KOKKOS_LAMBDA(const size_t i) {
        boxes(i) = pack_arborx_box(aabbs(i));
        identities(i) = identity_t{owner_entities(owner_indices(i)), shifts(i)};
      });
  return PeriodicArborXSearchBoxesT<MemSpace, ShiftScalar>(selector, boxes, identities);
}

}  // namespace impl

}  // namespace search

}  // namespace mundy

namespace ArborX {

#if ARBORX_VERSION < 10799
/// \struct AccessTraits<mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy ArborX search boxes (old ArborX API only).
///
/// Identity-agnostic: only the box view is read. For ArborX >= 1.7.99 the BVH is constructed via
/// `attach_indices<int>(source_boxes.boxes())` and this specialization is not needed.
/// \tparam MemorySpace Kokkos memory space for the search boxes.
/// \tparam Identity Per-element identity payload (unused by the BVH primitives).
template <typename MemorySpace, typename Identity>
struct AccessTraits<mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>, PrimitivesTag> {
  //! Search-box wrapper type.
  using boxes_type = mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of primitive boxes.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive box for a source ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const boxes_type& boxes, size_type index) {
    return boxes.box(index);
  }
};
#endif  // ARBORX_VERSION < 10799

/// \struct AccessTraits<mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy ArborX search boxes.
///
/// Identity-agnostic: each target box becomes an intersection predicate with its dense ordinal attached as data.
/// \tparam MemorySpace Kokkos memory space for the search boxes.
/// \tparam Identity Per-element identity payload (unused by the predicate boxes).
template <typename MemorySpace, typename Identity>
struct AccessTraits<mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Search-box wrapper type.
  using boxes_type = mundy::search::impl::SearchBoxes<MemorySpace, ArborX::Box, Identity>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of predicates.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target ordinal.
  static KOKKOS_FUNCTION auto get(const boxes_type& boxes, size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

}  // namespace ArborX

#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // MUNDY_SEARCH_IMPL_ARBORXSEARCHBOXES_HPP_
