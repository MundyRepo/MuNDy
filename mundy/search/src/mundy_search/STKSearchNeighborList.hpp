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

#ifndef MUNDY_SEARCH_STKSEARCHNEIGHBORLIST_HPP_
#define MUNDY_SEARCH_STKSEARCHNEIGHBORLIST_HPP_

/// \file STKSearchNeighborList.hpp
/// \brief STK-coarse-search-backed concrete neighbor-list types and their NeighborListBuildTraits specializations.

// C++ core
#include <concepts>       // for std::same_as
#include <cstddef>        // for size_t
#include <limits>         // for std::numeric_limits
#include <stdexcept>      // for std::invalid_argument, std::out_of_range
#include <unordered_set>  // for std::unordered_set (Phase E: extended source dedup)
#include <vector>         // for std::vector (Phase D: ghosting send list)

// Trilinos
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>     // for Kokkos::UnorderedMap (key-to-ordinal maps)
#include <stk_mesh/base/BulkData.hpp>  // for modification_begin/end, get_entity, entity_key
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntityKey.hpp>               // for stk::mesh::EntityKey
#include <stk_mesh/base/FieldParallel.hpp>           // for stk::mesh::communicate_field_data (ghosted geometry)
#include <stk_mesh/base/GetNgpMesh.hpp>              // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/HashEntityAndEntityKey.hpp>  // for std::hash<stk::mesh::EntityKey>
#include <stk_mesh/base/NgpMesh.hpp>                 // for stk::mesh::NgpMesh (entity_key on device)
#include <stk_mesh/base/Selector.hpp>
#include <stk_search/BoundingBox.hpp>   // for stk::search::Box
#include <stk_search/BoxIdent.hpp>      // for stk::search::BoxIdentProc, IdentProcIntersection
#include <stk_search/CoarseSearch.hpp>  // for stk::search::coarse_search
#include <stk_search/IdentProc.hpp>     // for stk::search::IdentProc
#include <stk_search/SearchMethod.hpp>  // for stk::search::MORTON_LBVH
#include <stk_util/ngp/NgpSpaces.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for MPI_Allreduce (global target bbox in the periodic build)

// Mundy
#include <mundy_math/Vector3.hpp>                    // for mundy::Vector3
#include <mundy_search/NeighborListBuildTraits.hpp>  // for NeighborListBuildTraits, NeighborListInputType
#include <mundy_search/SearchCandidate.hpp>          // for NeighborSearchCandidate
#include <mundy_search/SearchInput.hpp>              // for SearchInput / PeriodicSearchInput (component inputs)
#include <mundy_search/impl/NarrowPhaseFilter.hpp>   // for impl::apply_narrow_phase
#include <mundy_search/impl/STKSearchBoxes.hpp>      // for impl::make_stk_search_boxes, impl::PeriodicSTKSearchBoxesT
#include <mundy_utils/host_ptr.hpp>                  // for host_ptr
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

/// \class STKSearchNeighborList
/// \brief STK coarse-search neighbor list mapped into Mundy's common access surface.
///
/// Stores compressed target-to-source neighbor data in the same shape as `ArborX1dNeighborList`, backed by STK's
/// distributed coarse search.
template <typename MemorySpace = stk::ngp::MemSpace>
class STKSearchNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using size_type = size_t;
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  STKSearchNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList(const STKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList(STKSearchNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList& operator=(const STKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList& operator=(STKSearchNeighborList&&) = default;

  /// \brief Construct a list from already-built compressed storage.
  /// \param target_selector [in] Selector defining the target chunk used during the build.
  /// \param source_selector [in] Selector defining the source chunk used during the build.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  STKSearchNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                        const entity_view_t& target_entities, const entity_view_t& source_entities,
                        const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        source_indices_(source_indices),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "STKSearchNeighborList: offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return *target_selector_;
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return *source_selector_;
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source ordinal for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "STKSearchNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw source-index view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_indices() const noexcept {
    return source_indices_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target chunk.
  host_ptr<stk::mesh::Selector> target_selector_;
  //! Selector defining the source chunk.
  host_ptr<stk::mesh::Selector> source_selector_;
  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source ordinals for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "STKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

/// \class PeriodicSTKSearchNeighborList
/// \brief STK coarse-search neighbor list with compressed owner-pair storage and per-object periodic image shifts.
///
/// Consumes periodic STK coarse-search image pairs, collapses them to owner ordinals, and retains the per-object image
/// shifts: a target shift per owner and a source shift per stored pair. `target_image_shift(target_index)` and
/// `source_image_shift(target_index, neighbor_ordinal)` give the per-object shifts; a kernel that wants the pairwise
/// relative shift computes `source_image_shift − target_image_shift` itself.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicSTKSearchNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using source_index_type = size_type;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicSTKSearchNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList(const PeriodicSTKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList(PeriodicSTKSearchNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList& operator=(const PeriodicSTKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList& operator=(PeriodicSTKSearchNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built compressed storage.
  /// \param target_selector [in] Selector defining the target owner chunk used during the build.
  /// \param source_selector [in] Selector defining the source owner chunk used during the build.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param target_image_shifts [in] Per-target-owner image shift (target's original -> imaged reference point).
  /// \param source_owner_indices [in] Dense source owner ordinal for every stored pair.
  /// \param source_image_shifts [in] Per-pair source owner image shift (original -> imaged reference point).
  /// \param offsets [in] Target owner offsets into `source_owner_indices`; extent must be `num_targets + 1`.
  PeriodicSTKSearchNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                                const entity_view_t& target_entities, const entity_view_t& source_entities,
                                const image_shift_view_t& target_image_shifts,
                                const source_index_view_t& source_owner_indices,
                                const image_shift_view_t& source_image_shifts, const offset_view_t& offsets)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        target_image_shifts_(target_image_shifts),
        source_owner_indices_(source_owner_indices),
        source_image_shifts_(source_image_shifts),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: offsets extent must be num_targets + 1.");
    MUNDY_THROW_ASSERT(target_image_shifts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: target_image_shifts and target_entities must have the same "
                       "extent.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == source_image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: source_owner_indices and source_image_shifts must have the "
                       "same extent.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable target owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable source owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target owner chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return *target_selector_;
  }

  /// \brief Get the selector defining the source owner chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return *source_selector_;
  }

  /// \brief Get the total number of stored periodic neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_owner_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source owner ordinal for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_owner_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the source owner's image shift for a stored pair (original -> imaged reference point).
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type source_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    return source_image_shifts_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get a target owner's image shift: the displacement from its original to its imaged reference point.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  image_shift_type target_image_shift(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::target_image_shift target index out of range.");
    return target_image_shifts_(target_index);
  }

  /// \brief Get the neighbor owner entity for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target owner entity for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw flattened source-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_owner_indices() const noexcept {
    return source_owner_indices_;
  }

  /// \brief Get the raw flattened source-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t source_image_shifts() const noexcept {
    return source_image_shifts_;
  }

  /// \brief Get the raw per-target-owner image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t target_image_shifts() const noexcept {
    return target_image_shifts_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target owner chunk.
  host_ptr<stk::mesh::Selector> target_selector_;
  //! Selector defining the source owner chunk.
  host_ptr<stk::mesh::Selector> source_selector_;
  //! Target owner entities indexed by dense target owner ordinal.
  entity_view_t target_entities_;
  //! Source owner entities indexed by dense source owner ordinal.
  entity_view_t source_entities_;
  //! Per-target-owner image shift (original -> imaged reference point), indexed by dense target owner ordinal.
  image_shift_view_t target_image_shifts_;
  //! Flattened dense source owner ordinals for each stored periodic pair.
  source_index_view_t source_owner_indices_;
  //! Flattened per-pair source owner image shift (original -> imaged reference point).
  image_shift_view_t source_image_shifts_;
  //! Per-target-owner offsets into `source_owner_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

// ---------------------------------------------------------------------------
// Optional build-phase profiling
// ---------------------------------------------------------------------------
// Set `enable_stk_build_profiling = true` before calling build() to populate
// `stk_build_last_timings` on return.  Inserts Kokkos::fence() between each
// phase and records wall time in milliseconds.  Zero overhead when false.

/// Per-phase wall-time breakdown for one STKSearchNeighborList build call.
struct STKBuildPhaseTimings {
  double phase_a_ms{0};   ///< A: build BoxIdentProc search views
  double phase_b_ms{0};   ///< B: build target EntityKey->ordinal map
  double phase_c_ms{0};   ///< C: stk::search::coarse_search (MORTON_LBVH + MPI)
  double phase_d_ms{0};   ///< D: mirror results to host + ghosting coordination
  double phase_e_ms{0};   ///< E: build extended source entity view (host->device)
  double phase_f_ms{0};   ///< F: refresh NgpMesh + build source EntityKey->ordinal map
  double phase_g0_ms{0};  ///< G0: precompute valid target/source ordinal pairs
  double phase_g_ms{0};   ///< G: count pass (atomic per-target increments)
  double phase_h_ms{0};   ///< H: prefix scan + write-position init
  double phase_i_ms{0};   ///< I: fill pass (atomic slot allocation)
  double phase_j_ms{0};   ///< J: optional per-row insertion sort
  double phase_k_ms{0};   ///< K: construct list object
  double total_ms() const {
    return phase_a_ms + phase_b_ms + phase_c_ms + phase_d_ms + phase_e_ms + phase_f_ms + phase_g0_ms + phase_g_ms +
           phase_h_ms + phase_i_ms + phase_j_ms + phase_k_ms;
  }
};

/// Set to true before calling build() to enable per-phase timing.
inline bool enable_stk_build_profiling{false};
/// Populated by the most recent build() call when enable_stk_build_profiling is true.
inline STKBuildPhaseTimings stk_build_last_timings{};

// -----------------------------------------------------------------------------
// NeighborListBuildTraits specializations for STK-search-backed list types
// -----------------------------------------------------------------------------

/// \struct NeighborListBuildTraits<STKSearchNeighborList<MemorySpace>>
/// \brief Build traits for `STKSearchNeighborList`: non-periodic STK coarse-search compressed storage.
///
/// The build runs `stk::search::coarse_search`, applies the builder's excluder chain, groups results by target,
/// and returns compressed target-to-source storage.
template <typename MemorySpace>
struct NeighborListBuildTraits<STKSearchNeighborList<MemorySpace>> {
  //! \name Aliases
  //@{

  using list_type = STKSearchNeighborList<MemorySpace>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `STKSearchNeighborList` (none required).
  struct args_type {};
  //@}

  //! \name Build
  //@{

  /// \brief Build an up-to-date neighbor list from component-backed target and source inputs.
  ///
  /// Target and source inputs are `SearchInput`s pairing a selector with a geometry-yielding component. The build
  /// enumerates each chunk, reads its geometry from the component to generate broad-phase boxes, runs
  /// `stk::search::coarse_search`, applies the excluder chain, and returns compressed target-to-source storage.
  ///
  /// On multi-rank runs a ghosting side effect occurs: remote source entities are ghosted into the local process
  /// so that every target can reach its neighbors. The build communicates the source component's field over the
  /// ghosting group `"MUNDY_STK_SEARCH_NL_GHOSTING"` so the ghosted sources carry valid geometry; the group
  /// persists on `bulk_data` after the call returns for any additional fields the caller needs.
  ///
  /// \param builder [in] Complete builder.
  /// \param bulk_data [in] STK bulk data.
  /// \param args [in] Build-specific parameters (unused).
  template <typename Builder>
    requires AABBSearchInputType<typename Builder::target_input_type> &&
             AABBSearchInputType<typename Builder::source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& /*args*/) {
    // ===========================================================================
    // Type aliases
    // ===========================================================================

    using list_type = STKSearchNeighborList<MemorySpace>;
    using exec_space = typename Builder::execution_space;
    using size_type = typename list_type::size_type;
    using entity_view_t = typename list_type::entity_view_t;
    // The component geometry scalar is the user's choice; the STK coarse search runs in this (float) precision.
    using box_scalar = float;
    using box_type = stk::search::Box<box_scalar>;

    // STK search identifier: EntityKey encodes both entity rank and entity ID.
    // Required for `bulk_data.get_entity(entity_key)` in the ghosting phase;
    // EntityId alone lacks the entity rank and cannot be used there.
    using ident_proc_t = stk::search::IdentProc<stk::mesh::EntityKey, int>;

    // Per-entity search input for the Kokkos::View-based coarse_search overload.
    using box_ident_proc_t = stk::search::BoxIdentProc<box_type, ident_proc_t>;

    // Per-result type written by coarse_search: domain = target, range = source.
    using intersection_t = stk::search::IdentProcIntersection<ident_proc_t, ident_proc_t>;

    using search_view_t = Kokkos::View<box_ident_proc_t*, MemorySpace>;
    using results_view_t = Kokkos::View<intersection_t*, MemorySpace>;

    // Device-side key-to-ordinal map.  EntityKey is effectively a uint64_t,
    // so Kokkos::pod_hash gives correct, device-callable hashing.
    using key_map_t = Kokkos::UnorderedMap<stk::mesh::EntityKey, size_type, MemorySpace>;

    // ===========================================================================
    // Unpack builder state
    // ===========================================================================

    // Inputs are copied (non-const) so the build can synchronize each component to device before reading.
    auto target_input = builder.target_input();
    auto source_input = builder.source_input();
    const auto exec_sp = builder.exec_space();

    const int my_rank = bulk_data.parallel_rank();
    const bool single_rank = bulk_data.parallel_size() == 1;

    // ===========================================================================
    // Generate broad-phase search boxes from the component inputs (device)
    // ===========================================================================
    //
    // Each input is a (selector, component) pair.  make_stk_search_boxes enumerates the selected entities (defining
    // the dense ordinal ordering used below), reads each entity's geometry from the component on device, and packs
    // its broad-phase box.  Because geometry is read from a field, the build can later supply geometry for ghosted
    // sources (Phase D field comm).  Returned views alias reference-counted device storage (no deep copies).

    auto target_search = impl::make_stk_search_boxes<box_scalar>(bulk_data, exec_sp, target_input.rank(),
                                                                 target_input.selector(), target_input.component());
    auto source_search = impl::make_stk_search_boxes<box_scalar>(bulk_data, exec_sp, source_input.rank(),
                                                                 source_input.selector(), source_input.component());
    const entity_view_t target_entities_dev = target_search.first;
    const entity_view_t source_entities_dev = source_search.first;

    const size_type num_targets = target_entities_dev.extent(0);
    const size_type num_sources = source_entities_dev.extent(0);

    // NgpMesh must be obtained before any mesh modification (Phase D ghosting); the pack projects each box's local
    // owner Entity to the global EntityKey identity carried by `coarse_search`. The Entity views are kept for
    // final-list storage and ghosting; the search consumes the unified SearchBoxes (boxes + key identities).
    auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto target_boxes = impl::pack_stk_search_boxes(exec_sp, ngp_mesh, target_input.selector(), target_search.first,
                                                    target_search.second);
    auto source_boxes = impl::pack_stk_search_boxes(exec_sp, ngp_mesh, source_input.selector(), source_search.first,
                                                    source_search.second);
    auto target_boxes_dev = target_boxes.boxes();
    auto target_identities_dev = target_boxes.identities();
    auto source_boxes_dev = source_boxes.boxes();
    auto source_identities_dev = source_boxes.identities();

    // ===========================================================================
    // Optional phase-profiling timer (zero overhead when disabled)
    // ===========================================================================
    //
    // stk_prof_mark(slot) fences the device, records elapsed ms into `slot`,
    // and resets the timer for the next phase.  All calls are no-ops when
    // enable_stk_build_profiling is false.

    Kokkos::Timer stk_prof_timer_;
    const auto stk_prof_mark = [&](double& slot) {
      if (!enable_stk_build_profiling) return;
      Kokkos::fence();
      slot = stk_prof_timer_.seconds() * 1000.0;
      stk_prof_timer_.reset();
    };
    if (enable_stk_build_profiling) {
      stk_build_last_timings = {};
      Kokkos::fence();
      stk_prof_timer_.reset();
    }

    // ===========================================================================
    // Phase A — Build STK search input views (device)
    // ===========================================================================
    //
    // Target boxes form the *domain*; source boxes form the *range*.  This keeps
    // the result vocabulary consistent: `result.domainIdentProc` always refers to
    // a target entity and `result.rangeIdentProc` always to a source entity.
    //
    // `ngp_mesh.entity_key(entity)` is a KOKKOS_FUNCTION that returns the
    // `EntityKey` encoding both entity rank and entity ID.  We must obtain the
    // NgpMesh before any mesh modification; it is refreshed in Phase F after
    // modification_end().

    search_view_t target_search_view("mundy_stk_nl_tgt_search", num_targets);
    search_view_t source_search_view("mundy_stk_nl_src_search", num_sources);

    Kokkos::parallel_for(
        "mundy_stk_nl_build_target_search", Kokkos::RangePolicy<exec_space>(0, num_targets),
        KOKKOS_LAMBDA(const size_type i) {
          target_search_view(i).box = target_boxes_dev(i);
          target_search_view(i).identProc = ident_proc_t(target_identities_dev(i), my_rank);
        });

    Kokkos::parallel_for(
        "mundy_stk_nl_build_source_search", Kokkos::RangePolicy<exec_space>(0, num_sources),
        KOKKOS_LAMBDA(const size_type i) {
          source_search_view(i).box = source_boxes_dev(i);
          source_search_view(i).identProc = ident_proc_t(source_identities_dev(i), my_rank);
        });
    stk_prof_mark(stk_build_last_timings.phase_a_ms);

    // ===========================================================================
    // Phase B — Build target key-to-ordinal map (device)
    // ===========================================================================
    //
    // All domain (target) results reference locally owned targets, so this map
    // can be populated before the search.  The CSR count and fill passes use it
    // to skip symmetric copies where the domain entity belongs to a remote rank.
    //
    // 2x capacity keeps the load factor ≈50%, so insert failures are negligible for any well-distributed
    // collection of entity keys.

    key_map_t target_key_to_ordinal(2 * num_targets);

    Kokkos::parallel_for(
        "mundy_stk_nl_build_target_map", Kokkos::RangePolicy<exec_space>(0, num_targets),
        KOKKOS_LAMBDA(const size_type i) {
          const auto result = target_key_to_ordinal.insert(target_identities_dev(i), i);
          MUNDY_THROW_ASSERT(!result.failed(), std::runtime_error,
                             "mundy_stk_nl: target key-to-ordinal map insert failed; increase map capacity.");
        });
    Kokkos::fence();
    MUNDY_THROW_ASSERT(!target_key_to_ordinal.failed_insert(), std::runtime_error,
                       "mundy_stk_nl: target key-to-ordinal map has failed inserts after fence.");
    stk_prof_mark(stk_build_last_timings.phase_b_ms);

    // ===========================================================================
    // Phase C — Distributed coarse search (device + MPI)
    // ===========================================================================
    //
    // MORTON_LBVH is the only STK search method that accepts Kokkos::View I/O.
    // KDTREE forces host-side results regardless of MemorySpace, defeating the
    // device-side CSR passes.
    //
    // `enforceSearchResultSymmetry = true`: for every cross-process pair
    // (target on rank A, source on rank B), BOTH ranks receive the pair.  This
    // is structurally required for the cooperative ghosting protocol in Phase D:
    // rank B must see the pair to know it should ghost its source entity to rank A.
    //
    // `autoSwapDomainAndRange = false`: the caller has sized the target and source
    // views independently; auto-swapping would invert the domain/range semantics.

    results_view_t search_results_view;  // Resized by coarse_search.
    stk::search::coarse_search(target_search_view, source_search_view, stk::search::MORTON_LBVH, bulk_data.parallel(),
                               search_results_view, exec_sp,
                               /* enforceSearchResultSymmetry = */ true,
                               /* autoSwapDomainAndRange = */ false);

    const size_type num_results = search_results_view.extent(0);
    stk_prof_mark(stk_build_last_timings.phase_c_ms);

    // ===========================================================================
    // Phase D — Ghosting coordination (host)
    // ===========================================================================
    //
    // STK BulkData modification APIs are host-only.  Mirror results to host, then
    // apply the cooperative ghosting rule:
    //
    //   If I own the *range* (source) entity and the *domain* (target) is on a
    //   different rank, ghost my source to the target's rank.
    //
    // The symmetric result copies (where the domain belongs to a remote rank)
    // appear in `host_results` but produce no action here; they are filtered
    // by the target-map lookup in the device CSR passes (miss -> skip).
    //
    // Single-process short-circuit: no cross-process pairs exist on one rank;
    // the modification cycle is skipped entirely.
    //
    // `const_cast` rationale: `build()` takes `const BulkData&` for interface
    // uniformity with ArborX builds that make no mesh modifications.  The STK
    // build's ghosting side effect is documented in the function header.  The
    // caller's BulkData object is non-const; the cast is safe.

    entity_view_t extended_source_entities;
    size_type num_extended_sources = 0;

    if (single_rank) {
      // No cross-process ghosting can occur on one rank.  The extended source
      // list is exactly the generated source list, preserving ordinals.
      extended_source_entities = source_entities_dev;
      num_extended_sources = num_sources;
      stk_prof_mark(stk_build_last_timings.phase_d_ms);
    } else {
      auto host_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search_results_view);

      std::vector<stk::mesh::EntityProc> entities_to_ghost;
      for (size_type r = 0; r < num_results; ++r) {
        const auto& res = host_results(r);
        const int source_proc = res.rangeIdentProc.proc();
        const int target_proc = res.domainIdentProc.proc();
        if (source_proc != my_rank || target_proc == my_rank) continue;
        // I own this source; its paired target lives on `target_proc`.
        // Ghost the source to `target_proc` so that rank can form the neighbor pair.
        const stk::mesh::Entity source_ent = bulk_data.get_entity(res.rangeIdentProc.id());
        MUNDY_THROW_ASSERT(bulk_data.is_valid(source_ent), std::runtime_error,
                           "mundy_stk_nl: owned source entity not found in bulk_data during ghosting.");
        entities_to_ghost.emplace_back(source_ent, target_proc);
      }

      auto& mutable_bulk = const_cast<stk::mesh::BulkData&>(bulk_data);
      mutable_bulk.modification_begin();
      // The ghosting group persists on bulk_data after build() returns.  Callers
      // communicate field data they need via this group.
      stk::mesh::Ghosting& ghosting = mutable_bulk.create_ghosting("MUNDY_STK_SEARCH_NL_GHOSTING");
      mutable_bulk.change_ghosting(ghosting, entities_to_ghost);
      mutable_bulk.modification_end();

      // Communicate the source component's geometry field to the newly-ghosted source entities so they carry valid
      // geometry on this rank.  The build knows the source field because the input is a component; this removes the
      // old caller burden of hand-communicating geometry.  Targets are never ghosted, so only the source field is
      // sent.  The ghosting group persists for any additional fields the caller needs.
      const std::vector<const stk::mesh::FieldBase*> ghosted_geometry_fields{&source_input.component().field()};
      stk::mesh::communicate_field_data(ghosting, ghosted_geometry_fields);
      stk_prof_mark(stk_build_last_timings.phase_d_ms);

      // ===========================================================================
      // Phase E — Build extended source entity view (host -> device)
      // ===========================================================================
      //
      // After modification_end(), newly-ghosted source entities are locally
      // accessible.  Concatenate owned sources (ordinals 0..num_sources-1,
      // unchanged) with any additional ghosted sources discovered through the search
      // (ordinals num_sources..num_extended-1).
      //
      // `std::unordered_set` gives O(1) membership tests for deduplication.

      // Mirror owned source entities to host for key lookup.
      auto host_owned_sources = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source_entities_dev);

      // Build a set of owned source keys.
      std::unordered_set<stk::mesh::EntityKey, std::hash<stk::mesh::EntityKey>> owned_source_keys;
      owned_source_keys.reserve(num_sources);
      for (size_type i = 0; i < num_sources; ++i) {
        owned_source_keys.insert(bulk_data.entity_key(host_owned_sources(i)));
      }

      // Start the extended list with all owned sources in original order.
      std::vector<stk::mesh::Entity> extended_src_vec(host_owned_sources.data(), host_owned_sources.data() + num_sources);

      // Append ghosted sources not yet in the owned set.
      std::unordered_set<stk::mesh::EntityKey, std::hash<stk::mesh::EntityKey>> newly_added_keys;
      for (size_type r = 0; r < num_results; ++r) {
        const auto& res = host_results(r);
        if (res.domainIdentProc.proc() != my_rank) continue;  // Not our target row.
        const stk::mesh::EntityKey src_key = res.rangeIdentProc.id();
        if (owned_source_keys.count(src_key)) continue;          // Already owned.
        if (!newly_added_keys.insert(src_key).second) continue;  // Already appended.
        const stk::mesh::Entity src_ent = bulk_data.get_entity(src_key);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(src_ent), std::runtime_error,
                           "mundy_stk_nl: ghosted source entity is invalid after modification_end().");
        extended_src_vec.push_back(src_ent);
      }

      num_extended_sources = static_cast<size_type>(extended_src_vec.size());

      // Transfer to device.
      extended_source_entities = entity_view_t("mundy_stk_nl_ext_src", num_extended_sources);
      {
        auto host_ext = Kokkos::create_mirror_view(extended_source_entities);
        for (size_type i = 0; i < num_extended_sources; ++i) host_ext(i) = extended_src_vec[i];
        Kokkos::deep_copy(extended_source_entities, host_ext);
      }
      stk_prof_mark(stk_build_last_timings.phase_e_ms);
    }
    if (single_rank) {
      // Phase E is also a no-op on one rank: there are no remote sources to
      // append, and `extended_source_entities` already aliases the source input.
      stk_prof_mark(stk_build_last_timings.phase_e_ms);
    }

    // ===========================================================================
    // Phase F — Build source key-to-ordinal map (device)
    // ===========================================================================
    //
    // Built from the extended source view (owned + ghosted).  The NgpMesh must
    // be refreshed from the host mesh after modification_end() so that newly-
    // ghosted entities have valid fast mesh indices on device.

    auto updated_ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);

    key_map_t source_key_to_ordinal(2 * num_extended_sources);
    Kokkos::parallel_for(
        "mundy_stk_nl_build_source_map", Kokkos::RangePolicy<exec_space>(0, num_extended_sources),
        KOKKOS_LAMBDA(const size_type i) {
          const auto result = source_key_to_ordinal.insert(updated_ngp_mesh.entity_key(extended_source_entities(i)), i);
          MUNDY_THROW_ASSERT(!result.failed(), std::runtime_error,
                             "mundy_stk_nl: source key-to-ordinal map insert failed; increase map capacity.");
        });
    Kokkos::fence();
    MUNDY_THROW_ASSERT(!source_key_to_ordinal.failed_insert(), std::runtime_error,
                       "mundy_stk_nl: source key-to-ordinal map has failed inserts after fence.");
    stk_prof_mark(stk_build_last_timings.phase_f_ms);

    // ===========================================================================
    // Phase G0 — Precompute valid target/source ordinal pairs (device)
    // ===========================================================================
    //
    // For each search result, apply the full neighbor-list filter exactly once and
    // materialize the surviving dense ordinals.  Count and fill then read these
    // flat views without repeating the EntityKey map probes or excluder call.
    //
    // Filter steps:
    //   1. **Target lookup**: a map miss means the domain entity belongs to a
    //      remote rank (symmetric copy); mark invalid — not our list entry.
    //   2. **Source lookup**: must succeed — Phase E validated that every source for
    //      a locally-owned target is accessible and Phase F mapped all of those
    //      into `source_key_to_ordinal`.  A miss here means something went wrong
    //      (NgpMesh key mismatch, ghosting failure, etc.).
    //   3. **Excluder**: the builder's prepared excluder chain (e.g., ExcludeSelfInteraction).

    // Prepare the excluder chain immediately before the kernel that evaluates it, so its device geometry and
    // NgpMesh reflect the current mesh state. setup_broad/narrow_excluder return a prepared, device-capturable
    // copy (excluder.setup(...) already called); NoExcluder compiles away when there is no narrow phase.
    const auto excluder = builder.setup_broad_excluder(bulk_data);
    [[maybe_unused]] const auto narrow_excluder = [&]() {
      if constexpr (Builder::has_narrow_phase)
        return builder.setup_narrow_excluder(bulk_data);
      else
        return typename Builder::narrow_excluder_type{};
    }();

    constexpr size_type k_invalid_ordinal = std::numeric_limits<size_type>::max();
    Kokkos::View<size_type*, MemorySpace> precomputed_target_ordinals("mundy_stk_nl_precomp_trg", num_results);
    Kokkos::View<size_type*, MemorySpace> precomputed_source_ordinals("mundy_stk_nl_precomp_src", num_results);

    Kokkos::parallel_for(
        "mundy_stk_nl_precompute_ordinals", Kokkos::RangePolicy<exec_space>(0, num_results),
        KOKKOS_LAMBDA(const size_type r) {
          precomputed_target_ordinals(r) = k_invalid_ordinal;

          const intersection_t& result = search_results_view(r);

          // Step 1: target lookup — skip symmetric copies with a remote domain.
          const auto trg_slot = target_key_to_ordinal.find(result.domainIdentProc.id());
          if (!target_key_to_ordinal.valid_at(trg_slot)) return;
          const size_type trg_ord = target_key_to_ordinal.value_at(trg_slot);

          // Step 2: source lookup — must succeed after Phase E/F validation.
          const auto src_slot = source_key_to_ordinal.find(result.rangeIdentProc.id());
          MUNDY_THROW_ASSERT(source_key_to_ordinal.valid_at(src_slot), std::runtime_error,
                             "mundy_stk_nl: source entity missing from ordinal map after Phase E validation — "
                             "possible NgpMesh key mismatch or ghosting failure.");
          const size_type src_ord = source_key_to_ordinal.value_at(src_slot);

          // Step 3: broad-phase excluder, then narrow-phase excluder.
          // Both are applied here so the final precomputed ordinals reflect all
          // filtering in one pass — no post-CRS compaction needed for STK.
          const stk::mesh::Entity trg_ent = target_entities_dev(trg_ord);
          const stk::mesh::Entity src_ent = extended_source_entities(src_ord);
          const NeighborSearchCandidate<size_type> candidate(trg_ord, src_ord, trg_ent, src_ent);
          if (excluder(candidate)) return;
          // odr-use narrow_excluder so it is captured outside the constexpr-if below (nvcc rejects first-capture there).
          static_cast<void>(narrow_excluder);
          if constexpr (Builder::has_narrow_phase) {
            if (narrow_excluder(candidate)) return;
          }

          precomputed_target_ordinals(r) = trg_ord;
          precomputed_source_ordinals(r) = src_ord;
        });
    stk_prof_mark(stk_build_last_timings.phase_g0_ms);

    // ===========================================================================
    // Phase G — Count pass (device)
    // ===========================================================================
    //
    // Each valid precomputed ordinal pair contributes one neighbor to its target.
    // Invalid entries are remote symmetric copies or pairs rejected by the excluder.

    Kokkos::View<size_type*, MemorySpace> per_target_count("mundy_stk_nl_count", num_targets);

    Kokkos::parallel_for(
        "mundy_stk_nl_count", Kokkos::RangePolicy<exec_space>(0, num_results), KOKKOS_LAMBDA(const size_type r) {
          const size_type trg_ord = precomputed_target_ordinals(r);
          if (trg_ord == k_invalid_ordinal) return;
          Kokkos::atomic_fetch_add(&per_target_count(trg_ord), size_type(1));
        });
    stk_prof_mark(stk_build_last_timings.phase_g_ms);

    // ===========================================================================
    // Phase H — Prefix scan (device)
    // ===========================================================================
    //
    // `offsets[i] = sum(per_target_count[0..i))` for `i in [0, num_targets+1)`.
    // `offsets[num_targets] = total_pairs`.
    //
    // `write_positions` is a mutable copy of `offsets[0..num_targets)` used by
    // the fill pass for atomic slot allocation; `offsets` itself remains intact
    // for the list constructor.

    Kokkos::View<size_type*, MemorySpace> offsets("mundy_stk_nl_offsets", num_targets + 1);
    Kokkos::parallel_scan(
        "mundy_stk_nl_scan", Kokkos::RangePolicy<exec_space>(0, num_targets + 1),
        KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final_pass) {
          if (final_pass) offsets(i) = update;
          if (i < num_targets) update += per_target_count(i);
        });

    // Scalar copy to host for output-view allocation.
    size_type total_pairs = 0;
    Kokkos::deep_copy(total_pairs, Kokkos::subview(offsets, num_targets));

    Kokkos::View<size_type*, MemorySpace> source_indices("mundy_stk_nl_src_idx", total_pairs);

    // write_positions initialized from offsets[0..num_targets); modified in-place by fill pass.
    Kokkos::View<size_type*, MemorySpace> write_positions("mundy_stk_nl_wpos", num_targets);
    Kokkos::deep_copy(write_positions, Kokkos::subview(offsets, Kokkos::make_pair(size_type(0), num_targets)));
    stk_prof_mark(stk_build_last_timings.phase_h_ms);

    // ===========================================================================
    // Phase I — Fill pass (device)
    // ===========================================================================
    //
    // For each valid precomputed pair, `atomic_fetch_add` on
    // `write_positions[trg_ord]` allocates a unique slot; the source ordinal is
    // written into that slot.

    Kokkos::parallel_for(
        "mundy_stk_nl_fill", Kokkos::RangePolicy<exec_space>(0, num_results), KOKKOS_LAMBDA(const size_type r) {
          const size_type trg_ord = precomputed_target_ordinals(r);
          if (trg_ord == k_invalid_ordinal) return;
          const size_type pos = Kokkos::atomic_fetch_add(&write_positions(trg_ord), size_type(1));
          source_indices(pos) = precomputed_source_ordinals(r);
        });
    stk_prof_mark(stk_build_last_timings.phase_i_ms);

    // ===========================================================================
    // Phase J — Optional per-row insertion sort (device)
    // ===========================================================================
    //
    // Sorts each target's neighbor row by ascending source ordinal.  Insertion sort
    // is used because row sizes are small (~10–20 entries at typical densities) and
    // it avoids auxiliary allocation.  Sorted rows improve spatial locality when
    // kernels access per-source data (positions, radii, …) for multiple targets
    // that share neighbors.  Default is `false` (BVH traversal order preserved).

    if (builder.sort_neighbors()) {
      Kokkos::parallel_for(
          "mundy_stk_nl_sort", Kokkos::RangePolicy<exec_space>(0, num_targets), KOKKOS_LAMBDA(const size_type t) {
            const size_type beg = offsets(t);
            const size_type end = offsets(t + 1);
            for (size_type i = beg + 1; i < end; ++i) {
              const size_type key = source_indices(i);
              size_type j = i;
              while (j > beg && source_indices(j - 1) > key) {
                source_indices(j) = source_indices(j - 1);
                --j;
              }
              source_indices(j) = key;
            }
          });
    }
    stk_prof_mark(stk_build_last_timings.phase_j_ms);

    // ===========================================================================
    // Phase K — Construct and return the list
    // ===========================================================================
    //
    // `target_entities_dev` is the generated owned-target view (unchanged through the build).
    // `extended_source_entities` is the extended view (owned + ghosted sources).
    // `source_indices` and `offsets` are exclusively owned by the returned list.
    //
    // Kokkos::View is reference-counted; the list shares ownership of the entity
    // views with any other holders, so no copies are made.

    auto result = list_type(builder.target_selector(), builder.source_selector(), target_entities_dev,
                            extended_source_entities, source_indices, offsets);
    stk_prof_mark(stk_build_last_timings.phase_k_ms);
    return result;
  }
  //@}
};

/// \struct NeighborListBuildTraits<PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>>
/// \brief Build traits for `PeriodicSTKSearchNeighborList`: periodic STK coarse-search compressed storage.
///
/// The build generates periodic image boxes from component inputs, runs distributed `stk::search::coarse_search` over
/// the images (each image's identity carries its owner key and per-object image shift), ghosts remote source owners to
/// the target's rank, collapses results to owner ordinals, and stores per-object image shifts for each retained pair.
template <typename MemorySpace, typename ImageShiftScalar>
struct NeighborListBuildTraits<PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>> {
  //! \name Aliases
  //@{

  using list_type = PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `PeriodicSTKSearchNeighborList` (none required).
  struct args_type {};
  //@}

  //! \name Build
  //@{

  /// \brief Build the list from a complete builder and BulkData.
  ///
  /// \param builder [in] Complete builder carrying exec space, periodic component inputs, and the excluder chain.
  /// \param bulk_data [in] STK bulk data.
  /// \param args [in] Build-specific parameters (unused).
  template <typename Builder>
    requires PeriodicAABBSearchInputType<typename Builder::target_input_type> &&
             PeriodicAABBSearchInputType<typename Builder::source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& /*args*/) {
    using list_type = PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>;
    using exec_space = typename Builder::execution_space;
    using size_type = typename list_type::size_type;
    using entity_view_t = typename list_type::entity_view_t;
    using image_shift_type = typename list_type::image_shift_type;
    using image_shift_view_t = typename list_type::image_shift_view_t;
    using source_index_view_t = typename list_type::source_index_view_t;
    using offset_view_t = typename list_type::offset_view_t;

    using ident_t = impl::PeriodicImageIdentity<stk::mesh::EntityKey, ImageShiftScalar>;
    using ident_proc_t = stk::search::IdentProc<ident_t, int>;
    using box_type = stk::search::Box<float>;
    using box_ident_proc_t = stk::search::BoxIdentProc<box_type, ident_proc_t>;
    using intersection_t = stk::search::IdentProcIntersection<ident_proc_t, ident_proc_t>;
    using search_view_t = Kokkos::View<box_ident_proc_t*, MemorySpace>;
    using results_view_t = Kokkos::View<intersection_t*, MemorySpace>;
    using key_map_t = Kokkos::UnorderedMap<stk::mesh::EntityKey, size_type, MemorySpace>;

    // Inputs copied (non-const) so each component can be synced to device before reading.
    auto target_input = builder.target_input();
    auto source_input = builder.source_input();
    const auto exec_sp = builder.exec_space();
    const int my_rank = bulk_data.parallel_rank();
    const bool single_rank = bulk_data.parallel_size() == 1;

    // --- Generate backend-neutral periodic images (target: 1 wrapped image/owner + local bbox; source: <=3^d/owner,
    //     pruned), then pack each into STK search boxes whose identity is {owner key, image shift}. ---
    auto target_images = impl::make_periodic_target_images<ImageShiftScalar>(
        bulk_data, exec_sp, target_input.rank(), target_input.selector(), target_input.component(),
        target_input.periodic_metric());
    mundy::AABB<float> target_bbox = impl::periodic_images_bounding_box(exec_sp, target_images);

    // All-reduce the target bbox so source images that only reach a remote rank's targets survive local pruning.
    if (!single_rank) {
      float local_min[3] = {target_bbox.min_corner()[0], target_bbox.min_corner()[1], target_bbox.min_corner()[2]};
      float local_max[3] = {target_bbox.max_corner()[0], target_bbox.max_corner()[1], target_bbox.max_corner()[2]};
      float global_min[3];
      float global_max[3];
      MPI_Allreduce(local_min, global_min, 3, MPI_FLOAT, MPI_MIN, bulk_data.parallel());
      MPI_Allreduce(local_max, global_max, 3, MPI_FLOAT, MPI_MAX, bulk_data.parallel());
      target_bbox = mundy::AABB<float>{mundy::Point<float>{global_min[0], global_min[1], global_min[2]},
                                       mundy::Point<float>{global_max[0], global_max[1], global_max[2]}};
    }

    auto source_images = impl::make_periodic_source_images<ImageShiftScalar>(
        bulk_data, exec_sp, source_input.rank(), source_input.selector(), source_input.component(),
        source_input.periodic_metric(), target_bbox);

    // NgpMesh must be obtained before any mesh modification (Phase D ghosting); the pack projects each image's local
    // owner Entity to the global EntityKey carried by `coarse_search`.
    auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto target_boxes = impl::pack_periodic_stk_search_boxes(exec_sp, ngp_mesh, target_input.selector(), target_images);
    auto source_boxes = impl::pack_periodic_stk_search_boxes(exec_sp, ngp_mesh, source_input.selector(), source_images);

    const size_type num_target_owners = target_images.owner_entities.extent(0);
    const size_type num_target_images = target_boxes.size();
    const size_type num_source_owners = source_images.owner_entities.extent(0);
    const size_type num_source_images = source_boxes.size();

    // Raw views for device capture (SearchBoxes holds a Selector and so cannot itself be captured in a kernel).
    auto target_image_boxes = target_boxes.boxes();
    auto target_image_identities = target_boxes.identities();
    auto target_owner_entities = target_images.owner_entities;
    auto target_image_shifts = target_images.shifts;  // per target owner (1 image/owner)
    auto source_image_boxes = source_boxes.boxes();
    auto source_image_identities = source_boxes.identities();
    auto source_owner_entities = source_images.owner_entities;

    // --- Phase A: BoxIdentProc views over images; the per-image identity {owner key, image shift} is the search ident.
    // ---
    search_view_t target_search("mundy_stk_per_tgt_search", num_target_images);
    search_view_t source_search("mundy_stk_per_src_search", num_source_images);
    Kokkos::parallel_for(
        "mundy_stk_per_build_tgt_search", Kokkos::RangePolicy<exec_space>(0, num_target_images),
        KOKKOS_LAMBDA(const size_type i) {
          target_search(i).box = target_image_boxes(i);
          target_search(i).identProc = ident_proc_t(target_image_identities(i), my_rank);
        });
    Kokkos::parallel_for(
        "mundy_stk_per_build_src_search", Kokkos::RangePolicy<exec_space>(0, num_source_images),
        KOKKOS_LAMBDA(const size_type i) {
          source_search(i).box = source_image_boxes(i);
          source_search(i).identProc = ident_proc_t(source_image_identities(i), my_rank);
        });

    // --- Phase B: target OWNER key -> dense owner ordinal map. ---
    key_map_t target_owner_key_to_ordinal(2 * num_target_owners + 1);
    Kokkos::parallel_for(
        "mundy_stk_per_build_tgt_owner_map", Kokkos::RangePolicy<exec_space>(0, num_target_owners),
        KOKKOS_LAMBDA(const size_type o) {
          const auto r = target_owner_key_to_ordinal.insert(ngp_mesh.entity_key(target_owner_entities(o)), o);
          MUNDY_THROW_ASSERT(!r.failed(), std::runtime_error, "mundy_stk_per: target owner map insert failed.");
        });
    Kokkos::fence();

    // --- Phase C: distributed coarse search over images. ---
    results_view_t results;
    stk::search::coarse_search(target_search, source_search, stk::search::MORTON_LBVH, bulk_data.parallel(), results,
                               exec_sp, /*enforceSearchResultSymmetry=*/true, /*autoSwapDomainAndRange=*/false);
    const size_type num_results = results.extent(0);

    // --- Phase D/E: ghost source OWNERS to their paired target's rank; build the extended source owner set. ---
    entity_view_t extended_source_owners;
    size_type num_extended_source_owners = 0;
    if (single_rank) {
      extended_source_owners = source_owner_entities;
      num_extended_source_owners = num_source_owners;
    } else {
      auto host_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);

      // If I own a source image whose paired target is remote, ghost the source OWNER to the target's rank
      // (change_ghosting deduplicates repeated owner/proc pairs from multiple images).
      std::vector<stk::mesh::EntityProc> entities_to_ghost;
      for (size_type r = 0; r < num_results; ++r) {
        const auto& res = host_results(r);
        if (res.rangeIdentProc.proc() != my_rank || res.domainIdentProc.proc() == my_rank) continue;
        const stk::mesh::Entity owner = bulk_data.get_entity(res.rangeIdentProc.id().owner);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(owner), std::runtime_error,
                           "mundy_stk_per: owned source owner not found during ghosting.");
        entities_to_ghost.emplace_back(owner, res.domainIdentProc.proc());
      }

      auto& mutable_bulk = const_cast<stk::mesh::BulkData&>(bulk_data);
      mutable_bulk.modification_begin();
      stk::mesh::Ghosting& ghosting = mutable_bulk.create_ghosting("MUNDY_STK_SEARCH_NL_GHOSTING");
      mutable_bulk.change_ghosting(ghosting, entities_to_ghost);
      mutable_bulk.modification_end();

      const std::vector<const stk::mesh::FieldBase*> ghosted_geometry_fields{&source_input.component().field()};
      stk::mesh::communicate_field_data(ghosting, ghosted_geometry_fields);

      // Extended source owners = owned source owners (ordinals unchanged) + ghosted source owners reached by a local
      // target.
      auto host_owned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, source_owner_entities);
      std::unordered_set<stk::mesh::EntityKey, std::hash<stk::mesh::EntityKey>> owned_keys;
      owned_keys.reserve(num_source_owners);
      for (size_type i = 0; i < num_source_owners; ++i) owned_keys.insert(bulk_data.entity_key(host_owned(i)));

      std::vector<stk::mesh::Entity> ext(host_owned.data(), host_owned.data() + num_source_owners);
      std::unordered_set<stk::mesh::EntityKey, std::hash<stk::mesh::EntityKey>> added_keys;
      for (size_type r = 0; r < num_results; ++r) {
        const auto& res = host_results(r);
        if (res.domainIdentProc.proc() != my_rank) continue;  // not our target row
        const stk::mesh::EntityKey src_owner_key = res.rangeIdentProc.id().owner;
        if (owned_keys.count(src_owner_key)) continue;
        if (!added_keys.insert(src_owner_key).second) continue;
        const stk::mesh::Entity owner = bulk_data.get_entity(src_owner_key);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(owner), std::runtime_error,
                           "mundy_stk_per: ghosted source owner invalid after modification_end().");
        ext.push_back(owner);
      }
      num_extended_source_owners = static_cast<size_type>(ext.size());
      extended_source_owners = entity_view_t("mundy_stk_per_ext_src_owners", num_extended_source_owners);
      auto host_ext = Kokkos::create_mirror_view(extended_source_owners);
      for (size_type i = 0; i < num_extended_source_owners; ++i) host_ext(i) = ext[i];
      Kokkos::deep_copy(extended_source_owners, host_ext);
    }

    // --- Phase F: extended source OWNER key -> ordinal map (refreshed NgpMesh includes ghosts). ---
    auto updated_ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    key_map_t source_owner_key_to_ordinal(2 * num_extended_source_owners + 1);
    Kokkos::parallel_for(
        "mundy_stk_per_build_src_owner_map", Kokkos::RangePolicy<exec_space>(0, num_extended_source_owners),
        KOKKOS_LAMBDA(const size_type i) {
          const auto r = source_owner_key_to_ordinal.insert(updated_ngp_mesh.entity_key(extended_source_owners(i)), i);
          MUNDY_THROW_ASSERT(!r.failed(), std::runtime_error, "mundy_stk_per: source owner map insert failed.");
        });
    Kokkos::fence();

    // --- Phase G0: precompute per-result (target owner ordinal, source owner ordinal, source image shift). ---
    // Prepare the excluder chain immediately before the kernel that evaluates it (current, post-ghost mesh state).
    const auto excluder = builder.setup_broad_excluder(bulk_data);
    [[maybe_unused]] const auto narrow_excluder = [&]() {
      if constexpr (Builder::has_narrow_phase)
        return builder.setup_narrow_excluder(bulk_data);
      else
        return typename Builder::narrow_excluder_type{};
    }();

    constexpr size_type k_invalid_ordinal = std::numeric_limits<size_type>::max();
    Kokkos::View<size_type*, MemorySpace> pre_target("mundy_stk_per_pre_trg", num_results);
    Kokkos::View<size_type*, MemorySpace> pre_source("mundy_stk_per_pre_src", num_results);
    Kokkos::View<image_shift_type*, MemorySpace> pre_source_shift("mundy_stk_per_pre_src_shift", num_results);
    Kokkos::parallel_for(
        "mundy_stk_per_precompute", Kokkos::RangePolicy<exec_space>(0, num_results), KOKKOS_LAMBDA(const size_type r) {
          pre_target(r) = k_invalid_ordinal;
          const intersection_t& res = results(r);

          // Target lookup: a miss is a symmetric copy whose target lives on a remote rank — not our list entry.
          const auto tslot = target_owner_key_to_ordinal.find(res.domainIdentProc.id().owner);
          if (!target_owner_key_to_ordinal.valid_at(tslot)) return;
          const size_type t_ord = target_owner_key_to_ordinal.value_at(tslot);

          // Source lookup: must succeed (Phase D/E ghosted every source owner reached by a local target).
          const auto sslot = source_owner_key_to_ordinal.find(res.rangeIdentProc.id().owner);
          MUNDY_THROW_ASSERT(source_owner_key_to_ordinal.valid_at(sslot), std::runtime_error,
                             "mundy_stk_per: source owner missing from ordinal map — ghosting failure.");
          const size_type s_ord = source_owner_key_to_ordinal.value_at(sslot);

          const image_shift_type target_shift = res.domainIdentProc.id().shift;
          const image_shift_type source_shift = res.rangeIdentProc.id().shift;
          const PeriodicNeighborSearchCandidate<image_shift_type, size_type> candidate(
              t_ord, s_ord, target_owner_entities(t_ord), extended_source_owners(s_ord), target_shift, source_shift);
          if (excluder(candidate)) return;
          // odr-use narrow_excluder so it is captured outside the constexpr-if below (nvcc rejects first-capture there).
          static_cast<void>(narrow_excluder);
          if constexpr (Builder::has_narrow_phase) {
            if (narrow_excluder(candidate)) return;
          }

          pre_target(r) = t_ord;
          pre_source(r) = s_ord;
          pre_source_shift(r) = source_shift;
        });

    // --- Phase G: count surviving pairs per target owner. ---
    Kokkos::View<size_type*, MemorySpace> per_target_count("mundy_stk_per_count", num_target_owners);
    Kokkos::parallel_for(
        "mundy_stk_per_count", Kokkos::RangePolicy<exec_space>(0, num_results), KOKKOS_LAMBDA(const size_type r) {
          const size_type t_ord = pre_target(r);
          if (t_ord == k_invalid_ordinal) return;
          Kokkos::atomic_fetch_add(&per_target_count(t_ord), size_type(1));
        });

    // --- Phase H: prefix scan -> CSR offsets; allocate per-pair output. ---
    offset_view_t offsets("mundy_stk_per_offsets", num_target_owners + 1);
    Kokkos::parallel_scan(
        "mundy_stk_per_scan", Kokkos::RangePolicy<exec_space>(0, num_target_owners + 1),
        KOKKOS_LAMBDA(const size_type i, size_type& update, const bool final_pass) {
          if (final_pass) offsets(i) = update;
          if (i < num_target_owners) update += per_target_count(i);
        });
    size_type total_pairs = 0;
    Kokkos::deep_copy(total_pairs, Kokkos::subview(offsets, num_target_owners));

    source_index_view_t out_source_indices("mundy_stk_per_src_idx", total_pairs);
    image_shift_view_t out_source_shifts("mundy_stk_per_src_shift", total_pairs);
    Kokkos::View<size_type*, MemorySpace> write_positions("mundy_stk_per_wpos", num_target_owners);
    Kokkos::deep_copy(write_positions, Kokkos::subview(offsets, Kokkos::make_pair(size_type(0), num_target_owners)));

    // --- Phase I: fill per-pair source ordinal + source image shift. ---
    Kokkos::parallel_for(
        "mundy_stk_per_fill", Kokkos::RangePolicy<exec_space>(0, num_results), KOKKOS_LAMBDA(const size_type r) {
          const size_type t_ord = pre_target(r);
          if (t_ord == k_invalid_ordinal) return;
          const size_type pos = Kokkos::atomic_fetch_add(&write_positions(t_ord), size_type(1));
          out_source_indices(pos) = pre_source(r);
          out_source_shifts(pos) = pre_source_shift(r);
        });

    // --- Phase J: optional per-row sort by source owner ordinal (carry the matching source shift). ---
    if (builder.sort_neighbors()) {
      Kokkos::parallel_for(
          "mundy_stk_per_sort", Kokkos::RangePolicy<exec_space>(0, num_target_owners), KOKKOS_LAMBDA(const size_type t) {
            const size_type beg = offsets(t);
            const size_type end = offsets(t + 1);
            for (size_type i = beg + 1; i < end; ++i) {
              const size_type key = out_source_indices(i);
              const image_shift_type key_shift = out_source_shifts(i);
              size_type j = i;
              while (j > beg && out_source_indices(j - 1) > key) {
                out_source_indices(j) = out_source_indices(j - 1);
                out_source_shifts(j) = out_source_shifts(j - 1);
                --j;
              }
              out_source_indices(j) = key;
              out_source_shifts(j) = key_shift;
            }
          });
    }

    // --- Phase K: construct. target_image_shifts is per target owner (1 image/owner). ---
    return list_type(builder.target_selector(), builder.source_selector(), target_owner_entities, extended_source_owners,
                     target_image_shifts, out_source_indices, out_source_shifts, offsets);
  }
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_STKSEARCHNEIGHBORLIST_HPP_
