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
#include <concepts>   // for std::same_as
#include <cstddef>    // for size_t
#include <stdexcept>  // for std::invalid_argument, std::out_of_range

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>                    // for mundy::Vector3
#include <mundy_search/NeighborListBuildTraits.hpp>  // for NeighborListBuildTraits
#include <mundy_search/impl/STKSearchBoxes.hpp>      // for impl::STKSearchBoxesT, impl::PeriodicSTKSearchBoxesT
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

/// \class STKSearchNeighborList
/// \brief STK coarse-search neighbor list mapped into Mundy's common access surface.
///
/// This implementation is intended to consume STK coarse-search candidate pairs and materialize the same compressed
/// target-to-source storage shape as `ArborX1dNeighborList`.
/// \tparam MemorySpace Kokkos memory space for owned views.
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
    return target_selector_;
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
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
  stk::mesh::Selector target_selector_;
  //! Selector defining the source chunk.
  stk::mesh::Selector source_selector_;
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
/// \brief STK coarse-search neighbor list with compressed owner-pair storage and relative periodic image shifts.
///
/// This implementation is intended to consume periodic STK coarse-search image pairs, collapse them to owner ordinals,
/// and retain one relative image shift for each stored owner pair.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
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
  /// \param source_owner_indices [in] Dense source owner ordinal for every stored pair.
  /// \param relative_image_shifts [in] Source image shift minus target image shift for every stored pair.
  /// \param offsets [in] Target owner offsets into `source_owner_indices`; extent must be `num_targets + 1`.
  PeriodicSTKSearchNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                                const entity_view_t& target_entities, const entity_view_t& source_entities,
                                const source_index_view_t& source_owner_indices,
                                const image_shift_view_t& relative_image_shifts, const offset_view_t& offsets)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: offsets extent must be num_targets + 1.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == relative_image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: source_owner_indices and relative_image_shifts must have the "
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
    return target_selector_;
  }

  /// \brief Get the selector defining the source owner chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
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

  /// \brief Get the source image shift relative to the target image shift for a stored pair.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    return relative_image_shifts_(pair_index(target_index, neighbor_ordinal));
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

  /// \brief Get the raw flattened relative-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t relative_image_shifts() const noexcept {
    return relative_image_shifts_;
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
  stk::mesh::Selector target_selector_;
  //! Selector defining the source owner chunk.
  stk::mesh::Selector source_selector_;
  //! Target owner entities indexed by dense target owner ordinal.
  entity_view_t target_entities_;
  //! Source owner entities indexed by dense source owner ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source owner ordinals for each stored periodic pair.
  source_index_view_t source_owner_indices_;
  //! Flattened source-image shift minus target-image shift for each stored periodic pair.
  image_shift_view_t relative_image_shifts_;
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

// -----------------------------------------------------------------------------
// NeighborListBuildTraits specializations for STK-search-backed list types
// -----------------------------------------------------------------------------

/// \struct NeighborListBuildTraits<STKSearchNeighborList<MemorySpace>>
/// \brief Build traits for `STKSearchNeighborList`: non-periodic STK coarse-search compressed storage.
///
/// The build runs `stk::search::coarse_search`, applies the builder's excluder chain, groups results by target,
/// and returns compressed target-to-source storage. Declaration only for this design pass.
/// \tparam MemorySpace Kokkos memory space for the returned list.
template <typename MemorySpace>
struct NeighborListBuildTraits<STKSearchNeighborList<MemorySpace>> {
  //! \name Aliases
  //@{

  using list_type = STKSearchNeighborList<MemorySpace>;
  using target_input_type = impl::STKSearchBoxesT<MemorySpace>;
  using source_input_type = impl::STKSearchBoxesT<MemorySpace>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `STKSearchNeighborList` (none required).
  struct args_type {};
  //@}

  //! \name Build
  //@{

  /// \brief Build the list from a complete builder and BulkData.
  /// \tparam Builder Complete `NeighborListBuilder` type carrying exec space, inputs, and excluder.
  /// \param builder [in] Complete builder. Target and source inputs must be `STKSearchBoxesT<MemorySpace>`.
  /// \param bulk_data [in] STK bulk data for excluder setup.
  /// \param args [in] Build-specific parameters (unused).
  template <typename Builder>
    requires std::same_as<typename Builder::target_input_type, target_input_type> &&
             std::same_as<typename Builder::source_input_type, source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& args);
  //@}
};

/// \struct NeighborListBuildTraits<PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>>
/// \brief Build traits for `PeriodicSTKSearchNeighborList`: periodic STK coarse-search compressed storage.
///
/// The build runs STK coarse search on image boxes, collapses results to owner ordinals, groups them by target
/// owner, and stores one relative image shift for each retained owner pair. Declaration only for this design pass.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct NeighborListBuildTraits<PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>> {
  //! \name Aliases
  //@{

  using list_type = PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar>;
  using target_input_type = impl::PeriodicSTKSearchBoxesT<MemorySpace, ImageShiftScalar>;
  using source_input_type = impl::PeriodicSTKSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `PeriodicSTKSearchNeighborList` (none required).
  struct args_type {};
  //@}

  //! \name Build
  //@{

  /// \brief Build the list from a complete builder and BulkData.
  /// \tparam Builder Complete `NeighborListBuilder` type carrying exec space, inputs, and excluder.
  /// \param builder [in] Complete builder. Target and source inputs must be
  ///                     `PeriodicSTKSearchBoxesT<MemorySpace, ImageShiftScalar>`.
  /// \param bulk_data [in] STK bulk data for excluder setup.
  /// \param args [in] Build-specific parameters (unused).
  template <typename Builder>
    requires std::same_as<typename Builder::target_input_type, target_input_type> &&
             std::same_as<typename Builder::source_input_type, source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& args);
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_STKSEARCHNEIGHBORLIST_HPP_
