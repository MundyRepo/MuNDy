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

#ifndef MUNDY_SEARCH_ARBORX2DNEIGHBORLIST_HPP_
#define MUNDY_SEARCH_ARBORX2DNEIGHBORLIST_HPP_

/// \file ArborX2dNeighborList.hpp
/// \brief ArborX dense 2D neighbor-list types and their NeighborListBuildTraits specializations.

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
#include <mundy_search/impl/ArborXSearchBoxes.hpp>   // for impl::ArborXSearchBoxesT, impl::PeriodicArborXSearchBoxesT
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

/// \class ArborX2dNeighborList
/// \brief ArborX neighbor list with Cabana-style dense 2D per-target storage.
///
/// This implementation stores target entities, source entities, per-target neighbor counts, and dense rows of source
/// ordinals. It does not expose compact pair ids through the generic payload.
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename MemorySpace = stk::ngp::MemSpace>
class ArborX2dNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using size_type = size_t;
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborX2dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList(const ArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList(ArborX2dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList& operator=(const ArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList& operator=(ArborX2dNeighborList&&) = default;

  /// \brief Construct a list from already-built dense storage.
  /// \param target_selector [in] Selector defining the target chunk used during the build.
  /// \param source_selector [in] Selector defining the source chunk used during the build.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param neighbor_counts [in] Number of valid entries in each target row.
  /// \param source_indices [in] Dense target-by-neighbor source ordinal view.
  ArborX2dNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                       const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const count_view_t& neighbor_counts, const source_index_view_t& source_indices)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        neighbor_counts_(neighbor_counts),
        source_indices_(source_indices),
        total_pairs_(0) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_indices_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: source_indices row extent must equal num_targets.");
    size_type total = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, neighbor_counts_.extent(0)),
        KOKKOS_LAMBDA(size_type i, size_type & partial_sum) { partial_sum += neighbor_counts_(i); }, total);
    total_pairs_ = total;
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
    return total_pairs_;
  }

  /// \brief Get the allocated row width for each target.
  KOKKOS_INLINE_FUNCTION
  size_type max_neighbors_per_target() const noexcept {
    return source_indices_.extent(1);
  }

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX2dNeighborList::num_neighbors target index out of range.");
    return neighbor_counts_(target_index);
  }

  /// \brief Get the source ordinal for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX2dNeighborList::source_index neighbor ordinal out of range.");
    return source_indices_(target_index, neighbor_ordinal);
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
                       "ArborX2dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "ArborX2dNeighborList::source_entity source index out of range.");
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

  /// \brief Get the raw per-target neighbor count view.
  KOKKOS_INLINE_FUNCTION
  count_view_t neighbor_counts() const noexcept {
    return neighbor_counts_;
  }

  /// \brief Get the raw dense source-index view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_indices() const noexcept {
    return source_indices_;
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
  //! Number of valid entries in each dense target row.
  count_view_t neighbor_counts_;
  //! Dense per-target source ordinals; extent is `num_targets() x max_neighbors_per_target`.
  source_index_view_t source_indices_;
  //! Total number of stored neighbor pairs, computed once at construction.
  size_type total_pairs_;
  //@}
};

/// \class PeriodicArborX2dNeighborList
/// \brief ArborX dense 2D neighbor list whose stored entries carry relative periodic image shifts.
///
/// This layout stores a fixed-width row of source owner ordinals and relative shifts for each target owner. It is
/// useful when downstream kernels prefer dense per-target neighbor rows over compressed storage.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicArborX2dNeighborList {
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
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type**, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborX2dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList(const PeriodicArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList(PeriodicArborX2dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList& operator=(const PeriodicArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList& operator=(PeriodicArborX2dNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built dense storage.
  /// \param target_selector [in] Selector defining the target owner chunk used during the build.
  /// \param source_selector [in] Selector defining the source owner chunk used during the build.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param neighbor_counts [in] Number of valid entries in each target owner row.
  /// \param source_owner_indices [in] Dense source owner ordinals in target-by-neighbor rows.
  /// \param relative_image_shifts [in] Relative image shifts in target-by-neighbor rows.
  PeriodicArborX2dNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                               const entity_view_t& target_entities, const entity_view_t& source_entities,
                               const count_view_t& neighbor_counts, const source_index_view_t& source_owner_indices,
                               const image_shift_view_t& relative_image_shifts)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        neighbor_counts_(neighbor_counts),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "PeriodicArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "PeriodicArborX2dNeighborList: source_owner_indices row extent must equal num_targets.");
    MUNDY_THROW_ASSERT(relative_image_shifts_.extent(0) == source_owner_indices_.extent(0) &&
                           relative_image_shifts_.extent(1) == source_owner_indices_.extent(1),
                       std::invalid_argument,
                       "PeriodicArborX2dNeighborList: relative_image_shifts extent must equal source_owner_indices "
                       "extent.");
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
  ///
  /// This is intentionally a linear scan for this first-pass dense layout. If callers need this frequently, store the
  /// total during construction instead of introducing a compact pair-id abstraction.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    size_type total_neighbors = 0;
    for (size_type target_index = 0; target_index < num_targets(); ++target_index) {
      total_neighbors += num_neighbors(target_index);
    }
    return total_neighbors;
  }

  /// \brief Get the allocated row width for each target owner.
  KOKKOS_INLINE_FUNCTION
  size_type max_neighbors_per_target() const noexcept {
    return source_owner_indices_.extent(1);
  }

  /// \brief Get the number of neighbors for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicArborX2dNeighborList::num_neighbors target index out of range.");
    return neighbor_counts_(target_index);
  }

  /// \brief Get the source owner ordinal for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicArborX2dNeighborList::source_index neighbor ordinal out of range.");
    return source_owner_indices_(target_index, neighbor_ordinal);
  }

  /// \brief Get the source image shift relative to the target image shift for a stored pair.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicArborX2dNeighborList::relative_image_shift neighbor ordinal out of range.");
    return relative_image_shifts_(target_index, neighbor_ordinal);
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
                       "PeriodicArborX2dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicArborX2dNeighborList::source_entity source index out of range.");
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

  /// \brief Get the raw per-target-owner neighbor count view.
  KOKKOS_INLINE_FUNCTION
  count_view_t neighbor_counts() const noexcept {
    return neighbor_counts_;
  }

  /// \brief Get the raw dense source-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_owner_indices() const noexcept {
    return source_owner_indices_;
  }

  /// \brief Get the raw dense relative-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t relative_image_shifts() const noexcept {
    return relative_image_shifts_;
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
  //! Number of valid entries in each dense target-owner row.
  count_view_t neighbor_counts_;
  //! Dense per-target source owner ordinals.
  source_index_view_t source_owner_indices_;
  //! Dense per-target source-image shift minus target-image shift values.
  image_shift_view_t relative_image_shifts_;
  //@}
};

// -----------------------------------------------------------------------------
// NeighborListBuildTraits specializations for ArborX dense 2D list types
// -----------------------------------------------------------------------------

/// \struct NeighborListBuildTraits<ArborX2dNeighborList<MemorySpace>>
/// \brief Build traits for `ArborX2dNeighborList`: non-periodic ArborX dense 2D storage.
///
/// The build runs ArborX's two-pass count/fill flow over non-periodic boxes, applies the builder's excluder chain,
/// and returns dense per-target source rows. Declaration only for this design pass; the definition runs ArborX.
/// \tparam MemorySpace Kokkos memory space for the returned list.
template <typename MemorySpace>
struct NeighborListBuildTraits<ArborX2dNeighborList<MemorySpace>> {
  //! \name Aliases
  //@{

  using list_type = ArborX2dNeighborList<MemorySpace>;
  using target_input_type = impl::ArborXSearchBoxesT<MemorySpace>;
  using source_input_type = impl::ArborXSearchBoxesT<MemorySpace>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `ArborX2dNeighborList`.
  struct args_type {
    int buffer_size = 0;  ///< Optional maximum-neighbor preallocation guess.
  };
  //@}

  //! \name Build
  //@{

  /// \brief Build the list from a complete builder and BulkData.
  /// \tparam Builder Complete `NeighborListBuilder` type carrying exec space, inputs, and excluder.
  /// \param builder [in] Complete builder. Target and source inputs must be `ArborXSearchBoxesT<MemorySpace>`.
  /// \param bulk_data [in] STK bulk data for excluder setup.
  /// \param args [in] Build-specific parameters.
  template <typename Builder>
    requires std::same_as<typename Builder::target_input_type, target_input_type> &&
             std::same_as<typename Builder::source_input_type, source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& args);
  //@}
};

/// \struct NeighborListBuildTraits<PeriodicArborX2dNeighborList<MemorySpace, ImageShiftScalar>>
/// \brief Build traits for `PeriodicArborX2dNeighborList`: periodic ArborX dense 2D storage.
///
/// The build runs ArborX's periodic count/fill flow, collapses image matches to owner ordinals, and stores a
/// relative image shift in the same dense slot as each source owner ordinal. Declaration only for this design pass.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct NeighborListBuildTraits<PeriodicArborX2dNeighborList<MemorySpace, ImageShiftScalar>> {
  //! \name Aliases
  //@{

  using list_type = PeriodicArborX2dNeighborList<MemorySpace, ImageShiftScalar>;
  using target_input_type = impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  using source_input_type = impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `PeriodicArborX2dNeighborList`.
  struct args_type {
    int buffer_size = 0;  ///< Optional maximum-neighbor preallocation guess.
  };
  //@}

  //! \name Build
  //@{

  /// \brief Build the list from a complete builder and BulkData.
  /// \tparam Builder Complete `NeighborListBuilder` type carrying exec space, inputs, and excluder.
  /// \param builder [in] Complete builder. Target and source inputs must be
  ///                     `PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>`.
  /// \param bulk_data [in] STK bulk data for excluder setup.
  /// \param args [in] Build-specific parameters.
  template <typename Builder>
    requires std::same_as<typename Builder::target_input_type, target_input_type> &&
             std::same_as<typename Builder::source_input_type, source_input_type>
  static list_type build(const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& args);
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_ARBORX2DNEIGHBORLIST_HPP_
