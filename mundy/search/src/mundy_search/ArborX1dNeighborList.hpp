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

#ifndef MUNDY_SEARCH_ARBORX1DNEIGHBORLIST_HPP_
#define MUNDY_SEARCH_ARBORX1DNEIGHBORLIST_HPP_

/// \file ArborX1dNeighborList.hpp
/// \brief ArborX compressed 1D neighbor-list types and their NeighborListBuildTraits specializations.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// C++ core
#include <concepts>   // for std::same_as
#include <cstddef>    // for size_t
#include <stdexcept>  // for std::invalid_argument, std::out_of_range

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>                              // for mundy::Vector3
#include <mundy_search/NeighborListBuildTraits.hpp>            // for NeighborListBuildTraits
#include <mundy_search/impl/ArborXCallback.hpp>               // for ArborXExcluderCallback, ArborXSearchCandidateFactory, PeriodicArborXSearchCandidateFactory
#include <mundy_search/impl/ArborXPeriodicBuildCallbacks.hpp>  // for ArborXPeriodicCountCallback, ArborXPeriodic1dFillCallback
#include <mundy_search/impl/ArborXSearchBoxes.hpp>             // for impl::ArborXSearchBoxesT, impl::PeriodicArborXSearchBoxesT
#include <mundy_utils/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT, MUNDY_THROW_REQUIRE

namespace mundy {

namespace search {

/// \class ArborX1dNeighborList
/// \brief ArborX neighbor list with Cabana-style compressed 1D storage.
///
/// This implementation stores target entities, source entities, a flattened source-index array, and per-target offsets.
/// Search boxes are not retained after construction.
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename MemorySpace = stk::ngp::MemSpace>
class ArborX1dNeighborList {
 public:
  //! \name Aliases
  //@{

  // clang-format off
  using memory_space      = MemorySpace;
  using execution_space   = typename MemorySpace::execution_space;
  using size_type         = size_t;
  using source_index_type = size_type;
  using entity_view_t       = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t       = Kokkos::View<size_type*, memory_space>;
  // clang-format on
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborX1dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList(const ArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList(ArborX1dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList& operator=(const ArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList& operator=(ArborX1dNeighborList&&) = default;

  /// \brief Construct a list from already-built compressed storage.
  /// \param target_selector [in] Selector that defines the target chunk; stored for later inspection.
  /// \param source_selector [in] Selector that defines the source chunk; stored for later inspection.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  ArborX1dNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
                       const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_selector_(target_selector),
        source_selector_(source_selector),
        target_entities_(target_entities),
        source_entities_(source_entities),
        source_indices_(source_indices),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "ArborX1dNeighborList: offsets extent must be num_targets + 1.");
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
                       "ArborX1dNeighborList::num_neighbors target index out of range.");
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
                       "ArborX1dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "ArborX1dNeighborList::source_entity source index out of range.");
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
                       "ArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

/// \class PeriodicArborX1dNeighborList
/// \brief ArborX compressed 1D neighbor list whose stored pairs carry relative periodic image shifts.
///
/// Targets and sources are indexed by owner ordinals, not image ordinals. Multiple stored pairs may therefore reference
/// the same source owner with different relative shifts. Kernels should reconstruct shifted source geometry from the
/// source owner fields and `relative_image_shift(target_index, neighbor_ordinal)`.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicArborX1dNeighborList {
 public:
  //! \name Aliases
  //@{

  // clang-format off
  using memory_space        = MemorySpace;
  using execution_space     = typename MemorySpace::execution_space;
  using image_shift_scalar  = ImageShiftScalar;
  using size_type           = size_t;
  using source_index_type   = size_type;
  using image_shift_type    = mundy::Vector3<image_shift_scalar>;
  using entity_view_t       = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t       = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t  = Kokkos::View<image_shift_type*, memory_space>;
  // clang-format on
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborX1dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList(const PeriodicArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList(PeriodicArborX1dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList& operator=(const PeriodicArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList& operator=(PeriodicArborX1dNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built compressed storage.
  /// \param target_selector [in] Selector that defines the target owner chunk; stored for later inspection.
  /// \param source_selector [in] Selector that defines the source owner chunk; stored for later inspection.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param source_owner_indices [in] Dense source owner ordinal for every stored pair.
  /// \param relative_image_shifts [in] Source image shift minus target image shift for every stored pair.
  /// \param offsets [in] Target owner offsets into `source_owner_indices`; extent must be `num_targets + 1`.
  PeriodicArborX1dNeighborList(const stk::mesh::Selector& target_selector, const stk::mesh::Selector& source_selector,
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
                       "PeriodicArborX1dNeighborList: offsets extent must be num_targets + 1.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == relative_image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicArborX1dNeighborList: source_owner_indices and relative_image_shifts must have the "
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
                       "PeriodicArborX1dNeighborList::num_neighbors target index out of range.");
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
                       "PeriodicArborX1dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicArborX1dNeighborList::source_entity source index out of range.");
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
                       "PeriodicArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

// -----------------------------------------------------------------------------
// NeighborListBuildTraits specializations for ArborX compressed 1D list types
// -----------------------------------------------------------------------------

/// \struct NeighborListBuildTraits<ArborX1dNeighborList<MemorySpace>>
/// \brief Build traits for `ArborX1dNeighborList`: non-periodic ArborX compressed 1D storage.
///
/// The build runs ArborX over non-periodic boxes, applies the builder's excluder chain, and returns compressed
/// 1D target-to-source storage. Declaration only for this design pass; the definition runs ArborX.
/// \tparam MemorySpace Kokkos memory space for the returned list.
template <typename MemorySpace>
struct NeighborListBuildTraits<ArborX1dNeighborList<MemorySpace>> {
  //! \name Aliases
  //@{

  using list_type = ArborX1dNeighborList<MemorySpace>;
  using target_input_type = impl::ArborXSearchBoxesT<MemorySpace>;
  using source_input_type = impl::ArborXSearchBoxesT<MemorySpace>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `ArborX1dNeighborList`.
  struct args_type {
    int buffer_size = 0;  ///< Optional ArborX traversal buffer-size hint.
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

/// \struct NeighborListBuildTraits<PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar>>
/// \brief Build traits for `PeriodicArborX1dNeighborList`: periodic ArborX compressed 1D storage.
///
/// The build runs ArborX over periodic image boxes, collapses every match back to owner ordinals, and stores
/// `source_image_shift - target_image_shift` for each retained owner pair. Declaration only for this design pass.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct NeighborListBuildTraits<PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar>> {
  //! \name Aliases
  //@{

  using list_type = PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar>;
  using target_input_type = impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  using source_input_type = impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //@}

  //! \name Build parameters
  //@{

  /// \brief Build-specific parameters for `PeriodicArborX1dNeighborList`.
  struct args_type {
    int buffer_size = 0;  ///< Optional ArborX traversal buffer-size hint.
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

// -----------------------------------------------------------------------
// NeighborListBuildTraits::build() definitions — non-periodic 1D
// -----------------------------------------------------------------------

/// \brief Build an `ArborX1dNeighborList` using ArborX managed-CSR output.
///
/// Builds a BVH over source boxes, runs a single ArborX query pass with the excluder-filtering callback, and
/// casts the resulting `int`-typed ArborX output views to the `size_t`-typed views expected by the list constructor.
template <typename MemorySpace>
template <typename Builder>
  requires std::same_as<typename Builder::target_input_type, impl::ArborXSearchBoxesT<MemorySpace>> &&
           std::same_as<typename Builder::source_input_type, impl::ArborXSearchBoxesT<MemorySpace>>
ArborX1dNeighborList<MemorySpace>
NeighborListBuildTraits<ArborX1dNeighborList<MemorySpace>>::build(const Builder& builder,
                                                                   const stk::mesh::BulkData& bulk_data,
                                                                   const args_type& args) {
  MUNDY_THROW_REQUIRE(bulk_data.parallel_size() == 1, std::invalid_argument,
                      "ArborX1dNeighborList build currently supports only single-process runs.");

  using exec_space = typename Builder::execution_space;
  using size_type = typename list_type::size_type;
  using factory_type = impl::ArborXSearchCandidateFactory<target_input_type, source_input_type>;
  using callback_type = impl::ArborXExcluderCallback<factory_type, typename Builder::excluder_type>;

  const auto& target_boxes = builder.target_input();
  const auto& source_boxes = builder.source_input();
  const auto exec_sp = builder.exec_space();
  const auto excluder = builder.setup_excluder(bulk_data);
  const size_type num_targets = target_boxes.size();

  factory_type factory(target_boxes, source_boxes);
  callback_type callback(factory, excluder);

  // ArborX-allocated int output views; ArborX resizes these during the query call.
  Kokkos::View<int*, MemorySpace> raw_indices;
  Kokkos::View<int*, MemorySpace> raw_offsets;

#if ARBORX_VERSION >= 10799
  // New API: source primitives passed as a raw Box view with attached ordinals.
  // ArborX provides built-in AccessTraits for Kokkos::View<Box*, MemSpace>.
  ArborX::BoundingVolumeHierarchy bvh(exec_sp,
                                      ArborX::Experimental::attach_indices<int>(source_boxes.boxes()));
#else
  // Old API: pass the full wrapper; uses AccessTraits<ArborXSearchBoxesT, PrimitivesTag>.
  ArborX::BVH<MemorySpace> bvh(exec_sp, source_boxes);
#endif

  bvh.query(exec_sp, target_boxes, callback, raw_indices, raw_offsets,
            ArborX::Experimental::TraversalPolicy().setBufferSize(args.buffer_size));

  const size_type num_pairs = static_cast<size_type>(raw_indices.extent(0));
  Kokkos::View<size_type*, MemorySpace> source_indices("mundy_search_1d_source_idx", num_pairs);
  Kokkos::View<size_type*, MemorySpace> offsets("mundy_search_1d_offsets", num_targets + 1);

  Kokkos::parallel_for("mundy_search_1d_cast_idx", Kokkos::RangePolicy<exec_space>(0, num_pairs),
                       KOKKOS_LAMBDA(size_type i) { source_indices(i) = static_cast<size_type>(raw_indices(i)); });
  Kokkos::parallel_for("mundy_search_1d_cast_off", Kokkos::RangePolicy<exec_space>(0, num_targets + 1),
                       KOKKOS_LAMBDA(size_type i) { offsets(i) = static_cast<size_type>(raw_offsets(i)); });

  if (builder.sort_neighbors()) {
    // Sort each target's neighbor row by ascending source ordinal. Insertion sort is used because row
    // sizes are small (~10-20 entries at typical densities) and it incurs no auxiliary allocation.
    Kokkos::parallel_for(
        "mundy_search_1d_sort_rows", Kokkos::RangePolicy<exec_space>(0, num_targets),
        KOKKOS_LAMBDA(size_type t) {
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

  return list_type(builder.target_selector(), builder.source_selector(), target_boxes.entities(),
                   source_boxes.entities(), source_indices, offsets);
}

// -----------------------------------------------------------------------
// NeighborListBuildTraits::build() definitions — periodic 1D
// -----------------------------------------------------------------------

/// \brief Build a `PeriodicArborX1dNeighborList` using a two-pass owner-indexed count/fill strategy.
///
/// ArborX returns image ordinals; the count callback maps each surviving hit to the target owner ordinal for the
/// atomic count. A prefix scan converts counts to CSR offsets, and the fill callback writes into flat owner-indexed
/// pair arrays. The resulting list stores source owner ordinals and relative image shifts.
template <typename MemorySpace, typename ImageShiftScalar>
template <typename Builder>
  requires std::same_as<typename Builder::target_input_type,
                        impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>> &&
           std::same_as<typename Builder::source_input_type,
                        impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>>
PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar>
NeighborListBuildTraits<PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar>>::build(
    const Builder& builder, const stk::mesh::BulkData& bulk_data, const args_type& args) {
  MUNDY_THROW_REQUIRE(bulk_data.parallel_size() == 1, std::invalid_argument,
                      "PeriodicArborX1dNeighborList build currently supports only single-process runs.");

  using exec_space = typename Builder::execution_space;
  using size_type = typename list_type::size_type;
  using image_shift_type = typename list_type::image_shift_type;
  using factory_type = impl::PeriodicArborXSearchCandidateFactory<target_input_type, source_input_type>;
  using excluder_type = typename Builder::excluder_type;
  using count_cb_t = impl::ArborXPeriodicCountCallback<target_input_type, source_input_type, excluder_type>;
  using fill_cb_t =
      impl::ArborXPeriodic1dFillCallback<target_input_type, source_input_type, excluder_type, image_shift_type>;

  const auto& target_boxes = builder.target_input();
  const auto& source_boxes = builder.source_input();
  const auto exec_sp = builder.exec_space();
  const auto excluder = builder.setup_excluder(bulk_data);
  const size_type num_target_owners = target_boxes.num_owners();

  factory_type factory(target_boxes, source_boxes);

#if ARBORX_VERSION >= 10799
  ArborX::BoundingVolumeHierarchy bvh(exec_sp,
                                      ArborX::Experimental::attach_indices<int>(source_boxes.boxes()));
#else
  ArborX::BVH<MemorySpace> bvh(exec_sp, source_boxes);
#endif

  // Pass 1: count surviving pairs per target owner (image→owner mapping done inside callback).
  Kokkos::View<size_type*, MemorySpace> owner_counts("mundy_search_per1d_counts", num_target_owners);
  Kokkos::deep_copy(owner_counts, size_type(0));
  bvh.query(exec_sp, target_boxes, count_cb_t(factory, excluder, owner_counts));

  // Compute total pairs so output views can be pre-allocated.
  size_type total_pairs = 0;
  Kokkos::parallel_reduce("mundy_search_per1d_total",
                          Kokkos::RangePolicy<exec_space>(0, num_target_owners),
                          KOKKOS_LAMBDA(size_type i, size_type& partial) { partial += owner_counts(i); },
                          total_pairs);

  Kokkos::View<size_type*, MemorySpace> offsets("mundy_search_per1d_offsets", num_target_owners + 1);
  Kokkos::View<size_type*, MemorySpace> write_positions("mundy_search_per1d_wpos", num_target_owners);
  Kokkos::View<size_type*, MemorySpace> source_owner_indices("mundy_search_per1d_soi", total_pairs);
  Kokkos::View<image_shift_type*, MemorySpace> relative_image_shifts("mundy_search_per1d_ris", total_pairs);
  Kokkos::deep_copy(write_positions, size_type(0));

  // Exclusive prefix scan: offsets[i] = sum(owner_counts[0..i)), offsets[N] = total_pairs.
  Kokkos::parallel_scan("mundy_search_per1d_scan",
                        Kokkos::RangePolicy<exec_space>(0, num_target_owners + 1),
                        KOKKOS_LAMBDA(size_type i, size_type& update, bool final) {
                          if (final) offsets(i) = update;
                          update += (i < num_target_owners) ? owner_counts(i) : size_type(0);
                        });

  // Pass 2: fill source owner ordinals and relative image shifts.
  bvh.query(exec_sp, target_boxes,
            fill_cb_t(factory, excluder, write_positions, offsets, source_owner_indices, relative_image_shifts));

  if (builder.sort_neighbors()) {
    // Sort each target owner's neighbor row by ascending source owner ordinal, keeping
    // relative_image_shifts in sync so each pair's shift stays with its source ordinal.
    Kokkos::parallel_for(
        "mundy_search_per1d_sort_rows", Kokkos::RangePolicy<exec_space>(0, num_target_owners),
        KOKKOS_LAMBDA(size_type t) {
          const size_type beg = offsets(t);
          const size_type end = offsets(t + 1);
          for (size_type i = beg + 1; i < end; ++i) {
            const size_type     key_idx   = source_owner_indices(i);
            const image_shift_type key_shift = relative_image_shifts(i);
            size_type j = i;
            while (j > beg && source_owner_indices(j - 1) > key_idx) {
              source_owner_indices(j)   = source_owner_indices(j - 1);
              relative_image_shifts(j)  = relative_image_shifts(j - 1);
              --j;
            }
            source_owner_indices(j)  = key_idx;
            relative_image_shifts(j) = key_shift;
          }
        });
  }

  return list_type(builder.target_selector(), builder.source_selector(), target_boxes.owner_entities(),
                   source_boxes.owner_entities(), source_owner_indices, relative_image_shifts, offsets);
}

}  // namespace search

}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // MUNDY_SEARCH_ARBORX1DNEIGHBORLIST_HPP_
