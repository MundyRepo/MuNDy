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

#ifndef MUNDY_MESH_NEIGHBORLIST_HPP_
#define MUNDY_MESH_NEIGHBORLIST_HPP_

/// \file NeighborList.hpp
/// \brief First-pass Mundy neighbor-list interface sketch.

// C++ core
#include <cstddef>    // for size_t
#include <stdexcept>  // for std::invalid_argument

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

/// \struct FullNeighborTag
/// \brief Build a directed full neighbor list.
///
/// If target A neighbors source B, the reverse pair is also stored whenever the target/source sets make the reverse
/// pair meaningful. Self pairs are still removed.
struct FullNeighborTag {};

/// \struct HalfNeighborTag
/// \brief Build a half neighbor list.
///
/// Duplicate pairs are removed. The exact ordering policy is implementation-owned because ArborX and STKSearch expose
/// different candidate-pair shapes, but all implementations should converge on entity-order based duplicate removal for
/// same-rank entity sets.
struct HalfNeighborTag {};

/// \struct SelfNeighborTag
/// \brief Build a list that only removes self interactions.
///
/// This follows Cabana's current "self" discriminator meaning: remove pairs where target and source are the same
/// entity, but do not otherwise attempt full/half symmetry management.
struct SelfNeighborTag {};

/// \class NeighborList
/// \brief Static interface facade for a concrete neighbor-list implementation.
///
/// Concrete neighbor-list types own storage and search details. This facade defines the common access and iteration
/// surface used by kernels and higher-level Mundy algorithms. It intentionally uses static dispatch rather than virtual
/// functions so that Kokkos device code can inline through the implementation type.
/// \tparam NeighborListType The concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborList;

/// \struct NeighborListTraits
/// \brief Type traits that describe the target/source handle contract for a neighbor list.
///
/// A neighbor list may expose contiguous target/source handles, such as ordinals in `[0, num_targets)`, or
/// non-contiguous handles, such as `stk::mesh::FastMeshIndex`. Dense target enumeration is a separate contract: a list
/// can support iteration over target ordinals without using those ordinals as the public target handle type.
/// \tparam NeighborListType The concrete neighbor-list implementation type.
template <typename NeighborListType>
struct NeighborListTraits {
  //! Concrete neighbor-list type.
  using neighbor_list_type = NeighborListType;
  //! Integral size type used for counts, ordinals, and compact pair storage.
  using size_type = typename neighbor_list_type::size_type;
  //! Default execution space for iteration over this list.
  using execution_space = typename neighbor_list_type::execution_space;
  //! Handle type used to identify a target in this list.
  using target_index_type = typename neighbor_list_type::target_index_type;
  //! Handle type used to identify a source in this list.
  using source_index_type = typename neighbor_list_type::source_index_type;

  //! True if targets can be addressed by dense ordinals.
  static constexpr bool has_contiguous_targets = neighbor_list_type::has_contiguous_targets;
  //! True if sources can be addressed by dense ordinals.
  static constexpr bool has_contiguous_sources = neighbor_list_type::has_contiguous_sources;
  //! True if `[0, num_targets)` is a valid target-iteration space for this list.
  static constexpr bool has_dense_target_ordinals = neighbor_list_type::has_dense_target_ordinals;
  //! True if both targets and sources are dense ordinal-addressed.
  static constexpr bool is_contiguous = has_contiguous_targets && has_contiguous_sources;
};

namespace impl {

/// \class NeighborSearchEntitiesT
/// \brief Device-accessible map from neighbor-list handles to required STK entity metadata.
///
/// Search algorithms work with compact target/source handles, but ECS kernels need to recover the STK entity and fast
/// mesh index associated with each handle. Periodic-image and owner bookkeeping are intentionally not part of this core
/// map; those live in `NeighborImageMetadataT` so non-periodic lists do not inherit WCM-specific storage.
/// \tparam MemorySpace Kokkos memory space in which the metadata views live.
template <typename MemorySpace>
class NeighborSearchEntitiesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using fast_mesh_index_view_t = Kokkos::View<stk::mesh::FastMeshIndex*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborSearchEntitiesT() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION NeighborSearchEntitiesT(const NeighborSearchEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborSearchEntitiesT(NeighborSearchEntitiesT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborSearchEntitiesT& operator=(const NeighborSearchEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborSearchEntitiesT& operator=(NeighborSearchEntitiesT&&) = default;

  /// \brief Construct an entity metadata map from already materialized views.
  /// \param entity_rank [in] The STK rank shared by every entity in this map.
  /// \param entities [in] The entity associated with each local search handle.
  /// \param fast_mesh_indices [in] The `FastMeshIndex` associated with each local search handle.
  KOKKOS_INLINE_FUNCTION
  NeighborSearchEntitiesT(stk::mesh::EntityRank entity_rank, const entity_view_t& entities,
                          const fast_mesh_index_view_t& fast_mesh_indices)
      : entity_rank_(entity_rank), entities_(entities), fast_mesh_indices_(fast_mesh_indices) {
    MUNDY_THROW_ASSERT(entities_.extent(0) == fast_mesh_indices_.extent(0), std::invalid_argument,
                       "NeighborSearchEntitiesT: entities and fast mesh indices must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of mapped search handles.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return entities_.extent(0);
  }

  /// \brief Get the STK entity rank shared by this map.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank entity_rank() const noexcept {
    return entity_rank_;
  }

  /// \brief Get the STK entity for a local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity entity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "NeighborSearchEntitiesT::entity index out of range.");
    return entities_(index);
  }

  /// \brief Get the STK fast mesh index for a local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex fast_mesh_index(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range,
                       "NeighborSearchEntitiesT::fast_mesh_index index out of range.");
    return fast_mesh_indices_(index);
  }

  /// \brief Get the raw entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t entities() const noexcept {
    return entities_;
  }

  /// \brief Get the raw fast mesh index view.
  KOKKOS_INLINE_FUNCTION
  fast_mesh_index_view_t fast_mesh_indices() const noexcept {
    return fast_mesh_indices_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! STK entity rank shared by every mapped search handle.
  stk::mesh::EntityRank entity_rank_;
  //! Entity identity for each local target/source handle.
  entity_view_t entities_;
  //! Fast mesh index for field/component access associated with each handle.
  fast_mesh_index_view_t fast_mesh_indices_;
  //@}
};

/// \class NeighborImageMetadataT
/// \brief Optional periodic-image metadata for target/source handles.
///
/// This helper is deliberately separate from `NeighborSearchEntitiesT`. Periodic images and owner ordinals are
/// essential for some search workflows, but they are not part of the minimum neighbor-list access contract. Builders
/// that produce periodic-image candidates can carry this object beside a list or inside a higher-level
/// algorithm-specific wrapper.
/// \tparam MemorySpace Kokkos memory space in which the metadata views live.
template <typename MemorySpace>
class NeighborImageMetadataT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using periodic_shift_t = Kokkos::Array<float, 3>;
  using periodic_shift_view_t = Kokkos::View<periodic_shift_t*, memory_space>;
  using owner_index_view_t = Kokkos::View<int*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborImageMetadataT() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION NeighborImageMetadataT(const NeighborImageMetadataT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborImageMetadataT(NeighborImageMetadataT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborImageMetadataT& operator=(const NeighborImageMetadataT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NeighborImageMetadataT& operator=(NeighborImageMetadataT&&) = default;

  /// \brief Construct periodic-image metadata from already materialized views.
  /// \param periodic_shifts [in] Periodic-image shift associated with each local search handle.
  /// \param owner_indices [in] Index of the owning non-periodic entity for each local search handle.
  KOKKOS_INLINE_FUNCTION
  NeighborImageMetadataT(const periodic_shift_view_t& periodic_shifts, const owner_index_view_t& owner_indices)
      : periodic_shifts_(periodic_shifts), owner_indices_(owner_indices) {
    MUNDY_THROW_ASSERT(periodic_shifts_.extent(0) == owner_indices_.extent(0), std::invalid_argument,
                       "NeighborImageMetadataT: periodic shifts and owner indices must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of mapped image handles.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return periodic_shifts_.extent(0);
  }

  /// \brief Get the periodic-image shift for a local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  periodic_shift_t periodic_shift(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "NeighborImageMetadataT::periodic_shift index out of range.");
    return periodic_shifts_(index);
  }

  /// \brief Get the owning non-periodic entity index for a local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  int owner_index(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "NeighborImageMetadataT::owner_index index out of range.");
    return owner_indices_(index);
  }

  /// \brief Get the raw periodic shift view.
  KOKKOS_INLINE_FUNCTION
  periodic_shift_view_t periodic_shifts() const noexcept {
    return periodic_shifts_;
  }

  /// \brief Get the raw owner index view.
  KOKKOS_INLINE_FUNCTION
  owner_index_view_t owner_indices() const noexcept {
    return owner_indices_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Periodic-image shift carried by each handle.
  periodic_shift_view_t periodic_shifts_;
  //! Owning non-periodic entity index for each handle.
  owner_index_view_t owner_indices_;
  //@}
};

/// \class ArborXSearchBoxesT
/// \brief Device-accessible ArborX boxes paired with their STK entity metadata.
///
/// ArborX only sees geometric primitives and predicate payloads. Mundy kernels need to recover the target/source entity
/// metadata associated with each primitive, so this wrapper carries both the `ArborX::Box` view and the corresponding
/// `NeighborSearchEntitiesT`.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity map live.
template <typename MemorySpace>
class ArborXSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using entity_map_type = NeighborSearchEntitiesT<memory_space>;
  using box_view_t = Kokkos::View<ArborX::Box*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborXSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION ArborXSearchBoxesT(const ArborXSearchBoxesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborXSearchBoxesT(ArborXSearchBoxesT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborXSearchBoxesT& operator=(const ArborXSearchBoxesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborXSearchBoxesT& operator=(ArborXSearchBoxesT&&) = default;

  /// \brief Construct ArborX search boxes from an entity map and matching box view.
  /// \param entity_map [in] Entity metadata associated with each box.
  /// \param boxes [in] ArborX boxes used as primitives or predicates.
  KOKKOS_INLINE_FUNCTION
  ArborXSearchBoxesT(const entity_map_type& entity_map, const box_view_t& boxes)
      : entity_map_(entity_map), boxes_(boxes) {
    MUNDY_THROW_ASSERT(entity_map_.size() == boxes_.extent(0), std::invalid_argument,
                       "ArborXSearchBoxesT: entity map and boxes must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  ArborX::Box box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "ArborXSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get the entity map associated with these boxes.
  KOKKOS_INLINE_FUNCTION
  const entity_map_type& entities() const noexcept {
    return entity_map_;
  }

  /// \brief Get the mutable entity map associated with these boxes.
  KOKKOS_INLINE_FUNCTION
  entity_map_type& entities() noexcept {
    return entity_map_;
  }

  /// \brief Get the raw ArborX box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! STK entity metadata associated one-to-one with `boxes_`.
  entity_map_type entity_map_;
  //! ArborX boxes used as primitives or target predicates.
  box_view_t boxes_;
  //@}
};

/// \class STKSearchBoxesT
/// \brief Device-accessible STK search boxes paired with their STK entity metadata.
///
/// This is the STK coarse-search counterpart to `ArborXSearchBoxesT`. It intentionally keeps the same entity-map shape
/// so that STKSearch and ArborX neighbor lists can share Mundy's target/source access surface.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity map live.
template <typename MemorySpace>
class STKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using entity_map_type = NeighborSearchEntitiesT<memory_space>;
  using box_type = stk::search::Box<double>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  STKSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION STKSearchBoxesT(const STKSearchBoxesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchBoxesT(STKSearchBoxesT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchBoxesT& operator=(const STKSearchBoxesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchBoxesT& operator=(STKSearchBoxesT&&) = default;

  /// \brief Construct STK search boxes from an entity map and matching box view.
  /// \param entity_map [in] Entity metadata associated with each box.
  /// \param boxes [in] STK boxes used for coarse search.
  KOKKOS_INLINE_FUNCTION
  STKSearchBoxesT(const entity_map_type& entity_map, const box_view_t& boxes) : entity_map_(entity_map), boxes_(boxes) {
    MUNDY_THROW_ASSERT(entity_map_.size() == boxes_.extent(0), std::invalid_argument,
                       "STKSearchBoxesT: entity map and boxes must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search handle.
  /// \param index [in] Local search handle.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "STKSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get the entity map associated with these boxes.
  KOKKOS_INLINE_FUNCTION
  const entity_map_type& entities() const noexcept {
    return entity_map_;
  }

  /// \brief Get the mutable entity map associated with these boxes.
  KOKKOS_INLINE_FUNCTION
  entity_map_type& entities() noexcept {
    return entity_map_;
  }

  /// \brief Get the raw STK box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! STK entity metadata associated one-to-one with `boxes_`.
  entity_map_type entity_map_;
  //! STK search boxes used by coarse search.
  box_view_t boxes_;
  //@}
};

/// \struct IdenticalSelectorOverlapTag
/// \brief Internal build-time tag for identical target/source selectors.
struct IdenticalSelectorOverlapTag {};

/// \struct DisjointSelectorOverlapTag
/// \brief Internal build-time tag for disjoint target/source selectors.
struct DisjointSelectorOverlapTag {};

/// \struct PartialSelectorOverlapTag
/// \brief Internal build-time tag for partially overlapping target/source selectors.
struct PartialSelectorOverlapTag {};

/// \struct NeighborBuildSemantics
/// \brief Policy hook for applying full/half/self neighbor semantics to build candidate pairs.
///
/// Specializations expose a `keep` function that decides whether a geometric candidate pair belongs in the final
/// neighbor list. Selector-overlap state is part of the policy type rather than a runtime enum so factories can select
/// a callback that matches identical, disjoint, or partially overlapping target/source selectors.
/// \tparam NeighborTag One of `FullNeighborTag`, `HalfNeighborTag`, or `SelfNeighborTag`.
/// \tparam SelectorOverlapTag One of the internal selector-overlap tags above.
template <typename NeighborTag, typename SelectorOverlapTag>
struct NeighborBuildSemantics;

/// \struct NeighborBuildSemantics<FullNeighborTag, SelectorOverlapTag>
/// \brief Full-list candidate filter for any selector-overlap shape.
template <typename SelectorOverlapTag>
struct NeighborBuildSemantics<FullNeighborTag, SelectorOverlapTag> {
  /// \brief Return true for every non-self candidate pair.
  /// \param targets [in] Target entity metadata.
  /// \param target_index [in] Target local search handle.
  /// \param sources [in] Source entity metadata.
  /// \param source_index [in] Source local search handle.
  template <typename SearchEntitiesType>
  KOKKOS_INLINE_FUNCTION static bool keep(const SearchEntitiesType& targets, size_t target_index,
                                          const SearchEntitiesType& sources, size_t source_index) {
    return targets.entity(target_index) != sources.entity(source_index);
  }
};

/// \struct NeighborBuildSemantics<HalfNeighborTag, IdenticalSelectorOverlapTag>
/// \brief Half-list candidate filter for identical target/source selectors.
template <>
struct NeighborBuildSemantics<HalfNeighborTag, IdenticalSelectorOverlapTag> {
  /// \brief Return true for non-self candidate pairs that survive duplicate suppression.
  /// \param targets [in] Target entity metadata.
  /// \param target_index [in] Target local search handle.
  /// \param sources [in] Source entity metadata.
  /// \param source_index [in] Source local search handle.
  template <typename SearchEntitiesType>
  KOKKOS_INLINE_FUNCTION static bool keep(const SearchEntitiesType& targets, size_t target_index,
                                          const SearchEntitiesType& sources, size_t source_index) {
    const stk::mesh::Entity target = targets.entity(target_index);
    const stk::mesh::Entity source = sources.entity(source_index);
    return target != source && target < source;
  }
};

/// \struct NeighborBuildSemantics<HalfNeighborTag, DisjointSelectorOverlapTag>
/// \brief Half-list candidate filter for disjoint target/source selectors.
template <>
struct NeighborBuildSemantics<HalfNeighborTag, DisjointSelectorOverlapTag> {
  /// \brief Return true for every candidate pair except an entity paired with itself.
  /// \param targets [in] Target entity metadata.
  /// \param target_index [in] Target local search handle.
  /// \param sources [in] Source entity metadata.
  /// \param source_index [in] Source local search handle.
  template <typename SearchEntitiesType>
  KOKKOS_INLINE_FUNCTION static bool keep(const SearchEntitiesType& targets, size_t target_index,
                                          const SearchEntitiesType& sources, size_t source_index) {
    return targets.entity(target_index) != sources.entity(source_index);
  }
};

/// \struct NeighborBuildSemantics<HalfNeighborTag, PartialSelectorOverlapTag>
/// \brief Half-list candidate filter for partially overlapping target/source selectors.
///
/// The overlap predicate must return true when a local search handle belongs to the selector intersection. Pairs where
/// both entities are in the intersection are duplicate-suppressed; pairs outside the intersection keep full non-self
/// semantics.
template <>
struct NeighborBuildSemantics<HalfNeighborTag, PartialSelectorOverlapTag> {
  /// \brief Return true for partially overlapping half-list selector semantics.
  /// \param targets [in] Target entity metadata.
  /// \param target_index [in] Target local search handle.
  /// \param sources [in] Source entity metadata.
  /// \param source_index [in] Source local search handle.
  /// \param in_overlap [in] Device-callable predicate over a metadata map and local search handle.
  template <typename SearchEntitiesType, typename OverlapPredicate>
  KOKKOS_INLINE_FUNCTION static bool keep(const SearchEntitiesType& targets, size_t target_index,
                                          const SearchEntitiesType& sources, size_t source_index,
                                          const OverlapPredicate& in_overlap) {
    const stk::mesh::Entity target = targets.entity(target_index);
    const stk::mesh::Entity source = sources.entity(source_index);
    const bool both_in_overlap = in_overlap(targets, target_index) && in_overlap(sources, source_index);
    return target != source && (!both_in_overlap || target < source);
  }
};

/// \struct NeighborBuildSemantics<SelfNeighborTag, SelectorOverlapTag>
/// \brief Self-filter candidate policy for any selector-overlap shape.
template <typename SelectorOverlapTag>
struct NeighborBuildSemantics<SelfNeighborTag, SelectorOverlapTag> {
  /// \brief Return true for every candidate pair except an entity paired with itself.
  /// \param targets [in] Target entity metadata.
  /// \param target_index [in] Target local search handle.
  /// \param sources [in] Source entity metadata.
  /// \param source_index [in] Source local search handle.
  template <typename SearchEntitiesType>
  KOKKOS_INLINE_FUNCTION static bool keep(const SearchEntitiesType& targets, size_t target_index,
                                          const SearchEntitiesType& sources, size_t source_index) {
    return targets.entity(target_index) != sources.entity(source_index);
  }
};

}  // namespace impl

/// \class Neighbors
/// \brief Lightweight neighbor-range view for one target.
///
/// `Neighbors` stores the concrete list and a target handle. The handle type comes from `NeighborListTraits`, which
/// lets the same wrapper support both dense ordinal-addressed lists and future non-contiguous lists.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class Neighbors {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using traits_type = NeighborListTraits<neighbor_list_type>;
  using size_type = typename traits_type::size_type;
  using target_index_type = typename traits_type::target_index_type;
  using source_index_type = typename traits_type::source_index_type;

  static constexpr bool has_contiguous_targets = traits_type::has_contiguous_targets;
  static constexpr bool has_contiguous_sources = traits_type::has_contiguous_sources;
  static constexpr bool has_dense_target_ordinals = traits_type::has_dense_target_ordinals;
  static constexpr bool is_contiguous = traits_type::is_contiguous;
  //@}

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION Neighbors() = default;

  /// \brief Construct a neighbor range for a target.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  KOKKOS_INLINE_FUNCTION
  Neighbors(const neighbor_list_type& list, target_index_type target_index) : list_(list), target_index_(target_index) {
  }

  /// \brief Get the number of neighbors for the target.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    return NeighborList<neighbor_list_type>::num_neighbors(list_, target_index_);
  }

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator[](size_type neighbor_ordinal) const {
    return NeighborList<neighbor_list_type>::get_neighbor(list_, target_index_, neighbor_ordinal);
  }

  /// \brief Get the source handle for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type neighbor_ordinal) const {
    return list_.source_index(target_index_, neighbor_ordinal);
  }

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator()(size_type neighbor_ordinal) const {
    return (*this)[neighbor_ordinal];
  }

  /// \brief Get the target handle associated with this range.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index() const noexcept {
    return target_index_;
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Target handle whose neighbor range is being viewed.
  target_index_type target_index_;
  //@}
};

/// \class NeighborPair
/// \brief Explicit payload passed to pair-iteration functors.
///
/// A pair payload carries the target handle and neighbor ordinal. Source information is resolved lazily through the
/// underlying list so pair kernels can ask for handles, STK entities, or fast mesh indices without coupling the core
/// neighbor-list API to periodic-image or owner-index metadata.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborPair {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using traits_type = NeighborListTraits<neighbor_list_type>;
  using size_type = typename traits_type::size_type;
  using target_index_type = typename traits_type::target_index_type;
  using source_index_type = typename traits_type::source_index_type;

  static constexpr bool has_contiguous_targets = traits_type::has_contiguous_targets;
  static constexpr bool has_contiguous_sources = traits_type::has_contiguous_sources;
  static constexpr bool has_dense_target_ordinals = traits_type::has_dense_target_ordinals;
  //@}

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION NeighborPair() = default;

  /// \brief Construct a pair payload.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  /// \param neighbor_ordinal [in] Ordinal of the source neighbor for the target.
  KOKKOS_INLINE_FUNCTION
  NeighborPair(const neighbor_list_type& list, target_index_type target_index, size_type neighbor_ordinal)
      : list_(list), target_index_(target_index), neighbor_ordinal_(neighbor_ordinal) {
  }

  /// \brief Get the compact pair index for this stored neighbor pair.
  ///
  /// This index is valid in `[0, NeighborList<neighbor_list_type>::size(list))` and is suitable for compact pair-field
  /// storage. Dense implementations must not return an allocated row slot here unless dense slots and compact pair ids
  /// are identical.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index() const {
    return list_.pair_index(target_index_, neighbor_ordinal_);
  }

  /// \brief Get the implementation storage slot for this stored neighbor pair.
  ///
  /// For compressed storage this is normally identical to `pair_index()`. Dense-row storage may return a row-major slot
  /// in an overallocated target-by-neighbor table.
  KOKKOS_INLINE_FUNCTION
  size_type storage_index() const {
    return list_.storage_index(target_index_, neighbor_ordinal_);
  }

  /// \brief Get the target handle for this pair.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the source ordinal within the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type neighbor_ordinal() const noexcept {
    return neighbor_ordinal_;
  }

  /// \brief Get the source handle for this pair.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index() const {
    return list_.source_index(target_index_, neighbor_ordinal_);
  }

  /// \brief Get the target STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_.target_entity(target_index_);
  }

  /// \brief Get the source STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const {
    return list_.source_entity(source_index());
  }

  /// \brief Get the target fast mesh index.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex target_fast_mesh_index() const {
    return list_.target_fast_mesh_index(target_index_);
  }

  /// \brief Get the source fast mesh index.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex source_fast_mesh_index() const {
    return list_.source_fast_mesh_index(source_index());
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Target handle for the pair.
  target_index_type target_index_;
  //! Ordinal of the source inside the target's neighbor range.
  size_type neighbor_ordinal_;
  //@}
};

/// \class TargetNeighbors
/// \brief Explicit payload passed to target-with-neighbors iteration functors.
///
/// This object gives target-parallel kernels access to a target handle and its `Neighbors` range without requiring the
/// kernel to know whether the concrete list is contiguous or non-contiguous.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class TargetNeighbors {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using traits_type = NeighborListTraits<neighbor_list_type>;
  using size_type = typename traits_type::size_type;
  using target_index_type = typename traits_type::target_index_type;
  using neighbors_type = Neighbors<neighbor_list_type>;

  static constexpr bool has_contiguous_targets = traits_type::has_contiguous_targets;
  static constexpr bool has_dense_target_ordinals = traits_type::has_dense_target_ordinals;
  //@}

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION TargetNeighbors() = default;

  /// \brief Construct a target-neighbors payload.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  KOKKOS_INLINE_FUNCTION
  TargetNeighbors(const neighbor_list_type& list, target_index_type target_index)
      : list_(list), target_index_(target_index) {
  }

  /// \brief Get the target handle.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the target STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_.target_entity(target_index_);
  }

  /// \brief Get the target fast mesh index.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex target_fast_mesh_index() const {
    return list_.target_fast_mesh_index(target_index_);
  }

  /// \brief Get the neighbor range for this target.
  KOKKOS_INLINE_FUNCTION
  neighbors_type neighbors() const {
    return NeighborList<neighbor_list_type>::get_neighbors(list_, target_index_);
  }

  /// \brief Get the number of neighbors for this target.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors() const {
    return NeighborList<neighbor_list_type>::num_neighbors(list_, target_index_);
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Target handle whose neighbor range is exposed to the user functor.
  target_index_type target_index_;
  //@}
};

namespace impl {

/// \class DeployFunctorOnNeighborPairs
/// \brief Kokkos functor that expands target-parallel work into neighbor-pair callbacks.
///
/// The outer policy is over target ordinals because every supported implementation can enumerate targets. The functor
/// converts the ordinal to the implementation's `target_index_type` before invoking the user callback.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `NeighborPair<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnNeighborPairs {
 public:
  using traits_type = NeighborListTraits<NeighborListType>;
  using size_type = typename traits_type::size_type;
  using target_index_type = typename traits_type::target_index_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every neighbor pair.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor) : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for every neighbor of one target ordinal.
  /// \param target_ordinal [in] Dense ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_ordinal) const {
    const target_index_type target_index = NeighborList<NeighborListType>::target_index(list_, target_ordinal);
    const size_type num_neighbors = NeighborList<NeighborListType>::num_neighbors(list_, target_index);
    for (size_type neighbor_ordinal = 0; neighbor_ordinal < num_neighbors; ++neighbor_ordinal) {
      functor_(NeighborPair<NeighborListType>(list_, target_index, neighbor_ordinal));
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each stored neighbor pair.
  Functor functor_;
  //@}
};

/// \class DeployFunctorOnTargetNeighbors
/// \brief Kokkos functor that invokes a target-neighbors callback for each target ordinal.
///
/// The target ordinal is converted through `NeighborList::target_index` so non-contiguous implementations can choose
/// their own target handle type while still supporting target-parallel iteration.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `TargetNeighbors<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnTargetNeighbors {
 public:
  using traits_type = NeighborListTraits<NeighborListType>;
  using size_type = typename traits_type::size_type;
  using target_index_type = typename traits_type::target_index_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every target.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnTargetNeighbors(const NeighborListType& list, const Functor& functor)
      : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for one target ordinal.
  /// \param target_ordinal [in] Dense ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_ordinal) const {
    const target_index_type target_index = NeighborList<NeighborListType>::target_index(list_, target_ordinal);
    functor_(TargetNeighbors<NeighborListType>(list_, target_index));
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each target-neighbors payload.
  Functor functor_;
  //@}
};

}  // namespace impl

/// \class NeighborList
/// \brief Common static interface for all Mundy neighbor-list implementations.
///
/// This is the concrete definition of the static facade forward-declared near the top of the file. It centralizes
/// target/source handle traits, neighbor access, and Kokkos iteration helpers for every implementation family.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborList {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using traits_type = NeighborListTraits<neighbor_list_type>;
  using size_type = typename traits_type::size_type;
  using execution_space = typename traits_type::execution_space;
  using target_index_type = typename traits_type::target_index_type;
  using source_index_type = typename traits_type::source_index_type;
  using neighbors_type = Neighbors<neighbor_list_type>;
  using neighbor_pair_type = NeighborPair<neighbor_list_type>;
  using target_neighbors_type = TargetNeighbors<neighbor_list_type>;

  static constexpr bool has_contiguous_targets = traits_type::has_contiguous_targets;
  static constexpr bool has_contiguous_sources = traits_type::has_contiguous_sources;
  static constexpr bool has_dense_target_ordinals = traits_type::has_dense_target_ordinals;
  static constexpr bool is_contiguous = traits_type::is_contiguous;
  //@}

  //! \name Size and handle queries
  //@{

  /// \brief Get the total number of stored neighbor pairs.
  /// \param list [in] Concrete neighbor list.
  KOKKOS_INLINE_FUNCTION
  static size_type size(const neighbor_list_type& list) {
    return list.size();
  }

  /// \brief Get the number of enumerable targets.
  /// \param list [in] Concrete neighbor list.
  KOKKOS_INLINE_FUNCTION
  static size_type num_targets(const neighbor_list_type& list) {
    return list.num_targets();
  }

  /// \brief Get the number of enumerable sources.
  /// \param list [in] Concrete neighbor list.
  KOKKOS_INLINE_FUNCTION
  static size_type num_sources(const neighbor_list_type& list) {
    return list.num_sources();
  }

  /// \brief Convert a dense target ordinal to this list's target handle type.
  /// \param list [in] Concrete neighbor list.
  /// \param target_ordinal [in] Dense ordinal in `[0, num_targets(list))`.
  KOKKOS_INLINE_FUNCTION
  static target_index_type target_index(const neighbor_list_type& list, size_type target_ordinal) {
    return list.target_index(target_ordinal);
  }

  /// \brief Get the number of neighbors for a target handle.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  KOKKOS_INLINE_FUNCTION
  static size_type num_neighbors(const neighbor_list_type& list, target_index_type target_index) {
    return list.num_neighbors(target_index);
  }

  /// \brief Get the neighbor entity for a target handle and neighbor ordinal.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  /// \param neighbor_ordinal [in] Ordinal in `[0, num_neighbors(list, target_index))`.
  KOKKOS_INLINE_FUNCTION
  static stk::mesh::Entity get_neighbor(const neighbor_list_type& list, target_index_type target_index,
                                        size_type neighbor_ordinal) {
    return list.neighbor_entity(target_index, neighbor_ordinal);
  }

  /// \brief Get a neighbor-range view for a target handle.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Target handle in the list's declared `target_index_type`.
  KOKKOS_INLINE_FUNCTION
  static neighbors_type get_neighbors(const neighbor_list_type& list, target_index_type target_index) {
    return neighbors_type(list, target_index);
  }
  //@}

  //! \name Parallel iteration
  //@{

  /// \brief Run a callback for every stored neighbor pair using the list's default execution space.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] Callback callable with `NeighborPair<neighbor_list_type>`.
  template <typename Functor>
  static void for_each_neighbor_pair(const neighbor_list_type& list, const Functor& functor) {
    execution_space exec_space{};
    for_each_neighbor_pair(exec_space, list, functor);
  }

  /// \brief Run a callback for every stored neighbor pair using a provided execution space.
  /// \param exec_space [in] Execution space used for the outer target-parallel loop.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] Callback callable with `NeighborPair<neighbor_list_type>`.
  template <typename ExecutionSpace, typename Functor>
  static void for_each_neighbor_pair(const ExecutionSpace& exec_space, const neighbor_list_type& list,
                                     const Functor& functor) {
    static_assert(has_dense_target_ordinals,
                  "NeighborList::for_each_neighbor_pair requires dense target ordinals for the outer iteration space.");
    using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    impl::DeployFunctorOnNeighborPairs<neighbor_list_type, Functor> deploy_functor(list, functor);
    Kokkos::parallel_for("mundy::mesh::NeighborList::for_each_neighbor_pair",
                         range_policy_t(exec_space, 0, list.num_targets()), deploy_functor);
  }

  /// \brief Run a callback for every target and its neighbor range using the list's default execution space.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] Callback callable with `TargetNeighbors<neighbor_list_type>`.
  template <typename Functor>
  static void for_each_target_with_neighbors(const neighbor_list_type& list, const Functor& functor) {
    execution_space exec_space{};
    for_each_target_with_neighbors(exec_space, list, functor);
  }

  /// \brief Run a callback for every target and its neighbor range using a provided execution space.
  /// \param exec_space [in] Execution space used for the target-parallel loop.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] Callback callable with `TargetNeighbors<neighbor_list_type>`.
  template <typename ExecutionSpace, typename Functor>
  static void for_each_target_with_neighbors(const ExecutionSpace& exec_space, const neighbor_list_type& list,
                                             const Functor& functor) {
    static_assert(has_dense_target_ordinals,
                  "NeighborList::for_each_target_with_neighbors requires dense target ordinals for the outer iteration "
                  "space.");
    using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    impl::DeployFunctorOnTargetNeighbors<neighbor_list_type, Functor> deploy_functor(list, functor);
    Kokkos::parallel_for("mundy::mesh::NeighborList::for_each_target_with_neighbors",
                         range_policy_t(exec_space, 0, list.num_targets()), deploy_functor);
  }
  //@}
};

/// \class ArborX1dNeighborList
/// \brief ArborX neighbor list with Cabana-style compressed 1D storage.
///
/// This implementation stores a source-index array and target offsets. Targets and sources are currently addressed by
/// contiguous ordinals, but the generic `NeighborList` facade still goes through `NeighborListTraits` so future
/// non-contiguous lists do not inherit this assumption.
/// \tparam NeighborTag Neighbor-list semantic tag (`FullNeighborTag`, `HalfNeighborTag`, or `SelfNeighborTag`).
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
class ArborX1dNeighborList {
 public:
  //! \name Aliases
  //@{

  using neighbor_tag = NeighborTag;
  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using target_index_type = size_type;
  using source_index_type = size_type;
  using search_boxes_type = impl::ArborXSearchBoxesT<memory_space>;
  using entity_map_type = typename search_boxes_type::entity_map_type;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;

  static constexpr bool has_contiguous_targets = true;
  static constexpr bool has_contiguous_sources = true;
  static constexpr bool has_dense_target_ordinals = true;
  static constexpr bool is_contiguous = has_contiguous_targets && has_contiguous_sources;
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
  /// \param targets [in] Target search boxes and metadata.
  /// \param sources [in] Source search boxes and metadata.
  /// \param source_indices [in] Contiguous source handles for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  ArborX1dNeighborList(const search_boxes_type& targets, const search_boxes_type& sources,
                       const source_index_view_t& source_indices, const offset_view_t& offsets)
      : targets_(targets), sources_(sources), source_indices_(source_indices), offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == targets_.size() + 1, std::invalid_argument,
                       "ArborX1dNeighborList: offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return targets_.size();
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return sources_.size();
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_indices_.extent(0);
  }

  /// \brief Convert a target ordinal to a target handle.
  /// \param target_ordinal [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index(size_type target_ordinal) const {
    MUNDY_THROW_ASSERT(target_ordinal < num_targets(), std::out_of_range,
                       "ArborX1dNeighborList::target_index target ordinal out of range.");
    return target_ordinal;
  }

  /// \brief Get the number of neighbors for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(target_index_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX1dNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the compact pair index for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(target_index_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }

  /// \brief Get the implementation storage slot for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type storage_index(target_index_type target_index, size_type neighbor_ordinal) const {
    return pair_index(target_index, neighbor_ordinal);
  }

  /// \brief Get the source handle for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(target_index_type target_index, size_type neighbor_ordinal) const {
    return source_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity neighbor_entity(target_index_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(target_index_type target_index) const {
    return targets_.entities().entity(target_index);
  }

  /// \brief Get the source entity for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    return sources_.entities().entity(source_index);
  }

  /// \brief Get the target fast mesh index for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex target_fast_mesh_index(target_index_type target_index) const {
    return targets_.entities().fast_mesh_index(target_index);
  }

  /// \brief Get the source fast mesh index for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex source_fast_mesh_index(source_index_type source_index) const {
    return sources_.entities().fast_mesh_index(source_index);
  }

  /// \brief Get the target search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& targets() const noexcept {
    return targets_;
  }

  /// \brief Get the source search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& sources() const noexcept {
    return sources_;
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

  //! Target boxes and entity metadata.
  search_boxes_type targets_;
  //! Source boxes and entity metadata.
  search_boxes_type sources_;
  //! Flattened source handles for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;
  //@}
};

/// \class ArborX2dNeighborList
/// \brief ArborX neighbor list with Cabana-style dense 2D per-target storage.
///
/// This implementation stores a per-target neighbor count and a fixed-width row of source handles for every target. It
/// is most natural when neighbor counts are relatively uniform and pair iteration can tolerate row-based indexing.
/// \tparam NeighborTag Neighbor-list semantic tag (`FullNeighborTag`, `HalfNeighborTag`, or `SelfNeighborTag`).
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
class ArborX2dNeighborList {
 public:
  //! \name Aliases
  //@{

  using neighbor_tag = NeighborTag;
  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using target_index_type = size_type;
  using source_index_type = size_type;
  using search_boxes_type = impl::ArborXSearchBoxesT<memory_space>;
  using entity_map_type = typename search_boxes_type::entity_map_type;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type**, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;

  static constexpr bool has_contiguous_targets = true;
  static constexpr bool has_contiguous_sources = true;
  static constexpr bool has_dense_target_ordinals = true;
  static constexpr bool is_contiguous = has_contiguous_targets && has_contiguous_sources;
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
  /// \param targets [in] Target search boxes and metadata.
  /// \param sources [in] Source search boxes and metadata.
  /// \param neighbor_counts [in] Number of valid entries in each target row.
  /// \param source_indices [in] Dense target-by-neighbor source handle view.
  /// \param pair_offsets [in] Compact pair offsets for each target; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  ArborX2dNeighborList(const search_boxes_type& targets, const search_boxes_type& sources,
                       const count_view_t& neighbor_counts, const source_index_view_t& source_indices,
                       const offset_view_t& pair_offsets)
      : targets_(targets),
        sources_(sources),
        neighbor_counts_(neighbor_counts),
        source_indices_(source_indices),
        pair_offsets_(pair_offsets) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == targets_.size(), std::invalid_argument,
                       "ArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_indices_.extent(0) == targets_.size(), std::invalid_argument,
                       "ArborX2dNeighborList: source_indices row extent must equal num_targets.");
    MUNDY_THROW_ASSERT(pair_offsets_.extent(0) == targets_.size() + 1, std::invalid_argument,
                       "ArborX2dNeighborList: pair_offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return targets_.size();
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return sources_.size();
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return pair_offsets_(num_targets());
  }

  /// \brief Get the allocated row width for each target.
  KOKKOS_INLINE_FUNCTION
  size_type max_neighbors_per_target() const noexcept {
    return source_indices_.extent(1);
  }

  /// \brief Convert a target ordinal to a target handle.
  /// \param target_ordinal [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index(size_type target_ordinal) const {
    MUNDY_THROW_ASSERT(target_ordinal < num_targets(), std::out_of_range,
                       "ArborX2dNeighborList::target_index target ordinal out of range.");
    return target_ordinal;
  }

  /// \brief Get the number of neighbors for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(target_index_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX2dNeighborList::num_neighbors target index out of range.");
    return neighbor_counts_(target_index);
  }

  /// \brief Get the compact pair index for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(target_index_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX2dNeighborList::pair_index neighbor ordinal out of range.");
    return pair_offsets_(target_index) + neighbor_ordinal;
  }

  /// \brief Get the dense-row storage slot for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type storage_index(target_index_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX2dNeighborList::storage_index neighbor ordinal out of range.");
    return target_index * max_neighbors_per_target() + neighbor_ordinal;
  }

  /// \brief Get the source handle for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(target_index_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX2dNeighborList::source_index neighbor ordinal out of range.");
    return source_indices_(target_index, neighbor_ordinal);
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity neighbor_entity(target_index_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(target_index_type target_index) const {
    return targets_.entities().entity(target_index);
  }

  /// \brief Get the source entity for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    return sources_.entities().entity(source_index);
  }

  /// \brief Get the target fast mesh index for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex target_fast_mesh_index(target_index_type target_index) const {
    return targets_.entities().fast_mesh_index(target_index);
  }

  /// \brief Get the source fast mesh index for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex source_fast_mesh_index(source_index_type source_index) const {
    return sources_.entities().fast_mesh_index(source_index);
  }

  /// \brief Get the target search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& targets() const noexcept {
    return targets_;
  }

  /// \brief Get the source search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& sources() const noexcept {
    return sources_;
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

  /// \brief Get the raw compact pair-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t pair_offsets() const noexcept {
    return pair_offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Target boxes and entity metadata.
  search_boxes_type targets_;
  //! Source boxes and entity metadata.
  search_boxes_type sources_;
  //! Number of valid entries in each dense target row.
  count_view_t neighbor_counts_;
  //! Dense per-target source handles; extent is `num_targets() x max_neighbors_per_target`.
  source_index_view_t source_indices_;
  //! Per-target compact pair offsets; extent is `num_targets() + 1`.
  offset_view_t pair_offsets_;
  //@}
};

/// \class STKSearchNeighborList
/// \brief STK coarse-search neighbor list mapped into Mundy's common target/source access surface.
///
/// This implementation is intended to consume STK coarse-search candidate pairs and materialize the same compressed
/// target-to-source storage shape as `ArborX1dNeighborList`. The first sketch uses contiguous ordinals for
/// target/source handles, while the generic facade remains ready for a future non-contiguous implementation.
/// \tparam NeighborTag Neighbor-list semantic tag (`FullNeighborTag`, `HalfNeighborTag`, or `SelfNeighborTag`).
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
class STKSearchNeighborList {
 public:
  //! \name Aliases
  //@{

  using neighbor_tag = NeighborTag;
  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using target_index_type = size_type;
  using source_index_type = size_type;
  using search_boxes_type = impl::STKSearchBoxesT<memory_space>;
  using entity_map_type = typename search_boxes_type::entity_map_type;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;

  static constexpr bool has_contiguous_targets = true;
  static constexpr bool has_contiguous_sources = true;
  static constexpr bool has_dense_target_ordinals = true;
  static constexpr bool is_contiguous = has_contiguous_targets && has_contiguous_sources;
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
  /// \param targets [in] Target search boxes and metadata.
  /// \param sources [in] Source search boxes and metadata.
  /// \param source_indices [in] Contiguous source handles for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  STKSearchNeighborList(const search_boxes_type& targets, const search_boxes_type& sources,
                        const source_index_view_t& source_indices, const offset_view_t& offsets)
      : targets_(targets), sources_(sources), source_indices_(source_indices), offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == targets_.size() + 1, std::invalid_argument,
                       "STKSearchNeighborList: offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return targets_.size();
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return sources_.size();
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_indices_.extent(0);
  }

  /// \brief Convert a target ordinal to a target handle.
  /// \param target_ordinal [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  target_index_type target_index(size_type target_ordinal) const {
    MUNDY_THROW_ASSERT(target_ordinal < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::target_index target ordinal out of range.");
    return target_ordinal;
  }

  /// \brief Get the number of neighbors for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(target_index_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the compact pair index for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(target_index_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "STKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }

  /// \brief Get the implementation storage slot for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type storage_index(target_index_type target_index, size_type neighbor_ordinal) const {
    return pair_index(target_index, neighbor_ordinal);
  }

  /// \brief Get the source handle for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(target_index_type target_index, size_type neighbor_ordinal) const {
    return source_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Target handle.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity neighbor_entity(target_index_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(target_index_type target_index) const {
    return targets_.entities().entity(target_index);
  }

  /// \brief Get the source entity for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    return sources_.entities().entity(source_index);
  }

  /// \brief Get the target fast mesh index for a target handle.
  /// \param target_index [in] Target handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex target_fast_mesh_index(target_index_type target_index) const {
    return targets_.entities().fast_mesh_index(target_index);
  }

  /// \brief Get the source fast mesh index for a source handle.
  /// \param source_index [in] Source handle.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex source_fast_mesh_index(source_index_type source_index) const {
    return sources_.entities().fast_mesh_index(source_index);
  }

  /// \brief Get the target search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& targets() const noexcept {
    return targets_;
  }

  /// \brief Get the source search boxes and metadata.
  KOKKOS_INLINE_FUNCTION
  const search_boxes_type& sources() const noexcept {
    return sources_;
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

  //! Target boxes and entity metadata.
  search_boxes_type targets_;
  //! Source boxes and entity metadata.
  search_boxes_type sources_;
  //! Flattened source handles produced by grouping STK search pairs by target.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;
  //@}
};

//! \name Factory sketches
//@{

/// \brief Build a compressed 1D ArborX neighbor list from target and source search boxes.
///
/// This host-side factory owns search orchestration so `ArborX1dNeighborList` can remain a small device-facing storage
/// view. The current body intentionally returns an empty, structurally valid list while the ArborX query path is being
/// designed.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity metadata.
/// \param target_selector [in] Host-side target selector, used only to choose half-list overlap policy.
/// \param sources [in] Source search boxes and entity metadata.
/// \param source_selector [in] Host-side source selector, used only to choose half-list overlap policy.
/// \param buffer_size [in] Optional ArborX traversal buffer-size hint.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
ArborX1dNeighborList<NeighborTag, MemorySpace> make_arborx_1d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::ArborXSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::ArborXSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector, int buffer_size = 0) {
  using list_type = ArborX1dNeighborList<NeighborTag, MemorySpace>;
  typename list_type::source_index_view_t source_indices("mundy_arborx_1d_source_indices", 0);
  typename list_type::offset_view_t offsets("mundy_arborx_1d_offsets", targets.size() + 1);
  Kokkos::deep_copy(exec_space, offsets, 0);

  // TODO(palmerb4): Follow Cabana::Experimental::makeNeighborList:
  // 1. Build an ArborX BVH over source boxes.
  // 2. Query target boxes with a callback selected by NeighborTag.
  // 3. Fill source_indices and offsets in compressed 1D form.
  // 4. Keep selector-overlap classification in this host-side factory so hot accessors do not branch on selectors.
  // 5. Use a stateful half-list filter for identical, disjoint, and partially overlapping selectors.
  (void)target_selector;
  (void)source_selector;
  (void)buffer_size;
  return list_type(targets, sources, source_indices, offsets);
}

/// \brief Build a dense 2D ArborX neighbor list from target and source search boxes.
///
/// This host-side factory owns the two-pass ArborX count/fill sequence. The current body intentionally returns an
/// empty, structurally valid list while the query path is being designed.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity metadata.
/// \param target_selector [in] Host-side target selector, used only to choose half-list overlap policy.
/// \param sources [in] Source search boxes and entity metadata.
/// \param source_selector [in] Host-side source selector, used only to choose half-list overlap policy.
/// \param buffer_size [in] Optional maximum-neighbor preallocation guess.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
ArborX2dNeighborList<NeighborTag, MemorySpace> make_arborx_2d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::ArborXSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::ArborXSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector, int buffer_size = 0) {
  using list_type = ArborX2dNeighborList<NeighborTag, MemorySpace>;
  typename list_type::count_view_t neighbor_counts("mundy_arborx_2d_neighbor_counts", targets.size());
  typename list_type::source_index_view_t source_indices("mundy_arborx_2d_source_indices", targets.size(), 0);
  typename list_type::offset_view_t pair_offsets("mundy_arborx_2d_pair_offsets", targets.size() + 1);
  Kokkos::deep_copy(exec_space, neighbor_counts, 0);
  Kokkos::deep_copy(exec_space, pair_offsets, 0);

  // TODO(palmerb4): Follow Cabana::Experimental::make2DNeighborList:
  // 1. Build an ArborX BVH over source boxes.
  // 2. First pass counts candidates per target, optionally filling a user-provided buffer.
  // 3. Reduce to max neighbors, allocate dense rows, reset counts, and refill.
  // 4. Build pair_offsets from neighbor_counts so pair_index remains compact and storage_index remains dense-row.
  // 5. Reuse the same stateful NeighborTag filtering policy as the 1D factory.
  (void)target_selector;
  (void)source_selector;
  (void)buffer_size;
  return list_type(targets, sources, neighbor_counts, source_indices, pair_offsets);
}

/// \brief Build an STK coarse-search neighbor list from target and source search boxes.
///
/// This host-side factory owns `stk::search::coarse_search` setup and grouping so `STKSearchNeighborList` can remain a
/// compressed device-facing storage view. The current body intentionally returns an empty, structurally valid list
/// while the STK search path is being designed.
/// \tparam ExecutionSpace Execution-space tag associated with search preparation.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space associated with search preparation.
/// \param targets [in] Target search boxes and entity metadata.
/// \param target_selector [in] Host-side target selector, used only to choose half-list overlap policy.
/// \param sources [in] Source search boxes and entity metadata.
/// \param source_selector [in] Host-side source selector, used only to choose half-list overlap policy.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
STKSearchNeighborList<NeighborTag, MemorySpace> make_stk_search_neighbor_list(
    const ExecutionSpace& exec_space, const impl::STKSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::STKSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector) {
  using list_type = STKSearchNeighborList<NeighborTag, MemorySpace>;
  typename list_type::source_index_view_t source_indices("mundy_stk_search_source_indices", 0);
  typename list_type::offset_view_t offsets("mundy_stk_search_offsets", targets.size() + 1);
  Kokkos::deep_copy(exec_space, offsets, 0);

  // TODO(palmerb4): Use STK coarse_search to get candidate pairs, then map those pairs back into this implementation's
  // target/source handles:
  // 1. Produce STK search boxes on the host or a supported execution space.
  // 2. Run coarse_search with target/source ident-proc payloads carrying the list handle type.
  // 3. Apply the stateful NeighborTag filter selected by target/source selector overlap.
  // 4. Sort/group by target index and materialize offsets plus source_indices.
  (void)target_selector;
  (void)source_selector;
  return list_type(targets, sources, source_indices, offsets);
}

//@}

}  // namespace mesh

}  // namespace mundy

namespace ArborX {

/// \struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many source primitives exist and how to fetch the `ArborX::Box` for each source
/// ordinal. The specialization must live in namespace `ArborX`.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PrimitivesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of primitives.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive box for a source ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Source ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes,
                                         size_type index) {
    return boxes.box(index);
  }
};

/// \struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many target predicates exist and how to convert each target box into an
/// intersection predicate. The attached data is the dense target ordinal used by the first-pass ArborX implementations.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of predicates.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Target ordinal to attach as predicate data.
  static KOKKOS_FUNCTION auto get(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes, size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

}  // namespace ArborX

#endif  // MUNDY_MESH_NEIGHBORLIST_HPP_
