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
#include <stdexcept>  // for std::invalid_argument, std::out_of_range

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
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
/// pair meaningful. Self pairs are removed.
struct FullNeighborTag {};

/// \struct HalfNeighborTag
/// \brief Build a half neighbor list.
///
/// Duplicate pairs are removed when the target/source sets overlap. The exact duplicate-suppression callback is a
/// build-time concern, not part of the device-facing list object.
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
/// Concrete neighbor-list types own storage. This facade defines the common access and iteration surface used by
/// kernels and higher-level Mundy algorithms. It intentionally uses static dispatch rather than virtual functions so
/// Kokkos device code can inline through the implementation type.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborList;

namespace impl {

/// \class ArborXSearchBoxesT
/// \brief Build-time ArborX boxes paired with STK entity identities.
///
/// This object is an input to ArborX neighbor-list construction. It is not the storage model of the final neighbor
/// list. The final list stores target/source entities and neighbor indices, while search boxes remain a construction
/// detail.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity view live.
template <typename MemorySpace>
class ArborXSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using box_view_t = Kokkos::View<ArborX::Box*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
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

  /// \brief Construct ArborX search boxes from matching box and entity views.
  /// \param boxes [in] ArborX boxes used as primitives or predicates.
  /// \param entities [in] STK entity associated with each search box.
  KOKKOS_INLINE_FUNCTION
  ArborXSearchBoxesT(const box_view_t& boxes, const entity_view_t& entities) : boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "ArborXSearchBoxesT: boxes and entities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of search boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  ArborX::Box box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "ArborXSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get an entity by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity entity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "ArborXSearchBoxesT::entity index out of range.");
    return entities_(index);
  }

  /// \brief Get the raw box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t entities() const noexcept {
    return entities_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! ArborX boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

/// \class STKSearchBoxesT
/// \brief Build-time STK search boxes paired with STK entity identities.
///
/// This is the STK coarse-search counterpart to `ArborXSearchBoxesT`. It is a construction input, not persistent
/// neighbor-list storage.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity view live.
template <typename MemorySpace>
class STKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using box_type = stk::search::Box<double>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
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

  /// \brief Construct STK search boxes from matching box and entity views.
  /// \param boxes [in] STK boxes used for coarse search.
  /// \param entities [in] STK entity associated with each search box.
  KOKKOS_INLINE_FUNCTION
  STKSearchBoxesT(const box_view_t& boxes, const entity_view_t& entities) : boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "STKSearchBoxesT: boxes and entities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of search boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "STKSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get an entity by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity entity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "STKSearchBoxesT::entity index out of range.");
    return entities_(index);
  }

  /// \brief Get the raw box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t entities() const noexcept {
    return entities_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! STK boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

}  // namespace impl

/// \class Neighbors
/// \brief Lightweight neighbor-range view for one target.
///
/// `Neighbors` stores the concrete list and a dense target ordinal. This deliberately keeps the first-pass interface
/// simple. A future non-contiguous list should introduce its own handle-aware facade when the real use case appears.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class Neighbors {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using source_index_type = typename neighbor_list_type::source_index_type;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  Neighbors() = default;

  /// \brief Construct a neighbor range for a target.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  Neighbors(const neighbor_list_type& list, size_type target_index) : list_(list), target_index_(target_index) {
  }
  //@}

  //! \name Accessors
  //@{

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

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator()(size_type neighbor_ordinal) const {
    return (*this)[neighbor_ordinal];
  }

  /// \brief Get the source ordinal for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type neighbor_ordinal) const {
    return list_.source_index(target_index_, neighbor_ordinal);
  }

  /// \brief Get the dense target ordinal associated with this range.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Dense target ordinal whose neighbor range is being viewed.
  size_type target_index_;
  //@}
};

/// \class NeighborPair
/// \brief Payload passed to pair-iteration functors.
///
/// The payload carries a dense target ordinal and a neighbor ordinal. It exposes source/target entities and source
/// ordinals, but does not expose storage internals such as compact pair ids or dense row slots.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborPair {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using source_index_type = typename neighbor_list_type::source_index_type;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborPair() = default;

  /// \brief Construct a pair payload.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal of the source neighbor for the target.
  KOKKOS_INLINE_FUNCTION
  NeighborPair(const neighbor_list_type& list, size_type target_index, size_type neighbor_ordinal)
      : list_(list), target_index_(target_index), neighbor_ordinal_(neighbor_ordinal) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal for this pair.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the source ordinal within the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type neighbor_ordinal() const noexcept {
    return neighbor_ordinal_;
  }

  /// \brief Get the dense source ordinal for this pair.
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
  //@}

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Dense target ordinal for the pair.
  size_type target_index_;
  //! Ordinal of the source inside the target's neighbor range.
  size_type neighbor_ordinal_;
  //@}
};

/// \class TargetNeighbors
/// \brief Payload passed to target-with-neighbors iteration functors.
///
/// This object gives target-parallel kernels access to a target entity and its `Neighbors` range.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class TargetNeighbors {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using neighbors_type = Neighbors<neighbor_list_type>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  TargetNeighbors() = default;

  /// \brief Construct a target-neighbors payload.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  TargetNeighbors(const neighbor_list_type& list, size_type target_index) : list_(list), target_index_(target_index) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the target STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_.target_entity(target_index_);
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
  //@}

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Dense target ordinal whose neighbor range is exposed to the user functor.
  size_type target_index_;
  //@}
};

namespace impl {

/// \class DeployFunctorOnNeighborPairs
/// \brief Kokkos functor that expands target-parallel work into neighbor-pair callbacks.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `NeighborPair<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every neighbor pair.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor) : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for every neighbor of one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
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
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `TargetNeighbors<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnTargetNeighbors {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every target.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnTargetNeighbors(const NeighborListType& list, const Functor& functor)
      : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
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
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborList {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using execution_space = typename neighbor_list_type::execution_space;
  using source_index_type = typename neighbor_list_type::source_index_type;
  using neighbors_type = Neighbors<neighbor_list_type>;
  using neighbor_pair_type = NeighborPair<neighbor_list_type>;
  using target_neighbors_type = TargetNeighbors<neighbor_list_type>;
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

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  static size_type num_neighbors(const neighbor_list_type& list, size_type target_index) {
    return list.num_neighbors(target_index);
  }

  /// \brief Get the neighbor entity for a target ordinal and neighbor ordinal.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, num_neighbors(list, target_index))`.
  KOKKOS_INLINE_FUNCTION
  static stk::mesh::Entity get_neighbor(const neighbor_list_type& list, size_type target_index,
                                        size_type neighbor_ordinal) {
    return list.neighbor_entity(target_index, neighbor_ordinal);
  }

  /// \brief Get a neighbor-range view for a target ordinal.
  /// \param list [in] Concrete neighbor list.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  static neighbors_type get_neighbors(const neighbor_list_type& list, size_type target_index) {
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
/// This implementation stores target entities, source entities, a flattened source-index array, and per-target offsets.
/// Search boxes are not retained after construction.
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
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
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
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  ArborX1dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_entities_(target_entities),
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

  /// \brief Get the compact storage index for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
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
  stk::mesh::Entity neighbor_entity(size_type target_index, size_type neighbor_ordinal) const {
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

  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source ordinals for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;
  //@}
};

/// \class ArborX2dNeighborList
/// \brief ArborX neighbor list with Cabana-style dense 2D per-target storage.
///
/// This implementation stores target entities, source entities, per-target neighbor counts, and dense rows of source
/// ordinals. It does not expose compact pair ids through the generic payload.
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
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param neighbor_counts [in] Number of valid entries in each target row.
  /// \param source_indices [in] Dense target-by-neighbor source ordinal view.
  KOKKOS_INLINE_FUNCTION
  ArborX2dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const count_view_t& neighbor_counts, const source_index_view_t& source_indices)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        neighbor_counts_(neighbor_counts),
        source_indices_(source_indices) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_indices_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: source_indices row extent must equal num_targets.");
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

  /// \brief Get the total number of stored neighbor pairs.
  ///
  /// This is intentionally a linear scan for this first-pass dense layout. If callers need this frequently, store the
  /// total during construction instead of adding a generic pair-index abstraction.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    size_type total_neighbors = 0;
    for (size_type target_index = 0; target_index < num_targets(); ++target_index) {
      total_neighbors += num_neighbors(target_index);
    }
    return total_neighbors;
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
  stk::mesh::Entity neighbor_entity(size_type target_index, size_type neighbor_ordinal) const {
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

  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Number of valid entries in each dense target row.
  count_view_t neighbor_counts_;
  //! Dense per-target source ordinals; extent is `num_targets() x max_neighbors_per_target`.
  source_index_view_t source_indices_;
  //@}
};

/// \class STKSearchNeighborList
/// \brief STK coarse-search neighbor list mapped into Mundy's common access surface.
///
/// This implementation is intended to consume STK coarse-search candidate pairs and materialize the same compressed
/// target-to-source storage shape as `ArborX1dNeighborList`.
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
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  STKSearchNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                        const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_entities_(target_entities),
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

  /// \brief Get the compact storage index for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "STKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
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
  stk::mesh::Entity neighbor_entity(size_type target_index, size_type neighbor_ordinal) const {
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

  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source ordinals for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;
  //@}
};

//! \name Factory declarations
//@{

/// \brief Build a compressed 1D ArborX neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX, apply the selected neighbor
/// semantics, and return list storage containing entities plus source indices. It must not silently return an empty
/// list.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity identities.
/// \param target_selector [in] Host-side target selector used to choose duplicate-suppression policy.
/// \param sources [in] Source search boxes and entity identities.
/// \param source_selector [in] Host-side source selector used to choose duplicate-suppression policy.
/// \param buffer_size [in] Optional ArborX traversal buffer-size hint.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
ArborX1dNeighborList<NeighborTag, MemorySpace> make_arborx_1d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::ArborXSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::ArborXSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector, int buffer_size = 0);

/// \brief Build a dense 2D ArborX neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX's two-pass count/fill flow and
/// return list storage containing entities, per-target counts, and dense source rows. It must not silently return an
/// empty list.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity identities.
/// \param target_selector [in] Host-side target selector used to choose duplicate-suppression policy.
/// \param sources [in] Source search boxes and entity identities.
/// \param source_selector [in] Host-side source selector used to choose duplicate-suppression policy.
/// \param buffer_size [in] Optional maximum-neighbor preallocation guess.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
ArborX2dNeighborList<NeighborTag, MemorySpace> make_arborx_2d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::ArborXSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::ArborXSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector, int buffer_size = 0);

/// \brief Build an STK coarse-search neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run `stk::search::coarse_search`, apply the
/// selected neighbor semantics, group by target, and return compressed list storage. It must not silently return an
/// empty list.
/// \tparam ExecutionSpace Execution-space tag associated with search preparation.
/// \tparam NeighborTag Neighbor-list semantic tag.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space associated with search preparation.
/// \param targets [in] Target search boxes and entity identities.
/// \param target_selector [in] Host-side target selector used to choose duplicate-suppression policy.
/// \param sources [in] Source search boxes and entity identities.
/// \param source_selector [in] Host-side source selector used to choose duplicate-suppression policy.
template <typename ExecutionSpace, typename NeighborTag = FullNeighborTag, typename MemorySpace = stk::ngp::MemSpace>
STKSearchNeighborList<NeighborTag, MemorySpace> make_stk_search_neighbor_list(
    const ExecutionSpace& exec_space, const impl::STKSearchBoxesT<MemorySpace>& targets,
    const stk::mesh::Selector& target_selector, const impl::STKSearchBoxesT<MemorySpace>& sources,
    const stk::mesh::Selector& source_selector);

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
/// intersection predicate. The attached data is the dense target ordinal used during construction.
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
