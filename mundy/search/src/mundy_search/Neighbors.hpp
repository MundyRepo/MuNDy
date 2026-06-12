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

#ifndef MUNDY_SEARCH_NEIGHBORS_HPP_
#define MUNDY_SEARCH_NEIGHBORS_HPP_

/// \file Neighbors.hpp
/// \brief NeighborListType concept, Neighbors, and NeighborPair — the neighbor-access surface.

// C++ core
#include <concepts>  // for std::convertible_to, std::same_as
#include <cstddef>   // for size_t

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace mundy {

namespace search {

/// \concept NeighborListType
/// \brief Specifies the protocol that all Mundy neighbor-list implementations must satisfy.
///
/// Any type T that satisfies this concept can be used with for_each_neighbor_pair and
/// for_each_target_with_neighbors. Concrete list types are checked against this protocol at
/// every call site, providing better error messages than a static-facade approach.
template <typename T>
concept NeighborListType = requires {
  typename T::size_type;
  typename T::source_index_type;
  typename T::execution_space;
  typename T::memory_space;
} && requires(const T& list, typename T::size_type i, typename T::size_type j) {
  { list.num_targets() } -> std::convertible_to<typename T::size_type>;
  { list.num_sources() } -> std::convertible_to<typename T::size_type>;
  { list.size() } -> std::convertible_to<typename T::size_type>;
  { list.target_selector() } -> std::same_as<const stk::mesh::Selector&>;
  { list.source_selector() } -> std::same_as<const stk::mesh::Selector&>;
  { list.num_neighbors(i) } -> std::convertible_to<typename T::size_type>;
  { list.get_neighbor(i, j) } -> std::same_as<stk::mesh::Entity>;
  { list.target_entity(i) } -> std::same_as<stk::mesh::Entity>;
  { list.source_entity(i) } -> std::same_as<stk::mesh::Entity>;
  { list.source_index(i, j) } -> std::convertible_to<typename T::source_index_type>;
};

/// \class Neighbors
/// \brief Lightweight neighbor-range view for one target.
///
/// `Neighbors` stores the concrete list and a dense target ordinal. This deliberately keeps the first-pass interface
/// simple. Periodic concrete list types can forward relative image shifts without requiring the common range to carry
/// image state. A future non-contiguous list should introduce its own handle-aware facade when the real use case
/// appears.
///
/// The `list()` and `target_index()` accessors serve as typed escape hatches for callers that need type-specific
/// behavior not expressible through the common surface (e.g., reading image shifts from a periodic list directly).
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
  Neighbors(const neighbor_list_type& list, size_type target_index) : list_(&list), target_index_(target_index) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of neighbors for the target.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    return list_->num_neighbors(target_index_);
  }

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator[](size_type neighbor_ordinal) const {
    return list_->get_neighbor(target_index_, neighbor_ordinal);
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
    return list_->source_index(target_index_, neighbor_ordinal);
  }

  /// \brief Get the target STK entity for this neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_->target_entity(target_index_);
  }

  /// \brief Get the dense target ordinal associated with this range.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the underlying concrete neighbor list.
  ///
  /// Use this escape hatch when you need type-specific behavior not provided by the common `Neighbors` surface, such
  /// as reading image shifts directly from a periodic list type.
  KOKKOS_INLINE_FUNCTION
  const neighbor_list_type& list() const noexcept {
    return *list_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Pointer to the concrete list being viewed (never null after construction).
  const neighbor_list_type* list_ = nullptr;
  //! Dense target ordinal whose neighbor range is being viewed.
  size_type target_index_;
  //@}
};

/// \class NeighborPair
/// \brief Payload passed to pair-iteration functors.
///
/// The payload carries a dense target ordinal and a neighbor ordinal. It exposes source/target entities and source
/// ordinals, but does not expose storage internals such as compact pair ids or dense row slots. Periodic concrete list
/// types additionally provide per-object image shifts through the forwarding `target_image_shift()` /
/// `source_image_shift()` accessors; a kernel that wants the relative shift computes `source − target` itself.
///
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
      : list_(&list), target_index_(target_index), neighbor_ordinal_(neighbor_ordinal) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal for this pair.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the dense source ordinal for this pair.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index() const {
    return list_->source_index(target_index_, neighbor_ordinal_);
  }

  /// \brief Get the target STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_->target_entity(target_index_);
  }

  /// \brief Get the source STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const {
    return list_->source_entity(source_index());
  }

  /// \brief Get the target owner's image shift (original → imaged reference point).
  ///
  /// Forwards to periodic concrete list types. For non-periodic lists there is deliberately no neutral fake shift,
  /// because that would hide whether a kernel is using periodic geometry.
  KOKKOS_INLINE_FUNCTION
  auto target_image_shift() const {
    return list_->target_image_shift(target_index_);
  }

  /// \brief Get the source owner's image shift for this pair (original → imaged reference point).
  ///
  /// A kernel that wants the pairwise relative shift computes `source_image_shift() − target_image_shift()` itself.
  KOKKOS_INLINE_FUNCTION
  auto source_image_shift() const {
    return list_->source_image_shift(target_index_, neighbor_ordinal_);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Pointer to the concrete list being viewed (never null after construction).
  const neighbor_list_type* list_ = nullptr;
  //! Dense target ordinal for the pair.
  size_type target_index_;
  //! Ordinal of the source inside the target's neighbor range.
  size_type neighbor_ordinal_;
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORS_HPP_
