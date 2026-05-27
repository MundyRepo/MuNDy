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
/// \brief Build-time STK coarse-search box wrappers.

// C++ core
#include <cstddef>  // for size_t

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_search/BoundingBox.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>        // for mundy::Vector3
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

namespace impl {

/// \class STKSearchBoxesT
/// \brief Build-time STK search boxes paired with STK entity identities.
///
/// This is the STK coarse-search counterpart to `ArborXSearchBoxesT`. It is a construction input, not persistent
/// neighbor-list storage.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity view live.
/// \tparam BoxScalar Scalar type used by the STK search boxes.
template <typename MemorySpace, typename BoxScalar = float>
class STKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using box_scalar = BoxScalar;
  using size_type = size_t;
  using box_type = stk::search::Box<box_scalar>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  STKSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  STKSearchBoxesT(const STKSearchBoxesT&) = default;
  STKSearchBoxesT(STKSearchBoxesT&&) = default;
  STKSearchBoxesT& operator=(const STKSearchBoxesT&) = default;
  STKSearchBoxesT& operator=(STKSearchBoxesT&&) = default;

  /// \brief Construct STK search boxes from a selector, matching box and entity views.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] STK boxes used for coarse search.
  /// \param entities [in] STK entity associated with each search box.
  STKSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes, const entity_view_t& entities)
      : selector_(selector), boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "STKSearchBoxesT: boxes and entities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

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

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! STK boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

/// \class PeriodicSTKSearchBoxesT
/// \brief Build-time STK search boxes for periodic images of STK owner entities.
///
/// This is the STK coarse-search counterpart to `PeriodicArborXSearchBoxesT`. It expands owner entities into image
/// boxes for construction, while preserving the owner mapping needed to collapse search results back to owner entities.
/// \tparam MemorySpace Kokkos memory space in which the image boxes and metadata live.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar = float>
class PeriodicSTKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using box_type = stk::search::Box<double>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using owner_index_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  PeriodicSTKSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  PeriodicSTKSearchBoxesT(const PeriodicSTKSearchBoxesT&) = default;
  PeriodicSTKSearchBoxesT(PeriodicSTKSearchBoxesT&&) = default;
  PeriodicSTKSearchBoxesT& operator=(const PeriodicSTKSearchBoxesT&) = default;
  PeriodicSTKSearchBoxesT& operator=(PeriodicSTKSearchBoxesT&&) = default;

  /// \brief Construct periodic image search boxes from a selector, owner entities and per-image metadata.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] STK search boxes for each periodic image.
  /// \param owner_entities [in] STK owner entities indexed by dense owner ordinal.
  /// \param owner_indices [in] Dense owner ordinal for each image box.
  /// \param image_shifts [in] Translation applied to the owner geometry for each image box.
  PeriodicSTKSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes,
                          const entity_view_t& owner_entities, const owner_index_view_t& owner_indices,
                          const image_shift_view_t& image_shifts)
      : selector_(selector),
        boxes_(boxes),
        owner_entities_(owner_entities),
        owner_indices_(owner_indices),
        image_shifts_(image_shifts) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == owner_indices_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchBoxesT: boxes and owner_indices must have the same extent.");
    MUNDY_THROW_ASSERT(boxes_.extent(0) == image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchBoxesT: boxes and image_shifts must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of periodic image boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get the number of owner entities.
  KOKKOS_INLINE_FUNCTION
  size_type num_owners() const noexcept {
    return owner_entities_.extent(0);
  }

  /// \brief Get a periodic image box by local image ordinal.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::box image index out of range.");
    return boxes_(image_index);
  }

  /// \brief Get the owner ordinal associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type owner_index(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::owner_index image index out of range.");
    return owner_indices_(image_index);
  }

  /// \brief Get an owner entity by dense owner ordinal.
  /// \param owner_index [in] Dense owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity owner_entity(size_type owner_index) const {
    MUNDY_THROW_ASSERT(owner_index < num_owners(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::owner_entity owner index out of range.");
    return owner_entities_(owner_index);
  }

  /// \brief Get the owner entity associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity image_owner_entity(size_type image_index) const {
    return owner_entity(owner_index(image_index));
  }

  /// \brief Get the shift applied to an owner to generate a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  image_shift_type image_shift(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::image_shift image index out of range.");
    return image_shifts_(image_index);
  }

  /// \brief Get the raw periodic image box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t owner_entities() const noexcept {
    return owner_entities_;
  }

  /// \brief Get the raw image-to-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  owner_index_view_t owner_indices() const noexcept {
    return owner_indices_;
  }

  /// \brief Get the raw image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t image_shifts() const noexcept {
    return image_shifts_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! STK search boxes for periodic images of owner entities.
  box_view_t boxes_;
  //! Owner entities indexed by dense owner ordinal.
  entity_view_t owner_entities_;
  //! Dense owner ordinal for each image box.
  owner_index_view_t owner_indices_;
  //! Translation applied to each owner to generate its image box.
  image_shift_view_t image_shifts_;
  //@}
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_STKSEARCHBOXES_HPP_
