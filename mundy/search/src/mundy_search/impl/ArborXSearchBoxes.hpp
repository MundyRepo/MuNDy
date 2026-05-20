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
/// \brief Build-time ArborX search box wrappers and their ArborX::AccessTraits specializations.

// C++ core
#include <cstddef>  // for size_t

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>        // for mundy::Vector3
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

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
  ArborXSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  ArborXSearchBoxesT(const ArborXSearchBoxesT&) = default;
  ArborXSearchBoxesT(ArborXSearchBoxesT&&) = default;
  ArborXSearchBoxesT& operator=(const ArborXSearchBoxesT&) = default;
  ArborXSearchBoxesT& operator=(ArborXSearchBoxesT&&) = default;

  /// \brief Construct ArborX search boxes from a selector, matching box and entity views.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] ArborX boxes used as primitives or predicates.
  /// \param entities [in] STK entity associated with each search box.
  ArborXSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes, const entity_view_t& entities)
      : selector_(selector), boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "ArborXSearchBoxesT: boxes and entities must have the same extent.");
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

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! ArborX boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

/// \class PeriodicArborXSearchBoxesT
/// \brief Build-time ArborX boxes for periodic images of STK owner entities.
///
/// A periodic image is not a mesh entity. Each image box stores an owner ordinal into `owner_entities_` and a shift
/// vector describing the translation applied to that owner geometry when the box was generated. ArborX sees image boxes
/// during construction; the final periodic neighbor list collapses matches back to owner ordinals and stores only the
/// relative image shift needed by pair kernels.
/// \tparam MemorySpace Kokkos memory space in which the image boxes and metadata live.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar = float>
class PeriodicArborXSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using box_view_t = Kokkos::View<ArborX::Box*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using owner_index_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  PeriodicArborXSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  PeriodicArborXSearchBoxesT(const PeriodicArborXSearchBoxesT&) = default;
  PeriodicArborXSearchBoxesT(PeriodicArborXSearchBoxesT&&) = default;
  PeriodicArborXSearchBoxesT& operator=(const PeriodicArborXSearchBoxesT&) = default;
  PeriodicArborXSearchBoxesT& operator=(PeriodicArborXSearchBoxesT&&) = default;

  /// \brief Construct periodic image search boxes from a selector, owner entities and per-image metadata.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] Search boxes for each periodic image.
  /// \param owner_entities [in] STK owner entities indexed by dense owner ordinal.
  /// \param owner_indices [in] Dense owner ordinal for each image box.
  /// \param image_shifts [in] Translation applied to the owner geometry for each image box.
  PeriodicArborXSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes,
                             const entity_view_t& owner_entities, const owner_index_view_t& owner_indices,
                             const image_shift_view_t& image_shifts)
      : selector_(selector),
        boxes_(boxes),
        owner_entities_(owner_entities),
        owner_indices_(owner_indices),
        image_shifts_(image_shifts) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == owner_indices_.extent(0), std::invalid_argument,
                       "PeriodicArborXSearchBoxesT: boxes and owner_indices must have the same extent.");
    MUNDY_THROW_ASSERT(boxes_.extent(0) == image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicArborXSearchBoxesT: boxes and image_shifts must have the same extent.");
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
  ArborX::Box box(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::box image index out of range.");
    return boxes_(image_index);
  }

  /// \brief Get the owner ordinal associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type owner_index(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::owner_index image index out of range.");
    return owner_indices_(image_index);
  }

  /// \brief Get an owner entity by dense owner ordinal.
  /// \param owner_index [in] Dense owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity owner_entity(size_type owner_index) const {
    MUNDY_THROW_ASSERT(owner_index < num_owners(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::owner_entity owner index out of range.");
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
                       "PeriodicArborXSearchBoxesT::image_shift image index out of range.");
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
  //! ArborX boxes for periodic images of owner entities.
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

namespace ArborX {

/// \struct AccessTraits<mundy::search::impl::ArborXSearchBoxesT<MemorySpace>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many source primitives exist and how to fetch the `ArborX::Box` for each source
/// ordinal. The specialization must live in namespace `ArborX`.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::search::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PrimitivesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::search::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of primitives.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::search::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive box for a source ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Source ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const mundy::search::impl::ArborXSearchBoxesT<MemorySpace>& boxes,
                                         size_type index) {
    return boxes.box(index);
  }
};

/// \struct AccessTraits<mundy::search::impl::ArborXSearchBoxesT<MemorySpace>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many target predicates exist and how to convert each target box into an
/// intersection predicate. The attached data is the dense target ordinal used during construction.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::search::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::search::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of predicates.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::search::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Target ordinal to attach as predicate data.
  static KOKKOS_FUNCTION auto get(const mundy::search::impl::ArborXSearchBoxesT<MemorySpace>& boxes,
                                  size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

/// \struct AccessTraits<mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy periodic ArborX image boxes.
///
/// This specialization exposes periodic image boxes to ArborX. The owner-entity mapping and image shifts remain in the
/// Mundy wrapper and are consumed by the neighbor-list builder after ArborX reports image-image matches.
/// \tparam MemorySpace Kokkos memory space for the Mundy periodic search boxes.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct AccessTraits<mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>
#if ARBORX_VERSION < 10799
                    ,
                    PrimitivesTag
#endif
                    > {
  //! Periodic search-box wrapper type.
  using boxes_type = mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of primitive image boxes.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive image box for an image ordinal.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  /// \param index [in] Image ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const boxes_type& boxes, size_type index) {
    return boxes.box(index);
  }
};

/// \struct AccessTraits<mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy periodic ArborX image boxes.
///
/// The attached data is the dense target image ordinal. Builders must translate that image ordinal to a target owner
/// ordinal and image shift before materializing the final periodic neighbor-list storage.
/// \tparam MemorySpace Kokkos memory space for the Mundy periodic search boxes.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct AccessTraits<mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Periodic search-box wrapper type.
  using boxes_type = mundy::search::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of predicate image boxes.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target image ordinal.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  /// \param index [in] Target image ordinal to attach as predicate data.
  static KOKKOS_FUNCTION auto get(const boxes_type& boxes, size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

}  // namespace ArborX

#endif  // MUNDY_SEARCH_IMPL_ARBORXSEARCHBOXES_HPP_
