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

#ifndef MUNDY_SEARCH_IMPL_SEARCHBOXES_HPP_
#define MUNDY_SEARCH_IMPL_SEARCHBOXES_HPP_

/// \file impl/SearchBoxes.hpp
/// \brief Backend-neutral build-time search boxes paired with a per-element identity.

// C++ core
#include <cstddef>  // for size_t

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>        // for mundy::Vector3
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

namespace impl {

/// \struct PeriodicImageIdentity
/// \brief The identity of a periodic search element: the owner it images, plus the lattice shift that produced it.
///
/// A periodic image is not a mesh entity. Its identity is the owner it is an image of together with the lattice
/// translation applied to that owner's geometry. The owner handle is the template parameter so each backend carries
/// the owner in its native form — a local `stk::mesh::Entity` for a single-rank ArborX search, a global
/// `stk::mesh::EntityKey` for a distributed STK coarse search that must survive ghosting.
/// \tparam Owner Owner handle type (e.g. `stk::mesh::Entity` or `stk::mesh::EntityKey`).
/// \tparam ShiftScalar Scalar type of the lattice shift vector.
template <typename Owner, typename ShiftScalar>
struct PeriodicImageIdentity {
  //! \name Aliases
  //@{

  using owner_type = Owner;
  using shift_scalar = ShiftScalar;
  using shift_type = mundy::Vector3<ShiftScalar>;
  //@}

  //! \name Members
  //@{

  //! Owner this element is a periodic image of.
  Owner owner;
  //! Lattice translation from the owner's original reference point to this image's reference point.
  shift_type shift;
  //@}

  //! \name Comparison (lexicographic on owner then shift; required when used as an STK coarse-search identity)
  //@{

  KOKKOS_INLINE_FUNCTION bool operator==(const PeriodicImageIdentity& other) const {
    return owner == other.owner && shift[0] == other.shift[0] && shift[1] == other.shift[1] &&
           shift[2] == other.shift[2];
  }

  KOKKOS_INLINE_FUNCTION bool operator!=(const PeriodicImageIdentity& other) const {
    return !(*this == other);
  }

  KOKKOS_INLINE_FUNCTION bool operator<(const PeriodicImageIdentity& other) const {
    if (owner != other.owner) {
      return owner < other.owner;
    }
    if (shift[0] != other.shift[0]) {
      return shift[0] < other.shift[0];
    }
    if (shift[1] != other.shift[1]) {
      return shift[1] < other.shift[1];
    }
    return shift[2] < other.shift[2];
  }
  //@}
};

/// \class SearchBoxes
/// \brief Build-time search boxes paired one-to-one with a per-element identity.
///
/// This is the single build-time input shape shared by every backend/periodicity combination: a selector, one box
/// per search element, and a matching identity per element. The box primitive and the identity payload are template
/// parameters, so the backends differ only in their instantiation:
///   - non-periodic: the identity is the owner handle itself (`stk::mesh::Entity` for ArborX, `stk::mesh::EntityKey`
///     for STK);
///   - periodic: the identity is a `PeriodicImageIdentity<Owner, ShiftScalar>`.
///
/// It is a construction detail, never the storage model of the final neighbor list (which keeps owner entities and
/// neighbor indices). The per-owner dense ordinal a final list indexes by is recovered by the build from the
/// identity, not stored here.
/// \tparam MemorySpace Kokkos memory space in which the boxes and identities live.
/// \tparam BoxType Backend box primitive (e.g. `ArborX::Box`, `stk::search::Box<float>`).
/// \tparam Identity Per-element identity payload (an owner handle, or a `PeriodicImageIdentity`).
template <typename MemorySpace, typename BoxType, typename Identity>
class SearchBoxes {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using box_type = BoxType;
  using identity_type = Identity;
  using size_type = size_t;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using identity_view_t = Kokkos::View<identity_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  SearchBoxes() = default;

  /// \brief Default copy and move constructors/operators.
  SearchBoxes(const SearchBoxes&) = default;
  SearchBoxes(SearchBoxes&&) = default;
  SearchBoxes& operator=(const SearchBoxes&) = default;
  SearchBoxes& operator=(SearchBoxes&&) = default;

  /// \brief Construct from a selector and matching box and identity views.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] One box per search element.
  /// \param identities [in] Identity of each search element (parallel to `boxes`).
  SearchBoxes(const stk::mesh::Selector& selector, const box_view_t& boxes, const identity_view_t& identities)
      : selector_(selector), boxes_(boxes), identities_(identities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == identities_.extent(0), std::invalid_argument,
                       "SearchBoxes: boxes and identities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of search elements.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get the box of a search element.
  /// \param index [in] Local search-element ordinal.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "SearchBoxes::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get the identity of a search element.
  /// \param index [in] Local search-element ordinal.
  KOKKOS_INLINE_FUNCTION
  identity_type identity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "SearchBoxes::identity index out of range.");
    return identities_(index);
  }

  /// \brief Get the raw box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw identity view.
  KOKKOS_INLINE_FUNCTION
  identity_view_t identities() const noexcept {
    return identities_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! One box per search element.
  box_view_t boxes_;
  //! Identity of each search element, parallel to `boxes_`.
  identity_view_t identities_;
  //@}
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_SEARCHBOXES_HPP_
