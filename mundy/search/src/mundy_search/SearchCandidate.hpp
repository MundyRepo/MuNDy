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

#ifndef MUNDY_SEARCH_SEARCHCANDIDATE_HPP_
#define MUNDY_SEARCH_SEARCHCANDIDATE_HPP_

/// \file SearchCandidate.hpp
/// \brief Candidate types passed to excluders during neighbor-list construction.

// C++ core
#include <cstddef>  // for size_t

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3

namespace mundy {

namespace search {

/// \class NeighborSearchCandidate
/// \brief Non-periodic target/source candidate passed to excluders.
///
/// Candidate objects are produced during search construction and are not final neighbor-list storage. They expose owner
/// accessors as aliases of the normal target/source accessors so excluders can use one owner-based vocabulary for both
/// periodic and non-periodic search.
/// \tparam SizeType Integral type used for local target/source ordinals.
template <typename SizeType = size_t>
class NeighborSearchCandidate {
 public:
  //! \name Aliases
  //@{

  using size_type = SizeType;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborSearchCandidate() = default;

  /// \brief Construct a non-periodic search candidate.
  /// \param target_index [in] Dense target ordinal.
  /// \param source_index [in] Dense source ordinal.
  /// \param target_entity [in] STK target entity.
  /// \param source_entity [in] STK source entity.
  KOKKOS_INLINE_FUNCTION
  NeighborSearchCandidate(size_type target_index, size_type source_index, stk::mesh::Entity target_entity,
                          stk::mesh::Entity source_entity)
      : target_index_(target_index),
        source_index_(source_index),
        target_entity_(target_entity),
        source_entity_(source_entity) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type source_index() const noexcept {
    return source_index_;
  }

  /// \brief Get the target entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const noexcept {
    return target_entity_;
  }

  /// \brief Get the source entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const noexcept {
    return source_entity_;
  }

  //@}

 private:
  //! \name Internal members
  //@{

  //! Dense target ordinal.
  size_type target_index_;
  //! Dense source ordinal.
  size_type source_index_;
  //! STK target entity.
  stk::mesh::Entity target_entity_;
  //! STK source entity.
  stk::mesh::Entity source_entity_;
  //@}
};

/// \class PeriodicNeighborSearchCandidate
/// \brief Periodic owner-pair candidate passed to excluders.
///
/// The candidate stores owner ordinals/entities and the source image shift relative to the target image shift. Images
/// are not entities; excluders should reason in terms of owner identity plus relative shift.
/// \tparam ImageShiftType Vector type used for relative image shifts.
/// \tparam SizeType Integral type used for local owner ordinals.
template <typename ImageShiftType, typename SizeType = size_t>
class PeriodicNeighborSearchCandidate {
 public:
  //! \name Aliases
  //@{

  using image_shift_type = ImageShiftType;
  using size_type = SizeType;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicNeighborSearchCandidate() = default;

  /// \brief Construct a periodic search candidate.
  /// \param target_owner_index [in] Dense target owner ordinal.
  /// \param source_owner_index [in] Dense source owner ordinal.
  /// \param target_entity [in] STK target owner entity.
  /// \param source_entity [in] STK source owner entity.
  /// \param relative_image_shift [in] Source image shift minus target image shift.
  KOKKOS_INLINE_FUNCTION
  PeriodicNeighborSearchCandidate(size_type target_owner_index, size_type source_owner_index,
                                  stk::mesh::Entity target_entity, stk::mesh::Entity source_entity,
                                  const image_shift_type& relative_image_shift)
      : target_owner_index_(target_owner_index),
        source_owner_index_(source_owner_index),
        target_entity_(target_entity),
        source_entity_(source_entity),
        relative_image_shift_(relative_image_shift) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_owner_index_;
  }

  /// \brief Get the dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type source_index() const noexcept {
    return source_owner_index_;
  }

  /// \brief Get the target owner entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const noexcept {
    return target_entity_;
  }

  /// \brief Get the source owner entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const noexcept {
    return source_entity_;
  }

  /// \brief Get the source image shift relative to the target image shift.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift() const noexcept {
    return relative_image_shift_;
  }

  //@}

 private:
  //! \name Internal members
  //@{

  //! Dense target owner ordinal.
  size_type target_owner_index_;
  //! Dense source owner ordinal.
  size_type source_owner_index_;
  //! STK target owner entity.
  stk::mesh::Entity target_entity_;
  //! STK source owner entity.
  stk::mesh::Entity source_entity_;
  //! Source image shift minus target image shift.
  image_shift_type relative_image_shift_;
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_SEARCHCANDIDATE_HPP_
