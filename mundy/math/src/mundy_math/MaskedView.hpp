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

#ifndef MUNDY_MATH_MASKEDVIEW_HPP_
#define MUNDY_MATH_MASKEDVIEW_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core libs
#include <concepts>

// Mundy
#include <mundy_math/Accessor.hpp>  // for mundy::math::ValidAccessor

namespace mundy {

namespace math {

/// \brief Get a masked accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a subset of a contiguous accessor as through it were a contiguous but
/// without copying the underlying data. This class provides a way to do that. For example, we might want to mask-off
/// every other value.
///
/// \tparam T The type of the elements
/// \tparam N The size of the accessor
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor, typename OwnershipType>
class MaskedAccessor;

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
class MaskedAccessor<T, N, mask, Accessor, Ownership::Views> {
 private:
  static constexpr Kokkos::Array<size_t, N> create_index_array() {
    Kokkos::Array<size_t, N> indices{};
    size_t idx = 0;
    for (size_t i = 0; i < N; ++i) {
      if (mask[i]) {
        indices[idx++] = i;
      }
    }
    return indices;
  }

 public:
  //! \name Internal data
  //@{

  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;
  static constexpr Kokkos::Array<size_t, N> valid_indices_ = create_index_array();
  //@}

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedAccessor(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedAccessor(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(const MaskedAccessor<T, N, mask, Accessor, Ownership::Views>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(MaskedAccessor<T, N, mask, Accessor, Ownership::Views>&& other)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr const auto& operator[](size_t idx) const {
    return accessor_[valid_indices_[idx]];
  }
};  // class MaskedAccessor

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
class MaskedAccessor<T, N, mask, Accessor, Ownership::Owns> {
 private:
  static constexpr Kokkos::Array<size_t, N> create_index_array() {
    Kokkos::Array<size_t, N> indices{};
    size_t idx = 0;
    for (size_t i = 0; i < N; ++i) {
      if (mask[i]) {
        indices[idx++] = i;
      }
    }
    return indices;
  }

 public:
  //! \name Internal data
  //@{

  Accessor accessor_;
  static constexpr Kokkos::Array<size_t, N> valid_indices_ = create_index_array();
  //@}

  /// \brief Default constructor.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr MaskedAccessor()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] accessor The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedAccessor(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(const MaskedAccessor<T, N, mask, Accessor, Ownership::Owns>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(MaskedAccessor<T, N, mask, Accessor, Ownership::Owns>&& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr const auto& operator[](size_t idx) const {
    return accessor_[valid_indices_[idx]];
  }
};  // class MaskedAccessor

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
using MaskedView = MaskedAccessor<T, N, mask, Accessor, Ownership::Views>;

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
using OwningMaskedAccessor = MaskedAccessor<T, N, mask, Accessor, Ownership::Owns>;

//! \name MaskedAccessor views
//@{

/// \brief A helper function to create a MaskedAccessor<T, N, Accessor> based on a given accessor.
/// \param[in] accessor The accessor accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a MaskedAccessor<T, stride, Accessor> from a accessor accessor.
/// Instead of writing
/// \code
///   MaskedAccessor<T, N, mask, Accessor> vec(accessor);
/// \endcode
/// you can write
/// \code
///   auto masked_accessor = get_masked_view<T, N, mask>(accessor);
/// \endcode
template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_masked_view(Accessor& accessor) {
  return MaskedView<T, N, mask, Accessor>(accessor);
}

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_masked_view(Accessor&& accessor) {
  return MaskedView<T, N, mask, Accessor>(std::forward<Accessor>(accessor));
}

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_masked_accessor(const Accessor& accessor) {
  return OwningMaskedAccessor<T, N, mask, Accessor>(accessor);
}

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_masked_accessor(Accessor&& accessor) {
  return OwningMaskedAccessor<T, N, mask, Accessor>(std::forward<Accessor>(accessor));
}
//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MASKEDVIEW_HPP_
