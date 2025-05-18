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

#ifndef MUNDY_MATH_STRIDEDVIEW_HPP_
#define MUNDY_MATH_STRIDEDVIEW_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core libs
#include <concepts>

// Mundy
#include <mundy_math/Accessor.hpp>  // for mundy::math::ValidAccessor

namespace mundy {

namespace math {

/// \brief Get a strided accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a contiguous accessor with a stride between elements but without copying the
/// underlying data. This class provides a way to do that.
///
/// \tparam T The type of the elements
/// \tparam stride The stride between elements
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t stride, ValidAccessor<T> Accessor, typename OwnershipType>
class StridedAccessor;

template <typename T, size_t stride, ValidAccessor<T> Accessor>
class StridedAccessor<T, stride, Accessor, Ownership::Views> {
 public:
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr StridedAccessor(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr StridedAccessor(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedAccessor(const StridedAccessor<T, stride, Accessor, Ownership::Views>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedAccessor(StridedAccessor<T, stride, Accessor, Ownership::Views>&& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr auto& operator[](size_t idx) const {
    return accessor_[idx * stride];
  }
};  // class StridedAccessor

template <typename T, size_t stride, ValidAccessor<T> Accessor>
class StridedAccessor<T, stride, Accessor, Ownership::Owns> {
 public:
  Accessor accessor_;

  /// \brief Default constructor.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr StridedAccessor()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr StridedAccessor(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedAccessor(const StridedAccessor<T, stride, Accessor, Ownership::Owns>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedAccessor(StridedAccessor<T, stride, Accessor, Ownership::Owns>&& other)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr auto& operator[](size_t idx) const {
    return accessor_[idx * stride];
  }
};  // class StridedAccessor

template <typename T, size_t stride, ValidAccessor<T> Accessor>
using StridedView = StridedAccessor<T, stride, Accessor, Ownership::Views>;

template <typename T, size_t stride, ValidAccessor<T> Accessor>
using OwningStridedAccessor = StridedAccessor<T, stride, Accessor, Ownership::Owns>;

//! \name StridedAccessor views
//@{

/// \brief A helper function to create a StridedAccessor<T, stride, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a StridedAccessor<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   StridedAccessor<T, stride, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto strided_data = get_strided_view<T, stride>(data);
/// \endcode
template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_strided_view(Accessor& data) {
  return StridedView<T, stride, Accessor>(data);
}

template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_strided_view(Accessor&& data) {
  return StridedView<T, stride, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_strided_accessor(Accessor& data) {
  return OwningStridedAccessor<T, stride, Accessor>(data);
}

template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_strided_accessor(Accessor&& data) {
  return OwningStridedAccessor<T, stride, Accessor>(std::forward<Accessor>(data));
}
//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_STRIDEDVIEW_HPP_
