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

#ifndef MUNDY_MATH_SHIFTEDVIEW_HPP_
#define MUNDY_MATH_SHIFTEDVIEW_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core libs
#include <concepts>

// Mundy
#include <mundy_math/Accessor.hpp>  // for mundy::math::ValidAccessor

namespace mundy {

namespace math {

/// \brief Get a shifted accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a contiguous accessor (with only a [] operator) but with a shift. That is,
/// instead of calling accessor[i], we'd like to call accessor[i + shift]. This class provides a way to do that.
///
/// \tparam T The type of the elements
/// \tparam shift The shift in the accessor
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t shift, ValidAccessor<T> Accessor, typename OwnershipType>
class ShiftedAccessor;

template <typename T, size_t shift, ValidAccessor<T> Accessor>
class ShiftedAccessor<T, shift, Accessor, Ownership::Views> {
 public:
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr ShiftedAccessor(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr ShiftedAccessor(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedAccessor(const ShiftedAccessor<T, shift, Accessor, Ownership::Views>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedAccessor(ShiftedAccessor<T, shift, Accessor, Ownership::Views>&& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    return accessor_[idx + shift];
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    return accessor_[idx + shift];
  }
};  // class ShiftedAccessor

template <typename T, size_t shift, ValidAccessor<T> Accessor>
class ShiftedAccessor<T, shift, Accessor, Ownership::Owns> {
 public:
  Accessor accessor_;

  /// \brief Default constructor.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr ShiftedAccessor()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] accessor The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr ShiftedAccessor(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedAccessor(const ShiftedAccessor<T, shift, Accessor, Ownership::Owns>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedAccessor(ShiftedAccessor<T, shift, Accessor, Ownership::Owns>&& other)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    return accessor_[idx + shift];
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    return accessor_[idx + shift];
  }
};  // class ShiftedAccessor

template <typename T, size_t shift, ValidAccessor<T> Accessor>
using ShiftedView = ShiftedAccessor<T, shift, Accessor, Ownership::Views>;

template <typename T, size_t shift, ValidAccessor<T> Accessor>
using OwningShiftedAccessor = ShiftedAccessor<T, shift, Accessor, Ownership::Owns>;

//! \name ShiftedAccessor views
//@{

/// \brief A helper function to create a ShiftedAccessor<T, N, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a ShiftedAccessor<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   ShiftedAccessor<T, shift, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto shifted_data = get_shifted_view<T, shift>(data);
/// \endcode
template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_shifted_view(Accessor& data) {
  return ShiftedView<T, shift, Accessor>(data);
}

template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_shifted_view(Accessor&& data) {
  return ShiftedView<T, shift, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_shifted_accessor(Accessor& data) {
  return OwningShiftedAccessor<T, shift, Accessor>(data);
}

template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_shifted_accessor(Accessor&& data) {
  return OwningShiftedAccessor<T, shift, Accessor>(std::forward<Accessor>(data));
}
//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_SHIFTEDVIEW_HPP_
