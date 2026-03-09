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
#include <mundy_math/Accessor.hpp>  // for mundy::ValidAccessor

namespace mundy {

/// \brief Get a shifted accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a contiguous accessor (with only a [] operator) but with a shift. That is,
/// instead of calling accessor[i], we'd like to call accessor[i + shift]. This class provides a way to do that.
///
/// \tparam T The type of the elements
/// \tparam shift The shift in the accessor
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t shift, ValidAccessor<T> Accessor>
class ShiftedView {
 public:
  storage<Accessor> accessor_;

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr ShiftedView(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr ShiftedView(Accessor&& accessor)
    requires(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(accessor)) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedView(const ShiftedView<T, shift, Accessor>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr ShiftedView(ShiftedView<T, shift, Accessor>&& other) : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    return impl::access_at(accessor_, idx + shift);
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    return impl::access_at(accessor_, idx + shift);
  }
};  // class ShiftedView

//! \name ShiftedView views
//@{

/// \brief A helper function to create a ShiftedView<T, N, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a ShiftedView<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   ShiftedView<T, shift, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto shifted_data = get_shifted_view<T, shift>(data);
/// \endcode
template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_shifted_view(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::forward<Accessor>(data)));
  return ShiftedView<T, shift, decltype(data_storage)>(data_storage);
}

template <typename T, size_t shift, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_shifted_accessor(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::move(data)));
  return ShiftedView<T, shift, decltype(data_storage)>(data_storage);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_SHIFTEDVIEW_HPP_
