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
#include <mundy_math/Accessor.hpp>  // for mundy::ValidAccessor
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \brief Get a strided accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a contiguous accessor with a stride between elements but without copying the
/// underlying data. This class provides a way to do that.
///
/// \tparam T The type of the elements
/// \tparam stride The stride between elements
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t stride, ValidAccessor<T> Accessor>
class StridedView {
 public:
  storage<Accessor> accessor_;

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr StridedView(const Accessor& accessor)
    MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr StridedView(Accessor&& accessor)
    MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(accessor)) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedView(const StridedView<T, stride, Accessor>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr StridedView(StridedView<T, stride, Accessor>&& other) : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    return impl::access_at(accessor_, idx * stride);
  }
  //
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    return impl::access_at(accessor_, idx * stride);
  }
};  // class StridedView

//! \name StridedView views
//@{

/// \brief A helper function to create a StridedView<T, stride, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a StridedView<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   StridedView<T, stride, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto strided_data = get_strided_view<T, stride>(data);
/// \endcode
template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_strided_view(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::forward<Accessor>(data)));
  return StridedView<T, stride, decltype(data_storage)>(data_storage);
}

template <typename T, size_t stride, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_strided_accessor(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::move(data)));
  return StridedView<T, stride, decltype(data_storage)>(data_storage);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_STRIDEDVIEW_HPP_
