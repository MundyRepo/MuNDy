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

#ifndef MUNDY_MATH_TRANSPOSEDVIEW_HPP_
#define MUNDY_MATH_TRANSPOSEDVIEW_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core libs
#include <concepts>

// Mundy
#include <mundy_math/Accessor.hpp>  // for mundy::ValidAccessor
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \brief An accessor that represents the transpose of a row-major (NxM) matrix represented by a contiguous
/// accessor
///
/// Concept: Sometimes we'd like to access the transpose of a row-major matrix represented by a contiguous accessor
/// without copying the underlying data. This class provides a way to do that.
///
/// The resulting transpose has size MxN.
///
/// \tparam T The type of the elements
/// \tparam N The number of rows in the matrix
/// \tparam M The number of columns in the matrix
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
class TransposedView {
 public:
  storage<Accessor> accessor_;

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr TransposedView(const Accessor& accessor)
    MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr TransposedView(Accessor&& accessor)
    MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(accessor)) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedView(const TransposedView<T, N, M, Accessor>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedView(TransposedView<T, N, M, Accessor>&& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    // This idx is the contiguous index into the theoretical row-major transpose. We need to convert it to the
    // row-major index of the original matrix.
    const size_t i = idx / N;
    const size_t j = idx % N;
    const size_t matrix_idx = j * M + i;
    return impl::access_at(accessor_, matrix_idx);
  }
  //
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    // This idx is the contiguous index into the theoretical row-major transpose. We need to convert it to the
    // row-major index of the original matrix.
    const size_t i = idx / N;
    const size_t j = idx % N;
    const size_t matrix_idx = j * M + i;
    return impl::access_at(accessor_, matrix_idx);
  }
};  // class TransposedView

//! \name TransposedView views
//@{

/// \brief A helper function to create a TransposedView<T, N, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a TransposedView<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   TransposedView<T, N, M, Accessor> trans(data);
/// \endcode
/// you can write
/// \code
///   auto transposed_data = get_transposed_view<T, N, M>(data);
/// \endcode
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_transposed_view(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::forward<Accessor>(data)));
  return TransposedView<T, N, M, decltype(data_storage)>(data_storage);
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_transposed_accessor(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::move(data)));
  return TransposedView<T, N, M, decltype(data_storage)>(data_storage);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_TRANSPOSEDVIEW_HPP_
