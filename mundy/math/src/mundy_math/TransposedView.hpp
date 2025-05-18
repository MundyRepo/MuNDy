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
#include <mundy_math/Accessor.hpp>  // for mundy::math::ValidAccessor

namespace mundy {

namespace math {

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
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
class TransposedAccessor;

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
class TransposedAccessor<T, N, M, Accessor, Ownership::Views> {
 public:
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr TransposedAccessor(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr TransposedAccessor(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedAccessor(
      const TransposedAccessor<T, N, M, Accessor, Ownership::Views>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedAccessor(TransposedAccessor<T, N, M, Accessor, Ownership::Views>&& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr auto& operator[](size_t idx) const {
    // This idx is the contiguous index into the theoretical row-major transpose. We need to convert it to the
    // row-major index of the original matrix.
    const size_t i = idx / N;
    const size_t j = idx % N;
    const size_t matrix_idx = j * M + i;
    return accessor_[matrix_idx];
  }
};  // class TransposedAccessor

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
class TransposedAccessor<T, N, M, Accessor, Ownership::Owns> {
 public:
  Accessor accessor_;

  /// \brief Default constructor.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr TransposedAccessor()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr TransposedAccessor(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedAccessor(
      const TransposedAccessor<T, N, M, Accessor, Ownership::Owns>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr TransposedAccessor(TransposedAccessor<T, N, M, Accessor, Ownership::Owns>&& other)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr auto& operator[](size_t idx) const {
    // This idx is the contiguous index into the theoretical row-major transpose. We need to convert it to the
    // row-major index of the original matrix.
    const size_t i = idx / N;
    const size_t j = idx % N;
    const size_t matrix_idx = j * M + i;
    return accessor_[matrix_idx];
  }
};  // class TransposedAccessor

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
using TransposedView = TransposedAccessor<T, N, M, Accessor, Ownership::Views>;

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
using OwningTransposedAccessor = TransposedAccessor<T, N, M, Accessor, Ownership::Owns>;

//! \name TransposedAccessor views
//@{

/// \brief A helper function to create a TransposedAccessor<T, N, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a TransposedAccessor<T, stride, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   TransposedAccessor<T, N, M, Accessor> trans(data);
/// \endcode
/// you can write
/// \code
///   auto transposed_data = get_transposed_view<T, N, M>(data);
/// \endcode
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_transposed_view(Accessor& data) {
  return TransposedView<T, N, M, Accessor>(data);
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_transposed_view(Accessor&& data) {
  return TransposedView<T, N, M, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_transposed_accessor(Accessor& data) {
  return OwningTransposedAccessor<T, N, M, Accessor>(data);
}

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_transposed_accessor(Accessor&& data) {
  return OwningTransposedAccessor<T, N, M, Accessor>(std::forward<Accessor>(data));
}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_TRANSPOSEACCESSOR_HPP_
