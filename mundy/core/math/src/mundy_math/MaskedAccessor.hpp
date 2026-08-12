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

#ifndef MUNDY_MATH_MASKEDACCESSOR_HPP_
#define MUNDY_MATH_MASKEDACCESSOR_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <concepts>
#include <stdexcept>    // for std::out_of_range
#include <type_traits>  // for std::is_copy_constructible_v

// Mundy
#include <mundy_math/Accessor.hpp>  // for mundy::ValidAccessor
#include <mundy_utils/requires.hpp>

namespace mundy {

/// \brief Get a masked accessor into a contiguous accessor
///
/// Concept: Sometimes we'd like to access a subset of a contiguous accessor as through it were a contiguous but
/// without copying the underlying data. This class provides a way to do that. For example, we might want to mask-off
/// every other value.
///
/// \tparam T The type of the elements
/// \tparam N The size of the accessor
/// \tparam Accessor The type of the contiguous accessor
template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
class MaskedAccessor {
 private:
  KOKKOS_INLINE_FUNCTION
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

  KOKKOS_INLINE_FUNCTION
  static constexpr size_t count_masked_elements() {
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
      if (mask[i]) {
        ++count;
      }
    }
    return count;
  }

  /// \brief CUDA doesn't like static constexpr internal variables, so we use a constexpr variable in a static
  /// function instead
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t map_index(size_t continuous_reduced_idx) {
    constexpr size_t num_masked_elements = count_masked_elements();
    constexpr Kokkos::Array<size_t, N> valid_indices = create_index_array();
    MUNDY_THROW_ASSERT(continuous_reduced_idx < num_masked_elements, std::out_of_range,
                       "Index out of bounds for masked view");
    return valid_indices[continuous_reduced_idx];
  }

 public:
  //! \name Internal data
  //@{

  storage<Accessor> accessor_;
  //@}

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedAccessor(const Accessor& accessor) MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedAccessor(Accessor&& accessor)
      MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(accessor)) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(const MaskedAccessor<T, N, mask, Accessor>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedAccessor(MaskedAccessor<T, N, mask, Accessor>&& other)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) {
    return impl::access_at(accessor_, map_index(idx));
  }
  //
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator[](size_t idx) const {
    return impl::access_at(accessor_, map_index(idx));
  }
};  // class MaskedAccessor

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
///   auto masked_accessor = get_masked_accessor<T, N, mask>(accessor);
/// \endcode
/// \note How the accessor is held follows the argument's value category: an lvalue is referenced, so the referent must
/// outlive the result; `std::move(x)` or `own(x)` hands it over by value instead. Whether the result views or owns the
/// underlying data remains a property of the accessor, not of this function.
template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_masked_accessor(Accessor&& accessor) {
  using accessor_t = impl::stored_accessor_t<Accessor>;
  return MaskedAccessor<T, N, mask, accessor_t>(accessor_t(impl::unwrap_accessor(std::forward<Accessor>(accessor))));
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_MASKEDACCESSOR_HPP_
