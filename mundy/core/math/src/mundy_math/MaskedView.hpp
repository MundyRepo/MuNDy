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
#include <mundy_math/Accessor.hpp>  // for mundy::ValidAccessor

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
class MaskedView {
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
  static constexpr size_t map_index(size_t k) {
    constexpr size_t num_masked_elements = count_masked_elements();
    static_assert(k < num_masked_elements, "Index out of bounds for masked view");
    constexpr Kokkos::Array<size_t, N> valid_indices = create_index_array();
    return valid_indices[k];
  }

 public:
  //! \name Internal data
  //@{

  storage<Accessor> accessor_;
  //@}

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedView(const Accessor& accessor)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Constructor from a given accessor
  KOKKOS_INLINE_FUNCTION
  explicit constexpr MaskedView(Accessor&& accessor)
    requires(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(accessor)) {
  }

  /// \brief Shallow copy constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedView(const MaskedView<T, N, mask, Accessor>& other)
      : accessor_(other.accessor_) {
  }

  /// \brief Shallow move constructor.
  KOKKOS_INLINE_FUNCTION constexpr MaskedView(MaskedView<T, N, mask, Accessor>&& other)
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
};  // class MaskedView

//! \name MaskedView views
//@{

/// \brief A helper function to create a MaskedView<T, N, Accessor> based on a given accessor.
/// \param[in] accessor The accessor accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a MaskedView<T, stride, Accessor> from a accessor accessor.
/// Instead of writing
/// \code
///   MaskedView<T, N, mask, Accessor> vec(accessor);
/// \endcode
/// you can write
/// \code
///   auto masked_accessor = get_masked_view<T, N, mask>(accessor);
/// \endcode
template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_masked_view(Accessor&& accessor) {
  auto accessor_storage = store(impl::unwrap_accessor(std::forward<Accessor>(accessor)));
  return MaskedView<T, N, mask, decltype(accessor_storage)>(accessor_storage);
}

template <typename T, size_t N, Kokkos::Array<bool, N> mask, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_masked_accessor(Accessor&& accessor) {
  auto accessor_storage = store(impl::unwrap_accessor(std::move(accessor)));
  return MaskedView<T, N, mask, decltype(accessor_storage)>(accessor_storage);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_MASKEDVIEW_HPP_
