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

#ifndef MUNDY_MATH_ARRAY_HPP_
#define MUNDY_MATH_ARRAY_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/impl/ArrayImpl.hpp>

namespace mundy {

namespace math {

/// \brief A simplistic array type with a fixed size and type
template <typename T, size_t N>
class Array {
 public:
  //! \name Internal data
  //@{

  /// \brief Our data
  Kokkos::Array<T, N> data_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using value_type = T;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Elements are uninitialized.
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Array() = default;

  /// \brief Constructor to initialize all elements explicitly.
  /// Requires the number of arguments to be N and the type of each to be T.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (N != 1) &&
            (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Args>>, T> && ...)
  KOKKOS_INLINE_FUNCTION constexpr explicit Array(Args&&... args) : data_{std::forward<Args>(args)...} {
  }

  // /// \brief Constructor to initialize all elements via initializer list
  KOKKOS_INLINE_FUNCTION
  constexpr Array(const std::initializer_list<T>& list)
    requires(!std::is_const_v<T>)
  {
    if (list.size() == N) {
      size_t i = 0;
      for (auto it = list.begin(); it != list.end(); ++it) {
        data_[i] = *it;
        ++i;
      }
    } else if (list.size() == 1) {
      impl::fill_impl(std::make_index_sequence<N>{}, *this, *list.begin());
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Array: Initializer list must have either 1 or N elements.");
    }
  }

  /// \brief Constructor to initialize all elements to a single value
  KOKKOS_INLINE_FUNCTION
  constexpr Array(const T& value) : Array(value, std::make_index_sequence<N>{}) {
  }

  /// \brief Default destructor
  KOKKOS_DEFAULTED_FUNCTION constexpr ~Array() = default;

  /// \brief Default copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Array(const Array<T, N>&) = default;

  /// \brief Default move constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Array(Array<T, N>&&) = default;

  /// \brief Default copy assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Array<T, N>& operator=(const Array<T, N>&) = default;

  /// \brief Default move assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Array<T, N>& operator=(Array<T, N>&&) = default;
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t idx) {
    return data_[idx];
  }

  /// \brief Const element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t idx) const {
    return data_[idx];
  }

  /// \brief Get our size
  KOKKOS_INLINE_FUNCTION
  constexpr int size() const {
    return N;
  }

  /// \brief Get a pointer to our data
  KOKKOS_INLINE_FUNCTION
  constexpr Kokkos::Array<T, N>& data() {
    return data_;
  }

  /// \brief Get a pointer to our data
  KOKKOS_INLINE_FUNCTION
  constexpr const Kokkos::Array<T, N>& data() const {
    return data_;
  }
  //@}

 private:
  //! \name Private constructors
  //@{

  /// \brief Constructor to initialize all elements to a single value using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION constexpr Array(const T& value, std::index_sequence<I...>) : data_{((void)I, value)...} {
  }

  /// \brief Deep copy constructor using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION constexpr Array(const Array<T, N>& other, std::index_sequence<I...>)
      : data_{other.data_[I]...} {
  }

  /// \brief Deep move constructor using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION constexpr Array(Array<T, N>&& other, std::index_sequence<I...>) : data_{other.data_[I]...} {
  }
  //@}
};  // Array

// Just to double check
static_assert(std::is_trivially_copyable_v<Kokkos::Array<double, 3>>);
static_assert(std::is_trivially_copyable_v<Array<double, 3>>);
static_assert(std::is_trivially_destructible_v<Array<double, 3>>);
static_assert(std::is_copy_constructible_v<Array<double, 3>>);
static_assert(std::is_move_constructible_v<Array<double, 3>>);
static_assert(std::is_copy_assignable_v<Array<double, 3>>);
static_assert(std::is_move_assignable_v<Array<double, 3>>);

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_ARRAY_HPP_
