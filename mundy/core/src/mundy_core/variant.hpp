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

#ifndef MUNDY_CORE_VARIANT_HPP_
#define MUNDY_CORE_VARIANT_HPP_

// C++ core
#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_core/tuple.hpp>  // for mundy::core::tuple
#include <mundy_core/type_traits.hpp>  // for mundy::core::index_finder, contains_type

namespace mundy {

namespace core {

template <class... Types>
struct variant {
 private:
  static_assert((std::is_copy_assignable_v<Types> && ...), "All types must be copy assignable.");
  static_assert((std::is_default_constructible_v<Types> && ...), "All types must be default constructible.");
  tuple<Types...> storage_;
  size_t active_index_;

  //! \name Helpers
  //@{

  template <size_t... Ids>
  KOKKOS_FUNCTION void reset_active_type_impl(std::index_sequence<Ids...>) {
    ((active_index_ == Ids
          ? (storage_.template get<Ids>() = std::decay_t<decltype(storage_.template get<Ids>())>{}, true)
          : false),
     ...);
  }

  // Function to reset the current active type to its default value
  KOKKOS_FUNCTION void reset_active_type() {
    reset_active_type_impl(std::make_index_sequence<sizeof...(Types)>{});
  }
  //@}

 public:
  /// \brief Default constructor initializes the first type as active
  KOKKOS_FUNCTION constexpr variant() : storage_{}, active_index_{0} {
  }

  /// \brief Constructor for initializing with a specific type
  template <class T>
    requires(contains_type_v<T, Types...>)
  KOKKOS_FUNCTION constexpr variant(T const& value) : storage_{}, active_index_{index_of<T>()} {
    storage_.template get<T>() = value;
  }

  /// \brief Get the active type index
  KOKKOS_FUNCTION constexpr size_t index() const {
    return active_index_;
  }

  template <class T>
  static constexpr size_t index_of() {
    return index_finder_v<T, Types...>;
  }

  /// \brief Check if a specific type is active
  template <class T>
  KOKKOS_FUNCTION constexpr bool holds_alternative() const {
    return active_index_ == index_of<T>();
  }

  /// \brief The J'th alternative type
  template <size_t J>
  using alternative_t = typename tuple<Types...>::template element_t<J>;

  /// \brief Get the value of the active type
  template <class T>
  KOKKOS_FUNCTION constexpr T& get() {
    static_assert(contains_type_v<T, Types...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }
  template <class T>
  KOKKOS_FUNCTION constexpr const T& get() const {
    static_assert(contains_type_v<T, Types...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }

  /// \brief Set a new active type, default-constructing the previous type
  template <class T>
    requires(contains_type_v<T, Types...>)
  KOKKOS_FUNCTION constexpr void operator=(T const& value) {
    reset_active_type();
    active_index_ = index_of<T>();
    storage_.template get<T>() = value;
  }
};

//! \name Non-member functions
//@{

/// \brief Get the index of the given type
template <class T, class... Types>
constexpr size_t index_of() {
  return variant<Types...>::template index_of<T>();
}

/// \brief Check if a specific type is active
template <class T, class... Types>
KOKKOS_FUNCTION constexpr bool holds_alternative(const variant<Types...>& var) {
  return var.template holds_alternative<T>();
}

/// \brief Get the J'th alternative type
template <size_t J, class... Types>
using variant_alternative_t = typename variant<Types...>::template alternative_t<J>;

/// \brief Get the value of the active type
template <class T, class... Types>
KOKKOS_FUNCTION constexpr T& get(variant<Types...>& var) {
  return var.template get<T>();
}
template <class T, class... Types>
KOKKOS_FUNCTION constexpr const T& get(const variant<Types...>& var) {
  return var.template get<T>();
}
//@}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_VARIANT_HPP_
