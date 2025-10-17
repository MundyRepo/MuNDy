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

template <class... Alts>
struct variant {
 private:
  static_assert((std::is_copy_assignable_v<Alts> && ...), "All types must be copy assignable.");
  static_assert((std::is_default_constructible_v<Alts> && ...), "All types must be default constructible.");
  tuple<Alts...> storage_;
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
    reset_active_type_impl(std::make_index_sequence<sizeof...(Alts)>{});
  }
  //@}

 public:
  /// \brief Default constructor initializes the first type as active
  KOKKOS_FUNCTION constexpr variant() : storage_{}, active_index_{0} {
  }

  /// \brief Constructor for initializing with a specific type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  KOKKOS_FUNCTION constexpr variant(const T& value) : storage_{}, active_index_{index_of<T>()} {
    storage_.template get<T>() = value;
  }

  /// \brief Get the active type index
  KOKKOS_FUNCTION constexpr size_t index() const {
    return active_index_;
  }

  /// \brief Get the number of alternatives
  KOKKOS_FUNCTION static constexpr size_t size() {
    return sizeof...(Alts);
  }

  template <class T>
  KOKKOS_FUNCTION static constexpr size_t index_of() {
    return index_finder_v<T, Alts...>;
  }

  /// \brief Check if a specific type is active
  template <class T>
  KOKKOS_FUNCTION constexpr bool holds_alternative() const {
    return active_index_ == index_of<T>();
  }

  /// \brief The J'th alternative type
  template <size_t J>
  using alternative_t = type_at_index_t<J, Alts...>;

  /// \brief Get the value of the active type
  template <class T>
  KOKKOS_FUNCTION constexpr T& get() {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }
  template <class T>
  KOKKOS_FUNCTION constexpr const T& get() const {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }

  /// \brief Get the value of the active type based on the active index
  template <size_t ActiveIdx>
  KOKKOS_FUNCTION constexpr auto get() -> alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }
  template <size_t ActiveIdx>
  KOKKOS_FUNCTION constexpr auto get() const -> const alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }


  /// \brief Set a new active type, default-constructing the previous type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  KOKKOS_FUNCTION constexpr void operator=(T const& value) {
    reset_active_type();
    active_index_ = index_of<T>();
    storage_.template get<T>() = value;
  }
};

//! \name Non-member functions
//@{

/// \brief Get the index of the given type
template <class T, class... Alts>
constexpr size_t index_of() {
  return variant<Alts...>::template index_of<T>();
}

/// \brief Check if a specific type is active
template <class T, class... Alts>
KOKKOS_FUNCTION constexpr bool holds_alternative(const variant<Alts...>& var) {
  return var.template holds_alternative<T>();
}

/// \brief Get the J'th alternative type TODO(palmerb4): Make independent of concrete variant instance
template <size_t J, class VariantType>
using variant_alternative_t = typename VariantType::template alternative_t<J>;

/// \brief Get the value of the active type
template <class T, class... Alts>
KOKKOS_FUNCTION constexpr T& get(variant<Alts...>& var) {
  return var.template get<T>();
}
template <class T, class... Alts>
KOKKOS_FUNCTION constexpr const T& get(const variant<Alts...>& var) {
  return var.template get<T>();
}

  /// \brief Get the value of the active type based on the active index
template <size_t ActiveIdx, class... Alts>
KOKKOS_FUNCTION constexpr auto& get(variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}
template <size_t ActiveIdx, class... Alts>
KOKKOS_FUNCTION constexpr const auto& get(const variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}

// -------- variant_size
template<class T> struct variant_size; // primary

template<class... Alts>
struct variant_size<variant<Alts...>> {
  static constexpr std::size_t value = sizeof...(Alts);
};

template<class T>
static constexpr std::size_t variant_size_v = variant_size<T>::value;

//@}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_VARIANT_HPP_
