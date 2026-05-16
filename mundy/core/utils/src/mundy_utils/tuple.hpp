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

#ifndef MUNDY_UTILS_TUPLE_HPP_
#define MUNDY_UTILS_TUPLE_HPP_

// C++ core
#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/type_traits.hpp>  // for count_type_v
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace impl {

/// \brief A helper class for tuple construction: holds a single element of the tuple and is tagged with its index in
/// the tuple
template <class T, size_t Idx>
struct tuple_member {
  T value;
  using value_type = T;

  /// \brief Default constructor. Only valid if T is default constructible
  KOKKOS_DEFAULTED_FUNCTION
  constexpr tuple_member()
    MUNDY_REQUIRES(std::default_initializable<T>)
  = default;

  /// \brief Constructor that takes a single argument
  KOKKOS_FUNCTION
  constexpr tuple_member(T const& val) : value(val) {
  }

  /// \brief Get the value
  KOKKOS_FUNCTION
  constexpr T& get() {
    return value;
  }
  KOKKOS_FUNCTION
  constexpr T const& get() const {
    return value;
  }
};

/// \brief Helper class which will be used via a fold expression to select the member with the correct Idx in a pack of
/// tuple_members
template <size_t SearchIdx, size_t Idx, class T>
struct tuple_idx_matcher {
  using type = tuple_member<T, Idx>;

  template <class Other>
  KOKKOS_FUNCTION constexpr auto operator|([[maybe_unused]] Other v) const {
    if constexpr (Idx == SearchIdx) {
      return *this;
    } else {
      return v;
    }
  }
};

/// \brief Helper class which will be used via a fold expression to select the member with the correct type in a pack of
/// tuple_members
template <class SearchType, size_t Idx, class T>
struct tuple_type_matcher {
  using type = tuple_member<T, Idx>;

  template <class Other>
  KOKKOS_FUNCTION constexpr auto operator|([[maybe_unused]] Other v) const {
    if constexpr (std::is_same_v<T, SearchType>) {
      return *this;
    } else {
      return v;
    }
  }
};

// **********************************************************************************************************************
/// \brief The actual tuple class
template <class IdxSeq, class... Elements>
struct tuple_impl;
//
template <size_t... Idx, class... Elements>
struct tuple_impl<std::index_sequence<Idx...>, Elements...> : public tuple_member<Elements, Idx>... {
  /// \brief Default constructor. Only valid if all elements are default constructible
  KOKKOS_DEFAULTED_FUNCTION
  constexpr tuple_impl()
    MUNDY_REQUIRES((std::default_initializable<Elements> && ...))
  = default;

  /// \brief Copy constructor from a set of values. Only valid if all elements are copy constructible
  KOKKOS_FUNCTION
  constexpr tuple_impl(Elements... vals)
    MUNDY_REQUIRES(sizeof...(Elements) > 0)
      : tuple_member<Elements, Idx>{vals}... {
  }

  /// \brief Default copy/move/assign constructors
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple_impl(const tuple_impl&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple_impl(tuple_impl&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple_impl& operator=(const tuple_impl&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple_impl& operator=(tuple_impl&&) = default;

  /// \brief Get the element of the tuple at index N
  template <size_t N>
  KOKKOS_FUNCTION constexpr auto& get() {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <size_t N>
  KOKKOS_FUNCTION constexpr const auto& get() const {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  /// \brief Get the element of the tuple with the given type T (errors if T is not unique)
  template <typename T>
  KOKKOS_FUNCTION constexpr const auto& get() const {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <typename T>
  KOKKOS_FUNCTION constexpr auto& get() {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  /// \brief Helper alias: select the matching base; sentinel ensures fold is never empty.
  template <size_t N>
    MUNDY_REQUIRES(sizeof...(Elements) > 0)
  using base_of =
      typename decltype((tuple_idx_matcher<N, Idx, Elements>() | ... | tuple_idx_matcher<N, N, void>{}))::type;
};

}  // namespace impl

/// \brief A simple std::tuple-like class that can be used in device code with similar sementics to std::tuple (e.g.,
/// get<N>(), get<T>(), tuple_cat, etc.). All elements must be copyable but not necessarily moveable or default
/// constructible. The tuple itself is copyable, moveable, and default constructible if all of its elements are
/// copyable, moveable, and default constructible (respectively).
template <class... Elements>
struct tuple : public impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  /// \brief Default constructor. Only valid if all elements are default constructible
  KOKKOS_DEFAULTED_FUNCTION
  constexpr tuple()
    MUNDY_REQUIRES((std::default_initializable<Elements> && ...))
  = default;

  /// \brief Constructor that takes a set of values. Only valid if all elements are copy constructible
  KOKKOS_FUNCTION
  constexpr tuple(Elements... vals)
    MUNDY_REQUIRES(sizeof...(Elements) > 0)
      : impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...>(vals...) {
  }

  /// \brief Default copy/move/assign constructors
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(const tuple&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple(tuple&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(const tuple&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tuple& operator=(tuple&&) = default;

  /// \brief Get the size of the tuple
  KOKKOS_FUNCTION
  static constexpr size_t size() {
    return sizeof...(Elements);
  }

  /// \brief Get the type of the N'th element
  template <size_t N>
    MUNDY_REQUIRES(sizeof...(Elements) > 0)
  using element_t = type_at_index_t<N, Elements...>;
};

/// \brief Get the I'th element of a tuple
template <size_t Idx, class... Args>
KOKKOS_FUNCTION constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<Idx>();
}
//
template <size_t Idx, class... Args>
KOKKOS_FUNCTION constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<Idx>();
}

/// \brief Get the element of a tuple with the given type T (errors if T is not unique)
template <class T, class... Args>
KOKKOS_FUNCTION constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<T>();
}
//
template <class T, class... Args>
KOKKOS_FUNCTION constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<T>();
}

// **********************************************************************************************************************
/// \brief The size of a tuple
template <class T>
struct tuple_size;  // primary
//
template <class... Es>
struct tuple_size<tuple<Es...>> {
  static constexpr size_t value = sizeof...(Es);
};
//
template <class T>
static constexpr size_t tuple_size_v = tuple_size<T>::value;

// **********************************************************************************************************************
/// \brief An element of a tuple
template <size_t I, class T>
struct tuple_element;  // primary
//
template <size_t I, class... Es>
struct tuple_element<I, tuple<Es...>> {
  static_assert(I < sizeof...(Es), "tuple_element index out of bounds");
  using type = type_at_index_t<I, Es...>;  // your existing meta util; OK with incomplete types
};
//
template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// **********************************************************************************************************************
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
/// \brief Deduction guide for tuple
template <class... Elements>
tuple(Elements...) -> tuple<Elements...>;
#endif

// **********************************************************************************************************************
// Tuple cat
namespace impl {

/// \brief Concatenate two tuples using index sequences
template <class FirstTuple, class SecondTuple, size_t... FirstIndices, size_t... SecondIndices>
KOKKOS_FUNCTION constexpr auto tuple_cat_impl(const FirstTuple& first, const SecondTuple& second,
                                              std::index_sequence<FirstIndices...>,
                                              std::index_sequence<SecondIndices...>) {
  // Extract elements from both tuples and construct the new tuple
  // This copy the elements of the tuples into the new tuple, so we remove const and ref qualifiers
  return tuple<std::decay_t<decltype(get<FirstIndices>(first))>...,
               std::decay_t<decltype(get<SecondIndices>(second))>...>{get<FirstIndices>(first)...,
                                                                      get<SecondIndices>(second)...};
}

}  // namespace impl

/// \brief Concatenate two tuples into a single tuple containing all elements of both.
template <class... FirstElements, class... SecondElements>
KOKKOS_FUNCTION constexpr auto tuple_cat(const tuple<FirstElements...>& first, const tuple<SecondElements...>& second) {
  constexpr auto first_size = sizeof...(FirstElements);
  constexpr auto second_size = sizeof...(SecondElements);

  // Generate index sequences for both tuples
  using FirstIndices = std::make_index_sequence<first_size>;
  using SecondIndices = std::make_index_sequence<second_size>;

  // Delegate to the implementation
  return impl::tuple_cat_impl(first, second, FirstIndices{}, SecondIndices{});
}

/// \brief The type of the concatenation of two tuples
template <typename... input_t>
using tuple_cat_t = decltype(tuple_cat(std::declval<input_t>()...));

/// \brief Make a tuple from a list of values.
template <class... Elements>
KOKKOS_FUNCTION constexpr auto make_tuple(Elements... vals) {
  return tuple<Elements...>{vals...};
}

}  // namespace mundy

#endif  // MUNDY_UTILS_TUPLE_HPP_
