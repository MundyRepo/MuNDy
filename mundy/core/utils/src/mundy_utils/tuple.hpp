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
#include <concepts>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/requires.hpp>
#include <mundy_utils/type_traits.hpp>  // for count_type_v

namespace mundy {

namespace impl {

/// \brief Cast val to an rvalue if T is move constructible; otherwise pass it through as an lvalue so the caller's
/// construction resolves to T's copy constructor instead of selecting a deleted move constructor.
template <class T>
KOKKOS_FUNCTION constexpr decltype(auto) move_if_movable(T& val) {
  if constexpr (std::is_move_constructible_v<T>) {
    return std::move(val);
  } else {
    return (val);
  }
}

/// \brief A helper class for tuple construction: holds a single element of the tuple and is tagged with its index in
/// the tuple
template <class T, size_t Idx>
struct tuple_member {
  T value;
  using value_type = T;

  /// \brief Default constructor. Only valid if T is default constructible
  KOKKOS_DEFAULTED_FUNCTION
  constexpr tuple_member() MUNDY_REQUIRES(std::default_initializable<T>) = default;

  /// \brief Constructor that copies a single argument
  KOKKOS_FUNCTION
  constexpr tuple_member(T const& val) MUNDY_REQUIRES(std::is_copy_constructible_v<T>) : value(val) {
  }

  /// \brief Constructor that moves a single argument
  KOKKOS_FUNCTION
  constexpr tuple_member(T&& val) MUNDY_REQUIRES(std::is_move_constructible_v<T>) : value(std::move(val)) {
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
  constexpr tuple_impl() MUNDY_REQUIRES((std::default_initializable<Elements> && ...)) = default;

  /// \brief Constructor from a set of values; each is moved into its tuple_member if movable, else copied
  KOKKOS_FUNCTION
  constexpr tuple_impl(Elements... vals) MUNDY_REQUIRES(sizeof...(Elements) > 0)
      : tuple_member<Elements, Idx>{move_if_movable(vals)}... {
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
  using base_of = typename decltype((tuple_idx_matcher<N, Idx, Elements>() | ... |
                                     tuple_idx_matcher<N, N, void>{}))::type;
};

}  // namespace impl

/// \brief A simple std::tuple-like class that can be used in device code with similar semantics to std::tuple (e.g.,
/// get<N>(), get<T>(), tuple_cat, etc.). Constructing a tuple from a set of values moves each one that supports
/// moving and copies the rest; the tuple itself is copyable, movable, and default constructible if all of its
/// elements are (respectively).
template <class... Elements>
struct tuple : public impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  /// \brief Default constructor. Only valid if all elements are default constructible
  KOKKOS_DEFAULTED_FUNCTION
  constexpr tuple() MUNDY_REQUIRES((std::default_initializable<Elements> && ...)) = default;

  /// \brief Constructor that takes a set of values; each is moved into its element if movable, else copied
  KOKKOS_FUNCTION
  constexpr tuple(Elements... vals) MUNDY_REQUIRES(sizeof...(Elements) > 0)
      : impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...>(
            impl::move_if_movable(vals)...) {
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
//
template <size_t Idx, class... Args>
KOKKOS_FUNCTION constexpr auto&& get(tuple<Args...>&& vals) {
  return std::move(vals.template get<Idx>());
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

/// \brief Concatenate two tuples using index sequences. Each element is copied or moved out of its source tuple
/// depending on whether that tuple is passed as an lvalue or rvalue; std::decay_t strips the resulting reference
/// and cv qualifiers to get the new tuple's element types.
template <class FirstTuple, class SecondTuple, size_t... FirstIndices, size_t... SecondIndices>
KOKKOS_FUNCTION constexpr auto tuple_cat_impl(FirstTuple&& first, SecondTuple&& second,
                                              std::index_sequence<FirstIndices...>,
                                              std::index_sequence<SecondIndices...>) {
  return tuple<std::decay_t<decltype(get<FirstIndices>(std::forward<FirstTuple>(first)))>...,
               std::decay_t<decltype(get<SecondIndices>(std::forward<SecondTuple>(second)))>...>{
      get<FirstIndices>(std::forward<FirstTuple>(first))..., get<SecondIndices>(std::forward<SecondTuple>(second))...};
}

}  // namespace impl

/// \brief Concatenate two tuples into a single tuple containing all elements of both, copying or moving each
/// element depending on whether its source tuple is passed as an lvalue or rvalue.
template <class FirstTuple, class SecondTuple>
KOKKOS_FUNCTION constexpr auto tuple_cat(FirstTuple&& first, SecondTuple&& second) {
  constexpr auto first_size = tuple_size_v<std::remove_cvref_t<FirstTuple>>;
  constexpr auto second_size = tuple_size_v<std::remove_cvref_t<SecondTuple>>;

  using FirstIndices = std::make_index_sequence<first_size>;
  using SecondIndices = std::make_index_sequence<second_size>;

  return impl::tuple_cat_impl(std::forward<FirstTuple>(first), std::forward<SecondTuple>(second), FirstIndices{},
                              SecondIndices{});
}

/// \brief The type of the concatenation of two tuples
template <typename... input_t>
using tuple_cat_t = decltype(tuple_cat(std::declval<input_t>()...));

/// \brief Make a tuple from a list of values.
template <class... Elements>
KOKKOS_FUNCTION constexpr auto make_tuple(Elements... vals) {
  return tuple<Elements...>{impl::move_if_movable(vals)...};
}

// **********************************************************************************************************************
// Functional algorithms over tuples
namespace impl {

template <class Tuple, class F, size_t... Idx>
KOKKOS_FUNCTION constexpr void for_each_impl(Tuple&& t, F&& f, std::index_sequence<Idx...>) {
  (f(get<Idx>(t)), ...);
}

template <class Tuple, class Pred, size_t... Idx>
KOKKOS_FUNCTION constexpr bool all_of_impl(Tuple&& t, Pred&& pred, std::index_sequence<Idx...>) {
  bool result = true;
  ((result = result && pred(get<Idx>(t))), ...);
  return result;
}

template <class Tuple, class Pred, size_t... Idx>
KOKKOS_FUNCTION constexpr bool any_of_impl(Tuple&& t, Pred&& pred, std::index_sequence<Idx...>) {
  bool result = false;
  ((result = result || pred(get<Idx>(t))), ...);
  return result;
}

template <class F, class Tuple, size_t... Idx>
KOKKOS_FUNCTION constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<Idx...>) {
  return std::forward<F>(f)(get<Idx>(t)...);
}

}  // namespace impl

/// \brief Call f(element) for every element of the tuple, in order.
template <class F, class... Elements>
KOKKOS_FUNCTION constexpr void for_each(tuple<Elements...>& t, F&& f) {
  impl::for_each_impl(t, std::forward<F>(f), std::make_index_sequence<sizeof...(Elements)>{});
}
//
template <class F, class... Elements>
KOKKOS_FUNCTION constexpr void for_each(const tuple<Elements...>& t, F&& f) {
  impl::for_each_impl(t, std::forward<F>(f), std::make_index_sequence<sizeof...(Elements)>{});
}

/// \brief True if pred(element) holds for every element of the tuple (vacuously true for an empty tuple).
template <class Pred, class... Elements>
KOKKOS_FUNCTION constexpr bool all_of(const tuple<Elements...>& t, Pred&& pred) {
  return impl::all_of_impl(t, std::forward<Pred>(pred), std::make_index_sequence<sizeof...(Elements)>{});
}

/// \brief True if pred(element) holds for at least one element of the tuple (vacuously false for an empty tuple).
template <class Pred, class... Elements>
KOKKOS_FUNCTION constexpr bool any_of(const tuple<Elements...>& t, Pred&& pred) {
  return impl::any_of_impl(t, std::forward<Pred>(pred), std::make_index_sequence<sizeof...(Elements)>{});
}

/// \brief Invoke f with the tuple's elements as positional arguments: f(get<0>(t), get<1>(t), ...).
template <class F, class... Elements>
KOKKOS_FUNCTION constexpr decltype(auto) apply(F&& f, tuple<Elements...>& t) {
  return impl::apply_impl(std::forward<F>(f), t, std::make_index_sequence<sizeof...(Elements)>{});
}
//
template <class F, class... Elements>
KOKKOS_FUNCTION constexpr decltype(auto) apply(F&& f, const tuple<Elements...>& t) {
  return impl::apply_impl(std::forward<F>(f), t, std::make_index_sequence<sizeof...(Elements)>{});
}

}  // namespace mundy

#endif  // MUNDY_UTILS_TUPLE_HPP_
