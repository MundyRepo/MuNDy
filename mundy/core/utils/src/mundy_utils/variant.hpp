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

#ifndef MUNDY_UTILS_VARIANT_HPP_
#define MUNDY_UTILS_VARIANT_HPP_

// C++ core
#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>         // for mundy::tuple
#include <mundy_utils/type_traits.hpp>   // for mundy::index_finder, contains_type

namespace mundy {

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
    MUNDY_THROW_ASSERT(holds_alternative<T>(), std::runtime_error, "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }
  template <class T>
  KOKKOS_FUNCTION constexpr const T& get() const {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    MUNDY_THROW_ASSERT(holds_alternative<T>(), std::runtime_error, "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }

  /// \brief Get the value of the active type based on the active index
  template <size_t ActiveIdx>
  KOKKOS_FUNCTION constexpr auto get() -> alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    MUNDY_THROW_ASSERT(holds_alternative<Alt>(), std::runtime_error, "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }
  template <size_t ActiveIdx>
  KOKKOS_FUNCTION constexpr auto get() const -> const alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    MUNDY_THROW_ASSERT(holds_alternative<Alt>(), std::runtime_error, "Incorrect type access using active index");
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

namespace impl {

template <class Visitor, class Variant, size_t ActiveIdx>
using visit_result_t = decltype(std::declval<Visitor>()(get<ActiveIdx>(std::declval<Variant>())));

template <class Visitor, class Variant, size_t... Ids>
KOKKOS_FUNCTION constexpr bool visit_has_homogeneous_return_impl(std::index_sequence<Ids...>) {
  using FirstReturnType = visit_result_t<Visitor, Variant, 0>;
  return (std::is_same_v<FirstReturnType, visit_result_t<Visitor, Variant, Ids>> && ...);
}

template <class ReturnType>
KOKKOS_FUNCTION constexpr ReturnType unreachable_visit_return() {
  if constexpr (std::is_void_v<ReturnType>) {
    return;
  } else {
    using ValueType = std::remove_reference_t<ReturnType>;
    return *static_cast<ValueType*>(nullptr);
  }
}

template <size_t ActiveIdx, size_t NumAlts, class ReturnType, class Visitor, class Variant>
KOKKOS_FUNCTION constexpr ReturnType visit_dispatch(Visitor&& visitor, Variant&& var) {
  if constexpr (ActiveIdx < NumAlts) {
    if (var.index() == ActiveIdx) {
      return static_cast<Visitor&&>(visitor)(get<ActiveIdx>(static_cast<Variant&&>(var)));
    }
    return visit_dispatch<ActiveIdx + 1, NumAlts, ReturnType>(static_cast<Visitor&&>(visitor),
                                                              static_cast<Variant&&>(var));
  } else {
    MUNDY_THROW_ASSERT(false, std::runtime_error, "Invalid variant index in visit");
    return unreachable_visit_return<ReturnType>();
  }
}

}  // namespace impl

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

/// \brief Visit the active value in the variant
template <class Visitor, class... Alts>
KOKKOS_FUNCTION constexpr decltype(auto) visit(Visitor&& visitor, variant<Alts...>& var) {
  static_assert(sizeof...(Alts) > 0, "variant must have at least one alternative.");
  using VariantRef = variant<Alts...>&;
  using ReturnType = impl::visit_result_t<Visitor&&, VariantRef, 0>;
  static_assert(
      impl::visit_has_homogeneous_return_impl<Visitor&&, VariantRef>(std::make_index_sequence<sizeof...(Alts)>{}),
      "Visitor return type must be the same for all alternatives.");
  return impl::visit_dispatch<0, sizeof...(Alts), ReturnType>(static_cast<Visitor&&>(visitor), var);
}

/// \brief Visit the active value in the variant (const overload)
template <class Visitor, class... Alts>
KOKKOS_FUNCTION constexpr decltype(auto) visit(Visitor&& visitor, const variant<Alts...>& var) {
  static_assert(sizeof...(Alts) > 0, "variant must have at least one alternative.");
  using VariantRef = const variant<Alts...>&;
  using ReturnType = impl::visit_result_t<Visitor&&, VariantRef, 0>;
  static_assert(
      impl::visit_has_homogeneous_return_impl<Visitor&&, VariantRef>(std::make_index_sequence<sizeof...(Alts)>{}),
      "Visitor return type must be the same for all alternatives.");
  return impl::visit_dispatch<0, sizeof...(Alts), ReturnType>(static_cast<Visitor&&>(visitor), var);
}

// -------- variant_size
template <class T>
struct variant_size;  // primary

template <class... Alts>
struct variant_size<variant<Alts...>> {
  static constexpr size_t value = sizeof...(Alts);
};

template <class T>
static constexpr size_t variant_size_v = variant_size<T>::value;

//@}

}  // namespace mundy

#endif  // MUNDY_UTILS_VARIANT_HPP_
