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

#ifndef MUNDY_CORE_AGGREGATE_HPP_
#define MUNDY_CORE_AGGREGATE_HPP_

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

namespace mundy {

namespace core {

namespace impl {

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr const auto& find_const_component_recurse_impl(const First& first,
                                                                               const Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_const_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_const_component_impl(const core::tuple<Components...>& tuple,
                                                                 std::index_sequence<Is...>) {
  // Unpack into the
  return find_const_component_recurse_impl<Tag>(core::get<Is>(tuple)...);
}

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr auto& find_component_recurse_impl(First& first, Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_component_impl(core::tuple<Components...>& tuple,
                                                           std::index_sequence<Is...>) {
  // Unpack into the
  return find_component_recurse_impl<Tag>(core::get<Is>(tuple)...);
}

// A concept to check if a single component has a tag_type
template <typename T>
concept has_tag_type = requires { typename T::tag_type; };

template <typename T>
static constexpr bool has_tag_type_v = has_tag_type<T>;

// A concept to check if all components in a variadic list have a tag_type
template <typename... Components>
concept all_have_tags = (has_tag_type<Components> && ...);

/// \brief Helper type trait for determining if a list of tagged type contains a component with the given tag
template <typename Tag, typename... Components>
struct has_component : std::false_type {};
//
template <typename Tag, typename First, typename... Rest>
struct has_component<Tag, First, Rest...> {
  static_assert(all_have_tags<First, Rest...>, "All of the given components must have tags.");
  static constexpr bool value = std::is_same_v<typename First::tag_type, Tag> || has_component<Tag, Rest...>::value;
};
//
template <typename Tag, typename... Components>
static constexpr bool has_component_v = has_component<Tag, Components...>::value;

/// \brief Fetch the component corresponding to the given Tag (returns a const reference since the tuple is const)
template <typename Tag, typename... Components>
KOKKOS_FUNCTION static constexpr const auto& find_component(const core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  static_assert(has_component_v<Tag, Components...>,
                "Attempting to find a component that does not exist in the given tuple");
  return impl::find_const_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_FUNCTION static constexpr auto& find_component(core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  static_assert(has_component_v<Tag, Components...>,
                "Attempting to find a component that does not exist in the given tuple");
  return impl::find_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// \brief A small helper type for tying a Tag to an underlying component
template <typename Tag, typename Type>
class TaggedComponent {
 public:
  using tag_type = Tag;
  using component_type = Type;

  constexpr TaggedComponent(component_type component) : component_(component) {
  }

  /// \brief Default copy/move/assign constructors
  constexpr TaggedComponent(const TaggedComponent&) = default;
  constexpr TaggedComponent(TaggedComponent&&) = default;
  constexpr TaggedComponent& operator=(const TaggedComponent&) = default;
  constexpr TaggedComponent& operator=(TaggedComponent&&) = default;

  inline constexpr const component_type& component() const {
    // Our lifetime should be at least as long as the component's
    return component_;
  }

  inline constexpr component_type& component() {
    return component_;
  }

  component_type component_;
};  // TaggedComponent

}  // namespace impl

/// \brief An aggregate: A bag of tagged types or, more specifically, a compile-time unordered map indexed by type
///
/// They are compile-time compatable "structural types" compatable with NTTPs.
/// Their types must be default constructable and copyable.
///
/// Construct an aggregate via a fluent interface:
/// \code{.cpp}
///   auto agg = mundy::core::make_aggregate()
///       .append<Tag1>(component1)
///       .append<Tag2>(component2);
///
template <typename... Components>
class aggregate {
 public:
  static_assert(impl::all_have_tags<Components...>, "All of the given components must have tags.");
  using ComponentsTuple = core::tuple<Components...>;

  //! \name Constructors
  //@{

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr aggregate() = default;

  /// \brief Construct an aggregate that has the given components
  KOKKOS_FUNCTION
  constexpr aggregate(ComponentsTuple components)
    requires(sizeof...(Components) > 0)
      : components_(std::move(components)) {
  }

  /// \brief Default copy/move/assign constructors
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate(const aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate(aggregate&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate& operator=(const aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate& operator=(aggregate&&) = default;
  //@}

  /// \brief Add a component (fluent interface):
  template <typename Tag, typename NewComponent>
  KOKKOS_FUNCTION constexpr auto append(NewComponent new_component) const {
    auto new_tagged_comp = impl::TaggedComponent<Tag, NewComponent>{std::move(new_component)};
    auto new_tuple = core::tuple_cat(components_, core::make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = aggregate<Components..., decltype(new_tagged_comp)>;
    return NewType(new_tuple);
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION
  constexpr const auto& get() const {
    return impl::find_component<Tag>(components_).component();
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION
  constexpr auto& get() {
    return impl::find_component<Tag>(components_).component();
  }

  /// \brief Check if we have a component with the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION
  static constexpr bool has() {
    return impl::has_component_v<Tag, Components...>;
  }

  /// \brief Get the number of components in this aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t size() {
    return sizeof...(Components);
  }

  //! \name Private members (no touch)
  //@{

  ComponentsTuple components_;
  //@}
};  // aggregate



//! \name Non-member functions
//@{

/// \brief The type of aggregates is typically inferred, so this is the canonical way to construct one.
KOKKOS_INLINE_FUNCTION
constexpr auto make_aggregate() {
  return aggregate<>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION
constexpr const auto& get(const aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION
constexpr auto& get(aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Check if we have a component with the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION
constexpr bool has(const aggregate<Components...>& /*agg*/) {
  return aggregate<Components...>::template has<Tag>();
}

/// \brief Add a new component to an existing aggregate (fluent interface)
template <typename Tag, typename NewComponent, typename... Components>
KOKKOS_INLINE_FUNCTION
constexpr auto append(const aggregate<Components...>& agg, NewComponent new_component) {
  return agg.template append<Tag>(std::move(new_component));
}
//@}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_AGGREGATE_HPP_
