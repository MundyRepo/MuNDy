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

#ifndef MUNDY_UTILS_AGGREGATE_HPP_
#define MUNDY_UTILS_AGGREGATE_HPP_

// C++ core
#include <array>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/tuple.hpp>              // for mundy::tuple
#include <mundy_utils/type_traits.hpp>        // for count_type_v
#include <mundy_utils/variant.hpp>            // for mundy::variant

namespace mundy {

/// \brief A small helper type for tying a Tag to an underlying value.
///
/// TODO(palmerb4): While currently unneeded, we could allow T to be of storage type such that we can
/// support non-owning values within an aggregate (e.g. T could be a pointer or reference wrapper). This isn't that
/// heavy of a change but would require some thought to get the storage semantics right.
template <typename Tag, typename T>
class tagged {
 public:
  using tag_type = Tag;
  using value_type = T;

  KOKKOS_INLINE_FUNCTION constexpr tagged(value_type value) : value_(std::move(value)) {
  }

  KOKKOS_DEFAULTED_FUNCTION constexpr tagged(const tagged&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tagged(tagged&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tagged& operator=(const tagged&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr tagged& operator=(tagged&&) = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const value_type& get() const {
    return value_;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type& get() {
    return value_;
  }

  value_type value_;
};

/// \brief Helper function to attach a tag to a type
template <typename Tag, typename T>
KOKKOS_INLINE_FUNCTION constexpr auto apply_tag(T&& value) {
  using ValueType = std::remove_cvref_t<T>;
  return tagged<Tag, ValueType>(std::forward<T>(value));
}

/* What all do we offer

  - has_tag_type<T>          : check if T defines an alias named tag_type
  - all_have_tags<Ts...>     : check if every type in Ts... defines tag_type
  - all_tags_unique<Ts...>   : check if every type in Ts... defines a unique tag_type
  - contains_tag<Tag, Ts...> : check if a type with tag_type == Tag is in a variadic list of types
  - index_of_tag<Tag, Ts...> : index in the variadic list of types that satisfied T::tag_type == Tag

*/

// **********************************************************************************************************************
/// \brief Check if a type defines an alias named tag_type
template <typename T, typename = void>
struct has_tag_type : std::false_type {};

template <typename T>
struct has_tag_type<T, std::void_t<typename T::tag_type>> : std::true_type {};

template <typename T>
static constexpr bool has_tag_type_v = has_tag_type<T>::value;

// **********************************************************************************************************************
/// \brief Check if all types in a pack define an alias named tag_type
template <typename... Ts>
struct all_have_tags {
  static constexpr bool value = (has_tag_type_v<Ts> && ...);
};

template <typename... Ts>
static constexpr bool all_have_tags_v = all_have_tags<Ts...>::value;

// **********************************************************************************************************************
/// \brief Check if all tag_type values in a tagged value pack are unique
template <typename... Ts>
struct all_tags_unique {
  static_assert(all_have_tags_v<Ts...>, "All of the given components must have tags.");
  static constexpr bool value = ((count_type_v<typename Ts::tag_type, typename Ts::tag_type...> == 1) && ...);
};

template <typename... Ts>
static constexpr bool all_tags_unique_v = all_tags_unique<Ts...>::value;

// **********************************************************************************************************************
/// \brief Check if a tagged value pack contains a value with the given Tag
template <typename Tag, typename... Ts>
  requires(all_have_tags_v<Ts...>)
struct contains_tag : std::false_type {};

template <typename Tag, typename First, typename... Rest>
  requires(all_have_tags_v<First, Rest...>)
struct contains_tag<Tag, First, Rest...> {
  static constexpr bool value = std::is_same_v<typename First::tag_type, Tag> || contains_tag<Tag, Rest...>::value;
};

template <typename Tag, typename... Ts>
  requires(all_have_tags_v<Ts...>)
static constexpr bool contains_tag_v = contains_tag<Tag, Ts...>::value;

// **********************************************************************************************************************
/// \brief Find the index of Tag inside a pack of tagged components
template <typename Tag, typename... Ts>
  requires(all_have_tags_v<Ts...> && contains_tag_v<Tag, Ts...>)
struct index_of_tag {
  static constexpr size_t value = index_finder_v<Tag, typename Ts::tag_type...>;
};

template <typename Tag, typename... Ts>
  requires(all_have_tags_v<Ts...> && contains_tag_v<Tag, Ts...>)
static constexpr size_t index_of_tag_v = index_of_tag<Tag, Ts...>::value;

namespace impl {

/// \brief Helper function to locate the value that matches a Tag
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

template <typename Tag, typename First>
KOKKOS_FUNCTION static constexpr const auto& find_const_component_recurse_impl(const First& first) {
  return first;
}

/// \brief Fetch the value corresponding to the given Tag using an index sequence
template <typename Tag, typename... Ts, size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_const_component_impl(const tuple<Ts...>& tuple,
                                                                 std::index_sequence<Is...>) {
  // Unpack into the
  return find_const_component_recurse_impl<Tag>(get<Is>(tuple)...);
}

/// \brief Helper function to locate the value that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr auto& find_component_recurse_impl(First& first, Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_component_recurse_impl<Tag>(rest...);
  }
}

template <typename Tag, typename First>
KOKKOS_FUNCTION static constexpr auto& find_component_recurse_impl(First& first) {
  return first;
}

/// \brief Fetch the value corresponding to the given Tag using an index sequence
template <typename Tag, typename... Ts, size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_component_impl(tuple<Ts...>& tuple, std::index_sequence<Is...>) {
  // Unpack into the
  return find_component_recurse_impl<Tag>(get<Is>(tuple)...);
}

/// \brief Fetch the value corresponding to the given Tag (returns a const reference since the tuple is const)
template <typename Tag, typename... Ts>
KOKKOS_FUNCTION static constexpr const auto& find_component(const tuple<Ts...>& tuple) {
  static_assert(all_have_tags_v<Ts...>, "All of the given components must have tags.");
  static_assert(contains_tag_v<Tag, Ts...>, "Attempting to find a value that does not exist in the given tuple");
  return impl::find_const_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Ts)>{});
}

/// \brief Fetch the value corresponding to the given Tag
template <typename Tag, typename... Ts>
KOKKOS_FUNCTION static constexpr auto& find_component(tuple<Ts...>& tuple) {
  static_assert(all_have_tags_v<Ts...>, "All of the given components must have tags.");
  static_assert(contains_tag_v<Tag, Ts...>, "Attempting to find a value that does not exist in the given tuple");
  return impl::find_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Ts)>{});
}

}  // namespace impl

/// \brief A runtime_aggregate: A bag of runtime tagged variants or,
/// In other words, an unordered map of variants indexed by tag string.
///
/// Construct an runtime_aggregate via a fluent interface:
/// \code{.cpp}
///   using VariantType = variant<Type1, Type2, Type3>;
///   auto ragg = mundy::make_runtime_aggregate<VariantType>()
///       .append("Tag1", variant_component1)
///       .append("Tag2", variant_component2);
/// \endcode
template <typename VariantType>
class runtime_aggregate {
 public:
  using variant_t = VariantType;

  //! \name Constructors
  //@{

  /// \brief Default constructor
  runtime_aggregate() = default;

  /// \brief Default copy/move/assign constructors
  runtime_aggregate(const runtime_aggregate&) = default;
  runtime_aggregate(runtime_aggregate&&) = default;
  runtime_aggregate& operator=(const runtime_aggregate&) = default;
  runtime_aggregate& operator=(runtime_aggregate&&) = default;
  //@}

  /// \brief Add a value (fluent interface):
  runtime_aggregate<VariantType>& append(const std::string& tag, variant_t new_component) {
    component_map_.insert_or_assign(tag, std::move(new_component));
    return *this;
  }

  /// \brief Fetch the value corresponding to the given Tag
  const variant_t& get(const std::string& tag) const {
    return component_map_.at(tag);
  }
  variant_t& get(const std::string& tag) {
    return component_map_.at(tag);
  }

  /// \brief Check if we have a value with the given Tag
  bool has(const std::string& tag) const {
    return component_map_.find(tag) != component_map_.end();
  }

  /// \brief Get the number of components in this runtime_aggregate
  size_t size() const {
    return static_cast<size_t>(component_map_.size());
  }

  //! \name Private members (no touch)
  //@{

  std::map<std::string, variant_t> component_map_;
  //@}
};  // runtime_aggregate

//! \name Non-member functions
//@{

/// \brief Canonical way to construct a runtime_aggregate
template <typename VariantType>
auto make_runtime_aggregate() {
  return runtime_aggregate<VariantType>();
}
//@}

/// \brief A variant_aggregate: A bag of compile-time tagged variants.
/// In other words, a compile-time map of variants indexed by tag type.
///
/// Construct a variant_aggregate via a fluent interface:
/// \code{.cpp}
///   using VariantType = mundy::variant<int, double>;
///   auto vagg = mundy::make_variant_aggregate<VariantType>()
///       .append<Tag1>(VariantType(1))
///       .append<Tag2>(VariantType(2.0));
/// \endcode
///
/// Each Tag type must be unique within a variant_aggregate.
template <typename VariantType, typename... Tags>
class variant_aggregate {
 public:
  using variant_t = VariantType;
  using TagsTuple = tuple<Tags...>;
  static constexpr size_t N = sizeof...(Tags);

  //! \name Constructors
  //@{

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr variant_aggregate() = default;

  /// \brief Construct a variant_aggregate that has the given tagged variants
  KOKKOS_FUNCTION
  constexpr variant_aggregate(Kokkos::Array<variant_t, N> variants) : variants_(std::move(variants)) {
  }

  /// \brief Default copy/move/assign constructors
  KOKKOS_DEFAULTED_FUNCTION constexpr variant_aggregate(const variant_aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr variant_aggregate(variant_aggregate&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr variant_aggregate& operator=(const variant_aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr variant_aggregate& operator=(variant_aggregate&&) = default;
  //@}

  /// \brief Add a value (fluent interface):
  template <typename Tag>
    requires(!contains_type_v<Tag, Tags...>)
  KOKKOS_FUNCTION constexpr auto append(variant_t new_variant) const {
    // Copy the old variants into a new array with one extra slot
    Kokkos::Array<variant_t, N + 1> new_variants;
    for (size_t i = 0; i < N; ++i) {
      new_variants[i] = variants_[i];
    }
    new_variants[N] = std::move(new_variant);

    using NewType = variant_aggregate<VariantType, Tags..., Tag>;
    return NewType(new_variants);
  }

  template <typename Tag>
    requires(contains_type_v<Tag, Tags...>)
  KOKKOS_FUNCTION constexpr void append(variant_t /*new_variant*/) const {
    static_assert(!contains_type_v<Tag, Tags...>, "variant_aggregate::append called with duplicate Tag");
  }

  /// \brief The I'th tag type
  template <size_t I>
    requires(sizeof...(Tags) > 0 && I < sizeof...(Tags))
  using tag_type = tuple_element_t<I, TagsTuple>;

  /// \brief Fetch the I'th value (compile-time index)
  template <size_t I>
    requires(I < sizeof...(Tags))
  KOKKOS_INLINE_FUNCTION constexpr const variant_t& get() const {
    return variants_[I];
  }
  template <size_t I>
    requires(I < sizeof...(Tags))
  KOKKOS_INLINE_FUNCTION constexpr variant_t& get() {
    return variants_[I];
  }

  template <size_t I>
    requires(I >= sizeof...(Tags))
  KOKKOS_INLINE_FUNCTION constexpr const void get() const {
    static_assert(I < sizeof...(Tags), "Attempting to get a value with an index that is out of bounds");
  }
  template <size_t I>
    requires(I >= sizeof...(Tags))
  KOKKOS_INLINE_FUNCTION constexpr void get() {
    static_assert(I < sizeof...(Tags), "Attempting to get a value with an index that is out of bounds");
  }

  /// \brief Fetch the I'th value (runtime index)
  KOKKOS_INLINE_FUNCTION
  constexpr const variant_t& get(size_t I) const {
    return variants_[I];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr variant_t& get(size_t I) {
    return variants_[I];
  }

  /// \brief Fetch the value corresponding to the given Tag
  template <typename Tag>
    requires(contains_type_v<Tag, Tags...>)
  KOKKOS_INLINE_FUNCTION constexpr const variant_t& get() const {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  /// \brief Fetch the value corresponding to the given Tag
  template <typename Tag>
    requires(contains_type_v<Tag, Tags...>)
  KOKKOS_INLINE_FUNCTION constexpr variant_t& get() {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  template <typename Tag>
    requires(!contains_type_v<Tag, Tags...>)
  KOKKOS_INLINE_FUNCTION constexpr void get() const {
    static_assert(contains_type_v<Tag, Tags...>,
                  "Attempting to get a value that does not exist in the variant_aggregate");
  }
  template <typename Tag>
    requires(!contains_type_v<Tag, Tags...>)
  KOKKOS_INLINE_FUNCTION constexpr void get() {
    static_assert(contains_type_v<Tag, Tags...>,
                  "Attempting to get a value that does not exist in the variant_aggregate");
  }

  /// \brief Check if we have a value with the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION static constexpr bool has() {
    return contains_type_v<Tag, Tags...>;
  }

  /// \brief Get the number of components in this variant_aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t size() {
    return N;
  }

 private:
  //! \name Private members
  //@{

  Kokkos::Array<variant_t, N> variants_;
  //@}
};  // variant_aggregate

/// \brief Canonical way to construct a variant_aggregate
template <typename VariantType>
KOKKOS_INLINE_FUNCTION constexpr auto make_variant_aggregate() {
  return variant_aggregate<VariantType>();
}

/// \brief Fetch the variant corresponding to the given Tag
template <typename Tag, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr const VariantType& get(const variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<Tag>();
}

/// \brief Fetch the variant corresponding to the given Tag
template <typename Tag, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr VariantType& get(variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<Tag>();
}

/// \brief Fetch the variant at index I
template <size_t I, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr const VariantType& get(const variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<I>();
}
template <size_t I, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr VariantType& get(variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<I>();
}

/// \brief The I'th variant_aggregate tag
template <size_t I, typename VarAggType>
struct variant_aggregate_tag;

template <size_t I, typename VariantType, typename... Tags>
struct variant_aggregate_tag<I, variant_aggregate<VariantType, Tags...>> {
  using type = type_at_index_t<I, Tags...>;
};

template <size_t I, typename VarAggType>
using variant_aggregate_tag_t = variant_aggregate_tag<I, VarAggType>::type;

/// \brief Check if a variant_aggregate has a variant with the given Tag
template <typename Tag, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr bool has(const variant_aggregate<VariantType, Tags...>& /*v_agg*/) {
  return variant_aggregate<VariantType, Tags...>::template has<Tag>();
}

/// \brief Check whether a variant_aggregate type has a value with the given Tag.
/// Usage: variant_aggregate_has_v<Tag, VarAggType>
template <typename Tag, typename VarAggType>
struct variant_aggregate_has {
  static constexpr bool value = VarAggType::template has<Tag>();
};
//
template <typename Tag, typename VarAggType>
static constexpr bool variant_aggregate_has_v = variant_aggregate_has<Tag, VarAggType>::value;

/// \brief Add a new value to an existing variant_aggregate (fluent interface)
template <typename Tag, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr auto append(const variant_aggregate<VariantType, Tags...>& v_agg,
                                             VariantType new_variant) {
  return v_agg.template append<Tag>(std::move(new_variant));
}

/// \brief Project selected tags from a variant_aggregate into a new variant_aggregate.
/// Copies the corresponding variants and preserves the requested tag order.
template <typename... SelectedTags, typename VariantType, typename... Tags>
  requires((sizeof...(SelectedTags) > 0) && (contains_type_v<SelectedTags, Tags...> && ...))
KOKKOS_INLINE_FUNCTION constexpr auto project(variant_aggregate<VariantType, Tags...>& v_agg) {
  Kokkos::Array<VariantType, sizeof...(SelectedTags)> projected_variants = {v_agg.template get<SelectedTags>()...};
  return variant_aggregate<VariantType, SelectedTags...>(std::move(projected_variants));
}

template <typename... SelectedTags, typename VariantType, typename... Tags>
  requires(!((sizeof...(SelectedTags) > 0) && (contains_type_v<SelectedTags, Tags...> && ...)))
KOKKOS_INLINE_FUNCTION constexpr void project(variant_aggregate<VariantType, Tags...>& /*v_agg*/) {
  if constexpr (sizeof...(SelectedTags) == 0) {
    static_assert(sizeof...(SelectedTags) > 0, "project<Tags...>(v_agg) requires at least one Tag.");
  } else {
    static_assert((contains_type_v<SelectedTags, Tags...> && ...),
                  "project<Tags...>(v_agg) called with a Tag not present in v_agg.");
  }
}

/// \brief Project selected tags from a const variant_aggregate into a new variant_aggregate.
/// Copies the corresponding variants and preserves the requested tag order.
template <typename... SelectedTags, typename VariantType, typename... Tags>
  requires((sizeof...(SelectedTags) > 0) && (contains_type_v<SelectedTags, Tags...> && ...))
KOKKOS_INLINE_FUNCTION constexpr auto project(const variant_aggregate<VariantType, Tags...>& v_agg) {
  Kokkos::Array<VariantType, sizeof...(SelectedTags)> projected_variants = {v_agg.template get<SelectedTags>()...};
  return variant_aggregate<VariantType, SelectedTags...>(std::move(projected_variants));
}

template <typename... SelectedTags, typename VariantType, typename... Tags>
  requires(!((sizeof...(SelectedTags) > 0) && (contains_type_v<SelectedTags, Tags...> && ...)))
KOKKOS_INLINE_FUNCTION constexpr void project(const variant_aggregate<VariantType, Tags...>& /*v_agg*/) {
  if constexpr (sizeof...(SelectedTags) == 0) {
    static_assert(sizeof...(SelectedTags) > 0, "project<Tags...>(v_agg) requires at least one Tag.");
  } else {
    static_assert((contains_type_v<SelectedTags, Tags...> && ...),
                  "project<Tags...>(v_agg) called with a Tag not present in v_agg.");
  }
}

namespace impl {

/// \brief A concept to check if a given type has operator()(args...)
template <typename T, typename... Args>
concept callable_with = requires(T t, Args... args) { t(std::forward<Args>(args)...); };

}  // namespace impl

/// \brief An aggregate: A bag of compile-time tagged types
/// In other words, a compile-time unordered map of arbitrary types indexed by tag type.
///
/// They are compile-time compatable "structural types" compatable with NTTPs.
/// Their types must be default constructable and copyable.
///
/// Construct an aggregate via a fluent interface:
/// \code{.cpp}
///   auto agg = mundy::make_aggregate()
///       .append<Tag1>(component1)
///       .append<Tag2>(component2);
/// \endcode
///
///
/// # Example use cases include
///
/// 1. Compile-time extensible tuple:
/// \code{.cpp}
///   auto cfg = make_aggregate()
///       .append<DT>(0.01)
///       .append<MAX_ITERS>(1000);
///
///   double dt     = cfg.get<DT>();
///   size_t it_max = cfg.get<MAX_ITERS>();
///   // double dt  = cfg.get<DT>(0);  // error: DT is not callable
/// \endcode
///
/// 2. Aggregation of accessors:
/// \code{.cpp}
///   auto spheres = make_aggregate()
///       .append<CENTER>(center_accessor)
///       .append<RADIUS>(radius_accessor);
///
///   auto c = spheres.get<CENTER>(10);
///   auto r = spheres.get<RADIUS>(3);
///
///   auto stored_center_accessor = spheres.get<CENTER>();
/// \endcode
///
/// 3. Aggregation of policies/strategies:
/// \code{.cpp}
///   auto solver_policies = make_aggregate()
///       .append<SOLVER>(solver_policy)
///       .append<PRECONDITIONER>(preconditioner_policy);
///
///   solver_policies.get<SOLVER>().solve(..., solver_policies.get<PRECONDITIONER>(), ...);
/// \endcode
///
/// 4. Aggregation of algorithms/functors:
/// \code{.cpp}
///   auto algs = make_aggregate()
///       .append<SORT>(SortAlgorithm{})
///       .append<FILTER>(FilterAlgorithm{});
///
///   algs.get<SORT>(data);
///   auto filtered = algs.get<FILTER>(data);
/// \endcode
///
/// 5. Mixed usage:
/// \code{.cpp}
///   auto agg = make_aggregate()
///       .append<POS>(pos_accessor)
///       .append<VEL>(vel_accessor)
///       .append<DT>(0.01);
///
///   agg.get<POS>(i) += agg.get<VEL>(i) * agg.get<DT>();
/// \endcode
///
///
/// # Tag requirements
/// Each Tag type must be unique within an aggregate but can otherwise be any type (including incomplete types).
/// Indeed, to make declaring types easier, the simplest strategy is to use incomplete structs:
/// \code{.cpp}
///   struct DT; struct MAX_ITERS;
/// \endcode
template <typename... TaggedComponents>
  requires(all_have_tags_v<TaggedComponents...> /* All of the given components must have tags */
           && all_tags_unique_v<TaggedComponents...> /* All tags in an aggregate must be unique */)
class aggregate {
 public:
  using TaggedComponentsTuple = tuple<TaggedComponents...>;

  //! \name Constructors
  //@{

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr aggregate() = default;

  /// \brief Construct an aggregate that has the given components
  KOKKOS_FUNCTION
  constexpr aggregate(TaggedComponentsTuple tagged_components)
    requires(sizeof...(TaggedComponents) > 0)
      : tagged_components_(std::move(tagged_components)) {
  }

  /// \brief Default copy/move/assign constructors
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate(const aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate(aggregate&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate& operator=(const aggregate&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr aggregate& operator=(aggregate&&) = default;
  //@}

  /// \brief Add a value (fluent interface):
  template <typename Tag, typename NewComponent>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_FUNCTION constexpr auto append(NewComponent new_component) const {
    tagged<Tag, NewComponent> new_tagged_comp(std::move(new_component));
    auto new_tuple = ::mundy::tuple_cat(tagged_components_, ::mundy::make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = aggregate<TaggedComponents..., decltype(new_tagged_comp)>;
    return NewType(new_tuple);
  }
  //
  template <typename Tag, typename NewComponent>
    requires(contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_FUNCTION constexpr void append(NewComponent /*new_component*/) const {
    static_assert(!contains_tag_v<Tag, TaggedComponents...>, "aggregate::append called with duplicate Tag");
  }

  /// \brief The I'th tag type
  template <size_t I>
    requires(sizeof...(TaggedComponents) > 0 && I < sizeof...(TaggedComponents))
  using tag_type = typename tuple_element_t<I, TaggedComponentsTuple>::tag_type;

  /// \brief Fetch the I'th tagged object
  template <size_t I>
    requires(I < sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr const auto& get_tagged() const {
    return tagged_components_.template get<I>();
  }
  template <size_t I>
    requires(I < sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr auto& get_tagged() {
    return tagged_components_.template get<I>();
  }

  template <size_t I>
    requires(I >= sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr const void get_tagged() const {
    static_assert(I < sizeof...(TaggedComponents), "Attempting to get a value with an index that is out of bounds");
  }
  template <size_t I>
    requires(I >= sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr void get_tagged() {
    static_assert(I < sizeof...(TaggedComponents), "Attempting to get a value with an index that is out of bounds");
  }

  /// \brief Fetch the tagged object corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr const auto& get_tagged() const {
    constexpr size_t index = index_of_tag_v<Tag, TaggedComponents...>;
    return tagged_components_.template get<index>();
  }
  template <typename Tag>
    requires(contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr auto& get_tagged() {
    constexpr size_t index = index_of_tag_v<Tag, TaggedComponents...>;
    return tagged_components_.template get<index>();
  }

  template <typename Tag>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get_tagged() const {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get_tagged() {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }

  /// \brief Fetch the I'th value
  template <size_t I>
    requires(I < sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr const auto& get() const {
    return tagged_components_.template get<I>().get();
  }
  template <size_t I>
    requires(I < sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr auto& get() {
    return tagged_components_.template get<I>().get();
  }

  template <size_t I>
    requires(I >= sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr const void get() const {
    static_assert(I < sizeof...(TaggedComponents), "Attempting to get a value with an index that is out of bounds");
  }
  template <size_t I>
    requires(I >= sizeof...(TaggedComponents))
  KOKKOS_INLINE_FUNCTION constexpr void get() {
    static_assert(I < sizeof...(TaggedComponents), "Attempting to get a value with an index that is out of bounds");
  }

  /// \brief Fetch the value corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr const auto& get() const {
    constexpr size_t index = index_of_tag_v<Tag, TaggedComponents...>;
    return tagged_components_.template get<index>().get();
  }
  template <typename Tag>
    requires(contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr auto& get() {
    constexpr size_t index = index_of_tag_v<Tag, TaggedComponents...>;
    return tagged_components_.template get<index>().get();
  }

  template <typename Tag>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get() const {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get() {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }

  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

  /// \brief Get tagged object of the given args: Perform get<I'th tag>()(args...) with syntactic sugar
  template <size_t I, typename... Args>
    requires(impl::callable_with<const typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>)
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(Args&&... args) const {
    return get<I>()(std::forward<Args>(args)...);
  }
  template <size_t I, typename... Args>
    requires(impl::callable_with<typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>)
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(Args&&... args) {
    return get<I>()(std::forward<Args>(args)...);
  }

  template <size_t I, typename... Args>
    requires(!impl::callable_with<const typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) const {
    static_assert(impl::callable_with<const typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>,
                  "The I'th value is not callable with the given arguments.");
  }
  template <size_t I, typename... Args>
    requires(!impl::callable_with<typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) {
    static_assert(impl::callable_with<typename type_at_index_t<I, TaggedComponents...>::value_type&, Args...>,
                  "The I'th value is not callable with the given arguments.");
  }

  /// \brief Get tagged object of the given args: Perform get<TAG>()(args...) with syntactic sugar
  template <typename Tag, typename... Args>
    requires(contains_tag_v<Tag, TaggedComponents...> &&
             impl::callable_with<const typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>,
                                                                TaggedComponents...>::value_type&,
                                 Args...>)
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(Args&&... args) const {
    return get<Tag>()(std::forward<Args>(args)...);
  }
  template <typename Tag, typename... Args>
    requires(contains_tag_v<Tag, TaggedComponents...> &&
             impl::callable_with<
                 typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>, TaggedComponents...>::value_type&,
                 Args...>)
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(Args&&... args) {
    return get<Tag>()(std::forward<Args>(args)...);
  }

  template <typename Tag, typename... Args>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) const {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }
  template <typename Tag, typename... Args>
    requires(!contains_tag_v<Tag, TaggedComponents...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) {
    static_assert(contains_tag_v<Tag, TaggedComponents...>,
                  "Attempting to get a value that does not exist in the aggregate");
  }

  template <typename Tag, typename... Args>
    requires(contains_tag_v<Tag, TaggedComponents...> &&
             !impl::callable_with<const typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>,
                                                                 TaggedComponents...>::value_type&,
                                  Args...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) const {
    static_assert(
        impl::callable_with<
            const typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>, TaggedComponents...>::value_type&,
            Args...>,
        "The value with the given Tag is not callable with the given arguments.");
  }
  template <typename Tag, typename... Args>
    requires(contains_tag_v<Tag, TaggedComponents...> &&
             !impl::callable_with<
                 typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>, TaggedComponents...>::value_type&,
                 Args...>)
  KOKKOS_INLINE_FUNCTION constexpr void get(Args&&... /*args*/) {
    static_assert(
        impl::callable_with<
            typename type_at_index_t<index_of_tag_v<Tag, TaggedComponents...>, TaggedComponents...>::value_type&,
            Args...>,
        "The value with the given Tag is not callable with the given arguments.");
  }

  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP

  /// \brief Check if we have a value with the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION static constexpr bool has() {
    return contains_tag_v<Tag, TaggedComponents...>;
  }

  /// \brief Get the number of components in this aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t size() {
    return sizeof...(TaggedComponents);
  }

  //! \name Private members (no touch)
  //@{

  TaggedComponentsTuple tagged_components_;
  //@}
};  // aggregate

//! \name Non-member functions/helpers
//@{

/// \brief The type of aggregates is typically inferred, so this is the canonical way to construct one.
KOKKOS_INLINE_FUNCTION
constexpr auto make_aggregate() {
  return aggregate<>();
}

/// \brief Project selected tags from an aggregate into a new aggregate (copies their corresponding components).
template <typename... Tags, typename... Ts>
  requires((sizeof...(Tags) > 0) && (contains_tag_v<Tags, Ts...> && ...))
KOKKOS_INLINE_FUNCTION constexpr auto project(aggregate<Ts...>& agg) {
  return aggregate(::mundy::make_tuple(agg.template get_tagged<Tags>()...));
}

template <typename... Tags, typename... Ts>
  requires(!((sizeof...(Tags) > 0) && (contains_tag_v<Tags, Ts...> && ...)))
KOKKOS_INLINE_FUNCTION constexpr void project(aggregate<Ts...>& /*agg*/) {
  if constexpr (sizeof...(Tags) == 0) {
    static_assert(sizeof...(Tags) > 0, "project<Tags...>(agg) requires at least one Tag.");
  } else {
    static_assert((aggregate<Ts...>::template has<Tags>() && ...),
                  "project<Tags...>(agg) called with a Tag not present in agg.");
  }
}

/// \brief Project selected tags from a const aggregate into a new aggregate (copies their corresponding components).
template <typename... Tags, typename... Ts>
  requires((sizeof...(Tags) > 0) && (contains_tag_v<Tags, Ts...> && ...))
KOKKOS_INLINE_FUNCTION constexpr auto project(const aggregate<Ts...>& agg) {
  return aggregate(::mundy::make_tuple(agg.template get_tagged<Tags>()...));
}

template <typename... Tags, typename... Ts>
  requires(!((sizeof...(Tags) > 0) && (contains_tag_v<Tags, Ts...> && ...)))
KOKKOS_INLINE_FUNCTION constexpr void project(const aggregate<Ts...>& /*agg*/) {
  if constexpr (sizeof...(Tags) == 0) {
    static_assert(sizeof...(Tags) > 0, "project<Tags...>(agg) requires at least one Tag.");
  } else {
    static_assert((contains_tag_v<Tags, Ts...> && ...), "project<Tags...>(agg) called with a Tag not present in agg.");
  }
}

/// \brief Fetch the value corresponding to the given Tag
template <typename Tag, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr const auto& get(const aggregate<Ts...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the value corresponding to the given Tag
template <typename Tag, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr auto& get(aggregate<Ts...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the value at index I
template <size_t I, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr const auto& get(const aggregate<Ts...>& agg) {
  return agg.template get<I>();
}
template <size_t I, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr auto& get(aggregate<Ts...>& agg) {
  return agg.template get<I>();
}

/// \brief Check if an aggregate have a value with the given Tag
template <typename Tag, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr bool has(const aggregate<Ts...>& /*agg*/) {
  return contains_tag_v<Tag, Ts...>;
}

/// \brief Check if an aggregate type has a value with the given Tag usage aggregate_has_v<Tag, AggType>
template <typename Tag, typename AggType>
struct aggregate_has {
  static constexpr bool value = AggType::template has<Tag>();
};
//
template <typename Tag, typename AggType>
static constexpr bool aggregate_has_v = aggregate_has<Tag, AggType>::value;

/// \brief Add a new value to an existing aggregate (fluent interface)
template <typename Tag, typename NewComponent, typename... Ts>
KOKKOS_INLINE_FUNCTION constexpr auto append(const aggregate<Ts...>& agg, NewComponent new_component) {
  return agg.template append<Tag>(std::move(new_component));
}

/// \brief The I'th aggregate tag
template <size_t I, typename AggType>
struct aggregate_tag;

template <size_t I, typename... Ts>
struct aggregate_tag<I, aggregate<Ts...>> {
  using type = type_at_index_t<I, Ts...>::tag_type;
};

template <size_t I, typename AggType>
using aggregate_tag_t = aggregate_tag<I, AggType>::type;

/// \brief Overload the stream operator for aggregates
template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const aggregate<Ts...>& agg) {
  // Print the (tag, val) pairs
  os << "aggregate{";
  size_t i = 0;
  ((os << typeid(typename Ts::tag_type).name() << ": " << agg.template get<typename Ts::tag_type>()
       << (sizeof...(Ts) > 1 ? ", " : "")),
   ...);
  os << "}";
  return os;
}
//@}

}  // namespace mundy

#endif  // MUNDY_UTILS_AGGREGATE_HPP_
