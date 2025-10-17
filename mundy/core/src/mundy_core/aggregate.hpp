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
#include <mundy_core/tuple.hpp>        // for mundy::core::tuple
#include <mundy_core/type_traits.hpp>  // for core::count_type_v
#include <mundy_core/variant.hpp>      // for mundy::core::variant

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

/// \brief A runtime_aggregate: A bag of runtime tagged variants or,
/// In other words, an unordered map of variants indexed by tag string.
///
/// Construct an runtime_aggregate via a fluent interface:
/// \code{.cpp}
///   using VariantType = variant<Type1, Type2, Type3>;
///   auto ragg = mundy::core::make_runtime_aggregate<VariantType>()
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

  /// \brief Add a component (fluent interface):
  runtime_aggregate<VariantType>& append(const std::string& tag, variant_t new_component) const {
    component_map_.emplace(tag, std::move(new_component));
    return *this;
  }

  /// \brief Fetch the component corresponding to the given Tag
  const variant_t& get(const std::string& tag) const {
    return component_map_.at(tag);
  }
  template <typename Tag>
  variant_t& get(const std::string& tag) {
    return component_map_.at(tag);
  }

  /// \brief Check if we have a component with the given Tag
  bool has(const std::string& tag) {
    return component_map_.find(tag) != component_map_.end();
  }

  /// \brief Get the number of components in this runtime_aggregate
  size_t size() {
    return component_map_.size();
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

/// \brief A variant_aggregate: A bag of compile-time tagged variants
/// In other words, a compile-time map of variants indexed by tag type.
template <typename VariantType, typename... Tags>
class variant_aggregate {
 public:
  using variant_t = VariantType;
  using TagsTuple = core::tuple<Tags...>;
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

  /// \brief Add a component (fluent interface):
  template <typename Tag>
  KOKKOS_FUNCTION constexpr auto append(variant_t new_variant) const {
    // Copy the old variants into a new array with one extra slot
    Kokkos::Array<variant_t, N> new_variants;
    for (size_t i = 0; i < N; ++i) {
      new_variants[i] = variants_[i];
    }
    new_variants[N] = std::move(new_variant);

    using NewType = variant_aggregate<VariantType, Tags..., Tag>;
    return NewType(new_variants);
  }

  /// \brief The I'th tag type
  template <size_t I>
  using tag_t = tuple_element_t<I, TagsTuple>;

  /// \brief Fetch the I'th component (compile-time index)
  template <size_t I>
  KOKKOS_INLINE_FUNCTION constexpr const variant_t& get() const {
    return variants_[I];
  }
  template <size_t I>
  KOKKOS_INLINE_FUNCTION constexpr variant_t& get() {
    return variants_[I];
  }

  /// \brief Fetch the I'th component (runtime index)
  KOKKOS_INLINE_FUNCTION
  constexpr const variant_t& get(size_t I) const {
    return variants_[I];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr variant_t& get(size_t I) {
    return variants_[I];
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION constexpr const variant_t& get() const {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION constexpr variant_t& get() {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  /// \brief Check if we have a component with the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION static constexpr bool has() {
    return contains_type_v<Tag, Tags...>;
  }

  /// \brief Get the number of components in this variant_aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t size() {
    return N;
  }

  //! \name Private members (no touch)
  //@{

  Kokkos::Array<variant_t, N> variants_;
  //@}
};  // variant_aggregate

/// \brief Canonical way to construct a variant_aggregate
template <typename VariantType>
auto make_variant_aggregate() {
  return variant_aggregate<VariantType>();
}

/// \brief Check if a variant_aggregate have a variant with the given Tag
template <typename Tag, typename VariantType, typename... Tags>
KOKKOS_INLINE_FUNCTION constexpr bool has(const variant_aggregate<VariantType, Tags...>& /*v_agg*/) {
  return variant_aggregate<VariantType, Tags...>::template has<Tag>();
}

/// \brief Check if a variant aggregate type has a component with the given Tag usage variant_aggregate_has_v<Tag, VarAggType>
template <typename Tag, typename VarAggType>
struct variant_aggregate_has {
  static constexpr bool value = VarAggType::template has<Tag>();
};
//
template <typename Tag, typename VarAggType>
static constexpr bool variant_aggregate_has_v = variant_aggregate_has<Tag, VarAggType>::value;

/// \brief Add a new component to an existing aggregate (fluent interface)
template <typename Tag, typename NewComponent, typename... Components>
KOKKOS_INLINE_FUNCTION constexpr auto append(const aggregate<Components...>& agg, NewComponent new_component) {
  return agg.template append<Tag>(std::move(new_component));
}

/// \brief An aggregate: A bag of compile-time tagged types
/// In other words, a compile-time unordered map of arbitrary types indexed by tag type.
///
/// They are compile-time compatable "structural types" compatable with NTTPs.
/// Their types must be default constructable and copyable.
///
/// Construct an aggregate via a fluent interface:
/// \code{.cpp}
///   auto agg = mundy::core::make_aggregate()
///       .append<Tag1>(component1)
///       .append<Tag2>(component2);
/// \endcode
template <typename... TaggedComponents>
class aggregate {
 public:
  static_assert(impl::all_have_tags<TaggedComponents...>, "All of the given components must have tags.");
  using TaggedComponentsTuple = core::tuple<TaggedComponents...>;

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

  /// \brief Add a component (fluent interface):
  template <typename Tag, typename NewComponent>
  KOKKOS_FUNCTION constexpr auto append(NewComponent new_component) const {
    impl::TaggedComponent<Tag, NewComponent> new_tagged_comp(std::move(new_component));
    auto new_tuple = core::tuple_cat(tagged_components_, core::make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = aggregate<TaggedComponents..., decltype(new_tagged_comp)>;
    return NewType(new_tuple);
  }

  /// \brief Fetch the I'th component
  template <size_t I>
  KOKKOS_INLINE_FUNCTION constexpr const auto& get() const {
    return tagged_components_.template get<I>().component();
  }
  template <size_t I>
  KOKKOS_INLINE_FUNCTION constexpr auto& get() {
    return tagged_components_.template get<I>().component();
  }

  /// \brief The I'th tag type
  template <size_t I>
  using tag_t = typename tuple_element_t<I, TaggedComponentsTuple>::tag_type;

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION constexpr const auto& get() const {
    return impl::find_component<Tag>(tagged_components_).component();
  }
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION constexpr auto& get() {
    return impl::find_component<Tag>(tagged_components_).component();
  }

  /// \brief Check if we have a component with the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION static constexpr bool has() {
    return impl::has_component_v<Tag, TaggedComponents...>;
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

//! \name Non-member functions
//@{

/// \brief The type of aggregates is typically inferred, so this is the canonical way to construct one.
KOKKOS_INLINE_FUNCTION
constexpr auto make_aggregate() {
  return aggregate<>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION constexpr const auto& get(const aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION constexpr auto& get(aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Check if an aggregate have a component with the given Tag
template <typename Tag, typename... Components>
KOKKOS_INLINE_FUNCTION constexpr bool has(const aggregate<Components...>& /*agg*/) {
  return aggregate<Components...>::template has<Tag>();
}

/// \brief Check if an aggregate type has a component with the given Tag usage aggregate_has_v<Tag, AggType>
template <typename Tag, typename AggType>
struct aggregate_has {
  static constexpr bool value = AggType::template has<Tag>();
};
//
template <typename Tag, typename AggType>
static constexpr bool aggregate_has_v = aggregate_has<Tag, AggType>::value;

/// \brief Add a new component to an existing aggregate (fluent interface)
template <typename Tag, typename NewComponent, typename... Components>
KOKKOS_INLINE_FUNCTION constexpr auto append(const aggregate<Components...>& agg, NewComponent new_component) {
  return agg.template append<Tag>(std::move(new_component));
}

/// \brief Overload the stream operator for aggregates
template <typename... Components>
std::ostream& operator<<(std::ostream& os, const aggregate<Components...>& agg) {
  // Print the (tag, val) pairs
  os << "aggregate{";
  ((os << typeid(typename Components::tag_type).name() << ": " << agg.template get<typename Components::tag_type>()
       << (sizeof...(Components) > 1 ? ", " : "")),
   ...);
  os << "}";
  return os;
}
//@}

namespace impl {

/// \brief Compile-time binomial
constexpr unsigned long long binom(unsigned int n, unsigned int k) {
  if (k > n) return 0ULL;
  if (k > n - k) k = n - k;
  unsigned long long r = 1;
  for (unsigned int i = 1; i <= k; ++i) r = (r * (n - k + i)) / i;
  return r;
}

/// \brief Unrank R-combination (no replacement) in colex
template <std::size_t R>
constexpr std::array<int, R> unrank_comb_norep(unsigned int M, unsigned long long k) {
  std::array<int, R> b{};
  int x = static_cast<int>(M);
  for (int i = R; i >= 1; --i) {
    while (binom(x, static_cast<unsigned>(i)) > k) {
      --x;
    }
    b[i - 1] = x;
    k -= binom(x, static_cast<unsigned>(i));
    --x;
  }
  return b;
}

/// \brief All weakly-increasing R-tuples over [0..N-1] (with replacement)
template <int N, std::size_t R>
consteval auto all_multicomb_indices() {
  constexpr auto CNT = binom(N + R - 1, R);
  constexpr unsigned int M = static_cast<unsigned int>(N + R - 1);
  std::array<std::array<int, R>, CNT> out{};
  for (unsigned long long k = 0; k < CNT; ++k) {
    auto b = unrank_comb_norep<R>(M, k);
    for (int i = 0; i < R; ++i) {
      out[k][i] = b[i] - i;  // weakly increasing 0..N-1
    }
  }
  return out;
}

/// \brief Cache the indices table per (N choose R)
template <std::size_t N, std::size_t R>
struct multicomb_index_table {
  static constexpr auto idxs = all_multicomb_indices<N, R>();
  static constexpr std::size_t size = idxs.size();
  static constexpr std::size_t n = N;
  static constexpr std::size_t r = R;
};

template <typename VariantAggregateType, std::array<int, VariantAggregateType::size()> active_ids, std::size_t... Is>
auto make_aggregate_from_active_impl(const VariantAggregateType& v_agg, std::index_sequence<Is...>) {
  using variant_t = typename VariantAggregateType::variant_t;
  using AggregateType = aggregate<impl::TaggedComponent<typename VariantAggregateType::template tag_t<Is>, variant_alternative_t<active_ids[Is], variant_t>>...>;
  return AggregateType(std::get<active_ids[Is]>(v_agg.template get<Is>())...);
}

/// \brief Construct a concrete aggregate from the active indices of a variant_aggregate
template <typename VariantAggregateType, std::array<int, VariantAggregateType::size()> active_ids>
auto make_aggregate_from_active(const VariantAggregateType& v_agg) {
  return make_aggregate_from_active_impl<VariantAggregateType, active_ids>(v_agg, std::make_index_sequence<VariantAggregateType::size()>{});
}

/// \brief Compare active indices of vs against the I-th pattern
template <std::size_t I, typename VariantType, typename... Tags>
bool match_active(const variant_aggregate<VariantType, Tags...>& v_agg) {
  constexpr std::size_t R = sizeof...(Tags);
  for (int j = 0; j < R; ++j) {
    const VariantType& v = v_agg.get(j);
    if (v.index() != multicomb_index_table<VariantType::size(), R>::idxs[I][j]) {
      return false;
    }
  }
  return true;
}

/// \brief Try the I-th multichoose; on match, call f(tuple_of_refs)
/// \return true if matched.
template <std::size_t I, typename VariantType, typename... Tags, typename Visitor>
bool try_one(const variant_aggregate<VariantType, Tags...>& v_agg, const Visitor& visitor) {
  constexpr std::size_t R = sizeof...(Tags);
  if (match_active<I>(v_agg)) {
    // Success! multicomb_index_table<N, R>::idxs[I] gives the std::array<int, R> of active indices
    constexpr std::array<int, R> active_ids = multicomb_index_table<VariantType::size(), R>::idxs[I];
    auto agg = make_aggregate_from_active<variant_aggregate<VariantType, Tags...>, active_ids>(v_agg);
    visitor(agg);
    return true;
  }
  return false;
}

/// \brief Visit over all patterns until a match
template <typename VariantType, typename... Tags, typename Visitor, std::size_t... I>
void visit_impl(const variant_aggregate<VariantType, Tags...>& v_agg, const Visitor& visitor,
                std::index_sequence<I...>) {
  bool done = false;
  ((done = done || try_one<I>(v_agg, visitor)), ...);
  MUNDY_THROW_ASSERT(done, std::runtime_error, "Internal error: no matching type combo found in visit.");
}

}  // namespace impl

/// \brief Visit all active types within a variant_aggregate
///
/// Example usage:
/// \code{.cpp}
///   auto v_agg = core::make_variant_aggregate<int, double, std::string>()
///                .append<Tag1>(123)                    // Tag1 -> int
///                .append<Tag2>(std::string("hello"));  // Tag2 -> std::string
///                .append<Tag3>(3.14);                  // Tag3 -> double
///   core::visit(v_agg, [](const auto& agg) {
///      std::cout << agg.get<Tag1>() << " " << agg.get<Tag2>() << " " << agg.get<Tag3>() << std::endl;
///   });
/// \endcode
template <typename VariantType, typename... Tags, typename Visitor>
void visit(const variant_aggregate<VariantType, Tags...>& v_agg, const Visitor& visitor) {
  using cached_table = impl::multicomb_index_table<VariantType::size(), sizeof...(Tags)>;
  impl::visit_impl(v_agg, visitor, std::make_index_sequence<cached_table::size()>{});
}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_AGGREGATE_HPP_
