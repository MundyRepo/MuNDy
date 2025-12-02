//g++ -O3 -std=c++20 ./variant_bench.cpp

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

// ----- Accessor (vector-backed or shared scalar) -----
enum class variant_t : unsigned { SHARED = 0u, VECTOR, CONDITIONAL, MAPPED_SCALAR, MAPPED_VECTOR, INVALID };

// **********************************************************************************************************************
/// \brief Check if a type is in a variadic list of types
template <class T, class... Ts>
struct contains_type {
  static constexpr bool value = (std::is_same_v<T, Ts> || ...);
};
//
template <class T, class... Ts>
static constexpr bool contains_type_v = contains_type<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Count how many times a type appears in a variadic list of types
template <class T, class... Types>
struct count_type {
  static constexpr size_t value = (0 + ... + (std::is_same_v<T, Types> ? 1 : 0));
};
template <class T, class... Types>
static constexpr size_t count_type_v = count_type<T, Types...>::value;

// **********************************************************************************************************************
/// \brief Find the index in the variadic list of types that matches the given type
template <class T, class... Ts>
struct index_finder;
//
template <class T>
struct index_finder<T> {
  static constexpr size_t value = 0;
};
//
template <class T, class First, class... Rest>
struct index_finder<T, First, Rest...> {
  static_assert(sizeof...(Rest) + 1 > 0, "Type not found in list");
  static constexpr size_t value = std::is_same_v<T, First> ? 0 : 1 + index_finder<T, Rest...>::value;
};
//
template <class T, class... Ts>
  requires(count_type_v<T, Ts...> == 1)
static constexpr size_t index_finder_v = index_finder<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Get the I'th type in a variadic list of types
template <std::size_t I, typename... Ts>
struct type_at_index;
//
template <std::size_t I, typename Head, typename... Tail>
struct type_at_index<I, Head, Tail...> {
  static_assert(I < 1 + sizeof...(Tail), "Index out of bounds in type_at_index");
  using type = typename type_at_index<I - 1, Tail...>::type;
};
//
// Specialization for the base case (I = 0)
template <typename Head, typename... Tail>
struct type_at_index<0, Head, Tail...> {
  using type = Head;
};
//
template <size_t I, class... Ts>
  requires(I < sizeof...(Ts))
using type_at_index_t = typename type_at_index<I, Ts...>::type;

namespace impl {

// The tuple implementation only comes in play when using capabilities
template <class T, size_t Idx>
struct tuple_member {
  T value;

  using value_type = T;

  // If T is default constructible, provide a default constructor
  constexpr tuple_member()
    requires std::default_initializable<T>
  = default;

  // Provide a constructor that takes a single argument.
  constexpr tuple_member(T const& val) : value(val) {
  }

  // Provide get() or equivalent
  inline constexpr T& get() {
    return value;
  }

  inline constexpr T const& get() const {
    return value;
  }
};

/// \brief Helper class which will be used via a fold expression to select the member with the correct Idx in a pack of
/// tuple_members
template <size_t SearchIdx, size_t Idx, class T>
struct tuple_idx_matcher {
  using type = tuple_member<T, Idx>;

  template <class Other>
  inline constexpr auto operator|([[maybe_unused]] Other v) const {
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
  inline constexpr auto operator|([[maybe_unused]] Other v) const {
    if constexpr (std::is_same_v<T, SearchType>) {
      return *this;
    } else {
      return v;
    }
  }
};

template <class IdxSeq, class... Elements>
struct tuple_impl;

template <size_t... Idx, class... Elements>
struct tuple_impl<std::index_sequence<Idx...>, Elements...> : public tuple_member<Elements, Idx>... {
  // If all elements are default constructible, provide a default constructor
  inline constexpr tuple_impl()
    requires((std::default_initializable<Elements> && ...))
  = default;

  inline constexpr tuple_impl(Elements... vals)
    requires(sizeof...(Elements) > 0)
      : tuple_member<Elements, Idx>{vals}... {
  }

  /// \brief Default copy/move/assign constructors
  inline constexpr tuple_impl(const tuple_impl&) = default;

  inline constexpr tuple_impl(tuple_impl&&) = default;

  inline constexpr tuple_impl& operator=(const tuple_impl&) = default;

  inline constexpr tuple_impl& operator=(tuple_impl&&) = default;

  /// \brief Get the element of the tuple at index N
  template <size_t N>
  inline constexpr auto& get() {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <size_t N>
  inline constexpr const auto& get() const {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  /// \brief Get the element of the tuple with the given type T (errors if T is not unique)
  template <typename T>
  inline constexpr const auto& get() const {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <typename T>
  inline constexpr auto& get() {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  // Helper alias: select the matching base; sentinel ensures fold is never empty.
  template <size_t N>
    requires(sizeof...(Elements) > 0)
  using base_of =
      typename decltype((tuple_idx_matcher<N, Idx, Elements>() | ... | tuple_idx_matcher<N, N, void>{}))::type;
};

}  // namespace impl

// A simple tuple-like class for representing slices internally and is
// compatible with device code This doesn't support type access since we don't
// need it This is not meant as an external API
template <class... Elements>
struct tuple : public impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  // If all elements are default constructible, provide a default constructor
  inline constexpr tuple()
    requires((std::default_initializable<Elements> && ...))
  = default;

  inline constexpr tuple(Elements... vals)
    requires(sizeof...(Elements) > 0)
      : impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...>(vals...) {
  }

  /// \brief Default copy/move/assign constructors
  inline constexpr tuple(const tuple&) = default;

  inline constexpr tuple(tuple&&) = default;

  inline constexpr tuple& operator=(const tuple&) = default;

  inline constexpr tuple& operator=(tuple&&) = default;

  /// \brief Get the size of the tuple
  inline static constexpr size_t size() {
    return sizeof...(Elements);
  }

  /// \brief Get the type of the N'th element
  template <size_t N>
    requires(sizeof...(Elements) > 0)
  using element_t = type_at_index_t<N, Elements...>;
};

template <size_t Idx, class... Args>
inline constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<Idx>();
}

template <size_t Idx, class... Args>
inline constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<Idx>();
}

template <class T, class... Args>
inline constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<T>();
}

template <class T, class... Args>
inline constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<T>();
}

// -------- tuple_size
template <class T>
struct tuple_size;  // primary

template <class... Es>
struct tuple_size<tuple<Es...>> {
  static constexpr std::size_t value = sizeof...(Es);
};

template <class T>
static constexpr std::size_t tuple_size_v = tuple_size<T>::value;

// -------- tuple_element
template <std::size_t I, class T>
struct tuple_element;  // primary

template <std::size_t I, class... Es>
struct tuple_element<I, tuple<Es...>> {
  static_assert(I < sizeof...(Es), "tuple_element index out of bounds");
  using type = type_at_index_t<I, Es...>;  // your existing meta util; OK with incomplete types
};

template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

template <class... Elements>
tuple(Elements...) -> tuple<Elements...>;

// Implementation to concatenate two tuples using index sequences
template <class FirstTuple, class SecondTuple, std::size_t... FirstIndices, std::size_t... SecondIndices>
inline constexpr auto tuple_cat_impl(const FirstTuple& first, const SecondTuple& second, std::index_sequence<FirstIndices...>,
                              std::index_sequence<SecondIndices...>) {
  // Extract elements from both tuples and construct the new tuple
  // This copy the elements of the tuples into the new tuple, so we remove const and ref qualifiers
  return tuple<std::decay_t<decltype(get<FirstIndices>(first))>...,
               std::decay_t<decltype(get<SecondIndices>(second))>...>{get<FirstIndices>(first)...,
                                                                      get<SecondIndices>(second)...};
}

// Public-facing `tuple_cat` function
template <class... FirstElements, class... SecondElements>
inline constexpr auto tuple_cat(const tuple<FirstElements...>& first, const tuple<SecondElements...>& second) {
  constexpr auto first_size = sizeof...(FirstElements);
  constexpr auto second_size = sizeof...(SecondElements);

  // Generate index sequences for both tuples
  using FirstIndices = std::make_index_sequence<first_size>;
  using SecondIndices = std::make_index_sequence<second_size>;

  // Delegate to the implementation
  return tuple_cat_impl(first, second, FirstIndices{}, SecondIndices{});
}

template <typename... input_t>
using tuple_cat_t = decltype(tuple_cat(std::declval<input_t>()...));

/// Make a tuple from a list of values.
template <class... Elements>
inline constexpr auto make_tuple(Elements... vals) {
  return tuple<Elements...>{vals...};
}

namespace impl {

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
inline static constexpr const auto& find_const_component_recurse_impl(const First& first, const Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_const_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
inline static constexpr auto& find_const_component_impl(const tuple<Components...>& tuple, std::index_sequence<Is...>) {
  // Unpack into the
  return find_const_component_recurse_impl<Tag>(get<Is>(tuple)...);
}

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
inline static constexpr auto& find_component_recurse_impl(First& first, Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
inline static constexpr auto& find_component_impl(tuple<Components...>& tuple, std::index_sequence<Is...>) {
  // Unpack into the
  return find_component_recurse_impl<Tag>(get<Is>(tuple)...);
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
inline static constexpr const auto& find_component(const tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  static_assert(has_component_v<Tag, Components...>,
                "Attempting to find a component that does not exist in the given tuple");
  return impl::find_const_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
inline static constexpr auto& find_component(tuple<Components...>& tuple) {
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

  inline constexpr TaggedComponent(component_type component) : component_(component) {
  }

  /// \brief Default copy/move/assign constructors
  inline constexpr TaggedComponent(const TaggedComponent&) = default;
  inline constexpr TaggedComponent(TaggedComponent&&) = default;
  inline constexpr TaggedComponent& operator=(const TaggedComponent&) = default;
  inline constexpr TaggedComponent& operator=(TaggedComponent&&) = default;

  inline constexpr const component_type& component() const {
    // Our lifetime should be at least as long as the component's
    return component_;
  }

  inline constexpr component_type& component() {
    return component_;
  }

  component_type component_;
};  // TaggedComponent

template <typename Tag, typename Type>
inline TaggedComponent<Tag, Type> apply_tag(Type t) {
  return TaggedComponent<Tag, Type>(t);
}

}  // namespace impl

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
  inline void reset_active_type_impl(std::index_sequence<Ids...>) {
    ((active_index_ == Ids
          ? (storage_.template get<Ids>() = std::decay_t<decltype(storage_.template get<Ids>())>{}, true)
          : false),
     ...);
  }

  // Function to reset the current active type to its default value
  inline void reset_active_type() {
    reset_active_type_impl(std::make_index_sequence<sizeof...(Alts)>{});
  }
  //@}

 public:
  /// \brief Default constructor initializes the first type as active
  inline constexpr variant() : storage_{}, active_index_{0} {
  }

  /// \brief Constructor for initializing with a specific type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  inline constexpr variant(const T& value) : storage_{}, active_index_{index_of<T>()} {
    storage_.template get<T>() = value;
  }

  /// \brief Get the active type index
  inline constexpr size_t index() const {
    return active_index_;
  }

  /// \brief Get the number of alternatives
  inline static constexpr size_t size() {
    return sizeof...(Alts);
  }

  template <class T>
  inline static constexpr size_t index_of() {
    return index_finder_v<T, Alts...>;
  }

  /// \brief Check if a specific type is active
  template <class T>
  inline constexpr bool holds_alternative() const {
    return active_index_ == index_of<T>();
  }

  template <size_t I>
  inline constexpr bool holds_alternative() const {
    return active_index_ == I;
  }

  /// \brief The J'th alternative type
  template <size_t J>
  using alternative_t = type_at_index_t<J, Alts...>;

  /// \brief Get the value of the active type
  template <class T>
  inline constexpr T& get() {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }
  template <class T>
  inline constexpr const T& get() const {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }

  /// \brief Get the value of the active type based on the active index
  template <size_t ActiveIdx>
  inline constexpr auto get() -> alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }
  template <size_t ActiveIdx>
  inline constexpr const auto get() const -> const alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }

  /// \brief Set a new active type, default-constructing the previous type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  inline constexpr void operator=(T const& value) {
    reset_active_type();
    active_index_ = index_of<T>();
    storage_.template get<T>() = value;
  }
};

//! \name Non-member functions
//@{

/// \brief Get the index of the given type
template <class T, class... Alts>
inline constexpr size_t index_of() {
  return variant<Alts...>::template index_of<T>();
}

/// \brief Check if a specific type is active
template <class T, class... Alts>
inline constexpr bool holds_alternative(const variant<Alts...>& var) {
  return var.template holds_alternative<T>();
}
template <size_t I, class... Alts>
inline constexpr bool holds_alternative(const variant<Alts...>& var) {
  return var.template holds_alternative<I>();
}


/// \brief Get the J'th alternative type TODO(palmerb4): Make independent of concrete variant instance
template <size_t J, class VariantType>
using variant_alternative_t = typename VariantType::template alternative_t<J>;

/// \brief Get the value of the active type
template <class T, class... Alts>
inline constexpr T& get(variant<Alts...>& var) {
  return var.template get<T>();
}
template <class T, class... Alts>
inline constexpr const T& get(const variant<Alts...>& var) {
  return var.template get<T>();
}

/// \brief Get the value of the active type based on the active index
template <size_t ActiveIdx, class... Alts>
inline constexpr auto& get(variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}
template <size_t ActiveIdx, class... Alts>
inline constexpr const auto& get(const variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}

// -------- variant_size
template <class T>
struct variant_size;  // primary

template <class... Alts>
struct variant_size<variant<Alts...>> {
  static constexpr std::size_t value = sizeof...(Alts);
};

template <class T>
static constexpr std::size_t variant_size_v = variant_size<T>::value;
//@}

/// \brief A variant_aggregate: A bag of compile-time tagged variants
/// In other words, a compile-time map of variants indexed by tag type.
template <typename VariantType, typename... Tags>
class variant_aggregate {
 public:
  using variant_t = VariantType;
  using TagsTuple = tuple<Tags...>;
  static constexpr size_t N = sizeof...(Tags);

  //! \name Constructors
  //@{

  /// \brief Default constructor

  inline constexpr variant_aggregate() = default;

  /// \brief Construct a variant_aggregate that has the given tagged variants
  inline constexpr variant_aggregate(std::array<variant_t, N> variants) : variants_(std::move(variants)) {
  }

  /// \brief Default copy/move/assign constructors
  inline constexpr variant_aggregate(const variant_aggregate&) = default;
  inline constexpr variant_aggregate(variant_aggregate&&) = default;
  inline constexpr variant_aggregate& operator=(const variant_aggregate&) = default;
  inline constexpr variant_aggregate& operator=(variant_aggregate&&) = default;
  //@}

  /// \brief Add a component (fluent interface):
  template <typename Tag>
  inline constexpr auto append(variant_t new_variant) const {
    // Copy the old variants into a new array with one extra slot
    std::array<variant_t, N + 1> new_variants;
    for (size_t i = 0; i < N; ++i) {
      new_variants[i] = variants_[i];
    }
    new_variants[N] = std::move(new_variant);

    using NewType = variant_aggregate<VariantType, Tags..., Tag>;
    return NewType(new_variants);
  }

  /// \brief The I'th tag type
  template <size_t I>
    requires(sizeof...(Tags) > 0)
  using tag_t = tuple_element_t<I, TagsTuple>;

  /// \brief Fetch the I'th component (compile-time index)
  template <size_t I>
  inline constexpr const variant_t& get() const {
    return variants_[I];
  }
  template <size_t I>
  inline constexpr variant_t& get() {
    return variants_[I];
  }

  /// \brief Fetch the I'th component (runtime index)

  inline constexpr const variant_t& get(size_t I) const {
    return variants_[I];
  }

  inline constexpr variant_t& get(size_t I) {
    return variants_[I];
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  inline constexpr const variant_t& get() const {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  inline constexpr variant_t& get() {
    constexpr size_t index = index_finder_v<Tag, Tags...>;
    return variants_[index];
  }

  /// \brief Check if we have a component with the given Tag
  template <typename Tag>
  inline static constexpr bool has() {
    return contains_type_v<Tag, Tags...>;
  }

  /// \brief Get the number of components in this variant_aggregate

  inline static constexpr size_t size() {
    return N;
  }

  //! \name Private members (no touch)
  //@{

  std::array<variant_t, N> variants_;
  //@}
};  // variant_aggregate

/// \brief Canonical way to construct a variant_aggregate
template <typename VariantType>
inline auto make_variant_aggregate() {
  return variant_aggregate<VariantType>();
}

/// \brief Fetch the variant corresponding to the given Tag
template <typename Tag, typename VariantType, typename... Tags>
inline constexpr const VariantType& get(const variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<Tag>();
}

/// \brief Fetch the variant corresponding to the given Tag
template <typename Tag, typename VariantType, typename... Tags>
inline constexpr VariantType& get(variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<Tag>();
}

/// \brief Fetch the variant at index I
template <size_t I, typename VariantType, typename... Tags>
inline constexpr const VariantType& get(const variant_aggregate<VariantType, Tags...>& v_agg) {
  return v_agg.template get<I>();
}
template <size_t I, typename VariantType, typename... Tags>
inline constexpr VariantType& get(variant_aggregate<VariantType, Tags...>& v_agg) {
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

/// \brief Check if a variant_aggregate have a variant with the given Tag
template <typename Tag, typename VariantType, typename... Tags>
inline constexpr bool has(const variant_aggregate<VariantType, Tags...>& /*v_agg*/) {
  return variant_aggregate<VariantType, Tags...>::template has<Tag>();
}

/// \brief Check if a variant aggregate type has a component with the given Tag usage variant_aggregate_has_v<Tag,
/// VarAggType>
template <typename Tag, typename VarAggType>
struct variant_aggregate_has {
  static constexpr bool value = VarAggType::template has<Tag>();
};
//
template <typename Tag, typename VarAggType>
static constexpr bool variant_aggregate_has_v = variant_aggregate_has<Tag, VarAggType>::value;

/// \brief Add a new component to an existing aggregate (fluent interface)
template <typename Tag, typename VariantType, typename... Tags>
inline constexpr auto append(const variant_aggregate<VariantType, Tags...>& v_agg, VariantType new_variant) {
  return v_agg.template append<Tag>(std::move(new_variant));
}

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
template <typename... TaggedComponents>
class aggregate {
 public:
  static_assert(impl::all_have_tags<TaggedComponents...>, "All of the given components must have tags.");
  using TaggedComponentsTuple = tuple<TaggedComponents...>;

  //! \name Constructors
  //@{

  /// \brief Default constructor

  inline constexpr aggregate() = default;

  /// \brief Construct an aggregate that has the given components
  inline constexpr aggregate(TaggedComponentsTuple tagged_components)
    requires(sizeof...(TaggedComponents) > 0)
      : tagged_components_(std::move(tagged_components)) {
  }

  /// \brief Default copy/move/assign constructors
  inline constexpr aggregate(const aggregate&) = default;
  inline constexpr aggregate(aggregate&&) = default;
  inline constexpr aggregate& operator=(const aggregate&) = default;
  inline constexpr aggregate& operator=(aggregate&&) = default;
  //@}

  /// \brief Add a component (fluent interface):
  template <typename Tag, typename NewComponent>
  inline constexpr auto append(NewComponent new_component) const {
    impl::TaggedComponent<Tag, NewComponent> new_tagged_comp(std::move(new_component));
    auto new_tuple = tuple_cat(tagged_components_, ::make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = aggregate<TaggedComponents..., decltype(new_tagged_comp)>;
    return NewType(new_tuple);
  }

  /// \brief The I'th tag type
  template <size_t I>
    requires(sizeof...(TaggedComponents) > 0)
  using tag_t = typename tuple_element_t<I, TaggedComponentsTuple>::tag_type;

  /// \brief Fetch the I'th component
  template <size_t I>
  inline constexpr const auto& get() const {
    return tagged_components_.template get<I>().component();
  }
  template <size_t I>
  inline constexpr auto& get() {
    return tagged_components_.template get<I>().component();
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  inline constexpr const auto& get() const {
    return impl::find_component<Tag>(tagged_components_).component();
  }
  template <typename Tag>
  inline constexpr auto& get() {
    return impl::find_component<Tag>(tagged_components_).component();
  }

  /// \brief Check if we have a component with the given Tag
  template <typename Tag>
  inline static constexpr bool has() {
    return impl::has_component_v<Tag, TaggedComponents...>;
  }

  /// \brief Get the number of components in this aggregate

  inline static constexpr size_t size() {
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

inline constexpr auto make_aggregate() {
  return aggregate<>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
inline constexpr const auto& get(const aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
inline constexpr auto& get(aggregate<Components...>& agg) {
  return agg.template get<Tag>();
}

/// \brief Fetch the component at index I
template <size_t I, typename... Components>
inline constexpr const auto& get(const aggregate<Components...>& agg) {
  return agg.template get<I>();
}
template <size_t I, typename... Components>
inline constexpr auto& get(aggregate<Components...>& agg) {
  return agg.template get<I>();
}

/// \brief Check if an aggregate have a component with the given Tag
template <typename Tag, typename... Components>
inline constexpr bool has(const aggregate<Components...>& /*agg*/) {
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
inline constexpr auto append(const aggregate<Components...>& agg, NewComponent new_component) {
  return agg.template append<Tag>(std::move(new_component));
}

/// \brief The I'th aggregate tag
template <size_t I, typename AggType>
struct aggregate_tag;

template <size_t I, typename... Components>
struct aggregate_tag<I, aggregate<Components...>> {
  using type = type_at_index_t<I, Components...>::tag_type;
};

template <size_t I, typename AggType>
using aggregate_tag_t = aggregate_tag<I, AggType>::type;

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

template<typename T>
struct NonOwningVector {
  // Just a pointer and a size
  T* data;
  std::size_t size;

  inline NonOwningVector() = default;

  inline NonOwningVector(T* data_ptr, std::size_t sz) : data(data_ptr), size(sz) {
  }

  inline NonOwningVector(std::vector<T>& vec) : data(vec.data()), size(vec.size()) {
  }

  // Copy/move constructors and assignments
  inline NonOwningVector(const NonOwningVector& other) = default;
  inline NonOwningVector(NonOwningVector&& other) = default;
  inline NonOwningVector& operator=(const NonOwningVector& other) = default;
  inline NonOwningVector& operator=(NonOwningVector&& other) = default;

  inline const T& operator[](std::size_t i) const {
    return data[i];
  }
  inline T& operator[](std::size_t i) {
    return data[i];
  }
};



class Accessor {
 public:
  Accessor() = default;

  // explicit inline Accessor(double shared_value) : our_type_(variant_t::SHARED), variant_(shared_value) {
  // }

  // explicit inline Accessor(const std::vector<double>& vec) : our_type_(variant_t::VECTOR), variant_(vec) {
  // }


  // explicit inline Accessor(double shared_value) : variant_(shared_value), is_shared_(true) {
  // }

  // explicit inline Accessor(std::vector<double>& vec) : variant_(vec), is_shared_(false) {
  // }


  explicit inline Accessor(double shared_value) : shared_value_(shared_value), vector_{}, is_shared_(true) {
  }

  explicit inline Accessor(std::vector<double>& vec) : shared_value_{}, vector_(vec), is_shared_(false) {
  }



  inline const double& operator()(std::size_t i) const {
    return is_shared_ ? shared_value_ : vector_[i];

    // if (our_type_ == variant_t::SHARED) {
    //   return shared_value_;
    // } else if (our_type_ == variant_t::VECTOR) {
    //   return vector_[i];
    // } else if (our_type_ == variant_t::MAPPED_SCALAR) {
    //   return part_mapped_scalars_.at(parts_[i]);
    // } else {
    //   return part_mapped_vectors_.at(parts_[i])[i];
    // }
  }

  // const variant_t our_type_;
  // using actual_variant_t = variant<double, std::vector<double>>;
  // const actual_variant_t variant_;
  double shared_value_;
  std::vector<double> vector_;
  const bool is_shared_;
};

// class Accessor {
//  public:
//   Accessor() = default;

//   explicit inline Accessor(double shared_value)
//       : our_type_(variant_t::SHARED), shared_value_(shared_value), vector_(), parts_(), part_mapped_scalars_(),
//       part_mapped_vectors_(), flip_point_() {}

//   explicit inline Accessor(const std::vector<double>& vec)
//       : our_type_(variant_t::VECTOR), shared_value_(), vector_(vec), parts_(), part_mapped_scalars_(),
//       part_mapped_vectors_(), flip_point_() {}

//   inline Accessor(const std::vector<double>& vec1, const std::vector<double>& vec2, int flip_point)
//       : our_type_(variant_t::CONDITIONAL), shared_value_(), vector_(), parts_(), part_mapped_scalars_(),
//       part_mapped_vectors_(), vec1_(vec1), vec2_(vec2), flip_point_(flip_point) {}

//   inline Accessor(std::vector<int> parts, std::map<int, double> part_mapped_scalars)
//       : our_type_(variant_t::MAPPED_SCALAR), shared_value_(), vector_(), parts_(std::move(parts)),
//       part_mapped_scalars_(std::move(part_mapped_scalars)), part_mapped_vectors_(), flip_point_() {}

//   inline Accessor(std::vector<int> parts, std::map<int,  std::vector<double>> part_mapped_vectors)
//       : our_type_(variant_t::MAPPED_VECTOR), shared_value_(), vector_(), parts_(std::move(parts)),
//       part_mapped_scalars_(), part_mapped_vectors_(std::move(part_mapped_vectors)), flip_point_() {}

//   inline const double& operator()(std::size_t i) const {
//     if (our_type_ == variant_t::SHARED) {
//       return shared_value_;
//     } else {
//       return vector_[i];
//     }
//     // if (our_type_ == variant_t::SHARED) {
//     //   return shared_value_;
//     // } else if (our_type_ == variant_t::VECTOR) {
//     //   return vector_[i];
//     // } else if (our_type_ == variant_t::MAPPED_SCALAR) {
//     //   return part_mapped_scalars_.at(parts_[i]);
//     // } else {
//     //   return part_mapped_vectors_.at(parts_[i])[i];
//     // }
//   }

//  private:
//   const variant_t our_type_;
//   const double shared_value_;
//   const std::vector<double> vector_;

//   const std::vector<int> parts_;
//   const std::map<int, double> part_mapped_scalars_;
//   const std::map<int,  std::vector<double>> part_mapped_vectors_;

//   const std::vector<double> vec1_;
//   const std::vector<double> vec2_;
//   const int flip_point_;
// };

class ScalarAccessor {
 public:
  inline ScalarAccessor() = default;

  explicit inline ScalarAccessor(double shared_value) : shared_value_(shared_value) {
  }

  // Copy constructor
  inline ScalarAccessor(const ScalarAccessor& other) : shared_value_(other.shared_value_) {
  }

  inline ScalarAccessor& operator=(const ScalarAccessor& other) {
    if (this != &other) {
      shared_value_ = other.shared_value_;
    }
    return *this;
  }

  inline const double& operator()(std::size_t i) const {
    return shared_value_;
  }

 private:
  double shared_value_;
};

class VectorAccessor {
 public:
  VectorAccessor() = default;

  explicit inline VectorAccessor(const std::vector<double>& vec) : vector_(vec) {
  }

  // Copy constructor
  inline VectorAccessor(const VectorAccessor& other) : vector_(other.vector_) {
  }

  inline VectorAccessor& operator=(const VectorAccessor& other) {
    if (this != &other) {
      vector_ = other.vector_;
    }
    return *this;
  }

  inline const double& operator()(std::size_t i) const {
    return vector_[i];
  }

 private:
  std::vector<double> vector_;
};

// ----- Data setup -----
static void randomize(std::vector<double>& v, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& x : v) x = dist(rng);
}

struct Coeffs6 {
  std::vector<double> a, b, c, d, e, f, x;
  std::size_t N{};
};

static Coeffs6& get_coeffs(std::size_t N) {
  static Coeffs6 C;
  static std::size_t cached = 0;
  if (cached != N) {
    C.a.assign(N, 0.0);
    C.b.assign(N, 0.0);
    C.c.assign(N, 0.0);
    C.d.assign(N, 0.0);
    C.e.assign(N, 0.0);
    C.f.assign(N, 0.0);
    C.x.assign(N, 0.0);
    C.N = N;
    randomize(C.a, 101);
    randomize(C.b, 202);
    randomize(C.c, 303);
    randomize(C.d, 404);
    randomize(C.e, 505);
    randomize(C.f, 606);
    randomize(C.x, 707);
    cached = N;
  }
  return C;
}

// ----- Kernels (unchanged) -----
struct TagX;
struct TagA;
struct TagB;
struct TagC;
struct TagD;
struct TagE;
struct TagF;

// Agg-based Horner (6 coefficients)
template <typename Agg>
static inline double poly6_sum_agg(const Agg& agg, std::size_t N) {
  double s = 0.0;
  const auto& x = get<TagX>(agg);
  const auto& a = get<TagA>(agg);
  const auto& b = get<TagB>(agg);
  const auto& c = get<TagC>(agg);
  const auto& d = get<TagD>(agg);
  const auto& e = get<TagE>(agg);
  const auto& f = get<TagF>(agg);
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x(i);
    s += (((((a(i) * xi + b(i)) * xi + c(i)) * xi + d(i)) * xi + e(i)) * xi + f(i));
  }
  return s;
}

#define ALT_UNROLL(Ax, Aa, Ab, Ac, Ad, Ae, Af)                                                                         \
  if (holds_alternative<Ax>(vx) && holds_alternative<Aa>(va) &&                                    \
      holds_alternative<Ab>(vb) && holds_alternative<Ac>(vc) &&                                    \
      holds_alternative<Ad>(vd) && holds_alternative<Ae>(ve) &&                                    \
      holds_alternative<Af>(vf)) {                                                                           \
    const auto& x_accessor = get<Ax>(vx);                                                                    \
    const auto& a_accessor = get<Aa>(va);                                                                    \
    const auto& b_accessor = get<Ab>(vb);                                                                    \
    const auto& c_accessor = get<Ac>(vc);                                                                    \
    const auto& d_accessor = get<Ad>(vd);                                                                    \
    const auto& e_accessor = get<Ae>(ve);                                                                    \
    const auto& f_accessor = get<Af>(vf);                                                                    \
    for (std::size_t i = 0; i < N; ++i) {                                                                              \
      const double xi = x_accessor(i);                                                                                 \
      s +=                                                                                                             \
          (((((a_accessor(i) * xi + b_accessor(i)) * xi + c_accessor(i)) * xi + d_accessor(i)) * xi + e_accessor(i)) * \
               xi +                                                                                                    \
           f_accessor(i));                                                                                             \
    }                                                                                                                  \
    return s;                                                                                                          \
  }

// #define ALT_UNROLL(Ax, Aa, Ab, Ac, Ad, Ae, Af)                                                                         \
//   if (holds_alternative<Ax>(vx) && holds_alternative<Aa>(va) &&                                    \
//       holds_alternative<Ab>(vb) && holds_alternative<Ac>(vc) &&                                    \
//       holds_alternative<Ad>(vd) && holds_alternative<Ae>(ve) &&                                    \
//       holds_alternative<Af>(vf)) {                                                                           \
//     static const auto &x_accessor = get<Ax>(vx);                                                                    \
//     static const auto &a_accessor = get<Aa>(va);                                                                    \
//     static const auto &b_accessor = get<Ab>(vb);                                                                    \
//     static const auto &c_accessor = get<Ac>(vc);                                                                    \
//     static const auto &d_accessor = get<Ad>(vd);                                                                    \
//     static const auto &e_accessor = get<Ae>(ve);                                                                    \
//     static const auto &f_accessor = get<Af>(vf);                                                                    \
//     /* Make a concrete aggregate use static to avoid cipying vectors */ \
//     static auto agg = make_aggregate()                                                                             \
//                    .template append<TagX>(x_accessor)                                                        \
//                    .template append<TagA>(a_accessor)                                                        \
//                    .template append<TagB>(b_accessor)                                                        \
//                    .template append<TagC>(c_accessor)                                                        \
//                    .template append<TagD>(d_accessor)                                                        \
//                    .template append<TagE>(e_accessor)                                                        \
//                    .template append<TagF>(f_accessor);                                                       \
//     return poly6_sum_agg(agg, N);                                                                                 \
//   }



// template <typename VarAgg>
// static inline double poly6_sum_v_agg(const VarAgg& v_agg, std::size_t N) {
//   double s = 0.0;
//   const auto& vx = get<TagX>(v_agg);
//   const auto& va = get<TagA>(v_agg);
//   const auto& vb = get<TagB>(v_agg);
//   const auto& vc = get<TagC>(v_agg);
//   const auto& vd = get<TagD>(v_agg);
//   const auto& ve = get<TagE>(v_agg);
//   const auto& vf = get<TagF>(v_agg);
//   ALT_UNROLL(0, 0, 0, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 0, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 0, 1, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 0, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(0, 1, 1, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 0, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 0, 1, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 0, 1, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 0, 1, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 0, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 0, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 0, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 0, 1, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 1, 0, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 1, 0, 1)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 1, 1, 0)
//   else
//   ALT_UNROLL(1, 1, 1, 1, 1, 1, 1)
//   else {
//     throw std::runtime_error("Unhandled variant combination in poly6_sum_v_agg");
//   }
//   return s;
// }

template <typename VarAgg>
static inline double poly6_sum_v_agg(const VarAgg& v_agg, std::size_t N) {
  double s = 0.0;
  const auto& vx = get<TagX>(v_agg);
  const auto& va = get<TagA>(v_agg);
  const auto& vb = get<TagB>(v_agg);
  const auto& vc = get<TagC>(v_agg);
  const auto& vd = get<TagD>(v_agg);
  const auto& ve = get<TagE>(v_agg);
  const auto& vf = get<TagF>(v_agg);
  
  for (std::size_t i = 0; i < N; ++i) {

// clang-format off
    const double xi = holds_alternative<ScalarAccessor>(vx) ? get<ScalarAccessor>(vx)(i) : get<VectorAccessor>(vx)(i); 
    const double ai = holds_alternative<ScalarAccessor>(va) ? get<ScalarAccessor>(va)(i) : get<VectorAccessor>(va)(i); 
    const double bi = holds_alternative<ScalarAccessor>(vb) ? get<ScalarAccessor>(vb)(i) : get<VectorAccessor>(vb)(i); 
    const double ci = holds_alternative<ScalarAccessor>(vc) ? get<ScalarAccessor>(vc)(i) : get<VectorAccessor>(vc)(i); 
    const double di = holds_alternative<ScalarAccessor>(vd) ? get<ScalarAccessor>(vd)(i) : get<VectorAccessor>(vd)(i); 
    const double ei = holds_alternative<ScalarAccessor>(ve) ? get<ScalarAccessor>(ve)(i) : get<VectorAccessor>(ve)(i); 
    const double fi = holds_alternative<ScalarAccessor>(vf) ? get<ScalarAccessor>(vf)(i) : get<VectorAccessor>(vf)(i); 
// clang-format on
    s += (((((ai * xi + bi) * xi + ci) * xi + di) * xi + ei) * xi + fi);
  }
  return s;
}

// Accessor-based Horner (6 coefficients)
template <class AAcc, class BAcc, class CAcc, class DAcc, class EAcc, class FAcc, class XAcc>
static inline double poly6_sum_accessor(const AAcc& a, const BAcc& b, const CAcc& c, const DAcc& d, const EAcc& e,
                                        const FAcc& f, const XAcc& x, std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x(i);
    s += (((((a(i) * xi + b(i)) * xi + c(i)) * xi + d(i)) * xi + e(i)) * xi + f(i));
  }
  return s;
}

// Direct vectors
static inline double poly6_sum_direct_vecs(const std::vector<double>& a, const std::vector<double>& b,
                                           const std::vector<double>& c, const std::vector<double>& d,
                                           const std::vector<double>& e, const std::vector<double>& f,
                                           const std::vector<double>& x, std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x[i];
    s += (((((a[i] * xi + b[i]) * xi + c[i]) * xi + d[i]) * xi + e[i]) * xi + f[i]);
  }
  return s;
}

// Direct scalars
static inline double poly6_sum_direct_scalars(double a, double b, double c, double d, double e, double f, double x,
                                              std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x;
    s += (((((a * xi + b) * xi + c) * xi + d) * xi + e) * xi + f);
  }
  return s;
}

// ----- Simple timing harness -----
using our_clock_t = std::chrono::steady_clock;

template <class Fn>
static double time_avg_ns(Fn&& fn, int iters) {
  using namespace std::chrono;

  double s = 0.0;
  for (int warm = 0; warm < 1000; ++warm) {
    s += fn();
  }
  std::cout << "warm sum=" << std::setprecision(17) << s << '\n';

  long double total_ns = 0.0L;
  for (int k = 0; k < iters; ++k) {
    auto t0 = our_clock_t::now();
    s += fn();
    auto t1 = our_clock_t::now();
    // Print AFTER stopping timer to avoid timing I/O but still defeat DCE
    total_ns += duration_cast<nanoseconds>(t1 - t0).count();
  }
  std::cout << "sum=" << std::setprecision(17) << s << '\n';
  return static_cast<double>(total_ns / iters);
}

int main(int argc, char** argv) {
  // Parameters
  std::size_t length = 10000;  // elements
  int iters = 1000;                 // timing iterations
  if (argc > 1) {
    length = static_cast<std::size_t>(std::stoull(argv[1]));
  }
  if (argc > 2) {
    iters = std::max(1, std::stoi(argv[2]));
  }

  auto& C = get_coeffs(length);

  // -1.0) V-Agg-based Horner (6 coefficients)
  using variant_t = variant<ScalarAccessor, VectorAccessor>;
  auto v_agg = make_variant_aggregate<variant_t>()
                   .append<TagX>(VectorAccessor(C.x))
                   .append<TagA>(VectorAccessor(C.a))
                   .append<TagB>(VectorAccessor(C.b))
                   .append<TagC>(VectorAccessor(C.c))
                   .append<TagD>(VectorAccessor(C.d))
                   .append<TagE>(VectorAccessor(C.e))
                   .append<TagF>(VectorAccessor(C.f));
  double avg_ns_v_agg = time_avg_ns([&] { return poly6_sum_v_agg(v_agg, C.N); }, iters);

  // -1.5) Scalar V-Agg-based Horner (6 coefficients)
  auto v_agg_sca = make_variant_aggregate<variant_t>()
                       .append<TagX>(ScalarAccessor(0.77))
                       .append<TagA>(ScalarAccessor(0.11))
                       .append<TagB>(ScalarAccessor(0.22))
                       .append<TagC>(ScalarAccessor(0.33))
                       .append<TagD>(ScalarAccessor(0.44))
                       .append<TagE>(ScalarAccessor(0.55))
                       .append<TagF>(ScalarAccessor(0.66));
  double avg_ns_v_agg_sca = time_avg_ns([&] { return poly6_sum_v_agg(v_agg_sca, C.N); }, iters);

  // 0) Agg-based Horner (6 coefficients)
  auto agg = make_aggregate()
                 .append<TagX>(Accessor(C.x))
                 .append<TagA>(Accessor(C.a))
                 .append<TagB>(Accessor(C.b))
                 .append<TagC>(Accessor(C.c))
                 .append<TagD>(Accessor(C.d))
                 .append<TagE>(Accessor(C.e))
                 .append<TagF>(Accessor(C.f));
  double avg_ns_agg = time_avg_ns([&] { return poly6_sum_agg(agg, C.N); }, iters);

  /// 0.5) Scalar Agg-based Horner (6 coefficients)
  auto agg_sca = make_aggregate()
                     .append<TagX>(ScalarAccessor(0.77))
                     .append<TagA>(ScalarAccessor(0.11))
                     .append<TagB>(ScalarAccessor(0.22))
                     .append<TagC>(ScalarAccessor(0.33))
                     .append<TagD>(ScalarAccessor(0.44))
                     .append<TagE>(ScalarAccessor(0.55))
                     .append<TagF>(ScalarAccessor(0.66));
  double avg_ns_agg_sca = time_avg_ns([&] { return poly6_sum_agg(agg_sca, C.N); }, iters);

  // 1) 6 Accessors backed by vectors (each Accessor copies its vector by design)
  Accessor av(C.a), bv(C.b), cv(C.c), dv(C.d), ev(C.e), fv(C.f), xv(C.x);
  double avg_ns_acc_vec = time_avg_ns([&] { return poly6_sum_accessor(av, bv, cv, dv, ev, fv, xv, C.N); }, iters);

  // 1) 6 Accessors backed by vectors (each Accessor copies its vector by design)
  VectorAccessor avx(C.a), bvx(C.b), cvx(C.c), dvx(C.d), evx(C.e), fvx(C.f), xvx(C.x);
  double avg_ns_acc_vec_explicit =
      time_avg_ns([&] { return poly6_sum_accessor(avx, bvx, cvx, dvx, evx, fvx, xvx, C.N); }, iters);

  // 2) 6 Accessors backed by shared scalars
  Accessor as(0.11), bs(0.22), cs(0.33), ds(0.44), es(0.55), fs(0.66), xs(0.77);
  double avg_ns_acc_sca = time_avg_ns([&] { return poly6_sum_accessor(as, bs, cs, ds, es, fs, xs, C.N); }, iters);

  // 2) 6 Accessors backed by shared scalars
  ScalarAccessor asx(0.11), bsx(0.22), csx(0.33), dsx(0.44), esx(0.55), fsx(0.66), xsx(0.77);
  double avg_ns_acc_sca_explicit =
      time_avg_ns([&] { return poly6_sum_accessor(asx, bsx, csx, dsx, esx, fsx, xsx, C.N); }, iters);

  // 3) Direct vectors (no accessors)
  double avg_ns_dir_vec =
      time_avg_ns([&] { return poly6_sum_direct_vecs(C.a, C.b, C.c, C.d, C.e, C.f, C.x, C.N); }, iters);

  // 4) Direct scalars (no accessors)
  const double a0 = 0.11, b0 = 0.22, c0 = 0.33, d0 = 0.44, e0 = 0.55, f0 = 0.66, x0 = 0.77;
  double avg_ns_dir_sca = time_avg_ns([&] { return poly6_sum_direct_scalars(a0, b0, c0, d0, e0, f0, x0, C.N); }, iters);

  // Report (averages)
  auto to_ms = [](double ns) { return ns / 1e6; };
  // clang-format off
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nAverages over " << iters << " iteration(s) for N=" << length << ":\n";
  std::cout << "  Agg×6 (vector-backed): " << to_ms(avg_ns_agg) << " ms/iter | relative: " << (avg_ns_agg / avg_ns_dir_vec) << "\n";
  std::cout << "  Agg×6 (scalar-backed): " << to_ms(avg_ns_agg_sca) << " ms/iter | relative: " << (avg_ns_agg_sca / avg_ns_dir_sca) << "\n";
  std::cout << "  V-Agg×6 (vector-backed): " << to_ms(avg_ns_v_agg) << " ms/iter | relative: " << (avg_ns_v_agg / avg_ns_dir_vec) << "\n";
  std::cout << "  V-Agg×6 (scalar-backed): " << to_ms(avg_ns_v_agg_sca) << " ms/iter | relative: " << (avg_ns_v_agg_sca / avg_ns_dir_sca) << "\n";
  std::cout << "  Accessor×6 (vector-backed): " << to_ms(avg_ns_acc_vec) << " ms/iter | relative: " << (avg_ns_acc_vec / avg_ns_dir_vec) << "\n";
  std::cout << "  Accessor×6 (scalar-backed): " << to_ms(avg_ns_acc_sca) << " ms/iter | relative: " << (avg_ns_acc_sca / avg_ns_dir_sca) << "\n";
  std::cout << "  Accessor×6 (vector-backed explicit): " << to_ms(avg_ns_acc_vec_explicit) << " ms/iter | relative: " << (avg_ns_acc_vec_explicit / avg_ns_dir_vec) << "\n";
  std::cout << "  Accessor×6 (scalar-backed explicit): " << to_ms(avg_ns_acc_sca_explicit) << " ms/iter | relative: " << (avg_ns_acc_sca_explicit / avg_ns_dir_sca) << "\n";
  std::cout << "  Direct vectors: " << to_ms(avg_ns_dir_vec) << " ms/iter\n";
  std::cout << "  Direct scalars: " << to_ms(avg_ns_dir_sca) << " ms/iter\n";
  // clang-format on
  return 0;
}
