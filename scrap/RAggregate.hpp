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

#ifndef MUNDY_MESH_RAGGREGATES_HPP_
#define MUNDY_MESH_RAGGREGATES_HPP_

// C++ core
#include <tuple>
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Trilinos
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_topology/topology.hpp>      // for stk::topology::topology_t

// Mundy
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_core/tuple.hpp>          // for mundy::core::tuple
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/fmt_stk_types.hpp>  // for STK-compatible fmt::format

namespace mundy {

namespace mesh {

/// \brief Runtime aggregates
// Runtime aggregates perform pack and unpack.

struct FieldAccessor {
  std::string name = "FieldAccessor";
};
struct ScalarAccessor {
  std::string name = "ScalarAccessor";
};
struct AABBAccessor {
  std::string name = "AABBAccessor";
};
template <size_t N>
struct VectorAccessor {
  std::string name = "VectorAccessorN";
};

namespace accessor_t {
struct FIELD;

struct SCALAR;

template <size_t N>
struct VECTOR;

using VECTOR3 = VECTOR<3>;

template <size_t N, size_t M>
struct MATRIX;

using MATRIX33 = MATRIX<3, 3>;

struct AABB;
}  // namespace accessor_t

template <typename T>
struct to_accessor_type;

template <typename T>
using to_accessor_type_t = to_accessor_type<T>::type;

template <>
struct to_accessor_type<accessor_t::FIELD> {
  using type = FieldAccessor;
};
template <>
struct to_accessor_type<accessor_t::SCALAR> {
  using type = ScalarAccessor;
};
template <size_t N>
struct to_accessor_type<accessor_t::VECTOR<N>> {
  using type = VectorAccessor<N>;
};
template <>
struct to_accessor_type<accessor_t::AABB> {
  using type = AABBAccessor;
};

// ragg.get_accessor<accessor_t::ENTITY_FIELD_DATA<T>>(rank, name) -> FieldAccessor
// ragg.get_accessor<accessor_t::SCALAR<T>>(rank, name) -> ScalarAccessor
// ragg.get_accessor<accessor_t::AABB<T>>(rank, name) -> AABBAccessor
// ragg.get_accessor<accessor_t::USER_TYPE, T>(rank, name) -> TheirCustomAccessor

// every accessor has at most 4 types:
// - shared single value
// - singe value per part
// - field
// - one field per part

// auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
//             .add_component<CENTER>(center_accessor)
//             .add_component<COLLISION_RADIUS>(radius_accessor);

RuntimeAggregate ragg = make_runtime_aggregate(bulk_data, selector, stk::topology::PARTICLE)
                            .add_accessor(NODE_RANK, "OUR_CENTER", center_accessor)
                            .add_accessor(ELEM_RANK, "OUR_RADIUS", radius_accessor);

std::map<std::string, std::string> rename_map{{"CENTER", "OUR_CENTER"}, {"RADIUS", "OUR_RADIUS"}};

// Option 1: Runtime aggregate within kernel
auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
               .add_accessor<CENTER>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]))
               .add_accessor<RADIUS>(ragg.get_accessor<accessor_t::VECTOR3<double>>(ELEM_RANK, rename_map["RADIUS"]));
stk::mesh::for_each_entity_run(ngp_mesh, KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index)
                                             stk::mesh::FastMeshIndex center_node_index =
                                                 ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
                               auto center = agg.get<CENTER>(center_node_index);
                               auto radius = agg.get<RADIUS>(sphere_index); center += radius[0];);

// Option 2: Compile-time aggregate with connectivity helper
auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
               .add_accessor<CENTER>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]))
               .add_accessor<RADIUS>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["RADIUS"]));
agg.for_each(KOKKOS_LAMBDA(auto& sphere) auto center = sphere.get<CENTER>(0 /* node ord */);
             auto radius = sphere.get<RADIUS>(); center += radius[0];);

// Option 3: No aggregates within kernels
auto sphere_centers = ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]);
auto sphere_radii = ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["RADIUS"]);

stk::mesh::for_each_entity_run(ngp_mesh, KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index)
                                             stk::mesh::FastMeshIndex center_node_index =
                                                 ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
                               auto center = sphere_centers(center_node_index);
                               auto radius = sphere_radii(sphere_index); center += radius[0];);

class RuntimeAggregate {
 public:
  RuntimeAggregate() = default;
  RuntimeAggregate(stk::topology top) : rank_(top.rank()), topology_(top) {
  }
  RuntimeAggregate(stk::mesh::EntityRank rank) : rank_(rank), topology_(stk::topology::INVALID_TOPOLOGY) {
  }

  RuntimeAggregate& add_accessor(const stk::mesh::EntityRank rank, std::string name, AccessorBase accessor) {
    auto result = ranked_accessor_maps_[rank].insert({name, accessor});
    MUNDY_THROW_REQUIRE(
        result.second, std::logic_error,
        fmt::format("Accessor with rank {} and name '{}' already exists. No duplicates allowed.", rank, name));

    return this;
  }

  template <accessor_t A>
  auto get_accessor(const stk::mesh::EntityRank rank, std::string name) -> to_accessor_type_t<A> {
    // Check if an aggregate of the given rank/name exists
    MUNDY_THROW_REQUIRE(ranked_accessor_maps_[rank].contains(name), std::logic_error,
                        fmt::format("Failed to find aggregate of rank {} with name '{}'", rank, name));

    return dynamic_cast<to_accessor_type_t<A>>(ranked_accessor_maps_[rank][name]);
  }

 private:
  stk::mesh::EntitRank rank_;
  stk::topology topology_;

  using AggregateMap = std::map<std::string, AggregateBase>;
  AggregateMap ranked_accessor_maps_[stk::topology::NUM_RANKS];
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Given a runtime_aggregate containing N tagged runtime_accessors that are variants of V standard
accessor types, there are V choose N with replacement total combinations for the resulting compile-time aggregate type.
Can we enumerate this list explicitly to map our runtime aggregate to its accessor? Let's pretend that N is known at
compile-time.
*/
using RuntimeAccessor = std::variant<A1, A2, A3>;

template <unsigned NumAccessors>
class RuntimeAggregate {
  static constexpr num_accessors = NumAccessors;
  static constexpr num_variant_types = 3;
  static constexpr num_combinations = factorial(num_variant_types + num_accessors - 1) / factorial(num_accessors);

  [[A1, A2, A3], [A1, A2, A3], [A1, A2, A3], ..., [A1, A2, A3]]

  ...
};

/// Static functor wrapper
template <typename Functor, unsigned Value>
struct FunctorWrapper {
  static void apply(const Functor& functor) {
    using type = value_to_type_t<Value>;
    functor(type{});
  }
};

template <typename Functor, unsigned... Values>
const auto get_functor_jump_table_impl(std::integer_sequence<unsigned, Values...> /* int_seq */) {
  static constexpr void (*jump_table[])(const Functor& functor) = {FunctorWrapper<Functor, Values>::apply...};
  return jump_table;
}

template <typename Functor>
const auto get_functor_jump_table() {
  return get_functor_jump_table_impl<Functor>(std::make_integer_sequence<unsigned, static_cast<unsigned>(1000)>{});
}

template <typename Functor>
void run(unsigned runtime_value, const Functor& functor) {
  auto jump_table = get_functor_jump_table<Functor>();
  jump_table[runtime_value](functor);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// How to create a compile-time set of all combinations of N choose R with replacement:
#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

// ---------- compile-time binomial ----------
constexpr unsigned long long binom(unsigned int n, unsigned int k) {
  if (k > n) return 0ULL;
  if (k > n - k) k = n - k;
  unsigned long long r = 1;
  for (unsigned int i = 1; i <= k; ++i) {
    // exact division in integers
    r = (r * (n - k + i)) / i;
  }
  return r;
}

// ---------- unrank the k-th R-combination from {0..M} (no replacement) in colex ----------
// Returns strictly increasing b[0..R-1], where 0 <= b0 < ... < b{R-1} <= M.
template <int R>
constexpr std::array<int, R> unrank_comb_norep(unsigned int M, unsigned long long k) {
  static_assert(R >= 0);
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

// ---------- map back to weakly increasing a[i] = b[i] - i ----------
template <int N, int R>
consteval std::array<int, R> ith_multicomb(unsigned long long k) {
  static_assert(N >= 0 && R >= 0);
  constexpr unsigned int M = static_cast<unsigned int>(N + R - 1);  // upper bound for b
  auto b = unrank_comb_norep<R>(M, k);
  std::array<int, R> a{};
  for (int i = 0; i < R; ++i) a[i] = b[i] - i;  // 0 <= a0 <= ... <= a{R-1} <= N-1
  return a;
}

// ---------- build the whole table at compile time ----------
template <int N, int R>
consteval auto all_multicomb() {
  constexpr auto CNT = binom(N + R - 1, R);
  std::array<std::array<int, R>, CNT> out{};
  for (unsigned long long k = 0; k < CNT; ++k) {
    out[k] = ith_multicomb<N, R>(k);
  }
  return out;
}

// ---------- example usage ----------
static_assert(binom(6, 3) == 20);
constexpr auto combos = all_multicomb<4, 3>();        // N=4 items, choose R=3 with replacement
static_assert(combos.size() == binom(4 + 3 - 1, 3));  // 20
// e.g., combos[0] == {0,0,0}, combos[1] == {0,0,1}, ..., combos.back() == {3,3,3}

int main() {
  constexpr auto combos = all_multicomb<4, 3>();  // N=4 items, choose R=3 with replacement
  for (int c = 0; c < combos.size(); ++c) {
    for (int d = 0; d < 3; ++d) {
      std::cout << combos[c][d];
    }
    std::cout << std::endl;
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// How to take a tuple of T types and generate all combinations of T choose R with replacement:

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

// ---------- compile-time binomial ----------
constexpr unsigned long long binom(unsigned int n, unsigned int k) {
  if (k > n) return 0ULL;
  if (k > n - k) k = n - k;
  unsigned long long r = 1;
  for (unsigned int i = 1; i <= k; ++i) r = (r * (n - k + i)) / i;
  return r;
}

// ---------- unrank the k-th R-combination (no replacement) in colex ----------
template <int R>
constexpr std::array<int, R> unrank_comb_norep(unsigned int M, unsigned long long k) {
  std::array<int, R> b{};
  int x = static_cast<int>(M);
  for (int i = R; i >= 1; --i) {
    while (binom(x, static_cast<unsigned>(i)) > k) --x;
    b[i - 1] = x;
    k -= binom(x, static_cast<unsigned>(i));
    --x;
  }
  return b;
}

// ---------- all weakly-increasing R-tuples over [0..N-1] (with replacement) ----------
template <int N, int R>
consteval auto all_multicomb() {
  constexpr auto CNT = binom(N + R - 1, R);
  constexpr unsigned int M = static_cast<unsigned int>(N + R - 1);
  std::array<std::array<int, R>, CNT> out{};
  for (unsigned long long k = 0; k < CNT; ++k) {
    auto b = unrank_comb_norep<R>(M, k);               // strict increasing in [0..M]
    for (int i = 0; i < R; ++i) out[k][i] = b[i] - i;  // map to weakly increasing a
  }
  return out;
}

// ---------- map indices to types ----------
template <class Types, int... Idx>
using pick_types_t = std::tuple<std::tuple_element_t<Idx, Types>...>;

// Build a type: std::tuple< pick_types_t<Types, a0,a1,...>, pick_types_t<Types, ...>, ... >
template <class Types, int R>
struct all_type_combos_t {
 private:
  static constexpr auto idxs = all_multicomb<std::tuple_size_v<Types>, R>();
  template <std::size_t I>
  using elem_t = pick_types_t<Types, idxs[I][0], idxs[I][1],
                              (R > 2 ? idxs[I][2] : 0)  // guarded but ok for generalization
                              // If you generalize R, expand this via an index sequence.
                              >;
  template <std::size_t... I>
  static auto build(std::index_sequence<I...>) -> std::tuple<elem_t<I>...>;

 public:
  using type = decltype(build(std::make_index_sequence<idxs.size()>{}));
};

// Helper alias
template <class Types, int R>
using all_type_combos = typename all_type_combos_t<Types, R>::type;

// -------------------- Example --------------------
#include <string>

using Types = std::tuple<int, double, std::string, float>;
using TuplesR3 = all_type_combos<Types, 3>;

// size check: C(4+3-1, 3) = 20
static_assert(std::tuple_size_v<TuplesR3> == binom(4 + 3 - 1, 3));

// spot checks on first/last elements (colex order):
static_assert(std::is_same_v<std::tuple_element_t<0, TuplesR3>, std::tuple<int, int, int>>);
static_assert(std::is_same_v<std::tuple_element_t<19, TuplesR3>, std::tuple<float, float, float>>);

int main() {
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// How to map a runtime array to a compile time tuple of default constructed types:
#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <utility>
// ---------- compile-time binomial ----------
constexpr unsigned long long binom(unsigned int n, unsigned int k) {
  if (k > n) return 0ULL;
  if (k > n - k) k = n - k;
  unsigned long long r = 1;
  for (unsigned int i = 1; i <= k; ++i) r = (r * (n - k + i)) / i;
  return r;
}

// ---------- unrank R-combination (no replacement) in colex ----------
template <int R>
constexpr std::array<int, R> unrank_comb_norep(unsigned int M, unsigned long long k) {
  std::array<int, R> b{};
  int x = static_cast<int>(M);
  for (int i = R; i >= 1; --i) {
    while (binom(x, static_cast<unsigned>(i)) > k) --x;
    b[i - 1] = x;
    k -= binom(x, static_cast<unsigned>(i));
    --x;
  }
  return b;
}

// ---------- all weakly-increasing R-tuples over [0..N-1] ----------
template <int N, int R>
consteval auto all_multicomb_indices() {
  constexpr auto CNT = binom(N + R - 1, R);
  constexpr unsigned int M = static_cast<unsigned int>(N + R - 1);
  std::array<std::array<int, R>, CNT> out{};
  for (unsigned long long k = 0; k < CNT; ++k) {
    auto b = unrank_comb_norep<R>(M, k);
    for (int i = 0; i < R; ++i) out[k][i] = b[i] - i;  // weakly increasing 0..N-1
  }
  return out;
}

// ---------- turn index tuples into type tuples ----------
template <class Types, int... Idx>
using pick_types_t = std::tuple<std::tuple_element_t<Idx, Types>...>;

template <class Types, int R>
struct all_type_combos_t {
  static constexpr int N = static_cast<int>(std::tuple_size_v<Types>);
  static constexpr auto idxs = all_multicomb_indices<N, R>();

  template <std::size_t I, std::size_t... J>
  static auto make_elem(std::index_sequence<J...>) -> pick_types_t<Types, idxs[I][static_cast<int>(J)]...>;

  template <std::size_t I>
  using elem_t = decltype(make_elem<I>(std::make_index_sequence<R>{}));

  template <std::size_t... I>
  static auto build(std::index_sequence<I...>) -> std::tuple<elem_t<I>...>;

  using type = decltype(build(std::make_index_sequence<idxs.size()>{}));
};

template <class Types, int R>
using all_type_combos = typename all_type_combos_t<Types, R>::type;

// ---------- runtime match + visit ----------
namespace detail {

// Does `ti` match the type pattern of tuple `Tup` element-wise?
template <class Tup, std::size_t... J>
bool match_types(const std::array<std::type_index, std::tuple_size_v<Tup>>& ti, std::index_sequence<J...>) {
  return ((ti[J] == std::type_index(typeid(std::tuple_element_t<J, Tup>))) && ...);
}

template <class Types, int R, std::size_t I, class F>
bool try_one(const std::array<std::type_index, R>& ti, F&& f) {
  using Tup = typename all_type_combos_t<Types, R>::template elem_t<I>;
  if (match_types<Tup>(ti, std::make_index_sequence<R>{})) {
    std::invoke(std::forward<F>(f), Tup{});  // default-constructed concrete tuple
    return true;
  }
  return false;
}

template <class Types, int R, class F, std::size_t... I>
bool visit_impl(const std::array<std::type_index, R>& ti, F&& f, std::index_sequence<I...>) {
  bool done = false;
  // Short-circuit fold: evaluate left-to-right and stop once matched.
  ((done = done || try_one<Types, R, I>(ti, f)), ...);
  return done;
}

}  // namespace detail

// Public API:
// - Returns true if a matching multichoose type was found, and calls f(Tup{}).
// - Requires that the input type_index array is in the same weakly-increasing order
//   as the generated combinations (0..N-1 order of `Types`).
template <class Types, int R, class F>
bool visit_multicomb(const std::array<std::type_index, R>& ti, F&& f) {
  using Combos = all_type_combos<Types, R>;
  constexpr std::size_t CNT = std::tuple_size_v<Combos>;
  return detail::visit_impl<Types, R>(ti, std::forward<F>(f), std::make_index_sequence<CNT>{});
}

// -------------------- Example --------------------
using Types = std::tuple<int, double, std::string, float>;

int main() {
  // Example: want to visit the "001" (int,double,double) and "233" (double,float,float)
  std::array<std::type_index, 3> key1{typeid(int), typeid(double), typeid(double)};
  std::array<std::type_index, 3> key2{typeid(double), typeid(float), typeid(float)};

  auto printer = [](auto tup) {
    using T = decltype(tup);
    // do "visit" work here; we just print the demangled-ish type names count
    constexpr std::size_t R = std::tuple_size_v<T>;
    std::cout << typeid(T).name() << ": " << typeid(std::get<0>(tup)).name() << typeid(std::get<1>(tup)).name()
              << typeid(std::get<2>(tup)).name() << std::endl;
  };

  bool ok1 = visit_multicomb<Types, 3>(key1, printer);  // matches {0,1,1}
  bool ok2 = visit_multicomb<Types, 3>(key2, printer);  // matches {1,3,3}
  (void)ok1;
  (void)ok2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// How to map an array of variants to a compile time tuple of their active types:
#include <array>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

// ---------- compile-time binomial ----------
constexpr unsigned long long binom(unsigned int n, unsigned int k) {
  if (k > n) return 0ULL;
  if (k > n - k) k = n - k;
  unsigned long long r = 1;
  for (unsigned int i = 1; i <= k; ++i) r = (r * (n - k + i)) / i;
  return r;
}

// ---------- unrank R-combination (no replacement) in colex ----------
template <std::size_t R>
constexpr std::array<int, R> unrank_comb_norep(unsigned int M, unsigned long long k) {
  std::array<int, R> b{};
  int x = static_cast<int>(M);
  for (int i = R; i >= 1; --i) {
    while (binom(x, static_cast<unsigned>(i)) > k) --x;
    b[i - 1] = x;
    k -= binom(x, static_cast<unsigned>(i));
    --x;
  }
  return b;
}

// ---------- all weakly-increasing R-tuples over [0..N-1] ----------
template <int N, std::size_t R>
consteval auto all_multicomb_indices() {
  constexpr auto CNT = binom(N + R - 1, R);
  constexpr unsigned int M = static_cast<unsigned int>(N + R - 1);
  std::array<std::array<int, R>, CNT> out{};
  for (unsigned long long k = 0; k < CNT; ++k) {
    auto b = unrank_comb_norep<R>(M, k);
    for (int i = 0; i < R; ++i) out[k][i] = b[i] - i;  // weakly increasing 0..N-1
  }
  return out;
}

// Cache the indices table per (Variant,R)
template <class Variant, std::size_t R>
struct multicomb_index_table {
  static constexpr int N = static_cast<int>(std::variant_size_v<std::remove_cv_t<std::remove_reference_t<Variant>>>);
  static constexpr auto idxs = all_multicomb_indices<N, R>();
  static constexpr std::size_t size = idxs.size();
};

// ---------- helpers that use the cached table ----------
namespace detail {

// Compare active indices of vs against the I-th pattern
template <std::size_t I, class Variant, std::size_t R>
bool match_active(const std::array<Variant, R>& vs) {
  for (int j = 0; j < R; ++j) {
    if (vs[j].index() != multicomb_index_table<Variant, R>::idxs[I][j]) return false;
  }
  return true;
}

// Make a tuple of references to the active alternatives for the I-th pattern
template <std::size_t I, class Variant, std::size_t R, std::size_t... J>
auto tie_from_variants_impl(std::array<Variant, R>& vs, std::index_sequence<J...>) {
  return std::tie(std::get<multicomb_index_table<Variant, R>::idxs[I][static_cast<int>(J)]>(vs[J])...);
}
template <std::size_t I, class Variant, std::size_t R, std::size_t... J>
auto tie_from_variants_impl(const std::array<Variant, R>& vs, std::index_sequence<J...>) {
  return std::tie(std::get<multicomb_index_table<Variant, R>::idxs[I][static_cast<int>(J)]>(vs[J])...);
}
template <std::size_t I, class Variant, std::size_t R>
auto tie_from_variants(std::array<Variant, R>& vs) {
  return tie_from_variants_impl<I, Variant, R>(vs, std::make_index_sequence<R>{});
}
template <std::size_t I, class Variant, std::size_t R>
auto tie_from_variants(const std::array<Variant, R>& vs) {
  return tie_from_variants_impl<I, Variant, R>(vs, std::make_index_sequence<R>{});
}

// Try the I-th multichoose; on match, call f(tuple_of_refs)
// Returns true if matched.
template <std::size_t I, class Variant, std::size_t R, class F, typename... ExtraArgs>
bool try_one(std::array<Variant, R>& vs, const F& f, ExtraArgs&... args) {
  if (match_active<I>(vs)) {
    auto tup = tie_from_variants<I, Variant, R>(vs);
    std::invoke(f, tup, args...);
    return true;
  }
  return false;
}
template <std::size_t I, class Variant, std::size_t R, class F, typename... ExtraArgs>
bool try_one(const std::array<Variant, R>& vs, const F& f, ExtraArgs&... args) {
  if (match_active<I>(vs)) {
    auto tup = tie_from_variants<I, Variant, R>(vs);
    std::invoke(f, tup, args...);
    return true;
  }
  return false;
}

// Visit over all patterns until a match
template <class Variant, std::size_t R, class F, std::size_t... I, typename... ExtraArgs>
bool visit_impl(std::array<Variant, R>& vs, const F& f, std::index_sequence<I...>, ExtraArgs&... args) {
  bool done = false;
  ((done = done || try_one<I, Variant, R>(vs, f, args...)), ...);
  return done;
}
template <class Variant, std::size_t R, class F, std::size_t... I, typename... ExtraArgs>
bool visit_impl(const std::array<Variant, R>& vs, const F& f, std::index_sequence<I...>, ExtraArgs&... args) {
  bool done = false;
  ((done = done || try_one<I, Variant, R>(vs, f, args...)), ...);
  return done;
}

}  // namespace detail

// ---------- Public API ----------
// Requires the runtime array to be in the same weakly-increasing order
// as the multichoose enumeration over the variant alternatives.
//
// Calls f(tup_of_refs) where tup_of_refs is a tuple of (const) lvalue refs to each active value.
// Returns true if a match was found, false otherwise.
template <class Variant, std::size_t R, class F, typename... ExtraArgs>
bool visit_multicomb(std::array<Variant, R>& vs, const F& f, ExtraArgs&... args) {
  using Tab = multicomb_index_table<Variant, R>;
  return detail::visit_impl<Variant, R>(vs, f, std::make_index_sequence<Tab::size>{}, args...);
}
template <class Variant, std::size_t R, class F, typename... ExtraArgs>
bool visit_multicomb(const std::array<Variant, R>& vs, const F& f, ExtraArgs&... args) {
  using Tab = multicomb_index_table<Variant, R>;
  return detail::visit_impl<Variant, R>(vs, f, std::make_index_sequence<Tab::size>{}, args...);
}

// -------------------- Example --------------------
#include <iostream>
#include <string>

int main() {
  using V = std::variant<int, double, std::string, float>;
  std::array<V, 3> a1{5, 2.0, 2.5};                    // indices {0,1,1}
  std::array<V, 3> a2{std::string("hi"), 1.0f, 1.0f};  // indices {2,3,3}

  auto printer = [](const auto& tup) {
    // tup is std::tuple<const Alt0&, const Alt1&, const Alt2&>
    std::cout << typeid(tup).name() << std::endl;
    std::apply(
        [](const auto&... xs) {
          ((std::cout << xs << " "), ...);
          std::cout << "\n";
        },
        tup);
  };

  visit_multicomb<V, 3>(a1, printer);  // prints: 5 2 2.5
  visit_multicomb<V, 3>(a2, printer);  // prints: hi 1 1
}

int main() {
  // Example: want to visit the "001" (int,double,double) and "233" (double,float,float)
  std::array<Variant, 3> key1{int(42), double(3.14), double(41.3)};
  std::array<Variant, 3> key2{double(3.14), float(32), float(23)};

  auto printer = [](auto tup) {
    using T = decltype(tup);
    // do "visit" work here; we just print the demangled-ish type names count
    constexpr std::size_t R = std::tuple_size_v<T>;
    std::cout << typeid(T).name() << ": " << typeid(std::get<0>(tup)).name() << typeid(std::get<1>(tup)).name()
              << typeid(std::get<2>(tup)).name() << std::endl;
  };

  bool ok1 = visit_multicomb(key1, printer);  // matches {0,1,1}
  bool ok2 = visit_multicomb(key2, printer);  // matches {1,3,3}
  (void)ok1;
  (void)ok2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Current logic:
The order of accessors within an aggregate doesn't matter. This means that the set of concrete aggregate types is rather
small. We will sort them by their active variant index, allowing direct comparison to all_multicomb_indices.
visit_multicomb can then map the set of sorted runtime accessors to a tuple of concrete accessors. This tuple is now,
however, tagged.

How to handle tags?
We can store a tag string on the runtime accessors. The user would need to provide a map between compile-time tag and
runtime string. We would store a tuple of tags and an array of strings. How then, can we map the tuple of tags, array of
strings, and tuple of concrete accessors to a tuple of tagged concrete accessors?

(Tags, std::array<std::string, N> tag_strings, std::tuple<Accessor1, ..., AccessorN> accessors)
  -> std::tuple<TaggedAccessor1, ..., TaggedAccessorN>

Ok, you take the tuple of accessors and build a map from string to tuple element index. Sure, that works but going from
there to a tuple of tagged accessors is impossible because this map is runtime and that tuple type is compile-time.

Also can't go the other way around and loop over the tuple of tags because the runtime strings still get in the way.

.add_accessor<CENTER>(center_accessor) works because it allows the user to provide the tag at compile-time. We still
need that but how can it be paired with the runtime string? The problem is that we are bound to a design where the type
is explicitly known but we are inside of an if-else block unable to return. Here, add_accessor is templated on the
accessor type, which means no intermediary runtime object can change that type.

Let's ask a different question: What is the purpose of tags? Tags give us the ability to call view.get<TAG>(entity)
instead of view.node_coords_accessor(entity). The tag is used (at compile-time) to find the correct accessor.

Can we do a pre-step before the current one? We have a RuntimeAggregate, which users touch directly but lack an
intermediary TaggedRuntimeAggregate, which contains tagged variants. Visit, should be performed on this object
and when we sort the tuple of runtime aggregates, we should also sort the tuple of tags.

// User creates their RuntimeAggregate
RuntimeAggregate ragg = make_runtime_aggregate(bulk_data, selector, stk::topology::PARTICLE)
                            .add_accessor(NODE_RANK, "OUR_CENTER", center_accessor)
                            .add_accessor(ELEM_RANK, "OUR_RADIUS", radius_accessor);

// User creates a rename map to tell us how their names correspond to our expected names
std::map<std::string, std::string> rename_map{{"CENTER", "OUR_CENTER"}, {"RADIUS", "OUR_RADIUS"}};

// We use the given rename map to unpack their runtime aggregate and add custom compile-time tags to each accessor
auto center_racc = ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]);
auto radius_racc = ragg.get_accessor<accessor_t::SCALAR<double>>(ELEM_RANK, rename_map["RADIUS"]);
auto tagged_ragg = make_tagged_ragg<stk::topology::PARTICLE>(bulk_data, selector)
                       .add_accessor<CENTER>(center_racc)
                       .add_accessor<RADIUS>(radius_racc);

// We use visit to convert the tagged aggregate into a compile-time optimized aggregate object
tagged_ragg.visit([](auto& agg) {
  // We perform an action that acts on the aggregate
  stk::mesh::for_each_entity_run(
      ngp_mesh, KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index) {
        stk::mesh::FastMeshIndex center_node_index = ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
        auto center = agg.get<CENTER>(center_node_index);
        auto radius = agg.get<RADIUS>(sphere_index);
        center += radius[0];
      });
});
*/

// In the following, we extend our current prototype to include tags
//
// Notes:
// We are unable to sort the variants by active index because we cannot sort the tuple of tags in the same way
// due to the variant index being runtime. We can use a reorder map instead, which has no effect on performance.
template <size_t N, typename... Tags, typename... VariantTs>
struct TaggedBagOfVariants {
  using tags_t = std::tuple<Tags...>;
  using variant_types_t = std::tuple<VariantTs...>;
  using variant_t = std::variant<VariantTs...>;
  std::array<variant_t, N> variants;

  template <typename Tag>
  auto insert(Tag tag, variant_t var) {
    // Create a new TaggedBagOfVariants with the new tag/variant added
    auto new_tags = std::tuple_cat(tags_t{}, std::tuple<Tag>{});
    std::array<variant_t, N + 1> new_variants;
    for (size_t i = 0; i < N; ++i) {
      new_variants[i] = variants[i];
    }
    new_variants[N] = var;
    return TaggedBagOfVariants<N + 1, Tags..., Tag, VariantTs...>{new_variants};
  }

  std::array<size_t, N> build_reorder_map() {  // map[sorted_id] = original_id
    std::array<size_t, N> map_from_sorted_to_original{};
    std::iota(map_from_sorted_to_original.begin(), map_from_sorted_to_original.end(), 0);
    std::sort(map_from_sorted_to_original.begin(), map_from_sorted_to_original.end(),
              [this](size_t a, size_t b) { return variants[a].index() < variants[b].index(); });
    return map_from_sorted_to_original;
  }

  template <typename Visitor>
  bool visit(const Visitor& visitor) {
    bool success =
        visit_multicomb(variants, visitor, tags_t{});  // calls visitor(tuple_of_active_types, tuple_of_their_tags)
    return success;
  }
};

template <size_t N, typename... Tags, typename... Types>
struct TaggedBagOfObjects {
  using tags_t = std::tuple<Tags...>;
  using types_t = std::tuple<Types...>;
  std::tuple<Types...> objs;

  template <typename Tag, typename NewObject>
  auto insert(Tag tag, NewObject obj) {
    // Create a new TaggedBagOfVariants with the new tag/variant added
    auto new_tags = std::tuple_cat(tags_t{}, std::tuple<Tag>{});
    auto new_objs = std::tuple_cat(objs, std::tuple<NewObject>{obj});
    return TaggedBagOfObjects<N + 1, Tags..., Tag, Types..., NewObject>{new_objs};
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Same logic, a new day.
//
// Mundy now formally offers core::aggregate as a generalized data structure that mimics a compile-time map from key to
// object similar in many ways to boost::hana::map. We'll call our TaggedBagOfVariants variant_aggregate and write it in
// condensed form as v_agg.
//
// I am currently working about the issue of sorting by active type. We have a "working" prototype that correctly converts
// each variant of a variant_aggregate to its active type and uses this to populate a concrete aggregate. The problem is that
// this prototype doesn't maintain the correct tag to type mapping because we cannot sort the tags in the same way as the variants.
//
// We have unsorted tags in a tuple and sorted concrete types in a tuple.
// The only solution is to make get<tag>(this_weird_sorted_aggregate) do a compile-time map from tag to index and then use a
// runtime undo_sort_map to get the correct index into the sorted concrete types.
//
// // Some python code for a test
// import numpy as np
// import math
// def num_visits_direct(num_alts, num_variants):
//     return num_alts**num_variants
//
// def num_visits_sorted(num_alts, num_variants):
//     return math.factorial(num_alts + num_variants - 1) // (math.factorial(num_variants) * math.factorial(num_alts - 1))
//
// def num_visits_sorted_and_tagged(num_alts, num_variants):
//     return math.factorial(num_alts + num_variants - 1) // (math.factorial(num_variants) * math.factorial(num_alts - 1)) * math.factorial(num_variants)
//
// for num_alts in range(1, 5):
//     for num_variants in range(1, 13):
//         print(f"{num_alts:2d} {num_variants:2d} {num_visits_sorted(num_alts, num_variants):6d} {num_visits_direct(num_alts, num_variants):6d} {num_visits_sorted_and_tagged(num_alts, num_variants):6d}")

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_RAGGREGATES_HPP_
