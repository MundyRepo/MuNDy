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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRIMPL_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRIMPL_HPP_

/// \file NgpAccessorExprImpl.hpp

// Kokkos
#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA, etc.

// STL
#include <unordered_map>

// STK mesh
#include <stk_mesh/base/BulkData.hpp>            // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>              // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>               // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>            // for stk::mesh::MetaData
#include <stk_mesh/base/NgpField.hpp>            // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>             // for stk::mesh::NgpMesh
#include <stk_mesh/base/NgpReductions.hpp>       // for stk::mesh::for_each_entity_reduce
#include <stk_mesh/base/Selector.hpp>            // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>               // for stk::mesh::FastMeshIndex
#include <stk_util/parallel/ParallelReduce.hpp>  // for stk::all_reduce_*

// Mundy
#include <mundy_math/Matrix.hpp>          // for mundy::Matrix
#include <mundy_math/Quaternion.hpp>      // for mundy::Quaternion
#include <mundy_math/ScalarWrapper.hpp>   // for mundy::ScalarWrapper
#include <mundy_math/Vector.hpp>          // for mundy::Vector
#include <mundy_mesh/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <mundy_utils/StringLiteral.hpp>  // for mundy::StringLiteral
#include <mundy_utils/aggregate.hpp>      // for mundy::aggregate
#include <mundy_utils/rng.hpp>            // for mundy::make_philox
#include <mundy_utils/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>          // for mundy::tuple

namespace mundy {

namespace mesh {

/*
Accessor expressions!

Goal:
To allow fields/components to be used as though they were their underlying type by delayed expression evaluation.

For example, let vec*, mat*, quat* be accessors of the appropriate type. Then the following will create and evaluate
an inlined expression list:
  EntityExpr all_rods(rod_selector, rank);
  ConnectedEntitiesExpr rod_nodes = all_rods.get_connectivity(NODE_RANK);
  auto vec3_1 = avec3_1(rod_nodes[0]) + avec3_2(rod_nodes[1]);
  auto vec3_2 = avec3_2(all_rods);
  auto tmp_vec3 = reuse(vec3_1 + vec3_2 * 2.0 - cross(vec3_1, vec3_2));
  auto tmp_mat3 = outer(tmp_vec3, vec3_1) + mat3_1 * 3.0;

  // Performs a single kernel launch, evaluates tmp_vec3 and tmp_mat3 for each rods, then assigns them to the accessors
  fused_assign(vec3_1, tmp_vec3,   // vec3_1 = tmp_vec3
               mat3_1, tmp_mat3);  // mat3_1 = tmp_mat3

  // Performs a different kernel launch, recomputes tmp_mat3 for each rods, and finds its global min.
  Vector3<double> min_vel = reduce_min(tmp_mat3);

Expressions are evaluated upon assignment to an lvalue or when passed to a reduction operation. The following will all
perform evals
  - accessor(entity_expr) = expr;
  - accessor(entity_expr) += expr;
  - fused_assign(accessor1, expr1, accessor2, expr2, ...);
  - auto result = reduce_op(expr);

Upon evaluation, but before looping over the entities, all fields involved in the expression are synchronized to the
appropriate space and marked modified where necessary. The expression tree "knows" which fields are read and written.

# Caching:

Something important to consider here is that our design is carefully setup to ensure that identical sub-expressions in
the tree will always return the same result given the same input. That is, if you can identify a subset of the tree
(from a given node all the way to its leaves) that matches another subset, then they are compatible with reuse. Because
our reuse is based on "if constexpr" they have zero overhead and do not introduce branching. As such, it seems like your
tag system can be done using the collective type of the sub-expressions. So, we basically want our expression system to
use a tagged bag (aka Aggregate). To then decide what to cache, we need to perform something similar to update_is_cached
but instead of setting equal to true, we count the total number of occurrences of each tag in the bag. Then, whenever
ANYTHING in the tree is evaluated, we conditionally cache the result if the number of occurrences of that tag is > 1.
This way, the user never marks anything as reused, but rather the system automatically determines what to cache. The
fact that we are using "if constexpr" means that there is no runtime overhead to this approach.

Special functions
 -reuse: Flag an expression to be reused by multiple other expressions in a single fused kernel. Its return is memoized
instead of being re-evaluated.

 -fused_assign: Fuse N assignment operations into a single kernel to avoid either multiple evaluations of shared
sub-expressions or multiple kernel launches.
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Example of accessor expression vs for_each_entity_run
//
// void euler_update_rods_agg() {
//   rod_agg.sync_to_device<CENTER, QUAT, VELOCITY, OMEGA>();
//   stk::mesh::for_each_entity_run(
//       ngp_mesh, rod_selector, rank, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &rod_index) {
//         auto nodes = ngp_mesh.get_connected_entities(rod_index, NODE_RANK);
//         auto center = rod_agg.get<CENTER>(nodes[0]);
//         auto quat = rod_agg.get<QUAT>(nodes[0]);
//         auto velocity = rod_agg.get<VELOCITY>(nodes[0]);
//         auto omega = rod_agg.get<OMEGA>(nodes[0]);
//         center += dt * velocity;
//         quat = rotate_quaternion(quat, omega, dt);
//       });
//   rod_agg.modify_on_device<CENTER, QUAT>();
// }
//
// void euler_update_rods_expr() {
//   EntityExpr rods(rod_selector, rank);
//   ConnectedEntitiesExpr nodes = rods.get_connectivity(NODE_RANK);
//   auto center = rod_agg.get<CENTER>(nodes[0]);
//   auto quat = rod_agg.get<QUAT>(nodes[0]);
//   auto velocity = od_agg.get<VELOCITY>(nodes[0]);
//   auto omega = rod_agg.get<OMEGA>(nodes[0]);
//
//   fused_assign(center, /* = */ center + dt * velocity,  //
//                quat, /* = */ rotate_quaternion(quat, omega, dt));
// }
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Example reduction of max speed vs field blas
// void max_speed_agg() {
//   stk::mesh::for_each_entity_run(
//       ngp_mesh, rod_selector, rank, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &rod_index) {
//         // Need to stash speed
//         aspeed(rod_index) = norm(avelocity(rod_index));
//       });
//
//   double max_speed = field_max(ngp_mesh, rod_selector, aspeed);
// }
//
// void max_speed_expr() {
//   EntityExpr all_rods(rod_selector, rank);
//   double max_speed = reduce_max(norm(avelocity(all_rods)));
// }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace impl {

template <template <class> class B, class E>
struct is_crtp_base_of_impl : std::is_base_of<B<E>, E> {};

// Cached eval cannot safely return a raw reference alongside a by-value cache object.
// Instead, cached eval returns a Kokkos::pair<Handle, Cache>, where Handle either owns a temporary or names a tag in
// the cache. Callers then resolve the handle against whichever cache object is currently live.
template <typename ValueType>
class OwnedCachedValue {
 public:
  using value_type = std::remove_cvref_t<ValueType>;

  KOKKOS_DEFAULTED_FUNCTION
  OwnedCachedValue() = default;

  KOKKOS_FUNCTION
  explicit OwnedCachedValue(const value_type& value) : value_(value) {
  }

  KOKKOS_FUNCTION
  explicit OwnedCachedValue(value_type&& value) : value_(std::move(value)) {
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr value_type& get([[maybe_unused]] CacheType& cache) {
    return value_;
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr const value_type& get([[maybe_unused]] const CacheType& cache) const {
    return value_;
  }

 private:
  value_type value_;
};

template <typename ValueType>
OwnedCachedValue(ValueType&&) -> OwnedCachedValue<std::remove_cvref_t<ValueType>>;

template <typename Tag>
class CachedTagGetter {
 public:
  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(CacheType& cache) const {
    return ::mundy::get<Tag>(cache);
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(const CacheType& cache) const {
    return ::mundy::get<Tag>(cache);
  }
};

class RuntimeReuseValidator {
 public:
  RuntimeReuseValidator() = default;

  template <typename EvalCountsType, EvalCountsType eval_counts, typename Expr>
  void validate(const Expr& expr) {
    static_assert(has<typename Expr::our_tag>(eval_counts), "eval_counts must contain the expression tag");
    if constexpr (Expr::supports_runtime_reuse && !Expr::has_static_eval &&
                  get<typename Expr::our_tag>(eval_counts) > 1) {
      validate_or_track(expr);
    }
  }

 private:
  template <typename Expr>
  void validate_or_track(const Expr& expr) {
    using tag_t = typename Expr::our_tag;
    auto [it, inserted] = first_expr_ptr_by_tag_.emplace(tag_key<tag_t>(), static_cast<const void*>(&expr));
    if (!inserted) {
      const auto* first_expr = static_cast<const Expr*>(it->second);
      MUNDY_THROW_REQUIRE(
          expr.runtime_reuse_equivalent(*first_expr), std::logic_error,
          "Unsupported runtime reuse found in expression tree...\n"
          "\n"
          "Hi, you've surpassed mundy's current supported behavior related to runtime reuse in an expression tree.\n"
          "This means that you managed to create two identical random number generator expressions with different "
          "runtime seed/counter combos.\n"
          "\n"
          "For example:\n"
          ">>>  auto rng_a = rng(seed(es) + 1.0, counter(es));\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          ">>>  fused_assign(first_draw(es), /*=*/rng_a.template rand<double>(),\n"
          ">>>               second_draw(es), /*=*/rng_b.template rand<double>());\n"
          "\n"
          "This is worked around by changing the inputs to the random number generators to have different forms:\n"
          "Example workaround 1 (change order of addition):\n"
          ">>>  auto rng_a = rng(1.0 + seed(es), counter(es));\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          "\n"
          "Example workaround 2 (add zero):\n"
          ">>>  auto rng_a = rng(seed(es) + 1.0, counter(es) + 0.0);\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          "\n"
          "If this remains a problem, let us know and we'll test an arbitrary runtime-reuse implementation.\n"
          "It's fully possible, we simply haven't needed it yet.");
    }
  }

  template <typename Tag>
  static const void* tag_key() {
    static const int key = 0;
    return &key;
  }

  std::unordered_map<const void*, const void*> first_expr_ptr_by_tag_;
};

template <typename EvalCountsType, EvalCountsType eval_counts, size_t I = 0, class ExprTuple, size_t NumEntities,
          class CacheType, class Ctx>
KOKKOS_FUNCTION auto cached_expr_chain(const ExprTuple& exprs,
                                       const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                       CacheType&& cache, const Ctx& ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (I == num_expr) {
    // No more exprs; return empty values tuple and the current cache
    return Kokkos::make_pair(tuple<>{}, std::forward<CacheType>(cache));
  } else {
    // Evaluate current expr with the current cache
    auto& expr = get<I>(exprs);
    auto result_i = expr.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<CacheType>(cache), ctx);
    auto value_handle_i = std::move(result_i.first);
    auto next_cache = std::move(result_i.second);

    // Recurse for the rest, threading the updated cache
    auto [vals_tail, final_cache] =
        cached_expr_chain<EvalCountsType, eval_counts, I + 1>(exprs, fmis, std::move(next_cache), ctx);

    // Prepend this value handle to the tuple of later value handles.
    auto vals_all = tuple_cat(tuple{std::move(value_handle_i)}, std::move(vals_tail));
    return Kokkos::make_pair(vals_all, std::move(final_cache));
  }
}

template <size_t I = 0, class ExprTuple, size_t NumEntities, class Ctx>
KOKKOS_FUNCTION auto expr_chain(const ExprTuple& exprs,
                                const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis, const Ctx& ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (I == num_expr) {
    // No more exprs; return empty values tuple and the current cache
    return tuple<>{};
  } else {
    // Evaluate current expr with the current cache
    auto val_i = get<I>(exprs).eval(fmis, ctx);

    // Recurse for the rest, threading the updated cache
    auto vals_tail = expr_chain<I + 1>(exprs, fmis, ctx);

    // Prepend this value to the tuple of later values
    auto vals_all = tuple_cat(tuple{std::move(val_i)}, std::move(vals_tail));
    return vals_all;
  }
}

// A map from rank and selector to an std::any
template <StringLiteral map_name>
class AnyRankSelectorMap {
 public:
  AnyRankSelectorMap() = default;

  /// \brief The name of the map
  static std::string name() {
    return map_name.to_string();
  }

  /// \brief If a given rank/selector pair is in the map
  bool contains(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const {
    const auto& selector_map = ranked_selector_maps_[rank];
    return selector_map.find(selector) != selector_map.end();
  }

  template <typename T>
  void insert(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector, T value) {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(!contains(rank, selector), std::logic_error,
                       "Attempting to insert a rank and selector pair into AnyRankSelectorMap that is already present");
    selector_map.emplace(selector, std::move(value));
  }

  template <typename T>
  T& at(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<T&>(selector_map.at(selector));
  }

  template <typename T>
  const T& at(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<const T&>(selector_map.at(selector));
  }

 private:
  using selector_map_t = std::map<stk::mesh::Selector, std::any>;
  selector_map_t ranked_selector_maps_[stk::topology::NUM_RANKS];
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRIMPL_HPP_
