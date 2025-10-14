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

#ifndef MUNDY_MESH_NGPACCESSOREXPR_HPP_
#define MUNDY_MESH_NGPACCESSOREXPR_HPP_

/// \file NgpAccessorExpr.hpp

// Kokkos
#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA, etc.

// STK mesh
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/NgpField.hpp>  // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>   // for stk::mesh::NgpMesh
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::FastMeshIndex

// Mundy
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral
#include <mundy_core/aggregate.hpp>      // for mundy::core::aggregate
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_core/tuple.hpp>          // for mundy::core::tuple
#include <mundy_math/Vector.hpp>         // for mundy::math::Vector
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

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


Special functions
 -reuse: Flag an expression to be reused by multiple other expressions in a single fused kernel. Its return is memoized
instead of being re-evaluated.

 -fused_assign: Fuse N assignment operations into a single kernel to avoid either multiple evaluations of shared
sub-expressions or multiple kernel launches.

Some notes on caches:
  - Each expression cannot "know" the type of the cache it is given.
    - Cache type is propagated from leaf to root expressions at compile-time during construction.
    - The collective cache is created by the root and passed down to leaf expressions during eval where each leaf
      knows its compile-time offset into the cache.

  - Cache type is accumulated from past expressions and only known at the root expression that is evaluated.
    - This means that eval must be templated by its given cache type and the compile-time offset into the cache.
    - Each expression stores (as a static constexpr) the number of cache entries it needs.

FusedExpr is the one that's in charge of setting up the cache and forwarding it to the sub-expressions. It's the fused
root of the expression tree.

The thing I'm worrying about is the runtime if statement on the memoized return of ReusedExpr. I want to know if it
can be done at compile-time instead. We would need the expressions to be able to map an array of bools stating if
a given cached type is set or not (pre-eval) to an array of bools stating if a given cached type is set or not
(post-eval).


template <std::size_t Start, std::size_t Count, std::size_t M>
constexpr void set_true(Vector<bool, M>& a) {
  static_assert(Start + Count <= M, "range out of bounds");
  // unrolled at compile-time
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((a[Start + I] = true), ...);
  }(std::make_index_sequence<Count>{});
}

// Cache is just a mundy::core::tuple of types.
// IsCached is a math::Vector<bool, N>, so it's compatable with compile-time vector operations <3!!!

auto dot.eval<CacheOffset, CacheSize, IsCached>(fmas, cache, context)
  auto lhs_res = lhs.eval<CacheOffset, CacheSize, IsCached>(fmas, cache, context);

  // Evaluating the LHS may have changed the cache, so we need to update IsCached for the RHS
  constexpr auto updated_is_cached = LeftExpr::update_is_cached<CacheOffset, CacheSize, IsCached>();
  auto rhs_res = rhs.eval<CacheOffset + LeftExpr::num_cached_types, updated_is_cached>(fmas, cache, context));

  return dot(lhs_res, rhs_res);

auto reuse.eval<CacheOffset, CacheSize, IsCached>(fmas, cache, context) {
  // The cache offset is for us and all of our sub-expressions. Our cached object is the last one in that range.
  // Basically read from right to left in the cache when traversing via eval.
  //   [  unrelated expr cache | our sub-expr cache  |  our cache  | unrelated expr cache ]
  //                           ^                     ^
  //                      CacheOffset     CacheOffset + PrevExpr::num_cached_types
  constexpr size_t our_cache_offset = CacheOffset + PrevExpr::num_cached_types;
  if constexpr (IsCached[our_cache_offset]) {
    return get<our_cache_offset>(cache);
  } else {
    auto val = prev_expr_.eval<CacheOffset, CacheSize, IsCached>(fmas, cache, context);
    get<CacheOffset>(cache) = val;
    return val;
  }
}

void fuse.eval<CacheOffset, CacheSize, IsCached>(fmas, cache, context) {
  // Let's assume that this fuse is for three expressions expr1, expr2, expr3

  constexpr size_t expr1_offset = CacheOffset;
  constexpr size_t expr2_offset = CacheOffset + Expr1::num_cached_types;
  constexpr size_t expr3_offset = CacheOffset + Expr1::num_cached_types + Expr2::num_cached_types;

  constexpr auto is_cached_pre_expr1 = IsCached;
  constexpr auto is_cached_pre_expr2 = Expr1::update_is_cached<CacheOffset, is_cached_pre_expr1>();
  constexpr auto is_cached_pre_expr3 = Expr2::update_is_cached<CacheOffset + Expr1::num_cached_types,
is_cached_pre_expr2>();

  expr1.eval<expr1_offset, is_cached_pre_expr1>(fmas, cache, context);
  expr2.eval<expr2_offset, is_cached_pre_expr2>(fmas, cache, context);
  expr3.eval<expr3_offset, is_cached_pre_expr3>(fmas, cache, context);
}

constexpr void ReuseEval::update_is_cached<CacheOffset, IsCached>() {
  // Deep copy the IsCached array and set our cache entry to true
  auto updated_flags = IsCached;
  updated_flags = PreviousExpr::update_is_cached<CacheOffset, updated_flags>();
  updated_flags[CacheOffset + PrevExpr::num_cached_types] = true;
  return updated_flags;
}

constexpr void DotExpr::update_is_cached<CacheOffset, IsCached>() {
  // Deep copy the IsCached array and set our cache entries to the union of our sub-expressions
  auto updated_flags = IsCached;
  updated_flags = LeftExpr::update_is_cached<CacheOffset, updated_flags>();
  updated_flags = RightExpr::update_is_cached<CacheOffset + LeftExpr::num_cached_types, updated_flags>();
  return updated_flags;
}

auto expr.init_cache() {
  return cache_t{};
}

for_each_entity_eval_expr(Expr expr) {
  static_assert(EntityExpr::num_entities == 1,
                "for_each_entity_evaluate_expr only works with single-entity expressions");
  stk::mesh::EntityRank rank = expr.rank();
  stk::mesh::Selector selector = expr.selector();
  stk::mesh::NgpMesh ngp_mesh = expr.ngp_mesh();

  // Sync all fields to the appropriate space and mark modified where necessary
  NgpEvalContext evaluation_context(ngp_mesh);
  expr.propagate_synchronize(evaluation_context);

  // Perform the evaluation
  ::mundy::mesh::for_each_entity_run(
      ngp_mesh, selector, rank, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
        auto cache = expr.init_cache();
        constexpr auto is_cached = Expr::init_is_cached();
        constexpr size_t cache_offset = 0;
        expr.template eval<cache_offset, is_cached>(entity_index, cache, evaluation_context);
      });
}

// A comment on make_entity_expr. The entire tree must use one and only one EntityExpr. It can be reused
throughout the tree but there can only be one. It is what decides how to perform the for_each_entity_run.

// A step back. Part of the reason for this is to allow access to the rank of all involved entities, their selector,
and the mesh. We used the execution context to pass the mesh through the evals but also sadly had EntityExpr return
ngp mesh, making this separation of concerns invalid.

// The problem here is that all expressions need to access the same execution context within which we have the
// mesh, ranks, and selectors. Technically, the execution context of the ngp mesh could be separated from the selectors
// and ranks. If this is the case, then we still have an execution context that must be passed through all evals but
// we also have an additional ~thing~ that the entire expression must agree on. This ~thing~ is what we are calling
// the EntityExpr, which is a poor name since it is closer to the EntityContext. This is as apposed to the
ExecutionContext.
// But like, just loop that all together into Context and be done with it. We can still type specialize the evals on
// a template of the class, so there's no loss of functionality.

I think we're looking for something like a driver or evaluator or manager that we store a pointer to and only access on
the host.

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

template <typename EvalCountsType, EvalCountsType eval_counts, std::size_t I = 0, class ExprTuple, size_t NumEntities,
          class CacheType, class Ctx>
KOKKOS_FUNCTION auto cached_expr_chain_impl(const ExprTuple &exprs,
                                            const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                            CacheType &&cache, const Ctx &ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (num_expr == 1) {
    // Single expr; just eval it and return its value and the current cache
    auto &expr = core::get<0>(exprs);
    auto [val, next_cache] =
        expr.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<CacheType>(cache), ctx);
    return Kokkos::make_pair(core::tuple{val}, next_cache);
  } else if constexpr (I == num_expr) {
    // No more exprs; return empty values tuple and the current cache
    return Kokkos::make_pair(core::tuple<>{}, std::forward<CacheType>(cache));
  } else {
    // Evaluate current expr with the current cache
    auto &expr = core::get<I>(exprs);
    auto [val_i, next_cache] =
        expr.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<CacheType>(cache), ctx);

    // Recurse for the rest, threading the updated cache
    auto [vals_tail, final_cache] =
        cached_expr_chain_impl<EvalCountsType, eval_counts, I + 1>(exprs, fmis, std::move(next_cache), ctx);

    // Prepend this value to the tuple of later values
    auto vals_all = core::tuple_cat(core::tuple{std::move(val_i)}, std::move(vals_tail));
    return Kokkos::make_pair(vals_all, final_cache);
  }
}

template <std::size_t I = 0, class ExprTuple, size_t NumEntities, class Ctx>
KOKKOS_FUNCTION auto expr_chain_impl(const ExprTuple &exprs,
                                     const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, const Ctx &ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (num_expr == 1) {
    // Single expr; just eval it and return its value and the current cache
    auto val = core::get<0>(exprs).eval(fmis, ctx);
    return core::tuple{std::move(val)};
  } else if constexpr (I == num_expr) {
    // No more exprs; return empty values tuple and the current cache
    return core::tuple<>{};
  } else {
    // Evaluate current expr with the current cache
    auto val_i = core::get<I>(exprs).eval(fmis, ctx);

    // Recurse for the rest, threading the updated cache
    auto vals_tail = expr_chain_impl<I + 1>(exprs, fmis, ctx);

    // Prepend this value to the tuple of later values
    auto vals_all = core::tuple_cat(core::tuple{std::move(val_i)}, std::move(vals_tail));
    return vals_all;
  }
}

// Public interface
template <typename EvalCountsType, EvalCountsType eval_counts, class ExprTuple, size_t NumEntities, class CacheType,
          class Ctx>
KOKKOS_INLINE_FUNCTION auto cached_expr_chain(const ExprTuple &exprs,
                                              const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                              CacheType &&cache0, const Ctx &ctx) {
  return cached_expr_chain_impl<EvalCountsType, eval_counts>(exprs, fmis, std::forward<CacheType>(cache0), ctx);
}

template <class ExprTuple, size_t NumEntities, class Ctx>
KOKKOS_INLINE_FUNCTION auto expr_chain(const ExprTuple &exprs,
                                       const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                       const Ctx &ctx) {
  return expr_chain_impl(exprs, fmis, ctx);
}

// A map from rank and selector to an std::any
template <core::StringLiteral map_name>
class AnyRankSelectorMap {
 public:
  AnyRankSelectorMap() = default;

  /// \brief The name of the map
  static std::string name() {
    return map_name.to_string();
  }

  /// \brief If a given rank/selector pair is in the map
  bool contains(stk::mesh::EntityRank rank, const stk::mesh::Selector &selector) const {
    const auto &selector_map = ranked_selector_maps_[rank];
    return selector_map.find(selector) != selector_map.end();
  }

  template <typename T>
  void insert(stk::mesh::EntityRank rank, const stk::mesh::Selector &selector, T value) {
    auto &selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(!contains(rank, selector), std::logic_error,
                       "Attempting to insert a rank and selector pair into AnyRankSelectorMap that is already present");
    selector_map.emplace(selector, std::move(value));
  }

  template <typename T>
  T &at(stk::mesh::EntityRank rank, const stk::mesh::Selector &selector) {
    auto &selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<T &>(selector_map.at(selector));
  }

  template <typename T>
  const T &at(stk::mesh::EntityRank rank, const stk::mesh::Selector &selector) const {
    auto &selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<const T &>(selector_map.at(selector));
  }

 private:
  using selector_map_t = std::map<stk::mesh::Selector, std::any>;
  selector_map_t ranked_selector_maps_[stk::topology::NUM_RANKS];
};

}  // namespace impl

//! \name Evaluation contexts
//@{

class NgpEvalContext {
 public:
  KOKKOS_INLINE_FUNCTION
  NgpEvalContext(stk::mesh::NgpMesh ngp_mesh) : ngp_mesh_(ngp_mesh) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
};
//@}

//! \name Entity expressions (those whose eval returns an entity and have a rank)
//@{

template <typename DerivedExpr>
class CachableExprBase {
 public:
  using our_tag = DerivedExpr;

 private:
  template <typename Tag, typename AggregateType, AggregateType agg>
  static constexpr auto increment_tag_count() {
    if constexpr (has<Tag>(agg)) {
      auto new_agg = agg;
      get<Tag>(new_agg) += 1;
      return new_agg;
    } else {
      return append<Tag>(agg, 1);
    }
  }

  template <typename SubExprTuple, size_t I, typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  static constexpr auto increment_eval_counts_recurse() {
    if constexpr (I < SubExprTuple::size()) {
      using sub_expr_t = core::tuple_element_t<I, SubExprTuple>;
      // Recurse into the sub-expression
      constexpr auto updated_eval_counts =
          sub_expr_t::template increment_eval_counts<OldEvalCountsType, old_eval_counts>();
      return increment_eval_counts_recurse<SubExprTuple, I + 1, decltype(updated_eval_counts), updated_eval_counts>();
    } else {
      return old_eval_counts;
    }
  }

 public:
  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedExpr &self() const noexcept {
    return static_cast<const DerivedExpr &>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedExpr &self() noexcept {
    return static_cast<DerivedExpr &>(*this);
  }

  /// \brief Evaluate the expression
  template <size_t NumEntities, class Ctx>
  auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, const Ctx &context) const {
    return self().eval(fmis, context);
  }

  /// \brief Evaluate the expression
  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, OldCacheType &&old_cache,
                   const Ctx &context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
  }

  /// \brief Update eval_counts by incrementing the counts for our tag and our sub-expressions tags
  template <typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  static constexpr auto increment_eval_counts() {
    constexpr auto new_eval_counts = increment_tag_count<our_tag, OldEvalCountsType, old_eval_counts>();
    using sub_exprs = typename DerivedExpr::sub_expressions_t;
    return increment_eval_counts_recurse<sub_exprs, 0, decltype(new_eval_counts), new_eval_counts>();
  }

  //! \name Field synchronization and modification flagging
  //@{

  template <class Ctx>
  void propagate_synchronize(const Ctx &context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx &context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx &context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx &context) {
    self().flag_overwrite_all(context);
  }
  //@}
};

template <typename DerivedEntityExpr>
class EntityExprBase : public CachableExprBase<DerivedEntityExpr> {
 public:
  using our_t = EntityExprBase<DerivedEntityExpr>;
  using our_tag = typename CachableExprBase<DerivedEntityExpr>::our_tag;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedEntityExpr &self() const noexcept {
    return static_cast<const DerivedEntityExpr &>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedEntityExpr &self() noexcept {
    return static_cast<DerivedEntityExpr &>(*this);
  }

  /// \brief The rank of the entity we return
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return self().rank();
  }

  /// \brief The host-only driver for this expression tree
  const auto driver() const {
    return self().driver();
  }

  /// \brief Evaluate the expression
  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                                       const Ctx &context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, OldCacheType &&old_cache,
                   const Ctx &context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
  }

  //! \name Field synchronization and modification flagging
  //@{

  template <class Ctx>
  void propagate_synchronize(const Ctx &context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx &context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx &context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx &context) {
    self().flag_overwrite_all(context);
  }
  //@}
};

// Both rank and the index you use to fetch the N'th connected entity must be compile-time constants, lest we lose
// the ability to have such elegant reuse.
template <typename PrevEntityExpr>
class ConnectedEntitiesExpr : public EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>> {
 public:
  using our_t = ConnectedEntitiesExpr<PrevEntityExpr>;
  using our_tag = typename EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>>::our_tag;
  using sub_expressions_t = core::tuple<PrevEntityExpr>;
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;
  using ConnectedEntities = stk::mesh::NgpMesh::ConnectedEntities;

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(PrevEntityExpr prev_entity_expr, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(const EntityExprBase<PrevEntityExpr> &prev_entity_expr_base, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr_base.self()), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return conn_rank_;
  }

  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                                const NgpEvalContext &context) const {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext &context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        // Return the cached value and the old cache
        std::cout << "Reusing cached connectivity" << std::endl;
        auto cache = std::forward<OldCacheType>(old_cache);
        return Kokkos::make_pair(get<our_tag>(cache), cache);
      } else {
        // Eval our subexpressions first
        auto [entity_index, new_cache] = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);

        // Our eval result needs cached, but is not yet cached
        stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
        auto val = context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
        auto newest_cache = append<our_tag>(new_cache, val);
        return Kokkos::make_pair(val, newest_cache);
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
      auto [entity_index, new_cache] = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto val = context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
      return Kokkos::make_pair(val, new_cache);
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    prev_entity_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    return prev_entity_expr_.driver();
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  stk::mesh::EntityRank conn_rank_;
};

template <size_t NumEntities, size_t Ord, typename DriverType>
class EntityExpr : public EntityExprBase<EntityExpr<NumEntities, Ord, DriverType>> {
 public:
  using our_t = EntityExpr<NumEntities, Ord, DriverType>;
  using our_tag = typename EntityExprBase<EntityExpr<NumEntities, Ord, DriverType>>::our_tag;
  using sub_expressions_t = core::tuple<>;
  static constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  EntityExpr(const stk::mesh::EntityRank &rank, const DriverType *driver) : rank_(rank), driver_(driver) {
  }

  /// \brief The rank of the entity we return
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> fmis,
                                                       const NgpEvalContext & /*context*/) const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return fmis[Ord];
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext & /*context*/) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before. means that
        // our eval has cached its result before. Return the cached value
        std::cout << "Reusing cached entity" << std::endl;
        auto cache = std::forward<OldCacheType>(old_cache);
        auto val = get<our_tag>(cache);
        return Kokkos::make_pair(val, cache);
      } else {
        // Our eval result needs cached, but is not yet cached
        auto val = fmis[Ord];
        auto new_cache = append<our_tag>(std::forward<OldCacheType>(old_cache), val);
        return Kokkos::make_pair(val, new_cache);
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto val = fmis[Ord];
      return Kokkos::make_pair(val, old_cache);
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpEvalContext & /*context*/) {
    // Leaf node, nothing to do here.
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  const DriverType *driver() const {
    return driver_;
  }

 private:
  stk::mesh::EntityRank rank_;
  const DriverType *driver_;
};

class NgpForEachEntityExprDriver {
 public:
  NgpForEachEntityExprDriver(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector, stk::mesh::EntityRank rank)
      : ngp_mesh_(ngp_mesh), selector_(selector), rank_(rank) {
  }

  // Default copy/move constructor and assignment operator are fine
  NgpForEachEntityExprDriver(const NgpForEachEntityExprDriver &) = default;
  NgpForEachEntityExprDriver(NgpForEachEntityExprDriver &&) = default;
  NgpForEachEntityExprDriver &operator=(const NgpForEachEntityExprDriver &) = default;
  NgpForEachEntityExprDriver &operator=(NgpForEachEntityExprDriver &&) = default;
  virtual ~NgpForEachEntityExprDriver() = default;

  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

  stk::mesh::Selector selector() const {
    return selector_;
  }

  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  template <typename Expr>
  void run(CachableExprBase<Expr> &expr_base) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh_);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    ::mundy::mesh::for_each_entity_run(
        ngp_mesh_, rank_, selector_, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
          // expr.eval(Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, evaluation_context);

          // Sum the counts of each expression in the tree
          constexpr auto empty_eval_counts = core::make_aggregate();
          constexpr auto eval_counts =
              Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

          // Perform the eval
          auto empty_cache = core::make_aggregate();
          expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, empty_cache, evaluation_context);
        });
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
};

auto make_entity_expr(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                      const stk::mesh::EntityRank &rank) {
  // To ensure that all expressions have the same driver, we store a persistent driver manager
  // on the meta data and use it to memoize the driver for the given rank and selector.

  using driver_map_t = impl::AnyRankSelectorMap<core::make_string_literal("ExprDrivers")>;
  stk::mesh::MetaData &meta_data = bulk_data.mesh_meta_data();
  driver_map_t *driver_map = const_cast<driver_map_t *>(meta_data.get_attribute<driver_map_t>());
  if (driver_map == nullptr) {
    const driver_map_t *new_driver_map = new driver_map_t();
    driver_map = const_cast<driver_map_t *>(meta_data.declare_attribute_with_delete(new_driver_map));
  }

  // Stash our driver in the map if it doesn't already exist
  stk::mesh::NgpMesh &ngp_mesh = get_updated_ngp_mesh(bulk_data);
  const NgpForEachEntityExprDriver *driver_ptr;
  if (driver_map->contains(rank, selector)) {
    // Driver already exists for this rank and selector; reuse it
    NgpForEachEntityExprDriver &existing_driver = driver_map->at<NgpForEachEntityExprDriver>(rank, selector);

    // The NgpMesh may have updated since the last time we created this driver, so update it
    existing_driver = NgpForEachEntityExprDriver(ngp_mesh, selector, rank);
    driver_ptr = &existing_driver;
  } else {
    // Driver doesn't exist yet; create and insert it
    NgpForEachEntityExprDriver new_driver(ngp_mesh, selector, rank);
    driver_map->insert<NgpForEachEntityExprDriver>(rank, selector, std::move(new_driver));
    const NgpForEachEntityExprDriver &inserted_driver = driver_map->at<NgpForEachEntityExprDriver>(rank, selector);
    driver_ptr = &inserted_driver;
  }

  return EntityExpr<1, 0, NgpForEachEntityExprDriver>(rank, driver_ptr);
}
//@}

//! \name Views of mathematical expressions
//@{

/*
Let's assume that the return type of every AccessorExpr is compatable with
operator and we'll simply forward these to the eval.
 +
 -
 *
 \
 +=
 -=
 *=
 /=

Scalar, Vector, Matrix, Quaternion
*/

template <typename DerivedMathExpr>
class MathExprBase;

template <typename TargetExpr, typename SourceExpr>
class AssignExpr : public MathExprBase<AssignExpr<TargetExpr, SourceExpr>> {
 public:
  using our_t = AssignExpr<TargetExpr, SourceExpr>;
  using our_tag = typename MathExprBase<AssignExpr<TargetExpr, SourceExpr>>::our_tag;
  using sub_expressions_t = core::tuple<TargetExpr, SourceExpr>;
  static constexpr size_t num_entities = TargetExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AssignExpr(TargetExpr trg_expr, SourceExpr src_expr) : trg_expr_(trg_expr), src_expr_(src_expr) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    trg_expr_.eval(fmis, context) = src_expr_.eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext &context) const {
    static_assert(!core::has<our_tag, std::remove_reference_t<OldCacheType>>(),
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");

    // Eval our subexpressions first, allowing them to cache their results if necessary
    auto [trg_val, new_cache] = trg_expr_.template cached_eval<EvalCountsType, eval_counts>(
        fmis, std::forward<OldCacheType>(old_cache), context);
    auto [src_val, newer_cache] =
        src_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(new_cache), context);
    trg_val = src_val;
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    src_expr_.flag_read_only(context);
    trg_expr_.flag_overwrite_all(context);
    trg_expr_.propagate_synchronize(context);
    src_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

  auto driver() const {
    auto d = trg_expr_.driver();
    MUNDY_THROW_ASSERT(d == src_expr_.driver(), std::logic_error, "Mismatched drivers in assignment expression");
    return d;
  }

 private:
  TargetExpr trg_expr_;
  SourceExpr src_expr_;
};

#define MUNDY_ACCESSOR_EXPR_OP(OpName, op)                                                                        \
  template <typename LeftMathExpr, typename RightMathExpr>                                                        \
  class OpName##Expr : public MathExprBase<OpName##Expr<LeftMathExpr, RightMathExpr>> {                           \
   public:                                                                                                        \
    using our_t = OpName##Expr<LeftMathExpr, RightMathExpr>;                                                      \
    using our_tag = typename MathExprBase<OpName##Expr<LeftMathExpr, RightMathExpr>>::our_tag;                    \
    using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;                                           \
    static constexpr size_t num_entities = LeftMathExpr::num_entities;                                            \
                                                                                                                  \
    KOKKOS_INLINE_FUNCTION                                                                                        \
    OpName##Expr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {                           \
    }                                                                                                             \
                                                                                                                  \
    KOKKOS_INLINE_FUNCTION                                                                                        \
    OpName##Expr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)                \
        : left_(left.self()), right_(right.self()) {                                                              \
    }                                                                                                             \
                                                                                                                  \
    template <typename OtherExpr>                                                                                 \
    auto operator+(const MathExprBase<OtherExpr> &other) const {                                                  \
      return AddExpr<our_t, OtherExpr>(*this, other.self());                                                      \
    }                                                                                                             \
                                                                                                                  \
    template <typename OtherExpr>                                                                                 \
    auto operator-(const MathExprBase<OtherExpr> &other) const {                                                  \
      return SubExpr<our_t, OtherExpr>(*this, other.self());                                                      \
    }                                                                                                             \
                                                                                                                  \
    template <typename OtherExpr>                                                                                 \
    auto operator*(const MathExprBase<OtherExpr> &other) const {                                                  \
      return MulExpr<our_t, OtherExpr>(*this, other.self());                                                      \
    }                                                                                                             \
                                                                                                                  \
    template <typename OtherExpr>                                                                                 \
    auto operator/(const MathExprBase<OtherExpr> &other) const {                                                  \
      return DivExpr<our_t, OtherExpr>(*this, other.self());                                                      \
    }                                                                                                             \
                                                                                                                  \
    KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,           \
                                     const NgpEvalContext &context) const {                                       \
      return left_.eval(fmis, context) op right_.eval(fmis, context);                                             \
    }                                                                                                             \
                                                                                                                  \
    template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>                         \
    auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache, \
                     const NgpEvalContext &context) const {                                                       \
      static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");                               \
                                                                                                                  \
      if constexpr (get<our_tag>(eval_counts) > 1) {                                                              \
        if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {                              \
          /* The fact that our tag exists in the old cache means that our eval has cached its result before.*/    \
          /* Return the cached value */                                                                           \
          std::cout << "Reusing cached binary math expression for operator " #op << std::endl;            \
          auto cache = std::forward<OldCacheType>(old_cache);                                                     \
          auto val = get<our_tag>(cache);                                                                         \
          return Kokkos::make_pair(val, cache);                                                                   \
        } else {                                                                                                  \
          /* Eval our subexpressions first, allowing them to cache their results if necessary */                  \
          auto [left_val, new_cache] = left_.template cached_eval<EvalCountsType, eval_counts>(                   \
              fmis, std::forward<OldCacheType>(old_cache), context);                                              \
          auto [right_val, newer_cache] =                                                                         \
              right_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(new_cache), context);      \
                                                                                                                  \
          /* Our eval result needs cached, but is not yet cached */                                               \
          auto val = left_val op right_val;                                                                       \
          auto newest_cache = append<our_tag>(std::move(newer_cache), val);                                       \
          return Kokkos::make_pair(val, newest_cache);                                                            \
        }                                                                                                         \
      } else {                                                                                                    \
        /* Eval our subexpressions first, allowing them to cache their results if necessary */                    \
        auto [left_val, new_cache] = left_.template cached_eval<EvalCountsType, eval_counts>(                     \
            fmis, std::forward<OldCacheType>(old_cache), context);                                                \
        auto [right_val, newer_cache] =                                                                           \
            right_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(new_cache), context);        \
                                                                                                                  \
        /* Don't cache our result */                                                                              \
        auto val = left_val op right_val;                                                                         \
        return Kokkos::make_pair(val, newer_cache);                                                               \
      }                                                                                                           \
    }                                                                                                             \
                                                                                                                  \
    void propagate_synchronize(const NgpEvalContext &context) {                                                   \
      left_.flag_read_only(context);                                                                              \
      right_.flag_read_only(context);                                                                             \
      left_.propagate_synchronize(context);                                                                       \
      right_.propagate_synchronize(context);                                                                      \
    }                                                                                                             \
                                                                                                                  \
    void flag_read_only(const NgpEvalContext & /*context*/) {                                                     \
      /* Our return type is naturally read-only. Nothing to do here. */                                           \
    }                                                                                                             \
                                                                                                                  \
    void flag_read_write(const NgpEvalContext & /*context*/) {                                                    \
      std::cout << "Warning: Attempting to write to the return type of a binary expression, which returns a "     \
                   "temporary value."                                                                             \
                << std::endl;                                                                                     \
    }                                                                                                             \
                                                                                                                  \
    void flag_overwrite_all(const NgpEvalContext & /*context*/) {                                                 \
      std::cout << "Warning: Attempting to write to the return type of a binary expression, which returns a "     \
                   "temporary value."                                                                             \
                << std::endl;                                                                                     \
    }                                                                                                             \
                                                                                                                  \
    const auto driver() const {                                                                                   \
      auto d = left_.driver();                                                                                    \
      MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression"); \
      return d;                                                                                                   \
    }                                                                                                             \
                                                                                                                  \
   private:                                                                                                       \
    LeftMathExpr left_;                                                                                           \
    RightMathExpr right_;                                                                                         \
  };

#define MUNDY_ACCESSOR_EXPR_OP_EQUALS(OpName, op_equals)                                                               \
  template <typename LeftMathExpr, typename RightMathExpr>                                                             \
  class OpName##EqualsExpr : public MathExprBase<OpName##EqualsExpr<LeftMathExpr, RightMathExpr>> {                    \
   public:                                                                                                             \
    using our_t = OpName##EqualsExpr<LeftMathExpr, RightMathExpr>;                                                     \
    using our_tag = typename MathExprBase<OpName##EqualsExpr<LeftMathExpr, RightMathExpr>>::our_tag;                   \
    using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;                                                \
    static constexpr size_t num_entities = LeftMathExpr::num_entities;                                                 \
                                                                                                                       \
    KOKKOS_INLINE_FUNCTION                                                                                             \
    OpName##EqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {                          \
    }                                                                                                                  \
                                                                                                                       \
    KOKKOS_INLINE_FUNCTION                                                                                             \
    OpName##EqualsExpr(const EntityExprBase<LeftMathExpr> &left, const EntityExprBase<RightMathExpr> &right)           \
        : left_(left.self()), right_(right.self()) {                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,                \
                                     const NgpEvalContext &context) const {                                            \
      left_.eval(fmis, context) op_equals right_.eval(fmis, context);                                                  \
    }                                                                                                                  \
                                                                                                                       \
    template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>                              \
    void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,      \
                     const NgpEvalContext &context) const {                                                            \
      static_assert(!core::has<our_tag, std::remove_reference_t<OldCacheType>>(),                                      \
                    "The cache somehow contains our tag, but our eval returns void and should never cache anything."); \
      /* Eval our subexpressions first, allowing them to cache their results if necessary */                           \
      auto [left_val, new_cache] = left_.template cached_eval<EvalCountsType, eval_counts>(fmis, old_cache, context);  \
      auto [right_val, newer_cache] =                                                                                  \
          right_.template cached_eval<EvalCountsType, eval_counts>(fmis, new_cache, context);                          \
      left_val op_equals right_val;                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    void propagate_synchronize(const NgpEvalContext &context) {                                                        \
      left_.flag_read_write(context);                                                                                  \
      right_.flag_read_only(context);                                                                                  \
      left_.propagate_synchronize(context);                                                                            \
      right_.propagate_synchronize(context);                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    void flag_read_only(const NgpEvalContext & /*context*/) {                                                          \
      /* Our return type is naturally read-only. Nothing to do here. */                                                \
    }                                                                                                                  \
                                                                                                                       \
    void flag_read_write(const NgpEvalContext & /*context*/) {                                                         \
      std::cout << "Warning: Attempting to write to the return type of a binary expression, which returns a "          \
                   "temporary value."                                                                                  \
                << std::endl;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    void flag_overwrite_all(const NgpEvalContext & /*context*/) {                                                      \
      std::cout << "Warning: Attempting to write to the return type of a binary expression, which returns a "          \
                   "temporary value."                                                                                  \
                << std::endl;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    const auto driver() const {                                                                                        \
      auto d = left_.driver();                                                                                         \
      MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");      \
      return d;                                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
   private:                                                                                                            \
    LeftMathExpr left_;                                                                                                \
    RightMathExpr right_;                                                                                              \
  };

MUNDY_ACCESSOR_EXPR_OP(Add, +)
MUNDY_ACCESSOR_EXPR_OP(Sub, -)
MUNDY_ACCESSOR_EXPR_OP(Div, /)
MUNDY_ACCESSOR_EXPR_OP(Mul, *)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Add, +=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Sub, -=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Div, /=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Mul, *=)

template <typename TaggedAccessorT, typename PrevEntityExpr>
class AccessorExpr : public MathExprBase<AccessorExpr<TaggedAccessorT, PrevEntityExpr>> {
 public:
  using our_t = AccessorExpr<TaggedAccessorT, PrevEntityExpr>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = core::tuple<PrevEntityExpr>;
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(TaggedAccessorT tagged_accessor, const PrevEntityExpr &prev_entity_expr)
      : tagged_accessor_(tagged_accessor), prev_entity_expr_(prev_entity_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(TaggedAccessorT tagged_accessor, const EntityExprBase<PrevEntityExpr> &prev_entity_expr_base)
      : tagged_accessor_(tagged_accessor), prev_entity_expr_(prev_entity_expr_base.self()) {
  }

  auto operator=(const our_t &other) {
    auto expr = AssignExpr<our_t, our_t>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
    requires(!std::is_same_v<OtherExpr, our_t>)
  auto operator=(const MathExprBase<OtherExpr> &other) {
    auto expr = AssignExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
    requires(!std::is_same_v<OtherExpr, our_t>)
  auto operator=(const EntityExprBase<OtherExpr> &other) {
    auto expr = AssignExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  auto operator+(const MathExprBase<OtherExpr> &other) const {
    return AddExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator-(const MathExprBase<OtherExpr> &other) const {
    return SubExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator*(const MathExprBase<OtherExpr> &other) const {
    return MulExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator/(const MathExprBase<OtherExpr> &other) const {
    return DivExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  void operator+=(const MathExprBase<OtherExpr> &other) {
    auto expr = AddExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator-=(const MathExprBase<OtherExpr> &other) {
    auto expr = SubExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator*=(const MathExprBase<OtherExpr> &other) {
    auto expr = MulExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator/=(const MathExprBase<OtherExpr> &other) {
    auto expr = DivExpr<our_t, OtherExpr>(*this, other.self());
    expr.driver()->run(expr);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return tagged_accessor_(entity_index);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext &context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        // Return the cached value and the old cache
        std::cout << "Reusing cached accessor" << std::endl;
        auto cache = std::forward<OldCacheType>(old_cache);
        return Kokkos::make_pair(get<our_tag>(cache), cache);
      } else {
        // Eval our subexpressions first
        auto [entity_index, new_cache] = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);

        // Our eval result needs cached, but is not yet cached
        auto val = tagged_accessor_(entity_index);
        auto newest_cache = append<our_tag>(std::move(new_cache), val);
        return Kokkos::make_pair(val, newest_cache);
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto [entity_index, new_cache] = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto val = tagged_accessor_(entity_index);
      return Kokkos::make_pair(val, new_cache);
    }
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    tagged_accessor_.sync_to_device();
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    tagged_accessor_.sync_to_device();
    tagged_accessor_.modify_on_device();
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    tagged_accessor_.clear_host_sync_state();
    tagged_accessor_.modify_on_device();
  }

  void propagate_synchronize(const NgpEvalContext & /*context*/) {
  }

  const auto driver() const {
    return prev_entity_expr_.driver();
  }

  const PrevEntityExpr &prev_entity_expr() const {
    return prev_entity_expr_;
  }

 private:
  TaggedAccessorT tagged_accessor_;
  PrevEntityExpr prev_entity_expr_;
};

template <typename DerivedMathExpr>
class MathExprBase : public CachableExprBase<DerivedMathExpr> {
 public:
  using our_t = MathExprBase<DerivedMathExpr>;
  using our_tag = typename CachableExprBase<DerivedMathExpr>::our_tag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr MathExprBase() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedMathExpr &self() const noexcept {
    return static_cast<const DerivedMathExpr &>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedMathExpr &self() noexcept {
    return static_cast<DerivedMathExpr &>(*this);
  }

  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                   const Ctx &context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename CacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, CacheType &cache,
                   const Ctx &context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, cache, context);
  }

  template <class Ctx>
  void propagate_synchronize(const Ctx &context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx &context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx &context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx &context) {
    self().flag_overwrite_all(context);
  }

  const auto driver() const {
    return self().driver();
  }

  // template <typename OtherExpr>
  // auto operator+(const MathExprBase<OtherExpr> &other) const {
  //   return AddExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  // }

  // template <typename OtherExpr>
  // auto operator-(const MathExprBase<OtherExpr> &other) const {
  //   return SubExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  // }

  // template <typename OtherExpr>
  // auto operator*(const MathExprBase<OtherExpr> &other) const {
  //   return MulExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  // }

  // template <typename OtherExpr>
  // auto operator/(const MathExprBase<OtherExpr> &other) const {
  //   return DivExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  // }

  // auto operator=(const our_t &other) {
  //   std::cout << "In MathExprBase::operator=(const our_t &)" << std::endl;
  //   auto expr = AssignExpr<DerivedMathExpr, DerivedMathExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  //   requires(!std::is_same_v<OtherExpr, DerivedMathExpr>)
  // auto operator=(const MathExprBase<OtherExpr> &other) {
  //   std::cout << "In MathExprBase::operator=(const MathExprBase<OtherExpr> &)" << std::endl;
  //   auto expr = AssignExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  // auto operator=(const EntityExprBase<OtherExpr> &other) {
  //   std::cout << "In MathExprBase::operator=(const EntityExprBase<OtherExpr> &)" << std::endl;
  //   auto expr = AssignExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  // void operator+=(const MathExprBase<OtherExpr> &other) {
  //   auto expr = AddExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  // void operator-=(const MathExprBase<OtherExpr> &other) {
  //   auto expr = SubExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  // void operator*=(const MathExprBase<OtherExpr> &other) {
  //   auto expr = MulExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }

  // template <typename OtherExpr>
  // void operator/=(const MathExprBase<OtherExpr> &other) {
  //   auto expr = DivExpr<DerivedMathExpr, OtherExpr>(*this, other.self());
  //   expr.driver()->run(expr);
  // }
};
//@}

//! \name Helpers
//@{

template <typename PrevMathExpr>
class CopyExpr : public MathExprBase<CopyExpr<PrevMathExpr>> {
 public:
  using our_t = CopyExpr<PrevMathExpr>;
  using our_tag = typename MathExprBase<CopyExpr<PrevMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<PrevMathExpr>;
  static constexpr size_t num_entities = PrevMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  CopyExpr(const PrevMathExpr &prev_math_expr) : prev_math_expr_(prev_math_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  CopyExpr(const EntityExprBase<PrevMathExpr> &prev_math_expr_base) : prev_math_expr_(prev_math_expr_base.self()) {
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    return copy(prev_math_expr_.eval(fmis, context));
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext &context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before. means that
        // our eval has cached its result before. Return the cached value
        std::cout << "Reusing cached copy expression" << std::endl;
        auto cache = std::forward<OldCacheType>(old_cache);
        auto val = get<our_tag>(cache);
        return Kokkos::make_pair(val, cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto [view, new_cache] = prev_math_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto val = copy(view);
        auto newer_cache = append<our_tag>(std::move(new_cache), val);
        return Kokkos::make_pair(val, newer_cache);
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto [view, new_cache] = prev_math_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto val = copy(view);
      return Kokkos::make_pair(val, new_cache);
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    prev_math_expr_.flag_read_only(context);
    prev_math_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Nothing to do here
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout << "Warning: Attempting to write to the return type of a copy expression, which returns a "
                 "temporary value."
              << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout << "Warning: Attempting to write to the return type of a copy expression, which returns a "
                 "temporary value."
              << std::endl;
  }

  auto driver() const {
    return prev_math_expr_.driver();
  }

 private:
  PrevMathExpr prev_math_expr_;
};

/// \brief Deep copies the value of the given expression.
template <typename Expr>
auto copy(const MathExprBase<Expr> &expr) {
  return CopyExpr<Expr>(expr.self());
}

template <typename... TrgSrcExprPairs>
class FusedAssignExpr : public MathExprBase<FusedAssignExpr<TrgSrcExprPairs...>> {
 public:
  using our_t = FusedAssignExpr<TrgSrcExprPairs...>;
  using our_tag = typename MathExprBase<FusedAssignExpr<TrgSrcExprPairs...>>::our_tag;
  using sub_expressions_t = core::tuple<TrgSrcExprPairs...>;
  static constexpr size_t num_pairs = sizeof...(TrgSrcExprPairs) / 2;
  static_assert(sizeof...(TrgSrcExprPairs) % 2 == 0,
                "The number of target/source expression pairs in FusedAssignExpr must be even.");

  static constexpr size_t num_entities = core::tuple_element_t<0, core::tuple<TrgSrcExprPairs...>>::num_entities;
  static_assert(((num_entities == TrgSrcExprPairs::num_entities) && ...),
                "All expressions in the fused assign must have the same number of entities.");

  KOKKOS_INLINE_FUNCTION
  FusedAssignExpr(const TrgSrcExprPairs &...exprs) : exprs_(core::make_tuple(exprs...)) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    // Eval all expressions, storing their results for later.
    auto all_values = impl::expr_chain(exprs_, fmis, context);

    // Set all right hand sides to their corresponding left hand sides.
    set_impl(all_values, std::make_index_sequence<2 * num_pairs>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, OldCacheType &&old_cache,
                   const NgpEvalContext &context) const {
    static_assert(!core::has<our_tag, std::remove_reference_t<OldCacheType>>(),
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");

    // Eval all expressions, storing their results for later.
    auto [all_values, final_cache] = impl::cached_expr_chain<EvalCountsType, eval_counts>(
        exprs_, fmis, std::forward<OldCacheType>(old_cache), context);

    // Set all right hand sides to their corresponding left hand sides.
    set_impl(all_values, std::make_index_sequence<2 * num_pairs>{});
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    propagate_synchronize_impl(std::make_index_sequence<num_pairs>{}, context);
  }

  const auto driver() const {
    // TODO(palmerb4): Check that all drivers are the same.
    return core::get<0>(exprs_).driver();
  }

  //  private:
  template <typename AllValuesType, size_t... Is>
  KOKKOS_INLINE_FUNCTION static void set_impl(AllValuesType &all_values, std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == 2 * num_pairs, "Index sequence size must match number of target + source exprs.");
    (set_i_impl<Is>(all_values), ...);
  }

  template <size_t I, typename AllValuesType>
  KOKKOS_INLINE_FUNCTION static void set_i_impl(AllValuesType &all_values) {
    // I = 0, 1, 2, 3
    // 0 % 2 -> 0
    // 1 % 2 -> 1
    // 2 % 2 -> 0
    if constexpr (I % 2 == 0) {
      core::get<I>(all_values) = core::get<I + 1>(all_values);
    }
  }

  template <size_t... Is, typename Ctx>
  KOKKOS_INLINE_FUNCTION void propagate_synchronize_impl(std::index_sequence<Is...>, const Ctx &context) {
    static_assert(sizeof...(Is) == num_pairs, "Index sequence size must match number of target/source pairs.");

    // Flag all right hand sides as read-only and all left hand sides as overwrite-all.
    (core::get<2 * Is + 1>(exprs_).flag_read_only(context), ...);
    (core::get<2 * Is>(exprs_).flag_overwrite_all(context), ...);

    // Propagate synchronize to all expressions.
    (core::get<2 * Is + 1>(exprs_).propagate_synchronize(context), ...);
    (core::get<2 * Is>(exprs_).propagate_synchronize(context), ...);
  }

  core::tuple<TrgSrcExprPairs...> exprs_;
};

/// \brief Perform a fused assignment operation
/// fused_assign(
//       trg_expr1, /*=*/ src_expr1,
///      trg_expr2, /*=*/ src_expr2,
///               ...
///      trg_exprN, /*=*/ src_exprN);
template <typename... TrgSrcExprPairs>
void fused_assign(const TrgSrcExprPairs &...exprs) {
  constexpr size_t num_trg_src_pairs = sizeof...(TrgSrcExprPairs);
  static_assert(num_trg_src_pairs % 2 == 0,
                "The number of target/source expression pairs in fused_assign must be even.");
  FusedAssignExpr<TrgSrcExprPairs...> fused_expr(exprs...);
  fused_expr.driver()->run(fused_expr);
}

/*
I have some problems without cache system that I want to discuss and fledge out here.
First and foremost, if a math expression concatenates the cache of the left and the right, then there
can be no reuse between the two. If, instead, it uses the cache of the left and passes it to the right, then
there is no possibility for the right operation to add objects to the cache that the left isn't aware of.

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

How do we then construct the initial aggregate?
 - Every uniquely typed expression contributes a count to the eval_counts aggregate.
 - We only want the cache aggregate and is_cached aggregate to store types for which eval_count.get<TAG>() > 1.
 - A single init_cache<eval_counts>() function that returns a default constructed aggregate containing only TAGs for
    which eval_count.get<TAG>() > 1.

New interface:

/// \brief Evaluate the expression
auto eval<EvalCountsType, eval_counts, CacheType>(fmis, cache, context) const {
  static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

  if constexpr (get<our_tag>(eval_counts) > 1) {
    if constexpr (core::has<our_tag, std::remove_reference_t<OldCacheType>>()) {
      // The fact that our tag exists in the old cache means that our eval has cached its result before. means that our
eval has cached its result before.
      // Return the cached value
      return get<our_tag>(cache);
    } else {
      // Our eval result needs cached, but is not yet cached
      auto val = ... compute the value ...;
      get<our_tag>(cache) = val;
      return val;
    }
  } else {
    // We don't need to cache our value, so just compute and return it
    return ... compute the value ...;
  }
}

/// \brief Default construct eval_counts with our tag and our sub-expressions tags set to 0
template <OldEvalCountsType, old_eval_counts>
static constexpr auto default_init_eval_counts() {
  if constexpr (has<our_tag>(old_eval_counts)) {
    // Our tag already exists, which also means that the tags of our sub-expressions already exist.
    // Nothing to do here.
    return old_eval_counts;
  } else {
    // Our tag doesn't exist in the old eval_counts, so we need to add it
    constexpr auto new_eval_counts = old_eval_counts.append<our_tag>(0);

    // Propagate the tags from our sub-expressions
    auto newer_eval_counts =
        PrevExpr::default_init_eval_counts<decltype(new_eval_counts), new_eval_counts>();
    return newer_eval_counts;
  }
}

/// \brief Update is_cached if our eval cached its result.
template <OldIsCachedType, old_is_cached, EvalCountsType, eval_counts>
static constexpr auto update_is_cached() {
  static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

  if constexpr (get<our_tag>(eval_counts) > 1) {
    if constexpr (has<our_tag>(old_is_cached)) {
      // The fact that our tag exists in old_is_cached means that our eval has cached its result before.
      // Nothing to do here.
      return old_is_cached;
    } else {
      // Our eval result needs cached, but is not yet cached, so it's our responsibility to add it
      constexpr auto new_is_cached = old_is_cached.append<our_tag>(true);

      // Propagate the tags from our sub-expressions
      auto newer_is_cached =
          PrevExpr::update_is_cached<decltype(new_is_cached), new_is_cached, EvalCountsType, eval_counts>();
      return newer_is_cached;
    }
  } else {
    // We don't need to cache our value, so just propagate the tags from our sub-expressions
    return PrevExpr::update_is_cached<OldIsCachedType, old_is_cached, EvalCountsType, eval_counts>();
  }
}

/// \brief Fill the cache with default constructed objects for each reused object that needs to be cached
auto init_cache<EvalCountsType, eval_counts, OldCacheType>(old_cache) {
  static_assert(has<our_tag>(eval_counts), "The eval_counts type must contain our tag");

  if constexpr (get<our_tag>(eval_counts) > 1) {
    // We need to use our cache, either we are the evaluator or we are the recipient of a cached value
    if constexpr (has<our_tag>(OldCacheType)) {
      // Our tag already exists, which also means that the tags of our sub-expressions already exist.
      // Nothing to do here.
      return old_cache;
    } else {
      // Our tag doesn't exist in the old cache, so we need to add it
      auto new_cache = append<our_tag>(old_cache, ...default value...);

      // Propagate the tags from our sub-expressions
      auto newer_cache = PrevExpr::init_cache<eval_counts>(new_cache);
      return newer_cache;
    }
  } else {
    // We don't need to cache our value, so just propagate the tags from our sub-expressions
    return PrevExpr::init_cache<eval_counts>(old_cache);
  }
}


Final logic issue (hopefully):
Views are not always default constructable, meaning that we cannot construct an emtpy aggregate cache
and then populate it via copy assignment. Instead, if we intend to cache our result, we need to take in the old_cache
and return a new_cache, otherwise, we can just return our eval result.



AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

Well, I made a mistake in my cache logic above. Accessor expressions do not return a static value. They are the only
type with that property. The only way you can cache an accessor is if it is a tagged accessor. The reason is that the
tag says ensures that no two accessors with the same tag can ever return different values, effectively making
AccessorExpr's eval function static.

*/

//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPACCESSOREXPR_HPP_
