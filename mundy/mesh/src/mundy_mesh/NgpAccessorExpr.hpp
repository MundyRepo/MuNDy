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
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_core/tuple.hpp>          // for mundy::core::tuple
#include <mundy_core/aggregate.hpp>          // for mundy::core::aggregate
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
  template <typename Tag, typename AggregateType>
  static constexpr auto increment_tag_count(const Tag &, const AggregateType &agg) {
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

  template <typename SubExprTuple, size_t I, typename OldIsCachedType, OldIsCachedType old_is_cached,
            typename EvalCountsType, EvalCountsType eval_counts>
  static constexpr auto update_is_cached_recurse() {
    if constexpr (I < SubExprTuple::size()) {
      using sub_expr_t = core::tuple_element_t<I, SubExprTuple>;
      // Recurse into the sub-expression
      constexpr auto updated_is_cached =
          sub_expr_t::template update_is_cached<OldIsCachedType, old_is_cached, EvalCountsType, eval_counts>();
      return update_is_cached_recurse<SubExprTuple, I + 1, decltype(updated_is_cached), updated_is_cached,
                                      EvalCountsType, eval_counts>();
    } else {
      return old_is_cached;
    }
  }

  template <typename SubExprTuple, size_t I, size_t NumEntities, typename EvalCountsType, EvalCountsType eval_counts,
            typename OldCacheType>
  static auto init_cache_recurse(const OldCacheType &old_cache) {
    if constexpr (I < SubExprTuple::size()) {
      using sub_expr_t = core::tuple_element_t<I, SubExprTuple>;
      // Recurse into the sub-expression
      auto updated_cache = sub_expr_t::template init_cache<NumEntities, EvalCountsType, eval_counts>(old_cache);
      return init_cache_recurse<SubExprTuple, I + 1, NumEntities, EvalCountsType, eval_counts>(updated_cache);
    } else {
      return old_cache;
    }
  }

 public:
  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedExpr &self() const noexcept {
    return static_cast<const DerivedExpr &>(*this);
  }

  /// \brief Evaluate the expression
  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, CacheType &cache,
                   const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, class Ctx>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
  }

  /// \brief Update eval_counts by incrementing the counts for our tag and our sub-expressions tags
  template <typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  static constexpr auto increment_eval_counts() {
    constexpr auto new_eval_counts = increment_tag_count(our_tag{}, old_eval_counts);
    using sub_exprs = typename DerivedExpr::sub_expressions_t;
    return increment_eval_counts_recurse<sub_exprs, 0, decltype(new_eval_counts), new_eval_counts>();
  }

  /// \brief Update is_cached if our eval cached its result.
  template <typename OldIsCachedType, OldIsCachedType old_is_cached, typename EvalCountsType,
            EvalCountsType eval_counts>
  static constexpr auto update_is_cached() {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(old_is_cached)) {
        // The fact that our tag exists in old_is_cached means that our eval has cached its result before.
        // Nothing to do here.
        return old_is_cached;
      } else {
        // Our eval result needs cached, but is not yet cached, so it's our responsibility to add it
        constexpr auto new_is_cached = append<our_tag>(old_is_cached, true);

        // DerivedEntityExpr::sub_expressions_t is a tuple of all sub-expression types
        // recurse through the sub-expressions performing update_is_cached on the result of the previous
        using sub_exprs = typename DerivedExpr::sub_expressions_t;
        return update_is_cached_recurse<sub_exprs, 0, decltype(new_is_cached), new_is_cached, EvalCountsType, eval_counts>();
      }
    } else {
      // We don't need to cache our value, so just propagate the tags from our sub-expressions
      using sub_exprs = typename DerivedExpr::sub_expressions_t;
      return update_is_cached_recurse<sub_exprs, 0, OldIsCachedType, old_is_cached, EvalCountsType, eval_counts>();
    }
  }

  /// \brief Fill the cache with default constructed objects for each reused object that needs to be cached
  template <size_t NumEntities, typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  static auto init_cache(const OldCacheType &old_cache) {
    static_assert(has<our_tag>(eval_counts), "The eval_counts type must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      // We need to use our cache, either we are the evaluator or we are the recipient of a cached value
      if constexpr (has<our_tag>(old_cache)) {
        // Our tag already exists, which also means that the tags of our sub-expressions already exist.
        // Nothing to do here.
        return old_cache;
      } else {
        // Our tag doesn't exist in the old cache, so we need to add it
        using eval_return_t =
            decltype(self().eval(std::declval<const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &>(),
                                 std::declval<const NgpEvalContext &>()));
        auto new_cache = append<our_tag>(old_cache, eval_return_t{});

        // Propagate the tags from our sub-expressions
        using sub_exprs = typename DerivedExpr::sub_expressions_t;
        return init_cache_recurse<sub_exprs, 0, NumEntities, EvalCountsType, eval_counts>(new_cache);
      }
    } else {
      // We don't need to cache our value, so just propagate the tags from our sub-expressions
      using sub_exprs = typename DerivedExpr::sub_expressions_t;
      return init_cache_recurse<sub_exprs, 0, NumEntities, EvalCountsType, eval_counts>(old_cache);
    }
  }
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

  /// \brief Evaluate the expression
  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fmi, const Ctx &context) const {
    return self().eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, CacheType &cache,
                   const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, class Ctx>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
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

  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const stk::mesh::FastMeshIndex &fmi,
                                                const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmi, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext &context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first
        stk::mesh::FastMeshIndex entity_index =
            prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache,
                                                                                                         context);

        // Our eval result needs cached, but is not yet cached
        stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
        auto val = context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
      stk::mesh::FastMeshIndex entity_index =
          prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache,
                                                                                                       context);
      return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext &context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first
        stk::mesh::FastMeshIndex entity_index =
            prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache,
                                                                                                         context);

        // Our eval result needs cached, but is not yet cached
        stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
        auto val = context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
      stk::mesh::FastMeshIndex entity_index =
          prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache,
                                                                                                       context);
      return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
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
    return fmis[ordinal_];
  }

  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex fmi,
                                                       const NgpEvalContext & /*context*/) const
    requires(num_entities == 1)
  {
    static_assert(Ord == 0, "EntityExpr with a single entity must have Ord == 0");
    return fmi;
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & /*context*/) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Our eval result needs cached, but is not yet cached
        auto val = fmis[ordinal_];
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      return fmis[ordinal_];
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & /*context*/) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");
    static_assert(Ord == 0, "EntityExpr with a single entity must have Ord == 0");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Our eval result needs cached, but is not yet cached
        get<our_tag>(cache) = fmi;
        return fmi;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      return fmi;
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
  static constexpr size_t ordinal_ = Ord;
  const DriverType *driver_;
};

class NgpForEachEntityExprDriver {
 public:
  NgpForEachEntityExprDriver(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector, stk::mesh::EntityRank rank)
      : ngp_mesh_(ngp_mesh), selector_(selector), rank_(rank) {
  }

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
  void run(const CachableExprBase<Expr> &expr) const {
    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh_);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    ::mundy::mesh::for_each_entity_run(
        ngp_mesh_, rank_, selector_, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
          // Setup the reused cache for this entity
          constexpr auto empty_eval_counts = core::make_aggregate();
          constexpr auto empty_is_cached = core::make_aggregate();
          auto empty_cache = core::make_aggregate();

          // Sum the counts of each expression in the tree
          constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

          // Default initialize each cached object
          auto cache = expr.template init_cache<1, decltype(eval_counts), eval_counts>(core::make_tuple());

          expr.template cached_eval<decltype(empty_is_cached), empty_is_cached, decltype(eval_counts), eval_counts>(
              entity_index, cache, evaluation_context);
        });
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
};

auto make_entity_expr(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                      const stk::mesh::EntityRank &rank) {
  // Create a single driver and pass it to the EntityExpr
  static NgpForEachEntityExprDriver driver(ngp_mesh, selector, rank);
  return EntityExpr<1, 0, NgpForEachEntityExprDriver>(rank, &driver);
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

  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                   const Ctx &context) const {
    return self().eval(fmis, context);
  }

  template <class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, const Ctx &context) const {
    return self().eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, size_t NumEntities, class Ctx>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, CacheType &cache,
                   const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, class Ctx>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const Ctx &context) const {
    self().template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
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

  template <typename OtherExpr>
  auto operator+(const MathExprBase<OtherExpr> &other) const {
    return AddExpr<DerivedMathExpr, OtherExpr>(*this, other);
  }

  template <typename OtherExpr>
  auto operator-(const MathExprBase<OtherExpr> &other) const {
    return SubExpr<DerivedMathExpr, OtherExpr>(*this, other);
  }

  template <typename OtherExpr>
  auto operator*(const MathExprBase<OtherExpr> &other) const {
    return MulExpr<DerivedMathExpr, OtherExpr>(*this, other);
  }

  template <typename OtherExpr>
  auto operator/(const MathExprBase<OtherExpr> &other) const {
    return DivExpr<DerivedMathExpr, OtherExpr>(*this, other);
  }

  template <typename OtherExpr>
  auto operator=(const MathExprBase<OtherExpr> &other) {
    auto expr = AssignExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  auto operator=(const EntityExprBase<OtherExpr> &other) {
    auto expr = AssignExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator+=(const MathExprBase<OtherExpr> &other) {
    auto expr = AddExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator-=(const MathExprBase<OtherExpr> &other) {
    auto expr = SubExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator*=(const MathExprBase<OtherExpr> &other) {
    auto expr = MulExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator/=(const MathExprBase<OtherExpr> &other) {
    auto expr = DivExpr<DerivedMathExpr, OtherExpr>(*this, other);
    expr.driver()->run(expr);
  }
};

template <typename LeftMathExpr, typename RightMathExpr>
class AddExpr : public MathExprBase<AddExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = AddExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<AddExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AddExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  AddExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    return left_.eval(fmis, context) + right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    return left_.eval(fmi, context) + right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val + right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);

      // Don't cache our result
      return left_val + right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val + right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);

      // Don't cache our result
      return left_val + right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class SubExpr : public MathExprBase<SubExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = SubExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<SubExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  SubExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  SubExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    return left_.eval(fmis, context) - right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    return left_.eval(fmi, context) - right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val - right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);

      // Don't cache our result
      return left_val - right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val - right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);

      // Don't cache our result
      return left_val - right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class MulExpr : public MathExprBase<MulExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = MulExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<MulExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  MulExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  MulExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    return left_.eval(fmis, context) * right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    return left_.eval(fmi, context) * right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val * right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);

      // Don't cache our result
      return left_val * right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val * right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);

      // Don't cache our result
      return left_val * right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class DivExpr : public MathExprBase<DivExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = DivExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<DivExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  DivExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  DivExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    return left_.eval(fmis, context) / right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi,
                                   const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    return left_.eval(fmi, context) / right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val / right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);

      // Don't cache our result
      return left_val / right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);

        // Our eval result needs cached, but is not yet cached
        auto val = left_val / right_val;
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);

      // Don't cache our result
      return left_val / right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

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

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    trg_expr_.eval(fmi, context) = src_expr_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto trg_val =
            trg_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            TargetExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto src_val =
            src_expr_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);
        trg_val = src_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto trg_val =
          trg_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          TargetExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto src_val =
          src_expr_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);
      trg_val = src_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto trg_val =
            trg_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            TargetExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto src_val =
            src_expr_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);
        trg_val = src_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto trg_val =
          trg_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          TargetExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto src_val =
          src_expr_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);
      trg_val = src_val;
    }
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

template <typename LeftMathExpr, typename RightMathExpr>
class AddEqualsExpr : public MathExprBase<AddEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = AddEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<AddEqualsExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AddEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  AddEqualsExpr(const EntityExprBase<LeftMathExpr> &left, const EntityExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    left_.eval(fmis, context) += right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    left_.eval(fmi, context) += right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);
        left_val += right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);
      left_val += right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);
        left_val += right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);
      left_val += right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class SubEqualsExpr : public MathExprBase<SubEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = SubEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<SubEqualsExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  SubEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  SubEqualsExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    left_.eval(fmis, context) -= right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    left_.eval(fmi, context) -= right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);
        left_val -= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);
      left_val -= right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);
        left_val -= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);
      left_val -= right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
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

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class MulEqualsExpr : public MathExprBase<MulEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = MulEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<MulEqualsExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  MulEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  MulEqualsExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    left_.eval(fmis, context) *= right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    left_.eval(fmi, context) *= right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);
        left_val *= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);
      left_val *= right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);
        left_val *= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);
      left_val *= right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
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

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class DivEqualsExpr : public MathExprBase<DivEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = DivEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<DivEqualsExpr<LeftMathExpr, RightMathExpr>>::our_tag;
  using sub_expressions_t = core::tuple<LeftMathExpr, RightMathExpr>;
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  DivEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  DivEqualsExpr(const MathExprBase<LeftMathExpr> &left, const MathExprBase<RightMathExpr> &right)
      : left_(left.self()), right_(right.self()) {
  }

  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    left_.eval(fmis, context) /= right_.eval(fmis, context);
  }

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    left_.eval(fmi, context) /= right_.eval(fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmis, cache, context);
        left_val /= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmis, cache, context);
      left_val /= right_val;
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        auto left_val =
            left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        constexpr auto updated_is_cached =
            LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
        auto right_val =
            right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
                fmi, cache, context);
        left_val /= right_val;
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      // Eval our subexpressions first, allowing them to cache their results if necessary
      auto left_val =
          left_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      constexpr auto updated_is_cached =
          LeftMathExpr::template update_is_cached<IsCachedType, is_cached, EvalCountsType, eval_counts>();
      auto right_val =
          right_.template cached_eval<decltype(updated_is_cached), updated_is_cached, EvalCountsType, eval_counts>(
              fmi, cache, context);
      left_val /= right_val;
    }
  }

  void propagate_synchronize(const NgpEvalContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
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

  const auto driver() const {
    auto d = left_.driver();
    MUNDY_THROW_ASSERT(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
    return d;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename AccessorT, typename PrevEntityExpr>
class AccessorExpr : public MathExprBase<AccessorExpr<AccessorT, PrevEntityExpr>> {
 public:
  using our_t = AccessorExpr<AccessorT, PrevEntityExpr>;
  using our_tag = typename MathExprBase<AccessorExpr<AccessorT, PrevEntityExpr>>::our_tag;
  using sub_expressions_t = core::tuple<PrevEntityExpr>;
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(AccessorT accessor, const PrevEntityExpr &prev_entity_expr)
      : accessor_(accessor), prev_entity_expr_(prev_entity_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(AccessorT accessor, const EntityExprBase<PrevEntityExpr> &prev_entity_expr_base)
      : accessor_(accessor), prev_entity_expr_(prev_entity_expr_base.self()) {
  }

  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                   const NgpEvalContext &context) const {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return accessor_(entity_index);
  }

  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmi, context);
    return accessor_(entity_index);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Our eval result needs cached, but is not yet cached
        stk::mesh::FastMeshIndex entity_index =
            prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
        auto val = accessor_(entity_index);
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::FastMeshIndex entity_index =
          prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context);
      return accessor_(entity_index);
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  auto cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Our eval result needs cached, but is not yet cached
        stk::mesh::FastMeshIndex entity_index =
            prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
        auto val = accessor_(entity_index);
        get<our_tag>(cache) = val;
        return val;
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::FastMeshIndex entity_index =
          prev_entity_expr_.template cached_eval<IsCachedType, is_cached, EvalCountsType, eval_counts>(fmi, cache, context);
      return accessor_(entity_index);
    }
  }

  void flag_read_only(const NgpEvalContext & /*context*/) {
    accessor_.sync_to_device();
  }

  void flag_read_write(const NgpEvalContext & /*context*/) {
    accessor_.sync_to_device();
    accessor_.modify_on_device();
  }

  void flag_overwrite_all(const NgpEvalContext & /*context*/) {
    accessor_.clear_host_sync_state();
    accessor_.modify_on_device();
  }

  void propagate_synchronize(const NgpEvalContext & /*context*/) {
  }

  const auto driver() const {
    return prev_entity_expr_.driver();
  }

 private:
  AccessorT accessor_;
  PrevEntityExpr prev_entity_expr_;
};
//@}

//! \name Helpers
//@{

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
    eval_impl(std::make_index_sequence<num_pairs>{}, fmis, context);
  }

  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, const NgpEvalContext &context) const
    requires(num_entities == 1)
  {
    eval_impl(std::make_index_sequence<num_pairs>{}, fmi, context);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        cached_eval_without_reuse_impl<IsCachedType, is_cached, EvalCountsType, eval_counts>(
            fmis, cache, std::make_index_sequence<num_pairs>{});

        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      cached_eval_without_reuse_impl<IsCachedType, is_cached, EvalCountsType, eval_counts>(
          fmis, cache, std::make_index_sequence<num_pairs>{});
    }
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType>
  void cached_eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                   const NgpEvalContext & context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (get<our_tag>(eval_counts) > 1) {
      if constexpr (has<our_tag>(is_cached)) {
        // The fact that our tag exists in is_cached means that our eval has cached its result before.
        // Return the cached value
        return get<our_tag>(cache);
      } else {
        // Eval our subexpressions first, allowing them to cache their results if necessary
        cached_eval_without_reuse_impl<IsCachedType, is_cached, EvalCountsType, eval_counts>(
            fmi, cache, std::make_index_sequence<num_pairs>{});

        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Attempting to cache the result of an assignment expression, which returns void.");
      }
    } else {
      cached_eval_without_reuse_impl<IsCachedType, is_cached, EvalCountsType, eval_counts>(
          fmi, cache, std::make_index_sequence<num_pairs>{});
    }
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

  void propagate_synchronize(const NgpEvalContext & /*context*/) {
    propagate_synchronize_impl(std::make_index_sequence<num_pairs>{});
  }

  const auto driver() const {
    // TODO(palmerb4): Check that all drivers are the same.
    return core::get<0>(exprs_).driver();
  }

 private:
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION void eval_impl(std::index_sequence<Is...>,
                                        const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                        const NgpEvalContext &context) const {
    static_assert(sizeof...(Is) == num_pairs, "Index sequence size must match number of target/source pairs.");

    // Eval all right hand sides, storing their results for later.
    auto rhs_values = core::make_tuple(core::get<2 * Is + 1>(exprs_).eval(fmis, context)...);

    // Set all right hand sides to their corresponding left hand sides.
    ((core::get<2 * Is>(exprs_).eval(fmis, context) = core::get<Is>(rhs_values)), ...);
  }

  template <typename IsCachedType, IsCachedType is_cached, typename EvalCountsType, EvalCountsType eval_counts,
            typename CacheType, size_t... Is>
  void cached_eval_without_reuse_impl(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                      CacheType &cache, const NgpEvalContext &context,
                                      std::index_sequence<Is...>) const {
    static_assert(sizeof...(Is) == 2 * num_pairs, "Index sequence size must match 2 * number of target/source pairs.");
    // Eval all right hand sides, storing their results for later.
    auto all_values =
        core::make_tuple(eval_i<Is, IsCachedType, is_cached, EvalCountsType, eval_counts>(fmis, cache, context)...);

    // Set all right hand sides to their corresponding left hand sides.
    (set_impl<Is>(all_values), ...);
  }

  template <size_t I, typename AllValuesType>
  KOKKOS_INLINE_FUNCTION void set_impl(AllValuesType &all_values) const {
    if constexpr (I % 2 == 0) {
      core::get<I>(all_values) = core::get<I + 1>(all_values);
    }
  }

  template <size_t I, typename IsCachedType0, IsCachedType0 is_cached0, typename EvalCountsType,
            EvalCountsType eval_counts, typename CacheType>
  auto eval_i(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
              const NgpEvalContext &context) {
    // Update is_cached using all previous left and right hand sides

    // I'm fretting about order of operations. We need to make sure to store sub_expressions in the correct order so
    // that we are reusing a non-default constructed object.

    constexpr auto is_cached_i =
        update_is_cached_recurse<0, I, IsCachedType0, is_cached0, EvalCountsType, eval_counts>();
    return exprs_.template cached_eval<delctype(is_cached_i), is_cached_i, EvalCountsType, eval_counts>(fmis, cache,
                                                                                                        context);
  }

  template <size_t StartI, size_t EndI, typename IsCachedTypeOld, IsCachedTypeOld is_cached_old,
            typename EvalCountsType, EvalCountsType eval_counts>
  static constexpr auto update_is_cached_recurse() {
    if constexpr (StartI < EndI) {
      using sub_expr_t = core::tuple_element_t<StartI, core::tuple<TrgSrcExprPairs...>>;
      constexpr auto updated_is_cached =
          sub_expr_t::template update_is_cached<IsCachedTypeOld, is_cached_old, EvalCountsType, eval_counts>();
      return update_is_cached_recurse<StartI + 1, EndI, decltype(updated_is_cached), updated_is_cached, EvalCountsType,
                                      eval_counts>();
    } else {
      return is_cached_old;
    }
  }

  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION void propagate_synchronize_impl(std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == num_pairs, "Index sequence size must match number of target/source pairs.");

    // Flag all right hand sides as read-only and all left hand sides as overwrite-all.
    (core::get<2 * Is + 1>(exprs_).flag_read_only(), ...);
    (core::get<2 * Is>(exprs_).flag_overwrite_all(), ...);

    // Propagate synchronize to all expressions.
    (core::get<2 * Is + 1>(exprs_).propagate_synchronize(), ...);
    (core::get<2 * Is>(exprs_).propagate_synchronize(), ...);
  }

  core::tuple<TrgSrcExprPairs...> exprs_;
};

/// \brief Perform a fused assignment operation
/// fused_assign(
//       trg_expr1 = src_expr1,
///      trg_expr2 = src_expr2,
///               ...
///      trg_exprN = src_exprN);
template <typename... TrgSrcExprPairs>
void fused_assign(const TrgSrcExprPairs&... exprs) {
  constexpr size_t num_trg_src_pairs = sizeof...(TrgSrcExprPairs);
  static_assert(num_trg_src_pairs % 2 == 0,
                "The number of target/source expression pairs in fused_assign must be even.");
  FusedAssignExpr<TrgSrcExprPairs...> fused_expr(exprs...);
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
auto eval<IsCachedType, is_cached, EvalCountsType, eval_counts, CacheType>(fmis, cache, context) const {
  static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

  if constexpr (get<our_tag>(eval_counts) > 1) {
    if constexpr (has<our_tag>(is_cached)) {
      // The fact that our tag exists in is_cached means that our eval has cached its result before.
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
*/

//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPACCESSOREXPR_HPP_
