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
  NgpContext evaluation_context(ngp_mesh);
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
// the EntityExpr, which is a poor name since it is closer to the EntityContext. This is as apposed to the ExecutionContext.
// But like, just loop that all together into Context and be done with it. We can still type specialize the evals on 
// a template of the class, so there's no loss of functionality.

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

class NgpContext {
 public:
  KOKKOS_INLINE_FUNCTION
  NgpContext(stk::mesh::NgpMesh ngp_mesh) : ngp_mesh_(ngp_mesh) {
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

template <typename DerivedEntityExpr>
class EntityExprBase {
 public:
  using our_t = EntityExprBase<DerivedEntityExpr>;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedEntityExpr &self() const noexcept {
    return static_cast<const DerivedEntityExpr &>(*this);
  }

  stk::mesh::NgpMesh ngp_mesh() const {
    return self().ngp_mesh();
  }

  template<size_t Ord = 0>
  stk::mesh::Selector selector() const {
    return self().template selector<Ord>();
  }

  template<size_t Ord = 0>
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return self().template rank<Ord>();
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached,  //
            size_t NumEntities, typename CacheType, class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis,
                                                       CacheType &cache, const Ctx &context) const {
    return self().template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType, class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                                       const Ctx &context) const {
    return self().template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static auto init_cache() {
    return DerivedEntityExpr::init_cache();
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return DerivedEntityExpr::init_is_cached();
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    return DerivedEntityExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
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
};

// How would all of this look in a for_each_entity_expr_eval
// template <typename EntityExpr>
// void for_each_entity_evaluate_expr(
//     const stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
//     const EntityExprBase<EntityExpr> &entity_expr) {
//   static_assert(EntityExpr::num_entities == 1,
//                 "for_each_entity_evaluate_expr only works with single-entity expressions");
//   stk::mesh::EntityRank rank = entity_expr.rank();

//   // Sync all fields to the appropriate space and mark modified where necessary
//   NgpContext evaluation_context(ngp_mesh);
//   entity_expr.propagate_synchronize(evaluation_context);

//   // Perform the evaluation
//   ::mundy::mesh::for_each_entity_run(
//       ngp_mesh, selector, rank, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
//         // Setup the reused cache for this entity
//         auto cache = EntityExpr::init_cache();
//         constexpr auto is_cached = EntityExpr::init_is_cached();
//         constexpr size_t cache_offset = 0;
//         entity_expr.template eval<cache_offset, is_cached>(entity_index, cache, evaluation_context);
//       });
// }
template <typename EntityExpr>
void for_each_entity_evaluate_expr(const EntityExpr &entity_expr) {
  static_assert(EntityExpr::num_entities == 1,
                "for_each_entity_evaluate_expr only works with single-entity expressions");
  stk::mesh::EntityRank rank = entity_expr.rank();
  stk::mesh::Selector selector = entity_expr.selector();
  stk::mesh::NgpMesh ngp_mesh = entity_expr.ngp_mesh();

  // Sync all fields to the appropriate space and mark modified where necessary
  NgpContext evaluation_context(ngp_mesh);
  entity_expr.propagate_synchronize(evaluation_context);

  // Perform the evaluation
  ::mundy::mesh::for_each_entity_run(
      ngp_mesh, selector, rank, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
        // Setup the reused cache for this entity
        auto cache = EntityExpr::init_cache();
        constexpr auto is_cached = EntityExpr::init_is_cached();
        constexpr size_t cache_offset = 0;
        constexpr size_t cache_size = decltype(cache)::size();
        entity_expr.template eval<cache_offset, cache_size, is_cached>(entity_index, cache, evaluation_context);
      });
}

template <typename PrevEntityExpr>
class ConnectedEntitiesExpr : public EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>> {
 public:
  using our_t = ConnectedEntitiesExpr<PrevEntityExpr>;
  using our_cache_t = typename PrevEntityExpr::our_cache_t;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;
  using ConnectedEntities = stk::mesh::NgpMesh::ConnectedEntities;

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(PrevEntityExpr prev_entity_expr, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(EntityExprBase<PrevEntityExpr> prev_entity_expr_base, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr_base.self()), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh ngp_mesh() const {
    return prev_entity_expr_.ngp_mesh();
  }

  template<size_t Ord = 0>
  stk::mesh::Selector selector() const {
    return prev_entity_expr_.template selector<Ord>();
  }

  template<size_t Ord = 0>
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    // TODO(palmerb4): Need a better naming convention. This is the rank needed to perform
    // the evaluation, not the connectivity rank. The current name
    return prev_entity_expr_.template rank<Ord>();  
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis,
                                                CacheType &cache, const NgpContext &context) const {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index =
        prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                                const NgpContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index =
        prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // We don't add any cache entries, so just propagate the flags from our previous expression
    return PrevEntityExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext &context) {
    prev_entity_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  stk::mesh::EntityRank conn_rank_;
};

template <size_t NumEntities, size_t Ord>
class EntityExpr : public EntityExprBase<EntityExpr<NumEntities, Ord>> {
 public:
  using our_t = EntityExpr<NumEntities, Ord>;
  using our_cache_t = core::tuple<>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  EntityExpr(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                  const stk::mesh::EntityRank &rank)
      : ngp_mesh_(ngp_mesh), selector_(selector), rank_(rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

  template<size_t Ord = 0>
  stk::mesh::Selector selector() const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return selector_;
  }

  template<size_t Ord = 0>
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return rank_;
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> fmis,
                                                       CacheType &cache, const NgpContext & /*context*/) const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return fmis[ordinal_];
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex fmi, CacheType &cache,
                                                       const NgpContext & /*context*/) const
    requires(num_entities == 1)
  {
    static_assert(Ord == 0, "EntityExpr with a single entity must have Ord == 0");
    return fmi;
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Nothing changes. We neither add anything to the cache nor have any sub-expressions.
    return IsCached;
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
    // Leaf node, nothing to do here.
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
  static constexpr size_t ordinal_ = Ord;
};

auto make_entity_expr(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                      const stk::mesh::EntityRank &rank) {
  return EntityExpr<1, 0>(ngp_mesh, selector, rank);
}

template <size_t NumEntities, size_t Ord>
auto make_multi_entity_expr(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                            const stk::mesh::EntityRank &rank) {
  static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
  return EntityExpr<NumEntities, Ord>(ngp_mesh, selector, rank);
}

template <size_t NumEntities>
class RTimeEntityExpr : public EntityExprBase<RTimeEntityExpr<NumEntities>> {
 public:
  using our_t = RTimeEntityExpr<NumEntities>;
  using our_cache_t = core::tuple<>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  RTimeEntityExpr(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                  const stk::mesh::EntityRank &rank, size_t ordinal)
      : ngp_mesh_(ngp_mesh), selector_(selector), rank_(rank), ordinal_(ordinal) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

  template<size_t Ord = 0>
  stk::mesh::Selector selector() const {
    static_assert(Ord < NumEntities, "RTimeEntityExpr ordinal must be less than NumEntities");
    return selector_;
  }

  template<size_t Ord = 0>
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    static_assert(Ord < NumEntities, "RTimeEntityExpr ordinal must be less than NumEntities");
    return rank_;
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(
      const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
      const NgpContext & /*context*/) const {
    MUNDY_THROW_ASSERT(ordinal_ < NumEntities, std::out_of_range,
                       "RTimeEntityExpr ordinal must be less than NumEntities");
    return fmis[ordinal_];
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                                       const NgpContext & /*context*/) const
    requires(num_entities == 1)
  {
    MUNDY_THROW_ASSERT(ordinal_ == 0, std::out_of_range, "RTimeEntityExpr with a single entity must have Ord == 0");
    return fmi;
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Nothing changes. We neither add anything to the cache nor have any sub-expressions.
    return IsCached;
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
    // Leaf node, nothing to do here.
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of an entity expression, which returns a temporary value."
        << std::endl;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
  size_t ordinal_;
};
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

template <class Derived>
class MathExprBase;

template <typename LeftMathExpr, typename RightMathExpr>
class AddExpr : public MathExprBase<AddExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = AddExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  AddExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  AddExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) +
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) +
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class SubExpr : public MathExprBase<SubExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = SubExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  SubExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  SubExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) -
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) -
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class MulExpr : public MathExprBase<MulExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = MulExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  MulExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  MulExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) *
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) *
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class DivExpr : public MathExprBase<DivExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = DivExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  DivExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  DivExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) /
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    return left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) /
           right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename TargetExpr, typename SourceExpr>
class AssignExpr : public MathExprBase<AssignExpr<TargetExpr, SourceExpr>> {
 public:
  using our_t = AssignExpr<TargetExpr, SourceExpr>;
  using our_cache_t = core::tuple_cat_t<typename TargetExpr::our_cache_t, typename SourceExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = TargetExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AssignExpr(TargetExpr trg_expr, SourceExpr src_expr) : trg_expr_(trg_expr), src_expr_(src_expr) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    trg_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) =
        src_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    trg_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) =
        src_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = TargetExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        SourceExpr::template update_is_cached<CacheOffset + TargetExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    src_expr_.flag_read_only(context);
    trg_expr_.flag_overwrite_all(context);
    trg_expr_.propagate_synchronize(context);
    src_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

 private:
  TargetExpr trg_expr_;
  SourceExpr src_expr_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class AddEqualsExpr : public MathExprBase<AddEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = AddEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  AddEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  AddEqualsExpr(EntityExprBase<LeftMathExpr> left, EntityExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) +=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) +=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    std::cout
        << "Warning: Attempting to write to the return type of a binary expression, which returns a temporary value."
        << std::endl;
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class SubEqualsExpr : public MathExprBase<SubEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = SubEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  SubEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  SubEqualsExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) -=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) -=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class MulEqualsExpr : public MathExprBase<MulEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = MulEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  MulEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  MulEqualsExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) *=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) *=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename LeftMathExpr, typename RightMathExpr>
class DivEqualsExpr : public MathExprBase<DivEqualsExpr<LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = DivEqualsExpr<LeftMathExpr, RightMathExpr>;
  using our_cache_t = core::tuple_cat_t<typename LeftMathExpr::our_cache_t, typename RightMathExpr::our_cache_t>;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = LeftMathExpr::NumEntities;

  KOKKOS_INLINE_FUNCTION
  DivEqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  DivEqualsExpr(MathExprBase<LeftMathExpr> left, MathExprBase<RightMathExpr> right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context) /=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION void eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    left_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context) /=
        right_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    constexpr auto updated_flags_left = LeftMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    constexpr auto updated_flags =
        RightMathExpr::template update_is_cached<CacheOffset + LeftMathExpr::num_cached_types, updated_flags_left>();
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    MUNDY_THROW_ASSERT(false, std::logic_error,
                       "Attempting to write to the return type of an assignment expression, which returns void.");
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

template <typename AccessorT, typename PrevEntityExpr>
class AccessorExpr : public MathExprBase<AccessorExpr<AccessorT, PrevEntityExpr>> {
 public:
  using our_t = AccessorExpr<AccessorT, PrevEntityExpr>;
  using our_cache_t = typename PrevEntityExpr::our_cache_t;
  static constexpr size_t num_cached_types = our_cache_t::size();
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(AccessorT accessor, const PrevEntityExpr &prev_entity_expr)
      : accessor_(accessor), prev_entity_expr_(prev_entity_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(AccessorT accessor, const EntityExprBase<PrevEntityExpr> &prev_entity_expr_base)
      : accessor_(accessor), prev_entity_expr_(prev_entity_expr_base.self()) {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    stk::mesh::FastMeshIndex entity_index =
        prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
    return accessor_(entity_index);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::FastMeshIndex entity_index =
        prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
    return accessor_(entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // We don't add any cache entries, so just propagate the flags from our previous expression
    return PrevEntityExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
  }

  void flag_read_only(const NgpContext & /*context*/) {
    accessor_.sync_to_device();
  }

  void flag_read_write(const NgpContext & /*context*/) {
    accessor_.sync_to_device();
    accessor_.modify_on_device();
  }

  void flag_overwrite_all(const NgpContext & /*context*/) {
    accessor_.clear_host_sync_state();
    accessor_.modify_on_device();
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
  }

 private:
  AccessorT accessor_;
  PrevEntityExpr prev_entity_expr_;
};

template <typename DerivedMathExpr>
class MathExprBase {
 public:
  using our_t = MathExprBase<DerivedMathExpr>;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr MathExprBase() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedMathExpr &self() const noexcept {
    return static_cast<const DerivedMathExpr &>(*this);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached,  //
            size_t NumEntities, typename CacheType, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities> &fmis, CacheType &cache,
                                   const Ctx &context) const {
    return self().template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached,  //
            typename CacheType, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache, const Ctx &context) const {
    return self().template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
  }

  KOKKOS_INLINE_FUNCTION
  static auto init_cache() {
    return DerivedMathExpr::init_cache();
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return DerivedMathExpr::init_is_cached();
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    return DerivedMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
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
    for_each_entity_evaluate_expr(expr);
  }

  template <typename OtherExpr>
  auto operator=(const EntityExprBase<OtherExpr> &other) {
    auto expr = AssignExpr<DerivedMathExpr, OtherExpr>(*this, other);
    for_each_entity_evaluate_expr(expr);
  }

  template <typename OtherExpr>
  void operator+=(const MathExprBase<OtherExpr> &other) {
    auto expr = AddExpr<DerivedMathExpr, OtherExpr>(*this, other);
    for_each_entity_evaluate_expr(expr);
  }

  template <typename OtherExpr>
  void operator-=(const MathExprBase<OtherExpr> &other) {
    auto expr = SubExpr<DerivedMathExpr, OtherExpr>(*this, other);
    for_each_entity_evaluate_expr(expr);
  }

  template <typename OtherExpr>
  void operator*=(const MathExprBase<OtherExpr> &other) {
    auto expr = MulExpr<DerivedMathExpr, OtherExpr>(*this, other);
    for_each_entity_evaluate_expr(expr);
  }

  template <typename OtherExpr>
  void operator/=(const MathExprBase<OtherExpr> &other) {
    auto expr = DivExpr<DerivedMathExpr, OtherExpr>(*this, other);
    for_each_entity_evaluate_expr(expr);
  }
};
//@}

//! \name Helpers
//@{

template <typename PrevEntityExpr>
class ReuseEntityExpr : public EntityExprBase<ReuseEntityExpr<PrevEntityExpr>> {
 public:
  using our_t = ReuseEntityExpr<PrevEntityExpr>;
  static constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  ReuseEntityExpr(PrevEntityExpr prev_entity_expr)
      : prev_entity_expr_(prev_entity_expr), evaluated_(false), stored_value_() {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(
      const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
      const NgpContext &context) const {
    constexpr size_t our_cache_offset = CacheOffset + PrevEntityExpr::num_cached_types;
    if constexpr (IsCached[our_cache_offset]) {
      return get<our_cache_offset>(cache);
    } else {
      auto val = prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
      get<CacheOffset>(cache) = val;
      return val;
    }
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                                       const NgpContext &context) const
    requires(num_entities == 1)
  {
    constexpr size_t our_cache_offset = CacheOffset + PrevEntityExpr::num_cached_types;
    if constexpr (IsCached[our_cache_offset]) {
      return get<our_cache_offset>(cache);
    } else {
      auto val = prev_entity_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
      get<CacheOffset>(cache) = val;
      return val;
    }
  }

  using stored_value_t = stk::mesh::FastMeshIndex;
  using our_cache_t = core::tuple_cat_t<typename PrevEntityExpr::our_cache_t, core::tuple<stored_value_t>>;
  static constexpr size_t num_cached_types = our_cache_t::size();

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Our cached object comes after that of our sub-expression
    auto updated_flags = PrevEntityExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    updated_flags[CacheOffset + PrevEntityExpr::num_cached_types] = true;
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    prev_entity_expr_.propagate_synchronize(context);
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  bool evaluated_;
  mutable stored_value_t stored_value_;
};

template <typename PrevMathExpr>
class ReuseMathExpr : public MathExprBase<ReuseMathExpr<PrevMathExpr>> {
 public:
  using our_t = ReuseMathExpr<PrevMathExpr>;
  static constexpr size_t num_entities = PrevMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  ReuseMathExpr(PrevMathExpr prev_math_expr) : prev_math_expr_(prev_math_expr), evaluated_(false), stored_value_() {
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmis, CacheType &cache,
                                   const NgpContext &context) const {
    // The cache offset is for us and all of our sub-expressions. Our cached object is the last one in that range.
    // Basically read from right to left in the cache when traversing via eval.
    //   [  unrelated expr cache | our sub-expr cache  |  our cache  | unrelated expr cache ]
    //                           ^                     ^
    //                      CacheOffset     CacheOffset + PrevMathExpr::num_cached_types
    constexpr size_t our_cache_offset = CacheOffset + PrevMathExpr::num_cached_types;
    if constexpr (IsCached[our_cache_offset]) {
      return get<our_cache_offset>(cache);
    } else {
      auto val = prev_math_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmis, cache, context);
      get<CacheOffset>(cache) = val;
      return val;
    }
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached, typename CacheType>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fmi, CacheType &cache,
                                   const NgpContext &context) const
    requires(num_entities == 1)
  {
    constexpr size_t our_cache_offset = CacheOffset + PrevMathExpr::num_cached_types;
    if constexpr (IsCached[our_cache_offset]) {
      return get<our_cache_offset>(cache);
    } else {
      auto val = prev_math_expr_.template eval<CacheOffset, CacheSize, IsCached>(fmi, cache, context);
      get<CacheOffset>(cache) = val;
      return val;
    }
  }

  using stored_value_t = decltype(std::declval<PrevMathExpr>().eval(
      std::declval<Kokkos::Array<stk::mesh::FastMeshIndex, PrevMathExpr::num_entities>>(), std::declval<NgpContext>()));
  using our_cache_t = core::tuple_cat_t<typename PrevMathExpr::our_cache_t, core::tuple<stored_value_t>>;
  static constexpr size_t num_cached_types = our_cache_t::size();

  KOKKOS_INLINE_FUNCTION
  static our_cache_t init_cache() {
    return our_cache_t{};
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr auto init_is_cached() {
    return math::Vector<bool, our_cache_t::size()>{false};
  }

  template <size_t CacheOffset, size_t CacheSize, math::Vector<bool, CacheSize> IsCached>
  KOKKOS_INLINE_FUNCTION static constexpr auto update_is_cached() {
    // Propagate the flags from our left and right expressions
    auto updated_flags = PrevMathExpr::template update_is_cached<CacheOffset, CacheSize, IsCached>();
    updated_flags[CacheOffset + PrevMathExpr::num_cached_types] = true;
    return updated_flags;
  }

  void propagate_synchronize(const NgpContext &context) {
    prev_math_expr_.propagate_synchronize(context);
  }

 private:
  PrevMathExpr prev_math_expr_;
  bool evaluated_;
  mutable stored_value_t stored_value_;
};

/// \brief Reuse the return value of an expression in multiple places in a single or fused evaluation.
template <typename PrevMathExpr>
ReuseMathExpr<PrevMathExpr> reuse(MathExprBase<PrevMathExpr> expr) {
  return ReuseMathExpr<PrevMathExpr>(expr);
}

/// \brief Reuse the entity returned by an expression in multiple places in a single or fused evaluation.
template <typename PrevEntityExpr>
ReuseEntityExpr<PrevEntityExpr> reuse(EntityExprBase<PrevEntityExpr> expr) {
  return ReuseEntityExpr<PrevEntityExpr>(expr);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPACCESSOREXPR_HPP_
