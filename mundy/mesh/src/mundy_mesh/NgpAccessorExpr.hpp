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
/// \defgroup MundyMeshNgpAccessorExpr mundy::mesh::NgpAccessorExpr
/// \brief Expression-template layer for fusing STK NGP field reads, writes, and reductions.

// Kokkos
#include <Kokkos_Core.hpp>  // for KOKKOS_LAMBDA, etc.

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
#include <mundy_math/Matrix.hpp>         // for mundy::Matrix
#include <mundy_math/Quaternion.hpp>     // for mundy::Quaternion
#include <mundy_math/ScalarWrapper.hpp>  // for mundy::ScalarWrapper
#include <mundy_math/Vector.hpp>         // for mundy::Vector
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/impl/NgpAccessorExprImpl.hpp>
#include <mundy_utils/StringLiteral.hpp>  // for mundy::StringLiteral
#include <mundy_utils/aggregate.hpp>      // for mundy::aggregate
#include <mundy_utils/requires.hpp>
#include <mundy_utils/rng.hpp>           // for mundy::make_philox
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
#include <mundy_utils/tuple.hpp>         // for mundy::tuple

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

Type-based caching is only sound when a node's type fully determines its eval result within a tree. Each expression
therefore exposes `static constexpr bool has_static_eval` to indicate whether two nodes with the same tag are
guaranteed to evaluate to the same result for the same inputs. Automatic memoization is enabled only for such nodes.

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

/// \brief is_crtp_base_of<B, E>
///
/// Resembles std::is_base_of, but addresses the problem of whether _some_ instantiation
/// of a CRTP templated class B is a base of class E. A CRTP templated class is correctly
/// templated with the most derived type in the CRTP hierarchy. Using this assumption,
/// this implementation deals with either CRTP final classes (checks for inheritance
/// with E as the CRTP parameter of B) or CRTP base classes (which are singly templated
/// by the most derived class, and that's pulled out to use as a template parameter for B).
template <template <class> class B, class E>
using is_crtp_base_of = impl::is_crtp_base_of_impl<B, std::decay_t<E>>;

template <template <class> class B, class E>
static constexpr bool is_crtp_base_of_v = is_crtp_base_of<B, E>::value;

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

  // Derived expressions must expose:
  //   static constexpr bool has_static_eval;
  //
  // When true, the expression may participate in type-based memoization. When false, equal tags are not assumed to
  // imply equal eval results, so cached_eval must recompute the value each time.
  //
  // The rigorous rule is:
  //   - true only for pure expressions whose eval result is fully determined by the tag and their eval inputs
  //   - false for expressions with side effects
  //   - false for expressions whose eval depends on runtime state not represented by the tag
  //
  // Simply because an object has non-static members does not necessarily mean it must have has_static_eval=false. For
  // example, an AccessorExpr has_static_eval=true since the fields are themselves tagged (basically a contract stating
  // that each tag corresponds to a unique field).
  static constexpr bool has_static_eval = DerivedExpr::has_static_eval;

  // Optional opt-in for runtime reuse when has_static_eval == false.
  //
  // Runtime reuse is still cache-by-tag, but unlike static caching it requires host-side structural validation. If an
  // expression enables runtime reuse, and the same tag appears multiple times in a tree, each instance must be
  // structurally equivalent at runtime_reuse_equivalent(). Expressions opting in should therefore override
  // runtime_reuse_equivalent(const Self&) for their structural contract.
  static constexpr bool supports_runtime_reuse = false;

 private:
  template <typename Tag, typename AggregateType, AggregateType agg>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_tag_count() {
    if constexpr (aggregate_has_v<Tag, AggregateType>) {
      auto new_agg = agg;
      get<Tag>(new_agg) += 1;
      return new_agg;
    } else {
      return append<Tag>(agg, 1);
    }
  }

  template <typename SubExprTuple, size_t I, typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_eval_counts_recurse() {
    if constexpr (I < SubExprTuple::size()) {
      using sub_expr_t = tuple_element_t<I, SubExprTuple>;
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
  constexpr const DerivedExpr& self() const noexcept {
    return static_cast<const DerivedExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedExpr& self() noexcept {
    return static_cast<DerivedExpr&>(*this);
  }

  // Default structural-equivalence rule:
  // - static nodes are equivalent by tag contract
  // - non-static nodes are not equivalent unless a derived class overrides this method
  KOKKOS_INLINE_FUNCTION
  constexpr bool runtime_reuse_equivalent([[maybe_unused]] const DerivedExpr& other) const noexcept {
    return has_static_eval;
  }

  /// \brief Evaluate the expression
  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const Ctx& context) const {
    return self().eval(fmis, context);
  }

  /// \brief Evaluate the expression
  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
  }

  /// \brief Update eval_counts by incrementing the counts for our tag and our sub-expressions tags
  template <typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_eval_counts() {
    constexpr auto new_eval_counts = increment_tag_count<our_tag, OldEvalCountsType, old_eval_counts>();
    using sub_exprs = typename DerivedExpr::sub_expressions_t;
    return increment_eval_counts_recurse<sub_exprs, 0, decltype(new_eval_counts), new_eval_counts>();
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse([[maybe_unused]] impl::RuntimeReuseValidator& validator) const {
    // Leaf/default behavior: recurse only where derived classes explicitly expose their sub-expression ownership.
  }

  //! \name Field synchronization and modification flagging
  //@{

  template <class Ctx>
  void propagate_synchronize(const Ctx& context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx& context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx& context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx& context) {
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
  constexpr const DerivedEntityExpr& self() const noexcept {
    return static_cast<const DerivedEntityExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedEntityExpr& self() noexcept {
    return static_cast<DerivedEntityExpr&>(*this);
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
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                                       const Ctx& context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
  }

  //! \name Field synchronization and modification flagging
  //@{

  template <class Ctx>
  void propagate_synchronize(const Ctx& context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx& context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx& context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx& context) {
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
  using sub_expressions_t = tuple<PrevEntityExpr>;
  using ConnectedEntities = stk::mesh::NgpMesh::ConnectedEntities;
  static constexpr bool constrains_num_entities = false;
  // The connectivity rank is runtime state, so two nodes with the same type need not evaluate to the same result.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(PrevEntityExpr prev_entity_expr, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(const EntityExprBase<PrevEntityExpr>& prev_entity_expr_base, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr_base.self()), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return conn_rank_;
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                                const NgpEvalContext& context) const {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Eval our subexpressions first
        auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto entity_value = std::move(entity_result.first);

        // Our eval result needs cached, but is not yet cached
        stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
        auto val =
            context.ngp_mesh().get_connected_entities(entity_rank, entity_value.get(entity_result.second), conn_rank_);
        auto newest_cache = append<our_tag>(std::move(entity_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
      auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto entity_value = std::move(entity_result.first);
      auto val =
          context.ngp_mesh().get_connected_entities(entity_rank, entity_value.get(entity_result.second), conn_rank_);
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(entity_result.second));
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    prev_entity_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return conn_rank_ == other.conn_rank_ && prev_entity_expr_.runtime_reuse_equivalent(other.prev_entity_expr_);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    prev_entity_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
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
  using sub_expressions_t = tuple<>;
  static constexpr size_t num_entities = NumEntities;
  // Our eval result is just fmis[Ord]. The stored rank/driver affect metadata and execution, but not the returned FMI.
  static constexpr bool has_static_eval = true;

  KOKKOS_INLINE_FUNCTION
  EntityExpr(const stk::mesh::EntityRank& rank, const DriverType* driver) : rank_(rank), driver_(driver) {
  }

  /// \brief The rank of the entity we return
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> fmis,
                                                       const NgpEvalContext& /*context*/) const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return fmis[Ord];
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& /*context*/) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Our eval result needs cached, but is not yet cached
        auto val = fmis[Ord];
        auto new_cache = append<our_tag>(std::forward<OldCacheType>(old_cache), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(new_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto val = fmis[Ord];
      return Kokkos::make_pair(impl::OwnedCachedValue{val}, std::forward<OldCacheType>(old_cache));
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpEvalContext& /*context*/) {
    // Leaf node, nothing to do here.
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
  }

  const DriverType* driver() const {
    return driver_;
  }

 private:
  stk::mesh::EntityRank rank_;
  const DriverType* driver_;
};

// The goal of this class is to allow for the creation of EntityExpr from an array of entities.
// This class is not, itself, an EntityExpr, but allows for the creation of one.
template <size_t NumEntities, typename DriverType>
class IntermediaryEntityArray {
 public:
  static constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  IntermediaryEntityArray(const Kokkos::Array<stk::mesh::EntityRank, NumEntities>& ranks, const DriverType* driver)
      : ranks_(ranks), driver_(driver) {
  }

  template <size_t Ord>
  KOKKOS_INLINE_FUNCTION EntityExpr<NumEntities, Ord, DriverType> get() const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return EntityExpr<NumEntities, Ord, DriverType>(ranks_[Ord], driver_);
  }

  const DriverType* driver() const {
    return driver_;
  }

 private:
  Kokkos::Array<stk::mesh::EntityRank, NumEntities> ranks_;
  const DriverType* driver_;
};

template <typename DriverType>
class EntityPair {
 public:
  static constexpr size_t num_entities = 2;

  KOKKOS_INLINE_FUNCTION
  EntityPair(const stk::mesh::EntityRank& first_rank, const stk::mesh::EntityRank& second_rank,
             const DriverType* driver)
      : first_rank_(first_rank), second_rank_(second_rank), driver_(driver) {
  }

  template <size_t Ord>
  KOKKOS_INLINE_FUNCTION EntityExpr<2, Ord, DriverType> get() const {
    static_assert(Ord < 2, "EntityExpr ordinal must be less than 2");
    if constexpr (Ord == 0) {
      return EntityExpr<2, Ord, DriverType>(first_rank_, driver_);
    } else {
      return EntityExpr<2, Ord, DriverType>(second_rank_, driver_);
    }
  }

  KOKKOS_INLINE_FUNCTION
  EntityExpr<2, 0, DriverType> first() const {
    return EntityExpr<2, 0, DriverType>(first_rank_, driver_);
  }

  KOKKOS_INLINE_FUNCTION
  EntityExpr<2, 1, DriverType> second() const {
    return EntityExpr<2, 1, DriverType>(second_rank_, driver_);
  }

  const DriverType* driver() const {
    return driver_;
  }

 private:
  stk::mesh::EntityRank first_rank_;
  stk::mesh::EntityRank second_rank_;
  const DriverType* driver_;
};

template <typename ExecSpace = stk::ngp::ExecSpace>
class NgpForEachEntityExprDriver {
 public:
  NgpForEachEntityExprDriver(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector,
                             stk::mesh::EntityRank rank, const ExecSpace& exec_space = ExecSpace())
      : bulk_data_ptr_(&bulk_data), selector_(selector), rank_(rank), exec_space_(exec_space) {
  }

  // Default copy/move constructor and assignment operator are fine
  NgpForEachEntityExprDriver(const NgpForEachEntityExprDriver&) = default;
  NgpForEachEntityExprDriver(NgpForEachEntityExprDriver&&) = default;
  NgpForEachEntityExprDriver& operator=(const NgpForEachEntityExprDriver&) = default;
  NgpForEachEntityExprDriver& operator=(NgpForEachEntityExprDriver&&) = default;
  virtual ~NgpForEachEntityExprDriver() = default;

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::logic_error,
                        "NgpForEachEntityExprDriver has a null BulkData pointer");
    return *bulk_data_ptr_;
  }

  stk::mesh::Selector selector() const {
    return selector_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  ExecSpace exec_space() const {
    return exec_space_;
  }

  template <typename Expr>
  void run(CachableExprBase<Expr>& expr_base) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    ::mundy::mesh::for_each_entity_run(
        ngp_mesh, rank_, selector_, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
          // Non-cached eval for debugging
          // expr.eval(Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, evaluation_context);

          // Perform the eval
          auto empty_cache = aggregate();
          expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, empty_cache, evaluation_context);
        });
  }

  template <typename Expr, typename ReductionOp>
  void reduce_local(CachableExprBase<Expr>& expr_base, ReductionOp& reduction) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    using value_type = typename ReductionOp::value_type;
    stk::mesh::for_each_entity_reduce(
        ngp_mesh, rank_, selector_, reduction,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index, value_type& value) {
          // Perform the eval
          auto empty_cache = aggregate();
          auto result = expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, empty_cache, evaluation_context);

          // Combine into the reduction
          // To avoid CUDA being CUDA, we must "touch" the reduction
          [[maybe_unused]] auto meaningless_return_to_make_cuda_happy = reduction.reference();
          auto&& val_ref = result.first.get(result.second);
          using val_t = std::remove_cvref_t<decltype(val_ref)>;

          if constexpr (std::is_same_v<val_t, value_type>) {
            // Directly compatible types; just combine
            reduction.join(value, val_ref);
          } else if constexpr (is_scalar_wrapper_v<val_t>) {
            // val is a scalar wrapper; extract the underlying value and combine
            reduction.join(value, val_ref[0]);
          } else {
            // Unknown return type, attempt to use it directly
            reduction.join(value, val_ref);
          }
        });
  }

 private:
  const stk::mesh::BulkData* bulk_data_ptr_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
  ExecSpace exec_space_;
};

template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
class NgpForEachEntityPairExprDriver {
 public:
  NgpForEachEntityPairExprDriver(const stk::mesh::BulkData& bulk_data, const PairView& pair_view,
                                 const ExecSpace& exec_space = ExecSpace())
      : bulk_data_ptr_(&bulk_data), pair_view_(pair_view), exec_space_(exec_space) {
  }

  // Default copy/move constructor and assignment operator are fine
  NgpForEachEntityPairExprDriver(const NgpForEachEntityPairExprDriver&) = default;
  NgpForEachEntityPairExprDriver(NgpForEachEntityPairExprDriver&&) = default;
  NgpForEachEntityPairExprDriver& operator=(const NgpForEachEntityPairExprDriver&) = default;
  NgpForEachEntityPairExprDriver& operator=(NgpForEachEntityPairExprDriver&&) = default;
  virtual ~NgpForEachEntityPairExprDriver() = default;

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::logic_error,
                        "NgpForEachEntityPairExprDriver has a null BulkData pointer");
    return *bulk_data_ptr_;
  }

  ExecSpace exec_space() const {
    return exec_space_;
  }

  template <typename Expr>
  void run(CachableExprBase<Expr>& expr_base) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    auto pair_view = pair_view_;  // Make a local copy for lambda capture
    Kokkos::parallel_for(
        "NgpForEachEntityPairExprDriver::run", Kokkos::RangePolicy<ExecSpace>(exec_space(), 0, pair_view.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          auto entity_pair = pair_view(i);
          stk::mesh::FastMeshIndex left_fmi = FMIExtractor::get_left_index(entity_pair);
          stk::mesh::FastMeshIndex right_fmi = FMIExtractor::get_right_index(entity_pair);

          // Non-cached eval
          // expr.eval(Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, evaluation_context);

          // Perform the eval
          auto empty_cache = aggregate();
          expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, empty_cache, evaluation_context);
        });
  }

  template <typename Expr, typename ReductionOp>
  void reduce_local(CachableExprBase<Expr>& expr_base, ReductionOp& reduction) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    auto pair_view = pair_view_;  // Make a local copy for lambda capture
    using value_type = typename ReductionOp::value_type;
    Kokkos::parallel_reduce(
        "NgpForEachEntityPairExprDriver::reduce_local",
        Kokkos::RangePolicy<ExecSpace>(exec_space(), 0, pair_view.extent(0)),
        KOKKOS_LAMBDA(const int i, value_type& value) {
          auto entity_pair = pair_view(i);
          stk::mesh::FastMeshIndex left_fmi = FMIExtractor::get_left_index(entity_pair);
          stk::mesh::FastMeshIndex right_fmi = FMIExtractor::get_right_index(entity_pair);

          // Perform the eval
          auto empty_cache = aggregate();
          auto result = expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, empty_cache, evaluation_context);

          // Combine into the reduction
          // To avoid CUDA being CUDA, we must "touch" the reduction
          [[maybe_unused]] auto meaningless_return_to_make_cuda_happy = reduction.reference();
          auto&& val_ref = result.first.get(result.second);
          using val_t = std::remove_cvref_t<decltype(val_ref)>;

          if constexpr (std::is_same_v<val_t, value_type>) {
            // Directly compatible types; just combine
            reduction.join(value, val_ref);
          } else if constexpr (is_scalar_wrapper_v<val_t>) {
            // val is a scalar wrapper; extract the underlying value and combine
            reduction.join(value, val_ref[0]);
          } else {
            // Unknown return type, attempt to use it directly
            reduction.join(value, val_ref);
          }
        });
  }

 private:
  const stk::mesh::BulkData* bulk_data_ptr_;
  PairView pair_view_;
  ExecSpace exec_space_;
};

template <typename ExecSpace = stk::ngp::ExecSpace>
auto make_entity_expr(stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                      const stk::mesh::EntityRank& rank, const ExecSpace& /*exec_space*/ = ExecSpace()) {
  // To ensure that all expressions have the same driver, we store a persistent driver manager
  // on the meta data and use it to memoize the driver for the given rank and selector.

  using driver_t = NgpForEachEntityExprDriver<ExecSpace>;
  using driver_map_t = impl::AnyRankSelectorMap<make_string_literal("NgpExprDrivers")>;
  stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
  driver_map_t* driver_map = const_cast<driver_map_t*>(meta_data.get_attribute<driver_map_t>());
  if (driver_map == nullptr) {
    const driver_map_t* new_driver_map = new driver_map_t();
    driver_map = const_cast<driver_map_t*>(meta_data.declare_attribute_with_delete(new_driver_map));
  }

  // Stash our driver in the map if it doesn't already exist
  const driver_t* driver_ptr;
  if (driver_map->contains(rank, selector)) {
    // Driver already exists for this rank and selector; reuse it
    driver_t& existing_driver = driver_map->at<driver_t>(rank, selector);
    driver_ptr = &existing_driver;
  } else {
    // Driver doesn't exist yet; create and insert it
    driver_t new_driver(bulk_data, selector, rank);
    driver_map->insert<driver_t>(rank, selector, std::move(new_driver));
    const driver_t& inserted_driver = driver_map->at<driver_t>(rank, selector);
    driver_ptr = &inserted_driver;
  }

  return EntityExpr<1, 0, driver_t>(rank, driver_ptr);
}

template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
auto make_pairwise_entity_expr(stk::mesh::BulkData& bulk_data,                                    //
                               const stk::mesh::EntityRank& left_rank,                            //
                               const stk::mesh::EntityRank& right_rank,                           //
                               const PairView& pair_view, const FMIExtractor& /*fmi_extractor*/,  //
                               const ExecSpace& /*exec_space*/ = ExecSpace()) {
  using driver_t = NgpForEachEntityPairExprDriver<PairView, FMIExtractor, ExecSpace>;
  using driver_map_t = impl::AnyRankSelectorMap<make_string_literal("NgpPairExprDrivers")>;
  stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
  driver_map_t* driver_map = const_cast<driver_map_t*>(meta_data.get_attribute<driver_map_t>());
  if (driver_map == nullptr) {
    const driver_map_t* new_driver_map = new driver_map_t();
    driver_map = const_cast<driver_map_t*>(meta_data.declare_attribute_with_delete(new_driver_map));
  }

  // Stash our driver in the map if it doesn't already exist
  const driver_t* driver_ptr;
  stk::mesh::EntityRank dummy_rank = stk::topology::NODE_RANK;  // Rank is irrelevant for pairwise drivers
  stk::mesh::Selector dummy_selector = stk::mesh::Selector();   // Selector is irrelevant for pairwise drivers
  if (driver_map->contains(dummy_rank, dummy_selector)) {
    // Driver already exists; reuse it
    driver_t& existing_driver = driver_map->at<driver_t>(dummy_rank, dummy_selector);
    driver_ptr = &existing_driver;
  } else {
    // Driver doesn't exist yet; create and insert it
    driver_t new_driver(bulk_data, pair_view);
    driver_map->insert<driver_t>(dummy_rank, dummy_selector, std::move(new_driver));
    const driver_t& inserted_driver = driver_map->at<driver_t>(dummy_rank, dummy_selector);
    driver_ptr = &inserted_driver;
  }

  return EntityPair(left_rank, right_rank, driver_ptr);
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

template <typename ConstantType>
class ConstantMathExpr : public MathExprBase<ConstantMathExpr<ConstantType>> {
 public:
  using our_t = ConstantMathExpr<ConstantType>;
  using our_tag = typename MathExprBase<ConstantMathExpr<ConstantType>>::our_tag;
  using sub_expressions_t = tuple<>;
  static constexpr bool constrains_num_entities = false;
  // The constant value is runtime state that affects the evaluation, so equal tags do not imply equal eval results.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  ConstantMathExpr(ConstantType value) : value_(value) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION ConstantType eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& /*fmis*/,
                                           const NgpEvalContext& /*context*/) const {
    return value_;
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& /*fmis*/,
                                          OldCacheType&& old_cache, const NgpEvalContext& /*context*/) const {
    static_assert(
        !aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
        "The cache somehow contains our tag, but our eval returns a constant and should never cache anything.");
    return Kokkos::make_pair(impl::OwnedCachedValue{value_}, std::forward<OldCacheType>(old_cache));
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse([[maybe_unused]] impl::RuntimeReuseValidator& validator) const {
    // Constants are leaves. Nothing to do here.
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return value_ == other.value_;
  }

  void propagate_synchronize(const NgpEvalContext& /*context*/) {
    // Nothing to do here
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Nothing to do here
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error, "Attempting to write to a constant expression.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error, "Attempting to write to a constant expression.");
  }

  auto driver() const {
    return nullptr;
  }

 private:
  ConstantType value_;
};

template <typename TargetExpr, typename SourceExpr>
class AssignExpr : public MathExprBase<AssignExpr<TargetExpr, SourceExpr>> {
 public:
  using our_t = AssignExpr<TargetExpr, SourceExpr>;
  using our_tag = typename MathExprBase<AssignExpr<TargetExpr, SourceExpr>>::our_tag;
  using sub_expressions_t = tuple<TargetExpr, SourceExpr>;
  static constexpr bool constrains_num_entities = false;
  // Assignment, in its current form, is side-effecting and its return type is void, so it always has non-static eval.
  // In the future, we may wish to switch to a non-void return that changes from assignment as a sink to assignment as
  // an expression.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  AssignExpr(TargetExpr trg_expr, SourceExpr src_expr) : trg_expr_(trg_expr), src_expr_(src_expr) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    trg_expr_.eval(fmis, context) = src_expr_.eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(!aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");

    // Eval our subexpressions first, allowing them to cache their results if necessary
    auto trg_result = trg_expr_.template cached_eval<EvalCountsType, eval_counts>(
        fmis, std::forward<OldCacheType>(old_cache), context);
    auto trg_value = std::move(trg_result.first);
    auto src_result =
        src_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(trg_result.second), context);
    auto&& trg_ref = trg_value.get(src_result.second);
    auto&& src_ref = src_result.first.get(src_result.second);
    trg_ref = src_ref;
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    trg_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    src_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    src_expr_.flag_read_only(context);
    trg_expr_.flag_overwrite_all(context);
    trg_expr_.propagate_synchronize(context);
    src_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to write to the return type of an assignment expression, which returns void.");
  }

  auto driver() const {
    using nullptr_t = decltype(nullptr);

    constexpr bool has_trg_driver = !std::is_same_v<nullptr_t, decltype(trg_expr_.driver())>;
    constexpr bool has_src_driver = !std::is_same_v<nullptr_t, decltype(src_expr_.driver())>;
    static_assert(
        has_trg_driver || has_src_driver,
        "At least one of the source or target expressions in an assignment expression must have a non-null driver.");

    if constexpr (has_trg_driver) {
      auto d = trg_expr_.driver();
      if constexpr (has_src_driver) {
        MUNDY_THROW_REQUIRE(d == src_expr_.driver(), std::logic_error, "Mismatched drivers in assignment expression");
      }
      return d;
    } else {
      return src_expr_.driver();
    }
  }

 private:
  TargetExpr trg_expr_;
  SourceExpr src_expr_;
};

template <typename Policy, typename LeftMathExpr, typename RightMathExpr>
class BinaryValueExpr : public MathExprBase<BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<LeftMathExpr, RightMathExpr>;
  static constexpr bool constrains_num_entities = false;
  static constexpr bool has_static_eval = LeftMathExpr::has_static_eval && RightMathExpr::has_static_eval;

  KOKKOS_INLINE_FUNCTION
  BinaryValueExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  BinaryValueExpr(const MathExprBase<LeftMathExpr>& left, const MathExprBase<RightMathExpr>& right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    return Policy::apply(left_.eval(fmis, context), right_.eval(fmis, context));
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        auto left_result = left_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto left_value = std::move(left_result.first);
        auto right_result =
            right_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(left_result.second), context);
        auto val = Policy::apply(left_value.get(right_result.second), right_result.first.get(right_result.second));
        auto newest_cache = append<our_tag>(std::move(right_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      auto left_result =
          left_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache), context);
      auto left_value = std::move(left_result.first);
      auto right_result =
          right_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(left_result.second), context);
      auto val = Policy::apply(left_value.get(right_result.second), right_result.first.get(right_result.second));
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(right_result.second));
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return left_.runtime_reuse_equivalent(other.left_) && right_.runtime_reuse_equivalent(other.right_);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    left_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    right_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    left_.flag_read_only(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of a binary expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of a binary expression, which returns a temporary value.");
  }

  const auto driver() const {
    using nullptr_t = decltype(nullptr);

    constexpr bool has_left_driver = !std::is_same_v<nullptr_t, decltype(left_.driver())>;
    constexpr bool has_right_driver = !std::is_same_v<nullptr_t, decltype(right_.driver())>;
    static_assert(
        has_left_driver || has_right_driver,
        "At least one of the left or right expressions in a binary math expression must have a non-null driver.");

    if constexpr (has_left_driver) {
      auto d = left_.driver();
      if constexpr (has_right_driver) {
        MUNDY_THROW_REQUIRE(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
      }
      return d;
    } else {
      return right_.driver();
    }
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

#define MUNDY_ACCESSOR_EXPR_OP(OpName, op)                                                     \
  struct OpName##OpPolicy {                                                                    \
    template <typename LeftValue, typename RightValue>                                         \
    KOKKOS_INLINE_FUNCTION static auto apply(const LeftValue& left, const RightValue& right) { \
      return left op right;                                                                    \
    }                                                                                          \
  };                                                                                           \
  template <typename LeftMathExpr, typename RightMathExpr>                                     \
  using OpName##Expr = BinaryValueExpr<OpName##OpPolicy, LeftMathExpr, RightMathExpr>;

namespace impl {

template <typename...>
static constexpr bool dependent_false_v = false;

template <typename T>
static constexpr bool is_math_expr_arg_v = is_crtp_base_of_v<MathExprBase, std::decay_t<T>>;

template <typename Expr>
KOKKOS_INLINE_FUNCTION auto make_apply_expr_arg(const MathExprBase<Expr>& expr) {
  return expr.self();
}

template <typename T>
MUNDY_REQUIRES(!impl::is_math_expr_arg_v<T> && !is_crtp_base_of_v<EntityExprBase, T>)
KOKKOS_INLINE_FUNCTION auto make_apply_expr_arg(const T& value) {
  return ConstantMathExpr<std::decay_t<T>>(value);
}

enum class SinkAccessMode { ReadOnly, ReadWrite, OverwriteAll };

template <SinkAccessMode Mode, typename Expr>
class SinkArg {
 public:
  using expr_t = Expr;
  static constexpr SinkAccessMode mode = Mode;

  KOKKOS_INLINE_FUNCTION
  explicit SinkArg(Expr expr) : expr_(expr) {
  }

  KOKKOS_INLINE_FUNCTION
  const Expr& expr() const {
    return expr_;
  }

  KOKKOS_INLINE_FUNCTION
  Expr& expr() {
    return expr_;
  }

 private:
  Expr expr_;
};

template <typename T>
struct is_sink_arg : std::false_type {};

template <SinkAccessMode Mode, typename Expr>
struct is_sink_arg<SinkArg<Mode, Expr>> : std::true_type {};

template <typename T>
static constexpr bool is_sink_arg_v = is_sink_arg<std::decay_t<T>>::value;

template <typename T>
struct is_constant_math_expr : std::false_type {};

template <typename T>
struct is_constant_math_expr<ConstantMathExpr<T>> : std::true_type {};

template <typename T>
static constexpr bool is_constant_math_expr_v = is_constant_math_expr<std::decay_t<T>>::value;

template <typename T>
struct sink_arg_has_nonconstant_expr
    : std::bool_constant<is_math_expr_arg_v<T> && !is_constant_math_expr_v<T>> {};

template <SinkAccessMode Mode, typename Expr>
struct sink_arg_has_nonconstant_expr<SinkArg<Mode, Expr>>
    : std::bool_constant<!is_constant_math_expr_v<Expr>> {};

template <typename T>
static constexpr bool sink_arg_has_nonconstant_expr_v = sink_arg_has_nonconstant_expr<std::decay_t<T>>::value;

template <SinkAccessMode Mode, typename T>
KOKKOS_INLINE_FUNCTION auto make_sink_arg_with_mode(const T& value) {
  static_assert(!is_sink_arg_v<T>, "sink_expr access wrappers cannot be nested.");
  if constexpr (Mode == SinkAccessMode::ReadOnly) {
    return SinkArg<Mode, decltype(make_apply_expr_arg(value))>(make_apply_expr_arg(value));
  } else {
    static_assert(is_math_expr_arg_v<T>,
                  "sink_expr(func, args...): read_write(...) and overwrite_all(...) arguments must be math "
                  "expressions. Scalars can only be read-only sink arguments.");
    return SinkArg<Mode, decltype(make_apply_expr_arg(value))>(make_apply_expr_arg(value));
  }
}

template <SinkAccessMode Mode, typename Expr>
KOKKOS_INLINE_FUNCTION auto make_sink_expr_arg(const SinkArg<Mode, Expr>& arg) {
  return arg;
}

template <typename T>
MUNDY_REQUIRES(!is_sink_arg_v<T>)
KOKKOS_INLINE_FUNCTION auto make_sink_expr_arg(const T& value) {
  return make_sink_arg_with_mode<SinkAccessMode::ReadOnly>(value);
}

}  // namespace impl

template <typename Arg>
KOKKOS_INLINE_FUNCTION auto read_only(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::ReadOnly>(arg);
}

template <typename Arg>
KOKKOS_INLINE_FUNCTION auto read_write(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::ReadWrite>(arg);
}

template <typename Arg>
KOKKOS_INLINE_FUNCTION auto overwrite_all(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::OverwriteAll>(arg);
}

/// \brief Expression node for applying an arbitrary non-mutating function object to expression values.
template <typename Func, typename... Exprs>
class ApplyValueExpr : public MathExprBase<ApplyValueExpr<Func, Exprs...>> {
 public:
  using our_t = ApplyValueExpr<Func, Exprs...>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<Exprs...>;
  static constexpr bool constrains_num_entities = false;
  static constexpr bool has_static_eval = std::is_empty_v<Func> && (Exprs::has_static_eval && ...);
  static constexpr size_t num_args = sizeof...(Exprs);

  KOKKOS_INLINE_FUNCTION
  ApplyValueExpr(Func func, Exprs... exprs) : func_(func), exprs_(make_tuple(exprs...)) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto values = impl::expr_chain(exprs_, fmis, context);
    return apply_values(values, std::make_index_sequence<num_args>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        auto [value_handles, final_cache] = impl::cached_expr_chain<EvalCountsType, eval_counts>(
            exprs_, fmis, std::forward<OldCacheType>(old_cache), context);
        auto val = apply_cached_values(value_handles, final_cache, std::make_index_sequence<num_args>{});
        auto newer_cache = append<our_tag>(std::move(final_cache), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newer_cache));
      }
    } else {
      auto [value_handles, final_cache] = impl::cached_expr_chain<EvalCountsType, eval_counts>(
          exprs_, fmis, std::forward<OldCacheType>(old_cache), context);
      auto val = apply_cached_values(value_handles, final_cache, std::make_index_sequence<num_args>{});
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(final_cache));
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return runtime_reuse_equivalent_impl(other, std::make_index_sequence<num_args>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    validate_runtime_reuse_impl<EvalCountsType, eval_counts>(std::make_index_sequence<num_args>{}, validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    propagate_synchronize_impl(std::make_index_sequence<num_args>{}, context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an apply expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an apply expression, which returns a temporary value.");
  }

  const auto driver() const {
    return driver_impl<0>();
  }

 private:
  template <typename ValuesTuple, size_t... Is>
  KOKKOS_INLINE_FUNCTION auto apply_values(const ValuesTuple& values, std::index_sequence<Is...>) const {
    if constexpr (std::is_invocable_v<const Func&, decltype(get<Is>(values))...>) {
      using result_t = std::invoke_result_t<const Func&, decltype(get<Is>(values))...>;
      static_assert(!std::is_void_v<result_t>,
                    "apply_expr(func, args...): func returned void. Use an explicit side-effect expression for "
                    "mutating functions; value expressions must return a value.");
      return func_(get<Is>(values)...);
    } else {
      static_assert(impl::dependent_false_v<Func, decltype(get<Is>(values))...>,
                    "apply_expr(func, args...): func cannot be called with the read-only evaluated values of the "
                    "provided expressions. If the function needs non-const arguments, it is mutating and must use an "
                    "explicit side-effect expression.");
    }
  }

  template <typename ValueHandlesTuple, typename CacheType, size_t... Is>
  KOKKOS_INLINE_FUNCTION auto apply_cached_values(const ValueHandlesTuple& value_handles, const CacheType& cache,
                                                  std::index_sequence<Is...>) const {
    if constexpr (std::is_invocable_v<const Func&, decltype(get<Is>(value_handles).get(cache))...>) {
      using result_t = std::invoke_result_t<const Func&, decltype(get<Is>(value_handles).get(cache))...>;
      static_assert(!std::is_void_v<result_t>,
                    "apply_expr(func, args...): func returned void. Use an explicit side-effect expression for "
                    "mutating functions; value expressions must return a value.");
      return func_(get<Is>(value_handles).get(cache)...);
    } else {
      static_assert(impl::dependent_false_v<Func, decltype(get<Is>(value_handles).get(cache))...>,
                    "apply_expr(func, args...): func cannot be called with the read-only cached values of the "
                    "provided expressions. If the function needs non-const arguments, it is mutating and must use an "
                    "explicit side-effect expression.");
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t... Is>
  void validate_runtime_reuse_impl(std::index_sequence<Is...>, impl::RuntimeReuseValidator& validator) const {
    (get<Is>(exprs_).template validate_runtime_reuse<EvalCountsType, eval_counts>(validator), ...);
  }

  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION bool runtime_reuse_equivalent_impl(const our_t& other, std::index_sequence<Is...>) const {
    if constexpr (std::is_empty_v<Func>) {
      return (get<Is>(exprs_).runtime_reuse_equivalent(get<Is>(other.exprs_)) && ...);
    } else {
      return false;
    }
  }

  template <size_t... Is>
  void propagate_synchronize_impl(std::index_sequence<Is...>, const NgpEvalContext& context) {
    (get<Is>(exprs_).flag_read_only(context), ...);
    (get<Is>(exprs_).propagate_synchronize(context), ...);
  }

  template <size_t I>
  const auto driver_impl() const {
    using nullptr_t = decltype(nullptr);
    if constexpr (I == num_args) {
      static_assert(impl::dependent_false_v<Func, Exprs...>,
                    "apply_expr(func, args...): at least one argument must have a non-null driver. Did you pass only "
                    "constants?");
      return nullptr;
    } else {
      const auto& expr = get<I>(exprs_);
      if constexpr (std::is_same_v<nullptr_t, decltype(expr.driver())>) {
        return driver_impl<I + 1>();
      } else {
        auto d = expr.driver();
        check_remaining_drivers<I + 1>(d);
        return d;
      }
    }
  }

  template <size_t I, typename DriverType>
  void check_remaining_drivers(const DriverType& driver) const {
    using nullptr_t = decltype(nullptr);
    if constexpr (I < num_args) {
      const auto& expr = get<I>(exprs_);
      if constexpr (!std::is_same_v<nullptr_t, decltype(expr.driver())>) {
        MUNDY_THROW_REQUIRE(driver == expr.driver(), std::logic_error, "Mismatched drivers in apply expression");
      }
      check_remaining_drivers<I + 1>(driver);
    }
  }

  Func func_;
  tuple<Exprs...> exprs_;
};

template <typename Func, typename... Args>
auto apply_expr(Func func, const Args&... args) {
  static_assert(sizeof...(Args) > 0, "apply_expr(func, args...): at least one argument is required.");
  static_assert((impl::is_math_expr_arg_v<Args> || ...),
                "apply_expr(func, args...): at least one argument must be a math expression so Mundy knows which "
                "entity driver should evaluate the expression. Scalars are allowed, but they cannot be the only "
                "arguments.");
  return ApplyValueExpr<std::decay_t<Func>, decltype(impl::make_apply_expr_arg(args))...>(std::move(func),
                                                                                    impl::make_apply_expr_arg(args)...);
}

#define MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ExprClassName, FuncName, FuncCall)                                  \
  struct ExprClassName##Func {                                                                               \
    template <typename... Values>                                                                            \
    KOKKOS_INLINE_FUNCTION auto operator()(const Values&... values) const -> decltype(FuncCall(values...)) { \
      return FuncCall(values...);                                                                            \
    }                                                                                                        \
  };                                                                                                         \
  template <typename... Exprs>                                                                               \
  using ExprClassName##Expr = ApplyValueExpr<ExprClassName##Func, Exprs...>;                                 \
  template <typename... Args>                                                                                \
  MUNDY_REQUIRES((impl::is_math_expr_arg_v<Args> || ...))                                                    \
  auto FuncName(const Args&... args) {                                                                       \
    return apply_expr(ExprClassName##Func{}, args...);                                                       \
  }

template <typename Func, typename... SinkArgs>
class ApplySinkExpr : public MathExprBase<ApplySinkExpr<Func, SinkArgs...>> {
 public:
  using our_t = ApplySinkExpr<Func, SinkArgs...>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<typename SinkArgs::expr_t...>;
  static constexpr bool constrains_num_entities = false;
  static constexpr bool has_static_eval = false;
  static constexpr size_t num_args = sizeof...(SinkArgs);

  KOKKOS_INLINE_FUNCTION
  ApplySinkExpr(Func func, SinkArgs... sink_args) : func_(func), sink_args_(make_tuple(sink_args...)) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto values = eval_arg_chain(fmis, context);
    apply_values(values, std::make_index_sequence<num_args>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(!aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");
    auto [value_handles, final_cache] = cached_arg_chain<EvalCountsType, eval_counts>(
        fmis, std::forward<OldCacheType>(old_cache), context);
    apply_cached_values(value_handles, final_cache, std::make_index_sequence<num_args>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    validate_runtime_reuse_impl<EvalCountsType, eval_counts>(std::make_index_sequence<num_args>{}, validator);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    flag_sink_args<impl::SinkAccessMode::ReadOnly>(std::make_index_sequence<num_args>{}, context);
    flag_sink_args<impl::SinkAccessMode::ReadWrite>(std::make_index_sequence<num_args>{}, context);
    flag_sink_args<impl::SinkAccessMode::OverwriteAll>(std::make_index_sequence<num_args>{}, context);
    propagate_synchronize_impl(std::make_index_sequence<num_args>{}, context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to read the return type of a sink expression, which returns void.");
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to write to the return type of a sink expression, which returns void.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to write to the return type of a sink expression, which returns void.");
  }

  const auto driver() const {
    return driver_impl<0>();
  }

 private:
  template <size_t I>
  using sink_arg_i_t = tuple_element_t<I, tuple<SinkArgs...>>;

  template <size_t I = 0, size_t NumEntities>
  KOKKOS_FUNCTION auto eval_arg_chain(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                      const NgpEvalContext& context) const {
    if constexpr (I == num_args) {
      return tuple<>{};
    } else {
      auto val_i = get<I>(sink_args_).expr().eval(fmis, context);
      auto vals_tail = eval_arg_chain<I + 1>(fmis, context);
      return tuple_cat(tuple{std::move(val_i)}, std::move(vals_tail));
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t I = 0, size_t NumEntities, typename CacheType>
  KOKKOS_FUNCTION auto cached_arg_chain(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                        CacheType&& cache, const NgpEvalContext& context) const {
    if constexpr (I == num_args) {
      return Kokkos::make_pair(tuple<>{}, std::forward<CacheType>(cache));
    } else {
      auto result_i = get<I>(sink_args_).expr().template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<CacheType>(cache), context);
      auto value_handle_i = std::move(result_i.first);
      auto next_cache = std::move(result_i.second);
      auto [vals_tail, final_cache] =
          cached_arg_chain<EvalCountsType, eval_counts, I + 1>(fmis, std::move(next_cache), context);
      auto vals_all = tuple_cat(tuple{std::move(value_handle_i)}, std::move(vals_tail));
      return Kokkos::make_pair(vals_all, std::move(final_cache));
    }
  }

  template <size_t I, typename ValuesTuple>
  KOKKOS_INLINE_FUNCTION decltype(auto) sink_eval_arg(ValuesTuple& values) const {
    if constexpr (sink_arg_i_t<I>::mode == impl::SinkAccessMode::ReadOnly) {
      using value_t = std::remove_reference_t<decltype(get<I>(values))>;
      return static_cast<const value_t&>(get<I>(values));
    } else {
      return (get<I>(values));
    }
  }

  template <size_t I, typename ValueHandlesTuple, typename CacheType>
  KOKKOS_INLINE_FUNCTION decltype(auto) sink_cached_arg(ValueHandlesTuple& value_handles, CacheType& cache) const {
    if constexpr (sink_arg_i_t<I>::mode == impl::SinkAccessMode::ReadOnly) {
      using value_t = std::remove_reference_t<decltype(get<I>(value_handles).get(cache))>;
      return static_cast<const value_t&>(get<I>(value_handles).get(cache));
    } else {
      return (get<I>(value_handles).get(cache));
    }
  }

  template <typename ValuesTuple, size_t... Is>
  KOKKOS_INLINE_FUNCTION void apply_values(ValuesTuple& values, std::index_sequence<Is...>) const {
    if constexpr (std::is_invocable_v<const Func&, decltype(sink_eval_arg<Is>(values))...>) {
      using result_t = std::invoke_result_t<const Func&, decltype(sink_eval_arg<Is>(values))...>;
      static_assert(std::is_void_v<result_t>,
                    "sink_expr(func, args...): func must return void. Mutating functions that return a value are not "
                    "supported by generic sink expressions.");
      func_(sink_eval_arg<Is>(values)...);
    } else {
      static_assert(impl::dependent_false_v<Func, decltype(sink_eval_arg<Is>(values))...>,
                    "sink_expr(func, args...): func cannot be called with the requested read-only/read-write/"
                    "overwrite-all evaluated arguments. Check that every mutating argument is wrapped with "
                    "read_write(...) or overwrite_all(...), and every write-mode expression evaluates to a mutable "
                    "lvalue.");
    }
  }

  template <typename ValueHandlesTuple, typename CacheType, size_t... Is>
  KOKKOS_INLINE_FUNCTION void apply_cached_values(ValueHandlesTuple& value_handles, CacheType& cache,
                                                  std::index_sequence<Is...>) const {
    if constexpr (std::is_invocable_v<const Func&, decltype(sink_cached_arg<Is>(value_handles, cache))...>) {
      using result_t = std::invoke_result_t<const Func&, decltype(sink_cached_arg<Is>(value_handles, cache))...>;
      static_assert(std::is_void_v<result_t>,
                    "sink_expr(func, args...): func must return void. Mutating functions that return a value are not "
                    "supported by generic sink expressions.");
      func_(sink_cached_arg<Is>(value_handles, cache)...);
    } else {
      static_assert(impl::dependent_false_v<Func, decltype(sink_cached_arg<Is>(value_handles, cache))...>,
                    "sink_expr(func, args...): func cannot be called with the requested read-only/read-write/"
                    "overwrite-all cached arguments. Check that every mutating argument is wrapped with "
                    "read_write(...) or overwrite_all(...), and every write-mode expression evaluates to a mutable "
                    "lvalue.");
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t... Is>
  void validate_runtime_reuse_impl(std::index_sequence<Is...>, impl::RuntimeReuseValidator& validator) const {
    (get<Is>(sink_args_).expr().template validate_runtime_reuse<EvalCountsType, eval_counts>(validator), ...);
  }

  template <impl::SinkAccessMode Mode, size_t... Is>
  void flag_sink_args(std::index_sequence<Is...>, const NgpEvalContext& context) {
    (flag_sink_arg<Mode, Is>(context), ...);
  }

  template <impl::SinkAccessMode Mode, size_t I>
  void flag_sink_arg(const NgpEvalContext& context) {
    if constexpr (sink_arg_i_t<I>::mode == Mode) {
      if constexpr (Mode == impl::SinkAccessMode::ReadOnly) {
        get<I>(sink_args_).expr().flag_read_only(context);
      } else if constexpr (Mode == impl::SinkAccessMode::ReadWrite) {
        get<I>(sink_args_).expr().flag_read_write(context);
      } else {
        get<I>(sink_args_).expr().flag_overwrite_all(context);
      }
    }
  }

  template <size_t... Is>
  void propagate_synchronize_impl(std::index_sequence<Is...>, const NgpEvalContext& context) {
    (get<Is>(sink_args_).expr().propagate_synchronize(context), ...);
  }

  template <size_t I>
  const auto driver_impl() const {
    using nullptr_t = decltype(nullptr);
    if constexpr (I == num_args) {
      static_assert(impl::dependent_false_v<Func, SinkArgs...>,
                    "sink_expr(func, args...): at least one argument must have a non-null driver. Did you pass only "
                    "constants?");
      return nullptr;
    } else {
      const auto& expr = get<I>(sink_args_).expr();
      if constexpr (std::is_same_v<nullptr_t, decltype(expr.driver())>) {
        return driver_impl<I + 1>();
      } else {
        auto d = expr.driver();
        check_remaining_drivers<I + 1>(d);
        return d;
      }
    }
  }

  template <size_t I, typename DriverType>
  void check_remaining_drivers(const DriverType& driver) const {
    using nullptr_t = decltype(nullptr);
    if constexpr (I < num_args) {
      const auto& expr = get<I>(sink_args_).expr();
      if constexpr (!std::is_same_v<nullptr_t, decltype(expr.driver())>) {
        MUNDY_THROW_REQUIRE(driver == expr.driver(), std::logic_error, "Mismatched drivers in sink expression");
      }
      check_remaining_drivers<I + 1>(driver);
    }
  }

  Func func_;
  tuple<SinkArgs...> sink_args_;
};

/// \brief Build a side-effect expression that applies a mutating function object to evaluated expression arguments.
///
/// `sink_expr` is the low-level escape hatch for mutating functions that Mundy does not wrap directly. Most users
/// should prefer a named wrapper such as `rotate_quaternion(q(es), omega(es), dt)` when one exists. Named wrappers
/// run immediately; use `sink_expr` when you need to construct the expression object yourself.
///
/// Rules:
/// - At least one argument must be an expression with a non-null driver.
/// - Unwrapped arguments are treated as `read_only(...)`.
/// - Scalars are allowed only as read-only arguments and are converted to `ConstantMathExpr`.
/// - `read_write(expr)` means the old device value is synchronized, then the field is marked modified on device.
/// - `overwrite_all(expr)` means host sync state is cleared, then the field is marked modified on device.
/// - Write-mode arguments must be expressions that evaluate to mutable lvalues.
/// - The callable must return `void`.
/// - Mutating functions that also return a value are intentionally unsupported (for now).
template <typename Func, typename... Args>
auto sink_expr(Func func, const Args&... args) {
  static_assert(sizeof...(Args) > 0, "sink_expr(func, args...): at least one argument is required.");
  static_assert((impl::sink_arg_has_nonconstant_expr_v<Args> || ...),
                "sink_expr(func, args...): at least one argument must be a non-constant math expression so Mundy "
                "knows which entity driver should evaluate the expression. Scalars are allowed, but they cannot be "
                "the only arguments.");
  return ApplySinkExpr<std::decay_t<Func>, decltype(impl::make_sink_expr_arg(args))...>(
      std::move(func), impl::make_sink_expr_arg(args)...);
}

namespace impl {

template <SinkAccessMode... Modes>
struct SinkArgPolicy {
  static constexpr size_t num_args = sizeof...(Modes);
};

template <size_t I, SinkAccessMode First, SinkAccessMode... Rest>
struct NthSinkAccessMode : NthSinkAccessMode<I - 1, Rest...> {};

template <SinkAccessMode First, SinkAccessMode... Rest>
struct NthSinkAccessMode<0, First, Rest...> {
  static constexpr SinkAccessMode value = First;
};

template <typename Policy, size_t I>
struct SinkPolicyMode;

template <SinkAccessMode... Modes, size_t I>
struct SinkPolicyMode<SinkArgPolicy<Modes...>, I> {
  static_assert(I < sizeof...(Modes), "Named sink expression argument index is out of range for its access policy.");
  static constexpr SinkAccessMode value = NthSinkAccessMode<I, Modes...>::value;
};

template <typename Policy, typename Func, typename... Args, size_t... Is>
auto make_named_sink_expr_impl(Func func, std::index_sequence<Is...>, const Args&... args) {
  return sink_expr(std::move(func), make_sink_arg_with_mode<SinkPolicyMode<Policy, Is>::value>(args)...);
}

template <typename Policy, typename Func, typename... Args>
auto make_named_sink_expr(Func func, const Args&... args) {
  static_assert(sizeof...(Args) == Policy::num_args,
                "Named sink expression called with the wrong number of arguments for its access policy.");
  return make_named_sink_expr_impl<Policy>(std::move(func), std::make_index_sequence<sizeof...(Args)>{}, args...);
}

}  // namespace impl

#define MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY ::mundy::mesh::impl::SinkAccessMode::ReadOnly
#define MUNDY_ACCESSOR_EXPR_SINK_READ_WRITE ::mundy::mesh::impl::SinkAccessMode::ReadWrite
#define MUNDY_ACCESSOR_EXPR_SINK_OVERWRITE_ALL ::mundy::mesh::impl::SinkAccessMode::OverwriteAll

#define MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC(ExprClassName, FuncName, FuncCall, ...)                         \
  struct ExprClassName##SinkFunc {                                                                            \
    template <typename... Values>                                                                             \
    KOKKOS_INLINE_FUNCTION auto operator()(Values&&... values) const                                          \
        -> decltype(FuncCall(std::forward<Values>(values)...)) {                                              \
      return FuncCall(std::forward<Values>(values)...);                                                       \
    }                                                                                                         \
  };                                                                                                          \
  using ExprClassName##SinkPolicy = impl::SinkArgPolicy<__VA_ARGS__>;                                         \
  template <typename... SinkArgs>                                                                             \
  using ExprClassName##SinkExpr = ApplySinkExpr<ExprClassName##SinkFunc, SinkArgs...>;                        \
  template <typename... Args>                                                                                 \
  MUNDY_REQUIRES((impl::is_math_expr_arg_v<Args> || ...))                                                     \
  void FuncName(const Args&... args) {                                                                        \
    auto expr = impl::make_named_sink_expr<ExprClassName##SinkPolicy>(ExprClassName##SinkFunc{}, args...);    \
    expr.driver()->run(expr);                                                                                 \
  }

template <typename Policy, typename LeftMathExpr, typename RightMathExpr>
class BinarySideEffectExpr : public MathExprBase<BinarySideEffectExpr<Policy, LeftMathExpr, RightMathExpr>> {
 public:
  using our_t = BinarySideEffectExpr<Policy, LeftMathExpr, RightMathExpr>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<LeftMathExpr, RightMathExpr>;
  static constexpr bool constrains_num_entities = false;
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  BinarySideEffectExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {
  }

  KOKKOS_INLINE_FUNCTION
  BinarySideEffectExpr(const EntityExprBase<LeftMathExpr>& left, const EntityExprBase<RightMathExpr>& right)
      : left_(left.self()), right_(right.self()) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto&& left_ref = left_.eval(fmis, context);
    auto&& right_ref = right_.eval(fmis, context);
    Policy::apply(left_ref, right_ref);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(!aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");
    auto left_result =
        left_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache), context);
    auto left_value = std::move(left_result.first);
    auto right_result =
        right_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(left_result.second), context);
    auto&& left_ref = left_value.get(right_result.second);
    auto&& right_ref = right_result.first.get(right_result.second);
    Policy::apply(left_ref, right_ref);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    left_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    right_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    left_.flag_read_write(context);
    right_.flag_read_only(context);
    left_.propagate_synchronize(context);
    right_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of a binary expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of a binary expression, which returns a temporary value.");
  }

  const auto driver() const {
    using nullptr_t = decltype(nullptr);

    constexpr bool has_left_driver = !std::is_same_v<nullptr_t, decltype(left_.driver())>;
    constexpr bool has_right_driver = !std::is_same_v<nullptr_t, decltype(right_.driver())>;
    static_assert(
        has_left_driver || has_right_driver,
        "At least one of the left or right expressions in a binary math expression must have a non-null driver.");

    if constexpr (has_left_driver) {
      auto d = left_.driver();
      if constexpr (has_right_driver) {
        MUNDY_THROW_REQUIRE(d == right_.driver(), std::logic_error, "Mismatched drivers in binary math expression");
      }
      return d;
    } else {
      return right_.driver();
    }
  }

 private:
  LeftMathExpr left_;
  RightMathExpr right_;
};

#define MUNDY_ACCESSOR_EXPR_OP_EQUALS(OpName, op_equals)                             \
  struct OpName##EqualsPolicy {                                                      \
    template <typename LeftValue, typename RightValue>                               \
    KOKKOS_INLINE_FUNCTION static void apply(LeftValue&& left, RightValue&& right) { \
      left op_equals right;                                                          \
    }                                                                                \
  };                                                                                 \
  template <typename LeftMathExpr, typename RightMathExpr>                           \
  using OpName##EqualsExpr = BinarySideEffectExpr<OpName##EqualsPolicy, LeftMathExpr, RightMathExpr>;

#define MUNDY_ACCESSOR_EXPR_ATOMIC_OP(ExprClassName, AtomicName, atomic_op)                                        \
  struct ExprClassName##Policy {                                                                                   \
    template <typename LeftValue, typename RightValue>                                                             \
    KOKKOS_INLINE_FUNCTION static void apply(LeftValue&& left, RightValue&& right) {                               \
      atomic_op(&left, right);                                                                                     \
    }                                                                                                              \
  };                                                                                                               \
  template <typename LeftMathExpr, typename RightMathExpr>                                                         \
  using ExprClassName##Expr = BinarySideEffectExpr<ExprClassName##Policy, LeftMathExpr, RightMathExpr>;            \
  template <typename LeftExpr, typename RightExpr>                                                                 \
  MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, LeftExpr> || is_crtp_base_of_v<MathExprBase, RightExpr>)          \
  auto AtomicName(const MathExprBase<LeftExpr>& left_expr, const MathExprBase<RightExpr>& right_expr) {            \
    return ExprClassName##Expr<LeftExpr, RightExpr>(left_expr.self(), right_expr.self());                          \
  }                                                                                                                \
  template <typename LeftExpr, typename RightT>                                                                    \
  MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, LeftExpr> && !is_crtp_base_of_v<MathExprBase, RightT>)            \
  auto AtomicName(const MathExprBase<LeftExpr>& left_expr, const RightT& right_const) {                            \
    using RightExpr = ConstantMathExpr<RightT>;                                                                    \
    RightExpr right_expr(right_const);                                                                             \
    return ExprClassName##Expr<LeftExpr, RightExpr>(left_expr.self(), right_expr);                                 \
  }                                                                                                                \
  template <typename LeftT, typename RightExpr>                                                                    \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, LeftT> && is_crtp_base_of_v<MathExprBase, RightExpr>)            \
  auto AtomicName(const LeftT& left_const, const MathExprBase<RightExpr>& right_expr) {                            \
    using LeftExpr = ConstantMathExpr<LeftT>;                                                                      \
    LeftExpr left_expr(left_const);                                                                                \
    return ExprClassName##Expr<LeftExpr, RightExpr>(left_expr, right_expr.self());                                 \
  }                                                                                                                \
  template <typename LeftT, typename RightT>                                                                       \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, LeftT> && !is_crtp_base_of_v<MathExprBase, RightT>)              \
  void AtomicName(const LeftT& left_const, const RightT& right_const) {                                            \
    MUNDY_THROW_REQUIRE(                                                                                           \
        false, std::logic_error,                                                                                   \
        "At least one argument to " #AtomicName                                                                    \
        " must be a math expression.\n"                                                                            \
        "The provided arguments were both constants. How would we know how to run the expression over entities?"); \
  }

// Add + Add
// Add + Sub
// Add + Mul
// Add + Div
//
// Sub + Add
// Sub + Sub
// Sub + Mul
// Sub + Div
//
// Mul + Add
// Mul + Sub
// Mul + Mul
// Mul + Div
//
// Div + Add
// Div + Sub
// Div + Mul
// Div + Div

MUNDY_ACCESSOR_EXPR_OP(Add, +)
MUNDY_ACCESSOR_EXPR_OP(Sub, -)
MUNDY_ACCESSOR_EXPR_OP(Div, /)
MUNDY_ACCESSOR_EXPR_OP(Mul, *)

#define MUNDY_ACCESSOR_EXPR_BINARY_VALUE_OPERATOR(OpName, op)                                   \
  template <typename Policy, typename LeftMathExpr, typename RightMathExpr, typename OtherExpr> \
  auto operator op(const BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>& expr,            \
                   const MathExprBase<OtherExpr>& other) {                                      \
    using our_t = BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>;                         \
    return OpName##Expr<our_t, OtherExpr>(expr.self(), other.self());                           \
  }

#define MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_LEFT_OPERATOR(OpName, op)                                          \
  template <typename ConstantType, typename Policy, typename LeftMathExpr, typename RightMathExpr>                   \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ConstantType& c, const BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>& expr) {        \
    using expr_t = BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>;                                             \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<ConstantMathExpr<ConstantType>, expr_t>(constant_expr, expr);                                \
  }

#define MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_RIGHT_OPERATOR(OpName, op)                                         \
  template <typename ConstantType, typename Policy, typename LeftMathExpr, typename RightMathExpr>                   \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>& expr, const ConstantType& c) {        \
    using expr_t = BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>;                                             \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<expr_t, ConstantMathExpr<ConstantType>>(expr, constant_expr);                                \
  }

#define MUNDY_ACCESSOR_EXPR_APPLY_VALUE_OPERATOR(OpName, op)                                           \
  template <typename Func, typename... Exprs, typename OtherExpr>                                      \
  auto operator op(const ApplyValueExpr<Func, Exprs...>& expr, const MathExprBase<OtherExpr>& other) { \
    using our_t = ApplyValueExpr<Func, Exprs...>;                                                      \
    return OpName##Expr<our_t, OtherExpr>(expr.self(), other.self());                                  \
  }

#define MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_LEFT_OPERATOR(OpName, op)                                           \
  template <typename ConstantType, typename Func, typename... Exprs>                                                 \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ConstantType& c, const ApplyValueExpr<Func, Exprs...>& expr) {                              \
    using expr_t = ApplyValueExpr<Func, Exprs...>;                                                                   \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<ConstantMathExpr<ConstantType>, expr_t>(constant_expr, expr);                                \
  }

#define MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_RIGHT_OPERATOR(OpName, op)                                          \
  template <typename ConstantType, typename Func, typename... Exprs>                                                 \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ApplyValueExpr<Func, Exprs...>& expr, const ConstantType& c) {                              \
    using expr_t = ApplyValueExpr<Func, Exprs...>;                                                                   \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<expr_t, ConstantMathExpr<ConstantType>>(expr, constant_expr);                                \
  }

MUNDY_ACCESSOR_EXPR_BINARY_VALUE_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_LEFT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_LEFT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_LEFT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_LEFT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_RIGHT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_RIGHT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_RIGHT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_BINARY_VALUE_CONSTANT_RIGHT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_LEFT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_LEFT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_LEFT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_LEFT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_RIGHT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_RIGHT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_RIGHT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_APPLY_VALUE_CONSTANT_RIGHT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Add, +=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Sub, -=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Div, /=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Mul, *=)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicAdd, atomic_add, ::mundy::atomic_add)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicSub, atomic_sub, ::mundy::atomic_sub)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicMul, atomic_mul, ::mundy::atomic_mul)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicDiv, atomic_div, ::mundy::atomic_div)

// Vector/Matrix/Quaternion functions
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Copy, copy, copy)                       // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sum, sum, sum)                          // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Product, product, ::mundy::product)     // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Min, min, ::mundy::min)                 // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Max, max, ::mundy::max)                 // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Mean, mean, ::mundy::mean)              // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Variance, variance, ::mundy::variance)  // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(StdDev, stddev, ::mundy::stddev)        // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Norm, norm, ::mundy::norm)              // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(OneNorm, one_norm, ::mundy::one_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(InfNorm, inf_norm, ::mundy::inf_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(TwoNorm, two_norm, ::mundy::two_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(TwoNormSquared, two_norm_squared, ::mundy::two_norm_squared)  // v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(NormSquared, norm_squared, ::mundy::norm_squared)             // v, q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(InfinityNorm, infinity_norm, ::mundy::inf_norm)               // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Inverse, inverse, ::mundy::inverse)                           // m, q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Conjugate, conjugate, ::mundy::conjugate)                     // q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Normalize, normalize, ::mundy::normalize)                     // q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Trace, trace, ::mundy::trace)                                 // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Transpose, transpose, ::mundy::transpose)                     // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Determinant, determinant, ::mundy::determinant)               // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Adjugate, adjugate, ::mundy::adjugate)                        // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cofactors, cofactors, ::mundy::cofactors)                     // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(FrobeniusNorm, frobenius_norm, ::mundy::frobenius_norm)       // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Dot, dot, ::mundy::dot)                                       // v-v, q-q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(MinorAngle, minor_angle, ::mundy::minor_angle)                // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(MajorAngle, major_angle, ::mundy::major_angle)                // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(FrobeniusInnerProduct, frobenius_inner_product,
                                 ::mundy::frobenius_inner_product)                           // m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(OuterProduct, outer_product, ::mundy::outer_product)        // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cross, cross, ::mundy::cross)                               // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ElementwiseMul, elementwise_mul, ::mundy::elementwise_mul)  // v-v, m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ElementwiseDiv, elementwise_div, ::mundy::elementwise_div)  // v-v, m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Slerp, slerp, ::mundy::slerp)                               // q-q
MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC(RotateQuaternion, rotate_quaternion, ::mundy::rotate_quaternion,
                                      MUNDY_ACCESSOR_EXPR_SINK_READ_WRITE, MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY,
                                      MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY)  // q, v, s

// Scalar functions
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Abs, abs, Kokkos::abs)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sqrt, sqrt, Kokkos::sqrt)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Exp, exp, Kokkos::exp)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Log, log, Kokkos::log)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sin, sin, Kokkos::sin)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cos, cos, Kokkos::cos)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Tan, tan, Kokkos::tan)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Asin, asin, Kokkos::asin)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Acos, acos, Kokkos::acos)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Atan, atan, Kokkos::atan)

template <typename TaggedAccessorT, typename PrevEntityExpr>
class AccessorExpr : public MathExprBase<AccessorExpr<TaggedAccessorT, PrevEntityExpr>> {
 public:
  using our_t = AccessorExpr<TaggedAccessorT, PrevEntityExpr>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<PrevEntityExpr>;
  static constexpr bool constrains_num_entities = false;
  // Static under the tagged-accessor identity contract, given that the given entity is itself static.
  static constexpr bool has_static_eval = PrevEntityExpr::has_static_eval;

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(TaggedAccessorT tagged_accessor, const PrevEntityExpr& prev_entity_expr)
      : tagged_accessor_(tagged_accessor), prev_entity_expr_(prev_entity_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(TaggedAccessorT tagged_accessor, const EntityExprBase<PrevEntityExpr>& prev_entity_expr_base)
      : tagged_accessor_(tagged_accessor), prev_entity_expr_(prev_entity_expr_base.self()) {
  }

  KOKKOS_DEFAULTED_FUNCTION
  AccessorExpr(const our_t&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  AccessorExpr(our_t&&) = default;

  auto operator=(const our_t& other) {
    AssignExpr<our_t, our_t> expr(*this, other);
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  MUNDY_REQUIRES(!std::is_same_v<OtherExpr, our_t>)
  auto operator=(const MathExprBase<OtherExpr>& other) {
    AssignExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  MUNDY_REQUIRES(!std::is_same_v<OtherExpr, our_t>)
  auto operator=(const EntityExprBase<OtherExpr>& other) {
    AssignExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  auto operator+(const MathExprBase<OtherExpr>& other) const {
    return AddExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator-(const MathExprBase<OtherExpr>& other) const {
    return SubExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator*(const MathExprBase<OtherExpr>& other) const {
    return MulExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  auto operator/(const MathExprBase<OtherExpr>& other) const {
    return DivExpr<our_t, OtherExpr>(*this, other.self());
  }

  template <typename OtherExpr>
  void operator+=(const MathExprBase<OtherExpr>& other) {
    AddEqualsExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator-=(const MathExprBase<OtherExpr>& other) {
    SubEqualsExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator*=(const MathExprBase<OtherExpr>& other) {
    MulEqualsExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename OtherExpr>
  void operator/=(const MathExprBase<OtherExpr>& other) {
    DivEqualsExpr<our_t, OtherExpr> expr(*this, other.self());
    expr.driver()->run(expr);
  }

  template <typename ConstantType>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>)
  auto operator=(const ConstantType& c) {
    ConstantMathExpr<ConstantType> constant_expr(c);
    AssignExpr<our_t, ConstantMathExpr<ConstantType>> expr(*this, constant_expr);
    expr.driver()->run(expr);
  }

  template <typename ConstantType>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>)
  auto operator+=(const ConstantType& c) {
    ConstantMathExpr<ConstantType> constant_expr(c);
    AddEqualsExpr<our_t, ConstantMathExpr<ConstantType>> expr(*this, constant_expr);
    expr.driver()->run(expr);
  }

  template <typename ConstantType>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>)
  auto operator-=(const ConstantType& c) {
    ConstantMathExpr<ConstantType> constant_expr(c);
    SubEqualsExpr<our_t, ConstantMathExpr<ConstantType>> expr(*this, constant_expr);
    expr.driver()->run(expr);
  }

  template <typename ConstantType>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>)
  auto operator*=(const ConstantType& c) {
    ConstantMathExpr<ConstantType> constant_expr(c);
    MulEqualsExpr<our_t, ConstantMathExpr<ConstantType>> expr(*this, constant_expr);
    expr.driver()->run(expr);
  }

  template <typename ConstantType>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>)
  auto operator/=(const ConstantType& c) {
    ConstantMathExpr<ConstantType> constant_expr(c);
    DivEqualsExpr<our_t, ConstantMathExpr<ConstantType>> expr(*this, constant_expr);
    expr.driver()->run(expr);
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return tagged_accessor_(entity_index);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Eval our subexpressions first
        auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto entity_value = std::move(entity_result.first);

        // Our eval result needs cached, but is not yet cached
        auto val = tagged_accessor_(entity_value.get(entity_result.second));
        auto newest_cache = append<our_tag>(std::move(entity_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto entity_value = std::move(entity_result.first);
      auto val = tagged_accessor_(entity_value.get(entity_result.second));
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(entity_result.second));
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    prev_entity_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return prev_entity_expr_.runtime_reuse_equivalent(other.prev_entity_expr_);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    tagged_accessor_.sync_to_device();
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    tagged_accessor_.sync_to_device();
    tagged_accessor_.modify_on_device();
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    tagged_accessor_.clear_host_sync_state();
    tagged_accessor_.modify_on_device();
  }

  void propagate_synchronize(const NgpEvalContext& /*context*/) {
  }

  const auto driver() const {
    return prev_entity_expr_.driver();
  }

  const PrevEntityExpr& prev_entity_expr() const {
    return prev_entity_expr_;
  }

 private:
  TaggedAccessorT tagged_accessor_;
  PrevEntityExpr prev_entity_expr_;
};

#define MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_LEFT_OPERATOR(OpName, op)                                              \
  template <typename ConstantType, typename SubTaggedAccessorT, typename SubPrevEntityExpr>                          \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ConstantType& c, const AccessorExpr<SubTaggedAccessorT, SubPrevEntityExpr>& expr) {         \
    using expr_t = AccessorExpr<SubTaggedAccessorT, SubPrevEntityExpr>;                                              \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<ConstantMathExpr<ConstantType>, expr_t>(constant_expr, expr);                                \
  }

#define MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_RIGHT_OPERATOR(OpName, op)                                             \
  template <typename ConstantType, typename SubTaggedAccessorT, typename SubPrevEntityExpr>                          \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const AccessorExpr<SubTaggedAccessorT, SubPrevEntityExpr>& expr, const ConstantType& c) {         \
    using expr_t = AccessorExpr<SubTaggedAccessorT, SubPrevEntityExpr>;                                              \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                 \
    return OpName##Expr<expr_t, ConstantMathExpr<ConstantType>>(expr, constant_expr);                                \
  }

MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_LEFT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_LEFT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_LEFT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_LEFT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_RIGHT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_RIGHT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_RIGHT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_ACCESSOR_CONSTANT_RIGHT_OPERATOR(Div, /)

template <typename DerivedMathExpr>
class MathExprBase : public CachableExprBase<DerivedMathExpr> {
 public:
  using our_t = MathExprBase<DerivedMathExpr>;
  using our_tag = typename CachableExprBase<DerivedMathExpr>::our_tag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr MathExprBase() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedMathExpr& self() const noexcept {
    return static_cast<const DerivedMathExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedMathExpr& self() noexcept {
    return static_cast<DerivedMathExpr&>(*this);
  }

  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const Ctx& context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename CacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          CacheType& cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, cache, context);
  }

  template <class Ctx>
  void propagate_synchronize(const Ctx& context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx& context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx& context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx& context) {
    self().flag_overwrite_all(context);
  }

  const auto driver() const {
    return self().driver();
  }
};
//@}

//! \name RNG stuff
//@{

/// RNG.rand<double>()
template <typename RNGExpr, typename T>
class RandomDistributionExpr : public MathExprBase<RandomDistributionExpr<RNGExpr, T>> {
 public:
  using our_t = RandomDistributionExpr<RNGExpr, T>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<RNGExpr>;
  static constexpr bool constrains_num_entities = false;

  // This method has a side effect on the RNG stream, so repeated uses must re-enter the RNG subtree rather than reuse a
  // cached draw.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  RandomDistributionExpr(const RNGExpr& rng_expr) : rng_expr_(rng_expr) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto rng = rng_expr_.eval(fmis, context);
    return rng.template rand<T>();
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Eval our subexpressions first
        auto rng_result = rng_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto rng_value = std::move(rng_result.first);

        // Our eval result needs cached, but is not yet cached
        auto&& rng_ref = rng_value.get(rng_result.second);
        auto val = rng_ref.template rand<T>();
        auto newest_cache = append<our_tag>(std::move(rng_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto rng_result = rng_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto rng_value = std::move(rng_result.first);
      auto&& rng_ref = rng_value.get(rng_result.second);
      auto val = rng_ref.template rand<T>();
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(rng_result.second));
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    rng_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    rng_expr_.flag_read_only(context);
    rng_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to mark a random number generator expression as read-write, but the return type is "
                        "a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to mark a random number generator expression as overwrite-all, but the return type "
                        "is a temporary value.");
  }

  const auto driver() const {
    return rng_expr_.driver();
  }

 private:
  RNGExpr rng_expr_;
};

// RNG.uniform<double>(low, high)
template <typename RNGExpr, typename T, typename LowExpr, typename HighExpr>
class UniformDistributionExpr : public MathExprBase<UniformDistributionExpr<RNGExpr, T, LowExpr, HighExpr>> {
 public:
  using our_t = UniformDistributionExpr<RNGExpr, T, LowExpr, HighExpr>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<RNGExpr, LowExpr, HighExpr>;
  static constexpr bool constrains_num_entities = false;

  // This method has a side effect on the RNG stream, so repeated uses must re-enter the RNG subtree rather than reuse a
  // cached draw.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  UniformDistributionExpr(const RNGExpr& rng_expr, const LowExpr& low_expr, const HighExpr& high_expr)
      : rng_expr_(rng_expr), low_expr_(low_expr), high_expr_(high_expr) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto rng = rng_expr_.eval(fmis, context);
    auto low = low_expr_.eval(fmis, context);
    auto high = high_expr_.eval(fmis, context);
    return rng.template uniform<T>(low, high);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Eval our subexpressions first
        auto rng_result = rng_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto rng_value = std::move(rng_result.first);
        auto low_result =
            low_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(rng_result.second), context);
        auto low_value = std::move(low_result.first);
        auto high_result =
            high_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(low_result.second), context);
        // Our eval result needs cached, but is not yet cached
        auto&& rng_ref = rng_value.get(high_result.second);
        auto val =
            rng_ref.template uniform<T>(low_value.get(high_result.second), high_result.first.get(high_result.second));
        auto final_cache = append<our_tag>(std::move(high_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(final_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto rng_result = rng_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto rng_value = std::move(rng_result.first);
      auto low_result =
          low_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(rng_result.second), context);
      auto low_value = std::move(low_result.first);
      auto high_result =
          high_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(low_result.second), context);
      auto&& rng_ref = rng_value.get(high_result.second);
      auto val =
          rng_ref.template uniform<T>(low_value.get(high_result.second), high_result.first.get(high_result.second));
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(high_result.second));
    }
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    rng_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    low_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    high_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    rng_expr_.flag_read_only(context);
    rng_expr_.propagate_synchronize(context);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to mark a random number generator expression as read-write, but the return type is "
                        "a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to mark a random number generator expression as overwrite-all, but the return type "
                        "is a temporary value.");
  }

  const auto driver() const {
    using nullptr_t = decltype(nullptr);
    constexpr bool has_rng_driver = !std::is_same_v<nullptr_t, decltype(rng_expr_.driver())>;
    constexpr bool has_low_driver = !std::is_same_v<nullptr_t, decltype(low_expr_.driver())>;
    constexpr bool has_high_driver = !std::is_same_v<nullptr_t, decltype(high_expr_.driver())>;
    static_assert(has_rng_driver,
                  "The RNG expression in a uniform distribution expression must have a non-null driver.");

    if constexpr (has_low_driver) {
      MUNDY_THROW_REQUIRE(rng_expr_.driver() == low_expr_.driver(), std::logic_error,
                          "Mismatched drivers in uniform distribution expression.");
    }
    if constexpr (has_high_driver) {
      MUNDY_THROW_REQUIRE(rng_expr_.driver() == high_expr_.driver(), std::logic_error,
                          "Mismatched drivers in uniform distribution expression.");
    }
    return rng_expr_.driver();
  }

 private:
  RNGExpr rng_expr_;
  LowExpr low_expr_;
  HighExpr high_expr_;
};

/// \brief An expression for generating random number generator based on a given seed and counter expression
/// This class is then used to generate expressions for drawing random numbers from various distributions
template <typename SeedExpr, typename CounterExpr, typename RNGType, RNGType (*make_counter_based_rng)(size_t, size_t)>
class CounterBasedRNGExpr
    : public MathExprBase<CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>> {
 public:
  using our_t = CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>;
  using our_tag = typename MathExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<SeedExpr, CounterExpr>;
  static constexpr bool constrains_num_entities = false;
  // RNG construction is a pure function of seed and counter, so it is static iff both inputs are static.
  static constexpr bool has_static_eval = SeedExpr::has_static_eval && CounterExpr::has_static_eval;
  // Runtime reuse is validated structurally on host before kernel launch.
  static constexpr bool supports_runtime_reuse = true;

  KOKKOS_INLINE_FUNCTION
  CounterBasedRNGExpr(const SeedExpr& seed_expr, const CounterExpr& counter_expr)
      : seed_expr_(seed_expr), counter_expr_(counter_expr) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    auto seed = seed_expr_.eval(fmis, context);
    auto counter = counter_expr_.eval(fmis, context);
    return make_counter_based_rng(seed, counter);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr ((our_t::has_static_eval || our_t::supports_runtime_reuse) && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        // The fact that our tag exists in the old cache means that our eval has cached its result before.
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        // Eval our subexpressions first
        auto seed_result = seed_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto seed_value = std::move(seed_result.first);
        auto counter_result = counter_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::move(seed_result.second), context);

        // Our eval result needs cached, but is not yet cached
        auto val = make_counter_based_rng(seed_value.get(counter_result.second),
                                          counter_result.first.get(counter_result.second));
        auto newest_cache = append<our_tag>(std::move(counter_result.second), val);
        return Kokkos::make_pair(impl::CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      // We don't need to cache our value, so just compute and return it
      auto seed_result = seed_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto seed_value = std::move(seed_result.first);
      auto counter_result =
          counter_expr_.template cached_eval<EvalCountsType, eval_counts>(fmis, std::move(seed_result.second), context);
      auto val = make_counter_based_rng(seed_value.get(counter_result.second),
                                        counter_result.first.get(counter_result.second));
      return Kokkos::make_pair(impl::OwnedCachedValue{std::move(val)}, std::move(counter_result.second));
    }
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return seed_expr_.runtime_reuse_equivalent(other.seed_expr_) &&
           counter_expr_.runtime_reuse_equivalent(other.counter_expr_);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    seed_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    counter_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  // Allow the user to rand_gen_expr.rand<double>() to get an expression for generating random doubles between 0 and 1
  template <typename T>
  auto rand() const {
    return RandomDistributionExpr<our_t, T>(*this);
  }

  // Allow the user to rand_gen_expr.uniform(low, high) to get an expression for generating random numbers between low
  // and high Low & high are expressions
  template <typename T, typename LowExpr, typename HighExpr>
  MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, LowExpr>&& is_crtp_base_of_v<MathExprBase, HighExpr>)
  auto uniform(const LowExpr& low_expr, const HighExpr& high_expr) const {
    return UniformDistributionExpr<our_t, T, LowExpr, HighExpr>(*this, low_expr, high_expr);
  }
  // Low is an expression but high is a constant
  template <typename T, typename LowExpr, typename HighT>
  MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, LowExpr> && !is_crtp_base_of_v<MathExprBase, HighT>)
  auto uniform(const LowExpr& low_expr, const HighT& high) const {
    ConstantMathExpr<HighT> high_expr(high);
    using HighExpr = ConstantMathExpr<HighT>;
    return uniform<T, LowExpr, HighExpr>(low_expr, high_expr);
  }
  // Low is a constant but high is an expression
  template <typename T, typename LowT, typename HighExpr>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, LowT> && is_crtp_base_of_v<MathExprBase, HighExpr>)
  auto uniform(const LowT& low, const HighExpr& high_expr) const {
    ConstantMathExpr<LowT> low_expr(low);
    using LowExpr = ConstantMathExpr<LowT>;
    return uniform<T, LowExpr, HighExpr>(low_expr, high_expr);
  }
  // Low & high are constants (perfectly allowed since the rng has a driver)
  template <typename T, typename LowT, typename HighT>
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, LowT> && !is_crtp_base_of_v<MathExprBase, HighT>)
  auto uniform(const LowT& low, const HighT& high) const {
    ConstantMathExpr<LowT> low_expr(low);
    ConstantMathExpr<HighT> high_expr(high);
    using LowExpr = ConstantMathExpr<LowT>;
    using HighExpr = ConstantMathExpr<HighT>;
    return uniform<T, LowExpr, HighExpr>(low_expr, high_expr);
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Our return type is naturally read-only. Nothing to do here.
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to mark a random number generator expression as read-write, but the RNG is a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    // Nothing to do here
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    seed_expr_.flag_read_only(context);
    counter_expr_.flag_read_only(context);
    seed_expr_.propagate_synchronize(context);
    counter_expr_.propagate_synchronize(context);
  }

  auto driver() const {
    using nullptr_t = decltype(nullptr);

    constexpr bool has_seed_driver = !std::is_same_v<nullptr_t, decltype(seed_expr_.driver())>;
    constexpr bool has_counter_driver = !std::is_same_v<nullptr_t, decltype(counter_expr_.driver())>;
    static_assert(has_seed_driver || has_counter_driver,
                  "At least one of the seed or counter expressions in a random generator expression must have a "
                  "non-null driver.\n"
                  "For example, they can't both be constants. How would we know how to run the expression.");

    if constexpr (has_seed_driver) {
      auto d = seed_expr_.driver();
      if constexpr (has_counter_driver) {
        MUNDY_THROW_REQUIRE(d == counter_expr_.driver(), std::logic_error,
                            "Mismatched drivers in random generator expression");
      }
      return d;
    } else {
      return counter_expr_.driver();
    }
  }

 private:
  SeedExpr seed_expr_;
  CounterExpr counter_expr_;
};

/// \brief Create a counter-based random number generator using the given seed and counter
/// Seed and counter are expressions
template <typename SeedExpr, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, SeedExpr>&& is_crtp_base_of_v<MathExprBase, CounterExpr>)
auto rng(const SeedExpr& seed_expr, const CounterExpr& counter_expr) {
  return CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Seed is an expression but counter is a constant
template <typename SeedExpr, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, SeedExpr> && !is_crtp_base_of_v<MathExprBase, CounterT>)
auto rng(const SeedExpr& seed_expr, const CounterT& counter) {
  using CounterExpr = ConstantMathExpr<CounterT>;
  auto counter_expr = CounterExpr(counter);
  return rng<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Seed is a constant but counter is an expression
template <typename SeedT, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, SeedT> && is_crtp_base_of_v<MathExprBase, CounterExpr>)
auto rng(const SeedT& seed, const CounterExpr& counter_expr) {
  using SeedExpr = ConstantMathExpr<SeedT>;
  auto seed_expr = SeedExpr(seed);
  return rng<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Both seed and counter are constants (not allowed)
template <typename SeedT, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, SeedT> && !is_crtp_base_of_v<MathExprBase, CounterT>)
void rng(const SeedT& seed, const CounterT& counter) {
  MUNDY_THROW_REQUIRE(false, std::logic_error,
                      "Both seed and counter arguments to rng() cannot be constants.\n"
                      "At least one of them must be an expression, lest we have no idea how to run the expression over "
                      "multiple entities.");
}
//@}

//! \name Helpers
//@{

template <typename... TrgSrcExprPairs>
class FusedAssignExpr : public MathExprBase<FusedAssignExpr<TrgSrcExprPairs...>> {
 public:
  using our_t = FusedAssignExpr<TrgSrcExprPairs...>;
  using our_tag = typename MathExprBase<FusedAssignExpr<TrgSrcExprPairs...>>::our_tag;
  using sub_expressions_t = tuple<TrgSrcExprPairs...>;
  static constexpr size_t num_pairs = sizeof...(TrgSrcExprPairs) / 2;
  static_assert(sizeof...(TrgSrcExprPairs) % 2 == 0,
                "The number of target/source expression pairs in FusedAssignExpr must be even.");
  static constexpr bool constrains_num_entities = false;
  // Fused assignment is side-effecting and returns void, so it is never a static cached value.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  FusedAssignExpr(const TrgSrcExprPairs&... exprs) : exprs_(make_tuple(exprs...)) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const NgpEvalContext& context) const {
    // Eval all expressions, storing their results for later.
    auto all_values = impl::expr_chain(exprs_, fmis, context);

    // Set all right hand sides to their corresponding left hand sides.
    set_impl(all_values, std::make_index_sequence<2 * num_pairs>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION void cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(!aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
                  "The cache somehow contains our tag, but our eval returns void and should never cache anything.");

    // Eval all expressions, storing their results for later.
    auto [all_values, final_cache] = impl::cached_expr_chain<EvalCountsType, eval_counts>(
        exprs_, fmis, std::forward<OldCacheType>(old_cache), context);

    // Set all right hand sides to their corresponding left hand sides.
    set_impl(all_values, final_cache, std::make_index_sequence<2 * num_pairs>{});
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(impl::RuntimeReuseValidator& validator) const {
    validate_runtime_reuse_impl<EvalCountsType, eval_counts>(std::make_index_sequence<sizeof...(TrgSrcExprPairs)>{},
                                                             validator);
  }

 private:
  template <typename EvalCountsType, EvalCountsType eval_counts, size_t... Is>
  void validate_runtime_reuse_impl(std::index_sequence<Is...>, impl::RuntimeReuseValidator& validator) const {
    (get<Is>(exprs_).template validate_runtime_reuse<EvalCountsType, eval_counts>(validator), ...);
  }

 public:
  void flag_read_only(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to read the return type of an assignment expression, which returns void.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Attempting to write to the return type of an assignment expression, which returns void.");
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    propagate_synchronize_impl(std::make_index_sequence<num_pairs>{}, context);
  }

  const auto driver() const {
    // TODO(palmerb4): Check that all drivers are the same.
    return get<0>(exprs_).driver();
  }

  //  private:
  template <typename AllValuesType, size_t... Is>
  KOKKOS_INLINE_FUNCTION static void set_impl(AllValuesType& all_values, std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == 2 * num_pairs, "Index sequence size must match number of target + source exprs.");
    (set_i_impl<Is>(all_values), ...);
  }

  template <typename AllValuesType, typename CacheType, size_t... Is>
  KOKKOS_INLINE_FUNCTION static void set_impl(AllValuesType& all_values, CacheType& cache, std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == 2 * num_pairs, "Index sequence size must match number of target + source exprs.");
    (set_i_impl<Is>(all_values, cache), ...);
  }

  template <size_t I, typename AllValuesType>
  KOKKOS_INLINE_FUNCTION static void set_i_impl(AllValuesType& all_values) {
    if constexpr (I % 2 == 0) {
      auto&& trg_ref = get<I>(all_values);
      auto&& src_ref = get<I + 1>(all_values);
      trg_ref = src_ref;
    }
  }

  template <size_t I, typename AllValuesType, typename CacheType>
  KOKKOS_INLINE_FUNCTION static void set_i_impl(AllValuesType& all_values, CacheType& cache) {
    if constexpr (I % 2 == 0) {
      auto&& trg_ref = get<I>(all_values).get(cache);
      auto&& src_ref = get<I + 1>(all_values).get(cache);
      trg_ref = src_ref;
    }
  }

  template <size_t... Is, typename Ctx>
  void propagate_synchronize_impl(std::index_sequence<Is...>, const Ctx& context) {
    static_assert(sizeof...(Is) == num_pairs, "Index sequence size must match number of target/source pairs.");

    // Flag all right hand sides as read-only and all left hand sides as overwrite-all.
    (get<2 * Is + 1>(exprs_).flag_read_only(context), ...);
    (get<2 * Is>(exprs_).flag_overwrite_all(context), ...);

    // Propagate synchronize to all expressions.
    (get<2 * Is + 1>(exprs_).propagate_synchronize(context), ...);
    (get<2 * Is>(exprs_).propagate_synchronize(context), ...);
  }

  tuple<TrgSrcExprPairs...> exprs_;
};

/// \brief Perform a fused assignment operation
/// fused_assign(
//       trg_expr1, /*=*/ src_expr1,
///      trg_expr2, /*=*/ src_expr2,
///               ...
///      trg_exprN, /*=*/ src_exprN);
template <typename... TrgSrcExprPairs>
void fused_assign(const TrgSrcExprPairs&... exprs) {
  constexpr size_t num_trg_src_pairs = sizeof...(TrgSrcExprPairs);
  static_assert(num_trg_src_pairs % 2 == 0,
                "The number of target/source expression pairs in fused_assign must be even.");
  FusedAssignExpr<TrgSrcExprPairs...> fused_expr(exprs...);
  fused_expr.driver()->run(fused_expr);
}

/// \brief Reduces value of a given expression over all entities in the driver on this process
template <typename Expr, typename ReductionOp>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
void reduce_local(Expr&& expr, ReductionOp& reduction) {
  auto driver = expr.driver();
  driver->reduce_local(expr, reduction);
}

/// \brief Reduce sum (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_sum(Expr&& expr) {
  Scalar local_sum = 0;
  Kokkos::Sum<Scalar> sum_reduction(local_sum);
  reduce_local(std::forward<Expr>(expr), sum_reduction);
  return local_sum;
}

/// \brief Reduce max (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_max(Expr&& expr) {
  Scalar local_max;
  Kokkos::Max<Scalar> max_reduction(local_max);
  reduce_local(std::forward<Expr>(expr), max_reduction);
  return local_max;
}

/// \brief Reduce min (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_min(Expr&& expr) {
  Scalar local_min;
  Kokkos::Min<Scalar> min_reduction(local_min);
  reduce_local(std::forward<Expr>(expr), min_reduction);
  return local_min;
}

/// \brief Reduces sum (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_sum(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_sum = reduce_local_sum<Scalar>(std::forward<Expr>(expr));
  Scalar global_sum = 0;
  stk::all_reduce_sum(driver->bulk_data().parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

/// \brief Reduces max (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_max(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_max = reduce_local_max<Scalar>(std::forward<Expr>(expr));
  Scalar global_max = 0;
  stk::all_reduce_max(driver->bulk_data().parallel(), &local_max, &global_max, 1);
  return global_max;
}

/// \brief Reduces min (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_min(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_min = reduce_local_min<Scalar>(std::forward<Expr>(expr));
  Scalar global_min = 0;
  stk::all_reduce_min(driver->bulk_data().parallel(), &local_min, &global_min, 1);
  return global_min;
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPACCESSOREXPR_HPP_
