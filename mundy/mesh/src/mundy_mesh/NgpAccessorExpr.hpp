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
#include <mundy_mesh/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <mundy_utils/StringLiteral.hpp>  // for mundy::StringLiteral
#include <mundy_utils/aggregate.hpp>      // for mundy::aggregate
#include <mundy_utils/requires.hpp>
#include <mundy_utils/rng.hpp>           // for mundy::make_philox
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
#include <mundy_utils/tuple.hpp>         // for mundy::tuple

// Implementation headers — included in dependency order.
// Each header includes its own prerequisites, so explicit ordering here is for documentation.
#include <mundy_mesh/impl/NgpAccessorExprUtils.hpp>      // is_crtp_base_of, is_crtp_base_of_v
#include <mundy_mesh/impl/NgpAccessorExprTypes.hpp>      // NgpEvalContext, holder types
#include <mundy_mesh/impl/NgpAccessorExprCachable.hpp>   // CachableExprBase
#include <mundy_mesh/impl/NgpAccessorExprMathBase.hpp>   // MathExprBase
#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>  // EntityExprBase
#include <mundy_mesh/impl/NgpAccessorExprConnectedEntities.hpp>  // ConnectedEntitiesExpr
#include <mundy_mesh/impl/NgpAccessorExprEntityExpr.hpp>  // EntityExpr
#include <mundy_mesh/impl/NgpAccessorExprDrivers.hpp>    // NgpForEachEntityExprDriver, NgpForEachEntityPairExprDriver
#include <mundy_mesh/impl/NgpAccessorExprConstant.hpp>   // ConstantMathExpr
#include <mundy_mesh/impl/NgpAccessorExprAssign.hpp>     // AssignExpr
#include <mundy_mesh/impl/NgpAccessorExprBinaryValue.hpp>  // BinaryValueExpr, Add/Sub/Mul/DivExpr, operators
#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>   // ApplyValueExpr, impl helpers
#include <mundy_mesh/impl/NgpAccessorExprSink.hpp>         // SinkArg, ApplySinkExpr, sink_expr_impl, BinarySideEffectExpr
#include <mundy_mesh/impl/NgpAccessorExprAccessor.hpp>     // AccessorExpr
#include <mundy_mesh/impl/NgpAccessorExprRNG.hpp>          // RandomDistributionExpr, UniformDistributionExpr, CounterBasedRNGExpr
#include <mundy_mesh/impl/NgpAccessorExprFused.hpp>        // FusedAssignExpr

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

//! \name Entity expression factories
//@{

template <typename ExecSpace = stk::ngp::ExecSpace>
auto make_entity_expr(stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                      const stk::mesh::EntityRank& rank, const ExecSpace& /*exec_space*/ = ExecSpace()) {
  // To ensure that all expressions have the same driver, we store a persistent driver manager
  // on the meta data and use it to memoize the driver for the given rank and selector.

  using driver_t = impl::NgpForEachEntityExprDriver<ExecSpace>;
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

  return impl::EntityExpr<1, 0, driver_t>(rank, driver_ptr);
}

template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
auto make_pairwise_entity_expr(stk::mesh::BulkData& bulk_data,                                    //
                               const stk::mesh::EntityRank& left_rank,                            //
                               const stk::mesh::EntityRank& right_rank,                           //
                               const PairView& pair_view, const FMIExtractor& /*fmi_extractor*/,  //
                               const ExecSpace& /*exec_space*/ = ExecSpace()) {
  using driver_t = impl::NgpForEachEntityPairExprDriver<PairView, FMIExtractor, ExecSpace>;
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

  return impl::EntityPair(left_rank, right_rank, driver_ptr);
}
//@}

//! \name Value expressions
//@{

/// \brief Build a read-only value expression that applies an arbitrary function object to expression arguments.
///
/// Usage: `apply_expr(func, expr1, expr2, ...)` where at least one argument is a math expression.
/// Scalar values are automatically wrapped in `ConstantMathExpr`.
template <typename Func, typename... Args>
auto apply_expr(Func func, const Args&... args) {
  static_assert(sizeof...(Args) > 0, "apply_expr(func, args...): at least one argument is required.");
  static_assert((impl::is_math_expr_arg_v<Args> || ...),
                "apply_expr(func, args...): at least one argument must be a math expression so Mundy knows which "
                "entity driver should evaluate the expression. Scalars are allowed, but they cannot be the only "
                "arguments.");
  return impl::ApplyValueExpr<std::decay_t<Func>, decltype(impl::make_apply_expr_arg(args))...>(
      std::move(func), impl::make_apply_expr_arg(args)...);
}
//@}

}  // namespace mesh

}  // namespace mundy

#include <mundy_mesh/impl/NgpAccessorExprBuiltins.hpp>  // FORWARD_FUNC/SINK macros, standard wrappers (norm, dot, etc.)

namespace mundy {

namespace mesh {

//! \name Sink expressions
//@{

/// \brief Wrap an argument as read-only for use with sink_expr().
template <typename Arg>
KOKKOS_INLINE_FUNCTION auto read_only(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::ReadOnly>(arg);
}

/// \brief Wrap an argument as read-write for use with sink_expr().
template <typename Arg>
KOKKOS_INLINE_FUNCTION auto read_write(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::ReadWrite>(arg);
}

/// \brief Wrap an argument as overwrite-all for use with sink_expr().
template <typename Arg>
KOKKOS_INLINE_FUNCTION auto overwrite_all(const Arg& arg) {
  return impl::make_sink_arg_with_mode<impl::SinkAccessMode::OverwriteAll>(arg);
}

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
  return impl::sink_expr_impl(std::move(func), args...);
}

template <typename... Args>
auto atomic_add(const Args&... args) {
  return impl::atomic_add(args...);
}

template <typename... Args>
auto atomic_sub(const Args&... args) {
  return impl::atomic_sub(args...);
}

template <typename... Args>
auto atomic_mul(const Args&... args) {
  return impl::atomic_mul(args...);
}

template <typename... Args>
auto atomic_div(const Args&... args) {
  return impl::atomic_div(args...);
}
//@}

//! \name RNG factory functions
//@{

/// \brief Create a counter-based random number generator using the given seed and counter.
/// Seed and counter are expressions.
template <typename SeedExpr, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, SeedExpr>&&
                   impl::is_crtp_base_of_v<impl::MathExprBase, CounterExpr>)
auto rng(const SeedExpr& seed_expr, const CounterExpr& counter_expr) {
  return impl::CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Seed is an expression but counter is a constant.
template <typename SeedExpr, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, SeedExpr> &&
                   !impl::is_crtp_base_of_v<impl::MathExprBase, CounterT>)
auto rng(const SeedExpr& seed_expr, const CounterT& counter) {
  using CounterExpr = impl::ConstantMathExpr<CounterT>;
  auto counter_expr = CounterExpr(counter);
  return rng<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Seed is a constant but counter is an expression.
template <typename SeedT, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!impl::is_crtp_base_of_v<impl::MathExprBase, SeedT> &&
                   impl::is_crtp_base_of_v<impl::MathExprBase, CounterExpr>)
auto rng(const SeedT& seed, const CounterExpr& counter_expr) {
  using SeedExpr = impl::ConstantMathExpr<SeedT>;
  auto seed_expr = SeedExpr(seed);
  return rng<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
/// Both seed and counter are constants (not allowed).
template <typename SeedT, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!impl::is_crtp_base_of_v<impl::MathExprBase, SeedT> &&
                   !impl::is_crtp_base_of_v<impl::MathExprBase, CounterT>)
void rng(const SeedT& /*seed*/, const CounterT& /*counter*/) {
  MUNDY_THROW_REQUIRE(false, std::logic_error,
                      "Both seed and counter arguments to rng() cannot be constants.\n"
                      "At least one of them must be an expression, lest we have no idea how to run the expression over "
                      "multiple entities.");
}
//@}

//! \name Fused assignment
//@{

/// \brief Perform a fused assignment operation.
///
/// fused_assign(
///      trg_expr1, /*=*/ src_expr1,
///      trg_expr2, /*=*/ src_expr2,
///               ...
///      trg_exprN, /*=*/ src_exprN);
template <typename... TrgSrcExprPairs>
void fused_assign(const TrgSrcExprPairs&... exprs) {
  constexpr size_t num_trg_src_pairs = sizeof...(TrgSrcExprPairs);
  static_assert(num_trg_src_pairs % 2 == 0,
                "The number of target/source expression pairs in fused_assign must be even.");
  impl::FusedAssignExpr<TrgSrcExprPairs...> fused_expr(exprs...);
  fused_expr.driver()->run(fused_expr);
}
//@}

//! \name Reduction operations
//@{

/// \brief Reduces value of a given expression over all entities in the driver on this process
template <typename Expr, typename ReductionOp>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
void reduce_local(Expr&& expr, ReductionOp& reduction) {
  auto driver = expr.driver();
  driver->reduce_local(expr, reduction);
}

/// \brief Reduce sum (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_sum(Expr&& expr) {
  Scalar local_sum = 0;
  Kokkos::Sum<Scalar> sum_reduction(local_sum);
  reduce_local(std::forward<Expr>(expr), sum_reduction);
  return local_sum;
}

/// \brief Reduce max (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_max(Expr&& expr) {
  Scalar local_max;
  Kokkos::Max<Scalar> max_reduction(local_max);
  reduce_local(std::forward<Expr>(expr), max_reduction);
  return local_max;
}

/// \brief Reduce min (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_min(Expr&& expr) {
  Scalar local_min;
  Kokkos::Min<Scalar> min_reduction(local_min);
  reduce_local(std::forward<Expr>(expr), min_reduction);
  return local_min;
}

/// \brief Reduces sum (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto all_reduce_sum(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_sum = reduce_local_sum<Scalar>(std::forward<Expr>(expr));
  Scalar global_sum = 0;
  stk::all_reduce_sum(driver->bulk_data().parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

/// \brief Reduces max (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto all_reduce_max(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_max = reduce_local_max<Scalar>(std::forward<Expr>(expr));
  Scalar global_max = 0;
  stk::all_reduce_max(driver->bulk_data().parallel(), &local_max, &global_max, 1);
  return global_max;
}

/// \brief Reduces min (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> ||
                   impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
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
