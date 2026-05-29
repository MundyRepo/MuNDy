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
/// \brief Expression-template layer for NGP field operations.
///
/// Accessor expressions let you write mesh-field arithmetic against plain math objects without manually
/// managing field synchronization, kernel boundaries, or per-entity loops. Expressions are lazy—nothing
/// runs until an accessor expression appears as an lvalue or is passed to a reduction.
///
/// See the \ref MundyMesh "MundyMesh Primer" for a full walkthrough, examples, and the complete function reference.

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
#include <mundy_mesh/impl/NgpAccessorExprAccessor.hpp>           // AccessorExpr
#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>         // ApplyValueExpr, impl helpers
#include <mundy_mesh/impl/NgpAccessorExprAssign.hpp>             // AssignExpr
#include <mundy_mesh/impl/NgpAccessorExprBinaryValue.hpp>        // BinaryValueExpr, Add/Sub/Mul/DivExpr, operators
#include <mundy_mesh/impl/NgpAccessorExprBuiltins.hpp>           // standard wrappers (norm, dot, etc.)
#include <mundy_mesh/impl/NgpAccessorExprCachable.hpp>           // CachableExprBase
#include <mundy_mesh/impl/NgpAccessorExprConnectedEntities.hpp>  // ConnectedEntitiesExpr
#include <mundy_mesh/impl/NgpAccessorExprConstant.hpp>           // ConstantMathExpr
#include <mundy_mesh/impl/NgpAccessorExprDrivers.hpp>     // NgpForEachEntityExprDriver, NgpForEachEntityPairExprDriver
#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>  // EntityExprBase
#include <mundy_mesh/impl/NgpAccessorExprEntityExpr.hpp>  // EntityExpr
#include <mundy_mesh/impl/NgpAccessorExprFused.hpp>       // FusedAssignExpr
#include <mundy_mesh/impl/NgpAccessorExprMathBase.hpp>    // MathExprBase
#include <mundy_mesh/impl/NgpAccessorExprRNG.hpp>  // RandomDistributionExpr, UniformDistributionExpr, CounterBasedRNGExpr
#include <mundy_mesh/impl/NgpAccessorExprReductions.hpp>  // reduce helpers
#include <mundy_mesh/impl/NgpAccessorExprSink.hpp>   // SinkArg, ApplySinkExpr, sink_expr_impl, BinarySideEffectExpr
#include <mundy_mesh/impl/NgpAccessorExprTypes.hpp>  // NgpEvalContext, holder types
#include <mundy_mesh/impl/NgpAccessorExprUtils.hpp>  // is_crtp_base_of, is_crtp_base_of_v

namespace mundy {

namespace mesh {

//! \name Entity expression factories
//@{

/// \brief Create an entity expression for iterating over entities of a given rank in a selector.
template <typename ExecSpace = stk::ngp::ExecSpace>
auto make_entity_expr(stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                      const stk::mesh::EntityRank& rank, const ExecSpace& exec_space = ExecSpace()) {
  return impl::make_entity_expr_impl(bulk_data, selector, rank, exec_space);
}

/// \brief Create a pairwise entity expression for iterating over entity pairs defined by a pair view.
template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
auto make_pairwise_entity_expr(stk::mesh::BulkData& bulk_data,                                //
                               const stk::mesh::EntityRank& left_rank,                        //
                               const stk::mesh::EntityRank& right_rank,                       //
                               const PairView& pair_view, const FMIExtractor& fmi_extractor,  //
                               const ExecSpace& exec_space = ExecSpace()) {
  return impl::make_pairwise_entity_expr_impl(bulk_data, left_rank, right_rank, pair_view, fmi_extractor, exec_space);
}
//@}

//! \name Value expressions
//@{

/// \brief Build a read-only value expression by applying a function object to expression arguments.
template <typename Func, typename... Args>
auto apply_expr(Func func, const Args&... args) {
  return impl::apply_expr_impl(std::move(func), args...);
}
//@}

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

/// \brief Build a side-effect expression that applies a mutating function object to expression arguments.
template <typename Func, typename... Args>
auto sink_expr(Func func, const Args&... args) {
  return impl::sink_expr_impl(std::move(func), args...);
}

/// \brief Atomically add rhs to each element of the target expression.
template <typename... Args>
auto atomic_add(const Args&... args) {
  return impl::atomic_add_impl(args...);
}

/// \brief Atomically subtract rhs from each element of the target expression.
template <typename... Args>
auto atomic_sub(const Args&... args) {
  return impl::atomic_sub_impl(args...);
}

/// \brief Atomically multiply each element of the target expression by rhs.
template <typename... Args>
auto atomic_mul(const Args&... args) {
  return impl::atomic_mul_impl(args...);
}

/// \brief Atomically divide each element of the target expression by rhs.
template <typename... Args>
auto atomic_div(const Args&... args) {
  return impl::atomic_div_impl(args...);
}
//@}

//! \name RNG factory functions
//@{

/// \brief Create a counter-based random number generator using the given seed and counter.
/// At least one of seed or counter must be an expression for this code to compile.
template <typename SeedExpr, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
auto rng(const SeedExpr& seed_expr, const CounterExpr& counter_expr) {
  return impl::rng_impl<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}
//@}

//! \name Fused assignment
//@{

/// \brief Evaluate all RHS expressions before writing any LHS—simultaneous multi-target assignment in one kernel.
template <typename... TrgSrcExprPairs>
void fused_assign(const TrgSrcExprPairs&... exprs) {
  impl::fused_assign_impl(exprs...);
}
//@}

//! \name Reduction operations
//@{

/// \brief Reduces value of a given expression over all entities in the driver on this process
template <typename Expr, typename ReductionOp>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
void reduce_local(Expr&& expr, ReductionOp& reduction) {
  impl::reduce_local_impl(std::forward<Expr>(expr), reduction);
}

/// \brief Reduce sum (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_sum(Expr&& expr) {
  return impl::reduce_local_sum_impl<Scalar>(std::forward<Expr>(expr));
}

/// \brief Reduce max (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_max(Expr&& expr) {
  return impl::reduce_local_max_impl<Scalar>(std::forward<Expr>(expr));
}

/// \brief Reduce min (process local)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto reduce_local_min(Expr&& expr) {
  return impl::reduce_local_min_impl<Scalar>(std::forward<Expr>(expr));
}

/// \brief Reduces sum (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto all_reduce_sum(Expr&& expr) {
  return impl::all_reduce_sum_impl<Scalar>(std::forward<Expr>(expr));
}

/// \brief Reduces max (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto all_reduce_max(Expr&& expr) {
  return impl::all_reduce_max_impl<Scalar>(std::forward<Expr>(expr));
}

/// \brief Reduces min (all processes)
template <typename Scalar, typename Expr>
MUNDY_REQUIRES(impl::is_crtp_base_of_v<impl::MathExprBase, Expr> || impl::is_crtp_base_of_v<impl::EntityExprBase, Expr>)
auto all_reduce_min(Expr&& expr) {
  return impl::all_reduce_min_impl<Scalar>(std::forward<Expr>(expr));
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPACCESSOREXPR_HPP_
