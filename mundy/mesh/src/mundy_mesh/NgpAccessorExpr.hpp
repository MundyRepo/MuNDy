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
  -accessor(entity_expr) = expr; 
  -accessor(entity_expr) += expr; 
  -fused_assign(accessor1, expr1, accessor2, expr2, ...); 
  -auto result = reduce_op(expr);


Upon evaluation, but before looping over the entities, all fields involved in the expression are synchronized to the
appropriate space and marked modified where necessary. The expression tree "knows" which fields are read and written.


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

// How would all of this look in a for_each_entity_expr_eval
template <typename EntityExpr>
void for_each_entity_evaluate_expr(const EntityExprBase<EntityExpr> &entity_expr) {
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
        entity_expr.eval(entity_index, evaluation_context);
      });
}

//! \name Entity expressions (those whose eval returns an entity)
//@{

template <typename DerivedEntityExpr>
class EntityExprBase {
 public:
  using our_t = EntityExprBase<NumEntities, DerivedEntityExpr>;
  constexpr size_t num_entities = DerivedEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedEntityExpr &self() const noexcept {
    return static_cast<const DerivedEntityExpr &>(*this);
  }

  template <class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(
      const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const Ctx &context) const {
    return self().eval(fmas, context);
  }

  template <class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fma, const Ctx &context) const
    requires(num_entities == 1)
  {
    return self().eval(fma, context);
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
  }
};

template <typename PrevEntityExpr>
class ConnectedEntitiesExpr : public EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>> {
 public:
  using our_t = ConnectedEntitiesExpr<NumEntities, PrevEntityExpr>;
  constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(PrevEntityExpr prev_entity_expr, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas,
                                const NgpContext &context) const {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmas, context);
    return context.ngp_mesh().get_connected_entities(entity_index, rank_);
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fma, context);
    return context.ngp_mesh().get_connected_entities(entity_index, rank_);
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<NumEntities, our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext &context) {
    prev_entity_expr_.propagate_synchronize(context);
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  stk::mesh::EntityRank conn_rank_;
};

template <size_t NumEntities, size_t Ord>
class CTimeEntityExpr : public EntityExprBase<CTimeEntityExpr<NumEntities, Ord>> {
 public:
  using our_t = CTimeEntityExpr<NumEntities, Ord>;
  constexpr size_t num_entities = NumEntities;

  KOKKOS_DEFAULTED_FUNCTION
  CTimeEntityExpr() = default;

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> fmas,
                                const NgpContext & /*context*/) const {
    static_assert(Ord < NumEntities, "CTimeEntityExpr ordinal must be less than NumEntities");
    return fmas[ordinal_];
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex fma, const NgpContext & /*context*/) const
    requires(num_entities == 1)
  {
    static_assert(Ord == 0, "CTimeEntityExpr with a single entity must have Ord == 0");
    return fma;
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<NumEntities, our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
  }

 private:
  constexpr size_t ordinal_ = Ord;
};

template <size_t NumEntities>
class RTimeEntityExpr : public EntityExprBase<RTimeEntityExpr<NumEntities>> {
 public:
  using our_t = RTimeEntityExpr<NumEntities>;
  constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  RTimeEntityExpr(size_t ordinal) : ordinal_(ordinal) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas,
                                const NgpContext & /*context*/) const {
    MUNDY_THROW_ASSERT(ordinal_ < NumEntities, std::out_of_range,
                       "RTimeEntityExpr ordinal must be less than NumEntities");
    return fmas[ordinal_];
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fma, const NgpContext & /*context*/) const
    requires(num_entities == 1)
  {
    MUNDY_THROW_ASSERT(ordinal_ == 0, std::out_of_range, "RTimeEntityExpr with a single entity must have Ord == 0");
    return fma;
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<NumEntities, our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
  }

 private:
  size_t ordinal_;
};
//@}

//! \name Views of mathematical expressions
//@{

template <typename DerivedMathExpr>
class MathExprBase {
 public:
  using our_t = MathExprBase<DerivedMathExpr>;
  constexpr size_t num_entities = DerivedMathExpr::num_entities;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr MathExprBase() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedMathExpr &self() const noexcept {
    return static_cast<const DerivedMathExpr &>(*this);
  }

  template <class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas,
                                   const Ctx &context) const {
    return self().eval(fmas, context);
  }

  template <class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const stk::mesh::FastMeshIndex &fma, const Ctx &context) const
    requires(num_entities == 1)
  {
    return self().eval(fma, context);
  }

  template <class Ctx>
  void propagate_synchronize(const Ctx &context) {
    self().propagate_synchronize(context);
  }
};

template <typename PrevEntityExpr, typename AccessorT>
class AccessorExpr : public MathExprBase<AccessorExpr<PrevEntityExpr, AccessorT>> {
 public:
  using our_t = AccessorExpr<NumEntities, PrevEntityExpr, AccessorT>;
  constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AccessorExpr(PrevEntityExpr prev_entity_expr, AccessorT accessor)
      : prev_entity_expr_(prev_entity_expr), accessor_(accessor) {
  }

  KOKKOS_INLINE_FUNCTION
  auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmas, context);
    return accessor_(entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  auto eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const
    requires(num_entities == 1)
  {
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fma, context);
    return accessor_(entity_index);
  }

  void read_only(const NgpContext & /*context*/) {
    accessor_.sync_to_device();
  }

  void read_write(const NgpContext & /*context*/) {
    accessor_.sync_to_device();
    accessor_.modify_on_device();
  }

  void overwrite(const NgpContext & /*context*/) {
    accessor_.clear_host_sync_state();
    accessor_.modify_on_device();
  }

  void propagate_synchronize(const NgpContext & /*context*/) {
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  AccessorT accessor_;
};

template <typename TargetAccessorExpr, typename SourceMathExpr>
class AssignMathExpr {
 public:
  using our_t = AssignExpr<TargetAccessorExpr, SourceMathExpr>;
  constexpr size_t num_entities = TargetAccessorExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AssignExpr(TargetAccessorExpr trg_accessor_expr, SourceMathExpr src_math_expr)
      : trg_accessor_expr_(trg_accessor_expr), src_math_expr_(src_math_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const {
    // The left hand side is a view into the accessor data, so we can assign to it directly
    trg_accessor_expr_.eval(fmas, context) = src_math_expr_.eval(fmas, context);
  }

  void propagate_synchronize(const NgpContext &context) {
    trg_accessor_expr_.overwrite(context);
    trg_accessor_expr_.propagate_synchronize(context);
    src_math_expr_.propagate_synchronize(context);
  }

 private:
  TargetAccessorExpr trg_accessor_expr_;
  SourceMathExpr src_math_expr_;
};

template <typename TargetAccessorExpr, typename SourceEntityExpr>
class AssignEntityExpr {
 public:
  using our_t = AssignEntityExpr<TargetAccessorExpr, SourceEntityExpr>;
  constexpr size_t num_entities = TargetAccessorExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  AssignEntityExpr(TargetAccessorExpr trg_accessor_expr, SourceEntityExpr src_entity_expr)
      : trg_accessor_expr_(trg_accessor_expr), src_entity_expr_(src_entity_expr) {
  }

  KOKKOS_INLINE_FUNCTION
  void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const {
    // The left hand side is a view into the accessor data, so we can assign to it directly
    trg_accessor_expr_.eval(fmas, context) = src_entity_expr_.eval(fmas, context);
  }

  void propagate_synchronize(const NgpContext &context) {
    trg_accessor_expr_.overwrite(context);
    trg_accessor_expr_.propagate_synchronize(context);
    src_entity_expr_.propagate_synchronize(context);
  }

 private:
  TargetAccessorExpr trg_accessor_expr_;
  SourceEntityExpr src_entity_expr_;
};

/// \brief Calling assign(accessor_expr, /* = */ math_expr) creates an assignment expression for delayed evaluation
template <typename TargetPrevEntityExpr, typename TargetAccessorT, typename SourceMathExpr>
auto assign(AccessorExpr<TargetPrevEntityExpr, TargetAccessorT> trg, MathExprBase<SourceMathExpr> src) {
  static_assert(TargetPrevEntityExpr::num_entities == SourceMathExpr::num_entities,
                "Mismatched number of entities in assign");
  return AssignMathExpr<AccessorExpr<TargetPrevEntityExpr, TargetAccessorT>, MathExprBase<SourceMathExpr>>(trg, src);
}

/// \brief Calling assign(accessor_expr, /* = */ entity_expr) creates an assignment expression for delayed evaluation
template <typename TargetPrevEntityExpr, typename TargetAccessorT, typename SourceEntityExpr>
auto assign(AccessorExpr<TargetPrevEntityExpr, TargetAccessorT> trg, EntityExprBase<SourceEntityExpr> src) {
  static_assert(TargetPrevEntityExpr::num_entities == SourceEntityExpr::num_entities,
                "Mismatched number of entities in assign");
  return AssignEntityExpr<AccessorExpr<TargetPrevEntityExpr, TargetAccessorT>, EntityExprBase<SourceEntityExpr>>(trg,
                                                                                                                 src);
}

/// \brief Calling operator= on an accessor expression evaluates the expression and assigns it to the accessor
template <typename TargetPrevEntityExpr, typename TargetAccessorT, typename SourceMathExpr>
void operator=(AccessorExpr<TargetPrevEntityExpr, TargetAccessorT> trg, MathExprBase<SourceMathExpr> src)
  requires(Expr1::num_entities == 1)
{
  auto assignment_expr = assign(trg, /* = */ src);
  for_each_entity_evaluate_expr(assignment_expr);
}

/// \brief Calling operator= on an accessor expression evaluates the expression and assigns it to the accessor
template <typename TargetPrevEntityExpr, typename TargetAccessorT, typename SourceEntityExpr>
void operator=(AccessorExpr<TargetPrevEntityExpr, TargetAccessorT> trg, EntityExprBase<SourceEntityExpr> src)
  requires(Expr1::num_entities == 1)
{
  auto assignment_expr = assign(trg, /* = */ src);
  for_each_entity_evaluate_expr(assignment_expr);
}

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

#define MUNDY_MATH_EXPR_PAIRWISE_OP(OpName, op)                                                                     \
  template <typename LeftMathExpr, typename RightMathExpr>                                                          \
  class OpName##Expr : public MathExprBase<OpName##Expr<LeftMathExpr, RightMathExpr>> {                             \
   public:                                                                                                          \
    using our_t = OpName##Expr<LeftMathExpr, RightMathExpr>;                                                        \
    constexpr size_t num_entities = NumEntities;                                                                    \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    OpName##Expr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {                             \
    }                                                                                                               \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const { \
      return left_.eval(fmas, context) op right_.eval(fmas, context);                                               \
    }                                                                                                               \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    auto eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const                                 \
      requires(num_entities == 1)                                                                                   \
    {                                                                                                               \
      return left_.eval(fma, context) op right_.eval(fma, context);                                                 \
    }                                                                                                               \
                                                                                                                    \
    void propagate_synchronize(const NgpContext &context) {                                                         \
      left_.read_only(context);                                                                                     \
      right_.read_only(context);                                                                                    \
      left_.propagate_synchronize(context);                                                                         \
      right_.propagate_synchronize(context);                                                                        \
    }                                                                                                               \
                                                                                                                    \
   private:                                                                                                         \
    LeftMathExpr left_;                                                                                             \
    RightMathExpr right_;                                                                                           \
  };                                                                                                                \
                                                                                                                    \
  template <size_t NumEntities, typename Expr1, typename Expr2>                                                     \
  auto operator##op(MathExprBase<Expr1> e1, MathExprBase<Expr2> e2) {                                               \
    return OpName##Expr<MathExprBase<Expr1>, MathExprBase<Expr2>>(e1, e2);                                          \
  }

#define MUNDY_MATH_EXPR_PAIRWISE_OP_EQUALS(OpName, op_equals)                                                       \
  template <typename LeftMathExpr, typename RightMathExpr>                                                          \
  class OpName##EqualsExpr : public MathExprBase<OpName##EqualsExpr<LeftMathExpr, RightMathExpr>> {                 \
   public:                                                                                                          \
    using our_t = OpName##EqualsExpr<LeftMathExpr, RightMathExpr>;                                                  \
    constexpr size_t num_entities = NumEntities;                                                                    \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    OpName##EqualsExpr(LeftMathExpr left, RightMathExpr right) : left_(left), right_(right) {                       \
    }                                                                                                               \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    void eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const { \
      left_.eval(fmas, context) op_equals = right_.eval(fmas, context);                                             \
    }                                                                                                               \
                                                                                                                    \
    KOKKOS_INLINE_FUNCTION                                                                                          \
    void eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const                                 \
      requires(num_entities == 1)                                                                                   \
    {                                                                                                               \
      left_.eval(fma, context) op_equals right_.eval(fma, context);                                                 \
    }                                                                                                               \
                                                                                                                    \
    void propagate_synchronize(const NgpContext &context) {                                                         \
      left_.read_write(context);                                                                                    \
      right_.read_only(context);                                                                                    \
      left_.propagate_synchronize(context);                                                                         \
      right_.propagate_synchronize(context);                                                                        \
    }                                                                                                               \
                                                                                                                    \
   private:                                                                                                         \
    LeftMathExpr left_;                                                                                             \
    RightMathExpr right_;                                                                                           \
  };                                                                                                                \
                                                                                                                    \
  template <size_t NumEntities, typename Expr1, typename Expr2>                                                     \
  void operator##op =(MathExprBase<Expr1> e1, MathExprBase<Expr2> e2) {                                             \
    auto expr = OpName##Expr<MathExprBase<Expr1>, MathExprBase<Expr2>>(e1, e2);                                     \
    for_each_entity_evaluate_expr(expr);                                                                            \
  }

MUNDY_MATH_EXPR_PAIRWISE_OP(Add, +)
MUNDY_MATH_EXPR_PAIRWISE_OP(Sub, -)
MUNDY_MATH_EXPR_PAIRWISE_OP(Mul, *)
MUNDY_MATH_EXPR_PAIRWISE_OP(Div, /)
MUNDY_MATH_EXPR_PAIRWISE_OP_EQUALS(Add, +=)
MUNDY_MATH_EXPR_PAIRWISE_OP_EQUALS(Sub, -=)
MUNDY_MATH_EXPR_PAIRWISE_OP_EQUALS(Mul, *=)
MUNDY_MATH_EXPR_PAIRWISE_OP_EQUALS(Div, /=)
//@}

//! \name Helpers
//@{

/// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
/// Example:
///   auto v3_accessor = Vector3FieldComponent(v3_field);
///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
///   auto get_v3_expr = v3_accessor(all_nodes);
template <typename AccessorT, typename EntityExpr>
AccessorExpr<EntityExpr, AccessorT> operator()(AccessorT accessor, EntityExpr e) {
  return AccessorExpr<EntityExpr, AccessorT>(e, accessor);
}

template <typename PrevEntityExpr>
class ReuseEntityExpr : public EntityExprBase<ReuseEntityExpr<PrevEntityExpr>> {
 public:
  using our_t = ReuseEntityExpr<NumEntities, PrevEntityExpr>;
  using stored_value_t = stk::mesh::FastMeshIndex;
  constexpr size_t num_entities = PrevEntityExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  ReuseEntityExpr(PrevEntityExpr prev_entity_expr)
      : prev_entity_expr_(prev_entity_expr), evaluated_(false), stored_value_() {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas,
                                const NgpContext &context) const {
    if (!evaluated_) {
      stored_value_ = prev_entity_expr_.eval(fmas, context);
      evaluated_ = true;
    }
    return stored_value_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const
    requires(num_entities == 1)
  {
    if (!evaluated_) {
      stored_value_ = prev_entity_expr_.eval(fma, context);
      evaluated_ = true;
    }
    return stored_value_;
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
  using our_t = ReuseMathExpr<NumEntities, PrevMathExpr, AccessorT>;
  using stored_value_t = decltype(std::declval<PrevMathExpr>().eval(
      std::declval<Kokkos::Array<stk::mesh::FastMeshIndex, PrevMathExpr::num_entities>>(), std::declval<NgpContext>()));
  constexpr size_t num_entities = PrevMathExpr::num_entities;

  KOKKOS_INLINE_FUNCTION
  ReuseMathExpr(PrevMathExpr prev_math_expr) : prev_math_expr_(prev_math_expr), evaluated_(false), stored_value_() {
  }

  KOKKOS_INLINE_FUNCTION
  auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> &fmas, const NgpContext &context) const {
    if (!evaluated_) {
      stored_value_ = prev_math_expr_.eval(fmas, context);
      evaluated_ = true;
    }
    return stored_value_;
  }

  KOKKOS_INLINE_FUNCTION
  auto eval(const stk::mesh::FastMeshIndex &fma, const NgpContext &context) const
    requires(num_entities == 1)
  {
    if (!evaluated_) {
      stored_value_ = prev_math_expr_.eval(fma, context);
      evaluated_ = true;
    }
    return stored_value_;
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
