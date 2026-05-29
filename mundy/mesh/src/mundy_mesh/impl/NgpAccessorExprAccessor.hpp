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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRACCESSOR_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRACCESSOR_HPP_

/// \file NgpAccessorExprAccessor.hpp
/// \brief AccessorExpr: field-access expression node, plus its constant operator overloads.

#include <mundy_mesh/impl/NgpAccessorExprSink.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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

template <typename T>
struct is_accessor_expr : std::false_type {};

template <typename TaggedAccessorT, typename PrevEntityExpr>
struct is_accessor_expr<AccessorExpr<TaggedAccessorT, PrevEntityExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_accessor_expr_v = is_accessor_expr<std::decay_t<T>>::value;

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

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRACCESSOR_HPP_
