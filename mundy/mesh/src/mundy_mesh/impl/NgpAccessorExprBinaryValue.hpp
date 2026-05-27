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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRBINARYVALUE_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRBINARYVALUE_HPP_

/// \file NgpAccessorExprBinaryValue.hpp
/// \brief BinaryValueExpr, the MUNDY_ACCESSOR_EXPR_OP macro, standard arithmetic expression types
///        (AddExpr, SubExpr, MulExpr, DivExpr), and their constant-operand operator overloads.

#include <mundy_mesh/impl/NgpAccessorExprConstant.hpp>
#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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

template <typename T>
struct is_binary_value_expr : std::false_type {};

template <typename Policy, typename LeftMathExpr, typename RightMathExpr>
struct is_binary_value_expr<BinaryValueExpr<Policy, LeftMathExpr, RightMathExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_binary_value_expr_v = is_binary_value_expr<std::decay_t<T>>::value;

/// \brief Macro that defines an OpPolicy struct and an OpExpr type alias for BinaryValueExpr.
///
/// Usage: MUNDY_ACCESSOR_EXPR_OP(Add, +) creates AddOpPolicy and AddExpr.
#define MUNDY_ACCESSOR_EXPR_OP(OpName, op)                                                     \
  struct OpName##OpPolicy {                                                                    \
    template <typename LeftValue, typename RightValue>                                         \
    KOKKOS_INLINE_FUNCTION static auto apply(const LeftValue& left, const RightValue& right) { \
      return left op right;                                                                    \
    }                                                                                          \
  };                                                                                           \
  template <typename LeftMathExpr, typename RightMathExpr>                                     \
  using OpName##Expr = BinaryValueExpr<OpName##OpPolicy, LeftMathExpr, RightMathExpr>;

// Create the standard arithmetic binary expression types.
MUNDY_ACCESSOR_EXPR_OP(Add, +)
MUNDY_ACCESSOR_EXPR_OP(Sub, -)
MUNDY_ACCESSOR_EXPR_OP(Div, /)
MUNDY_ACCESSOR_EXPR_OP(Mul, *)

// Global binary value operator overloads for BinaryValueExpr op MathExprBase
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

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRBINARYVALUE_HPP_
