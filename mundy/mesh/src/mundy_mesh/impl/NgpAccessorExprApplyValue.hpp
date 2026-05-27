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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRAPPLYVALUE_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRAPPLYVALUE_HPP_

/// \file NgpAccessorExprApplyValue.hpp
/// \brief ApplyValueExpr: function-application value expression node, its operator overloads,
///        and the impl-namespace helpers (is_math_expr_arg_v, make_apply_expr_arg, etc.).
///
/// The apply_expr() factory function is public API defined in NgpAccessorExpr.hpp.

#include <mundy_mesh/impl/NgpAccessorExprAssign.hpp>
#include <mundy_mesh/impl/NgpAccessorExprBinaryValue.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename...>
static constexpr bool dependent_false_v = false;

template <typename Expr>
KOKKOS_INLINE_FUNCTION auto make_apply_expr_arg(const MathExprBase<Expr>& expr) {
  return expr.self();
}

template <typename T>
MUNDY_REQUIRES(!impl::is_math_expr_arg_v<T> && !is_crtp_base_of_v<EntityExprBase, T>)
KOKKOS_INLINE_FUNCTION auto make_apply_expr_arg(const T& value) {
  return ConstantMathExpr<std::decay_t<T>>(value);
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

template <typename T>
struct is_apply_value_expr : std::false_type {};

template <typename Func, typename... Exprs>
struct is_apply_value_expr<ApplyValueExpr<Func, Exprs...>> : std::true_type {};

template <typename T>
static constexpr bool is_apply_value_expr_v = is_apply_value_expr<std::decay_t<T>>::value;

// Apply value operator overloads for ApplyValueExpr op MathExprBase
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

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRAPPLYVALUE_HPP_
