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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRSINK_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRSINK_HPP_

/// \file NgpAccessorExprSink.hpp
/// \brief Sink expression types: SinkArg, impl sink helpers, ApplySinkExpr, impl::delayed_sink_expr_impl,
///        BinarySideEffectExpr, plus the OP_EQUALS and ATOMIC_OP macro definitions and their invocations.
///
/// Note: read_only(), read_write(), overwrite_all(), and sink_expr() are public functions
/// defined in NgpAccessorExpr.hpp. This file provides the underlying impl::delayed_sink_expr_impl()
/// which is called by the public sink_expr() and by NgpAccessorExprAccessor.hpp.

#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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
struct sink_arg_has_nonconstant_expr : std::bool_constant<is_math_expr_arg_v<T> && !is_constant_math_expr_v<T>> {};

template <SinkAccessMode Mode, typename Expr>
struct sink_arg_has_nonconstant_expr<SinkArg<Mode, Expr>> : std::bool_constant<!is_constant_math_expr_v<Expr>> {};

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
    auto [value_handles, final_cache] =
        cached_arg_chain<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache), context);
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
      auto result_i =
          get<I>(sink_args_)
              .expr()
              .template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<CacheType>(cache), context);
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

template <typename T>
struct is_apply_sink_expr : std::false_type {};

template <typename Func, typename... SinkArgs>
struct is_apply_sink_expr<ApplySinkExpr<Func, SinkArgs...>> : std::true_type {};

template <typename T>
static constexpr bool is_apply_sink_expr_v = is_apply_sink_expr<std::decay_t<T>>::value;

/// \brief Low-level sink expression factory (impl variant).
///
/// Returns an ApplySinkExpr without running it. Called internally by NgpAccessorExprAccessor.hpp
/// (lvalue assignment operator) and by the named-sink-wrapper macro. Users should call the
/// public sink_expr() in NgpAccessorExpr.hpp, which runs the expression immediately.
template <typename Func, typename... Args>
auto delayed_sink_expr_impl(Func func, const Args&... args) {
  static_assert(sizeof...(Args) > 0, "sink_expr(func, args...): at least one argument is required.");
  static_assert((sink_arg_has_nonconstant_expr_v<Args> || ...),
                "sink_expr(func, args...): at least one argument must be a non-constant math expression so Mundy "
                "knows which entity driver should evaluate the expression. Scalars are allowed, but they cannot be "
                "the only arguments.");
  return ApplySinkExpr<std::decay_t<Func>, decltype(make_sink_expr_arg(args))...>(std::move(func),
                                                                                  make_sink_expr_arg(args)...);
}

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
  return delayed_sink_expr_impl(std::move(func), make_sink_arg_with_mode<SinkPolicyMode<Policy, Is>::value>(args)...);
}

template <typename Policy, typename Func, typename... Args>
auto make_named_sink_expr(Func func, const Args&... args) {
  static_assert(sizeof...(Args) == Policy::num_args,
                "Named sink expression called with the wrong number of arguments for its access policy.");
  return make_named_sink_expr_impl<Policy>(std::move(func), std::make_index_sequence<sizeof...(Args)>{}, args...);
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

template <typename T>
struct is_binary_side_effect_expr : std::false_type {};

template <typename Policy, typename LeftMathExpr, typename RightMathExpr>
struct is_binary_side_effect_expr<BinarySideEffectExpr<Policy, LeftMathExpr, RightMathExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_binary_side_effect_expr_v = is_binary_side_effect_expr<std::decay_t<T>>::value;

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

MUNDY_ACCESSOR_EXPR_OP_EQUALS(Add, +=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Sub, -=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Div, /=)
MUNDY_ACCESSOR_EXPR_OP_EQUALS(Mul, *=)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicAdd, atomic_add_impl, ::mundy::atomic_add)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicSub, atomic_sub_impl, ::mundy::atomic_sub)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicMul, atomic_mul_impl, ::mundy::atomic_mul)
MUNDY_ACCESSOR_EXPR_ATOMIC_OP(AtomicDiv, atomic_div_impl, ::mundy::atomic_div)

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRSINK_HPP_
