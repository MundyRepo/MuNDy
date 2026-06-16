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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRRNG_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRRNG_HPP_

/// \file NgpAccessorExprRNG.hpp
/// \brief Random-number expression types: RandomDistributionExpr, UniformDistributionExpr,
///        and CounterBasedRNGExpr.
///
/// The rng() factory overloads are public API defined in NgpAccessorExpr.hpp.

#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/rng.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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

template <typename T>
struct is_random_distribution_expr : std::false_type {};

template <typename RNGExpr, typename T>
struct is_random_distribution_expr<RandomDistributionExpr<RNGExpr, T>> : std::true_type {};

template <typename T>
static constexpr bool is_random_distribution_expr_v = is_random_distribution_expr<std::decay_t<T>>::value;

#define MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(OpName, op)                                    \
  template <typename ConstantType, typename SubRNGExpr, typename T>                                                    \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ConstantType& c, const RandomDistributionExpr<SubRNGExpr, T>& expr) {                        \
    using expr_t = RandomDistributionExpr<SubRNGExpr, T>;                                                              \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                   \
    return OpName##Expr<ConstantMathExpr<ConstantType>, expr_t>(constant_expr, expr);                                  \
  }

#define MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(OpName, op)                                   \
  template <typename ConstantType, typename SubRNGExpr, typename T>                                                    \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const RandomDistributionExpr<SubRNGExpr, T>& expr, const ConstantType& c) {                        \
    using expr_t = RandomDistributionExpr<SubRNGExpr, T>;                                                              \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                   \
    return OpName##Expr<expr_t, ConstantMathExpr<ConstantType>>(expr, constant_expr);                                  \
  }

MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Div, /)

#undef MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR
#undef MUNDY_ACCESSOR_EXPR_RANDOM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR

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

template <typename T>
struct is_uniform_distribution_expr : std::false_type {};

template <typename RNGExpr, typename T, typename LowExpr, typename HighExpr>
struct is_uniform_distribution_expr<UniformDistributionExpr<RNGExpr, T, LowExpr, HighExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_uniform_distribution_expr_v = is_uniform_distribution_expr<std::decay_t<T>>::value;

#define MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(OpName, op)                                   \
  template <typename ConstantType, typename SubRNGExpr, typename T, typename LowExpr, typename HighExpr>              \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const ConstantType& c, const UniformDistributionExpr<SubRNGExpr, T, LowExpr, HighExpr>& expr) {    \
    using expr_t = UniformDistributionExpr<SubRNGExpr, T, LowExpr, HighExpr>;                                          \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                   \
    return OpName##Expr<ConstantMathExpr<ConstantType>, expr_t>(constant_expr, expr);                                  \
  }

#define MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(OpName, op)                                  \
  template <typename ConstantType, typename SubRNGExpr, typename T, typename LowExpr, typename HighExpr>              \
  MUNDY_REQUIRES(!is_crtp_base_of_v<MathExprBase, ConstantType> && !is_crtp_base_of_v<EntityExprBase, ConstantType>) \
  auto operator op(const UniformDistributionExpr<SubRNGExpr, T, LowExpr, HighExpr>& expr, const ConstantType& c) {    \
    using expr_t = UniformDistributionExpr<SubRNGExpr, T, LowExpr, HighExpr>;                                          \
    ConstantMathExpr<ConstantType> constant_expr(c);                                                                   \
    return OpName##Expr<expr_t, ConstantMathExpr<ConstantType>>(expr, constant_expr);                                  \
  }

MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR(Div, /)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Add, +)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Sub, -)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Mul, *)
MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR(Div, /)

#undef MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_LEFT_OPERATOR
#undef MUNDY_ACCESSOR_EXPR_UNIFORM_DISTRIBUTION_CONSTANT_RIGHT_OPERATOR

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
  // and high. Low & high are expressions
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

template <typename T>
struct is_counter_based_rng_expr : std::false_type {};

template <typename SeedExpr, typename CounterExpr, typename RNGType,
          RNGType (*make_counter_based_rng)(size_t, size_t)>
struct is_counter_based_rng_expr<CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>>
    : std::true_type {};

template <typename T>
static constexpr bool is_counter_based_rng_expr_v = is_counter_based_rng_expr<std::decay_t<T>>::value;

template <typename SeedExpr, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(is_math_expr_arg_v<SeedExpr>&& is_math_expr_arg_v<CounterExpr>)
auto rng_impl(const SeedExpr& seed_expr, const CounterExpr& counter_expr) {
  return CounterBasedRNGExpr<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}

template <typename SeedExpr, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(is_math_expr_arg_v<SeedExpr> && !is_math_expr_arg_v<CounterT>)
auto rng_impl(const SeedExpr& seed_expr, const CounterT& counter) {
  using CounterExpr = ConstantMathExpr<CounterT>;
  auto counter_expr = CounterExpr(counter);
  return rng_impl<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}

template <typename SeedT, typename CounterExpr, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!is_math_expr_arg_v<SeedT> && is_math_expr_arg_v<CounterExpr>)
auto rng_impl(const SeedT& seed, const CounterExpr& counter_expr) {
  using SeedExpr = ConstantMathExpr<SeedT>;
  auto seed_expr = SeedExpr(seed);
  return rng_impl<SeedExpr, CounterExpr, RNGType, make_counter_based_rng>(seed_expr, counter_expr);
}

template <typename SeedT, typename CounterT, typename RNGType = openrand::Philox,
          RNGType (*make_counter_based_rng)(size_t, size_t) = make_philox>
MUNDY_REQUIRES(!is_math_expr_arg_v<SeedT> && !is_math_expr_arg_v<CounterT>)
void rng_impl(const SeedT& /*seed*/, const CounterT& /*counter*/) {
  MUNDY_THROW_REQUIRE(false, std::logic_error,
                      "Both seed and counter arguments to rng() cannot be constants.\n"
                      "At least one of them must be an expression, lest we have no idea how to run the expression over "
                      "multiple entities.");
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRRNG_HPP_
