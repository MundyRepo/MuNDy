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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRFUSED_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRFUSED_HPP_

/// \file NgpAccessorExprFused.hpp
/// \brief FusedAssignExpr: a fused multi-assignment expression node that evaluates all sub-expressions
///        and assigns their results simultaneously, avoiding redundant field synchronizations.
///
/// The fused_assign() factory function is public API defined in NgpAccessorExpr.hpp.

#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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

template <typename T>
struct is_fused_assign_expr : std::false_type {};

template <typename... TrgSrcExprPairs>
struct is_fused_assign_expr<FusedAssignExpr<TrgSrcExprPairs...>> : std::true_type {};

template <typename T>
static constexpr bool is_fused_assign_expr_v = is_fused_assign_expr<std::decay_t<T>>::value;

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRFUSED_HPP_
