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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRASSIGN_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRASSIGN_HPP_

/// \file NgpAccessorExprAssign.hpp
/// \brief AssignExpr: a side-effecting assignment expression node in the NGP expression system.

#include <mundy_mesh/impl/NgpAccessorExprMathBase.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

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

template <typename T>
struct is_assign_expr : std::false_type {};

template <typename TargetExpr, typename SourceExpr>
struct is_assign_expr<AssignExpr<TargetExpr, SourceExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_assign_expr_v = is_assign_expr<std::decay_t<T>>::value;

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRASSIGN_HPP_
