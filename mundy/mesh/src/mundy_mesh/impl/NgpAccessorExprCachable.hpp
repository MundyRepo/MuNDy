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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRCACHABLE_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRCACHABLE_HPP_

/// \file NgpAccessorExprCachable.hpp
/// \brief CachableExprBase: CRTP base for all cachable expression nodes in the NGP expression system.

#include <mundy_mesh/impl/NgpAccessorExprTypes.hpp>

namespace mundy {

namespace mesh {

namespace impl {

//! \name Cachable expression base
//@{

template <typename DerivedExpr>
class CachableExprBase {
 public:
  using our_tag = DerivedExpr;

  // Derived expressions must expose:
  //   static constexpr bool has_static_eval;
  //
  // When true, the expression may participate in type-based memoization. When false, equal tags are not assumed to
  // imply equal eval results, so cached_eval must recompute the value each time.
  //
  // The rigorous rule is:
  //   - true only for pure expressions whose eval result is fully determined by the tag and their eval inputs
  //   - false for expressions with side effects
  //   - false for expressions whose eval depends on runtime state not represented by the tag
  //
  // Simply because an object has non-static members does not necessarily mean it must have has_static_eval=false. For
  // example, an AccessorExpr has_static_eval=true since the fields are themselves tagged (basically a contract stating
  // that each tag corresponds to a unique field).
  static constexpr bool has_static_eval = DerivedExpr::has_static_eval;

  // Optional opt-in for runtime reuse when has_static_eval == false.
  //
  // Runtime reuse is still cache-by-tag, but unlike static caching it requires host-side structural validation. If an
  // expression enables runtime reuse, and the same tag appears multiple times in a tree, each instance must be
  // structurally equivalent at runtime_reuse_equivalent(). Expressions opting in should therefore override
  // runtime_reuse_equivalent(const Self&) for their structural contract.
  static constexpr bool supports_runtime_reuse = false;

 private:
  template <typename Tag, typename AggregateType, AggregateType agg>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_tag_count() {
    if constexpr (aggregate_has_v<Tag, AggregateType>) {
      auto new_agg = agg;
      get<Tag>(new_agg) += 1;
      return new_agg;
    } else {
      return append<Tag>(agg, 1);
    }
  }

  template <typename SubExprTuple, size_t I, typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_eval_counts_recurse() {
    if constexpr (I < SubExprTuple::size()) {
      using sub_expr_t = tuple_element_t<I, SubExprTuple>;
      // Recurse into the sub-expression
      constexpr auto updated_eval_counts =
          sub_expr_t::template increment_eval_counts<OldEvalCountsType, old_eval_counts>();
      return increment_eval_counts_recurse<SubExprTuple, I + 1, decltype(updated_eval_counts), updated_eval_counts>();
    } else {
      return old_eval_counts;
    }
  }

 public:
  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedExpr& self() const noexcept {
    return static_cast<const DerivedExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedExpr& self() noexcept {
    return static_cast<DerivedExpr&>(*this);
  }

  // Default structural-equivalence rule:
  // - static nodes are equivalent by tag contract
  // - non-static nodes are not equivalent unless a derived class overrides this method
  KOKKOS_INLINE_FUNCTION
  constexpr bool runtime_reuse_equivalent([[maybe_unused]] const DerivedExpr& other) const noexcept {
    return has_static_eval;
  }

  /// \brief Evaluate the expression
  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const Ctx& context) const {
    return self().eval(fmis, context);
  }

  /// \brief Evaluate the expression
  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
  }

  /// \brief Update eval_counts by incrementing the counts for our tag and our sub-expressions tags
  template <typename OldEvalCountsType, OldEvalCountsType old_eval_counts>
  KOKKOS_INLINE_FUNCTION static constexpr auto increment_eval_counts() {
    constexpr auto new_eval_counts = increment_tag_count<our_tag, OldEvalCountsType, old_eval_counts>();
    using sub_exprs = typename DerivedExpr::sub_expressions_t;
    return increment_eval_counts_recurse<sub_exprs, 0, decltype(new_eval_counts), new_eval_counts>();
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse([[maybe_unused]] impl::RuntimeReuseValidator& validator) const {
    // Leaf/default behavior: recurse only where derived classes explicitly expose their sub-expression ownership.
  }

  //! \name Field synchronization and modification flagging
  //@{

  template <class Ctx>
  void propagate_synchronize(const Ctx& context) {
    self().propagate_synchronize(context);
  }

  template <class Ctx>
  void flag_read_only(const Ctx& context) {
    self().flag_read_only(context);
  }

  template <class Ctx>
  void flag_read_write(const Ctx& context) {
    self().flag_read_write(context);
  }

  template <class Ctx>
  void flag_overwrite_all(const Ctx& context) {
    self().flag_overwrite_all(context);
  }
  //@}
};

template <typename T>
struct is_cachable_expr_base : std::false_type {};

template <typename DerivedExpr>
struct is_cachable_expr_base<CachableExprBase<DerivedExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_cachable_expr_base_v = is_cachable_expr_base<std::decay_t<T>>::value;
//@}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRCACHABLE_HPP_
