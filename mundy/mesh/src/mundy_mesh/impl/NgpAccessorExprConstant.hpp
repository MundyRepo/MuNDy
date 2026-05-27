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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRCONSTANT_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRCONSTANT_HPP_

/// \file NgpAccessorExprConstant.hpp
/// \brief ConstantMathExpr: a literal-value leaf node in the NGP expression system.

#include <mundy_mesh/impl/NgpAccessorExprMathBase.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename ConstantType>
class ConstantMathExpr : public MathExprBase<ConstantMathExpr<ConstantType>> {
 public:
  using our_t = ConstantMathExpr<ConstantType>;
  using our_tag = typename MathExprBase<ConstantMathExpr<ConstantType>>::our_tag;
  using sub_expressions_t = tuple<>;
  static constexpr bool constrains_num_entities = false;
  // The constant value is runtime state that affects the evaluation, so equal tags do not imply equal eval results.
  static constexpr bool has_static_eval = false;

  KOKKOS_INLINE_FUNCTION
  ConstantMathExpr(ConstantType value) : value_(value) {
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION ConstantType eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& /*fmis*/,
                                           const NgpEvalContext& /*context*/) const {
    return value_;
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& /*fmis*/,
                                          OldCacheType&& old_cache, const NgpEvalContext& /*context*/) const {
    static_assert(
        !aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>,
        "The cache somehow contains our tag, but our eval returns a constant and should never cache anything.");
    return Kokkos::make_pair(impl::OwnedCachedValue{value_}, std::forward<OldCacheType>(old_cache));
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse([[maybe_unused]] impl::RuntimeReuseValidator& validator) const {
    // Constants are leaves. Nothing to do here.
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return value_ == other.value_;
  }

  void propagate_synchronize(const NgpEvalContext& /*context*/) {
    // Nothing to do here
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
    // Nothing to do here
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error, "Attempting to write to a constant expression.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error, "Attempting to write to a constant expression.");
  }

  auto driver() const {
    return nullptr;
  }

 private:
  ConstantType value_;
};

template <typename T>
struct is_constant_math_expr : std::false_type {};

template <typename T>
struct is_constant_math_expr<ConstantMathExpr<T>> : std::true_type {};

template <typename T>
static constexpr bool is_constant_math_expr_v = is_constant_math_expr<std::decay_t<T>>::value;

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRCONSTANT_HPP_
