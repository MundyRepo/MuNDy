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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRMATHBASE_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRMATHBASE_HPP_

/// \file NgpAccessorExprMathBase.hpp
/// \brief MathExprBase CRTP base class for all mathematical expression nodes.

#include <mundy_mesh/impl/NgpAccessorExprCachable.hpp>
#include <mundy_mesh/impl/NgpAccessorExprUtils.hpp>

namespace mundy {

namespace mesh {

namespace impl {

//! \name Views of mathematical expressions
//@{

template <typename DerivedMathExpr>
class MathExprBase : public CachableExprBase<DerivedMathExpr> {
 public:
  using our_t = MathExprBase<DerivedMathExpr>;
  using our_tag = typename CachableExprBase<DerivedMathExpr>::our_tag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr MathExprBase() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedMathExpr& self() const noexcept {
    return static_cast<const DerivedMathExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedMathExpr& self() noexcept {
    return static_cast<DerivedMathExpr&>(*this);
  }

  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                   const Ctx& context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename CacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          CacheType& cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, cache, context);
  }

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

  const auto driver() const {
    return self().driver();
  }
};

template <typename T>
struct is_math_expr_base : std::false_type {};

template <typename DerivedMathExpr>
struct is_math_expr_base<MathExprBase<DerivedMathExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_math_expr_base_v = is_math_expr_base<std::decay_t<T>>::value;

template <typename T>
struct is_math_expr_arg : std::bool_constant<is_crtp_base_of_v<MathExprBase, std::decay_t<T>>> {};

template <typename T>
static constexpr bool is_math_expr_arg_v = is_math_expr_arg<T>::value;
//@}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRMATHBASE_HPP_
