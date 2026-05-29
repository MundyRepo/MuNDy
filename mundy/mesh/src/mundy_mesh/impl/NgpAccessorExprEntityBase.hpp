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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYBASE_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYBASE_HPP_

/// \file NgpAccessorExprEntityBase.hpp
/// \brief EntityExprBase CRTP base class.

#include <mundy_mesh/impl/NgpAccessorExprCachable.hpp>
#include <mundy_mesh/impl/NgpAccessorExprUtils.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename DerivedEntityExpr>
class EntityExprBase : public CachableExprBase<DerivedEntityExpr> {
 public:
  using our_t = EntityExprBase<DerivedEntityExpr>;
  using our_tag = typename CachableExprBase<DerivedEntityExpr>::our_tag;

  template <typename T>
  struct is_this_expr : std::is_same<std::decay_t<T>, our_t> {};

  template <typename T>
  static constexpr bool is_this_expr_v = is_this_expr<T>::value;

  KOKKOS_INLINE_FUNCTION
  constexpr const DerivedEntityExpr& self() const noexcept {
    return static_cast<const DerivedEntityExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr DerivedEntityExpr& self() noexcept {
    return static_cast<DerivedEntityExpr&>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return self().rank();
  }

  const auto driver() const {
    return self().driver();
  }

  template <size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                                       const Ctx& context) const {
    return self().eval(fmis, context);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType, size_t NumEntities, class Ctx>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const Ctx& context) const {
    return self().template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<OldCacheType>(old_cache),
                                                                    context);
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
};

template <typename T>
struct is_entity_expr_base : std::false_type {};

template <typename DerivedEntityExpr>
struct is_entity_expr_base<EntityExprBase<DerivedEntityExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_entity_expr_base_v = is_entity_expr_base<std::decay_t<T>>::value;

template <typename T>
struct is_entity_expr_arg : std::bool_constant<is_crtp_base_of_v<EntityExprBase, std::decay_t<T>>> {};

template <typename T>
static constexpr bool is_entity_expr_arg_v = is_entity_expr_arg<T>::value;

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYBASE_HPP_
