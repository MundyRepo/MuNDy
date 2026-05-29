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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYEXPR_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYEXPR_HPP_

/// \file NgpAccessorExprEntityExpr.hpp
/// \brief EntityExpr leaf node and definitions for entity holder types.

#include <mundy_mesh/impl/NgpAccessorExprConnectedEntities.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <size_t NumEntities, size_t Ord, typename DriverType>
class EntityExpr : public EntityExprBase<EntityExpr<NumEntities, Ord, DriverType>> {
 public:
  using our_t = EntityExpr<NumEntities, Ord, DriverType>;
  using our_tag = typename EntityExprBase<EntityExpr<NumEntities, Ord, DriverType>>::our_tag;
  using sub_expressions_t = tuple<>;
  static constexpr size_t num_entities = NumEntities;
  static constexpr bool has_static_eval = true;

  template <typename T>
  struct is_this_expr : std::is_same<std::decay_t<T>, our_t> {};

  template <typename T>
  static constexpr bool is_this_expr_v = is_this_expr<T>::value;

  KOKKOS_INLINE_FUNCTION
  EntityExpr(const stk::mesh::EntityRank& rank, const DriverType* driver) : rank_(rank), driver_(driver) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities> fmis,
                                                       const NgpEvalContext& /*context*/) const {
    static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
    return fmis[Ord];
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, num_entities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& /*context*/) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        auto val = fmis[Ord];
        auto new_cache = append<our_tag>(std::forward<OldCacheType>(old_cache), val);
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::move(new_cache));
      }
    } else {
      auto val = fmis[Ord];
      return Kokkos::make_pair(OwnedCachedValue{val}, std::forward<OldCacheType>(old_cache));
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  void propagate_synchronize(const NgpEvalContext& /*context*/) {
  }

  void flag_read_only(const NgpEvalContext& /*context*/) {
  }

  void flag_read_write(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
  }

  void flag_overwrite_all(const NgpEvalContext& /*context*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "Attempting to write to the return type of an entity expression, which returns a temporary value.");
  }

  const DriverType* driver() const {
    return driver_;
  }

 private:
  stk::mesh::EntityRank rank_;
  const DriverType* driver_;
};

template <typename T>
struct is_entity_expr : std::false_type {};

template <size_t NumEntities, size_t Ord, typename DriverType>
struct is_entity_expr<EntityExpr<NumEntities, Ord, DriverType>> : std::true_type {};

template <typename T>
static constexpr bool is_entity_expr_v = is_entity_expr<std::decay_t<T>>::value;

template <size_t NumEntities, typename DriverType>
template <size_t Ord>
KOKKOS_INLINE_FUNCTION EntityExpr<NumEntities, Ord, DriverType> IntermediaryEntityArray<NumEntities, DriverType>::get()
    const {
  static_assert(Ord < NumEntities, "EntityExpr ordinal must be less than NumEntities");
  return EntityExpr<NumEntities, Ord, DriverType>(ranks_[Ord], driver_);
}

template <typename DriverType>
template <size_t Ord>
KOKKOS_INLINE_FUNCTION EntityExpr<2, Ord, DriverType> EntityPair<DriverType>::get() const {
  static_assert(Ord < 2, "EntityExpr ordinal must be less than 2");
  if constexpr (Ord == 0) {
    return EntityExpr<2, Ord, DriverType>(first_rank_, driver_);
  } else {
    return EntityExpr<2, Ord, DriverType>(second_rank_, driver_);
  }
}

template <typename DriverType>
KOKKOS_INLINE_FUNCTION EntityExpr<2, 0, DriverType> EntityPair<DriverType>::first() const {
  return EntityExpr<2, 0, DriverType>(first_rank_, driver_);
}

template <typename DriverType>
KOKKOS_INLINE_FUNCTION EntityExpr<2, 1, DriverType> EntityPair<DriverType>::second() const {
  return EntityExpr<2, 1, DriverType>(second_rank_, driver_);
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRENTITYEXPR_HPP_
