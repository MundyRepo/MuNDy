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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRCONNECTEDENTITIES_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRCONNECTEDENTITIES_HPP_

/// \file NgpAccessorExprConnectedEntities.hpp
/// \brief ConnectedEntitiesExpr and IndexedConnectedEntityExpr entity-expression nodes.

#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

// Forward declaration so ConnectedEntitiesExpr::operator[] can name this type.
template <typename PrevConnectedEntitiesExpr>
class IndexedConnectedEntityExpr;

template <typename PrevEntityExpr>
class ConnectedEntitiesExpr : public EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>> {
 public:
  using our_t = ConnectedEntitiesExpr<PrevEntityExpr>;
  using our_tag = typename EntityExprBase<ConnectedEntitiesExpr<PrevEntityExpr>>::our_tag;
  using sub_expressions_t = tuple<PrevEntityExpr>;
  using ConnectedEntities = stk::mesh::NgpMesh::ConnectedEntities;
  static constexpr bool constrains_num_entities = false;
  static constexpr bool has_static_eval = false;

  template <typename T>
  struct is_this_expr : std::is_same<std::decay_t<T>, our_t> {};

  template <typename T>
  static constexpr bool is_this_expr_v = is_this_expr<T>::value;

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(PrevEntityExpr prev_entity_expr, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  ConnectedEntitiesExpr(const EntityExprBase<PrevEntityExpr>& prev_entity_expr_base, stk::mesh::EntityRank conn_rank)
      : prev_entity_expr_(prev_entity_expr_base.self()), conn_rank_(conn_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return conn_rank_;
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION ConnectedEntities eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                                const NgpEvalContext& context) const {
    stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
    stk::mesh::FastMeshIndex entity_index = prev_entity_expr_.eval(fmis, context);
    return context.ngp_mesh().get_connected_entities(entity_rank, entity_index, conn_rank_);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto entity_value = std::move(entity_result.first);
        stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
        auto val =
            context.ngp_mesh().get_connected_entities(entity_rank, entity_value.get(entity_result.second), conn_rank_);
        auto newest_cache = append<our_tag>(std::move(entity_result.second), val);
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      stk::mesh::EntityRank entity_rank = prev_entity_expr_.rank();
      auto entity_result = prev_entity_expr_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto entity_value = std::move(entity_result.first);
      auto val =
          context.ngp_mesh().get_connected_entities(entity_rank, entity_value.get(entity_result.second), conn_rank_);
      return Kokkos::make_pair(OwnedCachedValue{std::move(val)}, std::move(entity_result.second));
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  /// \brief Select the entity at runtime index \p index from this connected-entities collection.
  /// Returns an entity expression whose eval() resolves to the FastMeshIndex of that specific entity.
  KOKKOS_INLINE_FUNCTION
  auto operator[](size_t index) const {
    return IndexedConnectedEntityExpr<our_t>(*this, index);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(RuntimeReuseValidator& validator) const {
    prev_entity_expr_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return conn_rank_ == other.conn_rank_ && prev_entity_expr_.runtime_reuse_equivalent(other.prev_entity_expr_);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    prev_entity_expr_.propagate_synchronize(context);
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

  const auto driver() const {
    return prev_entity_expr_.driver();
  }

 private:
  PrevEntityExpr prev_entity_expr_;
  stk::mesh::EntityRank conn_rank_;
};

template <typename T>
struct is_connected_entities_expr : std::false_type {};

template <typename PrevEntityExpr>
struct is_connected_entities_expr<ConnectedEntitiesExpr<PrevEntityExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_connected_entities_expr_v = is_connected_entities_expr<std::decay_t<T>>::value;

// ---------------------------------------------------------------------------
// IndexedConnectedEntityExpr
//
// Entity expression returned by ConnectedEntitiesExpr::operator[](index).
// eval() resolves to the FastMeshIndex of the entity at position `index_`
// in the connected-entities collection produced by the parent expression.
// ---------------------------------------------------------------------------
template <typename PrevConnectedEntitiesExpr>
class IndexedConnectedEntityExpr
    : public EntityExprBase<IndexedConnectedEntityExpr<PrevConnectedEntitiesExpr>> {
 public:
  using our_t   = IndexedConnectedEntityExpr<PrevConnectedEntitiesExpr>;
  using our_tag = typename EntityExprBase<our_t>::our_tag;
  using sub_expressions_t = tuple<PrevConnectedEntitiesExpr>;
  static constexpr bool constrains_num_entities = false;
  // Runtime index means two same-typed instances may differ; no static caching.
  static constexpr bool has_static_eval = false;

  template <typename T>
  struct is_this_expr : std::is_same<std::decay_t<T>, our_t> {};

  template <typename T>
  static constexpr bool is_this_expr_v = is_this_expr<T>::value;

  KOKKOS_INLINE_FUNCTION
  IndexedConnectedEntityExpr(const PrevConnectedEntitiesExpr& prev, size_t index)
      : prev_(prev), index_(index) {
  }

  KOKKOS_INLINE_FUNCTION
  IndexedConnectedEntityExpr(const EntityExprBase<PrevConnectedEntitiesExpr>& prev_base, size_t index)
      : prev_(prev_base.self()), index_(index) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return prev_.rank();
  }

  template <size_t NumEntities>
  KOKKOS_INLINE_FUNCTION stk::mesh::FastMeshIndex eval(
      const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
      const NgpEvalContext& context) const {
    auto conn_entities = prev_.eval(fmis, context);
    return context.ngp_mesh().fast_mesh_index(conn_entities[index_]);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts, size_t NumEntities, typename OldCacheType>
  KOKKOS_INLINE_FUNCTION auto cached_eval(const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                          OldCacheType&& old_cache, const NgpEvalContext& context) const {
    static_assert(has<our_tag>(eval_counts), "eval_counts must contain our tag");

    if constexpr (our_t::has_static_eval && get<our_tag>(eval_counts) > 1) {
      if constexpr (aggregate_has_v<our_tag, std::remove_reference_t<OldCacheType>>) {
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::forward<OldCacheType>(old_cache));
      } else {
        auto conn_result = prev_.template cached_eval<EvalCountsType, eval_counts>(
            fmis, std::forward<OldCacheType>(old_cache), context);
        auto conn_value = std::move(conn_result.first);
        auto val        = context.ngp_mesh().fast_mesh_index(conn_value.get(conn_result.second)[index_]);
        auto newest_cache = append<our_tag>(std::move(conn_result.second), val);
        return Kokkos::make_pair(CachedTagGetter<our_tag>{}, std::move(newest_cache));
      }
    } else {
      auto conn_result = prev_.template cached_eval<EvalCountsType, eval_counts>(
          fmis, std::forward<OldCacheType>(old_cache), context);
      auto conn_value = std::move(conn_result.first);
      auto val        = context.ngp_mesh().fast_mesh_index(conn_value.get(conn_result.second)[index_]);
      return Kokkos::make_pair(OwnedCachedValue{std::move(val)}, std::move(conn_result.second));
    }
  }

  KOKKOS_INLINE_FUNCTION
  auto get_connectivity(stk::mesh::EntityRank conn_rank) const {
    return ConnectedEntitiesExpr<our_t>(*this, conn_rank);
  }

  template <typename EvalCountsType, EvalCountsType eval_counts>
  void validate_runtime_reuse(RuntimeReuseValidator& validator) const {
    prev_.template validate_runtime_reuse<EvalCountsType, eval_counts>(validator);
    validator.template validate<EvalCountsType, eval_counts>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  bool runtime_reuse_equivalent(const our_t& other) const {
    return index_ == other.index_ && prev_.runtime_reuse_equivalent(other.prev_);
  }

  void propagate_synchronize(const NgpEvalContext& context) {
    prev_.propagate_synchronize(context);
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

  const auto driver() const {
    return prev_.driver();
  }

 private:
  PrevConnectedEntitiesExpr prev_;
  size_t                    index_;
};

template <typename T>
struct is_indexed_connected_entity_expr : std::false_type {};

template <typename PrevConnectedEntitiesExpr>
struct is_indexed_connected_entity_expr<IndexedConnectedEntityExpr<PrevConnectedEntitiesExpr>> : std::true_type {};

template <typename T>
static constexpr bool is_indexed_connected_entity_expr_v =
    is_indexed_connected_entity_expr<std::decay_t<T>>::value;

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRCONNECTEDENTITIES_HPP_
