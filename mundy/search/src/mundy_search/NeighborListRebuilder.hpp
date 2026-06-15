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

#ifndef MUNDY_SEARCH_NEIGHBORLISTREBUILDER_HPP_
#define MUNDY_SEARCH_NEIGHBORLISTREBUILDER_HPP_

/// \file NeighborListRebuilder.hpp
/// \brief RebuilderType concept, rebuilder implementations, and rebuilder chaining via operator|.

// C++ core
#include <concepts>     // for std::convertible_to, std::same_as
#include <cstddef>      // for size_t
#include <limits>       // for std::numeric_limits
#include <type_traits>  // for std::remove_cvref_t

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_selected_entities
#include <stk_mesh/base/MetaData.hpp>     // for stk::mesh::MetaData
#include <stk_mesh/base/Selector.hpp>     // for stk::mesh::Selector
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_geom/periodicity.hpp>     // for FreeSpaceMetric
#include <mundy_geom/primitives/OBB.hpp>  // for mundy::OBB, mundy::quaternion_to_rotation_matrix
#include <mundy_math/Quaternion.hpp>      // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>         // for mundy::Vector3
#include <mundy_math/cmath.hpp>
#include <mundy_mesh/EntityIndices.hpp>  // for mundy::mesh::get_local_entities, get_local_entity_indices
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::AABBFieldComponent/OBBFieldComponent, get_updated_ngp_component
#include <mundy_mesh/NgpFieldBLAS.hpp>    // for mundy::mesh::field_copy
#include <mundy_mesh/impl/DeclareUniqueFieldLike.hpp>  // for mundy::mesh::impl::declare_unique_field_like
#include <mundy_search/NeighborListBuildTraits.hpp>    // for AABBSearchInputTypeFor
#include <mundy_search/SearchInput.hpp>                // for SearchInput
#include <mundy_utils/host_ptr.hpp>                    // for host_ptr
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

namespace impl {

/// \class ScopedLateFields
/// \brief RAII guard that enables STK late-field declaration only if we are the ones who turned it on.
///
/// On a committed mesh, declaring a field requires `enable_late_fields()`. This guard enables it on construction
/// *only* when the mesh is committed and late fields were not already enabled, and restores the prior state on
/// destruction (so a caller that found late fields already on leaves them on). On an uncommitted mesh it is a no-op
/// (declaring fields pre-commit is already allowed). Exception-safe: the restore happens even if a declaration throws.
class ScopedLateFields {
 public:
  explicit ScopedLateFields(stk::mesh::MetaData& meta_data) {
    if (meta_data.is_commit() && !meta_data.are_late_fields_enabled()) {
      meta_data.enable_late_fields();
      meta_data_ = &meta_data;  // remember that *we* enabled it, so *we* disable it
    }
  }

  ~ScopedLateFields() {
    if (meta_data_ != nullptr) {
      meta_data_->disable_late_fields();
    }
  }

  ScopedLateFields(const ScopedLateFields&) = delete;
  ScopedLateFields& operator=(const ScopedLateFields&) = delete;
  ScopedLateFields(ScopedLateFields&&) = delete;
  ScopedLateFields& operator=(ScopedLateFields&&) = delete;

 private:
  stk::mesh::MetaData* meta_data_ = nullptr;  //!< Non-null iff this guard enabled late fields and must disable them.
};

/// \brief Debug-assert that a rebuilder is queried with the same selectors it snapshotted.
///
/// A neighbor list's selectors define its identity: entities may change *within* a selector (via mesh modification),
/// but changing the selector itself changes the nature of the list and requires building a new one. Every rebuilder
/// relies on this to compare snapshots soundly, so misuse is caught here (compiles out in release builds).
inline void assert_fixed_selectors(const stk::mesh::Selector& target, const stk::mesh::Selector& snapshot_target,
                                   const stk::mesh::Selector& source, const stk::mesh::Selector& snapshot_source) {
  MUNDY_THROW_ASSERT(target == snapshot_target && source == snapshot_source, std::runtime_error,
                     "NeighborListRebuilder was queried with a different selector than it snapshotted. A neighbor "
                     "list's selector defines its identity; changing it requires building a new list (entities may "
                     "change within the selector via mesh modification, but the selector itself must stay fixed).");
}

}  // namespace impl

/// \concept RebuilderType
/// \brief Specifies a stateful policy that decides when a neighbor list needs to be rebuilt.
template <typename T>
concept RebuilderType =
    requires(T& rebuilder, const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
             const SearchInput<mundy::mesh::AABBFieldComponent<double>>& input) {
      { rebuilder.setup(bulk_data, selector, selector) } -> std::same_as<void>;
      { rebuilder.needs_rebuild(bulk_data, input, input) } -> std::convertible_to<bool>;
      { rebuilder.snapshot(bulk_data, input, input) } -> std::same_as<void>;
    };

/// \class RebuilderChain
/// \brief Type-level OR chain of two rebuilders.
///
/// `needs_rebuild` short-circuits: if `prior_` triggers, `next_` is not consulted.
/// `snapshot` always fires on both members so that every rebuilder's snapshot stays current.
///
/// Build chains with any rebuilder's `.rebuild_if(next)` or `operator|` method.
template <typename PriorRebuilder, typename Rebuilder>
class RebuilderChain {
 public:
  //! \name Aliases
  //@{

  using prior_rebuilder_type = PriorRebuilder;
  using appended_rebuilder_type = Rebuilder;
  //@}

  //! \name Constructors
  //@{

  RebuilderChain() = default;

  RebuilderChain(const prior_rebuilder_type& prior, const appended_rebuilder_type& next) : prior_(prior), next_(next) {
  }
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief Forward per-update setup to both chain members.
  void setup(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& target_selector,
             const stk::mesh::Selector& source_selector) {
    prior_.setup(bulk, target_selector, source_selector);
    next_.setup(bulk, target_selector, source_selector);
  }

  /// \brief Return true if either rebuilder in the chain signals a rebuild is needed.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    return prior_.needs_rebuild(bulk, targets, sources) || next_.needs_rebuild(bulk, targets, sources);
  }

  /// \brief Snapshot state in both chain members to keep all snapshots current.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    prior_.snapshot(bulk, targets, sources);
    next_.snapshot(bulk, targets, sources);
  }
  //@}

  //! \name Chaining
  //@{

  /// \brief Return a chain of this chain OR-combined with `next`.
  template <RebuilderType Next>
  RebuilderChain<RebuilderChain, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<RebuilderChain, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<RebuilderChain, Next> operator|(const Next& next) const {
    return RebuilderChain<RebuilderChain, Next>(*this, next);
  }
  //@}

  //! \name Accessors
  //@{

  prior_rebuilder_type& prior() noexcept {
    return prior_;
  }
  const prior_rebuilder_type& prior() const noexcept {
    return prior_;
  }
  appended_rebuilder_type& next() noexcept {
    return next_;
  }
  const appended_rebuilder_type& next() const noexcept {
    return next_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  prior_rebuilder_type prior_;
  appended_rebuilder_type next_;
  //@}
};

/// \struct AlwaysRebuild
/// \brief Rebuilder that unconditionally triggers a rebuild on every update.
///
/// Useful as the default policy when a caller wants a fresh list every step, or as
/// a starting point in a rebuilder chain where a later policy may suppress rebuilds.
struct AlwaysRebuild {
  //! \name Rebuild policy
  //@{

  /// \brief No per-update setup needed.
  void setup(const stk::mesh::BulkData& /*bulk*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) noexcept {
  }

  /// \brief Always signal that a rebuild is needed.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                     const SourceInput& /*sources*/) noexcept {
    return true;
  }

  /// \brief No state to snapshot after a rebuild.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                const SourceInput& /*sources*/) noexcept {
  }
  //@}

  //! \name Chaining
  //@{

  /// \brief Return a chain of this rebuilder OR-combined with `next`.
  template <RebuilderType Next>
  RebuilderChain<AlwaysRebuild, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<AlwaysRebuild, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<AlwaysRebuild, Next> operator|(const Next& next) const {
    return RebuilderChain<AlwaysRebuild, Next>(*this, next);
  }
  //@}
};

/// \struct NeverRebuild
/// \brief Rebuilder that suppresses all rebuilds after the first.
///
/// The list is built exactly once on the first `update()` call and then reused
/// unchanged for all subsequent calls regardless of geometry or connectivity changes.
/// Call `ManagedNeighborList::invalidate()` to force a manual rebuild.
struct NeverRebuild {
  //! \name Rebuild policy
  //@{

  /// \brief No per-update setup needed.
  void setup(const stk::mesh::BulkData& /*bulk*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) noexcept {
  }

  /// \brief Never signal that a rebuild is needed after the first build.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                     const SourceInput& /*sources*/) noexcept {
    return false;
  }

  /// \brief No state to snapshot after a rebuild.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                const SourceInput& /*sources*/) noexcept {
  }
  //@}

  //! \name Chaining
  //@{

  /// \brief Return a chain of this rebuilder OR-combined with `next`.
  template <RebuilderType Next>
  RebuilderChain<NeverRebuild, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<NeverRebuild, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<NeverRebuild, Next> operator|(const Next& next) const {
    return RebuilderChain<NeverRebuild, Next>(*this, next);
  }
  //@}
};

/// \class RebuildOnEntityChange
/// \brief Rebuilder that triggers when the target or source entity sequence changes.
///
/// After each build, `snapshot()` records the ordered entity sequences for both inputs (and the mesh's
/// `synchronized_count()` at that point). `needs_rebuild()` reports a rebuild when that sequence changes — a
/// different count, a different entity at any position, or a changed ordering. When the mesh has not been modified
/// since the snapshot (unchanged `synchronized_count()`), it short-circuits to `false` without enumerating or
/// comparing; otherwise the element-wise compare runs on device.
///
/// This is stricter than a count-only check: an add-one / remove-one swap at constant count is
/// detected because the entity at some index will differ.  It is also stricter than an unordered
/// set check: reordering the same entities triggers a rebuild because the ordinal-to-entity
/// mapping embedded in the neighbor list has changed.
template <typename MemorySpace = stk::ngp::MemSpace>
class RebuildOnEntityChange {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  //@}

  //! \name Constructors
  //@{

  RebuildOnEntityChange() = default;
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief No per-update setup needed; entities are enumerated on demand from each input's selector.
  void setup(const stk::mesh::BulkData& /*bulk*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) noexcept {
  }

  /// \brief Return true if the entity sequence differs from the snapshot at the last build.
  ///
  /// On the first call (no snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) return true;
    impl::assert_fixed_selectors(targets.selector(), *snapshot_target_selector_, sources.selector(),
                                 *snapshot_source_selector_);
    // Fast path: with a fixed selector, the entity sequence can only change through a mesh modification, which
    // advances `synchronized_count()`. If it is unchanged since the snapshot, the sequence is provably unchanged —
    // return without enumerating the entities or running the element-wise compare.
    if (bulk.synchronized_count() == snapshot_sync_count_) return false;
    return entities_changed(current_entities(bulk, targets), snapshot_target_) ||
           entities_changed(current_entities(bulk, sources), snapshot_source_);
  }

  /// \brief Snapshot the current entity sequences into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    take_snapshot(current_entities(bulk, targets), snapshot_target_);
    take_snapshot(current_entities(bulk, sources), snapshot_source_);
    snapshot_target_selector_ = host_ptr<stk::mesh::Selector>(targets.selector());
    snapshot_source_selector_ = host_ptr<stk::mesh::Selector>(sources.selector());
    snapshot_sync_count_ = bulk.synchronized_count();
    has_snapshot_ = true;
  }
  //@}

  //! \name Chaining
  //@{

  /// \brief Return a chain of this rebuilder OR-combined with `next`.
  template <RebuilderType Next>
  RebuilderChain<RebuildOnEntityChange, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<RebuildOnEntityChange, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<RebuildOnEntityChange, Next> operator|(const Next& next) const {
    return RebuilderChain<RebuildOnEntityChange, Next>(*this, next);
  }
  //@}

 private:
  //! \name Internal helpers
  //@{

  /// \brief Enumerate the current entities of an input's chunk into a device view (selector order).
  template <typename Input>
  Kokkos::View<stk::mesh::Entity*, memory_space> current_entities(const stk::mesh::BulkData& bulk,
                                                                  const Input& input) const {
    auto ngp_entities = mundy::mesh::get_local_entities(bulk, input.rank(), input.selector(), execution_space{});
    ngp_entities.sync_to_device();
    Kokkos::View<stk::mesh::Entity*, memory_space> entities = ngp_entities.view_device();
    return entities;
  }

  template <typename EntityView>
  bool entities_changed(const EntityView& current, const Kokkos::View<stk::mesh::Entity*, memory_space>& snap) const {
    int n = static_cast<int>(current.extent(0));
    if (n != static_cast<int>(snap.extent(0))) return true;
    if (n == 0) return false;  // Kokkos::Max identity (INT_MIN) != 0 over empty range — guard explicitly
    int any_changed = 0;
    Kokkos::parallel_reduce(
        "mundy_entity_change_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          int changed = (current(i) != snap(i)) ? 1 : 0;
          lmax = lmax > changed ? lmax : changed;
        },
        Kokkos::Max<int>(any_changed));
    Kokkos::fence();
    return any_changed != 0;
  }

  template <typename EntityView>
  void take_snapshot(const EntityView& entities, Kokkos::View<stk::mesh::Entity*, memory_space>& snap) {
    int n = static_cast<int>(entities.extent(0));
    Kokkos::resize(snap, n);
    Kokkos::parallel_for(
        "mundy_entity_snapshot", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { snap(i) = entities(i); });
    Kokkos::fence();
  }
  //@}

  //! \name Internal members
  //@{

  //! Whether a snapshot has been taken (false until first `snapshot()` call).
  bool has_snapshot_ = false;
  //! `BulkData::synchronized_count()` at the last snapshot; an unchanged count means the entity sequence is unchanged.
  size_t snapshot_sync_count_ = 0;
  //! Selectors captured at the last snapshot; `needs_rebuild` must be queried with these same selectors (fixed-list).
  host_ptr<stk::mesh::Selector> snapshot_target_selector_;
  host_ptr<stk::mesh::Selector> snapshot_source_selector_;
  //! Device-resident snapshot of the target entity sequence.
  Kokkos::View<stk::mesh::Entity*, memory_space> snapshot_target_{"mundy_rebuilder_snap_tgt_ent", 0};
  //! Device-resident snapshot of the source entity sequence.
  Kokkos::View<stk::mesh::Entity*, memory_space> snapshot_source_{"mundy_rebuilder_snap_src_ent", 0};
  //@}
};

/// \class RebuildOnAABBDisplacement
/// \brief Rebuilder that triggers when any box corner moves beyond a displacement threshold.
///
/// After each build, `snapshot()` records all six corner coordinates (min/max per axis)
/// for every target and source box.  `needs_rebuild()` computes the per-corner displacement
/// via the supplied `Metric` and returns true when any corner has moved farther than its
/// threshold.
///
/// The `Metric` controls how displacement is measured.  The default `FreeSpaceMetric<Scalar>`
/// gives the raw Cartesian difference, reproducing the original aperiodic behaviour.  Any
/// periodic metric from `mundy_geom/periodicity.hpp` (e.g. `OrthorhombicMetric`,
/// `TriclinicMetric`) applies the minimum-image convention so that a particle crossing a
/// periodic boundary is not counted as having moved by a full cell length.
template <typename Scalar = double, typename MemorySpace = stk::ngp::MemSpace,
          typename Metric = FreeSpaceMetric<Scalar>>
class RebuildOnAABBDisplacement {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using metric_type = Metric;
  using scalar_type = Scalar;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct with a single displacement threshold for both targets and sources.
  ///
  /// Only available when `Metric` is `FreeSpaceMetric` (the aperiodic default). For
  /// periodic simulations or when using a custom metric, supply a metric explicitly via the overloads below.
  /// \param max_displacement [in] Rebuild if any corner moves farther than this.
  explicit RebuildOnAABBDisplacement(Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_max_displacement_(max_displacement), source_max_displacement_(max_displacement) {
  }

  /// \brief Construct with separate thresholds for target and source boxes.
  ///
  /// Only available when `Metric` is `FreeSpaceMetric` (the aperiodic default).
  /// \param target_max_displacement [in] Threshold for target box corners.
  /// \param source_max_displacement [in] Threshold for source box corners.
  RebuildOnAABBDisplacement(Scalar target_max_displacement, Scalar source_max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_max_displacement_(target_max_displacement), source_max_displacement_(source_max_displacement) {
  }

  /// \brief Construct with a single threshold and an explicit metric.
  ///
  /// Use this for periodic simulations where displacement must respect the periodic cell.
  /// \param max_displacement [in] Threshold applied to both targets and sources.
  /// \param metric [in] Metric used to compute minimum-image corner displacement.
  RebuildOnAABBDisplacement(Scalar max_displacement, const Metric& metric)
      : target_max_displacement_(max_displacement), source_max_displacement_(max_displacement), metric_(metric) {
  }

  /// \brief Construct with separate thresholds and an explicit metric.
  ///
  /// \param target_max_displacement [in] Threshold for target box corners.
  /// \param source_max_displacement [in] Threshold for source box corners.
  /// \param metric [in] Metric used to compute minimum-image corner displacement.
  RebuildOnAABBDisplacement(Scalar target_max_displacement, Scalar source_max_displacement, const Metric& metric)
      : target_max_displacement_(target_max_displacement),
        source_max_displacement_(source_max_displacement),
        metric_(metric) {
  }
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief No per-update setup needed; geometry is read on demand from each input's AABB component.
  void setup(const stk::mesh::BulkData& /*bulk*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) noexcept {
  }

  /// \brief Return true if any AABB corner has moved beyond its threshold since the last build.
  ///
  /// Geometry is read from each input's AABB component over its selector. On the first call (no snapshot yet),
  /// always returns true.
  template <typename TargetInput, typename SourceInput>
    requires AABBSearchInputTypeFor<TargetInput, scalar_type> && AABBSearchInputTypeFor<SourceInput, scalar_type>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) {
      return true;
    }
    impl::assert_fixed_selectors(targets.selector(), *snapshot_target_selector_, sources.selector(),
                                 *snapshot_source_selector_);
    // With a fixed selector the entity set can only change via mesh modification, which advances
    // `synchronized_count()`; a changed count therefore invalidates the entity-aligned comparison, so rebuild.
    if (bulk.synchronized_count() != snapshot_sync_count_) {
      return true;
    }

    using TComp = std::remove_cvref_t<decltype(targets.component())>;
    using SComp = std::remove_cvref_t<decltype(sources.component())>;

    // Empty targets or sources → the list is trivially empty, so never signal a (spurious) rebuild. A cheap bucket
    // count (no enumerated index views): the per-side displacement checks below cannot see cross-side emptiness — the
    // union hides a zero side, and the OR short-circuits on the target — so the whole-list-empty test must happen here.
    if (stk::mesh::count_selected_entities(targets.selector(), bulk.buckets(targets.rank())) == 0 ||
        stk::mesh::count_selected_entities(sources.selector(), bulk.buckets(sources.rank())) == 0) {
      return false;
    }

    auto target_live = targets.component();
    TComp target_scratch(*scratch_target_field_);
    auto ngp_target_live = mundy::mesh::get_updated_ngp_component(target_live);
    auto ngp_target_scratch = mundy::mesh::get_updated_ngp_component(target_scratch);

    if (fields_coincide_ && target_max_displacement_ == source_max_displacement_) {
      // Same field AND same threshold → check once over the UNION of the two selectors. The union dedups the
      // intersection and remains valid for disjoint selectors, and "any corner escaped over the union" equals
      // "escaped over targets OR over sources".
      auto union_indices = mundy::mesh::get_local_entity_indices(
          bulk, targets.rank(), targets.selector() | sources.selector(), execution_space{});
      return corners_moved(ngp_target_live, ngp_target_scratch, union_indices, target_max_displacement_);
    } else {
      // Two checks (different fields, or the same field with per-side thresholds). When the fields coincide the source
      // reads the shared target scratch field — the union snapshot populated it over both selectors.
      auto target_indices =
          mundy::mesh::get_local_entity_indices(bulk, targets.rank(), targets.selector(), execution_space{});
      auto source_indices =
          mundy::mesh::get_local_entity_indices(bulk, sources.rank(), sources.selector(), execution_space{});
      auto source_live = sources.component();
      SComp source_scratch(*scratch_source_field_);
      auto ngp_source_live = mundy::mesh::get_updated_ngp_component(source_live);
      auto ngp_source_scratch = mundy::mesh::get_updated_ngp_component(source_scratch);
      return corners_moved(ngp_target_live, ngp_target_scratch, target_indices, target_max_displacement_) ||
             corners_moved(ngp_source_live, ngp_source_scratch, source_indices, source_max_displacement_);
    }
  }

  /// \brief Snapshot the current AABB corners into a companion scratch field (declared on first use).
  template <typename TargetInput, typename SourceInput>
    requires AABBSearchInputTypeFor<TargetInput, scalar_type> && AABBSearchInputTypeFor<SourceInput, scalar_type>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    ensure_scratch_fields(targets, sources);
    auto exec = execution_space{};
    auto target_live = targets.component();
    if (fields_coincide_) {
      // One scratch field for both sides; snapshot it once over the union of the two selectors.
      mundy::mesh::field_copy<scalar_type>(target_live.field(), *scratch_target_field_,
                                           targets.selector() | sources.selector(), exec);
    } else {
      auto source_live = sources.component();
      mundy::mesh::field_copy<scalar_type>(target_live.field(), *scratch_target_field_, targets.selector(), exec);
      mundy::mesh::field_copy<scalar_type>(source_live.field(), *scratch_source_field_, sources.selector(), exec);
    }
    snapshot_target_selector_ = host_ptr<stk::mesh::Selector>(targets.selector());
    snapshot_source_selector_ = host_ptr<stk::mesh::Selector>(sources.selector());
    snapshot_sync_count_ = bulk.synchronized_count();
    has_snapshot_ = true;
  }
  //@}

  //! \name Chaining
  //@{

  /// \brief Return a chain of this rebuilder OR-combined with `next`.
  template <RebuilderType Next>
  RebuilderChain<RebuildOnAABBDisplacement, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<RebuildOnAABBDisplacement, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<RebuildOnAABBDisplacement, Next> operator|(const Next& next) const {
    return RebuilderChain<RebuildOnAABBDisplacement, Next>(*this, next);
  }
  //@}

 private:
  //! \name Internal helpers
  //@{

  /// \brief Declare the scratch field(s) on first use (lazily, since the components arrive at snapshot time).
  ///
  /// Each rebuilder instance owns *unique* scratch field(s) shaped like the live AABB field; on a committed mesh the
  /// declaration is scoped as a late field. When the target and source components read the **same** underlying field,
  /// a single shared scratch field is declared (and snapshot/checked once over the union of the two selectors).
  /// Subsequent calls are no-ops.
  template <typename TargetInput, typename SourceInput>
  void ensure_scratch_fields(const TargetInput& targets, const SourceInput& sources) {
    if (scratch_target_field_ != nullptr) {
      return;
    }
    auto target_live = targets.component();
    auto source_live = sources.component();
    fields_coincide_ = (&target_live.field() == &source_live.field());
    stk::mesh::MetaData& meta = target_live.field().mesh_meta_data();
    impl::ScopedLateFields late_fields(meta);
    scratch_target_field_ =
        &mundy::mesh::impl::declare_unique_field_like<scalar_type>(meta, target_live.field(), "rebuild_snapshot");
    if (fields_coincide_) {
      scratch_source_field_ = scratch_target_field_;
    } else {
      scratch_source_field_ =
          &mundy::mesh::impl::declare_unique_field_like<scalar_type>(meta, source_live.field(), "rebuild_snapshot");
    }
  }

  /// \brief Return true if any AABB corner has moved beyond `threshold` under the metric.
  ///
  /// `ngp_live` holds the current corners and `ngp_scratch` the snapshot corners; both are read at the same
  /// FastMeshIndex per entity. For `FreeSpaceMetric` this is the raw per-scalar absolute difference; for periodic
  /// metrics the minimum-image displacement is used, so a corner that wraps across the cell boundary is not counted
  /// as having moved by a full cell length.
  template <typename NgpComponent, typename IndexView>
  bool corners_moved(NgpComponent ngp_live, NgpComponent ngp_scratch, IndexView indices, scalar_type threshold) const {
    int n = static_cast<int>(indices.extent(0));
    if (n == 0) {
      return false;
    }
    ngp_live.sync_to_device();
    ngp_scratch.sync_to_device();
    indices.sync_to_device();
    auto d_indices = indices.view_device();
    auto met = metric_;  // device-capturable by value; all concrete metrics are trivially copyable
    int any_moved = 0;
    Kokkos::parallel_reduce(
        "mundy_aabb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const auto aabb = ngp_live(d_indices(i));
          const auto aabb_old = ngp_scratch(d_indices(i));
          const Point<scalar_type> old_min{aabb_old.min_corner()[0], aabb_old.min_corner()[1], aabb_old.min_corner()[2]};
          const Point<scalar_type> new_min{aabb.min_corner()[0], aabb.min_corner()[1], aabb.min_corner()[2]};
          const Point<scalar_type> old_max{aabb_old.max_corner()[0], aabb_old.max_corner()[1], aabb_old.max_corner()[2]};
          const Point<scalar_type> new_max{aabb.max_corner()[0], aabb.max_corner()[1], aabb.max_corner()[2]};
          // met.sep gives the minimum-image displacement (identity for FreeSpaceMetric).
          const auto d_min = met.sep(old_min, new_min);
          const auto d_max = met.sep(old_max, new_max);
          int moved = (abs(d_min[0]) > threshold ? 1 : 0) | (abs(d_min[1]) > threshold ? 1 : 0) |
                      (abs(d_min[2]) > threshold ? 1 : 0) | (abs(d_max[0]) > threshold ? 1 : 0) |
                      (abs(d_max[1]) > threshold ? 1 : 0) | (abs(d_max[2]) > threshold ? 1 : 0);
          lmax = lmax > moved ? lmax : moved;
        },
        Kokkos::Max<int>(any_moved));
    Kokkos::fence();
    return any_moved != 0;
  }
  //@}

  //! \name Internal members
  //@{

  //! Whether snapshot() has been called at least once.
  bool has_snapshot_ = false;
  //! Per-corner displacement threshold for target boxes.
  scalar_type target_max_displacement_;
  //! Per-corner displacement threshold for source boxes.
  scalar_type source_max_displacement_;
  //! Metric used to compute minimum-image corner displacement.  Default is FreeSpaceMetric<Scalar>.
  Metric metric_{};
  //! `BulkData::synchronized_count()` at the last snapshot; a changed count invalidates the entity-aligned compare.
  size_t snapshot_sync_count_ = 0;
  //! Selectors captured at the last snapshot; `needs_rebuild` must be queried with these same selectors (fixed-list).
  host_ptr<stk::mesh::Selector> snapshot_target_selector_;
  host_ptr<stk::mesh::Selector> snapshot_source_selector_;
  //! Whether the target and source components read the same field → one shared scratch field, union snapshot/check.
  bool fields_coincide_ = false;
  //! Unique scratch field holding the target corners at the last snapshot (declared on first snapshot).
  stk::mesh::Field<scalar_type>* scratch_target_field_ = nullptr;
  //! Unique scratch field holding the source corners (declared on first snapshot).
  stk::mesh::Field<scalar_type>* scratch_source_field_ = nullptr;
  //@}
};

/// \class RebuildOnOBBDisplacement
/// \brief Rebuilder that triggers when any OBB escapes its inflated snapshot.
///
/// After each build, `snapshot()` records every target and source OBB.  `needs_rebuild()`
/// checks whether each current OBB is still fully contained within the inflated snapshot OBB
/// (half-extents expanded by the displacement threshold `d`).
///
/// \par Containment check
/// For a pair of OBBs `obb_old` (snapshot) and `obb_new` (current), the check is
/// `obb_new ⊆ expand(obb_old, d)`.  Using `obb_old`'s three face normals as projection axes —
/// sufficient for containment in any convex body — this reduces to:
///
/// \code
///   T     = R_old^T * (c_new - c_old)          // center displacement in obb_old's local frame
///   R_rel = R_old^T * R_new                    // relative rotation matrix
///   for axis i = 0, 1, 2:
///       |T[i]| + sum_k |R_rel(i,k)| * h_new[k]  <=  h_old[i] + d
/// \endcode
///
/// All eight corners of `obb_new` are implicitly checked: the first term is the center
/// displacement and the second is the maximum extent of `obb_new` along axis `i`, achieved
/// at the worst-case corner.  If the inequality fails for any axis, the OBB has escaped its
/// inflated containment region and a rebuild is required.
///
/// Note that the check uses only the face normals of the snapshot OBB (not all 15 SAT axes).
/// This is correct because containment in a convex set is determined entirely by its own
/// supporting halfspaces — the three axis-pairs that define the OBB.
///
/// \par Relationship to RebuildOnAABBDisplacement
/// For axis-aligned boxes, the containment check above degenerates to the per-corner scalar
/// displacement check in `RebuildOnAABBDisplacement`: axes are fixed so T[i] = corner_disp[i]
/// and R_rel = I.  `RebuildOnOBBDisplacement` is the correct generalisation when orientations
/// can change between rebuilds.
///
/// \par Input and views
/// Unlike `RebuildOnAABBDisplacement` (which reads boxes from the search input), OBBs are not
/// part of the standard search input and must be supplied as caller-owned Kokkos views at
/// construction.  The user is responsible for keeping those views up to date before each call.
///
template <typename Scalar = double, typename MemorySpace = stk::ngp::MemSpace,
          typename Metric = FreeSpaceMetric<Scalar>>
class RebuildOnOBBDisplacement {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using metric_type = Metric;
  using scalar_type = Scalar;
  using obb_type = OBB<Scalar>;
  using obb_component_type = mundy::mesh::OBBFieldComponent<Scalar>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct with a symmetric OBB component and a single threshold (aperiodic).
  ///
  /// The same OBB component is read for both target and source sides (over each input's selector).  Available
  /// only for the default aperiodic `FreeSpaceMetric`; supply a metric explicitly for periodic simulations.
  /// \param obbs             [in] OBB component supplying each entity's OBB; shared by both sides.
  /// \param max_displacement [in] Rebuild when any OBB corner exits its inflated snapshot.
  explicit RebuildOnOBBDisplacement(obb_component_type obbs, Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : RebuildOnOBBDisplacement(obbs, obbs, max_displacement, max_displacement, Metric{}) {
  }

  /// \brief Construct with separate target/source OBB components and a single threshold (aperiodic).
  ///
  /// \param target_obbs     [in] OBB component for target entities.
  /// \param source_obbs     [in] OBB component for source entities.
  /// \param max_displacement [in] Threshold applied to both targets and sources.
  RebuildOnOBBDisplacement(obb_component_type target_obbs, obb_component_type source_obbs, Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : RebuildOnOBBDisplacement(target_obbs, source_obbs, max_displacement, max_displacement, Metric{}) {
  }

  /// \brief Construct with separate target/source OBB components and asymmetric thresholds (aperiodic).
  ///
  /// \param target_obbs              [in] OBB component for target entities.
  /// \param source_obbs              [in] OBB component for source entities.
  /// \param target_max_displacement  [in] Threshold for target OBBs.
  /// \param source_max_displacement  [in] Threshold for source OBBs.
  RebuildOnOBBDisplacement(obb_component_type target_obbs, obb_component_type source_obbs,
                           Scalar target_max_displacement, Scalar source_max_displacement)
    requires is_free_space_metric_v<Metric>
      : RebuildOnOBBDisplacement(target_obbs, source_obbs, target_max_displacement, source_max_displacement, Metric{}) {
  }

  /// \brief Construct with a symmetric OBB component, a single threshold, and an explicit metric.
  ///
  /// Use this for periodic simulations where the center-displacement term must respect
  /// the periodic cell geometry.
  /// \param obbs             [in] OBB component shared by both target and source sides.
  /// \param max_displacement [in] Threshold applied to both sides.
  /// \param metric           [in] Metric used to compute minimum-image center displacement.
  RebuildOnOBBDisplacement(obb_component_type obbs, Scalar max_displacement, const Metric& metric)
      : RebuildOnOBBDisplacement(obbs, obbs, max_displacement, max_displacement, metric) {
  }

  /// \brief Construct with separate target/source OBB components, asymmetric thresholds, and a metric.
  ///
  /// This is the canonical constructor that all others delegate to. Because the OBB components — hence their fields
  /// and metadata — are available here, the per-side scratch snapshot field(s) are declared now: one shared field
  /// when both sides read the same field, otherwise one field per side. On a committed mesh the declaration is
  /// scoped as a late field.
  /// \param target_obbs              [in] OBB component for target entities.
  /// \param source_obbs              [in] OBB component for source entities.
  /// \param target_max_displacement  [in] Threshold for target OBBs.
  /// \param source_max_displacement  [in] Threshold for source OBBs.
  /// \param metric                   [in] Metric for minimum-image center displacement.
  RebuildOnOBBDisplacement(obb_component_type target_obbs, obb_component_type source_obbs,
                           Scalar target_max_displacement, Scalar source_max_displacement, const Metric& metric)
      : target_obb_component_(target_obbs),
        source_obb_component_(source_obbs),
        target_max_displacement_(target_max_displacement),
        source_max_displacement_(source_max_displacement),
        metric_(metric) {
    // Target and source components live on the same mesh, so one metadata reference covers both.
    stk::mesh::MetaData& meta = target_obb_component_.field().mesh_meta_data();
    fields_coincide_ = (&target_obb_component_.field() == &source_obb_component_.field());
    impl::ScopedLateFields late_fields(meta);
    scratch_target_field_ = &mundy::mesh::impl::declare_unique_field_like<scalar_type>(
        meta, target_obb_component_.field(), "rebuild_snapshot");
    scratch_source_field_ = fields_coincide_ ? scratch_target_field_
                                             : &mundy::mesh::impl::declare_unique_field_like<scalar_type>(
                                                   meta, source_obb_component_.field(), "rebuild_snapshot");
  }
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief No per-update setup needed; OBBs are read on demand from the stored OBB components.
  void setup(const stk::mesh::BulkData& /*bulk*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) noexcept {
  }

  /// \brief Return true if any OBB has escaped its inflated snapshot containment region.
  ///
  /// The current OBBs (from the stored components) are compared against the snapshot OBBs (held in the per-side
  /// scratch fields), read at the same FastMeshIndex per entity over each input's selector. On the first call (no
  /// snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
    requires NeighborListInputType<TargetInput> && NeighborListInputType<SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) return true;
    impl::assert_fixed_selectors(targets.selector(), *snapshot_target_selector_, sources.selector(),
                                 *snapshot_source_selector_);
    // With a fixed selector the entity set can only change via mesh modification, which advances
    // `synchronized_count()`; a changed count therefore invalidates the entity-aligned comparison, so rebuild.
    if (bulk.synchronized_count() != snapshot_sync_count_) return true;

    // Empty targets or sources → the list is trivially empty, so never signal a (spurious) rebuild. A cheap bucket
    // count (no enumerated index views): the per-side checks below cannot see cross-side emptiness — the union hides a
    // zero side, and the OR short-circuits on the target — so the whole-list-empty test must happen here.
    if (stk::mesh::count_selected_entities(targets.selector(), bulk.buckets(targets.rank())) == 0 ||
        stk::mesh::count_selected_entities(sources.selector(), bulk.buckets(sources.rank())) == 0) {
      return false;
    }

    auto target_live = target_obb_component_;
    obb_component_type target_scratch(*scratch_target_field_);
    auto ngp_target_live = mundy::mesh::get_updated_ngp_component(target_live);
    auto ngp_target_scratch = mundy::mesh::get_updated_ngp_component(target_scratch);

    if (fields_coincide_ && target_max_displacement_ == source_max_displacement_) {
      // Same field AND same threshold → check once over the UNION of the two selectors. The union dedups the
      // intersection and remains valid for disjoint selectors, and "any OBB escaped over the union" equals
      // "escaped over targets OR over sources".
      auto union_indices = mundy::mesh::get_local_entity_indices(
          bulk, targets.rank(), targets.selector() | sources.selector(), execution_space{});
      return obbs_escaped(ngp_target_live, ngp_target_scratch, union_indices, target_max_displacement_);
    } else {
      // Two checks (different fields, or the same field with per-side thresholds). When the fields coincide the source
      // reads the shared target scratch field — the union snapshot populated it over both selectors.
      auto target_indices =
          mundy::mesh::get_local_entity_indices(bulk, targets.rank(), targets.selector(), execution_space{});
      auto source_indices =
          mundy::mesh::get_local_entity_indices(bulk, sources.rank(), sources.selector(), execution_space{});
      obb_component_type source_scratch(*scratch_source_field_);
      auto ngp_source_live = mundy::mesh::get_updated_ngp_component(source_obb_component_);
      auto ngp_source_scratch = mundy::mesh::get_updated_ngp_component(source_scratch);
      return obbs_escaped(ngp_target_live, ngp_target_scratch, target_indices, target_max_displacement_) ||
             obbs_escaped(ngp_source_live, ngp_source_scratch, source_indices, source_max_displacement_);
    }
  }

  /// \brief Snapshot the current OBBs into the per-side scratch fields.
  template <typename TargetInput, typename SourceInput>
    requires NeighborListInputType<TargetInput> && NeighborListInputType<SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    auto exec = execution_space{};
    if (fields_coincide_) {
      // One scratch field for both sides; snapshot it once over the union of the two selectors.
      mundy::mesh::field_copy<scalar_type>(target_obb_component_.field(), *scratch_target_field_,
                                           targets.selector() | sources.selector(), exec);
    } else {
      mundy::mesh::field_copy<scalar_type>(target_obb_component_.field(), *scratch_target_field_, targets.selector(),
                                           exec);
      mundy::mesh::field_copy<scalar_type>(source_obb_component_.field(), *scratch_source_field_, sources.selector(),
                                           exec);
    }
    snapshot_target_selector_ = host_ptr<stk::mesh::Selector>(targets.selector());
    snapshot_source_selector_ = host_ptr<stk::mesh::Selector>(sources.selector());
    snapshot_sync_count_ = bulk.synchronized_count();
    has_snapshot_ = true;
  }
  //@}

  //! \name Chaining
  //@{

  template <RebuilderType Next>
  RebuilderChain<RebuildOnOBBDisplacement, Next> rebuild_if(const Next& next) const {
    return RebuilderChain<RebuildOnOBBDisplacement, Next>(*this, next);
  }

  template <RebuilderType Next>
  RebuilderChain<RebuildOnOBBDisplacement, Next> operator|(const Next& next) const {
    return RebuilderChain<RebuildOnOBBDisplacement, Next>(*this, next);
  }
  //@}

 private:
  //! \name Internal helpers
  //@{

  /// \brief Return true if any current OBB has escaped its inflated snapshot entry.
  ///
  /// `ngp_live` holds the current OBBs and `ngp_scratch` the snapshot OBBs; both are read at the same FastMeshIndex
  /// per entity. For each entity i the containment check is (in obb_old's local frame):
  ///   for axis k:  |T[k]| + sum_j |R_rel(k,j)| * h_new[j]  <=  h_old[k] + threshold
  /// where T = R_old^T * sep(c_old, c_new) and R_rel = R_old^T * R_new.
  template <typename NgpComponent, typename IndexView>
  bool obbs_escaped(NgpComponent ngp_live, NgpComponent ngp_scratch, IndexView indices, Scalar threshold) const {
    const int n = static_cast<int>(indices.extent(0));
    if (n == 0) return false;
    ngp_live.sync_to_device();
    ngp_scratch.sync_to_device();
    indices.sync_to_device();
    auto d_indices = indices.view_device();
    auto met = metric_;
    int any_escaped = 0;
    Kokkos::parallel_reduce(
        "mundy_obb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const auto obb_old = ngp_scratch(d_indices(i));
          const auto obb_new = ngp_live(d_indices(i));

          // Center displacement in world frame, minimum-image for periodic metrics.
          const auto T_world = met.sep(obb_old.center(), obb_new.center());

          // Express center displacement in obb_old's local frame.
          const Vector3<Scalar> T = conjugate(obb_old.orientation()) * T_world;

          // Relative rotation: R_old^T * R_new.
          const auto R_rel = quaternion_to_rotation_matrix(conjugate(obb_old.orientation()) * obb_new.orientation());

          // Check containment along each of obb_old's face normals.
          // The maximum extent of obb_new along local axis k is |T[k]| + sum_j |R_rel(k,j)| * h_new[j].
          // If this exceeds h_old[k] + threshold, obb_new has escaped the inflated snapshot.
          int escaped = 0;
          for (int k = 0; k < 3; ++k) {
            Scalar extent = abs(T[k]);
            for (int j = 0; j < 3; ++j) extent += abs(R_rel(k, j)) * obb_new.half_extent(j);
            if (extent > obb_old.half_extent(k) + threshold) {
              escaped = 1;
              break;
            }
          }
          lmax = lmax > escaped ? lmax : escaped;
        },
        Kokkos::Max<int>(any_escaped));
    Kokkos::fence();
    return any_escaped != 0;
  }
  //@}

  //! \name Internal members
  //@{

  //! Whether snapshot() has been called at least once.
  bool has_snapshot_ = false;
  //! OBB component supplying target-entity OBBs (read over the target input's selector).
  obb_component_type target_obb_component_;
  //! OBB component supplying source-entity OBBs (read over the source input's selector).
  obb_component_type source_obb_component_;
  //! Per-entity displacement threshold for target OBBs.
  Scalar target_max_displacement_ = 0;
  //! Per-entity displacement threshold for source OBBs.
  Scalar source_max_displacement_ = 0;
  //! Metric used for center-displacement (minimum-image for periodic metrics).
  Metric metric_{};
  //! `BulkData::synchronized_count()` at the last snapshot; a changed count invalidates the entity-aligned compare.
  size_t snapshot_sync_count_ = 0;
  //! Selectors captured at the last snapshot; `needs_rebuild` must be queried with these same selectors (fixed-list).
  host_ptr<stk::mesh::Selector> snapshot_target_selector_;
  host_ptr<stk::mesh::Selector> snapshot_source_selector_;
  //! Whether the target and source components read the same field → one shared scratch field, union snapshot/check.
  bool fields_coincide_ = false;
  //! Unique scratch field holding the target OBBs at the last snapshot (declared at construction).
  stk::mesh::Field<scalar_type>* scratch_target_field_ = nullptr;
  //! Unique scratch field holding the source OBBs (aliases the target field when `fields_coincide_`).
  stk::mesh::Field<scalar_type>* scratch_source_field_ = nullptr;
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTREBUILDER_HPP_
