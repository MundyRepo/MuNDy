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
#include <concepts>  // for std::convertible_to, std::same_as
#include <cstddef>   // for size_t
#include <limits>    // for std::numeric_limits

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_geom/periodicity.hpp>     // for FreeSpaceMetric
#include <mundy_geom/primitives/OBB.hpp>  // for mundy::OBB, mundy::quaternion_to_rotation_matrix
#include <mundy_math/Quaternion.hpp>      // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>         // for mundy::Vector3
#include <mundy_math/cmath.hpp>
#include <mundy_mesh/EntityIndices.hpp>  // for mundy::mesh::get_local_entities, get_local_entity_indices
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::AABBFieldComponent/OBBFieldComponent, get_updated_ngp_component
#include <mundy_search/NeighborListBuildTraits.hpp>  // for AABBSearchInputTypeFor
#include <mundy_search/SearchInput.hpp>              // for SearchInput

namespace mundy {

namespace search {

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
///
/// \tparam PriorRebuilder Prior rebuilder in the chain.
/// \tparam Rebuilder Newly appended rebuilder.
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
/// After each build, `snapshot()` stores the ordered entity views for both inputs using a
/// Kokkos `parallel_for` on device. `needs_rebuild()` checks element-wise identity — same count
/// AND same entity at every position — using a `parallel_reduce` on device. Any change in count,
/// entity identity, or entity ordering triggers a rebuild.
///
/// This is stricter than a count-only check: an add-one / remove-one swap at constant count is
/// detected because the entity at some index will differ.  It is also stricter than an unordered
/// set check: reordering the same entities triggers a rebuild because the ordinal-to-entity
/// mapping embedded in the neighbor list has changed.
///
/// \tparam MemorySpace Kokkos memory space used for device-resident snapshots and kernels.
///   Must match the memory space of the entity views supplied to `needs_rebuild` and `snapshot`.
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
    return entities_changed(current_entities(bulk, targets), snapshot_target_) ||
           entities_changed(current_entities(bulk, sources), snapshot_source_);
  }

  /// \brief Snapshot the current entity sequences into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    take_snapshot(current_entities(bulk, targets), snapshot_target_);
    take_snapshot(current_entities(bulk, sources), snapshot_source_);
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
  //! Device-resident snapshot of the target entity sequence.
  Kokkos::View<stk::mesh::Entity*, memory_space> snapshot_target_{"mundy_rebuilder_snap_tgt_ent", 0};
  //! Device-resident snapshot of the source entity sequence.
  Kokkos::View<stk::mesh::Entity*, memory_space> snapshot_source_{"mundy_rebuilder_snap_src_ent", 0};
  //@}
};

namespace impl {

/// \struct ComponentGeometry
/// \brief A component's on-device geometry readout: the NGP component paired with the index view it is read at.
template <typename NgpComponent, typename IndexView>
struct ComponentGeometry {
  NgpComponent ngp_component;  //!< NGP component; evaluate at an index to read that entity's primitive.
  IndexView indices;           //!< Device FastMeshIndex view, in selector order.
};

}  // namespace impl

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
///
/// \tparam Scalar      Floating-point type for the displacement threshold and snapshot storage.
///   Defaults to `double`.
/// \tparam MemorySpace Kokkos memory space for device-resident snapshots and kernels.
/// \tparam Metric      Distance metric used for corner-displacement measurement.
///   Defaults to `FreeSpaceMetric<Scalar>` (aperiodic, raw Cartesian difference).
///   Any concrete metric type from `mundy_geom/periodicity.hpp` is valid.
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
    auto target_geom = current_geometry(bulk, targets);
    auto source_geom = current_geometry(bulk, sources);
    const int nt = static_cast<int>(target_geom.indices.extent(0));
    const int ns = static_cast<int>(source_geom.indices.extent(0));
    // Empty targets or sources → result is always empty regardless of geometry; skip displacement check.
    if (nt == 0 || ns == 0) {
      return false;
    }
    if (nt * 6 != static_cast<int>(snapshot_target_.extent(0)) ||
        ns * 6 != static_cast<int>(snapshot_source_.extent(0))) {
      return true;
    }
    return corners_moved(target_geom.ngp_component, target_geom.indices, snapshot_target_, target_max_displacement_) ||
           corners_moved(source_geom.ngp_component, source_geom.indices, snapshot_source_, source_max_displacement_);
  }

  /// \brief Snapshot the current AABB corners into device-resident storage.
  template <typename TargetInput, typename SourceInput>
    requires AABBSearchInputTypeFor<TargetInput, scalar_type> && AABBSearchInputTypeFor<SourceInput, scalar_type>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    auto target_geom = current_geometry(bulk, targets);
    auto source_geom = current_geometry(bulk, sources);
    take_snapshot(target_geom.ngp_component, target_geom.indices, snapshot_target_);
    take_snapshot(source_geom.ngp_component, source_geom.indices, snapshot_source_);
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

  /// \brief Enumerate an input's chunk and prepare to read its AABB geometry on device.
  template <typename Input>
  auto current_geometry(const stk::mesh::BulkData& bulk, const Input& input) const {
    auto component = input.component();  // copy → non-const, so the field can be synced to device
    component.sync_to_device();
    auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
    auto indices = mundy::mesh::get_local_entity_indices(bulk, input.rank(), input.selector(), execution_space{});
    indices.sync_to_device();
    return impl::ComponentGeometry{ngp_component, indices.view_device()};
  }

  /// \brief Return true if any AABB corner has moved beyond `threshold` under the metric.
  ///
  /// For `FreeSpaceMetric` this is identical to the raw per-scalar absolute difference.
  /// For periodic metrics the minimum-image displacement is used, so a corner that wraps
  /// across the cell boundary is not counted as having moved by a full cell length.
  template <typename NgpComponent, typename IndexView>
  bool corners_moved(const NgpComponent& ngp_component, const IndexView& indices,
                     const Kokkos::View<scalar_type*, memory_space>& snapshot, scalar_type threshold) const {
    int n = static_cast<int>(indices.extent(0));
    if (n == 0) {
      return false;
    }
    auto snap = snapshot;
    auto met = metric_;  // device-capturable by value; all concrete metrics are trivially copyable
    int any_moved = 0;
    Kokkos::parallel_reduce(
        "mundy_aabb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const auto aabb = ngp_component(indices(i));
          const int base = 6 * i;
          const Point<scalar_type> old_min{snap(base + 0), snap(base + 1), snap(base + 2)};
          const Point<scalar_type> new_min{aabb.min_corner()[0], aabb.min_corner()[1], aabb.min_corner()[2]};
          const Point<scalar_type> old_max{snap(base + 3), snap(base + 4), snap(base + 5)};
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

  /// \brief Snapshot AABB corners into a device-resident view (6 scalars per entity: min xyz, max xyz).
  template <typename NgpComponent, typename IndexView>
  void take_snapshot(const NgpComponent& ngp_component, const IndexView& indices,
                     Kokkos::View<scalar_type*, memory_space>& snapshot) {
    int n = static_cast<int>(indices.extent(0));
    Kokkos::resize(snapshot, 6 * n);
    auto snap = snapshot;
    Kokkos::parallel_for(
        "mundy_aabb_snapshot", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) {
          const auto aabb = ngp_component(indices(i));
          snap(6 * i + 0) = aabb.min_corner()[0];
          snap(6 * i + 1) = aabb.min_corner()[1];
          snap(6 * i + 2) = aabb.min_corner()[2];
          snap(6 * i + 3) = aabb.max_corner()[0];
          snap(6 * i + 4) = aabb.max_corner()[1];
          snap(6 * i + 5) = aabb.max_corner()[2];
        });
    Kokkos::fence();
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
  //! Snapshot of target box corners (6 scalars per box: min_xyz then max_xyz).
  Kokkos::View<scalar_type*, memory_space> snapshot_target_{"mundy_rebuilder_snap_tgt", 0};
  //! Snapshot of source box corners (6 scalars per box: min_xyz then max_xyz).
  Kokkos::View<scalar_type*, memory_space> snapshot_source_{"mundy_rebuilder_snap_src", 0};
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
/// \tparam Scalar      Floating-point type of the OBBs and the displacement threshold.
/// \tparam MemorySpace Kokkos memory space for device-resident OBB views and snapshots.
/// \tparam Metric      Distance metric for the center-displacement term.
///   Defaults to `FreeSpaceMetric<double>` (Cartesian, aperiodic).
///   Any periodic metric from `mundy_geom/periodicity.hpp` applies the minimum-image
///   convention to the center displacement so that boundary-crossing is handled correctly.
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
      : target_obb_component_(obbs),
        source_obb_component_(obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement) {
  }

  /// \brief Construct with separate target/source OBB components and a single threshold (aperiodic).
  ///
  /// \param target_obbs     [in] OBB component for target entities.
  /// \param source_obbs     [in] OBB component for source entities.
  /// \param max_displacement [in] Threshold applied to both targets and sources.
  RebuildOnOBBDisplacement(obb_component_type target_obbs, obb_component_type source_obbs, Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_obb_component_(target_obbs),
        source_obb_component_(source_obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement) {
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
      : target_obb_component_(target_obbs),
        source_obb_component_(source_obbs),
        target_max_displacement_(target_max_displacement),
        source_max_displacement_(source_max_displacement) {
  }

  /// \brief Construct with a symmetric OBB component, a single threshold, and an explicit metric.
  ///
  /// Use this for periodic simulations where the center-displacement term must respect
  /// the periodic cell geometry.
  /// \param obbs             [in] OBB component shared by both target and source sides.
  /// \param max_displacement [in] Threshold applied to both sides.
  /// \param metric           [in] Metric used to compute minimum-image center displacement.
  RebuildOnOBBDisplacement(obb_component_type obbs, Scalar max_displacement, const Metric& metric)
      : target_obb_component_(obbs),
        source_obb_component_(obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement),
        metric_(metric) {
  }

  /// \brief Construct with separate target/source OBB components, asymmetric thresholds, and a metric.
  ///
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
  /// OBBs are read from the stored OBB components over each input's selector. On the first call (no snapshot
  /// yet), always returns true.
  template <typename TargetInput, typename SourceInput>
    requires NeighborListInputType<TargetInput> && NeighborListInputType<SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) return true;
    auto target_geom = current_geometry(bulk, target_obb_component_, targets);
    auto source_geom = current_geometry(bulk, source_obb_component_, sources);
    if (target_geom.indices.extent(0) != snapshot_target_.extent(0) ||
        source_geom.indices.extent(0) != snapshot_source_.extent(0))
      return true;
    return obbs_escaped(target_geom.ngp_component, target_geom.indices, snapshot_target_, target_max_displacement_) ||
           obbs_escaped(source_geom.ngp_component, source_geom.indices, snapshot_source_, source_max_displacement_);
  }

  /// \brief Snapshot the current OBBs into device-resident storage.
  template <typename TargetInput, typename SourceInput>
    requires NeighborListInputType<TargetInput> && NeighborListInputType<SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets, const SourceInput& sources) {
    auto target_geom = current_geometry(bulk, target_obb_component_, targets);
    auto source_geom = current_geometry(bulk, source_obb_component_, sources);
    take_snapshot(target_geom.ngp_component, target_geom.indices, snapshot_target_);
    take_snapshot(source_geom.ngp_component, source_geom.indices, snapshot_source_);
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

  /// \brief Enumerate an input's chunk and prepare to read its OBB geometry on device.
  template <typename Input>
  auto current_geometry(const stk::mesh::BulkData& bulk, obb_component_type component, const Input& input) const {
    component.sync_to_device();
    auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
    auto indices = mundy::mesh::get_local_entity_indices(bulk, input.rank(), input.selector(), execution_space{});
    indices.sync_to_device();
    return impl::ComponentGeometry{ngp_component, indices.view_device()};
  }

  /// \brief Return true if any current OBB has escaped its inflated snapshot entry.
  ///
  /// For each entity i the containment check is (in obb_old's local frame):
  ///   for axis k:  |T[k]| + sum_j |R_rel(k,j)| * h_new[j]  <=  h_old[k] + threshold
  /// where T = R_old^T * sep(c_old, c_new) and R_rel = R_old^T * R_new.
  template <typename NgpComponent, typename IndexView>
  bool obbs_escaped(const NgpComponent& ngp_component, const IndexView& indices,
                    const Kokkos::View<obb_type*, MemorySpace>& snap, Scalar threshold) const {
    const int n = static_cast<int>(indices.extent(0));
    if (n == 0) return false;
    auto met = metric_;
    int any_escaped = 0;
    Kokkos::parallel_reduce(
        "mundy_obb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const obb_type& obb_old = snap(i);
          const auto obb_new = ngp_component(indices(i));

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

  /// \brief Snapshot OBBs (read from the component) into a device-resident view.
  template <typename NgpComponent, typename IndexView>
  void take_snapshot(const NgpComponent& ngp_component, const IndexView& indices,
                     Kokkos::View<obb_type*, MemorySpace>& snap) {
    const int n = static_cast<int>(indices.extent(0));
    Kokkos::resize(snap, n);
    auto s = snap;
    Kokkos::parallel_for(
        "mundy_obb_snapshot", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) {
          const auto v = ngp_component(indices(i));
          // Materialize the field-view OBB into an owning OBB<Scalar> (coordinate-wise to avoid cross-type ctors).
          s(i) = obb_type(
              Point<Scalar>{v.center()[0], v.center()[1], v.center()[2]},
              Quaternion<Scalar>{v.orientation().w(), v.orientation().x(), v.orientation().y(), v.orientation().z()},
              Vector3<Scalar>{v.half_extents()[0], v.half_extents()[1], v.half_extents()[2]});
        });
    Kokkos::fence();
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
  //! Device-resident snapshot of target OBBs at the last rebuild.
  Kokkos::View<obb_type*, MemorySpace> snapshot_target_{"mundy_rebuilder_snap_tgt_obb", 0};
  //! Device-resident snapshot of source OBBs at the last rebuild.
  Kokkos::View<obb_type*, MemorySpace> snapshot_source_{"mundy_rebuilder_snap_src_obb", 0};
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTREBUILDER_HPP_
