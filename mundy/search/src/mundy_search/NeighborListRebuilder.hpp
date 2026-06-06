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
#include <mundy_geom/periodicity.hpp>        // for FreeSpaceMetric (aperiodic default)
#include <mundy_geom/primitives/OBB.hpp>     // for mundy::OBB, mundy::quaternion_to_rotation_matrix
#include <mundy_math/Vector3.hpp>            // for mundy::Vector3 (point construction in corners_moved)
#include <mundy_math/cmath.hpp>
#include <mundy_search/impl/STKSearchBoxes.hpp>  // for impl::STKSearchBoxesT (concept representative)

namespace mundy {

namespace search {

/// \concept RebuilderType
/// \brief Specifies a stateful policy that decides when a neighbor list needs to be rebuilt.
///
/// Rebuilders are stored in `ManagedNeighborList` and consulted before each `update()` call.
/// `needs_rebuild(...)` returns true when the list should be discarded and rebuilt.
/// `snapshot(...)` is called after every successful build so the rebuilder can snapshot
/// whatever state it uses to detect staleness.
///
/// Both methods receive the full target and source input objects (not just selectors) so that
/// geometry-aware rebuilders can inspect box views directly on whatever memory space they live on.
/// The concept is checked against `impl::STKSearchBoxesT<stk::ngp::MemSpace, float>` as a
/// representative input type.
template <typename T>
concept RebuilderType = requires(T& rebuilder, const stk::mesh::BulkData& bulk_data,
                                 const impl::STKSearchBoxesT<stk::ngp::MemSpace, float>& input) {
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

  /// \brief Return true if the entity sequence differs from the snapshot at the last build.
  ///
  /// On the first call (no snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) return true;
    return entities_changed(targets.entities(), snapshot_target_) ||
           entities_changed(sources.entities(), snapshot_source_);
  }

  /// \brief Snapshot the current entity sequences into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets, const SourceInput& sources) {
    take_snapshot(targets.entities(), snapshot_target_);
    take_snapshot(sources.entities(), snapshot_source_);
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

  /// \brief Return true if any box corner has moved beyond its threshold since the last build.
  ///
  /// On the first call (no snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets, const SourceInput& sources) {
    if (!has_snapshot_) return true;
    // Empty targets or sources → result is always empty regardless of geometry; skip displacement check.
    if (targets.boxes().extent(0) == 0 || sources.boxes().extent(0) == 0) return false;
    if (targets.boxes().extent(0) * 6 != snapshot_target_.extent(0) ||
        sources.boxes().extent(0) * 6 != snapshot_source_.extent(0))
      return true;
    return corners_moved(targets.boxes(), snapshot_target_, target_max_displacement_) ||
           corners_moved(sources.boxes(), snapshot_source_, source_max_displacement_);
  }

  /// \brief Snapshot the current box corners into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets, const SourceInput& sources) {
    take_snapshot(targets.boxes(), snapshot_target_);
    take_snapshot(sources.boxes(), snapshot_source_);
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

  /// \brief Uniform min-corner accessor for both ArborX::Box (.minCorner()) and STK boxes (.min_corner()).
  template <typename BoxT>
  KOKKOS_INLINE_FUNCTION static scalar_type box_min(const BoxT& b, int i) {
    if constexpr (requires { b.minCorner(); }) {
      return static_cast<scalar_type>(b.minCorner()[i]);
    } else {
      return static_cast<scalar_type>(b.min_corner()[i]);
    }
  }

  /// \brief Uniform max-corner accessor for both ArborX::Box (.maxCorner()) and STK boxes (.max_corner()).
  template <typename BoxT>
  KOKKOS_INLINE_FUNCTION static scalar_type box_max(const BoxT& b, int i) {
    if constexpr (requires { b.maxCorner(); }) {
      return static_cast<scalar_type>(b.maxCorner()[i]);
    } else {
      return static_cast<scalar_type>(b.max_corner()[i]);
    }
  }

  /// \brief Return true if any box corner has moved beyond `threshold` under the metric.
  ///
  /// For `FreeSpaceMetric` this is identical to the raw per-scalar absolute difference.
  /// For periodic metrics the minimum-image displacement is used, so a corner that wraps
  /// across the cell boundary is not counted as having moved by a full cell length.
  template <typename BoxView>
  bool corners_moved(const BoxView& current_boxes, const Kokkos::View<scalar_type*, memory_space>& snapshot,
                     scalar_type threshold) const {
    int n = static_cast<int>(current_boxes.extent(0));
    if (n == 0) return false;
    auto snap = snapshot;
    auto met = metric_;  // device-capturable by value; all concrete metrics are trivially copyable
    int any_moved = 0;
    Kokkos::parallel_reduce(
        "mundy_aabb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const int base = 6 * i;
          // Form snapshotted and current min/max corner points in the metric's scalar type.
          const Point<scalar_type> old_min{snap(base + 0), snap(base + 1), snap(base + 2)};
          const Point<scalar_type> new_min{static_cast<scalar_type>(box_min(current_boxes(i), 0)),
                                           static_cast<scalar_type>(box_min(current_boxes(i), 1)),
                                           static_cast<scalar_type>(box_min(current_boxes(i), 2))};
          const Point<scalar_type> old_max{snap(base + 3), snap(base + 4), snap(base + 5)};
          const Point<scalar_type> new_max{static_cast<scalar_type>(box_max(current_boxes(i), 0)),
                                           static_cast<scalar_type>(box_max(current_boxes(i), 1)),
                                           static_cast<scalar_type>(box_max(current_boxes(i), 2))};
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

  /// \brief Snapshot box corners into a device-resident view using a Kokkos parallel_for.
  template <typename BoxView>
  void take_snapshot(const BoxView& boxes, Kokkos::View<scalar_type*, memory_space>& snapshot) {
    int n = static_cast<int>(boxes.extent(0));
    Kokkos::resize(snapshot, 6 * n);
    auto snap = snapshot;
    Kokkos::parallel_for(
        "mundy_aabb_snapshot", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) {
          snap(6 * i + 0) = static_cast<scalar_type>(box_min(boxes(i), 0));
          snap(6 * i + 1) = static_cast<scalar_type>(box_min(boxes(i), 1));
          snap(6 * i + 2) = static_cast<scalar_type>(box_min(boxes(i), 2));
          snap(6 * i + 3) = static_cast<scalar_type>(box_max(boxes(i), 0));
          snap(6 * i + 4) = static_cast<scalar_type>(box_max(boxes(i), 1));
          snap(6 * i + 5) = static_cast<scalar_type>(box_max(boxes(i), 2));
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

  using memory_space    = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using metric_type     = Metric;
  using scalar_type     = Scalar;
  using obb_type        = OBB<Scalar>;
  using obb_view_type   = Kokkos::View<const obb_type*, MemorySpace>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct with a symmetric OBB view and a single threshold (aperiodic).
  ///
  /// Same view is used for both target and source sides.  Available only for the default
  /// aperiodic `FreeSpaceMetric`; supply a metric explicitly for periodic simulations.
  /// \param obbs            [in] OBBs indexed by dense entity ordinals, shared by both sides.
  /// \param max_displacement [in] Rebuild when any OBB corner exits its inflated snapshot.
  explicit RebuildOnOBBDisplacement(obb_view_type obbs, Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_obbs_(obbs),
        source_obbs_(obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement) {}

  /// \brief Construct with separate target/source views and a single threshold (aperiodic).
  ///
  /// \param target_obbs     [in] OBBs for target entities, indexed by dense target ordinals.
  /// \param source_obbs     [in] OBBs for source entities, indexed by dense source ordinals.
  /// \param max_displacement [in] Threshold applied to both targets and sources.
  RebuildOnOBBDisplacement(obb_view_type target_obbs, obb_view_type source_obbs, Scalar max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_obbs_(target_obbs),
        source_obbs_(source_obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement) {}

  /// \brief Construct with separate target/source views and asymmetric thresholds (aperiodic).
  ///
  /// \param target_obbs              [in] OBBs for target entities.
  /// \param source_obbs              [in] OBBs for source entities.
  /// \param target_max_displacement  [in] Threshold for target OBBs.
  /// \param source_max_displacement  [in] Threshold for source OBBs.
  RebuildOnOBBDisplacement(obb_view_type target_obbs, obb_view_type source_obbs,
                            Scalar target_max_displacement, Scalar source_max_displacement)
    requires is_free_space_metric_v<Metric>
      : target_obbs_(target_obbs),
        source_obbs_(source_obbs),
        target_max_displacement_(target_max_displacement),
        source_max_displacement_(source_max_displacement) {}

  /// \brief Construct with a symmetric OBB view, a single threshold, and an explicit metric.
  ///
  /// Use this for periodic simulations where the center-displacement term must respect
  /// the periodic cell geometry.
  /// \param obbs             [in] OBBs shared by both target and source sides.
  /// \param max_displacement [in] Threshold applied to both sides.
  /// \param metric           [in] Metric used to compute minimum-image center displacement.
  RebuildOnOBBDisplacement(obb_view_type obbs, Scalar max_displacement, const Metric& metric)
      : target_obbs_(obbs),
        source_obbs_(obbs),
        target_max_displacement_(max_displacement),
        source_max_displacement_(max_displacement),
        metric_(metric) {}

  /// \brief Construct with separate target/source views, asymmetric thresholds, and a metric.
  ///
  /// \param target_obbs              [in] OBBs for target entities.
  /// \param source_obbs              [in] OBBs for source entities.
  /// \param target_max_displacement  [in] Threshold for target OBBs.
  /// \param source_max_displacement  [in] Threshold for source OBBs.
  /// \param metric                   [in] Metric for minimum-image center displacement.
  RebuildOnOBBDisplacement(obb_view_type target_obbs, obb_view_type source_obbs,
                            Scalar target_max_displacement, Scalar source_max_displacement,
                            const Metric& metric)
      : target_obbs_(target_obbs),
        source_obbs_(source_obbs),
        target_max_displacement_(target_max_displacement),
        source_max_displacement_(source_max_displacement),
        metric_(metric) {}
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief Return true if any OBB has escaped its inflated snapshot containment region.
  ///
  /// On the first call (no snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                     const SourceInput& /*sources*/) {
    if (!has_snapshot_) return true;
    if (target_obbs_.extent(0) != snapshot_target_.extent(0) ||
        source_obbs_.extent(0) != snapshot_source_.extent(0))
      return true;
    return obbs_escaped(target_obbs_, snapshot_target_, target_max_displacement_) ||
           obbs_escaped(source_obbs_, snapshot_source_, source_max_displacement_);
  }

  /// \brief Snapshot the current OBBs into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& /*targets*/,
                const SourceInput& /*sources*/) {
    take_snapshot(target_obbs_, snapshot_target_);
    take_snapshot(source_obbs_, snapshot_source_);
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

  /// \brief Return true if any OBB in `current` has escaped its inflated snapshot entry.
  ///
  /// For each entity i the containment check is (in obb_old's local frame):
  ///   for axis k:  |T[k]| + sum_j |R_rel(k,j)| * h_new[j]  <=  h_old[k] + threshold
  /// where T = R_old^T * sep(c_old, c_new) and R_rel = R_old^T * R_new.
  bool obbs_escaped(const obb_view_type& current,
                    const Kokkos::View<obb_type*, MemorySpace>& snap,
                    Scalar threshold) const {
    const int n = static_cast<int>(current.extent(0));
    if (n == 0) return false;
    auto met = metric_;
    int any_escaped = 0;
    Kokkos::parallel_reduce(
        "mundy_obb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const obb_type& obb_old = snap(i);
          const obb_type& obb_new = current(i);

          // Center displacement in world frame, minimum-image for periodic metrics.
          const auto T_world = met.sep(obb_old.center(), obb_new.center());

          // Express center displacement in obb_old's local frame.
          const Vector3<Scalar> T = conjugate(obb_old.orientation()) *
                                    Vector3<Scalar>{static_cast<Scalar>(T_world[0]),
                                                    static_cast<Scalar>(T_world[1]),
                                                    static_cast<Scalar>(T_world[2])};

          // Relative rotation: R_old^T * R_new.
          const auto R_rel = quaternion_to_rotation_matrix(
              conjugate(obb_old.orientation()) * obb_new.orientation());

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

  /// \brief Snapshot OBBs into a device-resident view.
  void take_snapshot(const obb_view_type& obbs, Kokkos::View<obb_type*, MemorySpace>& snap) {
    const int n = static_cast<int>(obbs.extent(0));
    Kokkos::resize(snap, n);
    auto s = snap;
    Kokkos::parallel_for(
        "mundy_obb_snapshot", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { s(i) = obbs(i); });
    Kokkos::fence();
  }
  //@}

  //! \name Internal members
  //@{

  //! Whether snapshot() has been called at least once.
  bool has_snapshot_ = false;
  //! Caller-maintained OBB view for target entities.
  obb_view_type target_obbs_;
  //! Caller-maintained OBB view for source entities.
  obb_view_type source_obbs_;
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
