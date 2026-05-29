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
concept RebuilderType = requires(
    T& rebuilder, const stk::mesh::BulkData& bulk_data,
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

  RebuilderChain(const prior_rebuilder_type& prior, const appended_rebuilder_type& next)
      : prior_(prior), next_(next) {}
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief Return true if either rebuilder in the chain signals a rebuild is needed.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& bulk, const TargetInput& targets,
                     const SourceInput& sources) {
    return prior_.needs_rebuild(bulk, targets, sources) || next_.needs_rebuild(bulk, targets, sources);
  }

  /// \brief Snapshot state in both chain members to keep all snapshots current.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& bulk, const TargetInput& targets,
                const SourceInput& sources) {
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

  prior_rebuilder_type& prior() noexcept { return prior_; }
  const prior_rebuilder_type& prior() const noexcept { return prior_; }
  appended_rebuilder_type& next() noexcept { return next_; }
  const appended_rebuilder_type& next() const noexcept { return next_; }
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
                const SourceInput& /*sources*/) noexcept {}
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
                const SourceInput& /*sources*/) noexcept {}
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
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets,
                     const SourceInput& sources) {
    if (!has_snapshot_) return true;
    return entities_changed(targets.entities(), snapshot_target_) ||
           entities_changed(sources.entities(), snapshot_source_);
  }

  /// \brief Snapshot the current entity sequences into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets,
                const SourceInput& sources) {
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
  bool entities_changed(const EntityView& current,
                        const Kokkos::View<stk::mesh::Entity*, memory_space>& snap) const {
    int n = static_cast<int>(current.extent(0));
    if (n != static_cast<int>(snap.extent(0))) return true;
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
  void take_snapshot(const EntityView& entities,
                     Kokkos::View<stk::mesh::Entity*, memory_space>& snap) {
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
/// After each build, `snapshot()` records all six scalar coordinates (min/max per axis)
/// for every target and source box using a device-resident view and a Kokkos parallel_for.
/// `needs_rebuild()` computes the maximum per-coordinate displacement using a Kokkos
/// parallel_reduce on device and returns true when any coordinate has moved more than
/// `max_displacement`.
///
/// Checking per-coordinate displacement of axis-aligned bounding box corners is stronger
/// than checking center displacement alone: the maximum displacement of any corner equals
/// the maximum displacement of any of the six bounding scalar values, so the check is exact
/// for AABBs and conservative for arbitrary shapes.
///
/// \tparam MemorySpace Kokkos memory space used for device-resident snapshots and kernels.
///   Must match the memory space of the box views supplied to `needs_rebuild` and `snapshot`.
template <typename MemorySpace = stk::ngp::MemSpace>
class RebuildOnAABBDisplacement {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct with a maximum per-corner displacement threshold.
  /// \param max_displacement [in] Rebuild is triggered if any box corner moves farther than this.
  explicit RebuildOnAABBDisplacement(double max_displacement) : max_displacement_(max_displacement) {}
  //@}

  //! \name Rebuild policy
  //@{

  /// \brief Return true if any box corner has moved more than `max_displacement` since the last build.
  ///
  /// On the first call (no snapshot yet), always returns true.
  template <typename TargetInput, typename SourceInput>
  bool needs_rebuild(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets,
                     const SourceInput& sources) {
    if (snapshot_target_.extent(0) == 0) return true;
    if (targets.boxes().extent(0) * 6 != snapshot_target_.extent(0) ||
        sources.boxes().extent(0) * 6 != snapshot_source_.extent(0))
      return true;
    return corners_moved(targets.boxes(), snapshot_target_) ||
           corners_moved(sources.boxes(), snapshot_source_);
  }

  /// \brief Snapshot the current box corners into device-resident storage.
  template <typename TargetInput, typename SourceInput>
  void snapshot(const stk::mesh::BulkData& /*bulk*/, const TargetInput& targets,
                const SourceInput& sources) {
    take_snapshot(targets.boxes(), snapshot_target_);
    take_snapshot(sources.boxes(), snapshot_source_);
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

  /// \brief Return true if any box corner has moved more than `max_displacement_`.
  ///
  /// Uses a Kokkos parallel_reduce on `execution_space` to evaluate all boxes on device
  /// without copying data to host.
  template <typename BoxView>
  bool corners_moved(const BoxView& current_boxes,
                     const Kokkos::View<double*, memory_space>& snapshot) const {
    int n = static_cast<int>(current_boxes.extent(0));
    if (n == 0) return false;
    double threshold = max_displacement_;
    auto snap = snapshot;
    int any_moved = 0;
    Kokkos::parallel_reduce(
        "mundy_aabb_displacement_check", Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, int& lmax) {
          const int base = 6 * i;
          int moved =
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_x_min()) - snap(base + 0)) > threshold ? 1
                                                                                                            : 0) |
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_y_min()) - snap(base + 1)) > threshold ? 1
                                                                                                            : 0) |
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_z_min()) - snap(base + 2)) > threshold ? 1
                                                                                                            : 0) |
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_x_max()) - snap(base + 3)) > threshold ? 1
                                                                                                            : 0) |
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_y_max()) - snap(base + 4)) > threshold ? 1
                                                                                                            : 0) |
              (Kokkos::abs(static_cast<double>(current_boxes(i).get_z_max()) - snap(base + 5)) > threshold ? 1
                                                                                                            : 0);
          lmax = lmax > moved ? lmax : moved;
        },
        Kokkos::Max<int>(any_moved));
    Kokkos::fence();
    return any_moved != 0;
  }

  /// \brief Snapshot box corners into a device-resident view using a Kokkos parallel_for.
  template <typename BoxView>
  void take_snapshot(const BoxView& boxes, Kokkos::View<double*, memory_space>& snapshot) {
    int n = static_cast<int>(boxes.extent(0));
    Kokkos::resize(snapshot, 6 * n);
    auto snap = snapshot;
    Kokkos::parallel_for(
        "mundy_aabb_snapshot", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) {
          snap(6 * i + 0) = static_cast<double>(boxes(i).get_x_min());
          snap(6 * i + 1) = static_cast<double>(boxes(i).get_y_min());
          snap(6 * i + 2) = static_cast<double>(boxes(i).get_z_min());
          snap(6 * i + 3) = static_cast<double>(boxes(i).get_x_max());
          snap(6 * i + 4) = static_cast<double>(boxes(i).get_y_max());
          snap(6 * i + 5) = static_cast<double>(boxes(i).get_z_max());
        });
    Kokkos::fence();
  }
  //@}

  //! \name Internal members
  //@{

  //! Maximum per-corner displacement before a rebuild is triggered.
  double max_displacement_;
  //! Device-resident snapshot of target box corners (6 doubles per box: min_xyz then max_xyz).
  Kokkos::View<double*, memory_space> snapshot_target_{"mundy_rebuilder_snap_tgt", 0};
  //! Device-resident snapshot of source box corners (6 doubles per box: min_xyz then max_xyz).
  Kokkos::View<double*, memory_space> snapshot_source_{"mundy_rebuilder_snap_src", 0};
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTREBUILDER_HPP_
