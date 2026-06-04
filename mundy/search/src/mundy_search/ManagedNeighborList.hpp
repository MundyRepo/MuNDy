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

#ifndef MUNDY_SEARCH_MANAGEDNEIGHBORLIST_HPP_
#define MUNDY_SEARCH_MANAGEDNEIGHBORLIST_HPP_

/// \file ManagedNeighborList.hpp
/// \brief ManagedNeighborList: a cached neighbor list driven by a stateful RebuilderType policy.

// C++ core
#include <optional>     // for std::optional
#include <stdexcept>    // for std::runtime_error
#include <type_traits>  // for std::move

// Trilinos
#include <stk_mesh/base/BulkData.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>                 // for ExcluderType
#include <mundy_search/NeighborListBuildTraits.hpp>  // for NeighborListInputType, NeighborListBuildTraits
#include <mundy_search/NeighborListRebuilder.hpp>    // for RebuilderType
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace search {

/// \class ManagedNeighborList
/// \brief A neighbor list cache driven by a stateful rebuilder policy.
///
/// `ManagedNeighborList` owns a cached `ListType` instance and a `Rebuilder` that determines
/// when the cache is stale. On each call to `update(bulk, targets, sources, args)`, the rebuilder
/// is consulted: if it signals that a rebuild is needed (or if no list has been built yet), a fresh
/// list is constructed from the stored builder state with the supplied targets and sources; `snapshot`
/// is then called on the rebuilder so it can record current geometry or entity state.
///
/// A `ManagedNeighborList` is obtained by calling `.manage(rebuilder)` on a `NeighborListBuilder`
/// at **any point** in the fluent chain. The remaining builder configuration methods — `exec_space`,
/// `exclude`, and `sort_neighbors` — are also available directly on the returned `ManagedNeighborList`,
/// so the full chain can be written in any order:
///
/// \code{.cpp}
///   // manage() at any position — chain continues after it:
///   auto managed = make_neighbor_list_builder<STKSearchNeighborList<>>()
///       .manage(RebuildOnEntityChange{} | RebuildOnAABBDisplacement<>{skin_distance})
///       .exec_space(exec)
///       .exclude(ExcludeSelfInteraction{});
///
///   // Each time-step — passes fresh box views:
///   const auto& nl = managed.update(bulk, target_boxes, source_boxes);
/// \endcode
///
/// `update()` requires that the execution space has been set (via `exec_space(...)`) at or before
/// the call; failing to do so produces a compile-time constraint violation.
///
/// \tparam BuilderState The `NeighborListBuilder<...>` type captured at the point of `.manage()`.
///   Further fluent calls produce new `ManagedNeighborList` specializations with updated builder state.
/// \tparam Rebuilder Stateful policy that decides when to rebuild.
template <typename BuilderState, RebuilderType Rebuilder>
class ManagedNeighborList {
 public:
  //! \name Aliases
  //@{

  using builder_state_type = BuilderState;
  using list_type = typename BuilderState::neighbor_list_type;
  using rebuilder_type = Rebuilder;
  using build_args_type = typename NeighborListBuildTraits<list_type>::args_type;
  //@}

  //! \name Update result
  //@{

  /// \brief Return type for update() carrying both the list and a rebuild indicator.
  ///
  /// Implicitly converts to `const list_type&` so existing call sites that bind the result
  /// to a reference compile unchanged (so long as the call site isn't templated). Contextual 
  /// bool conversion is `true` when the list waw rebuilt during this call, `false` 
  /// when the cached copy was returned as-is:
  ///
  /// \code{.cpp}
  ///   // Unchanged existing usage:
  ///   const auto& nl = managed.update(bulk, targets, sources);
  ///
  ///   // New: react only when the list changed:
  ///   if (managed.update(bulk, targets, sources)) { rehash_bins(); }
  ///
  ///   // Capture both at once:
  ///   auto result = managed.update(bulk, targets, sources);
  ///   use(result);         // implicit list conversion
  ///   if (result) { ... }  // bool branch
  /// \endcode
  struct UpdateResult {
    const list_type& list;
    bool             rebuilt;

    operator const list_type&() const noexcept { return list; }
    explicit operator bool()    const noexcept { return rebuilt; }
  };
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct from a builder state snapshot and a rebuilder.
  ///
  /// Called by `NeighborListBuilder::manage(rebuilder)` — prefer that factory.
  ManagedNeighborList(const BuilderState& builder, rebuilder_type rebuilder)
      : builder_(builder), rebuilder_(std::move(rebuilder)) {}
  //@}

  //! \name Fluent configuration (mirrors NeighborListBuilder)
  //@{

  /// \brief Return a new managed list with the execution space supplied.
  /// \tparam NewExecutionSpace Execution space type.
  /// \param es [in] Execution space used by the eventual build.
  template <typename NewExecutionSpace>
  auto exec_space(const NewExecutionSpace& es) const {
    auto new_builder = builder_.exec_space(es);
    return ManagedNeighborList<decltype(new_builder), Rebuilder>(new_builder, rebuilder_);
  }

  /// \brief Return a new managed list with an appended excluder.
  /// \tparam NextExcluder Excluder type to append.
  /// \param ex [in] Excluder to append.
  template <ExcluderType NextExcluder>
  auto exclude(const NextExcluder& ex) const {
    auto new_builder = builder_.exclude(ex);
    return ManagedNeighborList<decltype(new_builder), Rebuilder>(new_builder, rebuilder_);
  }

  /// \brief Return a new managed list with the neighbor-sort flag set.
  /// \param sort [in] Whether to sort neighbor rows by source ordinal after each build.
  auto sort_neighbors(bool sort) const {
    return ManagedNeighborList<BuilderState, Rebuilder>(builder_.sort_neighbors(sort), rebuilder_);
  }
  //@}

  //! \name Core API
  //@{

  /// \brief Return a valid, up-to-date neighbor list for the given targets and sources.
  ///
  /// Consults the rebuilder. If no list has been built yet or the rebuilder signals a rebuild is
  /// needed, constructs a fresh list from the stored builder state with the supplied targets and
  /// sources, then calls `snapshot` on the rebuilder. Otherwise returns the cached list.
  ///
  /// \tparam TargetInput Selected target input type (must satisfy `NeighborListInputType`).
  /// \tparam SourceInput Selected source input type (must satisfy `NeighborListInputType`).
  /// \param bulk [in] STK bulk data used for excluder setup and build.
  /// \param targets [in] Target input (boxes, selector, …) for this update.
  /// \param sources [in] Source input (boxes, selector, …) for this update.
  /// \param args [in] Backend-specific build parameters; default-constructed when omitted.
  template <NeighborListInputType TargetInput, NeighborListInputType SourceInput>
  UpdateResult update(const stk::mesh::BulkData& bulk, const TargetInput& targets,
                      const SourceInput& sources, const build_args_type& args = {})
    requires(BuilderState::has_exec_space)
  {
    const bool did_rebuild = !cached_list_.has_value() || rebuilder_.needs_rebuild(bulk, targets, sources);
    if (did_rebuild) {
      cached_list_.emplace(builder_.target_input(targets).source_input(sources).build(bulk, args));
      rebuilder_.snapshot(bulk, targets, sources);
    }
    return {*cached_list_, did_rebuild};
  }

  /// \brief Discard the cached list so that the next `update()` unconditionally rebuilds.
  void invalidate() noexcept { cached_list_.reset(); }

  /// \brief Return whether a valid cached list exists.
  bool has_valid_list() const noexcept { return cached_list_.has_value(); }

  /// \brief Return the cached list without consulting the rebuilder.
  ///
  /// Requires that `has_valid_list()` is true; call `update()` first.
  const list_type& current() const {
    MUNDY_THROW_REQUIRE(has_valid_list(), std::runtime_error,
                        "ManagedNeighborList::current() called before any successful update().");
    return *cached_list_;
  }
  //@}

  //! \name Accessors
  //@{

  const BuilderState& builder() const noexcept { return builder_; }
  rebuilder_type& rebuilder() noexcept { return rebuilder_; }
  const rebuilder_type& rebuilder() const noexcept { return rebuilder_; }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Builder state snapshot captured at the point of `.manage()`, further updated by fluent calls.
  BuilderState builder_;
  //! Stateful policy that decides when a rebuild is needed.
  rebuilder_type rebuilder_;
  //! Cached neighbor list; empty until the first successful `update()`.
  std::optional<list_type> cached_list_;
  //@}
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_MANAGEDNEIGHBORLIST_HPP_
