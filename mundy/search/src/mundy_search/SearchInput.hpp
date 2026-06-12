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

#ifndef MUNDY_SEARCH_SEARCHINPUT_HPP_
#define MUNDY_SEARCH_SEARCHINPUT_HPP_

/// \file SearchInput.hpp
/// \brief Component-backed neighbor-list inputs: `SearchInput` and `PeriodicSearchInput`.
///
/// A search input binds a `stk::mesh::Selector` to a geometry-yielding component (e.g. an
/// `OBBFieldComponent` / `AABBFieldComponent`). The pair `(selector, component)` encodes, in one
/// object: which entities are searched (selector + rank), their dense ordering, and the rule for
/// reading each entity's geometry (the component). This replaces the older "selector + entity view +
/// box view" triple, which forced the caller to keep `box[i] == compute_box(entity[i])` consistent by
/// hand and could not supply geometry for entities ghosted mid-build.

// C++ core
#include <concepts>  // for std::convertible_to

// Trilinos
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank
#include <stk_topology/topology.hpp>   // for stk::topology

namespace mundy {

namespace search {

namespace impl {

/// \brief True when `Component` exposes a backing field whose entity rank can be queried.
template <typename Component>
concept HasFieldEntityRank = requires(const Component& component) {
  { component.field().entity_rank() } -> std::convertible_to<stk::mesh::EntityRank>;
};

}  // namespace impl

/// \class SearchInput
/// \brief A selector paired with a geometry-yielding component for a non-periodic neighbor-list build.
/// \tparam Component A component whose `operator()` yields a geom primitive (AABB, OBB, Sphere, ...).
template <typename Component>
class SearchInput {
 public:
  //! \name Aliases
  //@{
  using component_type = Component;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct from a selector and a field-backed component; the entity rank is taken from the field.
  ///
  /// The component is taken by non-const reference because the build synchronizes it to device before reading
  /// (it is copied into this input by value; the copy shares the underlying STK field, so the sync is effective).
  SearchInput(const stk::mesh::Selector& selector, Component& component)
    requires impl::HasFieldEntityRank<Component>
      : selector_(selector), component_(component), rank_(component.field().entity_rank()) {
  }

  /// \brief Construct from a selector, component, and an explicit entity rank.
  SearchInput(const stk::mesh::Selector& selector, Component& component, stk::mesh::EntityRank rank)
      : selector_(selector), component_(component), rank_(rank) {
  }

  SearchInput() = default;
  SearchInput(const SearchInput&) = default;
  SearchInput(SearchInput&&) = default;
  SearchInput& operator=(const SearchInput&) = default;
  SearchInput& operator=(SearchInput&&) = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector defining the searched entity chunk.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the geometry-yielding component (const).
  const Component& component() const noexcept {
    return component_;
  }

  /// \brief Get the geometry-yielding component (mutable, e.g. for sync / NGP refresh).
  Component& component() noexcept {
    return component_;
  }

  /// \brief Get the entity rank of the searched chunk.
  stk::mesh::EntityRank rank() const noexcept {
    return rank_;
  }
  //@}

 private:
  stk::mesh::Selector selector_;
  Component component_{};
  stk::mesh::EntityRank rank_{stk::topology::INVALID_RANK};
};

/// \class PeriodicSearchInput
/// \brief A `SearchInput` augmented with a periodicity metric for periodic neighbor-list builds.
/// \tparam Component A component whose `operator()` yields a geom primitive.
/// \tparam Metric    A periodicity metric (e.g. `OrthorhombicMetric`) describing the periodic domain.
template <typename Component, typename Metric>
class PeriodicSearchInput {
 public:
  //! \name Aliases
  //@{
  using component_type = Component;
  using metric_type = Metric;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct from a selector, field-backed component, and metric; rank is taken from the field.
  ///
  /// The component is taken by non-const reference because the build synchronizes it to device before reading.
  PeriodicSearchInput(const stk::mesh::Selector& selector, Component& component, const Metric& metric)
    requires impl::HasFieldEntityRank<Component>
      : selector_(selector), component_(component), metric_(metric), rank_(component.field().entity_rank()) {
  }

  /// \brief Construct from a selector, component, metric, and an explicit entity rank.
  PeriodicSearchInput(const stk::mesh::Selector& selector, Component& component, const Metric& metric,
                      stk::mesh::EntityRank rank)
      : selector_(selector), component_(component), metric_(metric), rank_(rank) {
  }

  PeriodicSearchInput() = default;
  PeriodicSearchInput(const PeriodicSearchInput&) = default;
  PeriodicSearchInput(PeriodicSearchInput&&) = default;
  PeriodicSearchInput& operator=(const PeriodicSearchInput&) = default;
  PeriodicSearchInput& operator=(PeriodicSearchInput&&) = default;
  //@}

  //! \name Getters
  //@{

  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  const Component& component() const noexcept {
    return component_;
  }

  Component& component() noexcept {
    return component_;
  }

  stk::mesh::EntityRank rank() const noexcept {
    return rank_;
  }

  /// \brief Get the periodicity metric describing the periodic domain.
  const Metric& periodic_metric() const noexcept {
    return metric_;
  }
  //@}

 private:
  stk::mesh::Selector selector_;
  Component component_{};
  Metric metric_{};
  stk::mesh::EntityRank rank_{stk::topology::INVALID_RANK};
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Class template argument deduction guides
//@{
template <typename Component>
SearchInput(const stk::mesh::Selector&, Component&) -> SearchInput<Component>;

template <typename Component>
SearchInput(const stk::mesh::Selector&, Component&, stk::mesh::EntityRank) -> SearchInput<Component>;

template <typename Component, typename Metric>
PeriodicSearchInput(const stk::mesh::Selector&, Component&, const Metric&) -> PeriodicSearchInput<Component, Metric>;

template <typename Component, typename Metric>
PeriodicSearchInput(const stk::mesh::Selector&, Component&, const Metric&, stk::mesh::EntityRank)
    -> PeriodicSearchInput<Component, Metric>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_SEARCHINPUT_HPP_
