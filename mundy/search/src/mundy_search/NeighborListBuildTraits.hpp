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

#ifndef MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_
#define MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_

/// \file NeighborListBuildTraits.hpp
/// \brief NeighborListInputType concept and NeighborListBuildTraits primary template.
///
/// Include this header when specializing NeighborListBuildTraits for a new concrete list type.

// C++ core
#include <concepts>  // for std::same_as

// Trilinos
#include <stk_mesh/base/Selector.hpp>

namespace mundy {

namespace search {

/// \concept NeighborListInputType
/// \brief Specifies a selected source or target chunk used to build a neighbor list.
///
/// Search boxes, periodic image boxes, and future source/target input types may have different geometry and indexing
/// APIs, but they must expose the selector that defines the semantic entity chunk being searched.
template <typename T>
concept NeighborListInputType = requires(const T& input) {
  { input.selector() } -> std::same_as<const stk::mesh::Selector&>;
};

/// \struct NeighborListBuildTraits
/// \brief Traits that couple a concrete neighbor-list type to its build logic and type-specific parameters.
///
/// Specialize this struct for each concrete neighbor-list type. A valid specialization must provide:
///   - `target_input_type` — expected `TargetInput` type for the builder.
///   - `source_input_type` — expected `SourceInput` type for the builder.
///   - `args_type` — a struct holding any build-specific parameters (empty struct if none are needed).
///   - `static list_type build(const Builder&, const stk::mesh::BulkData&, const args_type&)` — a static
///     function template that builds the list from a complete `NeighborListBuilder`, `BulkData`, and args.
///
/// The primary template leaves `build()` undefined so that accessing it on an unspecialized type is a
/// compile error. The `struct args_type {}` in the primary allows function signatures in `NeighborListBuilder`
/// to compile even before a specialization is visible.
/// \tparam ListType Concrete neighbor-list type being described.
template <typename ListType>
struct NeighborListBuildTraits {
  /// \brief Default empty args; specializations override this with build-specific parameters.
  struct args_type {};
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_
