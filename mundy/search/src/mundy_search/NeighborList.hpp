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

#ifndef MUNDY_SEARCH_NEIGHBORLIST_HPP_
#define MUNDY_SEARCH_NEIGHBORLIST_HPP_

/// \file NeighborList.hpp
/// \brief Umbrella header: includes all public neighbor-list types for mundy::search.
///
/// Including this header brings in the full neighbor-list interface:
///   - Search candidate types (SearchCandidate.hpp)
///   - Excluder concept and built-in excluders (Excluder.hpp)
///   - NeighborListType concept and Neighbors/NeighborPair access types (Neighbors.hpp)
///   - NeighborListIterationTraits primary template and 2D specializations (NeighborListIterationTraits.hpp,
///     ArborX2dNeighborList.hpp)
///   - ForEach parallel iteration entry points (ForEach.hpp)
///   - NeighborListInputType concept and NeighborListBuildTraits primary template (NeighborListBuildTraits.hpp)
///   - Type-state fluent builder (NeighborListBuilder.hpp)
///   - ArborX-backed concrete list types and their build-traits specializations (ArborXNeighborList.hpp)
///   - STK-coarse-search-backed concrete list types and their build-traits specializations (STKSearchNeighborList.hpp)
///
/// Downstream code that only uses a subset of the interface may include individual headers directly. This umbrella is
/// provided for convenience when the full interface is needed.

// mundy_search public interface — dependency order
#include <mundy_search/ArborX1dNeighborList.hpp>  // ArborX1dNeighborList, PeriodicArborX1dNeighborList + traits
#include <mundy_search/ArborX2dNeighborList.hpp>  // ArborX2dNeighborList, PeriodicArborX2dNeighborList + traits
#include <mundy_search/Excluder.hpp>  // ExcluderType, NoExcluder, ExcluderChain, ExcludeSelfInteraction, ExcludeSymmetricDuplicates
#include <mundy_search/ForEach.hpp>                  // for_each_neighbor_pair, for_each_target_with_neighbors
#include <mundy_search/NeighborListBuildTraits.hpp>  // NeighborListInputType, NeighborListBuildTraits primary template
#include <mundy_search/NeighborListBuilder.hpp>      // NeighborListBuilder, make_neighbor_list_builder
#include <mundy_search/Neighbors.hpp>                // NeighborListType, Neighbors, NeighborPair
#include <mundy_search/STKSearchNeighborList.hpp>    // STKSearchNeighborList, PeriodicSTKSearchNeighborList + traits
#include <mundy_search/SearchCandidate.hpp>          // NeighborSearchCandidate, PeriodicNeighborSearchCandidate

#endif  // MUNDY_SEARCH_NEIGHBORLIST_HPP_
