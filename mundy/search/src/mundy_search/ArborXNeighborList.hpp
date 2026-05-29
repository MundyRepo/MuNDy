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

#ifndef MUNDY_SEARCH_ARBORXNEIGHBORLIST_HPP_
#define MUNDY_SEARCH_ARBORXNEIGHBORLIST_HPP_

/// \file ArborXNeighborList.hpp
/// \brief Umbrella header: includes both ArborX 1D and 2D neighbor-list types.
///
/// Most code should include only one of the two individual headers:
///   - ArborX1dNeighborList.hpp — compressed 1D storage (lower memory, best for sparse neighbor lists)
///   - ArborX2dNeighborList.hpp — dense 2D per-target storage (best for dense neighbor lists)
///
/// Include this umbrella only when both storage layouts are needed in the same translation unit.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

#include <mundy_search/ArborX1dNeighborList.hpp>
#include <mundy_search/ArborX2dNeighborList.hpp>

#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // MUNDY_SEARCH_ARBORXNEIGHBORLIST_HPP_
