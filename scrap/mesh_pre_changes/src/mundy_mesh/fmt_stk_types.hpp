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

#ifndef MUNDY_MESH_FMT_STK_TYPES_HPP_
#define MUNDY_MESH_FMT_STK_TYPES_HPP_

/// \file fmt_stk_types.hpp
/// \brief fmt ostream support for STK types

// External
#include <fmt/format.h>
#include <fmt/ostream.h>

// STK
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>

#define MUNDY_ADD_FMT_OSTREAM_SUPPORT(type) \
  template <>                               \
  struct fmt::formatter<type> : fmt::ostream_formatter {};

MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::mesh::Entity)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::mesh::EntityId)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::mesh::EntityKey)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::mesh::Selector)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::topology)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::topology::rank_t)
MUNDY_ADD_FMT_OSTREAM_SUPPORT(stk::topology::topology_t)

#endif  // MUNDY_MESH_FMT_STK_TYPES_HPP_
