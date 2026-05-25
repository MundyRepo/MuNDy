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

#ifndef MUNDY_MESH_IMPL_COMPONENTSIMPL_HPP_
#define MUNDY_MESH_IMPL_COMPONENTSIMPL_HPP_

/// \file ComponentImpl.hpp
/// \brief Stable re-export header for component implementation utilities.
///
/// Restriction-layer headers (DeclarePart, DeclareClass) include this file to access
/// `component_backing_field` without depending on the full declaration machinery in
/// DeclareComponent.hpp. The function is defined in FieldComponent.hpp; this header
/// provides a stable include path that does not expose the FieldComponent class hierarchy
/// directly to callers who only need the backing-field accessor.

// Mundy
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::impl::component_backing_field

#endif  // MUNDY_MESH_IMPL_COMPONENTSIMPL_HPP_
