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
/// \brief A set of helpers for working with components with reduced boilerplate code.

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <iostream>   // for std::ostream
#include <stdexcept>  // for std::runtime_error
#include <type_traits>
#include <utility>
#include <vector>  // for std::vector

// Trilinos
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace mesh {

namespace impl {

// TODO(palmerb4): How to pull the IMPL code out of Component.hpp given that it depends on public Component.hpp types?

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_COMPONENTSIMPL_HPP_
