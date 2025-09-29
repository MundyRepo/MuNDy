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

#ifndef MUNDY_MESH_TYPES_HPP_
#define MUNDY_MESH_TYPES_HPP_

namespace stk {

namespace ngp {

using HostMemSpace = stk::ngp::HostExecSpace::memory_space;
using MemSpace = stk::ngp::ExecSpace::memory_space;

}  // namespace ngp

}  // namespace stk

namespace mundy {

namespace mesh {

enum NgpDataAccessTag : uint8_t {
  ReadWrite = 0,     // Sync values to memory space and mark as modified; Allow modification
  ReadOnly = 1,      // Sync values to memory space and do not mark as modified; Disallow modification
  OverwriteAll = 2,  // Do not sync values to memory space and mark as modified; Allow modification

  Unsynchronized,       // Do not sync values to memory space and do not mark as modified; Allow modification
  ConstUnsynchronized,  // Do not sync values to memory space and do not mark as modified; Disallow modification

  InvalidAccess,
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_TYPES_HPP_