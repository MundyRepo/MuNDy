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

#ifndef MUNDY_MESH_IMPL_HOSTDEVICESYNCHRONIZER_HPP_
#define MUNDY_MESH_IMPL_HOSTDEVICESYNCHRONIZER_HPP_

/// \file LinkData.hpp
/// \brief Declaration of the LinkData class

namespace mundy {

namespace mesh {

namespace impl {

class HostDeviceSynchronizer {
 public:
  virtual ~HostDeviceSynchronizer() = default;
  virtual void sync_to_device() = 0;
  virtual void sync_to_host() = 0;
  virtual void modify_on_host() = 0;
  virtual void modify_on_device() = 0;
  virtual void update_post_mesh_mod() = 0;
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_HOSTDEVICESYNCHRONIZER_HPP_
