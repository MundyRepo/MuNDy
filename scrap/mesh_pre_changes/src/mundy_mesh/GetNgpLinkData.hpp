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

#ifndef MUNDY_MESH_NEW_GETNGPLINKDATA_HPP_
#define MUNDY_MESH_NEW_GETNGPLINKDATA_HPP_

/// \file GetNgpLinkData.hpp
/// \brief Declaration of the GetNgpLinkData class

// Mundy libs
#include <mundy_core/throw_assert.hpp>        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/NewLinkData.hpp>         // for mundy::mesh::NewLinkData
#include <mundy_mesh/NewNgpLinkData.hpp>      // for mundy::mesh::NewNgpLinkDataT

namespace mundy {

namespace mesh {

/// \brief Get an updated ngp link data object.
template<typename NgpMemSpace = stk::ngp::MemSpace>
inline NewNgpLinkDataT<NgpMemSpace> &get_updated_ngp_link_data(const NewLinkData &link_data) {
  static_assert(Kokkos::SpaceAccessibility<NgpMemSpace,stk::ngp::MemSpace>::accessible,
                "In a GPU-enabled build, get_updated_ngp_mesh requires a device-accessible memory-space.");
  MUNDY_THROW_REQUIRE(link_data.is_valid(), std::invalid_argument, "Given link data is not valid.");
  MUNDY_THROW_REQUIRE(!link_data.bulk_data().in_modifiable_state(), std::invalid_argument, 
    "The link data cannot be updated during a mesh modification.");

  NewNgpLinkDataBase* ngp_link_data_base = impl::get_ngp_link_data(link_data);

  if (ngp_link_data_base == nullptr) {
    ngp_link_data_base = new NewNgpLinkDataT<NgpMemSpace>(link_data);
    impl::set_ngp_link_data(link_data, ngp_link_data_base);
  } else {
    ngp_link_data_base->update_link_data();
  }
  return dynamic_cast<NewNgpLinkDataT<NgpMemSpace>&>(*ngp_link_data_base);
}

inline NewNgpLinkData& get_updated_ngp_link_data(const NewLinkData &link_data) {
  MUNDY_THROW_REQUIRE(!link_data.bulk_data().in_modifiable_state(), std::invalid_argument, "The link data cannot be updated during a mesh modification.");
  return get_updated_ngp_mesh<NgpMeshDefaultMemSpace>(link_data);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_GETNGPLINKDATA_HPP_
