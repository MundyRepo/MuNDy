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

#ifndef MUNDY_MESH_GETNGPLINKDATA_HPP_
#define MUNDY_MESH_GETNGPLINKDATA_HPP_

/// \file GetNgpLinkData.hpp
/// \brief Declaration of the GetNgpLinkData class

// C++ core
#include <memory>  // for std::shared_ptr, std::unique_ptr

// STK
#include <stk_util/ngp/NgpSpaces.hpp>  // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy
#include <mundy_core/throw_assert.hpp>                    // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkData.hpp>                        // for mundy::mesh::LinkData
#include <mundy_mesh/NgpLinkData.hpp>                     // for mundy::mesh::NgpLinkDataT
#include <mundy_mesh/impl/HostDeviceCOOSynchronizer.hpp>  // for mundy::mesh::impl::HostDeviceCOOSynchronizer
#include <mundy_mesh/impl/HostDeviceCRSSynchronizer.hpp>  // for mundy::mesh::impl::HostDeviceCRSSynchronizer
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>     // for mundy::mesh::impl::HostDeviceSynchronizer

namespace mundy {

namespace mesh {

/// \brief Get an updated ngp link data object.
///
/// \note I believe this is the best place to explain what I have perceived as STK's intent for NgpMemSpace.
/// Currently, you may have one and only one NGP memory space per bulk data object. Effectively, this works
/// like a deconstructed Kokkos::DualView<NgpMemSpace> design where the host never knows about the device memory space
/// and only learns about the device view when get_updated_ngp_* is called. The device memory space is then "locked in"
/// for the lifetime of the bulk data. Consequently, you must use the same NgpMemSpace for ALL get_updated_ngp_* calls
/// for a given bulk data. We follow the same design here with the NgpLinkDataT.
template <typename NgpMemSpace = stk::ngp::MemSpace>
NgpLinkDataT<NgpMemSpace>& get_updated_ngp_link_data(const LinkData& link_data) {
  static_assert(Kokkos::SpaceAccessibility<NgpMemSpace, stk::ngp::MemSpace>::accessible,
                "In a GPU-enabled build, get_updated_ngp_mesh requires a device-accessible memory-space.");
  MUNDY_THROW_REQUIRE(link_data.is_valid(), std::invalid_argument, "Given link data is not valid.");
  MUNDY_THROW_REQUIRE(!link_data.bulk_data().in_modifiable_state(), std::invalid_argument,
                      "The link data cannot be updated during a mesh modification.");

  std::any& any_ngp_link_data = impl::get_ngp_link_data(link_data);

  if (!any_ngp_link_data.has_value()) {
    any_ngp_link_data = NgpLinkDataT<NgpMemSpace>(link_data);  // Let LinkData manage the lifetime of the NGP data.
    NgpLinkDataT<NgpMemSpace>& ngp_link_data = std::any_cast<NgpLinkDataT<NgpMemSpace>&>(any_ngp_link_data);

    // Setup the synchronizers
    LinkCOOData& coo_data = const_cast<LinkCOOData&>(link_data.coo_data());
    LinkCRSData& crs_data = const_cast<LinkCRSData&>(link_data.crs_data());
    NgpLinkCOOData& ngp_coo_data = ngp_link_data.coo_data();
    NgpLinkCRSData& ngp_crs_data = ngp_link_data.crs_data();

    impl::set_crs_synchronizer(
        link_data, std::move(std::make_shared<impl::LinkCRSDataSynchronizerT<NgpMemSpace>>(crs_data, ngp_crs_data)));
    impl::set_coo_synchronizer(
        link_data, std::move(std::make_shared<impl::LinkCOODataSynchronizerT<NgpMemSpace>>(coo_data, ngp_coo_data)));
  } else {
    std::any_cast<NgpLinkDataT<NgpMemSpace>&>(any_ngp_link_data).update_post_mesh_mod();
  }

  return std::any_cast<NgpLinkDataT<NgpMemSpace>&>(any_ngp_link_data);
}

NgpLinkData& get_updated_ngp_link_data(const LinkData& link_data) {
  return get_updated_ngp_link_data<stk::ngp::MemSpace>(link_data);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_GETNGPLINKDATA_HPP_
