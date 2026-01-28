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

#ifndef MUNDY_MESH_IMPL_HOSTDEVICECSRSYNCHRONIZER_HPP_
#define MUNDY_MESH_IMPL_HOSTDEVICECSRSYNCHRONIZER_HPP_

/// \file LinkCSRDataSynchronizerT.hpp
/// \brief Declaration of the LinkCSRDataSynchronizerT class

// C++ core libs
#include <any>  // for std::any

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>  // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>   // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>    // stk::mesh::Part
#include <stk_mesh/base/Types.hpp>   // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_core/throw_assert.hpp>                 // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkCOOData.hpp>                  // for mundy::mesh::LinkCOOData/NgpLinkCOOData
#include <mundy_mesh/LinkCSRData.hpp>                  // for mundy::mesh::LinkCSRData/NgpLinkCSRData
#include <mundy_mesh/LinkMetaData.hpp>                 // for mundy::mesh::LinkMetaData
#include <mundy_mesh/MetaData.hpp>                     // for mundy::mesh::MetaData
#include <mundy_mesh/Types.hpp>                        // for mundy::mesh::NgpDataAccessTag
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>  // for mundy::mesh::impl::HostDeviceSynchronizer

namespace mundy {

namespace mesh {

namespace impl {

template <typename NgpMemSpace>
class LinkCSRDataSynchronizerT : public HostDeviceSynchronizer {
 public:
  LinkCSRDataSynchronizerT(LinkCSRData &crs_data, NgpLinkCSRDataT<NgpMemSpace> &ngp_crs_data)
      : crs_data_(crs_data), ngp_crs_data_(ngp_crs_data) {
  }

  virtual ~LinkCSRDataSynchronizerT() = default;

  virtual void modify_on_host() override {
    // We have been informed of a modification. Nothing to do.
  }

  virtual void modify_on_device() override {
    // We have been informed of a modification. Nothing to do.
  }

  virtual void sync_to_device() override {
    ngp_crs_data_.synchronize_with(crs_data_);
  }

  virtual void sync_to_host() override {
    crs_data_.synchronize_with(ngp_crs_data_);
  }

  virtual void update_post_mesh_mod() override {
    // No-op for now
    //
    // There are a bunch of synchronization steps that we need to figure out how to handle
    // once we make it possible to keep the CSR data up to date during mesh modifications.
    //
    // For now, we will just rebuild the CSR data from the COO data after mesh modifications
    // std::cout << "WARNING: update_post_mesh_mod() is a currently a no-op for LinkCSRDataSynchronizerT" << std::endl;
  }

 private:
  LinkCSRData &crs_data_;
  NgpLinkCSRDataT<NgpMemSpace> &ngp_crs_data_;
};  // LinkCSRDataSynchronizerT

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_HOSTDEVICECSRSYNCHRONIZER_HPP_
