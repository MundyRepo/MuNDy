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

#ifndef MUNDY_MESH_IMPL_HOSTDEVICECOOSYNCHRONIZER_HPP_
#define MUNDY_MESH_IMPL_HOSTDEVICECOOSYNCHRONIZER_HPP_

/// \file LinkCOODataSynchronizerT.hpp
/// \brief Declaration of the LinkCOODataSynchronizerT class

// C++ core libs
#include <any>  // for std::any

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>  // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>   // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>    // stk::mesh::Part
#include <stk_mesh/base/Types.hpp>   // for stk::mesh::EntityRank
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field

// Mundy libs
#include <mundy_core/throw_assert.hpp>                 // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkCOOData.hpp>                  // for mundy::mesh::LinkCOOData/NgpLinkCOOData
#include <mundy_mesh/LinkCRSData.hpp>                  // for mundy::mesh::LinkCRSData/NgpLinkCRSData
#include <mundy_mesh/LinkMetaData.hpp>                 // for mundy::mesh::LinkMetaData
#include <mundy_mesh/MetaData.hpp>                     // for mundy::mesh::MetaData
#include <mundy_mesh/Types.hpp>                        // for mundy::mesh::NgpDataAccessTag
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>  // for mundy::mesh::impl::HostDeviceSynchronizer
#include <mundy_mesh/impl/NgpLinkMetaData.hpp>         // for mundy::mesh::impl::NgpLinkMetaDataT

namespace mundy {

namespace mesh {

namespace impl {

template <typename NgpMemSpace>
class LinkCOODataSynchronizerT : public HostDeviceSynchronizer {
 public:
  LinkCOODataSynchronizerT(LinkCOOData &coo_data, NgpLinkCOODataT<NgpMemSpace> &ngp_coo_data)
      : coo_data_(coo_data), ngp_coo_data_(ngp_coo_data) {
  }

  virtual ~LinkCOODataSynchronizerT() = default;

  virtual void modify_on_host() override {
    LinkMetaData &lmd = coo_data_.link_meta_data();
    impl::get_linked_entities_field(lmd).modify_on_host();
    impl::get_linked_entities_crs_field(lmd).modify_on_host();
    impl::get_linked_entity_ids_field(lmd).modify_on_host();
    impl::get_linked_entity_ranks_field(lmd).modify_on_host();
    impl::get_linked_entity_bucket_ids_field(lmd).modify_on_host();
    impl::get_linked_entity_bucket_ords_field(lmd).modify_on_host();
    impl::get_link_crs_needs_updated_field(lmd).modify_on_host();
    impl::get_link_marked_for_destruction_field(lmd).modify_on_host();
  }

  virtual void modify_on_device() override {
    LinkMetaData &lmd = coo_data_.link_meta_data();
    impl::get_linked_entities_field(lmd).modify_on_device();
    impl::get_linked_entities_crs_field(lmd).modify_on_device();
    impl::get_linked_entity_ids_field(lmd).modify_on_device();
    impl::get_linked_entity_ranks_field(lmd).modify_on_device();
    impl::get_linked_entity_bucket_ids_field(lmd).modify_on_device();
    impl::get_linked_entity_bucket_ords_field(lmd).modify_on_device();
    impl::get_link_crs_needs_updated_field(lmd).modify_on_device();
    impl::get_link_marked_for_destruction_field(lmd).modify_on_device();
  }

  virtual void sync_to_host() override {
    LinkMetaData &lmd = coo_data_.link_meta_data();
    impl::get_linked_entities_field(lmd).sync_to_host();
    impl::get_linked_entities_crs_field(lmd).sync_to_host();
    impl::get_linked_entity_ids_field(lmd).sync_to_host();
    impl::get_linked_entity_ranks_field(lmd).sync_to_host();
    impl::get_linked_entity_bucket_ids_field(lmd).sync_to_host();
    impl::get_linked_entity_bucket_ords_field(lmd).sync_to_host();
    impl::get_link_crs_needs_updated_field(lmd).sync_to_host();
    impl::get_link_marked_for_destruction_field(lmd).sync_to_host();
  }

  virtual void sync_to_device() override {
    LinkMetaData &lmd = coo_data_.link_meta_data();
    impl::get_linked_entities_field(lmd).sync_to_device();
    impl::get_linked_entities_crs_field(lmd).sync_to_device();
    impl::get_linked_entity_ids_field(lmd).sync_to_device();
    impl::get_linked_entity_ranks_field(lmd).sync_to_device();
    impl::get_linked_entity_bucket_ids_field(lmd).sync_to_device();
    impl::get_linked_entity_bucket_ords_field(lmd).sync_to_device();
    impl::get_link_crs_needs_updated_field(lmd).sync_to_device();
    impl::get_link_marked_for_destruction_field(lmd).sync_to_device();
  }

  virtual void update_post_mesh_mod() override {
    stk::mesh::NgpMesh &ngp_mesh = impl::get_ngp_mesh(ngp_coo_data_);
    NgpLinkMetaDataT<NgpMemSpace> &ngp_link_meta_data = impl::get_ngp_link_meta_data(ngp_coo_data_);
    LinkMetaData &link_meta_data = coo_data_.link_meta_data();
    ngp_mesh.update_mesh();

    ngp_link_meta_data.ngp_linked_entities_field() =
        our_get_updated_ngp_field(impl::get_linked_entities_field(link_meta_data));
    ngp_link_meta_data.ngp_linked_entities_crs_field() =
        our_get_updated_ngp_field(impl::get_linked_entities_crs_field(link_meta_data));
    ngp_link_meta_data.ngp_linked_entity_ids_field() =
        our_get_updated_ngp_field(impl::get_linked_entity_ids_field(link_meta_data));
    ngp_link_meta_data.ngp_linked_entity_ranks_field() =
        our_get_updated_ngp_field(impl::get_linked_entity_ranks_field(link_meta_data));
    ngp_link_meta_data.ngp_linked_entity_bucket_ids_field() =
        our_get_updated_ngp_field(impl::get_linked_entity_bucket_ids_field(link_meta_data));
    ngp_link_meta_data.ngp_linked_entity_bucket_ords_field() =
        our_get_updated_ngp_field(impl::get_linked_entity_bucket_ords_field(link_meta_data));
    ngp_link_meta_data.ngp_link_crs_needs_updated_field() =
        our_get_updated_ngp_field(impl::get_link_crs_needs_updated_field(link_meta_data));
    ngp_link_meta_data.ngp_link_marked_for_destruction_field() =
        our_get_updated_ngp_field(impl::get_link_marked_for_destruction_field(link_meta_data));
  }

 private:

  template<typename T>
  auto &our_get_updated_ngp_field(const stk::mesh::Field<T> &field) {
    return stk::mesh::get_updated_ngp_field<T>(field);
  }


  LinkCOOData &coo_data_;
  NgpLinkCOODataT<NgpMemSpace> &ngp_coo_data_;
};  // LinkCRSDataSynchronizerT

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_HOSTDEVICECOOSYNCHRONIZER_HPP_
