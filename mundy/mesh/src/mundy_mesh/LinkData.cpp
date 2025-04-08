// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                       Copyright 2025 Michigan State University
//                                                 Author: Bryce Palmer
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

/// \file LinkData.cpp

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>   // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>         // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>     // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>        // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>          // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/LinkData.hpp>               // for mundy::mesh::LinkData
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>           // for mundy::mesh::field_copy
#include <mundy_mesh/impl/LinkedBucketConn.hpp>  // for mundy::mesh::impl::LinkedBucketConn

namespace mundy {

namespace mesh {

LinkPartition::LinkPartition(LinkData &link_data, const PartitionKey &key, unsigned dimensionality)
    : link_data_(link_data),
      bulk_data_(link_data.bulk_data()),
      link_meta_data_(link_data.link_meta_data()),
      key_(key),
      link_rank_(link_meta_data_.link_rank()),
      dimensionality_(dimensionality),
      bucket_to_linked_conn_ptr_(std::make_shared<BucketToLinkedConn>()),
      link_requests_size_view_("link_requests_size_view"),
      link_requests_capacity_view_("link_requests_capacity_view"),
      requested_links_("requested_links", 0, dimensionality) {
  link_requests_size_view_() = 0;
  link_requests_capacity_view_() = 0;
}
//@}

//! \name Internal functions
//@{

void LinkPartition::process_link_requests_fully_consistent_multi_process() {
  // 1. Count the number of links we need to declare (i.e., the number of rows for which we own the first entity)
  // 2. Declare all links that we need to manage and connect them to their linked entities.
  // 3. Ghost the entities and the newly created links.
  //    3.1. If the source is owned but not the target, send the source and our linker to the target. If the target
  //    is owned but not the source, receive the source and linker from the target.
  // 4. Communicate and update the linker fields to the ghosts. (update needs to reset the local info like bucket ord)
  // 5. Update the CRS for the ghosts (notedly, a different partition than us)

  MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Fully consistent multi-process is not yet supported.");
  // TODO(palmerb4): Implement this.
}

void LinkPartition::process_link_requests_fully_consistent_single_process() {
  // 1. Declare as many links as requests
  // 2. Loop over each request in parallel and connect the linkers to their linked entities
  // 3. Clear the requests

  stk::mesh::PartVector add_parts(key_.size());
  for (size_t i = 0; i < key_.size(); ++i) {
    add_parts[i] = &link_meta_data_.mesh_meta_data().get_part(key_[i]);
  }

  size_t num_requested_links = request_link_size();
  stk::mesh::EntityIdVector new_link_ids;
  stk::mesh::EntityVector new_link_entities;
  new_link_ids.reserve(num_requested_links);
  new_link_entities.reserve(num_requested_links);

  bulk_data_.generate_new_ids(link_rank_, num_requested_links, new_link_ids);
  bulk_data_.declare_entities(link_rank_, new_link_ids, add_parts, new_link_entities);

  // TODO(palmerb4): Consider syncing the the entities to the device, performing the link there, and modifying on
  // device.
  using host_range_policy = stk::ngp::HostRangePolicy;
  unsigned dimensionality = dimensionality_;
  auto requested_links = requested_links_;
  LinkData &link_data = link_data_;
  Kokkos::parallel_for(host_range_policy(0, num_requested_links),
                       [&link_data, &new_link_entities, &requested_links, &dimensionality](size_t i) {
                         stk::mesh::Entity linker = new_link_entities[i];
                         for (size_t j = 0; j < dimensionality; ++j) {
                           stk::mesh::Entity linked_entity = requested_links(i, j);
                           link_data.declare_relation(linker, linked_entity, j);
                         }
                       });
  // link_data_.modify_on_host();  // TODO(palmerb4): Only valid for NGP link data.
  std::cout << "Need to somehow mark link_data as modified. At this point, the host data has been modified but the ngp "
               "link data has no idea way to know this."
            << std::endl;

  // Clear the requests
  // Note, there is no need to erase the entities from the requested links view since we can just reset the size and
  // overwrite the old requests.
  link_requests_size_view_() = 0;
}

void LinkPartition::process_link_requests_partially_consistent_multi_process() {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Partial consistency is not yet supported.");
}

void LinkPartition::process_link_requests_partially_consistent_single_process() {
  // Single process partial consistency is the same as full consistency.
  process_link_requests_fully_consistent_single_process();
}
//@}

LinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank) {
  return LinkMetaData(meta_data, our_name, link_rank);
}

LinkData declare_link_data(BulkData &bulk_data, LinkMetaData link_meta_data) {
  return LinkData(bulk_data, link_meta_data);
}

void LinkData::modify_on_host() {
  auto ngp_link_data = get_updated_ngp_data(*this);
  ngp_link_data.modify_on_host();
}

void LinkData::modify_on_device() {
  auto ngp_link_data = get_updated_ngp_data(*this);
  ngp_link_data.modify_on_device();
}

void LinkData::sync_to_host() {
  auto ngp_link_data = get_updated_ngp_data(*this);
  ngp_link_data.sync_to_host();
}

void LinkData::sync_to_device() {
  auto ngp_link_data = get_updated_ngp_data(*this);
  ngp_link_data.sync_to_device();
}


}  // namespace mesh

}  // namespace mundy
