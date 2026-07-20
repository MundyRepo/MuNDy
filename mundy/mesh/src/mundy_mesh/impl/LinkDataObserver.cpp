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

/// \file LinkDataObserver.cpp
/// \brief Defines the LinkDataObserver class.

#include <mundy_mesh/impl/LinkDataObserver.hpp>

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::OrdinalVector, EntityProcVec, EntityRank

// Mundy libs
#include <mundy_mesh/LinkData.hpp>       // for mundy::mesh::LinkData (+ impl::notify_crs_may_be_invalid)
#include <mundy_mesh/LinkMetaData.hpp>   // for mundy::mesh::LinkMetaData
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

namespace impl {

LinkDataObserver::LinkDataObserver(LinkData& link_data)
    : stk::mesh::ModificationObserver(stk::mesh::ModificationObserverPriority::APPLICATION),
      link_data_ptr_(&link_data) {
}

LinkDataObserver::~LinkDataObserver() = default;

void LinkDataObserver::entity_added(stk::mesh::Entity entity) {
  if (is_link_rank(entity)) {
    notify_crs_may_be_invalid(link_data());
  }
}

void LinkDataObserver::entity_deleted(stk::mesh::Entity /*entity*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::entity_parts_added(stk::mesh::Entity /*entity*/, const stk::mesh::OrdinalVector& /*parts*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::entity_parts_removed(stk::mesh::Entity /*entity*/, const stk::mesh::OrdinalVector& /*parts*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::elements_about_to_move_procs_notification(
    const stk::mesh::EntityProcVec& /*elem_proc_pairs_to_move*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::elements_moved_procs_notification(
    const stk::mesh::EntityProcVec& /*elem_proc_pairs_to_move*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::local_entities_created_or_deleted_notification(stk::mesh::EntityRank /*rank*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::local_entity_comm_info_changed_notification(stk::mesh::EntityRank /*rank*/) {
  notify_crs_may_be_invalid(link_data());
}

void LinkDataObserver::local_buckets_changed_notification(stk::mesh::EntityRank /*rank*/) {
  notify_crs_may_be_invalid(link_data());
}

LinkData& LinkDataObserver::link_data() const {
  MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::logic_error, "LinkData pointer is null.");
  return *link_data_ptr_;
}

bool LinkDataObserver::is_link_rank(stk::mesh::Entity entity) const {
  return link_data().bulk_data().entity_rank(entity) == link_data().link_meta_data().link_rank();
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy
