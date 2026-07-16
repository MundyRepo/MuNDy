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

#ifndef MUNDY_MESH_IMPL_LINKDATAOBSERVER_HPP_
#define MUNDY_MESH_IMPL_LINKDATAOBSERVER_HPP_

/// \file LinkDataObserver.hpp
/// \brief Declares the LinkDataObserver class.

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>              // for stk::mesh::BulkData
#include <stk_mesh/base/ModificationObserver.hpp>  // for stk::mesh::ModificationObserver

// Mundy libs
#include <mundy_mesh/LinkMetaData.hpp>   // for mundy::mesh::LinkMetaData
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

namespace impl {

// TODO(palmerb4): If a link entity was created during the modification process, upon modification end signal
// we must update the runtime link data from the static data (entity id + rank -> entity)

class LinkDataObserver : public stk::mesh::ModificationObserver {
 public:
  LinkDataObserver(stk::mesh::BulkData& bulk_data, LinkMetaData& link_meta_data, bool& crs_structure_dirty)
      : stk::mesh::ModificationObserver(stk::mesh::ModificationObserverPriority::APPLICATION),
        bulk_data_ptr_(&bulk_data),
        link_meta_data_ptr_(&link_meta_data),
        crs_structure_dirty_ptr_(&crs_structure_dirty) {
  }

  virtual ~LinkDataObserver() = default;

  void entity_added(stk::mesh::Entity entity) override {
    if (is_link_rank(entity)) {
      request_structural_rebuild();
    }
  }

  void entity_deleted(stk::mesh::Entity /*entity*/) override {
    request_structural_rebuild();
  }

  void entity_parts_added(stk::mesh::Entity /*entity*/, const stk::mesh::OrdinalVector& /*parts*/) override {
    request_structural_rebuild();
  }

  void entity_parts_removed(stk::mesh::Entity /*entity*/, const stk::mesh::OrdinalVector& /*parts*/) override {
    request_structural_rebuild();
  }

  void elements_about_to_move_procs_notification(const stk::mesh::EntityProcVec& /*elemProcPairsToMove*/) override {
    request_structural_rebuild();
  }

  void elements_moved_procs_notification(const stk::mesh::EntityProcVec& /*elemProcPairsToMove*/) override {
    request_structural_rebuild();
  }

  void local_entities_created_or_deleted_notification(stk::mesh::EntityRank rank) override {
    request_structural_rebuild_for_rank(rank);
  }

  void local_entity_comm_info_changed_notification(stk::mesh::EntityRank rank) override {
    request_structural_rebuild_for_rank(rank);
  }

  void local_buckets_changed_notification(stk::mesh::EntityRank rank) override {
    request_structural_rebuild_for_rank(rank);
  }

 private:
  void request_structural_rebuild() {
    MUNDY_THROW_ASSERT(crs_structure_dirty_ptr_ != nullptr, std::logic_error,
                       "CSR structure-dirty flag pointer is null.");
    *crs_structure_dirty_ptr_ = true;
  }

  void request_structural_rebuild_for_rank(stk::mesh::EntityRank rank) {
    if (rank < stk::topology::NUM_RANKS) {
      request_structural_rebuild();
    }
  }

  bool is_link_rank(stk::mesh::Entity entity) const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::logic_error, "BulkData pointer is null.");
    return bulk_data_ptr_->entity_rank(entity) == link_meta_data().link_rank();
  }

  const LinkMetaData& link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::logic_error, "LinkMetaData pointer is null.");
    return *link_meta_data_ptr_;
  }

  stk::mesh::BulkData* bulk_data_ptr_;
  LinkMetaData* link_meta_data_ptr_;
  bool* crs_structure_dirty_ptr_;
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_LINKDATAOBSERVER_HPP_
