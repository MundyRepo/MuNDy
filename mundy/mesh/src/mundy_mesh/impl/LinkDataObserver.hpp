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
#include <stk_mesh/base/Entity.hpp>                // for stk::mesh::Entity
#include <stk_mesh/base/ModificationObserver.hpp>  // for stk::mesh::ModificationObserver
#include <stk_mesh/base/Types.hpp>                 // for stk::mesh::OrdinalVector, EntityProcVec, EntityRank

namespace mundy {

namespace mesh {

class LinkData;

namespace impl {

/// \brief LinkDataObserver translates BulkData mesh-modification signals into LinkData notifications.
///
/// The observer is a message interpreter: it filters the stream of BulkData modification signals, decides which of them
/// are meaningful to the link data, and forwards a *semantic* notification to the LinkData, which then decides how to
/// react. It deliberately holds only a reference to its LinkData and never reaches into its internals; it speaks in
/// terms of what happened to the mesh, not in terms of how the link data should react.
///
/// It forwards a single notification:
///   - notify_crs_may_be_invalid: the mesh changed in a way that may invalidate the CSR (entities/parts/buckets moved,
///   entities created or destroyed, etc.). This can fire many times per modification cycle.
///
/// COO runtime caches are reconciled by the LinkData constructor; on restart the LinkData is declared after the read.
/// TODO(palmerb4): reconcile the COO automatically during the restart read. The trigger must fire only after STK
/// commits both persisted id/rank transient fields, which happens after the mesh-creation modification cycle closes,
/// so it cannot be driven from a modification-end callback.
class LinkDataObserver : public stk::mesh::ModificationObserver {
 public:
  explicit LinkDataObserver(LinkData& link_data);

  ~LinkDataObserver() override;

  void entity_added(stk::mesh::Entity entity) override;

  void entity_deleted(stk::mesh::Entity entity) override;

  void entity_parts_added(stk::mesh::Entity entity, const stk::mesh::OrdinalVector& parts) override;

  void entity_parts_removed(stk::mesh::Entity entity, const stk::mesh::OrdinalVector& parts) override;

  void elements_about_to_move_procs_notification(const stk::mesh::EntityProcVec& elem_proc_pairs_to_move) override;

  void elements_moved_procs_notification(const stk::mesh::EntityProcVec& elem_proc_pairs_to_move) override;

  void local_entities_created_or_deleted_notification(stk::mesh::EntityRank rank) override;

  void local_entity_comm_info_changed_notification(stk::mesh::EntityRank rank) override;

  void local_buckets_changed_notification(stk::mesh::EntityRank rank) override;

 private:
  /// \brief Fetch the LinkData we notify.
  LinkData& link_data() const;

  /// \brief Is the given entity of link rank?
  bool is_link_rank(stk::mesh::Entity entity) const;

  LinkData* link_data_ptr_;
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_LINKDATAOBSERVER_HPP_
