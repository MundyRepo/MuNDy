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

#ifndef MUNDY_MESH_LINKDESTRUCTIONREQUESTS_HPP_
#define MUNDY_MESH_LINKDESTRUCTIONREQUESTS_HPP_

/// \file LinkDestructionRequests.hpp

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::field_data
#include <stk_util/ngp/NgpSpaces.hpp>  // for stk::ngp::MemSpace

// Mundy libs
#include <mundy_mesh/LinkMetaData.hpp>   // for mundy::mesh::LinkMetaData, impl::get_link_marked_for_destruction_field
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

/// \class LinkDestructionRequests
/// \brief Host-side request queue for link destruction.
class LinkDestructionRequests {  // Host only | Valid during mesh modifications
 public:
  //! \name Constructors
  //@{

  LinkDestructionRequests() = default;

  explicit LinkDestructionRequests(stk::mesh::BulkData& bulk_data, LinkMetaData& link_meta_data)
      : bulk_data_ptr_(&bulk_data), link_meta_data_ptr_(&link_meta_data) {}

  LinkDestructionRequests(const LinkDestructionRequests&) = default;
  LinkDestructionRequests(LinkDestructionRequests&&) = default;
  LinkDestructionRequests& operator=(const LinkDestructionRequests&) = default;
  LinkDestructionRequests& operator=(LinkDestructionRequests&&) = default;
  //@}

  //! \name Getters
  //@{

  const LinkMetaData& link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Mark `linker` for destruction. Processed on the next `LinkData::process_requests()` call.
  void request_destruction(const stk::mesh::Entity& linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");
    auto& field = impl::get_link_marked_for_destruction_field(link_meta_data());
    stk::mesh::field_data(field, linker)[0] = true;
  }
  //@}

 private:
  stk::mesh::BulkData* bulk_data_ptr_ = nullptr;
  LinkMetaData* link_meta_data_ptr_ = nullptr;
};  // LinkDestructionRequests

/// \class NgpLinkDestructionRequestsT
/// \brief Device-side request queue for link destruction.
template <typename NgpMemSpace>
class NgpLinkDestructionRequestsT {  // Device only | Invalid during mesh modifications
 public:
  //! \name Constructors
  //@{

  NgpLinkDestructionRequestsT() = default;

  explicit NgpLinkDestructionRequestsT(stk::mesh::BulkData& bulk_data, LinkMetaData& link_meta_data)
      : bulk_data_ptr_(&bulk_data), link_meta_data_ptr_(&link_meta_data) {}

  NgpLinkDestructionRequestsT(const NgpLinkDestructionRequestsT&) = default;
  NgpLinkDestructionRequestsT(NgpLinkDestructionRequestsT&&) = default;
  NgpLinkDestructionRequestsT& operator=(const NgpLinkDestructionRequestsT&) = default;
  NgpLinkDestructionRequestsT& operator=(NgpLinkDestructionRequestsT&&) = default;
  //@}

  //! \name Getters
  //@{

  const LinkMetaData& link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Mark `linker` for destruction. Processed on the next `LinkData::process_requests()` call.
  void request_destruction(const stk::mesh::Entity& linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");
    auto& field = impl::get_link_marked_for_destruction_field(link_meta_data());
    stk::mesh::field_data(field, linker)[0] = true;
  }
  //@}

 private:
  stk::mesh::BulkData* bulk_data_ptr_ = nullptr;
  LinkMetaData* link_meta_data_ptr_ = nullptr;
};  // NgpLinkDestructionRequestsT

using NgpLinkDestructionRequests = NgpLinkDestructionRequestsT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDESTRUCTIONREQUESTS_HPP_
