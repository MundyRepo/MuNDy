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

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/Entity.hpp>               // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>                // for stk::mesh::Field
#include <stk_mesh/base/FindRestriction.hpp>      // for stk::mesh::find_restriction
#include <stk_mesh/base/GetEntities.hpp>          // for stk::mesh::get_selected_entities
#include <stk_mesh/base/GetNgpField.hpp>          // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>           // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>             // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>              // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>                 // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>             // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>                // for stk::mesh::EntityRank
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets

// Mundy libs
#include <mundy_core/NgpView.hpp>        // for mundy::core::NgpView
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>   // for mundy::mesh::field_copy

namespace mundy {

namespace mesh {

class LinkDestructionRequests {  // Host only | Valid during mesh modifications
 public:
  /// \brief Default constructor.
  LinkDestructionRequests() = default;

  /// \brief Canonical constructor.
  LinkDestructionRequests(const LinkMetaData &link_meta_data, const stk::mesh::PartVector &add_parts,
                          unsigned link_dimensionality, unsigned initial_capacity)
      : link_meta_data_ptr_(&link_meta_data),
        link_parts_(add_parts),
        link_rank_(link_meta_data.link_rank()),
        link_dimensionality_(link_dimensionality),
        capacity_("capacity"),
        size_("size"),
        requests_(Kokkos::view_alloc(Kokkos::WithoutInitializing, "requests"), initial_capacity, link_dimensionality_) {
    MUNDY_THROW_ASSERT(!link_parts_.empty(), std::invalid_argument, "Link requests must have at least one part.");
    MUNDY_THROW_ASSERT(link_rank_ != stk::topology::INVALID_RANK, std::invalid_argument, "Link rank must be valid.");
    MUNDY_THROW_ASSERT(link_dimensionality_ > 0, std::invalid_argument,
                       "Link dimensionality must be greater than zero.");
    MUNDY_THROW_ASSERT(
        get_linker_dimensionality_host(add_parts) >= link_dimensionality_, std::invalid_argument,
        "Requested a dimensionality that is smaller, than the dimensionality supported by the given parts.");

    Kokkos::deep_copy(capacity_.view_device(), initial_capacity);
    Kokkos::deep_copy(size_.view_device(), 0);

    // Sync up the host views. If host == device, this is a no-op.
    capacity_.modify_on_device();
    size_.modify_on_device();
    capacity_.sync_to_host();
    size_.sync_to_host();
  }

  /// \brief Default copy/move constructors/operators.
  LinkDestructionRequests(const LinkDestructionRequests &) = default;
  LinkDestructionRequests(LinkDestructionRequests &&) = default;
  LinkDestructionRequests &operator=(const LinkDestructionRequests &) = default;
  LinkDestructionRequests &operator=(LinkDestructionRequests &&) = default;

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call.
  inline void request_destruction(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_marked_for_destruction_field = impl::get_link_marked_for_destruction_field(link_meta_data());
    stk::mesh::field_data(link_marked_for_destruction_field, linker)[0] = true;
  }

};  // LinkDestructionRequests

template <typename NgpMemSpace>
class NgpLinkDestructionRequestsT {  // Device only | Invalid during mesh modifications | Can become stale after mesh
                                     // modifications
 public:
  /// \brief Default constructor.
  NgpLinkDestructionRequestsT() = default;

  /// \brief Canonical constructor.
  NgpLinkDestructionRequestsT(const LinkMetaData &link_meta_data, const stk::mesh::PartVector &add_parts,
                              unsigned link_dimensionality, unsigned initial_capacity)
      : link_meta_data_ptr_(&link_meta_data),
        link_parts_(add_parts),
        link_rank_(link_meta_data.link_rank()),
        link_dimensionality_(link_dimensionality),
        capacity_("capacity"),
        size_("size"),
        requests_(Kokkos::view_alloc(Kokkos::WithoutInitializing, "requests"), initial_capacity, link_dimensionality_) {
    MUNDY_THROW_ASSERT(!link_parts_.empty(), std::invalid_argument, "Link requests must have at least one part.");
    MUNDY_THROW_ASSERT(link_rank_ != stk::topology::INVALID_RANK, std::invalid_argument, "Link rank must be valid.");
    MUNDY_THROW_ASSERT(link_dimensionality_ > 0, std::invalid_argument,
                       "Link dimensionality must be greater than zero.");
    MUNDY_THROW_ASSERT(
        get_linker_dimensionality_host(add_parts) >= link_dimensionality_, std::invalid_argument,
        "Requested a dimensionality that is smaller, than the dimensionality supported by the given parts.");

    Kokkos::deep_copy(capacity_.view_device(), initial_capacity);
    Kokkos::deep_copy(size_.view_device(), 0);

    // Sync up the host views. If host == device, this is a no-op.
    capacity_.modify_on_device();
    size_.modify_on_device();
    capacity_.sync_to_host();
    size_.sync_to_host();
  }

  /// \brief Default copy/move constructors/operators.
  NgpLinkDestructionRequestsT(const NgpLinkDestructionRequestsT &) = default;
  NgpLinkDestructionRequestsT(NgpLinkDestructionRequestsT &&) = default;
  NgpLinkDestructionRequestsT &operator=(const NgpLinkDestructionRequestsT &) = default;
  NgpLinkDestructionRequestsT &operator=(NgpLinkDestructionRequestsT &&) = default;

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call.
  inline void request_destruction(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_marked_for_destruction_field = impl::get_link_marked_for_destruction_field(link_meta_data());
    stk::mesh::field_data(link_marked_for_destruction_field, linker)[0] = true;
  }
};  // NgpLinkDestructionRequestsT

using NgpLinkDestructionRequests = NgpLinkDestructionRequestsT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDESTRUCTIONREQUESTS_HPP_
