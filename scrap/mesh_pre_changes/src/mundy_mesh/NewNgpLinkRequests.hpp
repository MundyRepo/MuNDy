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

#ifndef MUNDY_MESH_NEW_NGPLINKREQUEST_HPP_
#define MUNDY_MESH_NEW_NGPLINKREQUEST_HPP_

/// \file NewNgpLinkRequests.hpp

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
#include <mundy_core/NgpView.hpp>         // for mundy::core::NgpView
#include <mundy_core/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>        // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>        // for mundy::mesh::MetaData
#include <mundy_mesh/NewNgpLinkData.hpp>  // for mundy::mesh::NewNgpLinkData
#include <mundy_mesh/NgpFieldBLAS.hpp>    // for mundy::mesh::field_copy

namespace mundy {

namespace mesh {

template <typename NgpMemSpace>
class NewNgpLinkRequestsT {
 public:
  /// \brief Default constructor.
  NewNgpLinkRequestsT() = default;

  /// \brief Canonical constructor.
  NewNgpLinkRequestsT(const NewLinkMetaData &link_meta_data, const stk::mesh::PartVector &add_parts,
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
  NewNgpLinkRequestsT(const NewNgpLinkRequestsT &) = default;
  NewNgpLinkRequestsT(NewNgpLinkRequestsT &&) = default;
  NewNgpLinkRequestsT &operator=(const NewNgpLinkRequestsT &) = default;
  NewNgpLinkRequestsT &operator=(NewNgpLinkRequestsT &&) = default;

  /// \brief Fetch the link meta data.
  const NewLinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::runtime_error,
                       "Attempting to access link meta data before it has been set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the part vector within which the links will be declared.
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::PartVector &get_link_parts() const {
    return link_parts_;
  }

  /// \brief Fetch the link rank.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank link_rank() const {
    return link_rank_;
  }

  /// \brief Fetch the link dimensionality (requested, not actual).
  KOKKOS_INLINE_FUNCTION
  unsigned link_dimensionality() const {
    return link_dimensionality_;
  }

  /// \brief Get the current capacity of request_link requests.
  KOKKOS_INLINE_FUNCTION
  size_t capacity() const {
    return capacity_.view_device()();
  }
  /// \brief Get the current capacity of request_link requests.
  inline size_t capacity_host() const {
    return capacity_.view_host()();
  }

  /// \brief Get the current number of request_link requests.
  KOKKOS_INLINE_FUNCTION
  size_t size() const {
    return size_.view_device()();
  }
  /// \brief Get the current number of request_link requests.
  inline size_t size_host() const {
    return size_.view_host()();
  }

  /// \brief Clear all requests.
  void clear() {
    Kokkos::deep_copy(size_.view_device()(), 0);
    size_.modify_on_device();
    size_.sync_to_host();
  }

  /// \brief Reserve a given number of request_link requests.
  ///
  /// This must be called before attempting to make any request_link requests. Importantly, only locally owned
  /// partitions can request links (i.e., those that contain the locally owned part). This is similar to how STK
  /// declares entities since all declared entities begin in the locally owned part.
  ///
  /// Because we are using GPUs and cannot have dynamic memory allocation, we need to reserve a given number of
  /// requests upfront. This sets a capacity for the number of requests that can be made. In debug mode, we will
  /// throw if you try to make more requests than the capacity.
  ///
  /// If you reserve less than the current capacity, the capacity remains unchanged.
  void reserve(size_t new_capacity) {
    if (new_capacity > capacity_host()) {
      Kokkos::deep_copy(capacity_.view_device(), new_capacity);
      capacity_.modify_on_device();
      capacity_.sync_to_host();

      requests_.resize(new_capacity, link_dimensionality_);
    }
  }

  /// \brief Request a link between the given entities. This will be processed in the next process_requests call.
  ///
  /// This function can be called like request_link(linked_entity0, linked_entity1) to request a link between
  /// entities 0 and 1. The number of entities you pass in must match the link dimensionality of the partition.
  ///
  /// This function is thread safe but is assumed to be called relatively infrequently.
  ///
  /// \param linked_entities [in] Any number of entities to link.
  template <typename... LinkedEntities>
    requires(std::is_same_v<std::decay_t<LinkedEntities>, stk::mesh::Entity> && ...)
  KOKKOS_INLINE_FUNCTION void request_link(LinkedEntities &&...linked_entities) const {
    MUNDY_THROW_ASSERT(link_dimensionality() >= sizeof...(linked_entities), std::invalid_argument,
                       "The number of linked entities cannot exceed the link dimensionality.");

    // For those not familiar with atomic_fetch_sub, it returns the value before the subtraction.
    size_t old_size = Kokkos::atomic_fetch_add(&size_.view_device()(), 1);

    MUNDY_THROW_ASSERT(old_size + 1 <= capacity_.view_device()(), std::invalid_argument,
                       "The number of requests exceeds the capacity.");
    insert_request(std::make_index_sequence<sizeof...(linked_entities)>(), old_size,
                   std::forward<LinkedEntities>(linked_entities)...);
  }
  template <typename... LinkedEntities>
    requires(std::is_same_v<std::decay_t<LinkedEntities>, stk::mesh::Entity> && ...)
  inline void request_link_host(LinkedEntities &&...linked_entities) const {
    MUNDY_THROW_ASSERT(link_dimensionality() >= sizeof...(linked_entities), std::invalid_argument,
                       "The number of linked entities cannot exceed the link dimensionality.");

    // For those not familiar with atomic_fetch_sub, it returns the value before the subtraction.
    size_t old_size = Kokkos::atomic_fetch_add(&size_.view_host()(), 1);

    MUNDY_THROW_ASSERT(old_size + 1 <= capacity_.view_host()(), std::invalid_argument,
                       "The number of requests exceeds the capacity.");
    insert_request_host(std::make_index_sequence<sizeof...(linked_entities)>(), old_size,
                        std::forward<LinkedEntities>(linked_entities)...);
  }

  /// \brief Process all requests for creation/destruction made since the last process_requests call.
  ///
  /// Note, on a single process or if the entities you wish to link are all of element rank or higher, then partial
  /// consistency is the same as full consistency.
  ///
  /// If the global number of requests is non-zero, this function will enter a modification cycle if not already in one.
  ///
  /// \param assume_fully_consistent [in] If we should assume that the requests are fully consistent or not.
  // void process_requests(bool assume_fully_consistent = false) {
  //   MUNDY_THROW_REQUIRE(size_() <= capacity_(), std::invalid_argument,
  //                       "The number of requests exceeds the capacity. You wrote to invalid memory when requesting "
  //                       "links and somehow didn't get a segfault. Neat!");
  //   size_t global_requests_size = 0;
  //   stk::all_reduce_sum(bulk_data_.parallel(), &size_(), &global_requests_size, 1);

  //   if (global_requests_size > 0) {
  //     bool we_started_modification = false;
  //     if (!bulk_data_.in_modifiable_state()) {
  //       bulk_data_.modification_begin();
  //       we_started_modification = true;
  //     }

  //     if (assume_fully_consistent) {
  //       if (bulk_data_.parallel_size() == 1) {
  //         process_link_requests_fully_consistent_single_process();
  //       } else {
  //         process_link_requests_fully_consistent_multi_process();
  //       }
  //     } else {
  //       if (bulk_data_.parallel_size() == 1) {
  //         process_link_requests_partially_consistent_single_process();
  //       } else {
  //         process_link_requests_partially_consistent_multi_process();
  //       }
  //     }

  //     if (we_started_modification) {
  //       bulk_data_.modification_end();
  //     }
  //   }
  // }

 private:
  //! \name Internal functions
  //@{

  /// \brief Get the dimensionality for a collection of linker parts
  inline unsigned get_linker_dimensionality_host(const stk::mesh::PartVector &parts) const {
    // The restriction may be empty if the parts are not a subset of the universal link part.
    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data);
    const stk::mesh::FieldRestriction &restriction = stk::mesh::find_restriction(linked_es_field, link_rank_, parts);
    return restriction.num_scalars_per_entity();
  }

  /// \brief Unrole the entities into the requested links view.
  template <size_t... Is, typename... LinkedEntities>
  KOKKOS_INLINE_FUNCTION void insert_request(std::index_sequence<Is...>, size_t request_index,
                                             LinkedEntities &&...linked_entities) const {
    ((requests_.view_device()(request_index, Is) = std::forward<LinkedEntities>(linked_entities)), ...);
  }
  template <size_t... Is, typename... LinkedEntities>
  inline void insert_request_host(std::index_sequence<Is...>, size_t request_index,
                                  LinkedEntities &&...linked_entities) const {
    ((requests_.view_host()(request_index, Is) = std::forward<LinkedEntities>(linked_entities)), ...);
  }

  /// \brief Process all link requests (fully consistent, multiple processes)
  void process_link_requests_fully_consistent_multi_process();

  /// \brief Process all link requests (fully consistent, single process)
  void process_link_requests_fully_consistent_single_process();

  /// \brief Process all link requests (partially consistent, multiple processes)
  void process_link_requests_partially_consistent_multi_process();

  /// \brief Process all link requests (partially consistent, single process)
  void process_link_requests_partially_consistent_single_process();
  //@}

  //! \name Internal members
  //@{

  using SizeDualView = core::NgpViewT<size_t, NgpMemSpace>;
  using RequestsDualView = core::NgpViewT<stk::mesh::Entity **, NgpMemSpace>;

  // Core data
  const NewLinkMetaData *link_meta_data_ptr_;
  stk::mesh::PartVector link_parts_;
  stk::mesh::EntityRank link_rank_;
  unsigned link_dimensionality_;
  SizeDualView capacity_;
  SizeDualView size_;
  RequestsDualView requests_;
  //@}
};  // NewNgpLinkRequestsT

using NewNgpLinkRequests = NewNgpLinkRequestsT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_NGPLINKREQUEST_HPP_
