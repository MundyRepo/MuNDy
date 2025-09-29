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

#ifndef MUNDY_MESH_NEW_NGPLINKDATA_HPP_
#define MUNDY_MESH_NEW_NGPLINKDATA_HPP_

/// \file NewNgpLinkData.hpp
/// \brief Declaration of the NewNgpLinkData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <Kokkos_UnorderedMap.hpp>                // for Kokkos::UnorderedMap
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
#include <mundy_core/throw_assert.hpp>        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>            // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>       // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>            // for mundy::mesh::MetaData
#include <mundy_mesh/NewLinkMetaData.hpp>     // for mundy::mesh::NewLinkMetaData
#include <mundy_mesh/NewNgpCRSPartition.hpp>  // for mundy::mesh::NewNgpCRSPartition
#include <mundy_mesh/NewNgpLinkMetaData.hpp>  // for mundy::mesh::NewNgpLinkMetaData
#include <mundy_mesh/NewNgpLinkRequests.hpp>  // for mundy::mesh::NewNgpLinkRequests
#include <mundy_mesh/NgpFieldBLAS.hpp>        // for mundy::mesh::field_copy
#include <mundy_mesh/NewLinkData.hpp>         // for mundy::mesh::NewLinkData

namespace mundy {

namespace mesh {

class NewNgpLinkDataBase {
  virtual update_link_data() = 0;
};

/// \class NewNgpLinkData
template <typename NgpMemSpace>
class NewNgpLinkDataT : public NewNgpLinkDataBase {
 public:
  //! \name Aliases
  //@{

  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  using NgpCRSPartitionView = Kokkos::View<NewNgpCRSPartition *, stk::ngp::UVMMemSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  NewNgpLinkDataT() = default;

  /// \brief Default copy or move constructors/operators.
  NewNgpLinkDataT(const NewNgpLinkDataT &) = default;
  NewNgpLinkDataT(NewNgpLinkDataT &&) = default;
  NewNgpLinkDataT &operator=(const NewNgpLinkDataT &) = default;
  NewNgpLinkDataT &operator=(NewNgpLinkDataT &&) = default;

  /// \brief Canonical constructor.
  /// \param bulk_data [in] The bulk data manager we extend.
  /// \param link_meta_data [in] Our meta data manager.
  NewNgpLinkDataT(NewLinkData& link_data)
      : link_data_ptr_(&link_data),
        bulk_data_ptr_(&link_data.bulk_data()),
        mesh_meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()),
        link_meta_data_ptr_(&link_data.link_meta_data()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(link_data.bulk_data())),
        ngp_link_meta_data_(get_updated_ngp_link_meta_data(link_data.link_meta_data())),
        all_crs_partitions_(link_data.get_all_crs_partitions(NgpMemSpace{})) {
    MUNDY_THROW_REQUIRE(link_data.is_valid(), std::invalid_argument, "Given link meta data is not valid.");
  }

  /// \brief Destructor.
  virtual ~NewNgpLinkDataT() {std::cout << "DESTRUCTOR FOR NewNgpLinkDataT" << std::endl;}
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the link data is valid.
  KOKKOS_INLINE_FUNCTION
  bool is_valid() const {
    return mesh_meta_data_ptr_ != nullptr && link_meta_data_ptr_ != nullptr && bulk_data_ptr_ != nullptr;
  }

  /// \brief Fetch the bulk data's meta data manager
  const MetaData &mesh_meta_data() const {
    MUNDY_THROW_ASSERT(mesh_meta_data_ptr_ != nullptr, std::invalid_argument, "Mesh meta data is not set.");
    return *mesh_meta_data_ptr_;
  }
  MetaData &mesh_meta_data() {
    MUNDY_THROW_ASSERT(mesh_meta_data_ptr_ != nullptr, std::invalid_argument, "Mesh meta data is not set.");
    return *mesh_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  const BulkData &bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }
  BulkData &bulk_data() {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_FUNCTION
  const stk::mesh::NgpMesh &ngp_mesh() const noexcept {
    return ngp_mesh_;
  }
  KOKKOS_FUNCTION
  stk::mesh::NgpMesh &ngp_mesh() noexcept {
    return ngp_mesh_;
  }

  /// \brief Fetch the link meta data manager
  const NewLinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }
  NewLinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the ngp link meta data manager
  KOKKOS_FUNCTION
  const NewNgpLinkMetaData &ngp_link_meta_data() const noexcept {
    return ngp_link_meta_data_;
  }
  KOKKOS_FUNCTION
  NewNgpLinkMetaData &ngp_link_meta_data() noexcept {
    return ngp_link_meta_data_;
  }

  /// \brief Fetch the link data manager we extend
  const NewLinkData &link_data() const {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }
  NewLinkData &link_data() {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }

  /// \brief Fetch the link rank
  KOKKOS_FUNCTION
  stk::mesh::EntityRank link_rank() const noexcept {
    return ngp_link_meta_data_.link_rank();
  }
  //@}

  //! \name Base type methods
  //@{

  void update_link_data() override {
    ngp_mesh().update_mesh();
    ngp_link_meta_data_.ngp_linked_entities_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_linked_entities_crs_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_linked_entity_ids_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().update_field(NgpMemSpace{});
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().update_field(NgpMemSpace{});
  }
  //@}

  //! \name Dynamic link to linked entity relationships
  //@{

  /// \brief Declare a relation between a linker and a linked entity.
  ///
  /// # To explain ordinals:
  /// If a linker has dimensionality 3 then it can have up to 3 linked entities. The first
  /// linked entity has ordinal 0, the second has ordinal 1, and so on.
  ///
  /// Importantly, the relationship between links and its linked entities is static with fixed size.
  /// If you fetch the linked entities and have only declared the first two, then the third will be invalid.
  /// This is a slight deviation from STK, which would return a set of two valid entities and provide access to their
  /// ordinals.
  ///
  /// # How does a link attain a certain dimensionality?
  /// A link's dimensionality is determined by the set of parts that it belongs to. When link parts are declared, they
  /// are assigned a dimensionality. If a link belongs to multiple link parts, then the maximum dimensionality of
  /// those parts is the link's dimensionality.
  ///
  /// TODO(palmerb4): Bounds check the link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param linked_entity [in] The linked entity (may be invalid).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  void declare_relation(const stk::mesh::FastMeshIndex &linker_index,         //
                        const stk::mesh::EntityRank &linked_entity_rank,      //
                        const stk::mesh::FastMeshIndex &linked_entity_index,  //
                        unsigned link_ordinal) const {
    stk::mesh::Entity linked_entity = ngp_mesh_.get_entity(linked_entity_rank, linked_entity_index);
    stk::mesh::EntityKey linked_entity_key = ngp_mesh_.entity_key(linked_entity);

    ngp_link_meta_data_.ngp_linked_entities_field()(linker_index, link_ordinal) = linked_entity.local_offset();
    ngp_link_meta_data_.ngp_linked_entity_ids_field()(linker_index, link_ordinal) = linked_entity_key.id();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field()(linker_index, link_ordinal) = linked_entity_rank;
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field()(linker_index, link_ordinal) =
        linked_entity_index.bucket_id;
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field()(linker_index, link_ordinal) =
        linked_entity_index.bucket_ord;
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field()(linker_index, 0) = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  void delete_relation(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    ngp_link_meta_data_.ngp_linked_entities_field()(linker_index, link_ordinal) = stk::mesh::Entity().local_offset();
    ngp_link_meta_data_.ngp_linked_entity_ids_field()(linker_index, link_ordinal) = stk::mesh::EntityId();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field()(linker_index, link_ordinal) =
        static_cast<NewLinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field()(linker_index, link_ordinal) = 0;
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field()(linker_index, link_ordinal) = 0;
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field()(linker_index, 0) = true;
  }

  /// \brief Get the linked entity for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_linked_entity(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return stk::mesh::Entity(ngp_link_meta_data_.ngp_linked_entities_field()(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity index for a given linker and link ordinal.
  ///
  /// TODO(palmerb4): With Trilinos 16.2, we can remove the bucket_ids and bucket_ords fields and just use
  /// the ngp_mesh.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex get_linked_entity_index(const stk::mesh::FastMeshIndex &linker_index,
                                                   unsigned link_ordinal) const {
    return stk::mesh::FastMeshIndex(
        ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field()(linker_index, link_ordinal),
        ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field()(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId get_linked_entity_id(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return ngp_link_meta_data_.ngp_linked_entity_ids_field()(linker_index, link_ordinal);
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::FastMeshIndex &linker_index,
                                               unsigned link_ordinal) const {
    return static_cast<stk::mesh::EntityRank>(
        ngp_link_meta_data_.ngp_linked_entity_ranks_field()(linker_index, link_ordinal));
  }

  /// \brief Get all CRS partitions.
  ///
  /// Safety: This view is always safe to use and the reference is valid as long as the NewNgpLinkDataT is valid, which
  /// has the same lifetime as the bulk data manager. It remains valid even during mesh modifications.
  KOKKOS_INLINE_FUNCTION
  const NgpCRSPartitionView &get_all_crs_partitions() const noexcept {
    return all_crs_partitions_;
  }

  /// \brief Get the CRS partitions for a given link subset selector (Memoized/host only/not thread safe).
  ///
  /// This is the only way for either mundy or users to create a new partition. The returned view is persistent
  /// but its contents/size will change dynamically as new partitions are created and destroyed. The only promise
  /// we will make is to never delete a partition outside of a modification cycle.
  const NgpCRSPartitionView &get_or_create_crs_partitions(const stk::mesh::Selector &selector) {
    return link_data().template get_or_create_crs_partitions<NgpMemSpace>(selector);
  }

  /// \brief Get all links in the given partition that connect to the given entity in the CRS connectivity.
  KOKKOS_INLINE_FUNCTION
  ConnectedEntities get_connected_links(const stk::mesh::Ordinal partition_id, stk::mesh::EntityRank rank,
                                        const stk::mesh::FastMeshIndex &entity_index) const {
    return get_all_crs_partitions()[partition_id].get_connected_links(rank, entity_index);
  }

  /// \brief Get the number of links in the given partition that connect to the given entity in the CRS connectivity.
  KOKKOS_INLINE_FUNCTION
  unsigned num_connected_links(const stk::mesh::Ordinal partition_id, stk::mesh::EntityRank rank,
                               const stk::mesh::FastMeshIndex &entity_index) const {
    return get_all_crs_partitions()[partition_id].num_connected_links(rank, entity_index);
  }

  /// \brief Get the number of current CRS partitions (they aren't all necessarily up-to-date).
  /// partition_id is contiguous in [0, num_crs_partitions).
  KOKKOS_INLINE_FUNCTION
  unsigned num_crs_partitions() const {
    return get_all_crs_partitions().extent(0);
  }

  /// \brief Check if the CRS connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CRS buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  bool is_crs_up_to_date(const stk::mesh::Selector &selector) {
    return link_data().template is_crs_up_to_date<NgpMemSpace>(selector);
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  bool is_crs_up_to_date() {
    return link_data().template is_crs_up_to_date<NgpMemSpace>();
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  void update_crs_from_coo(const stk::mesh::Selector &selector) {
    link_data().template update_crs_from_coo<NgpMemSpace>(selector);
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  void update_crs_from_coo() {
    link_data().template update_crs_from_coo<NgpMemSpace>();
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  void check_crs_coo_consistency(const stk::mesh::Selector &selector) {
    link_data().template check_crs_coo_consistency<NgpMemSpace>(selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  void check_crs_coo_consistency() {
    link_data().template check_crs_coo_consistency<NgpMemSpace>();
  }
  //@}

  //! \name Delayed creation and destruction
  //@{

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call.
  KOKKOS_INLINE_FUNCTION
  void request_destruction(const stk::mesh::FastMeshIndex &linker_index) const {
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field()(linker_index, 0) = true;
  }

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call (host version).
  inline void request_destruction_host(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_marked_for_destruction_field = link_meta_data().link_marked_for_destruction_field();
    stk::mesh::field_data(link_marked_for_destruction_field, linker)[0] = true;
  }

  /// \brief Get a helper for requesting links for a collection of parts (memoized).
  const NewNgpLinkRequests &get_or_create_link_requests(const stk::mesh::PartVector &link_parts,
                                                        unsigned requested_dimensionality = 0,
                                                        unsigned requested_capacity = 0) const {
    // Order and repetition don't matter for the link parts, so we sort and uniquify them.
    stk::mesh::PartVector sorted_uniqued_link_parts = link_parts;
    stk::util::sort_and_unique(sorted_uniqued_link_parts);

    auto it = part_vector_to_request_links_map_.find(sorted_uniqued_link_parts);
    if (it != part_vector_to_request_links_map_.end()) {
      // Return the existing helper
      return it->second;
    } else {
      // Create a new helper
      auto [other_it, inserted] = part_vector_to_request_links_map_.try_emplace(
          sorted_uniqued_link_parts, link_meta_data(), sorted_uniqued_link_parts, requested_dimensionality,
          requested_capacity);
      return other_it->second;
    }
  }

  /// \brief Process all requests for creation/destruction made since the last process_requests call.
  ///
  /// Note, on a single process or if the entities you wish to link are all of element rank or higher, then partial
  /// consistency is the same as full consistency.
  ///
  /// If the global number of requests is non-zero, this function will enter a modification cycle if not already in one.
  ///
  /// \param assume_fully_consistent [in] If we should assume that the requests are fully consistent or not.
  void process_requests(bool assume_fully_consistent = false) {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Processing requests not implemented yet.");
  }
  //@}

  //! \name STK NGP interface
  //@{

  void modify_coo_on_host() {
    ngp_link_meta_data_.ngp_linked_entities_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().modify_on_host();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().modify_on_host();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().modify_on_host();
  }

  void modify_coo_on_device() {
    ngp_link_meta_data_.ngp_linked_entities_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().modify_on_device();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().modify_on_device();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().modify_on_device();
  }

  void sync_coo_to_host() {
    ngp_link_meta_data_.ngp_linked_entities_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().sync_to_host();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().sync_to_host();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().sync_to_host();
  }

  void sync_coo_to_device() {
    ngp_link_meta_data_.ngp_linked_entities_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().sync_to_device();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().sync_to_device();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().sync_to_device();
  }

  /// \brief Synchronize the CRS connectivity with the most up-to-date memory space.
  void sync_crs_to_device() {
    link_data().template sync_crs<NgpMemSpace>();
  }
  void sync_crs_to_host() {
    link_data().sync_crs_to_host();
  }
  //@}

 protected:
  //! \name Internal aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ngp_linked_entities_field_t = stk::mesh::NgpField<entity_value_t>;
  using ngp_linked_entity_ids_field_t = stk::mesh::NgpField<NewLinkMetaData::entity_id_value_t>;
  using ngp_linked_entity_ranks_field_t = stk::mesh::NgpField<NewLinkMetaData::entity_rank_value_t>;
  using ngp_linked_entity_bucket_ids_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_linked_entity_bucket_ords_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_link_crs_needs_updated_field_t = stk::mesh::NgpField<int>;
  using ngp_link_marked_for_destruction_field_t = stk::mesh::NgpField<unsigned>;
  //@}

  //! \name Friends <3
  //@{

  template <typename LinkDataType, typename OtherNgpMemSpace>
  friend class impl::NgpLinkDataCRSManagerT;
  //@}

  //! \name Internal actions
  //@{

  /// \brief Get the linked entity for a given linker and link ordinal (as last seen by the CRS connectivity).
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_linked_entity_crs(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return stk::mesh::Entity(ngp_link_meta_data_.ngp_linked_entities_crs_field()(linker_index, link_ordinal));
  }

  /// \brief Get if the CRS connectivity for a link needs to be updated.
  KOKKOS_INLINE_FUNCTION
  bool get_link_crs_needs_updated(const stk::mesh::FastMeshIndex &linker_index) const {
    return ngp_link_meta_data_.ngp_link_crs_needs_updated_field()(linker_index, 0);
  }

  /// \brief Destroy all links that have been marked for destruction.
  void destroy_marked_links() {
    auto &link_marked_for_destruction_field = link_meta_data().link_marked_for_destruction_field();
    stk::mesh::EntityVector links_to_maybe_destroy;
    stk::mesh::get_selected_entities(link_meta_data().universal_link_part(),
                                     bulk_data().buckets(link_meta_data().link_rank()), links_to_maybe_destroy);

    for (const stk::mesh::Entity &link : links_to_maybe_destroy) {
      const bool should_destroy_entity =
          static_cast<bool>(stk::mesh::field_data(link_marked_for_destruction_field, link)[0]);
      if (should_destroy_entity) {
        bool success = bulk_data().destroy_entity(link);
        MUNDY_THROW_ASSERT(success, std::runtime_error,
                           fmt::format("Failed to destroy link. Link rank: {}, entity id: {}",
                                       bulk_data().entity_rank(link), bulk_data().identifier(link)));
      }
    }
  }
  //@}

 private:
  //! \name Internal members (host only)
  //@{

  NewLinkData *link_data_ptr_;
  BulkData *bulk_data_ptr_;
  MetaData *mesh_meta_data_ptr_;
  NewLinkMetaData *link_meta_data_ptr_;

  using PartVectorToRequestLinks =
      std::map<stk::mesh::PartVector, NewNgpLinkRequests>;  // TODO(palmerb4): Move to a request manager class.
  mutable PartVectorToRequestLinks part_vector_to_request_links_map_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  stk::mesh::NgpMesh ngp_mesh_;
  NewNgpLinkMetaData ngp_link_meta_data_;
  NgpCRSPartitionView all_crs_partitions_;
  //@}
};  // NewNgpLinkDataT

using NewNgpLinkData = NewNgpLinkDataT<stk::ngp::MemSpace>;

/// \brief Get an updated ngp link data object.
// inline NewNgpLinkData get_updated_ngp_link_data(NewLinkData &link_data) {
//   MUNDY_THROW_REQUIRE(link_data.is_valid(), std::invalid_argument, "Given link data is not valid.");
//   return NewNgpLinkData(link_data);
// }

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_NGPLINKDATA_HPP_
