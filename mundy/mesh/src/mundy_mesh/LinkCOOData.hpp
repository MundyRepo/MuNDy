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

#ifndef MUNDY_MESH_LINKCOODATA_HPP_
#define MUNDY_MESH_LINKCOODATA_HPP_

/// \file LinkCOOData.hpp
/// \brief Declaration of the LinkCOOData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Kokkos_Sort.hpp>                        // for Kokkos::sort
#include <Kokkos_UnorderedMap.hpp>                // for Kokkos::UnorderedMap
#include <stk_mesh/base/Entity.hpp>               // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>                 // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>             // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>                // for stk::mesh::EntityRank
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets
#include <stk_util/ngp/NgpSpaces.hpp>             // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace
#include <stk_mesh/base/BulkData.hpp>        // for stk::mesh::BulkData

// Mundy libs
#include <mundy_core/throw_assert.hpp>        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkMetaData.hpp>        // for mundy::mesh::LinkMetaData
#include <mundy_mesh/impl/NgpLinkMetaData.hpp> // for mundy::mesh::impl::NgpLinkMetaDataT

namespace mundy {

namespace mesh {

// Forward declare the LinkData
class LinkData;

class LinkCOOData {  // Host only | Valid during mesh modifications
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  LinkCOOData() = default;

  /// \brief Default copy or move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION LinkCOOData(const LinkCOOData &) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCOOData(LinkCOOData &&) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCOOData &operator=(const LinkCOOData &) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCOOData &operator=(LinkCOOData &&) = default;

  /// \brief Canonical constructor.
  explicit LinkCOOData(stk::mesh::BulkData &bulk_data, LinkMetaData &link_meta_data)
      : bulk_data_ptr_(&bulk_data), link_meta_data_ptr_(&link_meta_data) {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
  }

  /// \brief Destructor.
  KOKKOS_FUNCTION
  virtual ~LinkCOOData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the link data is valid.
  inline bool is_valid() const noexcept {
    return link_meta_data_ptr_ != nullptr && bulk_data_ptr_ != nullptr;
  }

  /// \brief Fetch the link meta data manager
  inline const LinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  inline LinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  inline const stk::mesh::BulkData &bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  inline stk::mesh::BulkData &bulk_data() {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the link rank
  inline stk::mesh::EntityRank link_rank() const noexcept {
    return link_meta_data().link_rank();
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
  inline void declare_relation(const stk::mesh::Entity &linker, const stk::mesh::Entity &linked_entity,
                               unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    auto &linked_e_ids_field = impl::get_linked_entity_ids_field(link_meta_data());
    auto &linked_e_ranks_field = impl::get_linked_entity_ranks_field(link_meta_data());
    auto &linked_e_bucket_ids_field = impl::get_linked_entity_bucket_ids_field(link_meta_data());
    auto &linked_e_bucket_ords_field = impl::get_linked_entity_bucket_ords_field(link_meta_data());
    auto &link_needs_updated_field = impl::get_link_crs_needs_updated_field(link_meta_data());

    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = linked_entity.local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = bulk_data().identifier(linked_entity);
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] = bulk_data().entity_rank(linked_entity);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] =
        bulk_data().bucket(linked_entity).bucket_id();
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = bulk_data().bucket_ordinal(linked_entity);
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline void delete_relation(const stk::mesh::Entity &linker, unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    auto &linked_e_ids_field = impl::get_linked_entity_ids_field(link_meta_data());
    auto &linked_e_ranks_field = impl::get_linked_entity_ranks_field(link_meta_data());
    auto &linked_e_bucket_ids_field = impl::get_linked_entity_bucket_ids_field(link_meta_data());
    auto &linked_e_bucket_ords_field = impl::get_linked_entity_bucket_ords_field(link_meta_data());
    auto &link_needs_updated_field = impl::get_link_crs_needs_updated_field(link_meta_data());

    // Intentionally avoids updating the CRS linked entities field so that we can properly detect deletions.
    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = stk::mesh::Entity().local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = stk::mesh::EntityId();
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] =
        static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }

  /// \brief Get the linked entity for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::Entity get_linked_entity(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }

  /// \brief Get the linked entity index for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::FastMeshIndex get_linked_entity_index(const stk::mesh::Entity &linker,
                                                          unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_bucket_ids_field = impl::get_linked_entity_bucket_ids_field(link_meta_data());
    auto &linked_e_bucket_ords_field = impl::get_linked_entity_bucket_ords_field(link_meta_data());
    return stk::mesh::FastMeshIndex(stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal],
                                    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal]);
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::EntityId get_linked_entity_id(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ids_field =  impl::get_linked_entity_ids_field(link_meta_data());
    return stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal];
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ranks_field = impl::get_linked_entity_ranks_field(link_meta_data());
    return static_cast<stk::mesh::EntityRank>(stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal]);
  }
  //@}

 protected:
  /// \brief Get the linked entity for a given linker and link ordinal (as last seen by the CRS connectivity).
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::Entity get_linked_entity_crs(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_crs_field = impl::get_linked_entities_crs_field(link_meta_data());
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_crs_field, linker)[link_ordinal]);
  }

  /// \brief Get if the CRS connectivity for a link needs to be updated.
  inline bool get_link_crs_needs_updated(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_needs_updated_field = impl::get_link_crs_needs_updated_field(link_meta_data());
    return static_cast<bool>(stk::mesh::field_data(link_needs_updated_field, linker)[0]);
  }

 private:
  //! \name Internal members (host only)
  //@{

  stk::mesh::BulkData *bulk_data_ptr_;
  LinkMetaData *link_meta_data_ptr_;
  //@}
};  // LinkCOOData


template<typename NgpMemSpace>
class NgpLinkCOODataT;

namespace impl {
template<typename NgpMemSpace>
NgpLinkMetaDataT<NgpMemSpace> &get_ngp_link_meta_data(NgpLinkCOODataT<NgpMemSpace>& ngp_coo_data);

template<typename NgpMemSpace>
stk::mesh::NgpMesh &get_ngp_mesh(NgpLinkCOODataT<NgpMemSpace>& ngp_coo_data);

template<typename NgpMemSpace>
class NgpCOOToCRSSynchronizerT;
}  // namespace impl

template<typename NgpMemSpace>
class NgpLinkCOODataT {  // Device only | Invalid during mesh modifications | Can become stale after mesh modifications
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NgpLinkCOODataT() = default;

  /// \brief Default copy or move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION NgpLinkCOODataT(const NgpLinkCOODataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkCOODataT(NgpLinkCOODataT &&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkCOODataT &operator=(const NgpLinkCOODataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkCOODataT &operator=(NgpLinkCOODataT &&) = default;

  /// \brief Canonical constructor.
  explicit NgpLinkCOODataT(stk::mesh::BulkData &bulk_data, LinkMetaData &link_meta_data)
      : bulk_data_ptr_(&bulk_data),
        link_meta_data_ptr_(&link_meta_data),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(bulk_data)),
        ngp_link_meta_data_(impl::get_updated_ngp_link_meta_data(link_meta_data)) {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
  }


  /// \brief Construct from a LinkCOOData.
  /// Does NOT perform a deep copy. Simply steals their pointers to the bulk data and meta data.
  explicit NgpLinkCOODataT(LinkCOOData &host_other)
      : bulk_data_ptr_(&host_other.bulk_data()),
        link_meta_data_ptr_(&host_other.link_meta_data()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(host_other.bulk_data())),
        ngp_link_meta_data_(impl::get_updated_ngp_link_meta_data(host_other.link_meta_data())) {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
  }

  /// \brief Destructor.
  KOKKOS_DEFAULTED_FUNCTION
  virtual ~NgpLinkCOODataT() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the link data is valid.
  inline bool is_valid() const noexcept {
    return link_meta_data_ptr_ != nullptr && bulk_data_ptr_ != nullptr;
  }

  /// \brief Fetch the link meta data manager
  inline const LinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  inline LinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  inline const stk::mesh::BulkData &bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  inline stk::mesh::BulkData &bulk_data() {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the link rank
  KOKKOS_FUNCTION
  stk::mesh::EntityRank link_rank() const noexcept {
    return ngp_link_meta_data_.link_rank();
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
        static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
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
  //@}

 protected:
  /// \brief Fetch the ngp mesh
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh &ngp_mesh() const noexcept {
    return ngp_mesh_;
  }
  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh &ngp_mesh() noexcept {
    return ngp_mesh_;
  }

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

 private:
  
  template<typename T>
  friend impl::NgpLinkMetaDataT<T> &impl::get_ngp_link_meta_data(NgpLinkCOODataT<T>& ngp_coo_data);
  
  template<typename T>
  friend stk::mesh::NgpMesh &impl::get_ngp_mesh(NgpLinkCOODataT<T>& ngp_coo_data);
  
  template<typename T>
  friend class impl::NgpCOOToCRSSynchronizerT;

  //! \name Internal members (host only)
  //@{

  stk::mesh::BulkData *bulk_data_ptr_;
  LinkMetaData *link_meta_data_ptr_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  stk::mesh::NgpMesh ngp_mesh_;
  impl::NgpLinkMetaDataT<NgpMemSpace> ngp_link_meta_data_;
  //@}
};  // NgpLinkCOODataT

// Following STK's default naming convention, to make return statements of our functions more readable.
using NgpLinkCOOData = NgpLinkCOODataT<stk::ngp::MemSpace>;

namespace impl {
template<typename NgpMemSpace>
NgpLinkMetaDataT<NgpMemSpace> &get_ngp_link_meta_data(NgpLinkCOODataT<NgpMemSpace>& ngp_coo_data) {
  return ngp_coo_data.ngp_link_meta_data_;
}
template<typename NgpMemSpace>
stk::mesh::NgpMesh &get_ngp_mesh(NgpLinkCOODataT<NgpMemSpace>& ngp_coo_data) {
  return ngp_coo_data.ngp_mesh_;
}
}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKCOODATA_HPP_
