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

#ifndef MUNDY_MESH_NGPLINKMETADATA_HPP_
#define MUNDY_MESH_NGPLINKMETADATA_HPP_

/// \file NewNgpLinkMetaData.hpp
/// \brief Declaration of the NewNgpLinkMetaData class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <string>     // for std::string
#include <typeindex>  // for std::type_index
#include <vector>     // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>      // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>  // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NewLinkMetaData.hpp>  // for mundy::mesh::NewLinkMetaData

namespace mundy {

namespace mesh {

namespace impl {
  // Forward declaration (needed due to declaring a friend in a namespaced scope)
  template <typename NgpMemSpace>
  class NgpLinkDataCRSManagerT;
}

class NewNgpLinkMetaData {
 public:
  //! \name Type aliases
  //@{

  using entity_rank_value_t = std::underlying_type_t<stk::mesh::EntityRank>;
  using entity_id_value_t = stk::mesh::EntityId;
  using entity_value_t = stk::mesh::Entity::entity_value_type;

  using linked_entity_ids_field_t = stk::mesh::Field<entity_id_value_t>;
  using linked_entity_ranks_field_t = stk::mesh::Field<entity_rank_value_t>;
  using linked_entities_field_t = stk::mesh::Field<entity_value_t>;

  using ngp_linked_entity_ids_field_t = stk::mesh::NgpField<NewLinkMetaData::entity_id_value_t>;
  using ngp_linked_entity_ranks_field_t = stk::mesh::NgpField<NewLinkMetaData::entity_rank_value_t>;
  using ngp_linked_entities_field_t = stk::mesh::NgpField<entity_value_t>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Defualt constructor.
  NewNgpLinkMetaData() = default;

  /// \brief Canonical constructor.
  NewNgpLinkMetaData(NewLinkMetaData &link_meta_data)
      : link_meta_data_ptr_(&link_meta_data),
        link_rank_(link_meta_data.link_rank()),
        ngp_linked_entities_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entities_field_t::value_type>(
                link_meta_data.linked_entities_field())),
        ngp_linked_entities_crs_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entities_field_t::value_type>(
                link_meta_data.linked_entities_crs_field())),
        ngp_linked_entity_ids_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entity_ids_field_t::value_type>(
                link_meta_data.linked_entity_ids_field())),
        ngp_linked_entity_ranks_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entity_ranks_field_t::value_type>(
                link_meta_data.linked_entity_ranks_field())),
        ngp_linked_entity_bucket_ids_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entity_bucket_ids_field_t::value_type>(
                link_meta_data.linked_entity_bucket_ids_field())),
        ngp_linked_entity_bucket_ords_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::linked_entity_bucket_ords_field_t::value_type>(
                link_meta_data.linked_entity_bucket_ords_field())),
        ngp_link_crs_needs_updated_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::link_crs_needs_updated_field_t::value_type>(
                link_meta_data.link_crs_needs_updated_field())),
        ngp_link_marked_for_destruction_field_(
            stk::mesh::get_updated_ngp_field<NewLinkMetaData::link_marked_for_destruction_field_t::value_type>(
                link_meta_data.link_marked_for_destruction_field())),
        universal_link_part_ord_(link_meta_data.universal_link_part().mesh_meta_data_ordinal()) {
  }

  /// \brief Destructor.
  virtual ~NewNgpLinkMetaData() = default;

  /// \brief Default copy/move constructors/operators.
  NewNgpLinkMetaData(const NewNgpLinkMetaData &) = default;
  NewNgpLinkMetaData(NewNgpLinkMetaData &&) = default;
  NewNgpLinkMetaData &operator=(const NewNgpLinkMetaData &) = default;
  NewNgpLinkMetaData &operator=(NewNgpLinkMetaData &&) = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the this object is in a valid state.
  KOKKOS_INLINE_FUNCTION
  bool is_valid() const {
    return link_meta_data_ptr_ != nullptr;
  }

  /// \brief Get the link meta data object (if valid).
  const NewLinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(is_valid(), std::runtime_error, "Attempted to access an invalid link meta data.");
    return *link_meta_data_ptr_;
  }
  NewLinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(is_valid(), std::runtime_error, "Attempted to access an invalid link meta data.");
    return *link_meta_data_ptr_;
  }

  /// \brief Get the name of this link data.
  const std::string &name() const {
    return link_meta_data().name();
  }

  /// \brief Fetch the link rank.
  stk::mesh::EntityRank link_rank() const {
    return link_rank_;
  }

  /// \brief Fetch the linked entity ids field.
  ///
  /// \note Users should not edit this field yourself. We expose it to you because it's how you'll interact with the
  /// linked entities when doing things like post-processing the output EXO file, but it should be seen as read-only.
  /// Use declare/delete_relation to modify it since they perform additional behind-the-scenes bookkeeping.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_ids_field_t &ngp_linked_entity_ids_field() const {
    return ngp_linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  ///
  /// Same comment as linked_entity_ids_field. Treat this field as read-only.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_ranks_field_t &ngp_linked_entity_ranks_field() const {
    return ngp_linked_entity_ranks_field_;
  }

  /// \brief Fetch the universal link part.
  stk::mesh::PartOrdinal universal_link_part_ord() const {
    return universal_link_part_ord_;
  }
  //@}

 protected:
  //! \name Internal aliases
  //@{

  using linked_entity_bucket_ids_field_t = stk::mesh::Field<unsigned>;
  using linked_entity_bucket_ords_field_t = stk::mesh::Field<unsigned>;
  using link_crs_needs_updated_field_t = stk::mesh::Field<int>;
  using link_marked_for_destruction_field_t = stk::mesh::Field<unsigned>;

  using ngp_linked_entity_bucket_ids_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_linked_entity_bucket_ords_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_link_crs_needs_updated_field_t = stk::mesh::NgpField<int>;
  using ngp_link_marked_for_destruction_field_t = stk::mesh::NgpField<unsigned>;
  //@}

  //! \name Internal getters
  //@{

  /// \brief Fetch the linked entity ids field.
  ngp_linked_entity_ids_field_t &ngp_linked_entity_ids_field() {
    return ngp_linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  ngp_linked_entity_ranks_field_t &ngp_linked_entity_ranks_field() {
    return ngp_linked_entity_ranks_field_;
  }

  /// \brief Fetch the linked entities field
  const ngp_linked_entities_field_t &ngp_linked_entities_field() const {
    return ngp_linked_entities_field_;
  }
  ngp_linked_entities_field_t &ngp_linked_entities_field() {
    return ngp_linked_entities_field_;
  }

  /// \brief Fetch the linked entities field (as last seen by the CRS).
  const ngp_linked_entities_field_t &ngp_linked_entities_crs_field() const {
    return ngp_linked_entities_crs_field_;
  }
  ngp_linked_entities_field_t &ngp_linked_entities_crs_field() {
    return ngp_linked_entities_crs_field_;
  }

  /// \brief Fetch the linked entity bucket id field.
  const ngp_linked_entity_bucket_ids_field_t &ngp_linked_entity_bucket_ids_field() const {
    return ngp_linked_entity_bucket_ids_field_;
  }
  ngp_linked_entity_bucket_ids_field_t &ngp_linked_entity_bucket_ids_field() {
    return ngp_linked_entity_bucket_ids_field_;
  }

  /// \brief Fetch the linked entity bucket ord field.
  const ngp_linked_entity_bucket_ords_field_t &ngp_linked_entity_bucket_ords_field() const {
    return ngp_linked_entity_bucket_ords_field_;
  }
  ngp_linked_entity_bucket_ords_field_t &ngp_linked_entity_bucket_ords_field() {
    return ngp_linked_entity_bucket_ords_field_;
  }

  /// \brief Fetch the link crs needs updated field.
  const ngp_link_crs_needs_updated_field_t &ngp_link_crs_needs_updated_field() const {
    return ngp_link_crs_needs_updated_field_;
  }
  ngp_link_crs_needs_updated_field_t &ngp_link_crs_needs_updated_field() {
    return ngp_link_crs_needs_updated_field_;
  }

  /// \brief Fetch the link marked for destruction field.
  const ngp_link_marked_for_destruction_field_t &ngp_link_marked_for_destruction_field() const {
    return ngp_link_marked_for_destruction_field_;
  }
  ngp_link_marked_for_destruction_field_t &ngp_link_marked_for_destruction_field() {
    return ngp_link_marked_for_destruction_field_;
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  template<typename T>
  friend class NewNgpLinkDataT;

  template <typename T>
  friend class impl::NgpLinkDataCRSManagerT;

  template <typename T>
  friend class NewNgpLinkRequestsT;
  //@}

  //! \name Internal members
  //@{

  NewLinkMetaData *link_meta_data_ptr_;
  stk::mesh::EntityRank link_rank_;
  ngp_linked_entities_field_t ngp_linked_entities_field_;
  ngp_linked_entities_field_t ngp_linked_entities_crs_field_;
  ngp_linked_entity_ids_field_t ngp_linked_entity_ids_field_;
  ngp_linked_entity_ranks_field_t ngp_linked_entity_ranks_field_;
  ngp_linked_entity_bucket_ids_field_t ngp_linked_entity_bucket_ids_field_;
  ngp_linked_entity_bucket_ords_field_t ngp_linked_entity_bucket_ords_field_;
  ngp_link_crs_needs_updated_field_t ngp_link_crs_needs_updated_field_;
  ngp_link_marked_for_destruction_field_t ngp_link_marked_for_destruction_field_;
  stk::mesh::PartOrdinal universal_link_part_ord_;
  //@}
};  // NewNgpLinkMetaData

/// \brief Get an updated ngp link meta data object.
inline NewNgpLinkMetaData get_updated_ngp_link_meta_data(NewLinkMetaData &link_meta_data) {
  return NewNgpLinkMetaData(link_meta_data);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPLINKMETADATA_HPP_
