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

#ifndef MUNDY_MESH_LINKMETADATA_HPP_
#define MUNDY_MESH_LINKMETADATA_HPP_

/// \file NewLinkMetaData.hpp
/// \brief Declaration of the NewLinkMetaData class

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
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>  // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

class NewLinkMetaData {
 public:
  //! \name Type aliases
  //@{

  using entity_rank_value_t = std::underlying_type_t<stk::mesh::EntityRank>;
  using entity_id_value_t = stk::mesh::EntityId;
  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using linked_entity_ids_field_t = stk::mesh::Field<entity_id_value_t>;
  using linked_entity_ranks_field_t = stk::mesh::Field<entity_rank_value_t>;
  using linked_entities_field_t = stk::mesh::Field<entity_value_t>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Destructor.
  virtual ~NewLinkMetaData() = default;

  /// \brief Default copy/move constructors/operators.
  NewLinkMetaData(const NewLinkMetaData &) = default;
  NewLinkMetaData(NewLinkMetaData &&) = default;
  NewLinkMetaData &operator=(const NewLinkMetaData &) = default;
  NewLinkMetaData &operator=(NewLinkMetaData &&) = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the name of this link data.
  const std::string &name() const {
    return our_name_;
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
  const linked_entity_ids_field_t &linked_entity_ids_field() const {
    return linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  ///
  /// Same comment as linked_entity_ids_field. Treat this field as read-only.
  const linked_entity_ranks_field_t &linked_entity_ranks_field() const {
    return linked_entity_ranks_field_;
  }

  /// \brief Fetch the universal link part.
  const stk::mesh::Part &universal_link_part() const {
    return universal_link_part_;
  }

  /// \brief Fetch the universal link part.
  stk::mesh::Part &universal_link_part() {
    return universal_link_part_;
  }

  /// \brief Fetch the mesh meta data manager for this bulk data manager.
  const MetaData &mesh_meta_data() const {
    return meta_data_;
  }

  /// \brief Fetch the mesh meta data manager for this bulk data manager.
  MetaData &mesh_meta_data() {
    return meta_data_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare a link-part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &declare_link_part(const std::string &part_name, unsigned link_dimensionality_for_this_part) {
    stk::mesh::Part &part = meta_data_.declare_part(part_name, link_rank_);
    add_link_support_to_part(part, link_dimensionality_for_this_part);
    return part;
  }

  /// \brief Declare a link assembly part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  stk::mesh::Part &declare_link_assembly_part(const std::string &part_name) {
    stk::mesh::Part &part = meta_data_.declare_part(part_name, link_rank_);
    add_link_support_to_assembly_part(part);
    return part;
  }

  /// \brief Make an existing part into a link-compatible part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &add_link_support_to_part(stk::mesh::Part &part, unsigned link_dimensionality_for_this_part) {
    meta_data_.declare_part_subset(universal_link_part_, part);
    put_link_fields_on_part(part, link_dimensionality_for_this_part);
    return part;
  }

  /// \brief Make an existing assembly part into a link-compatible part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &add_link_support_to_assembly_part(stk::mesh::Part &part) {
    meta_data_.declare_part_subset(universal_link_part_, part);
    return part;
  }
  //@}

 protected:
  //! \name Internal aliases
  //@{

  using linked_entity_bucket_ids_field_t = stk::mesh::Field<unsigned>;
  using linked_entity_bucket_ords_field_t = stk::mesh::Field<unsigned>;
  using link_crs_needs_updated_field_t = stk::mesh::Field<int>;
  using link_marked_for_destruction_field_t = stk::mesh::Field<unsigned>;
  //@}

  //! \name Constructor
  //@{

  /// \brief No default constructor.
  NewLinkMetaData() = delete;

  /// \brief Construct and declare
  NewLinkMetaData(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank)
      : our_name_(our_name),
        meta_data_(meta_data),
        link_rank_(link_rank),
        linked_entities_field_(meta_data.declare_field<entity_value_t>(link_rank_, "MUNDY_LINKED_ENTITIES")),
        linked_entities_crs_field_(meta_data.declare_field<entity_value_t>(link_rank_, "MUNDY_LINKED_ENTITIES_CRS")),
        linked_entity_ids_field_(meta_data.declare_field<entity_id_value_t>(link_rank_, "MUNDY_LINKED_ENTITY_IDS")),
        linked_entity_ranks_field_(
            meta_data.declare_field<entity_rank_value_t>(link_rank_, "MUNDY_LINKED_ENTITY_RANKS")),
        linked_entity_bucket_ids_field_(meta_data.declare_field<unsigned>(link_rank_, "MUNDY_LINKED_ENTITY_BUCKET_ID")),
        linked_entity_bucket_ords_field_(
            meta_data.declare_field<unsigned>(link_rank_, "MUNDY_LINKED_ENTITY_BUCKET_ORD")),
        link_crs_needs_updated_field_(meta_data.declare_field<int>(link_rank_, "MUNDY_LINK_CRS_NEEDS_UPDATED")),
        link_marked_for_destruction_field_(
            meta_data.declare_field<unsigned>(link_rank_, "MUNDY_LINK_MARKED_FOR_DESTRUCTION")),
        universal_link_part_(meta_data.declare_part(std::string("MUNDY_UNIVERSAL_") + our_name, link_rank_)) {
    unsigned links_start_valid[1] = {0};
    stk::mesh::put_field_on_mesh(link_marked_for_destruction_field_, meta_data.universal_part(), 1, links_start_valid);
  }
  //@}

  //! \name Internal getters
  //@{

  /// \brief Fetch the linked entity ids field.
  linked_entity_ids_field_t &linked_entity_ids_field() {
    return linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  linked_entity_ranks_field_t &linked_entity_ranks_field() {
    return linked_entity_ranks_field_;
  }

  /// \brief Fetch the linked entities field
  const linked_entities_field_t &linked_entities_field() const {
    return linked_entities_field_;
  }
  linked_entities_field_t &linked_entities_field() {
    return linked_entities_field_;
  }

  /// \brief Fetch the linked entities field (as last seen by the CRS).
  const linked_entities_field_t &linked_entities_crs_field() const {
    return linked_entities_crs_field_;
  }
  linked_entities_field_t &linked_entities_crs_field() {
    return linked_entities_crs_field_;
  }

  /// \brief Fetch the linked entity bucket id field.
  const linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field() const {
    return linked_entity_bucket_ids_field_;
  }
  linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field() {
    return linked_entity_bucket_ids_field_;
  }

  /// \brief Fetch the linked entity bucket ord field.
  const linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field() const {
    return linked_entity_bucket_ords_field_;
  }
  linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field() {
    return linked_entity_bucket_ords_field_;
  }

  /// \brief Fetch the link crs needs updated field.
  const link_crs_needs_updated_field_t &link_crs_needs_updated_field() const {
    return link_crs_needs_updated_field_;
  }
  link_crs_needs_updated_field_t &link_crs_needs_updated_field() {
    return link_crs_needs_updated_field_;
  }

  /// \brief Fetch the link marked for destruction field.
  const link_marked_for_destruction_field_t &link_marked_for_destruction_field() const {
    return link_marked_for_destruction_field_;
  }
  link_marked_for_destruction_field_t &link_marked_for_destruction_field() {
    return link_marked_for_destruction_field_;
  }
  //@}

  //! \name Helper functions
  //@{

  /// \brief Add the linked entities and keys field to the part with the given dimensionality
  void put_link_fields_on_part(stk::mesh::Part &part, unsigned link_dimensionality) {
    std::vector<entity_value_t> initial_linked_entities(link_dimensionality, stk::mesh::Entity().local_offset());
    std::vector<entity_id_value_t> initial_linked_entity_ids(link_dimensionality, stk::mesh::EntityId());
    std::vector<entity_rank_value_t> initial_linked_entity_ranks(
        link_dimensionality, static_cast<entity_rank_value_t>(stk::topology::INVALID_RANK));
    int initial_link_crs_needs_updated[1] = {true};
    stk::mesh::put_field_on_mesh(linked_entities_field_, part, link_dimensionality, initial_linked_entities.data());
    stk::mesh::put_field_on_mesh(linked_entities_crs_field_, part, link_dimensionality, initial_linked_entities.data());
    stk::mesh::put_field_on_mesh(linked_entity_ids_field_, part, link_dimensionality, initial_linked_entity_ids.data());
    stk::mesh::put_field_on_mesh(linked_entity_ranks_field_, part, link_dimensionality,
                                 initial_linked_entity_ranks.data());
    stk::mesh::put_field_on_mesh(linked_entity_bucket_ids_field_, part, link_dimensionality, nullptr);
    stk::mesh::put_field_on_mesh(linked_entity_bucket_ords_field_, part, link_dimensionality, nullptr);
    stk::mesh::put_field_on_mesh(link_crs_needs_updated_field_, part, 1, initial_link_crs_needs_updated);
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  friend class NewNgpLinkMetaData;
  friend class NewLinkData;
  friend class NewLinkPartition;
  friend class NewNgpLinkData;
  friend class NewNgpLinkPartition;
  template <typename T>
  friend class NewNgpLinkRequestsT;
  friend NewLinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name,
                                             stk::mesh::EntityRank link_rank);
  friend std::shared_ptr<NewLinkMetaData> declare_link_meta_data_ptr(MetaData &meta_data, const std::string &our_name,
                                             stk::mesh::EntityRank link_rank);
  //@}

  //! \name Internal members
  //@{

  std::string our_name_;
  MetaData &meta_data_;
  stk::mesh::EntityRank link_rank_;
  linked_entities_field_t &linked_entities_field_;
  linked_entities_field_t &linked_entities_crs_field_;
  linked_entity_ids_field_t &linked_entity_ids_field_;
  linked_entity_ranks_field_t &linked_entity_ranks_field_;
  linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field_;
  linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field_;
  link_crs_needs_updated_field_t &link_crs_needs_updated_field_;
  link_marked_for_destruction_field_t &link_marked_for_destruction_field_;
  stk::mesh::Part &universal_link_part_;
  //@}
};

/// \brief Construct a new NewLinkMetaData object.
/// \param meta_data [in] The mesh meta data manager. (pre-commit)
/// \param our_name [in] The name of this link data.
/// \param link_rank [in] The rank of the link entities.
NewLinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank) {
  // TODO(palmerb4): Tie the lifetime of this object to the BulkData object so we can return a reference to it.
  return NewLinkMetaData(meta_data, our_name, link_rank);
}

std::shared_ptr<NewLinkMetaData> declare_link_meta_data_ptr(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank) {
  // TODO(palmerb4): Tie the lifetime of this object to the BulkData object so we can return a reference to it.
  return std::shared_ptr<NewLinkMetaData>(new NewLinkMetaData(meta_data, our_name, link_rank));
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKMETADATA_HPP_
