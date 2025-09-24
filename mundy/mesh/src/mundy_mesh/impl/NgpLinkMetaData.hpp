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

#ifndef MUNDY_MESH_IMPL_NGPLINKMETADATA_HPP_
#define MUNDY_MESH_IMPL_NGPLINKMETADATA_HPP_

/// \file NgpLinkMetaData.hpp
/// \brief Declaration of the NgpLinkMetaData class

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
#include <stk_mesh/base/GetNgpField.hpp>     // for stk::mesh::get_updated_ngp_field
#include <stk_util/ngp/NgpSpaces.hpp>  // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy libs
#include <mundy_mesh/LinkMetaData.hpp>  // for mundy::mesh::LinkMetaData

namespace mundy {

namespace mesh {

namespace impl {

template<typename NgpMemSpace>
class NgpLinkMetaDataT {
 public:
  //! \name Type aliases
  //@{

  using entity_rank_value_t = std::underlying_type_t<stk::mesh::EntityRank>;
  using entity_id_value_t = stk::mesh::EntityId;
  using entity_value_t = stk::mesh::Entity::entity_value_type;

  using linked_entity_ids_field_t = stk::mesh::Field<entity_id_value_t>;
  using linked_entity_ranks_field_t = stk::mesh::Field<entity_rank_value_t>;
  using linked_entities_field_t = stk::mesh::Field<entity_value_t>;
  using linked_entity_bucket_ids_field_t = stk::mesh::Field<unsigned>;
  using linked_entity_bucket_ords_field_t = stk::mesh::Field<unsigned>;
  using link_crs_needs_updated_field_t = stk::mesh::Field<int>;
  using link_marked_for_destruction_field_t = stk::mesh::Field<unsigned>;

  using ngp_linked_entity_ids_field_t = stk::mesh::NgpField<LinkMetaData::entity_id_value_t>;
  using ngp_linked_entity_ranks_field_t = stk::mesh::NgpField<LinkMetaData::entity_rank_value_t>;
  using ngp_linked_entities_field_t = stk::mesh::NgpField<entity_value_t>;
  using ngp_linked_entity_bucket_ids_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_linked_entity_bucket_ords_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_link_crs_needs_updated_field_t = stk::mesh::NgpField<int>;
  using ngp_link_marked_for_destruction_field_t = stk::mesh::NgpField<unsigned>;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Defualt constructor.
  NgpLinkMetaDataT() = default;

  /// \brief Canonical constructor.
  NgpLinkMetaDataT(LinkMetaData &link_meta_data)
      : link_meta_data_ptr_(&link_meta_data),
        link_rank_(link_meta_data.link_rank()),
        ngp_linked_entities_field_(
            stk::mesh::get_updated_ngp_field<linked_entities_field_t::value_type>(
                impl::get_linked_entities_field(link_meta_data))),
        ngp_linked_entities_crs_field_(
            stk::mesh::get_updated_ngp_field<linked_entities_field_t::value_type>(
                impl::get_linked_entities_crs_field(link_meta_data))),
        ngp_linked_entity_ids_field_(
            stk::mesh::get_updated_ngp_field<linked_entity_ids_field_t::value_type>(
                impl::get_linked_entity_ids_field(link_meta_data))),
        ngp_linked_entity_ranks_field_(
            stk::mesh::get_updated_ngp_field<linked_entity_ranks_field_t::value_type>(
                impl::get_linked_entity_ranks_field(link_meta_data))),
        ngp_linked_entity_bucket_ids_field_(
            stk::mesh::get_updated_ngp_field<linked_entity_bucket_ids_field_t::value_type>(
                impl::get_linked_entity_bucket_ids_field(link_meta_data))),
        ngp_linked_entity_bucket_ords_field_(
            stk::mesh::get_updated_ngp_field<linked_entity_bucket_ords_field_t::value_type>(
                impl::get_linked_entity_bucket_ords_field(link_meta_data))),
        ngp_link_crs_needs_updated_field_(
            stk::mesh::get_updated_ngp_field<link_crs_needs_updated_field_t::value_type>(
                impl::get_link_crs_needs_updated_field(link_meta_data))),
        ngp_link_marked_for_destruction_field_(
            stk::mesh::get_updated_ngp_field<link_marked_for_destruction_field_t::value_type>(
                impl::get_link_marked_for_destruction_field(link_meta_data))),
        universal_link_part_ord_(link_meta_data.universal_link_part().mesh_meta_data_ordinal()) {
  }

  /// \brief Destructor.
  virtual ~NgpLinkMetaDataT() = default;

  /// \brief Default copy/move constructors/operators.
  NgpLinkMetaDataT(const NgpLinkMetaDataT &) = default;
  NgpLinkMetaDataT(NgpLinkMetaDataT &&) = default;
  NgpLinkMetaDataT &operator=(const NgpLinkMetaDataT &) = default;
  NgpLinkMetaDataT &operator=(NgpLinkMetaDataT &&) = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the this object is in a valid state.
  bool is_valid() const {
    return link_meta_data_ptr_ != nullptr;
  }

  /// \brief Get the link meta data object (if valid).
  const LinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(is_valid(), std::runtime_error, "Attempted to access an invalid link meta data.");
    return *link_meta_data_ptr_;
  }
  LinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(is_valid(), std::runtime_error, "Attempted to access an invalid link meta data.");
    return *link_meta_data_ptr_;
  }

  /// \brief Get the name of this link data.
  const std::string &name() const {
    return link_meta_data().name();
  }

  /// \brief Fetch the link rank.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank link_rank() const noexcept {
    return link_rank_;
  }

  /// \brief Fetch the universal link part.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::PartOrdinal universal_link_part_ord() const noexcept {
    return universal_link_part_ord_;
  }

  /// \brief Fetch the linked entity ids field.
  ///
  /// \note Users should not edit this field yourself. We expose it to you because it's how you'll interact with the
  /// linked entities when doing things like post-processing the output EXO file, but it should be seen as read-only.
  /// Use declare/delete_relation to modify it since they perform additional behind-the-scenes bookkeeping.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_ids_field_t &ngp_linked_entity_ids_field() const noexcept {
    return ngp_linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  ///
  /// Same comment as linked_entity_ids_field. Treat this field as read-only.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_ranks_field_t &ngp_linked_entity_ranks_field() const noexcept {
    return ngp_linked_entity_ranks_field_;
  }

  /// \brief Fetch the linked entity ids field.
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entity_ids_field_t &ngp_linked_entity_ids_field() noexcept {
    return ngp_linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entity_ranks_field_t &ngp_linked_entity_ranks_field() noexcept {
    return ngp_linked_entity_ranks_field_;
  }

  /// \brief Fetch the linked entities field
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entities_field_t &ngp_linked_entities_field() const noexcept {
    return ngp_linked_entities_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entities_field_t &ngp_linked_entities_field() {
    return ngp_linked_entities_field_;
  }

  /// \brief Fetch the linked entities field (as last seen by the CRS).
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entities_field_t &ngp_linked_entities_crs_field() const noexcept {
    return ngp_linked_entities_crs_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entities_field_t &ngp_linked_entities_crs_field() noexcept {
    return ngp_linked_entities_crs_field_;
  }

  /// \brief Fetch the linked entity bucket id field.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_bucket_ids_field_t &ngp_linked_entity_bucket_ids_field() const noexcept {
    return ngp_linked_entity_bucket_ids_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entity_bucket_ids_field_t &ngp_linked_entity_bucket_ids_field() noexcept {
    return ngp_linked_entity_bucket_ids_field_;
  }

  /// \brief Fetch the linked entity bucket ord field.
  KOKKOS_INLINE_FUNCTION
  const ngp_linked_entity_bucket_ords_field_t &ngp_linked_entity_bucket_ords_field() const noexcept {
    return ngp_linked_entity_bucket_ords_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_linked_entity_bucket_ords_field_t &ngp_linked_entity_bucket_ords_field() noexcept {
    return ngp_linked_entity_bucket_ords_field_;
  }

  /// \brief Fetch the link crs needs updated field.
  KOKKOS_INLINE_FUNCTION
  const ngp_link_crs_needs_updated_field_t &ngp_link_crs_needs_updated_field() const noexcept {
    return ngp_link_crs_needs_updated_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_link_crs_needs_updated_field_t &ngp_link_crs_needs_updated_field() noexcept {
    return ngp_link_crs_needs_updated_field_;
  }

  /// \brief Fetch the link marked for destruction field.
  KOKKOS_INLINE_FUNCTION
  const ngp_link_marked_for_destruction_field_t &ngp_link_marked_for_destruction_field() const noexcept {
    return ngp_link_marked_for_destruction_field_;
  }
  KOKKOS_INLINE_FUNCTION
  ngp_link_marked_for_destruction_field_t &ngp_link_marked_for_destruction_field() noexcept {
    return ngp_link_marked_for_destruction_field_;
  }
  //@}

 private:

  //! \name Internal members
  //@{

  LinkMetaData *link_meta_data_ptr_;
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
};  // NgpLinkMetaDataT

using NgpLinkMetaData = NgpLinkMetaDataT<stk::ngp::MemSpace>;

/// \brief Get an updated ngp link meta data object.
template<typename NgpMemSpace = stk::ngp::MemSpace>
inline NgpLinkMetaDataT<NgpMemSpace> get_updated_ngp_link_meta_data(LinkMetaData &link_meta_data) {
  return NgpLinkMetaDataT<NgpMemSpace>(link_meta_data);
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPLINKMETADATA_HPP_
