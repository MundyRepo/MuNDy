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
#include <mundy_mesh/impl/NgpLinkDataCRSManager.hpp>        // for mundy::mesh::impl::NgpLinkDataCRSManager

namespace mundy {

namespace mesh {

/// \class NewNgpLinkData
/// \brief The main interface for interacting with the link data on the mesh.
///
/// # Links vs. Connectivity
/// Unlike STK's connectivity, links
///   - do not induce changes to part membership, ownership, or sharing
///   - have no restrictions on which entities they can connect (may link same rank entities)
///   - allow data to be stored on the links themselves in a bucket-aware manner
///   - allow either looping over the links or the entities to which they are connected
///   - enforce a weaker aura condition whereby every locally-owned-or-shared link and every locally-owned-or-shared
///   linked entity that it connects have at least ghosted access to one another
///   - allow for the creation and destruction of link to linked entity relations outside of a modification cycle and in
///   parallel
///
/// Links are not meant to be a replacement for connectivity, but rather a more flexible and dynamic alternative. They
/// are great for encoding neighbor relationships that require data to be stored on the link itself, such as reused
/// invariants. They also work well for composite relationships whereby an entity needs to "know about" the state of
/// another entity in a way that cannot be expressed topologically. For example, a quad face storing a set of nodes that
/// are "attached" to it at locations other than the corners.
///
/// From a user's perspective, links are Entities with a connected entity ids and connected entity rank field. You are
/// free to use links like regular entities by adding them to parts, using subset relations/selectors, and by adding
/// additional fields to them. This makes them fully compatible with our Aggregates. Importantly, they may even be
/// written out to the EXO file using standard STK_IO functionality with care to avoid accidentally loadbalancing them
/// if they don't have spacial information.
///
/// # NewNgpLinkData
/// The NewNgpLinkData class is the main interface for interacting with the link data on the mesh. It is meant to mirror
/// BulkData's connectivity interface while allowing multiple NewNgpLinkData objects to be used on the same mesh, each
/// with separately managed data. Use NewNgpLinkData to connect links to linked entities and to get the linked entities
/// for a given link. And similar to STK's for_each_entity_run, use non-member functions acting on the NewNgpLinkData to
/// loop over links and linked entities.
///
/// ## Declaring/Connecting Links
/// Declaring links can be done via the standard declare_entity interface, however, connecting these links to their
/// linked entities must be mediated via the link data. For this reason, we provide only const access to the linked
/// entities field. Use it to ~view~ the connections, not to ~modify~ them. Instead, use the link data's
/// declare_relation(linker, linked_entity, link_ordinal) and delete_relation(linker, link_ordinal) to modify the
/// relationship between a link and its linked entities. These functions are thread-safe and may be called in parallel
/// so long as you do not call declare_relation(linker, *, link_ordinal) for the same linker and ordinal on two
/// different threads (something that would be weird to do anyway).
///
/// Some comments on requirements:
///  - Both declare_relation and delete_relation require that the linker be valid, but not necessarily the linked
///  entity.
///  - To maintain parallel consistency, we require that declare/delete_relation be performed consistently for each
/// process that locally owns or shares the given linker or linked entity.
///
/// \note Once a relationship between a link and a linked entity is declared or destroyed, the link data is marked as
/// "needs updated", as the changes to the linked data are only reflected by the get_linked_entity function and not some
/// of the other infrastructure needed to maintain parallel consistency. As such, you must call
/// update_crs_from_coo() before entering a modification cycle or before using the for_each_linked_entity_run
/// function.
///
/// ## Getting Linked Entities
/// In contrast to STK's connectivity, links are designed to be declared dynamically and to be created and destroyed at
/// any time. This must be done *outside* of a mesh modification cycle. Once a relation between a link and a linked
/// entity is declared, you may call get_linked_entity(linker, link_ordinal) to get the linked entity at the given
/// ordinal. If no relation exists, this will return an invalid entity and is valid so
/// long as link_ordinal is less than the linker dimensionality, otherwise, it will throw an exception (in debug).
/// This function is thread-safe and immediately reflects any changes made to the link data.
///
/// Note, we do not offer a get_linked_entities function, as this would require either dynamic memory allocation or
/// compile-time link dimensionality. Similarly, we do not offer a get_connected_links(entity) function. Instead, we
/// offer two for_each_entity_run-esk functions for looping over links and linked entities.
///
///  - for_each_link_run(link_data, link_subset_selector, functor) works the same as for_each_entity_run, but for links.
///  The functor must either have an
///     operator(const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& linker) or an
//      operator(const NewNgpLinkData& link_data, const stk::mesh::Entity& linker).
///  This function is thread parallel over each link in the given link_data that falls within the given
///  link_subset_selector. Notice that for_each_link does not require this synchronization, meaning that
/// dynamic linker creation and destruction can be done in parallel within a for_each_link loop AND you can continue
/// to perform for_each_link loops while the link data is out-of_sync.
///
/// - for_each_linked_entity_run(link_data, linked_entity_selector, linker_subset_selector, functor)
/// The functor must either have an
///     operator(const stk::mesh::BulkData&, const stk::mesh::Entity&linked_entity, const stk::mesh::Entity&linker) or
///     an operator(const NewNgpLinkData&, const stk::mesh::Entity& linked_entity, const stk::mesh::Entity& linker).
/// This functions is thread parallel over each linked entity in the given linked_entity_selector that falls within a
/// bucket that a linker in the given linker_subset_selector connects to. This means that the functor will be called in
/// serial for each link that connects to a linked entity. Importantly, this function requires infrastructure that
/// requires the link data to be "up-to-date" (i.e., you must call link_data.update_crs_from_coo() before using it
/// if you have modified the link data).
///
/// You might think, why not just provide a get_connected_links(entity) function? The reason is that the links
/// themselves are heterogeneous and bucketized. As such, there is no practical way to provide contiguous access to the
/// set of links that an entity connects to while supporting subset selection of links and without dynamic memory
/// allocation.
///
/// \note To Devs: Hi! Welcome. If you want to better understand the NewNgpLinkData or our links in general, I recommend
/// looking at it as maintaining two connectivity structures: a COO-like structure providing access from a linker to its
/// linked entities and a CRS-like structure providing access from a linked entity to its linkers. The COO is the
/// dynamic "driver" that is trivial to modify (even in parallel) and the CRS is a more heavy-weight sparse data
/// structure that is non-trivial to modify in parallel since it often requires memory allocation. The
/// update_crs_from_coo function is responsible for mapping all of the modifications to the COO structure to the CRS
/// structure. There are some operations that fundamentally require a CRS-like structure such as maintaining parallel
/// consistency as entities are removed from the mesh or performing operations that require a serial loop over each
/// linker that connects to a given linked entity.
///
/// # Delayed link declaration and destruction
/// We offer helper functions for delayed destruction and declaration of links. Users call request_destruction(linker)
/// and request_link(linked_entity0, linked_entity1, ... linked_entityN) to request the destruction of a link and the
/// creation of a link between the given entities, respectively. These requests may be made in parallel and are
/// processed in the next process_requests call. These functions streamline the enforcement of the requirement that
/// "declare/delete_relation are performed consistently for each process that locally owns or shares the given linker or
/// linked entity." We do so at two ~levels ~of user investment, each with different costs.
///
/// ## FULLY_CONSISTENT: You did all the work
/// At a fully consistent level, request_link must be called by each process that locally owns or shares any of the
/// given linked entities and request_destruction must be called by each process that locally owns or shares the given
/// linker. This is the most user-intensive level, but it requires the least amount of MPI communication. At this level,
/// our role is to declare the linker on a single process, ghost the linker to all owners and sharers of the linked
/// entities, and then connect the linker to the linked entities on each process.
///
/// ## PARTIALLY_CONSISTENT: You did some of the work
/// Partial consistency is the default level. It has all of the same requirements as fully consistent, but without
/// considering sharers of either the linker or the linked entities. It is often quite arduous to ensure consistency
/// across all sharers, particularly when attempting to link an entity that is ghosted to the current process. This
/// level is the most user-friendly but does come with a cost. We must perform two pass MPI communication, first
/// broadcasting information to the owners and then to the sharers. Sometimes this is simply unavoidable.
///
/// If using a single process or if only linking element or constraint-rank entities, then partial consistency is the
/// same as full consistency. The level of consistency is passed to the process_requests function, accepting a bool
/// stating if the requests are fully consistent or not. This function will enter a modification cycle only if needed.
template <typename NgpMemSpace>
class NewNgpLinkDataT {
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
  NewNgpLinkDataT(BulkData &bulk_data, NewNgpLinkMetaData ngp_link_meta_data)
      : bulk_data_ptr_(&bulk_data),
        mesh_meta_data_ptr_(&bulk_data.mesh_meta_data()),
        link_meta_data_ptr_(&ngp_link_meta_data.link_meta_data()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(bulk_data)),
        ngp_link_meta_data_(ngp_link_meta_data),
        ngp_crs_manager_(this) {
    MUNDY_THROW_ASSERT(ngp_link_meta_data.is_valid(), std::invalid_argument, "Given link meta data is not valid.");
  }

  /// \brief Destructor.
  virtual ~NewNgpLinkDataT() = default;
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

  /// \brief Fetch the bulk data's meta data manager
  MetaData &mesh_meta_data() {
    MUNDY_THROW_ASSERT(mesh_meta_data_ptr_ != nullptr, std::invalid_argument, "Mesh meta data is not set.");
    return *mesh_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  const NewLinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  NewLinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  const BulkData &bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
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
  inline void declare_relation_host(const stk::mesh::Entity &linker, const stk::mesh::Entity &linked_entity,
                                    unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data().linked_entities_field();
    auto &linked_e_ids_field = link_meta_data().linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data().linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data().linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data().linked_entity_bucket_ords_field();
    auto &link_needs_updated_field = link_meta_data_ptr_->link_crs_needs_updated_field();

    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = linked_entity.local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = bulk_data().identifier(linked_entity);
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] = bulk_data().entity_rank(linked_entity);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] =
        bulk_data().bucket(linked_entity).bucket_id();
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = bulk_data().bucket_ordinal(linked_entity);
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }
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
  inline void delete_relation_host(const stk::mesh::Entity &linker, unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data().linked_entities_field();
    auto &linked_e_ids_field = link_meta_data().linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data().linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data().linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data().linked_entity_bucket_ords_field();
    auto &link_needs_updated_field = link_meta_data().link_crs_needs_updated_field();

    // Intentionally avoids updating the CRS linked entities field so that we can properly detect deletions.
    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = stk::mesh::Entity().local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = stk::mesh::EntityId();
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] =
        static_cast<NewLinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }
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
  inline stk::mesh::Entity get_linked_entity_host(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data().linked_entities_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_linked_entity(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return stk::mesh::Entity(ngp_link_meta_data_.ngp_linked_entities_field()(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity index for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::FastMeshIndex get_linked_entity_index_host(const stk::mesh::Entity &linker,
                                                               unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_bucket_ids_field = link_meta_data().linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data().linked_entity_bucket_ords_field();
    return stk::mesh::FastMeshIndex(stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal],
                                    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal]);
  }
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
  inline stk::mesh::EntityId get_linked_entity_id_host(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ids_field = link_meta_data().linked_entity_ids_field();
    return stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal];
  }
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId get_linked_entity_id(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return ngp_link_meta_data_.ngp_linked_entity_ids_field()(linker_index, link_ordinal);
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::EntityRank get_linked_entity_rank_host(const stk::mesh::Entity &linker,
                                                           unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ranks_field = link_meta_data().linked_entity_ranks_field();
    return static_cast<stk::mesh::EntityRank>(stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal]);
  }
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
    return ngp_crs_manager_.get_all_crs_partitions();
  }

  /// \brief Get the CRS partitions for a given link subset selector (Memoized/host only/not thread safe).
  ///
  /// This is the only way for either mundy or users to create a new partition. The returned view is persistent
  /// but its contents/size will change dynamically as new partitions are created and destroyed. The only promise
  /// we will make is to never delete a partition outside of a modification cycle.
  const NgpCRSPartitionView &get_or_create_crs_partitions(const stk::mesh::Selector &selector) const {
    return ngp_crs_manager_.get_or_create_crs_partitions(selector);
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
    return ngp_crs_manager_.is_crs_up_to_date(selector);
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  bool is_crs_up_to_date() {
    return ngp_crs_manager_.is_crs_up_to_date();
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  void update_crs_from_coo(const stk::mesh::Selector &selector) {
    ngp_crs_manager_.update_crs_from_coo(selector);
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  void update_crs_from_coo() {
    ngp_crs_manager_.update_crs_from_coo();
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  void check_crs_coo_consistency(const stk::mesh::Selector &selector) {
    ngp_crs_manager_.check_crs_coo_consistency(selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  void check_crs_coo_consistency() {
    ngp_crs_manager_.check_crs_coo_consistency();
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

  void modify_on_host() {
    ngp_link_meta_data_.ngp_linked_entities_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().modify_on_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().modify_on_host();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().modify_on_host();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().modify_on_host();
  }

  void modify_on_device() {
    ngp_link_meta_data_.ngp_linked_entities_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().modify_on_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().modify_on_device();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().modify_on_device();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().modify_on_device();
  }

  void sync_to_host() {
    ngp_link_meta_data_.ngp_linked_entities_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().sync_to_host();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().sync_to_host();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().sync_to_host();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().sync_to_host();
  }

  void sync_to_device() {
    ngp_link_meta_data_.ngp_linked_entities_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entities_crs_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_ids_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_ranks_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ids_field().sync_to_device();
    ngp_link_meta_data_.ngp_linked_entity_bucket_ords_field().sync_to_device();
    ngp_link_meta_data_.ngp_link_crs_needs_updated_field().sync_to_device();
    ngp_link_meta_data_.ngp_link_marked_for_destruction_field().sync_to_device();
    std::cout << "End of sync_to_device" << std::endl;
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

  template <typename T>
  friend class impl::NgpLinkDataCRSManagerT;
  //@}

  //! \name Internal actions
  //@{

  /// \brief Get the linked entity for a given linker and link ordinal (as last seen by the CRS connectivity).
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::Entity get_linked_entity_crs_host(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_crs_field = link_meta_data().linked_entities_crs_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_crs_field, linker)[link_ordinal]);
  }
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_linked_entity_crs(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return stk::mesh::Entity(ngp_link_meta_data_.ngp_linked_entities_crs_field()(linker_index, link_ordinal));
  }

  /// \brief Get the dimensionality of a linker
  /// \param linker [in] The linker (must be valid and of the correct rank).
  inline unsigned get_linker_dimensionality_host(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    return stk::mesh::field_scalars_per_entity(link_meta_data().linked_entities_field(), linker);
  }

  /// \brief Get the dimensionality of a linker bucket
  inline unsigned get_linker_dimensionality_host(const stk::mesh::Bucket &linker_bucket) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == linker_bucket.entity_rank(), std::invalid_argument,
                       "Linker bucket is not of the correct rank.");
    MUNDY_THROW_ASSERT(linker_bucket.member(link_meta_data().universal_link_part()), std::invalid_argument,
                       "Linker bucket is not a subset of our universal link part.");

    auto &linked_es_field = link_meta_data().linked_entities_field();
    return stk::mesh::field_scalars_per_entity(linked_es_field, linker_bucket);
  }

  /// \brief Get the dimensionality for a collection of linker parts
  inline unsigned get_linker_dimensionality_host(const stk::mesh::PartVector &parts) const {
    // The restriction may be empty if the parts are not a subset of the universal link part.
    auto &linked_es_field = link_meta_data().linked_entities_field();
    const stk::mesh::FieldRestriction &restriction =
        stk::mesh::find_restriction(linked_es_field, link_meta_data().link_rank(), parts);
    return restriction.num_scalars_per_entity();
  }

  /// \brief Get the dimensionality of a linker partition
  inline unsigned get_linker_dimensionality_host(const PartitionKey &partition_key) const {
    MUNDY_THROW_REQUIRE(partition_key.size() > 0, std::invalid_argument, "Partition key is empty.");

    // Fetch the parts
    stk::mesh::PartVector parts(partition_key.size());
    for (size_t i = 0; i < partition_key.size(); ++i) {
      parts[i] = &mesh_meta_data().get_part(partition_key[i]);
    }

    return get_linker_dimensionality_host(parts);
  }

  /// \brief Get if the CRS connectivity for a link needs to be updated.
  inline bool get_link_crs_needs_updated_host(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_needs_updated_field = link_meta_data().link_crs_needs_updated_field();
    return static_cast<bool>(stk::mesh::field_data(link_needs_updated_field, linker)[0]);
  }
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
  impl::NgpLinkDataCRSManagerT<NgpMemSpace> ngp_crs_manager_;
  //@}
};  // NewNgpLinkDataT

using NewNgpLinkData = NewNgpLinkDataT<stk::ngp::MemSpace>;

/// \brief Declare a new NewNgpLinkData object.
///
/// Note, this is the de-facto constructor for NewNgpLinkData. In the future, we will have it return a reference to
/// the constructed NewNgpLinkData object by tying its lifetime to the BulkData object.
///
/// \param bulk_data [in] The bulk data manager we extend.
/// \param link_meta_data [in] Our meta data manager.
/// \return A new NewNgpLinkData object.
NewNgpLinkData declare_ngp_link_data(BulkData &bulk_data, NewLinkMetaData &link_meta_data) {
  // TODO(palmerb4): Store a vector of NewNgpLinkData objects in the BulkData object using an attribute
  //  this way, we can tie the lifetime of the NewNgpLinkData object to the BulkData object and return
  //  a reference here.
  return NewNgpLinkData(bulk_data, link_meta_data);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_NGPLINKDATA_HPP_
