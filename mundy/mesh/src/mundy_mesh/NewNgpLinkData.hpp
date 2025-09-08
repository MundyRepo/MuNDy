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
/// update_crs_connectivity() before entering a modification cycle or before using the for_each_linked_entity_run
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
/// requires the link data to be "up-to-date" (i.e., you must call link_data.update_crs_connectivity() before using it
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
/// update_crs_connectivity function is responsible for mapping all of the modifications to the COO structure to the CRS
/// structure. There are some operations that fundamentally require a CRS-like structure such as maintaining parallel
/// consistency as entities are removed from the mesh or change parallel ownership/sharing or performing operations that
/// require a serial loop over each linker that connects to a given linked entity.
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
///
/// This class will need templated by Ngp memory space come Trilinos 16.2.
class NewNgpLinkData {
 public:
  //! \name Aliases
  //@{

  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  using NgpCRSPartitionView = Kokkos::View<NewNgpCRSPartition *, Kokkos::DefaultExecutionSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  NewNgpLinkData() = default;

  /// \brief Default copy or move constructors/operators.
  NewNgpLinkData(const NewNgpLinkData &) = default;
  NewNgpLinkData(NewNgpLinkData &&) = default;
  NewNgpLinkData &operator=(const NewNgpLinkData &) = default;
  NewNgpLinkData &operator=(NewNgpLinkData &&) = default;

  /// \brief Canonical constructor.
  /// \param bulk_data [in] The bulk data manager we extend.
  /// \param link_meta_data [in] Our meta data manager.
  NewNgpLinkData(BulkData &bulk_data, NewNgpLinkMetaData ngp_link_meta_data)
      : bulk_data_ptr_(&bulk_data),
        mesh_meta_data_ptr_(&bulk_data.mesh_meta_data()),
        link_meta_data_ptr_(&ngp_link_meta_data.link_meta_data()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(bulk_data)),
        ngp_link_meta_data_(ngp_link_meta_data),
        all_crs_partitions_("AllCRSPartitions", 0),
        stk_link_bucket_to_partition_id_map_(10) {
    MUNDY_THROW_ASSERT(ngp_link_meta_data.is_valid(), std::invalid_argument, "Given link meta data is not valid.");
  }

  /// \brief Destructor.
  virtual ~NewNgpLinkData() = default;
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
  stk::mesh::NgpMesh &ngp_mesh()  noexcept{
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
  /// Safety: This view is always safe to use and the reference is valid as long as the NewNgpLinkData is valid, which
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
  const NgpCRSPartitionView &get_or_create_crs_partitions(const stk::mesh::Selector &selector) const {
    MUNDY_THROW_ASSERT(is_valid(), std::invalid_argument, "Link data is not valid.");

    // We only care about the intersection of the given selector and our universe link selector.
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Memoized return
    SelectorToPartitionsMap::iterator it = selector_to_partitions_map_.find(link_subset_selector);
    if (it != selector_to_partitions_map_.end()) {
      // Return the existing view
      return it->second;
    } else {
      // Create a new view
      // 1. Map selector to buckets
      const stk::mesh::BucketVector &selected_buckets =
          bulk_data().get_buckets(link_meta_data().link_rank(), link_subset_selector);

      // 2. Sort and unique the keys for each buckets
      std::set<PartitionKey> new_keys;
      std::set<PartitionKey> old_keys;
      for (const stk::mesh::Bucket *bucket : selected_buckets) {
        PartitionKey key = get_partition_key(*bucket);
        if (partition_key_to_id_map_.find(key) == partition_key_to_id_map_.end()) {
          new_keys.insert(key);
        } else {
          old_keys.insert(key);
        }
      }

      size_t num_previous_partitions = all_crs_partitions_.extent(0);
      size_t num_new_partitions = new_keys.size();
      size_t num_old_partitions = old_keys.size();
      if (num_new_partitions > 0) {
        // 3. Grow the size of the partition view by the number of new unique keys
        Kokkos::resize(Kokkos::WithoutInitializing, all_crs_partitions_, num_previous_partitions + num_new_partitions);

        // 4. Create a new NewNgpCRSPartition (for each unique new key) and store it within the all_crs_partitions_ view
        stk::mesh::Ordinal partition_id = static_cast<stk::mesh::Ordinal>(num_previous_partitions);
        for (const PartitionKey &key : new_keys) {
          new (&all_crs_partitions_(partition_id))
              NewNgpCRSPartition(partition_id, key, link_rank(), get_linker_dimensionality_host(key), bulk_data());

          partition_key_to_id_map_[key] = partition_id;
          ++partition_id;
        }
      }

      // 5. Create a new view of CRS partitions of size equal to the number of unique keys (both existing and new)
      NgpCRSPartitionView new_crs_partitions("NewCRSPartitions", num_new_partitions + num_old_partitions);

      // 6. Copy the corresponding NewNgpCRSPartition from the all_crs_partitions_ view to the new view using the key to
      // partition_id map
      unsigned count = 0;
      for (const PartitionKey &key : old_keys) {
        auto it = partition_key_to_id_map_.find(key);
        if (it != partition_key_to_id_map_.end()) {
          stk::mesh::Ordinal partition_id = it->second;
          new_crs_partitions(count++) = all_crs_partitions_(partition_id);
        } else {
          MUNDY_THROW_ASSERT(false, std::logic_error,
                             "Partition key not found in partition key to id map. This should never happen.");
        }
      }
      for (const PartitionKey &key : new_keys) {
        auto it = partition_key_to_id_map_.find(key);
        if (it != partition_key_to_id_map_.end()) {
          stk::mesh::Ordinal partition_id = it->second;
          new_crs_partitions(count++) = all_crs_partitions_(partition_id);
        } else {
          MUNDY_THROW_ASSERT(false, std::logic_error,
                             "Partition key not found in partition key to id map. This should never happen.");
        }
      }

      // 7. Store the new view in the selector to partitions map
      selector_to_partitions_map_[link_subset_selector] = new_crs_partitions;

      // 8. Return a reference to the new view
      return selector_to_partitions_map_[link_subset_selector];
    }
  }

  /// \brief Get all links in the given partition that connect to the given entity in the CRS connectivity.
  KOKKOS_FUNCTION
  ConnectedEntities get_connected_links(const stk::mesh::Ordinal partition_id, stk::mesh::EntityRank rank,
                                        const stk::mesh::FastMeshIndex &entity_index) const {
    return get_all_crs_partitions()[partition_id].get_connected_links(rank, entity_index);
  }

  /// \brief Get the number of links in the given partition that connect to the given entity in the CRS connectivity.
  KOKKOS_FUNCTION
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
  bool is_crs_connectivity_up_to_date(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NewNgpLinkData::is_crs_connectivity_up_to_date");

    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Two types of out-of-date:
    //  1. The CRS connectivity of a selected partition is dirty.
    //    - Team loop over each selected partition and thread loop over each bucket in the partition. If any bucket is
    //    dirty, atomically set the needs updated flag to true.
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);
    unsigned num_partitions = partitions.extent(0);
    bool crs_buckets_up_to_date = true;
    for (unsigned i = 0; i < num_partitions; ++i) {
      const NewNgpCRSPartition &partition = partitions(i);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        const unsigned num_buckets = partition.num_buckets(rank);
        for (unsigned bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
          const auto &crs_bucket_conn = partition.get_crs_bucket_conn(rank, bucket_index);
          if (crs_bucket_conn.dirty_) {
            crs_buckets_up_to_date = false;
            goto done_checking_crs_buckets;
          }
        }
      }
    }  
    done_checking_crs_buckets:

    // TODO(palmerb4): It appears as though counting the number of dirty buckets in a parallel_for is slower than doing it
    // serially (at least for a CPU build). Is this true for GPU builds too?

    // int num_dirty_buckets = 0;
    // typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    // const auto &team_policy =
    //     stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(partitions.extent(0), Kokkos::AUTO);
    // Kokkos::parallel_reduce(
    //     "NewNgpLinkData::is_crs_connectivity_up_to_date", team_policy,
    //     KOKKOS_LAMBDA(const TeamHandleType &team, int &team_local_count) {
    //       const stk::mesh::Ordinal partition_id = team.league_rank();
    //       const NewNgpCRSPartition &partition = partitions(partition_id);

    //       int tmp_team_local_count = 0;

    //       for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    //         const unsigned num_buckets = partition.num_buckets(rank);
    //         int rank_local_count = 0;
    //         Kokkos::parallel_reduce(
    //             Kokkos::TeamThreadRange(team, num_buckets),
    //             KOKKOS_LAMBDA(const unsigned bucket_index, int &count) {
    //               const auto &crs_bucket_conn = partition.get_crs_bucket_conn(rank, bucket_index);
    //               count += crs_bucket_conn.dirty_;
    //             },
    //             Kokkos::Sum<int>(rank_local_count));
    //         tmp_team_local_count += rank_local_count;
    //       }

    //       team_local_count += tmp_team_local_count;
    //     },
    //     Kokkos::Sum<int>(num_dirty_buckets));
    // bool crs_buckets_up_to_date = num_dirty_buckets == 0;
    // std::cout << "num_dirty_buckets: " << num_dirty_buckets << std::endl;

    if (crs_buckets_up_to_date) {  // No need to perform the second check if the first fails.
      //  2. A selected link is out-of-date.
      int link_needs_updated_count = ::mundy::mesh::field_sum<int>(link_meta_data().link_crs_needs_updated_field(),
                                                                   link_subset_selector, stk::ngp::ExecSpace());
      bool links_up_to_date = (link_needs_updated_count == 0);
      return links_up_to_date;
    }

    Kokkos::Profiling::popRegion();
    return crs_buckets_up_to_date;
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  bool is_crs_connectivity_up_to_date() {
    return is_crs_connectivity_up_to_date(bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  void update_crs_connectivity(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    if (is_crs_connectivity_up_to_date(link_subset_selector)) {
      return;
    }

    MUNDY_THROW_ASSERT(is_valid(), std::invalid_argument, "Link data is not valid.");

    Kokkos::Profiling::pushRegion("NewNgpLinkData::update_crs_connectivity");

    // There are a couple options here for the types of state that we need to address:
    //  1. Nothing happened: All STK buckets are up-to-date (technically we only care about link buckets and the buckets
    //  of entities they link but we'll ignore that for now), all selected links are up-to-date, and all CRS buckets of
    //  selected partitions are up-to-date.
    //  2. A link was added or removed: Some STK buckets are out-of-date but all selected links are up-to-date.
    //  3. A link relation was added or removed: All STK buckets are up-to-date but some selected links are out-of-date.
    //  4. A combination of 2 and 3: Some STK buckets are out-of-date and some selected links are out-of-date.
    //
    // If some links are out-of-date, we need to mark the CRS connectivity of each downward linked entity as dirty. Upon
    // update, all dirty buckets will be updated.
    //
    // If link is created (or becomes a member of a link part), this has no effect on the CRS connectivity until it is
    // connected to linked entities, at which point, the regular update procedure would handle propagating changes.
    //
    // If a link is deleted (or loses its link part membership), then we need to update the CRS connectivity of the
    // downward linked CRS entities (regardless of the currently linked entities).
    //
    // If a the ownership of a link changes, the process losing ownership must mark the buckets of downward linked
    // entities are dirty and the receiving process must mark the link as needing an update to its CRS connectivity.
    //
    // If a link ever enters a state where it is linked to non-empty entities and none of those entities are owned by
    // its owning process, the link will transfer ownership to the process that owns the first non-empty linked entity.
    //
    // An observer will detect deletions, loss of link part membership, and changes of ownership. It will properly flag
    // the buckets of the linked entities as dirty or the link itself as being out-of-date, which will then be processed
    // in the next update.
    //
    // This function is independent of said observer and is only responsible for updating the CRS connectivity given
    // that some links are out-of-date or some crs buckets are dirty. As such, it simply loops over the links in the
    // given selector, flags the buckets of linked entities as dirty if the link is out-of-date, and then updates all
    // dirty crs buckets.
    //
    // stk_link_bucket_to_partition_id_map_ is a weird animal in that it must have the same size as the number of
    // buckets but the number of buckets may change during a modification cycle. We need to be certain if buckets may
    // even change their IDs during a modification cycle or not. If not, then we need to delete and shift this map each
    // time a bucket is destroyed.
    //
    // Each link bucket needs to be able to access its link partition. Buckets may change parts, but the observer isn't
    // informed of this change. Every tome that STK send a signal for local_buckets_changed_notification(link_rank), we
    // need to rebuilt this map by looping over all selected link buckets, fetching their partition key, and using
    // partition_key_to_id_map_ to get the corresponding id. We'll then store this in the bucket to id map.
    //
    // This tells us that the LinkDataObserver is in charge of deciding when to rebuild this map but not when to build
    // it in the first place. We could use a memoized getter that sees if the list is empty or not. If it's empty, it
    // calls rebuild_stk_link_bucket_to_partition_id_map.
    flag_dirty_linked_buckets_of_modified_links(link_subset_selector);

    reset_dirty_linked_buckets(link_subset_selector);

    gather_part_1_count(link_subset_selector);

    gather_part_2_partial_sum(link_subset_selector);

    scatter_part_1_setup(link_subset_selector);

    scatter_part_2_fill(link_subset_selector);

    finalize_crs_update(link_subset_selector);

    Kokkos::Profiling::popRegion();

// If in debug, check consistency
#ifndef NDEBUG
    check_crs_coo_consistency(link_subset_selector);
#endif
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  void update_crs_connectivity() {
    update_crs_connectivity(bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  void check_crs_coo_consistency(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;
    check_all_links_in_sync(link_subset_selector);
    check_linked_bucket_conn_size(link_subset_selector);
    check_coo_to_crs_conn(link_subset_selector);
    check_crs_to_coo_conn(link_subset_selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  void check_crs_coo_consistency() {
    check_crs_coo_consistency(bulk_data().mesh_meta_data().universal_part());
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
  using LinkBucketToPartitionIdMap = Kokkos::UnorderedMap<unsigned, unsigned, stk::ngp::MemSpace>;
  //@}

  //! \name CRS connectivity implementation details
  //@{

  /// \brief Get the partition key for a given set of link parts (independent of their order, host only)
  PartitionKey get_partition_key(const stk::mesh::PartVector &link_parts) const {
    stk::mesh::OrdinalVector link_parts_and_supersets;
    stk::mesh::impl::fill_add_parts_and_supersets(link_parts, link_parts_and_supersets);
    return link_parts_and_supersets;
  }

  /// \brief Get the partition key for a given link bucket (host only)
  PartitionKey get_partition_key(const stk::mesh::Bucket &link_bucket) const {
    return get_partition_key(link_bucket.supersets());
  }

  /// \brief Get the map from link bucket id to partition id (memoized/host only/not thread safe).
  LinkBucketToPartitionIdMap get_updated_stk_link_bucket_to_partition_id_map() {
    // If the map is empty, populate it.
    if (stk_link_bucket_to_partition_id_map_.size() == 0) {
      update_stk_link_bucket_to_partition_id_map();
    }

    return stk_link_bucket_to_partition_id_map_;
  }

  /// \brief Update the map from link bucket id to partition id (host only/not thread safe).
  void update_stk_link_bucket_to_partition_id_map() {
    // Get all link buckets that currently have selectors.
    stk::mesh::Selector all_selector;
    for (const auto &pair : selector_to_partitions_map_) {
      all_selector |= pair.first;
    }
    const stk::mesh::BucketVector &all_link_buckets =
        bulk_data().get_buckets(link_meta_data().link_rank(), all_selector);
    const unsigned num_link_buckets = static_cast<unsigned>(all_link_buckets.size());

    // Resize the map if needed.
    if (stk_link_bucket_to_partition_id_map_.capacity() < num_link_buckets) {
      stk_link_bucket_to_partition_id_map_.rehash(num_link_buckets);
    }

    // Loop over each bucket, get its partition key, map the key to an id, and store the id in the map.
    for (const stk::mesh::Bucket *bucket : all_link_buckets) {
      PartitionKey key = get_partition_key(*bucket);

      auto it = partition_key_to_id_map_.find(key);
      if (it != partition_key_to_id_map_.end()) {
        stk::mesh::Ordinal partition_id = it->second;
        bool insert_success = stk_link_bucket_to_partition_id_map_.insert(bucket->bucket_id(), partition_id).success();
        MUNDY_THROW_ASSERT(insert_success, std::runtime_error,
                           "Failed to insert bucket -> partition pair into the map. This is an internal error.");
      } else {
        MUNDY_THROW_ASSERT(false, std::logic_error,
                           "Partition key not found in partition key to id map. This should never happen.");
      }
    }
  }

  void flag_dirty_linked_buckets_of_modified_links(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::flag_dirty_linked_buckets");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Flag dirty buckets: Team loop over selected link buckets, fetch their partition, thread loop over links,
    // determine if any of those links are flagged as modified. If so, determine if their links were created or
    // destroyed. Flag the linked bucket of new or deleted entities as dirty.

    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    const stk::mesh::EntityRank link_rank = link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");
          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);

          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            if (get_link_crs_needs_updated(link_index)) {
              // Loop over the linked entities of this link
              for (unsigned d = 0; d < dimensionality; ++d) {
                stk::mesh::Entity linked_entity_crs = get_linked_entity_crs(link_index, d);
                stk::mesh::Entity linked_entity = get_linked_entity(link_index, d);
                bool things_changed = linked_entity_crs != linked_entity;
                if (things_changed) {
                  bool old_entity_is_valid = (linked_entity_crs != stk::mesh::Entity());
                  if (old_entity_is_valid) {
                    // Mark the old linked entity's crs bucket conn as dirty
                    const stk::mesh::FastMeshIndex linked_entity_crs_index =
                        ngp_mesh.fast_mesh_index(linked_entity_crs);
                    const stk::mesh::EntityRank linked_entity_crs_rank = ngp_mesh.entity_rank(linked_entity_crs);
                    auto &crs_bucket_conn =
                        crs_partition.get_crs_bucket_conn(linked_entity_crs_rank, linked_entity_crs_index.bucket_id);
                    Kokkos::atomic_store(&crs_bucket_conn.dirty_,
                                         true);  // TODO: This should be a protected function (flag_as_dirty_atomically)
                  }

                  bool new_entity_is_valid = (linked_entity != stk::mesh::Entity());
                  if (new_entity_is_valid) {
                    // Mark the new linked entity's crs bucket conn as dirty
                    const stk::mesh::FastMeshIndex new_linked_entity_index = ngp_mesh.fast_mesh_index(linked_entity);
                    const stk::mesh::EntityRank linked_entity_rank = ngp_mesh.entity_rank(linked_entity);
                    auto &crs_bucket_conn =
                        crs_partition.get_crs_bucket_conn(linked_entity_rank, new_linked_entity_index.bucket_id);
                    Kokkos::atomic_store(&crs_bucket_conn.dirty_,
                                         true);  // TODO: This should be a protected function (flag_as_dirty_atomically)
                  }
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  void reset_dirty_linked_buckets(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::reset_dirty_linked_buckets");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    //  Reset dirty buckets: Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over
    //  the partitions, if its corresponding linked bucket has been modified, thread loop over the linked entities and
    //  reset the connectivity counts.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, team.league_rank());
            unsigned bucket_size = bucket.size();

            // Serial loop over the partitions
            for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
              NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);

              // Fetch the crs bucket conn for this rank and bucket
              auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

              // If the bucket is dirty, reset the connectivity counts
              if (crs_bucket_conn.dirty_) {
                // Reset the connectivity counts for each entity in the bucket
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                     [&](const int &i) { crs_bucket_conn.num_connected_links_(i) = 0; });
              }
            }
          });
    }

    Kokkos::Profiling::popRegion();
  }

  void gather_part_1_count(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::gather_part_1_count");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Gather part 1 (count): Team loop over selected link buckets, fetch their partition, team loop over the links,
    // serial loop over the downward linked entities, if their bucket is dirty, atomically increment the connectivity
    // counts of the downward connected entities.

    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    const stk::mesh::EntityRank link_rank = link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            // Loop over the linked entities of this link
            for (unsigned d = 0; d < dimensionality; ++d) {
              // Only consider non-empty links
              if (get_linked_entity(link_index, d) != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = get_linked_entity_rank(link_index, d);
                auto &crs_bucket_conn =
                    crs_partition.get_crs_bucket_conn(linked_entity_rank, linked_entity_index.bucket_id);

                if (crs_bucket_conn.dirty_) {
                  // Atomically increment the connectivity count
                  Kokkos::atomic_add(&crs_bucket_conn.num_connected_links_(linked_entity_index.bucket_ord), 1u);
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  void gather_part_2_partial_sum(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::gather_part_2_partial_sum");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Gather part 2 (partial sum): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop
    // over the partitions, if its corresponding linked bucket has been modified, thread loop over the linked bucket to
    // partial sum the connectivity counts into the connectivity offsets.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, team.league_rank());
            unsigned bucket_size = bucket.size();

            // Serial loop over the partitions
            for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
              NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);

              // Fetch the crs bucket conn for this rank and bucket
              auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

              // If the bucket is dirty, partial sum the connectivity counts into the connectivity offsets.
              if (crs_bucket_conn.dirty_) {
                // Use a parallel_scan to compute the offsets
                Kokkos::parallel_scan(
                    Kokkos::TeamThreadRange(team, 0u, bucket_size),
                    KOKKOS_LAMBDA(unsigned i, unsigned &partial_sum, bool final_pass) {
                      const unsigned num_connected_links = crs_bucket_conn.num_connected_links_(i);
                      if (final_pass) {
                        // exclusive offset
                        crs_bucket_conn.sparse_connectivity_offsets_(i) = partial_sum;

                        if (i == bucket_size - 1) {
                          // Store the total number of connected links at the end of the offsets array
                          crs_bucket_conn.sparse_connectivity_offsets_(bucket_size) = partial_sum + num_connected_links;
                        }
                      }
                      partial_sum += num_connected_links;
                    });
              }
            }
          });
    }

    Kokkos::Profiling::popRegion();
  }

  void scatter_part_1_setup(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_1_setup");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Scatter part 1 (setup): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop over
    // the partitions, if its corresponding linked bucket has been modified, reset the connectivity counts to zero.
    //
    //
    reset_dirty_linked_buckets(link_subset_selector);

    // Resize the bucket sparse connectivity arrays
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
      NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        // Only attempt to resize dirty buckets that have non-zero connections
        for (unsigned bucket_id = 0; bucket_id < crs_partition.num_buckets(rank); ++bucket_id) {
          auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
          if (crs_bucket_conn.dirty_ && crs_bucket_conn.sparse_connectivity_offsets_.extent(0) > 0) {
            // Only resize if needed
            unsigned new_size = crs_bucket_conn.sparse_connectivity_offsets_(crs_bucket_conn.size());
            if (new_size > crs_bucket_conn.sparse_connectivity_.extent(0)) {  // Only grow
              Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), crs_bucket_conn.sparse_connectivity_,
                             new_size);
            }
          }
        }
      }
    }

    Kokkos::Profiling::popRegion();
  }

  void scatter_part_2_fill(const stk::mesh::Selector &selector) {  // This is currently broken.
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_2_fill");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Scatter part 2 (fill): Team loop over each selected link buckets, fetch
    // their partition ID, thread loop over the links, serial loop over their downward linked entities, and if their
    // bucket is dirty, scatter the link. Copy the link into the old field. Update the count as each entity is inserted.

    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    const stk::mesh::EntityRank link_rank = link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            // Loop over the linked entities of this link
            for (unsigned d = 0; d < dimensionality; ++d) {
              // Only consider non-empty links
              stk::mesh::Entity linked_entity = get_linked_entity(link_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = get_linked_entity_rank(link_index, d);
                auto &crs_bucket_conn =
                    crs_partition.get_crs_bucket_conn(linked_entity_rank, linked_entity_index.bucket_id);

                if (crs_bucket_conn.dirty_) {
                  // Atomically increment the connectivity count
                  const unsigned offset = crs_bucket_conn.sparse_connectivity_offsets_(linked_entity_index.bucket_ord);
                  const unsigned num_inserted_old = Kokkos::atomic_fetch_add(
                      &crs_bucket_conn.num_connected_links_(linked_entity_index.bucket_ord), 1);
                  crs_bucket_conn.sparse_connectivity_(offset + num_inserted_old) = link;
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  void finalize_crs_update(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_3_finalize");
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Finalize CRS update: Mark all buckets as no longer dirty, mark all selected links are up-to-date, and copy the
    // old COO connectivity to the new COO connectivity (for the given selector)

    // Serial loop over each rank, parallel loop over the stk buckets of said rank, serial loop over the partitions,
    // if its corresponding linked bucket has been modified, reset the dirty flag.
    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    Kokkos::Timer timer;
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Regular for loop over each stk bucket of said rank
      for (unsigned bucket_id = 0; bucket_id < ngp_mesh.num_buckets(rank); ++bucket_id) {
        // Serial loop over the partitions
        for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
          NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);

          // Fetch the crs bucket conn for this rank and bucket
          auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
          crs_bucket_conn.dirty_ = false;  // Reset the dirty flag
        }
      }

      // TODO(palmerb4): It appears as though resetting the flag in a parallel_for is slower than doing it
      // serially (at least for a CPU build). Is this true for GPU builds too?

      // Regular parallel_for over each stk bucket of said rank
      // Kokkos::parallel_for(
      //     Kokkos::RangePolicy<stk::mesh::NgpMesh::MeshExecSpace>(0, ngp_mesh.num_buckets(rank)),
      //     KOKKOS_LAMBDA(const int &bucket_id) {
      //       // Serial loop over the partitions
      //       for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
      //         NewNgpCRSPartition &crs_partition = crs_partitions(partition_id);

      //         // Fetch the crs bucket conn for this rank and bucket
      //         auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
      //         crs_bucket_conn.dirty_ = false;  // Reset the dirty flag
      //       }
      //     });
    }
    std::cout << " Reset dirty flag time: " << timer.seconds() << " seconds" << std::endl;

    // Mark all selected links as up-to-date
    timer.reset();
    auto &link_needs_updated_field = link_meta_data().link_crs_needs_updated_field();
    ::mundy::mesh::field_fill(0, link_needs_updated_field, link_subset_selector, stk::ngp::ExecSpace());
    std::cout << " Mark links up-to-date time: " << timer.seconds() << " seconds" << std::endl;

    // Copy the old COO connectivity to the new COO connectivity
    timer.reset();
    ::mundy::mesh::field_copy<entity_value_t>(link_meta_data().linked_entities_field(),
                                              link_meta_data().linked_entities_crs_field(), link_subset_selector,
                                              stk::ngp::ExecSpace());
    std::cout << " Copy old to new COO time: " << timer.seconds() << " seconds" << std::endl;

    Kokkos::Profiling::popRegion();
  }

  void check_all_links_in_sync(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    int needs_updated_count =
        field_sum<int>(link_meta_data().link_crs_needs_updated_field(), link_subset_selector, stk::ngp::ExecSpace());
    MUNDY_THROW_REQUIRE(needs_updated_count == 0, std::logic_error, "There are still links that are out of sync.");
  }

  void check_linked_bucket_conn_size(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Serial loop over each selected partition. Serial loop over each rank.
    // Assert that the size of the bucket conn is the same as the number of STK buckets of the given rank.
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NewNgpCRSPartition &partition = partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        unsigned num_buckets = partition.num_buckets(rank);
        unsigned num_stk_buckets = bulk_data().buckets(rank).size();
        MUNDY_THROW_REQUIRE(num_buckets == num_stk_buckets, std::logic_error,
                            "The number of linked buckets does not match the number of STK buckets.");
      }
    }
  }

  void check_coo_to_crs_conn(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Serial loop over each partial, hierarchical parallelism over each link in said selector,
    // serial loop over each of its downward connections, if it is non-empty, fetch their CRS conn,
    // serial loop over each link in the CRS conn, and check if it is the same as the source link.

    const stk::mesh::NgpMesh &local_ngp_mesh = ngp_mesh();
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NewNgpCRSPartition &partition = partitions(partition_id);
      const unsigned dimensionality = partition.link_dimensionality();
      stk::mesh::EntityRank link_rank = link_meta_data().link_rank();

      stk::mesh::for_each_entity_run(
          local_ngp_mesh, link_rank, partition.selector(), KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &linker_index) {
            // Loop over each linked entity in the linker
            for (unsigned d = 0; d < dimensionality; ++d) {
              stk::mesh::Entity linked_entity = get_linked_entity(linker_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                // Fetch the CRS connectivity of the linked entity
                stk::mesh::EntityRank linked_entity_rank = get_linked_entity_rank(linker_index, d);
                stk::mesh::FastMeshIndex linked_entity_index = get_linked_entity_index(linker_index, d);
                ConnectedEntities connected_links =
                    partition.get_connected_links(linked_entity_rank, linked_entity_index);

                MUNDY_THROW_REQUIRE(partition.num_connected_links(linked_entity_rank, linked_entity_index) > 0,
                                    std::logic_error,
                                    "A linked entity in the CRS connectivity is not connected to any links.");
                MUNDY_THROW_REQUIRE(
                    partition.num_connected_links(linked_entity_rank, linked_entity_index) == connected_links.size(),
                    std::logic_error,
                    "The number of connected links in the CRS connectivity does not match the size of the connected "
                    "links array.");

                // Loop over each connected link in the CRS connectivity
                bool found_link = false;
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::FastMeshIndex connected_link_index =
                      local_ngp_mesh.fast_mesh_index(connected_links[connected_link_ord]);
                  if (fma_equal(connected_link_index, linker_index)) {
                    found_link = true;
                    break;
                  }
                }

                MUNDY_THROW_REQUIRE(found_link, std::logic_error,
                                    "A linker in the CRS connectivity is missing from the COO connectivity.");
              }
            }
          });
    }
  }

  void check_crs_to_coo_conn(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over each CRS partition,
    // fetch the corresponding CRS bucket conn, thread loop over the entities in said bucket, serial loop over their
    // connected links, and check if the source entity is linked to the link.

    const stk::mesh::NgpMesh &local_ngp_mesh = ngp_mesh();
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);

    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      stk::NgpVector<unsigned> bucket_ids =
          local_ngp_mesh.get_bucket_ids(rank, bulk_data().mesh_meta_data().universal_part());

      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

      Kokkos::parallel_for(
          team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
            const stk::mesh::NgpMesh::BucketType &bucket = local_ngp_mesh.get_bucket(rank, bucket_id);
            unsigned num_entities = bucket.size();

            // Serial loop over each partition
            unsigned num_partitions = partitions.extent(0);
            for (unsigned partition_id = 0; partition_id < num_partitions; ++partition_id) {
              const NewNgpCRSPartition &partition = partitions(partition_id);
              const unsigned dimensionality = partition.link_dimensionality();

              // Thread loop over each entity in the bucket
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_entities), [&](const int &i) {
                stk::mesh::Entity entity = bucket[i];
                stk::mesh::FastMeshIndex entity_index = local_ngp_mesh.fast_mesh_index(entity);

                // Each connected link better be attached to us
                ConnectedEntities connected_links = partition.get_connected_links(rank, entity_index);
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::Entity connected_link = connected_links[connected_link_ord];
                  stk::mesh::FastMeshIndex connected_link_index = local_ngp_mesh.fast_mesh_index(connected_link);

                  MUNDY_THROW_REQUIRE(connected_link != stk::mesh::Entity(), std::logic_error,
                                      "A connected link in the CRS connectivity is empty.");

                  // Serial loop over each linked entity in the connected link
                  bool found_entity = false;
                  for (unsigned d = 0; d < dimensionality; ++d) {
                    stk::mesh::Entity linked_entity = get_linked_entity(connected_link_index, d);
                    if (linked_entity == entity) {
                      found_entity = true;
                      break;
                    }
                  }

                  MUNDY_THROW_REQUIRE(found_entity, std::logic_error,
                                      "A linked entity in the COO connectivity is missing from the CRS connectivity.");
                }
              });
            }
          });
    }
  }
  //@}

  //! \name Internal actions
  //@{

  KOKKOS_INLINE_FUNCTION
  bool fma_equal(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
    return (lhs.bucket_id == rhs.bucket_id) && (lhs.bucket_ord == rhs.bucket_ord);
  }

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

  using SelectorToPartitionsMap = std::map<stk::mesh::Selector, NgpCRSPartitionView>;
  using PartitionKeyToIdMap = std::map<PartitionKey, unsigned>;
  using PartVectorToRequestLinks = std::map<stk::mesh::PartVector, NewNgpLinkRequests>;
  mutable SelectorToPartitionsMap
      selector_to_partitions_map_;  // Maybe we want to use a view here to reduce copy overhead.
  mutable PartitionKeyToIdMap partition_key_to_id_map_;
  mutable PartVectorToRequestLinks part_vector_to_request_links_map_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  stk::mesh::EntityRank link_rank_;
  stk::mesh::NgpMesh ngp_mesh_;
  NewNgpLinkMetaData ngp_link_meta_data_;
  mutable NgpCRSPartitionView all_crs_partitions_;
  LinkBucketToPartitionIdMap stk_link_bucket_to_partition_id_map_;
  //@}
};  // NewNgpLinkData

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
