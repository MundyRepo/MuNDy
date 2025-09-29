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

#ifndef MUNDY_MESH_LINKDATA_HPP_
#define MUNDY_MESH_LINKDATA_HPP_

/// \file LinkData.hpp
/// \brief Declaration of the LinkData class

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
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>          // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>           // for mundy::mesh::field_copy
#include <mundy_mesh/impl/LinkedBucketConn.hpp>  // for mundy::mesh::impl::LinkedBucketConn

namespace mundy {

namespace mesh {

//! \name Forward declarations
//@{

class LinkData;
class LinkPartition;
class NgpLinkData;
//@}

using PartitionKey = stk::mesh::OrdinalVector;  // sorted list of part ordinals

/// \class LinkData
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
/// # LinkData
/// The LinkData class is the main interface for interacting with the link data on the mesh. It is meant to mirror
/// BulkData's connectivity interface while allowing multiple LinkData objects to be used on the same mesh, each with
/// separately managed data. Use LinkData to connect links to linked entities and to get the linked entities for a given
/// link. And similar to STK's for_each_entity_run, use non-member functions acting on the LinkData to loop over links
/// and linked entities.
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
/// of the other infrastructure needed to maintain parallel consistency. As such, you must call propagate_updates()
/// before entering a modification cycle or before using the for_each_linked_entity_run function.
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
//      operator(const LinkData& link_data, const stk::mesh::Entity& linker).
///  This function is thread parallel over each link in the given link_data that falls within the given
///  link_subset_selector. Notice that for_each_link does not require this synchronization, meaning that
/// dynamic linker creation and destruction can be done in parallel within a for_each_link loop AND you can continue
/// to perform for_each_link loops while the link data is out-of_sync.
///
/// - for_each_linked_entity_run(link_data, linked_entity_selector, linker_subset_selector, functor)
/// The functor must either have an
///     operator(const stk::mesh::BulkData&, const stk::mesh::Entity&linked_entity, const stk::mesh::Entity&linker) or
///     an operator(const LinkData&, const stk::mesh::Entity& linked_entity, const stk::mesh::Entity& linker).
/// This functions is thread parallel over each linked entity in the given linked_entity_selector that falls within a
/// bucket that a linker in the given linker_subset_selector connects to. This means that the functor will be called in
/// serial for each link that connects to a linked entity. Importantly, this function requires infrastructure that
/// requires the link data to be "up-to-date" (i.e., you must call link_data.propagate_updates() before using it if you
/// have modified the link data).
///
/// You might think, why not just provide a get_connected_links(entity) function? The reason is that the links
/// themselves are heterogeneous and bucketized. As such, there is no practical way to provide contiguous access to the
/// set of links that an entity connects to while supporting subset selection of links and without dynamic memory
/// allocation.
///
/// \note To Devs: Hi! Welcome. If you want to better understand the LinkData or our links in general, I recommend
/// looking at it as maintaining two connectivity structures: a COO-like structure providing access from a linker to its
/// linked entities and a CRS-like structure providing access from a linked entity to its linkers. The COO is the
/// dynamic "driver" that is trivial to modify (even in parallel) and the CRS is a more heavy-weight sparse data
/// structure that is non-trivial to modify in parallel since it often requires memory allocation. The propagate_updates
/// function is responsible for mapping all of the modifications to the COO structure to the CRS structure. There are
/// some operations that fundamentally require a CRS-like structure such as maintaining parallel consistency as entities
/// are removed from the mesh or change parallel ownership/sharing or performing operations that require a serial loop
/// over each linker that connects to a given linked entity.
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
/// # NgpLinkData
/// Following STK's lead, we provide an NgpLinkData class, which is device compatible and a
/// get_updated_ngp_data(link_data)-> NgpLinkData for getting the NgpLinkData for a given LinkData. With the
/// NgpLinkData, you may perform declare/delete_relation and get_linked_entity on either the host or device. As with all
/// of our Ngp* classes, modifications need to be synchronized to and from the host via the modify_on_host/device and
/// and sync_to_host/device functions. We also offer an ngp versions of
/// for_each_link_run; although, we do not currently support for_each_linked_entity_run on the device. We hope to
/// add support for this in the future.

class LinkMetaData {
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
  virtual ~LinkMetaData() = default;

  /// \brief Default copy/move constructors/operators.
  LinkMetaData(const LinkMetaData &) = default;
  LinkMetaData(LinkMetaData &&) = default;
  LinkMetaData &operator=(const LinkMetaData &) = default;
  LinkMetaData &operator=(LinkMetaData &&) = default;
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

  /// \brief Fetch the linked entities field.
  const linked_entities_field_t &linked_entities_field() const {
    return linked_entities_field_;
  }
  linked_entities_field_t &linked_entities_field() {
    return linked_entities_field_;
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
  LinkMetaData() = delete;

  /// \brief Construct and declare
  LinkMetaData(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank)
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

  /// \brief Fetch the linked entity ids field (non-const).
  linked_entity_ids_field_t &linked_entity_ids_field() {
    return linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field (non-const).
  linked_entity_ranks_field_t &linked_entity_ranks_field() {
    return linked_entity_ranks_field_;
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

  friend class LinkData;
  friend class LinkPartition;
  friend class NgpLinkData;
  friend LinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name,
                                             stk::mesh::EntityRank link_rank);

  template <typename FunctionToRunPerLinkedEntity>
  friend void for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
                                         const FunctionToRunPerLinkedEntity &);
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
  mutable bool link_coo_has_been_updated_ = false;
  //@}
};

LinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank);

/// \brief The main interface for interacting with partitioned link data on the mesh.
///
/// Unlike STK, we expose the concept of partitions. These are subsets of the link data corresponding to
/// links within a mathematical partition of the set space of all links. From a programming perspective,
/// a partition is uniquely identified by the set of parts that links within it belong to. That includes STK-specific
/// parts like the universal, locally owned, and shared parts.
///
/// Some of our core functionality for interacting with links and linked entities is partition aware, storing contiguous
/// or bucketized data for each partition. For example, get_connected_links(entity) will return a contiguous array of
/// links within the current partition that attach to the given entity. This information is otherwise unavailable at the
/// LinkData level since get_connected_links(entity) would require dynamic memory allocation or compile-time link
/// dimensionality.
///
/// Partitions may either be fetched via passing in a link bucket, a vector of parts, or a partition key.
///
/// \note Link partitions must be used with some care to avoid inextensibility. Specifically, be wary of harcoding
/// partition keys into your code. Instead, try to use other ways to get the partition keys that returns them in a way
/// that is flexible to the addition of new parts (such as bucket.supersets()). As a result, we recommend that
/// partitions only be touched by power users. Typically, you will use for_each_link_run, for_each_linked_entity_run,
/// and basic getters to interact with the link data.
class LinkPartition {
 public:
  //! \name Aliases
  //@{

  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  //@}

  //! \name Public constructors and destructor
  //@{

  /// \brief No default constructor.
  LinkPartition() = delete;

  /// \brief Default copy/move constructors/operators.
  LinkPartition(const LinkPartition &) = default;
  LinkPartition(LinkPartition &&) = default;
  LinkPartition &operator=(const LinkPartition &) = default;
  LinkPartition &operator=(LinkPartition &&) = default;

  /// \brief Canonical constructor.
  LinkPartition(LinkData &link_data, const PartitionKey &key, unsigned dimensionality);

  /// \brief Destructor.
  virtual ~LinkPartition() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the partition key.
  const PartitionKey &key() const {
    return key_;
  }

  /// \brief Fetch the link data manager.
  const LinkData &link_data() const {
    return link_data_;
  }
  LinkData &link_data() {
    return link_data_;
  }

  /// \brief Fetch the link meta data manager.
  const LinkMetaData &link_meta_data() const {
    return link_meta_data_;
  }
  LinkMetaData &link_meta_data() {
    return link_meta_data_;
  }

  /// \brief Get the dimensionality of links in this partition
  unsigned link_dimensionality() const {
    return dimensionality_;
  }

  /// \brief Check if this partition contains a given part
  bool contains(const stk::mesh::Part &part) const {
    return contains(part.mesh_meta_data_ordinal());
  }

  /// \brief Check if this partition contains a given part
  bool contains(unsigned part_ordinal) const {
    return std::find(key_.begin(), key_.end(), part_ordinal) != key_.end();
  }
  //@}

  //! \name CRS connectivity
  //@{

  /// @brief \brief If any of our linkers connect to an entity in the given bucket within the CRS connectivity.
  /// @param bucket
  /// @return
  inline bool connects_to(const stk::mesh::Bucket &bucket) const {
    stk::mesh::Bucket *bucket_ptr = const_cast<stk::mesh::Bucket *>(&bucket);
    return bucket_to_linked_conn_ptr_->find(bucket_ptr) != bucket_to_linked_conn_ptr_->end();
  }

  /// \brief Get all links in the current partition that connect to the given entity in the CRS connectivity.
  ///
  /// \param entity [in] The entity to get connected links for.
  /// \return The connected entities object.
  inline ConnectedEntities get_connected_links(const stk::mesh::Entity &entity) const {
    stk::mesh::Bucket &bucket = bulk_data_.bucket(entity);
    stk::mesh::Ordinal bucket_ordinal = bulk_data_.bucket_ordinal(entity);
    return connects_to(bucket) ? bucket_to_linked_conn_ptr_->at(&bucket).get_connected_entities(bucket_ordinal)
                               : ConnectedEntities();
  }

  /// \brief Get the number of links in the current partition that connect to the given entity in the CRS connectivity.
  inline size_t num_connected_links(const stk::mesh::Entity &entity) const {
    stk::mesh::Bucket &bucket = bulk_data_.bucket(entity);
    stk::mesh::Ordinal bucket_ordinal = bulk_data_.bucket_ordinal(entity);
    return connects_to(bucket) ? bucket_to_linked_conn_ptr_->at(&bucket).num_connectivity(bucket_ordinal) : 0;
  }
  //@}

  //! \name Delayed creation and destruction
  //@{

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
  void increase_request_link_capacity(size_t capacity) {
    MUNDY_THROW_REQUIRE(contains(bulk_data_.mesh_meta_data().locally_owned_part()), std::invalid_argument,
                        "Requesting links is only valid for partitions that are locally owned "
                        "(i.e., those that contain the locally owned part).");

    if (capacity > link_requests_capacity_view_()) {
      link_requests_capacity_view_() = capacity;
      Kokkos::resize(requested_links_, capacity, dimensionality_);
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
  void request_link(LinkedEntities &&...linked_entities) const {
    MUNDY_THROW_ASSERT(link_dimensionality() == sizeof...(linked_entities), std::invalid_argument,
                       "The number of linked entities must match the link dimensionality.");

    // For those not familiar with atomic_fetch_sub, it returns the value before the subtraction.
    size_t old_size = Kokkos::atomic_fetch_add(&link_requests_size_view_(), 1);

    MUNDY_THROW_ASSERT(old_size + 1 <= link_requests_capacity_view_(), std::invalid_argument,
                       "The number of requests exceeds the capacity.");
    insert_request(std::make_index_sequence<sizeof...(linked_entities)>(), old_size,
                   std::forward<LinkedEntities>(linked_entities)...);
  }

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call.
  inline void request_destruction(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_marked_for_destruction_field = link_meta_data_.link_marked_for_destruction_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();
    stk::mesh::field_data(link_marked_for_destruction_field, linker)[0] = true;
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;  // CRS conn of linked entities needs updated
  }

  /// \brief Get the current number of request_link requests.
  size_t request_link_size() const {
    return link_requests_size_view_();
  }

  /// \brief Get the current capacity of request_link requests.
  size_t request_link_capacity() const {
    return link_requests_capacity_view_();
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
    MUNDY_THROW_REQUIRE(link_requests_size_view_() <= link_requests_capacity_view_(), std::invalid_argument,
                        "The number of requests exceeds the capacity. You wrote to invalid memory when requesting "
                        "links and somehow didn't get a segfault. Neat!");
    size_t global_requests_size = 0;
    stk::all_reduce_sum(bulk_data_.parallel(), &link_requests_size_view_(), &global_requests_size, 1);

    if (global_requests_size > 0) {
      bool we_started_modification = false;
      if (!bulk_data_.in_modifiable_state()) {
        bulk_data_.modification_begin();
        we_started_modification = true;
      }

      if (assume_fully_consistent) {
        if (bulk_data_.parallel_size() == 1) {
          process_link_requests_fully_consistent_single_process();
        } else {
          process_link_requests_fully_consistent_multi_process();
        }
      } else {
        if (bulk_data_.parallel_size() == 1) {
          process_link_requests_partially_consistent_single_process();
        } else {
          process_link_requests_partially_consistent_multi_process();
        }
      }

      if (we_started_modification) {
        bulk_data_.modification_end();
      }
    }
  }
  //@}

 protected:
  //! \name Type aliases
  //@{

  using BucketToLinkedConn = std::map<stk::mesh::Bucket *, impl::LinkedBucketConn>;
  //@}

  //! \name Bucketized CRS connectivity details
  //@{

  /// \brief Add a connected link to the given entity in the CRS connectivity.
  ///
  /// This is not thread safe.
  ///
  /// \param entity [in] The entity that might be connected to the given linker.
  /// \param linker [in] The linker to add.
  inline void add_connected_link(const stk::mesh::Entity &entity, const stk::mesh::Entity &linker) {
    const stk::mesh::Bucket &bucket = bulk_data_.bucket(entity);
    stk::mesh::Ordinal bucket_ordinal = bulk_data_.bucket_ordinal(entity);
    add_connected_link(bucket, bucket_ordinal, linker);
  }

  /// \brief Add a connected link to the given entity in the CRS connectivity.
  ///
  /// This is not thread safe.
  ///
  /// \param bucket [in] The bucket that the entity belongs to.
  /// \param bucket_ordinal [in] The ordinal of the entity in the bucket.
  /// \param linker [in] The linker to add.
  inline void add_connected_link(const stk::mesh::Bucket &bucket, const stk::mesh::Ordinal bucket_ordinal,
                                 const stk::mesh::Entity &linker) {
    impl::LinkedBucketConn &linked_bucket_conn =
        (*bucket_to_linked_conn_ptr_)[const_cast<stk::mesh::Bucket *>(&bucket)];
    unsigned current_num_connected_links = linked_bucket_conn.num_connectivity(bucket_ordinal);
    linked_bucket_conn.add_connectivity(bucket_ordinal, linker, current_num_connected_links);
  }

  /// \brief Remove a connected link from the given entity in the CRS connectivity.
  ///
  /// This is not thread safe if multiple threads act on this bucket at the same time.
  ///
  /// \param entity [in] The entity that might be connected to the given linker.
  /// \param linker [in] The linker to remove.
  /// \return If the linker was removed.
  inline bool remove_connected_link(const stk::mesh::Entity &entity, const stk::mesh::Entity &linker) {
    const stk::mesh::Bucket &bucket = bulk_data_.bucket(entity);
    stk::mesh::Ordinal bucket_ordinal = bulk_data_.bucket_ordinal(entity);
    return remove_connected_link(bucket, bucket_ordinal, linker);
  }

  /// \brief Remove a connected link from the given entity in the CRS connectivity.
  ///
  /// This is not thread safe if multiple threads act on this bucket at the same time.
  ///
  /// \param bucket [in] The bucket that the entity belongs to.
  /// \param bucket_ordinal [in] The ordinal of the entity in the bucket.
  /// \param linker [in] The linker to remove.
  /// \return If the linker was removed.
  inline bool remove_connected_link(const stk::mesh::Bucket &bucket, const stk::mesh::Ordinal bucket_ordinal,
                                    const stk::mesh::Entity &linker) {
    bool we_connect_to_bucket = connects_to(bucket);
    MUNDY_THROW_REQUIRE(we_connect_to_bucket, std::invalid_argument,
                        "This partition does not connect to the given bucket.");

    bool removed = false;
    if (we_connect_to_bucket) {
      impl::LinkedBucketConn &linked_bucket_conn =
          bucket_to_linked_conn_ptr_->at(const_cast<stk::mesh::Bucket *>(&bucket));
      ConnectedEntities connected_links = linked_bucket_conn.get_connected_entities(bucket_ordinal);
      unsigned num_connected_links = connected_links.size();
      for (unsigned j = 0; j < num_connected_links; ++j) {
        if (connected_links[j] == linker) {
          linked_bucket_conn.remove_connectivity(bucket_ordinal, linker, j);
          removed = true;
          break;
        }
      }
    }

    return removed;
  }
  //@}

  //! \name Internal getters
  //@{

  /// \brief Get an iterator over the set of active (Bucket*, LinkedBucketConn) pairs.
  BucketToLinkedConn::iterator linked_bucket_begin() {
    return bucket_to_linked_conn_ptr_->begin();
  }
  BucketToLinkedConn::iterator linked_bucket_end() {
    return bucket_to_linked_conn_ptr_->end();
  }
  BucketToLinkedConn::const_iterator linked_bucket_begin() const {
    return bucket_to_linked_conn_ptr_->begin();
  }
  BucketToLinkedConn::const_iterator linked_bucket_end() const {
    return bucket_to_linked_conn_ptr_->end();
  }

  /// \brief Get linked bucket con for a given bucket
  impl::LinkedBucketConn &get_linked_bucket_conn(const stk::mesh::Bucket &bucket) {
    return bucket_to_linked_conn_ptr_->at(const_cast<stk::mesh::Bucket *>(&bucket));
  }
  const impl::LinkedBucketConn &get_linked_bucket_conn(const stk::mesh::Bucket &bucket) const {
    return bucket_to_linked_conn_ptr_->at(const_cast<stk::mesh::Bucket *>(&bucket));
  }
  //@}

 private:
  //! \name Friends
  //@{

  friend class LinkData;
  template <typename FunctionToRunPerPairOfLinkedEntityAndLink>
  friend void for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
                                         const FunctionToRunPerPairOfLinkedEntityAndLink &);
  //@}

  //! \name Internal functions
  //@{

  /// \brief Unrole the entities into the requested links view.
  template <size_t... Is, typename... LinkedEntities>
  void insert_request(std::index_sequence<Is...>, size_t request_index, LinkedEntities &&...linked_entities) const {
    ((requested_links_(request_index, Is) = std::forward<LinkedEntities>(linked_entities)), ...);
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

  // Core data
  LinkData &link_data_;
  BulkData &bulk_data_;
  LinkMetaData &link_meta_data_;
  PartitionKey key_;
  stk::mesh::EntityRank link_rank_;
  unsigned dimensionality_;

  // Host-bucketized-crs connectivity. Used for tracking many small serial changes during mesh modification
  impl::LinkedBucketConn AnEmptyLinkedBucketConn_;
  std::shared_ptr<BucketToLinkedConn> bucket_to_linked_conn_ptr_;

  // Stuff for requests
  using size_t_view = Kokkos::View<size_t, typename stk::ngp::HostExecSpace::memory_space>;
  mutable size_t_view link_requests_size_view_;
  mutable size_t_view link_requests_capacity_view_;
  Kokkos::View<stk::mesh::Entity **, typename stk::ngp::HostExecSpace::memory_space> requested_links_;
  //@}
};

class LinkData {
 public:
  //! \name Aliases
  //@{

  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor.
  LinkData() = delete;

  /// \brief No copy or move constructors/operators.
  LinkData(const LinkData &) = delete;
  LinkData(LinkData &&) = delete;
  LinkData &operator=(const LinkData &) = delete;
  LinkData &operator=(LinkData &&) = delete;

  /// \brief Canonical constructor.
  /// \param bulk_data [in] The bulk data manager we extend.
  /// \param link_meta_data [in] Our meta data manager.
  LinkData(BulkData &bulk_data, LinkMetaData link_meta_data)
      : bulk_data_(bulk_data), mesh_meta_data_(bulk_data.mesh_meta_data()), link_meta_data_(link_meta_data) {
  }

  /// \brief Destructor.
  virtual ~LinkData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the bulk data's meta data manager
  const MetaData &mesh_meta_data() const {
    return mesh_meta_data_;
  }

  /// \brief Fetch the bulk data's meta data manager
  MetaData &mesh_meta_data() {
    return mesh_meta_data_;
  }

  /// \brief Fetch the link meta data manager
  const LinkMetaData &link_meta_data() const {
    return link_meta_data_;
  }

  /// \brief Fetch the link meta data manager
  LinkMetaData &link_meta_data() {
    return link_meta_data_;
  }

  /// \brief Fetch the bulk data manager we extend
  const BulkData &bulk_data() const {
    return bulk_data_;
  }

  /// \brief Fetch the bulk data manager we extend
  BulkData &bulk_data() {
    return bulk_data_;
  }
  //@}

  //!\name General getters
  //@{

  /// \brief Get the partition key for a given set of link parts (independent of their order)
  PartitionKey get_partition_key(const stk::mesh::PartVector &link_parts) {
    stk::mesh::OrdinalVector link_parts_and_supersets;
    stk::mesh::impl::fill_add_parts_and_supersets(link_parts, link_parts_and_supersets);
    return link_parts_and_supersets;
  }

  /// \brief Get the partition key for a given link bucket.
  PartitionKey get_partition_key(const stk::mesh::Bucket &link_bucket) {
    return get_partition_key(link_bucket.supersets());
  }

  /// \brief Get a specific partition of the link data.
  /// \param link_bucket [in] The link bucket to get the partition for.
  LinkPartition &get_partition(const stk::mesh::Bucket &link_bucket) {
    return get_partition(get_partition_key(link_bucket));
  }

  /// \brief Get a specific partition of the link data.
  /// \param parts [in] The parts to get the partition for.
  LinkPartition &get_partition(const stk::mesh::PartVector &parts) {
    return get_partition(get_partition_key(parts));
  }

  /// \brief Get a specific partition of the link data.
  LinkPartition &get_partition(const PartitionKey &key) {
    auto [it, inserted] = partitions_.try_emplace(key, *this, key, get_linker_dimensionality(key));
    return it->second;
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
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    declare_relation_no_update(linker, linked_entity, link_ordinal);

    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline void delete_relation(const stk::mesh::Entity &linker, unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data_.linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data_.linked_entity_bucket_ords_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();

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
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }

  /// \brief Get the linked entity index for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::FastMeshIndex get_linked_entity_index(const stk::mesh::Entity &linker,
                                                          unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_bucket_ids_field = link_meta_data_.linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data_.linked_entity_bucket_ords_field();
    return stk::mesh::FastMeshIndex(stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal],
                                    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal]);
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::EntityId get_linked_entity_id(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    return stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal];
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    return static_cast<stk::mesh::EntityRank>(stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal]);
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links
  ///
  /// This function must be called before either entering a modification cycle or using the for_each_linked_entity_run
  /// function.
  void propagate_updates() {
    // 1. get each stk link bucket,
    // 2. get the key for that bucket's partition,
    // 3. insert a new partition conn into the map if one doesn't already exist,
    // 4. loop over each link in said stk partition
    // 5. if the link is out of sync, loop over each of the linked entities and its crs version
    // 6. if the linked entity and its crs version are the same, the connectivity is in sync, skip
    // 7. if the linked entity is valid add the linker to the end of the linked entity's connectivity
    // 8. set all entities to up-to-data and stash current crs entities.
    using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

    stk::mesh::BucketVector link_buckets =
        bulk_data_.get_buckets(link_meta_data_.link_rank(), link_meta_data_.universal_link_part());
    for (stk::mesh::Bucket *link_bucket : link_buckets) {
      unsigned link_dim = get_linker_dimensionality(*link_bucket);
      PartitionKey partition_key = get_partition_key(link_bucket->supersets());
      LinkPartition &link_partition = get_partition(partition_key);

      for (const stk::mesh::Entity &linker : *link_bucket) {
        const auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();
        const bool is_out_of_sync = stk::mesh::field_data(link_needs_updated_field, linker)[0];
        if (!is_out_of_sync) {
          continue;
        }

        for (unsigned i = 0; i < link_dim; ++i) {
          std::cout << "i: " << i << std::endl;
          stk::mesh::Entity linked_entity = get_linked_entity(linker, i);
          stk::mesh::Entity old_linked_entity = get_linked_entity_crs(linker, i);

          if (linked_entity == old_linked_entity) {
            // Already up to date, skip
            continue;
          }

          // If the old_linked_entity is valid, remove the linker from its connectivity
          if (bulk_data_.is_valid(old_linked_entity)) {
            link_partition.remove_connected_link(old_linked_entity, linker);
          }

          // If the new linked entity is valid, add the linker to the end of its connectivity
          // PLEASE NOTE: We assume that a link will never link to the same entity multiple times
          //  we enforce this in debug mode.
          if (bulk_data_.is_valid(linked_entity)) {
#ifndef NDEBUG
            // If in debug, check if the linker is already connected to the linked entity.
            ConnectedEntities connected_links = link_partition.get_connected_links(linked_entity);
            unsigned num_connected_links = connected_links.size();
            std::cout << "num_connected_links: " << num_connected_links << std::endl;
            bool linker_already_connected = false;
            for (unsigned j = 0; j < num_connected_links; ++j) {
              std::cout << "Linked entity: " << bulk_data_.identifier(linked_entity)
                        << "Linker: " << bulk_data_.identifier(linker)
                        << " connected_links[j]: " << bulk_data_.identifier(connected_links[j]) << std::endl;
              if (connected_links[j] == linker && bulk_data_.is_valid(connected_links[j])) {
                linker_already_connected = true;
              }
            }
            MUNDY_THROW_ASSERT(
                !linker_already_connected, std::logic_error,
                "Attempting to add a linker to linked entity crs relation that somehow already exists???."
                "This is possible if the link somehow links the same entity multiple times.");
#endif
            // Assume that the linker is not already connected
            link_partition.add_connected_link(linked_entity, linker);
          }
        }

        // At this point, the CRS and COO link connectivity for this link are in sync.
        // We can now reset the out-of-sync flag.
        stk::mesh::field_data(link_needs_updated_field, linker)[0] = false;
      }
    }

    // At this point, all links are in sync. We can copy the current linked entities field to the crs linked entities
    // field for the current set of links.
    using entity_value_t = stk::mesh::Entity::entity_value_type;
    field_copy<entity_value_t>(link_meta_data_.linked_entities_field(),      // source
                               link_meta_data_.linked_entities_crs_field(),  // target
                               link_meta_data_.universal_link_part(),        //
                               stk::ngp::HostExecSpace());
  }
  //@}

  //! \name Delayed creation and destruction
  //@{

  /// \brief Request the destruction of a link. This will be processed in the next process_requests call.
  inline void request_destruction(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &link_marked_for_destruction_field = link_meta_data_.link_marked_for_destruction_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();
    stk::mesh::field_data(link_marked_for_destruction_field, linker)[0] = true;
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;  // CRS conn of linked entities needs updated
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
    // 1. If needed, enter a modification cycle if not already in one.
    // 2. Destroy all links that have been requested for destruction.
    // 4. Loop over all partitions and call process_requests on them.

    // We only enter a mod cycle if the  global number of links marked for destruction or declaration are non-zero.

    // TODO(palmerb4): We've never encountered this before. This is a sum over a simple field that need to reduce into a
    // potentially larger type. This needs updated with a size_t reduction over an unsigned field.
    unsigned global_num_marked_for_destruction =
        field_sum<unsigned>(link_meta_data_.link_marked_for_destruction_field(), link_meta_data_.universal_link_part(),
                            stk::ngp::ExecSpace{});

    unsigned global_num_request_for_creation = 0;
    unsigned local_num_request_for_creation = 0;
    for (const auto &partition : partitions_) {
      local_num_request_for_creation += partition.second.request_link_size();
    }
    stk::all_reduce_sum(bulk_data_.parallel(), &local_num_request_for_creation, &global_num_request_for_creation, 1);

    // Process requests
    bool we_started_modification = false;
    if (global_num_marked_for_destruction > 0 || global_num_request_for_creation > 0) {
      if (!bulk_data_.in_modifiable_state()) {
        // propagate_updates();  // Must be called before entering a mod cycle!
        bulk_data_.modification_begin();
        we_started_modification = true;
      }
    }

    std::cout << "global_num_marked_for_destruction: " << global_num_marked_for_destruction
              << " global_num_request_for_creation: " << global_num_request_for_creation << std::endl;
    if (global_num_marked_for_destruction > 0) {
      destroy_marked_links();
    }

    if (global_num_request_for_creation > 0) {
      for (auto &partition : partitions_) {
        partition.second.process_requests(assume_fully_consistent);
      }
    }

    if (we_started_modification) {
      bulk_data_.modification_end();
    }
  }

  void modify_on_host();
  void modify_on_device();
  void sync_to_host();
  void sync_to_device();
  //@}

 protected:
  //! \name Internal getters
  //@{

  /// \brief Get the (CRS) linked entity for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  /// \return The linked entity as seen by the CRS.
  inline stk::mesh::Entity get_linked_entity_crs(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_crs_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }

  /// \brief Get the dimensionality of a linker
  /// \param linker [in] The linker (must be valid and of the correct rank).
  inline unsigned get_linker_dimensionality(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    return stk::mesh::field_scalars_per_entity(link_meta_data_.linked_entities_field(), linker);
  }

  /// \brief Get the dimensionality of a linker bucket
  inline unsigned get_linker_dimensionality(const stk::mesh::Bucket &linker_bucket) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == linker_bucket.entity_rank(), std::invalid_argument,
                       "Linker bucket is not of the correct rank.");
    MUNDY_THROW_ASSERT(linker_bucket.member(link_meta_data_.universal_link_part()), std::invalid_argument,
                       "Linker bucket is not a subset of our universal link part.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    return stk::mesh::field_scalars_per_entity(linked_es_field, linker_bucket);
  }

  /// \brief Get the dimensionality of a linker partition
  inline unsigned get_linker_dimensionality(const PartitionKey &partition_key) const {
    MUNDY_THROW_REQUIRE(partition_key.size() > 0, std::invalid_argument, "Partition key is empty.");

    // Fetch the parts
    stk::mesh::PartVector parts(partition_key.size());
    for (size_t i = 0; i < partition_key.size(); ++i) {
      parts[i] = &mesh_meta_data().get_part(partition_key[i]);
    }

    // FieldBase::restrictions
    auto &linked_es_field = link_meta_data_.linked_entities_field();
    const stk::mesh::FieldRestriction &restriction =
        stk::mesh::find_restriction(linked_es_field, link_meta_data_.link_rank(), parts);
    return restriction.num_scalars_per_entity();
  }

  /// \brief Get an iterator over the set of active (key, partition) pairs
  auto partition_begin() const {
    return partitions_.begin();
  }
  /// \brief Get an iterator over the set of active (key, partition) pairs
  auto partition_end() const {
    return partitions_.end();
  }
  auto partition_begin() {
    return partitions_.begin();
  }
  /// \brief Get an iterator over the set of active (key, partition) pairs
  auto partition_end() {
    return partitions_.end();
  }

  /// \brief Get the number of partitions
  size_t num_active_partitions() const {
    return partitions_.size();
  }
  //@}

  //! \name Internal actions
  //@{

  /// \brief Same as declare_relation, but does not flag the link as needing an update.
  /// This is useful for internal functions that will handle performing the update.
  inline void declare_relation_no_update(const stk::mesh::Entity &linker, const stk::mesh::Entity &linked_entity,
                                         unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data_.linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data_.linked_entity_bucket_ords_field();

    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = linked_entity.local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = bulk_data_.identifier(linked_entity);
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] = bulk_data_.entity_rank(linked_entity);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] =
        bulk_data_.bucket(linked_entity).bucket_id();
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = bulk_data_.bucket_ordinal(linked_entity);
  }

  /// \brief Destroy all links that have been marked for destruction.
  void destroy_marked_links() {
    auto &link_marked_for_destruction_field = link_meta_data_.link_marked_for_destruction_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();
    stk::mesh::EntityVector links_to_maybe_destroy;
    stk::mesh::get_selected_entities(link_meta_data_.universal_link_part(),
                                     bulk_data_.buckets(link_meta_data_.link_rank()), links_to_maybe_destroy);

    // Iterate over the entities to destroy and destroy them
    for (const stk::mesh::Entity &link : links_to_maybe_destroy) {
      const bool should_destroy_entity =
          static_cast<bool>(stk::mesh::field_data(link_marked_for_destruction_field, link)[0]);
      if (should_destroy_entity) {
        // MUNDY_THROW_ASSERT(!stk::mesh::field_data(link_needs_updated_field, link)[0], std::logic_error,
        //                    "Attempting to destroy a non-up-to-date link. "
        //                    "This can cause issues with maintaining consistency between the COO and CRS.");

        // Destroy all downward connections from the link
        //   TODO(palmerb4): This is a temporary workaround to intercept destruction of a link
        //    and propagate the effects to the CRS connectivity and will be replaced by a modification observer.
        // stk::mesh::Bucket &link_bucket = bulk_data_.bucket(link);
        // LinkPartition &link_partition = get_partition(get_partition_key(link_bucket));
        // for (unsigned d = 0; d < link_partition.link_dimensionality(); ++d) {
        //   std::cout << "d: " << d << std::endl;
        //   stk::mesh::Entity linked_entity = get_linked_entity(link, d);
        //   if (bulk_data_.is_valid(linked_entity)) {
        //     std::cout << "Destroying link: " << bulk_data_.identifier(link)
        //               << " linked entity: " << bulk_data_.identifier(linked_entity) << std::endl;
        //     // Remove the link from the linked entity's connectivity
        //     link_partition.remove_connected_link(linked_entity, link);
        //   }
        // }

        bool success = bulk_data_.destroy_entity(link);
        MUNDY_THROW_ASSERT(success, std::runtime_error,
                           fmt::format("Failed to destroy link. Link rank: {}, entity id: {}",
                                       bulk_data_.entity_rank(link), bulk_data_.identifier(link)));
      }
    }
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  friend class NgpLinkData;

  template <typename FunctionToRunPerPairOfLinkedEntityAndLink>
  friend void for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
                                         const FunctionToRunPerPairOfLinkedEntityAndLink &);
  //@}

  //! \name Internal members
  //@{

  BulkData &bulk_data_;
  MetaData &mesh_meta_data_;
  LinkMetaData link_meta_data_;
  std::map<PartitionKey, LinkPartition> partitions_;
  //@}
};  // LinkData

LinkData declare_link_data(BulkData &bulk_data, LinkMetaData link_meta_data);

class NgpLinkData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor.
  /// If you find that you need one, let us know and we can consider added one.
  /// The only real reason to not have one is that we want to store a reference to LinkData.
  /// We could replace this with a pointer and add setters + a concretize function.
  KOKKOS_FUNCTION
  NgpLinkData() = delete;

  /// \brief Default copy/move constructors/operators.
  NgpLinkData(const NgpLinkData &) = default;
  NgpLinkData(NgpLinkData &&) = default;
  NgpLinkData &operator=(const NgpLinkData &) = default;
  NgpLinkData &operator=(NgpLinkData &&) = default;

  /// \brief Construct from a LinkData.
  /// Either build this class directly or use get_updated_ngp_data(link_data).
  ///
  /// \param link_meta_data_ptr [in] A pointer to this link's meta data manager.
  /// \param bulk_data [in] The bulk data manager we extend.
  NgpLinkData(const LinkData &link_data)
      : link_data_(link_data),
        link_rank_(link_data_.link_meta_data().link_rank()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(link_data.bulk_data())),
        ngp_linked_entities_field_(                            //
            stk::mesh::get_updated_ngp_field<entity_value_t>(  //
                link_data.impl::get_linked_entities_field(link_meta_data))),
        ngp_linked_entity_ids_field_(                                           //
            stk::mesh::get_updated_ngp_field<LinkMetaData::entity_id_value_t>(  //
                link_data.link_meta_data().linked_entity_ids_field())),
        ngp_linked_entity_ranks_field_(                                           //
            stk::mesh::get_updated_ngp_field<LinkMetaData::entity_rank_value_t>(  //
                link_data.link_meta_data().linked_entity_ranks_field())),
        ngp_linked_entity_bucket_ids_field_(             //
            stk::mesh::get_updated_ngp_field<unsigned>(  //
                link_data.link_meta_data().linked_entity_bucket_ids_field())),
        ngp_linked_entity_bucket_ords_field_(            //
            stk::mesh::get_updated_ngp_field<unsigned>(  //
                link_data.link_meta_data().linked_entity_bucket_ords_field())),
        ngp_link_crs_needs_updated_field_(          //
            stk::mesh::get_updated_ngp_field<int>(  //
                link_data.link_meta_data().link_crs_needs_updated_field())) {
  }

  /// \brief Destructor.
  KOKKOS_FUNCTION
  virtual ~NgpLinkData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the link data
  const LinkData &host_link_data() const {
    return link_data_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_FUNCTION
  const stk::mesh::NgpMesh &ngp_mesh() const {
    return ngp_mesh_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_FUNCTION
  stk::mesh::NgpMesh &ngp_mesh() {
    return ngp_mesh_;
  }

  /// \brief Fetch the link rank
  KOKKOS_FUNCTION
  stk::mesh::EntityRank link_rank() const {
    return link_rank_;
  }
  //@}

  //!\name Actions
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
  /// TODO(palmerb4): Bounds check the link ordinal using the link dimensionality.
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

    ngp_linked_entities_field_(linker_index, link_ordinal) = linked_entity.local_offset();
    ngp_linked_entity_ids_field_(linker_index, link_ordinal) = linked_entity_key.id();
    ngp_linked_entity_ranks_field_(linker_index, link_ordinal) = linked_entity_rank;
    ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal) = linked_entity_index.bucket_id;
    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal) = linked_entity_index.bucket_ord;
    ngp_link_crs_needs_updated_field_(linker_index, 0) = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  void delete_relation(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    ngp_linked_entities_field_(linker_index, link_ordinal) = stk::mesh::Entity().local_offset();
    ngp_linked_entity_ids_field_(linker_index, link_ordinal) = stk::mesh::EntityId();
    ngp_linked_entity_ranks_field_(linker_index, link_ordinal) =
        static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal) = 0;
    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal) = 0;
    ngp_link_crs_needs_updated_field_(linker_index, 0) = true;
  }

  /// \brief Get the linked entity for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_linked_entity(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return stk::mesh::Entity(ngp_linked_entities_field_(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity fast mesh index for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex get_linked_entity_index(const stk::mesh::FastMeshIndex &linker_index,
                                                   unsigned link_ordinal) const {
    return stk::mesh::FastMeshIndex(ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal),
                                    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId get_linked_entity_id(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return ngp_linked_entity_ids_field_(linker_index, link_ordinal);
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::FastMeshIndex &linker_index,
                                               unsigned link_ordinal) const {
    return static_cast<stk::mesh::EntityRank>(ngp_linked_entity_ranks_field_(linker_index, link_ordinal));
  }

  void modify_on_host() {
    ngp_linked_entities_field_.modify_on_host();
    ngp_linked_entity_ids_field_.modify_on_host();
    ngp_linked_entity_ranks_field_.modify_on_host();
    ngp_linked_entity_bucket_ids_field_.modify_on_host();
    ngp_linked_entity_bucket_ords_field_.modify_on_host();
    ngp_link_crs_needs_updated_field_.modify_on_host();
  }

  void modify_on_device() {
    ngp_linked_entities_field_.modify_on_device();
    ngp_linked_entity_ids_field_.modify_on_device();
    ngp_linked_entity_ranks_field_.modify_on_device();
    ngp_linked_entity_bucket_ids_field_.modify_on_device();
    ngp_linked_entity_bucket_ords_field_.modify_on_device();
    ngp_link_crs_needs_updated_field_.modify_on_device();
  }

  void sync_to_host() {
    ngp_linked_entities_field_.sync_to_host();
    ngp_linked_entity_ids_field_.sync_to_host();
    ngp_linked_entity_ranks_field_.sync_to_host();
    ngp_linked_entity_bucket_ids_field_.sync_to_host();
    ngp_linked_entity_bucket_ords_field_.sync_to_host();
    ngp_link_crs_needs_updated_field_.sync_to_host();
  }

  void sync_to_device() {
    ngp_linked_entities_field_.sync_to_device();
    ngp_linked_entity_ids_field_.sync_to_device();
    ngp_linked_entity_ranks_field_.sync_to_device();
    ngp_linked_entity_bucket_ids_field_.sync_to_device();
    ngp_linked_entity_bucket_ords_field_.sync_to_device();
    ngp_link_crs_needs_updated_field_.sync_to_device();
  }
  //@}

 private:
  //! \name Internal aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ngp_linked_entities_field_t = stk::mesh::NgpField<entity_value_t>;
  using ngp_linked_entity_ids_field_t = stk::mesh::NgpField<LinkMetaData::entity_id_value_t>;
  using ngp_linked_entity_ranks_field_t = stk::mesh::NgpField<LinkMetaData::entity_rank_value_t>;
  using ngp_linked_entity_bucket_ids_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_linked_entity_bucket_ords_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_link_crs_needs_updated_field_t = stk::mesh::NgpField<int>;
  //@}

  //! \name Internal members
  //@{

  const LinkData &link_data_;
  stk::mesh::EntityRank link_rank_;
  stk::mesh::NgpMesh ngp_mesh_;
  ngp_linked_entities_field_t ngp_linked_entities_field_;
  ngp_linked_entity_ids_field_t ngp_linked_entity_ids_field_;
  ngp_linked_entity_ranks_field_t ngp_linked_entity_ranks_field_;
  ngp_linked_entity_bucket_ids_field_t ngp_linked_entity_bucket_ids_field_;
  ngp_linked_entity_bucket_ords_field_t ngp_linked_entity_bucket_ords_field_;
  ngp_link_crs_needs_updated_field_t ngp_link_crs_needs_updated_field_;
  //@}
};  // NgpLinkData

inline NgpLinkData get_updated_ngp_data(const LinkData &link_data) {
  return NgpLinkData(link_data);
}

//! \name Link iteration
//@{

/// \brief Run a host function over each link in the link_data that falls in the given selector in parallel.
///
/// The functor must have one of the following signatures:
///   1. operator()(const BulkData &, const Entity &) -> void
///   2. operator()(const BulkData &, const MeshIndex &) -> void
///   3. operator()(const LinkData &, const Entity &) -> void
///   4. operator()(const LinkData &, const MeshIndex &) -> void
template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  if constexpr (std::is_invocable_v<FunctionToRunPerLink, const BulkData &, const stk::mesh::Entity &>) {
    // Just use the standard for_each_entity_run
    ::mundy::mesh::for_each_entity_run(link_data.bulk_data(), link_data.link_meta_data().link_rank(),
                                       linker_subset_selector & link_data.link_meta_data().universal_link_part(),
                                       functor);
  } else {
    // Use the link data version
    const stk::mesh::BucketVector &buckets =
        link_data.bulk_data().get_buckets(link_data.link_meta_data().link_rank(),
                                          linker_subset_selector & link_data.link_meta_data().universal_link_part());
    const size_t num_buckets = buckets.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < num_buckets; j++) {
      stk::mesh::Bucket *bucket = buckets[j];
      const size_t bucket_size = bucket->size();
      for (size_t i = 0; i < bucket_size; i++) {
        if constexpr (std::is_invocable_v<FunctionToRunPerLink, const stk::mesh::BulkData &,
                                          const stk::mesh::MeshIndex &>) {
          functor(link_data, stk::mesh::MeshIndex({bucket, i}));
        } else {
          functor(link_data, (*bucket)[i]);
        }
      }
    }
  }
}

/// \brief Run a host function over each link in the link_data in parallel.
/// See for_each_link_run(const LinkData &, const stk::mesh::Selector &, const FunctionToRunPerLink &)
template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const FunctionToRunPerLink &functor) {
  for_each_link_run(link_data, link_data.link_meta_data().universal_link_part(), functor);
}

/// \brief Run an ngp-compatible function over each link in the ngp_link_data that falls in the given selector in
/// parallel.
///
/// Thread parallel over each link.
///
/// The functor must have the following signature:
///   operator()(const FastMeshIndex &) -> void
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(
      ngp_link_data.ngp_mesh(), ngp_link_data.link_rank(),
      ngp_link_data.host_link_data().link_meta_data().universal_link_part() & linker_subset_selector, functor);
}

/// \brief Run an ngp-compatible function over each link in the ngp_link_data in parallel.
/// See for_each_link_run(const NgpLinkData &, const stk::mesh::Selector &, const FunctionToRunPerLink &)
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const FunctionToRunPerLink &functor) {
  for_each_link_run(ngp_link_data, ngp_link_data.host_link_data().link_meta_data().universal_link_part(), functor);
}
//@}

//! \name Linked entity iteration
//@{

/// \brief Run a host function over each linked entity in the given selector and each of its connected links in
/// parallel.
///
/// Thread parallel over each linked entity. Serial over each link.
///
/// Same function signature as for_each_link_run.
///
/// Must call link_data.propagate_updates() before using this function.
template <typename FunctionToRunPerPairOfLinkedEntityAndLink>
void for_each_linked_entity_run(const LinkData &link_data, const stk::mesh::Selector &linked_entity_selector,
                                const stk::mesh::Selector &linker_subset_selector,
                                const FunctionToRunPerPairOfLinkedEntityAndLink &functor) {
  // TODO(palmerb4): Reimplement this function. using the static CRS data structure.
  //   // Procedure:
  //   //  1. If in debug, validate that all links are up to date
  //   //  2. Loop over each linker partition in serial and check if the partition is in the given selector
  //   //  3. Loop over each linked bucket in parallel
  //   //  4. For each entity in the bucket, loop over each of its linked entities in serial and evaluate the functor

#ifndef NDEBUG
  for_each_link_run(link_data, [](const LinkData &link_data, const stk::mesh::Entity &linker) {
    const auto &link_needs_updated = link_data.link_meta_data().link_crs_needs_updated_field();
    MUNDY_THROW_ASSERT(!stk::mesh::field_data(link_needs_updated, linker)[0], std::logic_error,
                       "Linker is out of sync with its linked entities. Make sure to call propagate_updates()"
                       "before using the for_each_linked_entity_run() function.");
  });
#endif

  // use partition_begin() and partition_end() to loop over each partition
  for (auto it = link_data.partition_begin(); it != link_data.partition_end(); ++it) {
    const PartitionKey &partition_key = it->first;
    const LinkPartition &link_partition = it->second;

    // Get the selector for this partition
    stk::mesh::PartVector link_parts(partition_key.size());
    for (size_t i = 0; i < partition_key.size(); ++i) {
      link_parts[i] = &link_data.mesh_meta_data().get_part(partition_key[i]);
    }
    stk::mesh::Selector partition_subset = stk::mesh::selectIntersection(link_parts) & linker_subset_selector;

    // Get the link buckets this selects
    stk::mesh::BucketVector link_buckets_in_subset =
        link_data.bulk_data().get_buckets(link_data.link_meta_data().link_rank(), partition_subset);
    if (link_buckets_in_subset.empty()) {
      continue;  // No link buckets in this partition are in the given selector
    }

    // Get each bucket in the linked_partition_conn keys that is in the given selector.
    stk::mesh::BucketVector linked_buckets_in_subset;
    for (auto it = link_partition.linked_bucket_begin(); it != link_partition.linked_bucket_end(); ++it) {
      stk::mesh::Bucket *linked_bucket = (*it).first;
      if (linked_entity_selector(linked_bucket)) {
        linked_buckets_in_subset.push_back(linked_bucket);
      }
    }

    // Loop over each linked bucket in parallel
    size_t num_linked_buckets = linked_buckets_in_subset.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < num_linked_buckets; ++i) {
      stk::mesh::Bucket *linked_bucket = linked_buckets_in_subset[i];

      MUNDY_THROW_ASSERT(link_partition.connects_to(*linked_bucket), std::logic_error,
                         "Bug: Linked bucket not found in linked partition conn map.");
      const impl::LinkedBucketConn &linked_bucket_conn = link_partition.get_linked_bucket_conn(*linked_bucket);
      size_t linked_bucket_size = linked_bucket->size();
      for (size_t j = 0; j < linked_bucket_size; ++j) {
        stk::mesh::Entity linked_entity = (*linked_bucket)[j];
        using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
        ConnectedEntities connected_links = linked_bucket_conn.get_connected_entities(j);
        unsigned num_connected_links = connected_links.size();

        // Loop over each of the linked entities in serial and evaluate the functor
        for (unsigned k = 0; k < num_connected_links; ++k) {
          stk::mesh::Entity linker = connected_links[k];

          // Two options for functors. Accepts BulkData/LinkData.
          if constexpr (std::is_invocable_v<FunctionToRunPerPairOfLinkedEntityAndLink, const stk::mesh::BulkData &,
                                            const stk::mesh::Entity &, const stk::mesh::Entity &>) {
            functor(link_data.bulk_data(), linked_entity, linker);
          } else {
            functor(link_data, linked_entity, linker);
          }
        }
      }
    }
  }
}

/// \brief Run a host function over each linked entity in the link_data in parallel.
/// See for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
/// const FunctionToRunPerLinkedEntity &)
template <typename FunctionToRunPerLinkedEntity>
void for_each_linked_entity_run(const LinkData &link_data, const FunctionToRunPerLinkedEntity &functor) {
  for_each_linked_entity_run(link_data, link_data.bulk_data().mesh_meta_data().universal_part(),  //
                             link_data.link_meta_data().universal_link_part(), functor);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDATA_HPP_
