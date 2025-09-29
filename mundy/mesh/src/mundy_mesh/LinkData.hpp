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
#include <any>     // for std::any
#include <memory>  // for std::shared_ptr, std::unique_ptr

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // stk::mesh::Part
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_core/throw_assert.hpp>                 // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkCOOData.hpp>                  // for mundy::mesh::LinkCOOData/NgpLinkCOOData
#include <mundy_mesh/LinkCRSData.hpp>                  // for mundy::mesh::LinkCRSData/NgpLinkCRSData
#include <mundy_mesh/LinkMetaData.hpp>                 // for mundy::mesh::LinkMetaData
#include <mundy_mesh/Types.hpp>                        // for mundy::mesh::NgpDataAccessTag
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>  // for mundy::mesh::impl::HostDeviceSynchronizer

namespace mundy {

namespace mesh {

class LinkData;
namespace impl {
std::any &get_ngp_link_data(const LinkData &link_data);
void set_coo_synchronizer(const LinkData &link_data, std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer);
void set_crs_synchronizer(const LinkData &link_data, std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer);
}  // namespace impl

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
/// BulkData's connectivity interface while allowing multiple LinkData objects to be used on the same mesh, each
/// with separately managed data. Use LinkData to connect links to linked entities and to get the linked entities
/// for a given link. And similar to STK's for_each_entity_run, use non-member functions acting on the LinkData to
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
/// requires the link data to be "up-to-date" (i.e., you must call link_data.update_crs_from_coo() before using it
/// if you have modified the link data).
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
class LinkData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  LinkData() = default;

  /// \brief Default copy or move constructors/operators.
  LinkData(const LinkData &) = default;
  LinkData(LinkData &&) = default;
  LinkData &operator=(const LinkData &) = default;
  LinkData &operator=(LinkData &&) = default;

  /// \brief Canonical constructor.
  /// \param bulk_data [in] The bulk data manager we extend.
  /// \param link_meta_data [in] Our meta data manager.
  LinkData(stk::mesh::BulkData &bulk_data,
           LinkMetaData &link_meta_data)  // We do NOT take ownership of the LinkMetaData
      : bulk_data_ptr_(&bulk_data),
        mesh_meta_data_ptr_(&bulk_data.mesh_meta_data()),
        link_meta_data_ptr_(&link_meta_data),
        coo_data_(bulk_data, link_meta_data),
        crs_data_(bulk_data, link_meta_data),
        coo_synchronizer_(nullptr),
        crs_synchronizer_(nullptr),
        any_ngp_link_data_(),
        crs_modified_on_host_(false),
        crs_modified_on_device_(false),
        coo_modified_on_host_(false),
        coo_modified_on_device_(false),
        crs_num_syncs_to_host_(0),
        crs_num_syncs_to_device_(0),
        coo_num_syncs_to_host_(0),
        coo_num_syncs_to_device_(0) {
  }

  /// \brief Destructor.
  virtual ~LinkData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the link data is valid.
  bool is_valid() const {
    return mesh_meta_data_ptr_ != nullptr && link_meta_data_ptr_ != nullptr && bulk_data_ptr_ != nullptr;
  }

  /// \brief Fetch the bulk data's meta data manager
  const stk::mesh::MetaData &mesh_meta_data() const {
    MUNDY_THROW_ASSERT(mesh_meta_data_ptr_ != nullptr, std::invalid_argument, "Mesh meta data is not set.");
    return *mesh_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data's meta data manager
  stk::mesh::MetaData &mesh_meta_data() {
    MUNDY_THROW_ASSERT(mesh_meta_data_ptr_ != nullptr, std::invalid_argument, "Mesh meta data is not set.");
    return *mesh_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  const LinkMetaData &link_meta_data() const {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the link meta data manager
  LinkMetaData &link_meta_data() {
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
    return *link_meta_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  const stk::mesh::BulkData &bulk_data() const {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the bulk data manager we extend
  stk::mesh::BulkData &bulk_data() {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    return *bulk_data_ptr_;
  }

  /// \brief Fetch the link rank
  stk::mesh::EntityRank link_rank() const noexcept {
    return link_meta_data().link_rank();
  }
  //@}

  //! \name CRS interface
  //@{

  LinkCRSData &crs_data() noexcept {
    return crs_data_;
  }
  const LinkCRSData &crs_data() const noexcept {
    return crs_data_;
  }
  void crs_modify_on_host() {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        "The host CRS is a read-only copy of the device CRS and may not be modified directly.");
    if (crs_has_device_data()) {
      crs_synchronizer_->modify_on_host();
    }
  }
  void crs_modify_on_device() {
    MUNDY_THROW_REQUIRE(crs_modified_on_host_ == false, std::invalid_argument,
                        "The device CRS may not be modified while the host CRS is modified."
                        "Either sync the host CRS to device or clear the host modification state.");
    crs_modified_on_device_ = true;
    if (crs_has_device_data()) {
      crs_synchronizer_->modify_on_device();
    }
  }
  bool crs_need_sync_to_host() const {
    return crs_modified_on_device_;
  }
  bool crs_need_sync_to_device() const {
    return crs_modified_on_host_;
  }
  void crs_sync_to_host() {
    if (crs_need_sync_to_host()) {
      if (crs_has_device_data()) {
        crs_synchronizer_->sync_to_host();
      } else {
        MUNDY_THROW_REQUIRE(
            false, std::logic_error,
            "Why has sync_crs_to_host been called on a LinkData with no device CRS that somehow needs synced to host?");
      }
      crs_increment_num_syncs_to_host();
      crs_clear_device_sync_state();
    }
  }
  void crs_sync_to_device() {
    if (crs_need_sync_to_device()) {
      if (crs_has_device_data()) {
        crs_synchronizer_->update_post_mesh_mod();
        crs_synchronizer_->sync_to_device();
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error,
                            "Why has crs_sync_to_device been called on a LinkData with no device CRS that somehow "
                            "needs synced to host?");
      }
      crs_increment_num_syncs_to_device();
      crs_clear_host_sync_state();
    }
  }
  void crs_clear_host_sync_state() {
    crs_modified_on_host_ = false;
  }
  void crs_clear_device_sync_state() {
    crs_modified_on_device_ = false;
  }
  bool crs_has_device_data() const {
    return crs_synchronizer_ != nullptr;
  }
  void crs_increment_num_syncs_to_host() {
    ++crs_num_syncs_to_host_;
  }
  void crs_increment_num_syncs_to_device() {
    ++crs_num_syncs_to_device_;
  }
  //@}

  //! \name COO interface
  //@{

  LinkCOOData &coo_data() noexcept {
    return coo_data_;
  }
  const LinkCOOData &coo_data() const noexcept {
    return coo_data_;
  }
  void coo_modify_on_host() {
    MUNDY_THROW_REQUIRE(coo_modified_on_device_ == false, std::invalid_argument,
                        "The host COO may not be modified while the device COO is also modified."
                        "Either sync the device COO to host or clear the device modification state.");
    coo_modified_on_host_ = true;
    if (coo_has_device_data()) {
      coo_synchronizer_->modify_on_host();
    }
  }
  void coo_modify_on_device() {
    MUNDY_THROW_REQUIRE(coo_modified_on_host_ == false, std::invalid_argument,
                        "The device COO may not be modified while the host COO is also modified."
                        "Either sync the host COO to device or clear the host modification state.");
    coo_modified_on_device_ = true;
    if (coo_has_device_data()) {
      coo_synchronizer_->modify_on_device();
    }
  }
  bool coo_need_sync_to_host() const {
    return coo_modified_on_device_;
  }
  bool coo_need_sync_to_device() const {
    return coo_modified_on_host_;
  }
  void coo_sync_to_host() {
    if (coo_need_sync_to_host()) {
      if (coo_has_device_data()) {
        coo_synchronizer_->sync_to_host();
      } else {
        MUNDY_THROW_REQUIRE(
            false, std::logic_error,
            "sync_coo_to_host been called on a LinkData with no device COO that is somehow marked modified.");
      }
      coo_increment_num_syncs_to_host();
      coo_clear_device_sync_state();
    }
  }
  void coo_sync_to_device() {
    if (coo_need_sync_to_device()) {
      if (coo_has_device_data()) {
        coo_synchronizer_->update_post_mesh_mod();
        coo_synchronizer_->sync_to_device();
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error,
                            "coo_sync_to_device been called on a LinkData with no device COO.");
      }
      coo_increment_num_syncs_to_device();
      coo_clear_host_sync_state();
    }
  }
  void coo_clear_host_sync_state() {
    coo_modified_on_host_ = false;
  }
  void coo_clear_device_sync_state() {
    coo_modified_on_device_ = false;
  }
  bool coo_has_device_data() const {
    return coo_synchronizer_ != nullptr;
  }
  void coo_increment_num_syncs_to_host() {
    ++coo_num_syncs_to_host_;
  }
  void coo_increment_num_syncs_to_device() {
    ++coo_num_syncs_to_device_;
  }
  //@}

  //! \name CRS/COO interactions
  //@{

  /// \brief Rectify potentially stale data post-mesh modification.
  void update_post_mesh_mod() {
    coo_synchronizer_->update_post_mesh_mod();
    crs_synchronizer_->update_post_mesh_mod();
  }
  //@}

  //! \name Declaration/destruction requests
  //@{

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

 private:
  //! \name Internal methods
  //@{

  std::any &get_ngp_link_data() const {
    return any_ngp_link_data_;
  }

  void set_coo_synchronizer(std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer) const {
    coo_synchronizer_ = std::move(synchronizer);
  }

  void set_crs_synchronizer(std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer) const {
    crs_synchronizer_ = std::move(synchronizer);
  }
  //@}

  //! \name Friends <3
  //@{

  friend std::any &impl::get_ngp_link_data(const LinkData &link_data);
  friend void impl::set_coo_synchronizer(const LinkData &link_data,
                                         std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer);
  friend void impl::set_crs_synchronizer(const LinkData &link_data,
                                         std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer);
  //@}

  //! \name Internal members
  //@{

  stk::mesh::BulkData *bulk_data_ptr_;
  stk::mesh::MetaData *mesh_meta_data_ptr_;
  LinkMetaData *link_meta_data_ptr_;

  LinkCOOData coo_data_;
  LinkCRSData crs_data_;
  mutable std::unique_ptr<impl::HostDeviceSynchronizer> coo_synchronizer_;
  mutable std::unique_ptr<impl::HostDeviceSynchronizer> crs_synchronizer_;
  mutable std::any any_ngp_link_data_;
  mutable bool crs_modified_on_host_;
  mutable bool crs_modified_on_device_;
  mutable bool coo_modified_on_host_;
  mutable bool coo_modified_on_device_;
  mutable size_t crs_num_syncs_to_host_;
  mutable size_t crs_num_syncs_to_device_;
  mutable size_t coo_num_syncs_to_host_;
  mutable size_t coo_num_syncs_to_device_;
  //@}
};

namespace impl {
inline std::any &get_ngp_link_data(const LinkData &link_data) {
  return link_data.get_ngp_link_data();
}

inline void set_crs_synchronizer(const LinkData &link_data,
                                 std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer) {
  link_data.set_crs_synchronizer(std::move(synchronizer));
}

inline void set_coo_synchronizer(const LinkData &link_data,
                                 std::unique_ptr<impl::HostDeviceSynchronizer> synchronizer) {
  link_data.set_coo_synchronizer(std::move(synchronizer));
}

}  // namespace impl

/// \brief Declare a new LinkData object.
///
/// Note, this is the de-facto constructor for LinkData. In the future, we will have it return a reference to
/// the constructed LinkData object by tying its lifetime to the BulkData object.
///
/// \param bulk_data [in] The bulk data manager we extend.
/// \param link_meta_data [in] Our meta data manager. Must be persistant with a lifetime at least as long as the
///   generated LinkData.
/// \return A new LinkData object.
LinkData declare_link_data(stk::mesh::BulkData &bulk_data, LinkMetaData &link_meta_data) {
  // TODO(palmerb4): Store a map of LinkData objects in the BulkData object using an attribute
  //  this way, we can tie the lifetime of the LinkData object to the BulkData object and return
  //  a reference here.
  return LinkData(bulk_data, link_meta_data);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDATA_HPP_
