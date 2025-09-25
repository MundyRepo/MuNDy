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

#ifndef MUNDY_MESH_NGPLINKDATA_HPP_
#define MUNDY_MESH_NGPLINKDATA_HPP_

/// \file LinkData.hpp
/// \brief Declaration of the LinkData class

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>  // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>   // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>    // stk::mesh::Part
#include <stk_mesh/base/Types.hpp>   // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_core/throw_assert.hpp>                  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_mesh/LinkCOOData.hpp>                   // for mundy::mesh::LinkCOOData/NgpLinkCOOData
#include <mundy_mesh/LinkCRSData.hpp>                   // for mundy::mesh::LinkCRSData/NgpLinkCRSData
#include <mundy_mesh/LinkData.hpp>                      // for mundy::mesh::LinkData
#include <mundy_mesh/LinkMetaData.hpp>                  // for mundy::mesh::LinkMetaData
#include <mundy_mesh/MetaData.hpp>                      // for mundy::mesh::MetaData
#include <mundy_mesh/Types.hpp>                         // for mundy::mesh::NgpDataAccessTag
#include <mundy_mesh/impl/NgpCOOToCRSSynchronizer.hpp>  // for mundy::mesh::impl::NgpCOOToCRSSynchronizerT

namespace mundy {

namespace mesh {

template <typename NgpMemSpace>
class NgpLinkDataT {
  static_assert(Kokkos::is_memory_space_v<NgpMemSpace>,
                "NgpLinkDataT: The NgpMemSpace template parameter must be a Kokkos memory space such as "
                "stk::ngp::HostMemSpace or "
                "stk::ngp::MemSpace.");

 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NgpLinkDataT() = default;

  /// \brief Default copy or move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataT(const NgpLinkDataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataT(NgpLinkDataT &&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataT &operator=(const NgpLinkDataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataT &operator=(NgpLinkDataT &&) = default;

  /// \brief Canonical constructor.
  /// \param link_data The host link data to mirror in the given memory space.
  explicit NgpLinkDataT(const LinkData &link_data)
      : link_data_ptr_(const_cast<LinkData *>(&link_data)),
        bulk_data_ptr_(&link_data_ptr_->bulk_data()),
        mesh_meta_data_ptr_(&link_data_ptr_->bulk_data().mesh_meta_data()),
        link_meta_data_ptr_(&link_data_ptr_->link_meta_data()),
        link_rank_(link_data_ptr_->link_rank()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(link_data_ptr_->bulk_data())),
        ngp_crs_data_(link_data_ptr_->crs_data()),
        ngp_coo_data_(link_data_ptr_->coo_data()) {
  }

  /// \brief Destructor.
  virtual ~NgpLinkDataT() = default;
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

  /// \brief Fetch our link data on the host.
  LinkData &link_data() {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }

  /// \brief Fetch the link rank
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank link_rank() const noexcept {
    return link_rank_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh &ngp_mesh() const noexcept {
    return ngp_mesh_;
  }
  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh &ngp_mesh() noexcept {
    return ngp_mesh_;
  }
  //@}

  //! \name CRS interface
  //@{

  KOKKOS_FUNCTION
  NgpLinkCRSDataT<NgpMemSpace> &crs_data() noexcept {
    return ngp_crs_data_;
  }
  KOKKOS_FUNCTION
  const NgpLinkCRSDataT<NgpMemSpace> &crs_data() const noexcept {
    return ngp_crs_data_;
  }
  void crs_modify_on_host() {
    link_data().crs_modify_on_host();
  }
  void crs_modify_on_device() {
    link_data().crs_modify_on_device();
  }
  void crs_need_sync_to_host() const {
    link_data().crs_need_sync_to_host();
  }
  void crs_need_sync_to_device() const {
    link_data().crs_need_sync_to_device();
  }
  void crs_sync_to_host() {
    link_data().crs_sync_to_host();
  }
  void crs_sync_to_device() {
    link_data().crs_sync_to_device();
  }
  void crs_clear_host_sync_state() {
    link_data().crs_clear_host_sync_state();
  }
  void crs_clear_device_sync_state() {
    link_data().crs_clear_device_sync_state();
  }
  //@}

  //! \name COO interface
  //@{

  KOKKOS_FUNCTION
  NgpLinkCOODataT<NgpMemSpace> &coo_data() noexcept {
    return ngp_coo_data_;
  }
  KOKKOS_FUNCTION
  const NgpLinkCOODataT<NgpMemSpace> &coo_data() const noexcept {
    return ngp_coo_data_;
  }
  void coo_modify_on_host() {
    link_data().coo_modify_on_host();
  }
  void coo_modify_on_device() {
    link_data().coo_modify_on_device();
  }
  void coo_need_sync_to_host() const {
    link_data().coo_need_sync_to_host();
  }
  void coo_need_sync_to_device() const {
    link_data().coo_need_sync_to_device();
  }
  void coo_sync_to_host() {
    link_data().coo_sync_to_host();
  }
  void coo_sync_to_device() {
    link_data().coo_sync_to_device();
  }
  void coo_clear_host_sync_state() {
    link_data().coo_clear_host_sync_state();
  }
  void coo_clear_device_sync_state() {
    link_data().coo_clear_device_sync_state();
  }
  //@}

  //! \name CRS/COO interactions
  //@{

  /// \brief Check if the CRS connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CRS buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  bool is_crs_up_to_date(const stk::mesh::Selector &selector) {
    return impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::is_crs_up_to_date(ngp_crs_data_, ngp_coo_data_, selector);
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  bool is_crs_up_to_date() {
    return impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::is_crs_up_to_date(ngp_crs_data_, ngp_coo_data_);
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  void update_crs_from_coo(const stk::mesh::Selector &selector) {
    impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::update_crs_from_coo(ngp_crs_data_, ngp_coo_data_, selector);
    crs_modify_on_device();
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  void update_crs_from_coo() {
    impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::update_crs_from_coo(ngp_crs_data_, ngp_coo_data_);
    crs_modify_on_device();
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  void check_crs_coo_consistency(const stk::mesh::Selector &selector) {
    impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::check_crs_coo_consistency(ngp_crs_data_, ngp_coo_data_, selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  void check_crs_coo_consistency() {
    impl::NgpCOOToCRSSynchronizerT<NgpMemSpace>::check_crs_coo_consistency(ngp_crs_data_, ngp_coo_data_);
  }

  /// \brief Rectify potentially stale data post-mesh modification.
  void update_post_mesh_mod() {
    link_data().update_post_mesh_mod();
    ngp_mesh_.update_mesh();
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
  //! \name Internal members (host only)
  //@{

  LinkData *link_data_ptr_;
  stk::mesh::BulkData *bulk_data_ptr_;
  stk::mesh::MetaData *mesh_meta_data_ptr_;
  LinkMetaData *link_meta_data_ptr_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  stk::mesh::EntityRank link_rank_;
  stk::mesh::NgpMesh ngp_mesh_;
  NgpLinkCRSDataT<NgpMemSpace> ngp_crs_data_;
  NgpLinkCOODataT<NgpMemSpace> ngp_coo_data_;
  //@}
};  // NgpLinkDataT

using NgpLinkData = NgpLinkDataT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPLINKDATA_HPP_
