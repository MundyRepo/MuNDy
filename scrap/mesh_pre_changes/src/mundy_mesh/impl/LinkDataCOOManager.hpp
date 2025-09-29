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

#ifndef MUNDY_MESH_NEW_LINKDATACOOMANAGER_HPP_
#define MUNDY_MESH_NEW_LINKDATACOOMANAGER_HPP_

/// \file LinkDataCOOManager.hpp
/// \brief Declaration of the LinkDataCOOManager class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <Kokkos_Sort.hpp>                        // for Kokkos::sort
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
#include <stk_util/ngp/NgpSpaces.hpp>             // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy libs
#include <mundy_core/throw_assert.hpp>        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>            // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>       // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>            // for mundy::mesh::MetaData
#include <mundy_mesh/NewLinkMetaData.hpp>     // for mundy::mesh::NewLinkMetaData
#include <mundy_mesh/NewNgpCRSPartition.hpp>  // for mundy::mesh::NewNgpCRSPartition
#include <mundy_mesh/NewNgpLinkMetaData.hpp>  // for mundy::mesh::NewNgpLinkMetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>        // for mundy::mesh::field_copy
#include <mundy_mesh/GetNgpLinkData.hpp>      // for mundy::mesh::get_updated_ngp_link_data

namespace stk {
namespace ngp {
using HostMemSpace = stk::ngp::HostExecSpace::memory_space;
using MemSpace = stk::ngp::ExecSpace::memory_space;
}  // namespace ngp
}  // namespace stk

namespace mundy {

namespace mesh {

namespace impl {

struct NgpLinkDataCOOManagerBase {
  virtual ~NgpLinkDataCOOManagerBase() = default;
  virtual void sync_to_host() = 0;
  virtual void sync_from_host() = 0;
};

template <typename LinkDataType, typename NgpMemSpace>
class NgpLinkDataCOOManagerT;

/// \brief Link data CRS manager
///
/// This class manages the fact that ngp CRS partition managers may exist in multiple memory spaces (e.g., host and
/// device). It ensures that they all view the same host data and that they can be updated independently and
/// synchronized upon request.
///
/// Data-wise, it contains a map from memory space to
template <typename LinkDataType>
class LinkDataCOOManager {
 public:
  //! \name Aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

  using CRSManagerBasePtr = std::shared_ptr<NgpLinkDataCOOManagerBase>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  LinkDataCOOManager() = default;

  /// \brief Default copy or move constructors/operators.
  LinkDataCOOManager(const LinkDataCOOManager &) = default;
  LinkDataCOOManager(LinkDataCOOManager &&) = default;
  LinkDataCOOManager &operator=(const LinkDataCOOManager &) = default;
  LinkDataCOOManager &operator=(LinkDataCOOManager &&) = default;

  /// \brief Canonical constructor.
  explicit LinkDataCOOManager(LinkDataType *link_data_ptr)
      : link_data_ptr_(link_data_ptr),
        most_up_to_date_crs_manager_ptr_(nullptr),
        crs_manager_map_(),
        host_crs_manager_ptr_(
            std::make_shared<NgpLinkDataCOOManagerT<LinkDataType, stk::ngp::HostMemSpace>>(link_data_ptr_)) {
    MUNDY_THROW_REQUIRE(link_data_ptr_, std::invalid_argument, "Link data pointer cannot be null.");
    MUNDY_THROW_REQUIRE(host_crs_manager_ptr_, std::invalid_argument, "Host CRS manager pointer cannot be null.");
  }

  /// \brief Destructor.
  virtual ~LinkDataCOOManager() {std::cout << "DESTRUCTOR FOR LinkDataCOOManager" << std::endl;}
  //@}

  //! \name Public methods
  //@{

  /// \brief Get our link data manager.
  const LinkDataType &link_data() const {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }
  LinkDataType &link_data() {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }

  /// \brief Get or create CRS manager for a given memory space.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  CRSManagerBasePtr get_or_create_crs_manager(const NgpMemSpace &mem_space = NgpMemSpace{}) {
    auto it = crs_manager_map_.find(std::type_index(typeid(NgpMemSpace)));
    if(it != crs_manager_map_.end()) {
      return it->second;
    } else {
      CRSManagerBasePtr crs_manager_ptr = std::make_shared<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace>>(link_data_ptr_);
      crs_manager_map_[std::type_index(typeid(NgpMemSpace))] = crs_manager_ptr;
      return crs_manager_map_[std::type_index(typeid(NgpMemSpace))];
    }
  }

  /// \brief Get all CRS partitions (Memoized/host only/not thread safe).
  ///
  /// Safety: This view is always safe to use and the reference is valid as long as the LinkDataType is valid, which
  /// has the same lifetime as the bulk data manager. It remains valid even during mesh modifications.
  ///
  /// Stale State: While always valid, if you call update_crs_from_coo(some_other_mem_space), only the partitions
  /// within that memory space will be updated. All others will remain as is. If you want to ensure that your space
  /// is up to date without performing an update, call synchronize_crs_partitions(mem_space).
  ///
  /// \tparam NgpMemSpace The memory space within which the crs connectivity data will reside.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  const NgpCRSPartitionViewT<NgpMemSpace> &get_all_crs_partitions(
      const NgpMemSpace &mem_space = NgpMemSpace{}) {
    const CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    return dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->get_all_crs_partitions();
  }

  /// \brief Get the CRS partitions for a given link subset selector (Memoized/host only/not thread safe).
  ///
  /// This is the only way for either mundy or users to create a new partition. The returned view is persistent
  /// but its contents/size will change dynamically as new partitions are created and destroyed. The only promise
  /// we will make is to never delete a partition outside of a modification cycle.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  const NgpCRSPartitionViewT<NgpMemSpace> &get_or_create_crs_partitions(
      const stk::mesh::Selector &selector, const NgpMemSpace &mem_space = NgpMemSpace{}) {
    const CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    return dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->get_or_create_crs_partitions(selector);
  }

  /// \brief Check if the CRS connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CRS buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  bool is_crs_up_to_date(const stk::mesh::Selector &selector, const NgpMemSpace &mem_space = NgpMemSpace{}) {
    const CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    MUNDY_THROW_REQUIRE(crs_manager_ptr, std::runtime_error, "CRS manager pointer is null.");
    bool our_mem_space_is_up_to_date = (most_up_to_date_crs_manager_ptr_ == crs_manager_ptr);
    return our_mem_space_is_up_to_date &&
           dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
               ->is_crs_up_to_date(selector);
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  bool is_crs_up_to_date(const NgpMemSpace &mem_space = NgpMemSpace{}) {
    const CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    bool our_mem_space_is_up_to_date = (most_up_to_date_crs_manager_ptr_ == crs_manager_ptr);
    return our_mem_space_is_up_to_date &&
           dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
               ->is_crs_up_to_date();
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  void update_crs_from_coo(const stk::mesh::Selector &selector, const NgpMemSpace &mem_space = NgpMemSpace{}) {
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->update_crs_from_coo(selector);
    most_up_to_date_crs_manager_ptr_ = crs_manager_ptr;
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  void update_crs_from_coo(const NgpMemSpace &mem_space = NgpMemSpace{}) {
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())->update_crs_from_coo();
    most_up_to_date_crs_manager_ptr_ = crs_manager_ptr;
  }

  /// \brief Synchronize the CRS connectivity with the most up-to-date memory space.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  void sync(const NgpMemSpace &mem_space = NgpMemSpace{}) {
    // Get the crs manager for the given mem space. If it isn't the most up-to-date manager, sync from the most
    // up-to-date to host, then sync from host to the requested memory space.
    //
    // This is done because we are unable to store the memory space of the most up-to-date manager, so we need
    // a common ground to sync through.
    MUNDY_THROW_REQUIRE(most_up_to_date_crs_manager_ptr_, std::runtime_error,
                        "No memory space has up-to-date CRS connectivity. Call update_crs_from_coo() in one memory "
                        "space before calling sync().");
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    if (crs_manager_ptr != most_up_to_date_crs_manager_ptr_) {
      most_up_to_date_crs_manager_ptr_->sync_to_host();
      crs_manager_ptr->sync_from_host();
    }
  }

  /// \brief Get the map from link bucket id to partition id (memoized/host only/not thread safe).
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  LinkBucketToPartitionIdMapT<NgpMemSpace> get_updated_stk_link_bucket_to_partition_id_map(
      const NgpMemSpace &mem_space = NgpMemSpace{}) {
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    return dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->get_updated_stk_link_bucket_to_partition_id_map(mem_space);
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  void check_crs_coo_consistency(const stk::mesh::Selector &selector, const NgpMemSpace &mem_space = NgpMemSpace{}) {
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->check_crs_coo_consistency(selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  template <typename NgpMemSpace = stk::ngp::HostMemSpace>
    requires(Kokkos::is_memory_space_v<NgpMemSpace>)
  void check_crs_coo_consistency(const NgpMemSpace &mem_space = NgpMemSpace{}) {
    CRSManagerBasePtr crs_manager_ptr = get_or_create_crs_manager(mem_space);
    dynamic_cast<NgpLinkDataCOOManagerT<LinkDataType, NgpMemSpace> *>(crs_manager_ptr.get())
        ->check_crs_coo_consistency();
  }
  //@}

 private:
  LinkDataType *link_data_ptr_ = nullptr;
  CRSManagerBasePtr most_up_to_date_crs_manager_ptr_ = nullptr;
  std::map<std::type_index, CRSManagerBasePtr> crs_manager_map_;
  std::shared_ptr<NgpLinkDataCOOManagerT<LinkDataType, stk::ngp::HostMemSpace>> host_crs_manager_ptr_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename LinkDataType, typename NgpMemSpace>
class NgpLinkDataCOOManagerT : public NgpLinkDataCOOManagerBase {
 public:
  //! \name Aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  using NgpCRSPartitionView = Kokkos::View<NewNgpCRSPartitionT<NgpMemSpace> *, stk::ngp::UVMMemSpace>;
  using LinkBucketToPartitionIdMap = Kokkos::UnorderedMap<unsigned, unsigned, NgpMemSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NgpLinkDataCOOManagerT() = default;

  /// \brief Default copy or move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataCOOManagerT(const NgpLinkDataCOOManagerT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataCOOManagerT(NgpLinkDataCOOManagerT &&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataCOOManagerT &operator=(const NgpLinkDataCOOManagerT &) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpLinkDataCOOManagerT &operator=(NgpLinkDataCOOManagerT &&) = default;

  /// \brief Canonical constructor for general NgpMemSpace's
  /// Store the given host view internally and copy it to the device.
  explicit NgpLinkDataCOOManagerT(
      std::shared_ptr<NgpLinkDataCOOManagerT<LinkDataType, stk::ngp::HostMemSpace>> &host_crs_manager_ptr)
      : link_data_ptr_(&host_crs_manager_ptr->link_data()),
      selector_to_partitions_map_(), partition_key_to_id_map_(),
        all_crs_partitions_("AllCRSPartitions", 0),
        stk_link_bucket_to_partition_id_map_(
            host_crs_manager_ptr->get_updated_stk_link_bucket_to_partition_id_map().capacity()),
        host_crs_manager_ptr_(host_crs_manager_ptr) {
    std::cout << "Inside NgpLinkDataCOOManagerT constructor" << std::endl;
    MUNDY_THROW_REQUIRE(host_crs_manager_ptr_, std::invalid_argument, "Host CRS manager pointer is null.");
    Kokkos::deep_copy(stk_link_bucket_to_partition_id_map_,
                      host_crs_manager_ptr_->get_updated_stk_link_bucket_to_partition_id_map());
  }

  /// \brief Directly construct a host NgpLinkDataCOOManagerT.
  explicit NgpLinkDataCOOManagerT(LinkDataType *link_data_ptr)
    requires(std::is_same_v<NgpMemSpace, stk::ngp::HostMemSpace>)
      : link_data_ptr_(link_data_ptr),
      selector_to_partitions_map_(), partition_key_to_id_map_(),
        all_crs_partitions_("AllCRSPartitions", 0),
        stk_link_bucket_to_partition_id_map_(10),
        host_crs_manager_ptr_(nullptr) {
    MUNDY_THROW_REQUIRE(link_data_ptr_, std::invalid_argument, "Link data pointer is null.");
  }

  /// \brief Destructor.
  virtual ~NgpLinkDataCOOManagerT() {std::cout << "DESTRUCTOR FOR NgpLinkDataCOOManagerT" << std::endl;}
  //@}

  //! \name Public methods
  //@{

  /// \brief Get our link data manager.
  const LinkDataType &link_data() const {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }
  LinkDataType &link_data() {
    MUNDY_THROW_ASSERT(link_data_ptr_ != nullptr, std::invalid_argument, "Link data is not set.");
    return *link_data_ptr_;
  }

  void sync_to_host() override {
    if constexpr (!is_host_mem_space_v<NgpMemSpace>) {
      // For each partition in our memory space, loop over each of its buckets and deep copy them to the corresponding
      // host bucket.
      //
      // At this point their number of partitions aren't guaranteed to be the same since get_or_create_crs_partitions
      // may have only been called on one memory space and not the other.
      MUNDY_THROW_REQUIRE(host_crs_manager_ptr_, std::invalid_argument, "Host CRS manager pointer is null.");
      stk::mesh::Selector our_all_selector = all_selector();
      auto &our_crs_partitions = get_all_crs_partitions();
      auto &host_crs_partitions = host_crs_manager_ptr_->get_or_create_crs_partitions(our_all_selector);
      host_crs_manager_ptr_->sort_partitions_by_id();
      sort_partitions_by_id();

      MUNDY_THROW_ASSERT(our_crs_partitions.extent(0) == host_crs_partitions.extent(0), std::logic_error,
                         "Internal error, inform the devs. Number of partitions in our memory space somehow differs "
                         "from the host's number of partitions.");

      for (unsigned partition_id = 0u; partition_id < our_crs_partitions.extent(0); ++partition_id) {
        // The only way the following is true, is if we sort the partition view by partition id.
        auto &our_partition = our_crs_partitions(partition_id);
        auto &host_partition = host_crs_partitions(partition_id);

        MUNDY_THROW_ASSERT(
            our_partition.id() == host_partition.id(), std::logic_error,
            "Internal error, inform the devs. Partition ids mismatch between memory spaces during sync_to_host().");
        MUNDY_THROW_ASSERT(
            our_partition.key() == host_partition.key(), std::logic_error,
            "Internal error, inform the devs. Partition keys mismatch between memory spaces during sync_to_host().");
        MUNDY_THROW_ASSERT(our_partition.link_rank() == host_partition.link_rank(), std::logic_error,
                           "Internal error, inform the devs. Partition link ranks mismatch between memory spaces "
                           "during sync_to_host().");
        MUNDY_THROW_ASSERT(our_partition.link_dimensionality() == host_partition.link_dimensionality(),
                           std::logic_error,
                           "Internal error, inform the devs. Partition link dimensionalities mismatch between memory "
                           "spaces during sync_to_host().");
        for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; rank++) {
          size_t num_buckets = our_partition.num_buckets(rank);
          MUNDY_THROW_ASSERT(our_partition.num_buckets(rank) == host_partition.num_buckets(rank), std::logic_error,
                             "Internal error, inform the devs. Number of link buckets mismatch between memory spaces "
                             "during sync_to_host().");
          for (unsigned i_bucket = 0; i_bucket < num_buckets; ++i_bucket) {
            deep_copy(host_partition.get_crs_bucket_conn(rank, i_bucket),
                      our_partition.get_crs_bucket_conn(rank, i_bucket));
          }
        }
      }
    }
  }

  void sync_from_host() override {
    if constexpr (!is_host_mem_space_v<NgpMemSpace>) {
      // For each partition in the host space, loop over each of its buckets and deep copy them to our corresponding
      // bucket.
      //
      // At this point their number of partitions aren't guaranteed to be the same since get_or_create_crs_partitions
      // may have only been called on one memory space and not the other.
      MUNDY_THROW_REQUIRE(host_crs_manager_ptr_, std::invalid_argument, "Host CRS manager pointer is null.");
      stk::mesh::Selector host_all_selector = host_crs_manager_ptr_->all_selector();
      auto &our_crs_partitions = get_or_create_crs_partitions(host_all_selector);
      auto &host_crs_partitions = host_crs_manager_ptr_->get_all_crs_partitions();
      host_crs_manager_ptr_->sort_partitions_by_id();
      sort_partitions_by_id();

      MUNDY_THROW_ASSERT(our_crs_partitions.extent(0) == host_crs_partitions.extent(0), std::logic_error,
                         "Internal error, inform the devs. Number of partitions in our memory space somehow differs "
                         "from the host's number of partitions.");

      for (unsigned partition_id = 0u; partition_id < our_crs_partitions.extent(0); ++partition_id) {
        // The only way the following is true, is if we sort the partition view by partition id.
        auto &our_partition = our_crs_partitions(partition_id);
        auto &host_partition = host_crs_partitions(partition_id);

        MUNDY_THROW_ASSERT(
            our_partition.id() == host_partition.id(), std::logic_error,
            "Internal error, inform the devs. Partition ids mismatch between memory spaces during sync_to_host().");
        MUNDY_THROW_ASSERT(
            our_partition.key() == host_partition.key(), std::logic_error,
            "Internal error, inform the devs. Partition keys mismatch between memory spaces during sync_to_host().");
        MUNDY_THROW_ASSERT(our_partition.link_rank() == host_partition.link_rank(), std::logic_error,
                           "Internal error, inform the devs. Partition link ranks mismatch between memory spaces "
                           "during sync_to_host().");
        MUNDY_THROW_ASSERT(our_partition.link_dimensionality() == host_partition.link_dimensionality(),
                           std::logic_error,
                           "Internal error, inform the devs. Partition link dimensionalities mismatch between memory "
                           "spaces during sync_to_host().");
        for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; rank++) {
          size_t num_buckets = our_partition.num_buckets(rank);
          MUNDY_THROW_ASSERT(our_partition.num_buckets(rank) == host_partition.num_buckets(rank), std::logic_error,
                             "Internal error, inform the devs. Number of link buckets mismatch between memory spaces "
                             "during sync_to_host().");
          for (unsigned i_bucket = 0; i_bucket < num_buckets; ++i_bucket) {
            deep_copy(our_partition.get_crs_bucket_conn(rank, i_bucket),
                      host_partition.get_crs_bucket_conn(rank, i_bucket));
          }
        }
      }
    }
  }

  /// \brief Check if the CRS connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CRS buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  bool is_crs_up_to_date(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("LinkDataCOOManager::is_crs_up_to_date");

    // Dereference just once
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

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

    // TODO(palmerb4): It appears as though counting the number of dirty buckets in a parallel_for is slower than doing
    // it serially (at least for a CPU build). Is this true for GPU builds too?

    // int num_dirty_buckets = 0;
    // typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    // const auto &team_policy =
    //     stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(partitions.extent(0), Kokkos::AUTO);
    // Kokkos::parallel_reduce(
    //     "LinkDataCOOManager::is_crs_up_to_date", team_policy,
    //     KOKKOS_LAMBDA(const TeamHandleType &team, int &team_local_count) {
    //       const stk::mesh::Ordinal partition_id = team.league_rank();
    //       const NewNgpCRSPartition &partition = partitions(partition_id);

    //       int tmp_team_local_count = 0;

    //       for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    //         const unsigned num_buckets = partition.num_buckets(rank);
    //         int rank_local_count = 0;
    //         Kokkos::parallel_reduce(
    //             Kokkos::TeamThreadRange(team, num_buckets),
    //             [&](const unsigned bucket_index, int &count) {
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
      int link_needs_updated_count = ::mundy::mesh::field_sum<int>(
          link_data().link_meta_data().link_crs_needs_updated_field(), link_subset_selector, stk::ngp::ExecSpace());
      bool links_up_to_date = (link_needs_updated_count == 0);
      return links_up_to_date;
    }

    Kokkos::Profiling::popRegion();
    return crs_buckets_up_to_date;
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  bool is_crs_up_to_date() {
    return is_crs_up_to_date(link_data().bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  void update_crs_from_coo(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    if (is_crs_up_to_date(link_subset_selector)) {
      return;
    }

    Kokkos::Profiling::pushRegion("LinkDataCOOManager::update_crs_from_coo");

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
  void update_crs_from_coo() {
    update_crs_from_coo(link_data().bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Update the map from link bucket id to partition id (host only/not thread safe).
  void update_stk_link_bucket_to_partition_id_map() {
    if constexpr (is_host_mem_space_v<NgpMemSpace>) {
      std::cout << "Host update_stk_link_bucket_to_partition_id_map" << std::endl;
      std::cout << "Is link data valid? " << link_data().is_valid() << std::endl;
      // Get all link buckets that currently have selectors.
      stk::mesh::Selector our_all_selector = all_selector();
      const stk::mesh::BucketVector &all_link_buckets =
          link_data().bulk_data().get_buckets(link_data().link_meta_data().link_rank(), our_all_selector);
      const unsigned num_link_buckets = static_cast<unsigned>(all_link_buckets.size());

      // Resize the map if needed. (only grow)
      if (stk_link_bucket_to_partition_id_map_.capacity() < num_link_buckets) {
        stk_link_bucket_to_partition_id_map_.rehash(num_link_buckets);
      }

      // Loop over each bucket, get its partition key, map the key to an id, and store the id in the map.
      for (const stk::mesh::Bucket *bucket : all_link_buckets) {
        PartitionKey key = get_partition_key(*bucket);

        auto it = partition_key_to_id_map_.find(key);
        if (it != partition_key_to_id_map_.end()) {
          stk::mesh::Ordinal partition_id = it->second;
          bool insert_success =
              stk_link_bucket_to_partition_id_map_.insert(bucket->bucket_id(), partition_id).success();
          MUNDY_THROW_ASSERT(insert_success, std::runtime_error,
                             "Failed to insert bucket -> partition pair into the map. This is an internal error.");
        } else {
          MUNDY_THROW_ASSERT(false, std::logic_error,
                             "Partition key not found in partition key to id map. This should never happen.");
        }
      }
    } else {
      std::cout << "Device update_stk_link_bucket_to_partition_id_map" << std::endl;
      MUNDY_THROW_REQUIRE(host_crs_manager_ptr_, std::runtime_error, "Host CRS manager pointer is null.");
      host_crs_manager_ptr_->update_stk_link_bucket_to_partition_id_map();
      Kokkos::deep_copy(stk_link_bucket_to_partition_id_map_,
                        host_crs_manager_ptr_->stk_link_bucket_to_partition_id_map_);
    }
  }

  /// \brief Get the map from link bucket id to partition id (memoized/host only/not thread safe).
  LinkBucketToPartitionIdMap get_updated_stk_link_bucket_to_partition_id_map() {
    std::cout << "Inside get_updated_stk_link_bucket_to_partition_id_map" << std::endl;
    // If the map is empty, populate it.  MOD MARK: There are so many better ways to do this.
    if (stk_link_bucket_to_partition_id_map_.size() == 0) {
      update_stk_link_bucket_to_partition_id_map();
    }

    return stk_link_bucket_to_partition_id_map_;
  }

  void flag_dirty_linked_buckets_of_modified_links(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::flag_dirty_linked_buckets");

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Flag dirty buckets: Team loop over selected link buckets, fetch their partition, thread loop over links,
    // determine if any of those links are flagged as modified. If so, determine if their links were created or
    // destroyed. Flag the linked bucket of new or deleted entities as dirty.

    auto ngp_link_data = get_updated_ngp_link_data(link_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data().bulk_data());
    const stk::mesh::EntityRank link_rank = link_data().link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "flag_dirty_linked_buckets_of_modified_links", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
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
            if (ngp_link_data.get_link_crs_needs_updated(link_index)) {
              // Loop over the linked entities of this link
              for (unsigned d = 0; d < dimensionality; ++d) {
                stk::mesh::Entity linked_entity_crs = ngp_link_data.get_linked_entity_crs(link_index, d);
                stk::mesh::Entity linked_entity = ngp_link_data.get_linked_entity(link_index, d);
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

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    //  Reset dirty buckets: Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over
    //  the partitions, if its corresponding linked bucket has been modified, thread loop over the linked entities and
    //  reset the connectivity counts.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data_ptr_->bulk_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          "reset_dirty_linked_buckets", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
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

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Gather part 1 (count): Team loop over selected link buckets, fetch their partition, team loop over the links,
    // serial loop over the downward linked entities, if their bucket is dirty, atomically increment the connectivity
    // counts of the downward connected entities.

    auto ngp_link_data = get_updated_ngp_link_data(link_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data().bulk_data());
    const stk::mesh::EntityRank link_rank = link_data().link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "gather_part_1_count", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
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
              if (ngp_link_data.get_linked_entity(link_index, d) != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = ngp_link_data.get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = ngp_link_data.get_linked_entity_rank(link_index, d);
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

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Gather part 2 (partial sum): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop
    // over the partitions, if its corresponding linked bucket has been modified, thread loop over the linked bucket to
    // partial sum the connectivity counts into the connectivity offsets.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data_ptr_->bulk_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          "gather_part_2_partial_sum", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
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
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                      [&](unsigned i, unsigned &partial_sum, bool final_pass) {
                                        const unsigned num_connected_links = crs_bucket_conn.num_connected_links_(i);
                                        if (final_pass) {
                                          // exclusive offset
                                          crs_bucket_conn.sparse_connectivity_offsets_(i) = partial_sum;

                                          if (i == bucket_size - 1) {
                                            // Store the total number of connected links at the end of the offsets array
                                            crs_bucket_conn.sparse_connectivity_offsets_(bucket_size) =
                                                partial_sum + num_connected_links;
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

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

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

  void scatter_part_2_fill(const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_2_fill");

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Scatter part 2 (fill): Team loop over each selected link buckets, fetch
    // their partition ID, thread loop over the links, serial loop over their downward linked entities, and if their
    // bucket is dirty, scatter the link. Copy the link into the old field. Update the count as each entity is inserted.

    auto ngp_link_data = get_updated_ngp_link_data(link_data());
    auto crs_partitions = get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data().bulk_data());
    const stk::mesh::EntityRank link_rank = link_data().link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "scatter_part_2_fill", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
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
              stk::mesh::Entity linked_entity = ngp_link_data.get_linked_entity(link_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = ngp_link_data.get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = ngp_link_data.get_linked_entity_rank(link_index, d);
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

    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Finalize CRS update: Mark all buckets as no longer dirty, mark all selected links are up-to-date, and copy the
    // old COO connectivity to the new COO connectivity (for the given selector)

    // Serial loop over each rank, parallel loop over the stk buckets of said rank, serial loop over the partitions,
    // if its corresponding linked bucket has been modified, reset the dirty flag.
    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data_ptr_->bulk_data());
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
      // Kokkos::parallel_for("finalize_crs_update_reset_dirty_flag",
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
    auto &link_needs_updated_field = link_data().link_meta_data().link_crs_needs_updated_field();
    ::mundy::mesh::field_fill(0, link_needs_updated_field, link_subset_selector, stk::ngp::ExecSpace());
    std::cout << " Mark links up-to-date time: " << timer.seconds() << " seconds" << std::endl;

    // Copy the old COO connectivity to the new COO connectivity
    timer.reset();
    ::mundy::mesh::field_copy<entity_value_t>(link_data().link_meta_data().linked_entities_field(),
                                              link_data().link_meta_data().linked_entities_crs_field(),
                                              link_subset_selector, stk::ngp::ExecSpace());
    std::cout << " Copy old to new COO time: " << timer.seconds() << " seconds" << std::endl;

    Kokkos::Profiling::popRegion();
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  void check_crs_coo_consistency(const stk::mesh::Selector &selector) {
    MUNDY_THROW_REQUIRE(link_data_ptr_, std::invalid_argument, "Link data hasn't been set.");
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;
    check_all_links_in_sync(link_subset_selector);
    check_linked_bucket_conn_size(link_subset_selector);
    check_coo_to_crs_conn(link_subset_selector);
    check_crs_to_coo_conn(link_subset_selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  void check_crs_coo_consistency() {
    MUNDY_THROW_REQUIRE(link_data_ptr_, std::invalid_argument, "Link data hasn't been set.");
    check_crs_coo_consistency(link_data_ptr_->bulk_data().mesh_meta_data().universal_part());
  }

  void check_all_links_in_sync(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;
    int needs_updated_count = field_sum<int>(link_data().link_meta_data().link_crs_needs_updated_field(),
                                             link_subset_selector, stk::ngp::ExecSpace());
    MUNDY_THROW_REQUIRE(needs_updated_count == 0, std::logic_error, "There are still links that are out of sync.");
  }

  void check_linked_bucket_conn_size(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Serial loop over each selected partition. Serial loop over each rank.
    // Assert that the size of the bucket conn is the same as the number of STK buckets of the given rank.
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NewNgpCRSPartition &partition = partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        unsigned num_buckets = partition.num_buckets(rank);
        unsigned num_stk_buckets = link_data_ptr_->bulk_data().buckets(rank).size();
        MUNDY_THROW_REQUIRE(num_buckets == num_stk_buckets, std::logic_error,
                            "The number of linked buckets does not match the number of STK buckets.");
      }
    }
  }

  void check_coo_to_crs_conn(const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Serial loop over each partial, hierarchical parallelism over each link in said selector,
    // serial loop over each of its downward connections, if it is non-empty, fetch their CRS conn,
    // serial loop over each link in the CRS conn, and check if it is the same as the source link.

    auto ngp_link_data = get_updated_ngp_link_data(link_data());
    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data().bulk_data());
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NewNgpCRSPartition &partition = partitions(partition_id);
      const unsigned dimensionality = partition.link_dimensionality();
      stk::mesh::EntityRank link_rank = link_data().link_meta_data().link_rank();

      stk::mesh::for_each_entity_run(
          ngp_mesh, link_rank, partition.selector(), KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &linker_index) {
            // Loop over each linked entity in the linker
            for (unsigned d = 0; d < dimensionality; ++d) {
              stk::mesh::Entity linked_entity = ngp_link_data.get_linked_entity(linker_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                // Fetch the CRS connectivity of the linked entity
                stk::mesh::EntityRank linked_entity_rank = ngp_link_data.get_linked_entity_rank(linker_index, d);
                stk::mesh::FastMeshIndex linked_entity_index = ngp_link_data.get_linked_entity_index(linker_index, d);
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
                      ngp_mesh.fast_mesh_index(connected_links[connected_link_ord]);
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
    stk::mesh::Selector link_subset_selector = link_data().link_meta_data().universal_link_part() & selector;

    // Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over each CRS partition,
    // fetch the corresponding CRS bucket conn, thread loop over the entities in said bucket, serial loop over their
    // connected links, and check if the source entity is linked to the link.

    auto ngp_link_data = get_updated_ngp_link_data(link_data());
    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(link_data().bulk_data());
    const NgpCRSPartitionView &partitions = get_or_create_crs_partitions(link_subset_selector);

    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      stk::NgpVector<unsigned> bucket_ids =
          ngp_mesh.get_bucket_ids(rank, link_data().bulk_data().mesh_meta_data().universal_part());

      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

      Kokkos::parallel_for(
          "check_crs_to_coo_conn", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, bucket_id);
            unsigned num_entities = bucket.size();

            // Serial loop over each partition
            unsigned num_partitions = partitions.extent(0);
            for (unsigned partition_id = 0; partition_id < num_partitions; ++partition_id) {
              const NewNgpCRSPartition &partition = partitions(partition_id);
              const unsigned dimensionality = partition.link_dimensionality();

              // Thread loop over each entity in the bucket
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_entities), [&](const int &i) {
                stk::mesh::Entity entity = bucket[i];
                stk::mesh::FastMeshIndex entity_index = ngp_mesh.fast_mesh_index(entity);

                // Each connected link better be attached to us
                ConnectedEntities connected_links = partition.get_connected_links(rank, entity_index);
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::Entity connected_link = connected_links[connected_link_ord];
                  stk::mesh::FastMeshIndex connected_link_index = ngp_mesh.fast_mesh_index(connected_link);

                  MUNDY_THROW_REQUIRE(connected_link != stk::mesh::Entity(), std::logic_error,
                                      "A connected link in the CRS connectivity is empty.");

                  // Serial loop over each linked entity in the connected link
                  bool found_entity = false;
                  for (unsigned d = 0; d < dimensionality; ++d) {
                    stk::mesh::Entity linked_entity = ngp_link_data.get_linked_entity(connected_link_index, d);
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

  KOKKOS_INLINE_FUNCTION
  static bool fma_equal(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
    return (lhs.bucket_id == rhs.bucket_id) && (lhs.bucket_ord == rhs.bucket_ord);
  }

 private:
  //! \name Internal members (host only)
  //@{

  LinkDataType *link_data_ptr_;

  using SelectorToPartitionsMap = std::map<stk::mesh::Selector, NgpCRSPartitionView>;
  using PartitionKeyToIdMap = std::map<PartitionKey, unsigned>;
  mutable SelectorToPartitionsMap
      selector_to_partitions_map_;  // Maybe we want to use a view here to reduce copy overhead.
  mutable PartitionKeyToIdMap partition_key_to_id_map_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  mutable NgpCRSPartitionView all_crs_partitions_;
  LinkBucketToPartitionIdMap stk_link_bucket_to_partition_id_map_;
  std::shared_ptr<NgpLinkDataCOOManagerT<LinkDataType, stk::ngp::HostMemSpace>> host_crs_manager_ptr_;
  //@}
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_LINKDATACOOMANAGER_HPP_
