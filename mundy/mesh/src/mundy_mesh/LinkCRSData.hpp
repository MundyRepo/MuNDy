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

#ifndef MUNDY_MESH_LINKCRSDATA_HPP_
#define MUNDY_MESH_LINKCRSDATA_HPP_

/// \file LinkCRSData.hpp
/// \brief Declaration of the LinkCRSData class

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
#include <stk_mesh/base/FindRestriction.hpp>      // for stk::mesh::find_restriction
#include <stk_mesh/base/BulkData.hpp>           // for stk::mesh::BulkData

// Mundy libs
#include <mundy_core/throw_assert.hpp>        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkCRSPartition.hpp>    // for mundy::mesh::LinkCRSPartition
#include <mundy_mesh/LinkMetaData.hpp>        // for mundy::mesh::LinkMetaData

namespace mundy {

namespace mesh {

// Forward declare the LinkData
class LinkData;

template <typename MemSpace>
class LinkCRSDataT {  // Raw data in any space
 public:
  //! \name Aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  using LinkCRSPartition = LinkCRSPartitionT<MemSpace>;
  using LinkCRSPartitionView = Kokkos::View<LinkCRSPartition *, stk::ngp::UVMMemSpace>;
  using LinkBucketToPartitionIdMap = Kokkos::UnorderedMap<unsigned, unsigned, MemSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  LinkCRSDataT() = default;

  /// \brief Default copy or move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION LinkCRSDataT(const LinkCRSDataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCRSDataT(LinkCRSDataT &&) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCRSDataT &operator=(const LinkCRSDataT &) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCRSDataT &operator=(LinkCRSDataT &&) = default;

  /// \brief Construct from scratch
  LinkCRSDataT(stk::mesh::BulkData &bulk_data, LinkMetaData &link_meta_data) : 
        bulk_data_ptr_(&bulk_data), 
        link_meta_data_ptr_(&link_meta_data),
        selector_to_partitions_map_(),
        partition_key_to_id_map_(),
        all_crs_partitions_("AllCRSPartitions", 0),
        stk_link_bucket_to_partition_id_map_host_(10),
        stk_link_bucket_to_partition_id_map_(10) {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
  }

  /// \brief Construct from a LinkCRSDataT with a different memory space.
  /// Does NOT perform a deep copy. Simply steals their pointers to the bulk data and meta data.
  ///
  /// This constructor exists so that we can use NgpLinkCRSData(LinkCRSData) to perform a shallow 
  /// copy if NgpMemSpace == MemSpace and this operator otherwise.
  template<typename OtherMemSpace>
  explicit LinkCRSDataT(LinkCRSDataT<OtherMemSpace> &other)
      requires(!std::is_same_v<OtherMemSpace, MemSpace>) :
        bulk_data_ptr_(&other.bulk_data()),
        link_meta_data_ptr_(&other.link_meta_data()),
        selector_to_partitions_map_(),
        partition_key_to_id_map_(),
        all_crs_partitions_("AllCRSPartitions", 0),
        stk_link_bucket_to_partition_id_map_host_(10),
        stk_link_bucket_to_partition_id_map_(10) {
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Bulk data is not set.");
    MUNDY_THROW_ASSERT(link_meta_data_ptr_ != nullptr, std::invalid_argument, "Link meta data is not set.");
  }

  /// \brief Destructor.
  KOKKOS_FUNCTION virtual ~LinkCRSDataT() {
    clear_partitions_and_views();
  } 
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

  //! \name CRS Partition management
  //@{

  /// \brief Get all CRS partitions.
  ///
  /// Safety: This view is always safe to use and the reference is valid as long as the LinkData is valid, which
  /// has the same lifetime as the bulk data manager. It remains valid even during mesh modifications.
  KOKKOS_INLINE_FUNCTION
  const LinkCRSPartitionView &get_all_crs_partitions() const noexcept {
    return all_crs_partitions_;
  }

  /// \brief Get the CRS partitions for a given link subset selector (Memoized/host only/not thread safe).
  ///
  /// This is the only way for either mundy or users to create a new partition. The returned view is persistent
  /// but its contents/size will change dynamically as new partitions are created and destroyed. The only promise
  /// we will make is to never delete a partition outside of a modification cycle.
  ///
  /// \note Other classes perform synchronization of LinkCRSData objects between memory spaces. Importantly, creating
  /// an empty partition as a result of this function is not considered a "modification" and does not dirty
  /// other memory spaces. Instead, we only maintain consistency of populated partitions across memory spaces.
  const LinkCRSPartitionView &get_or_create_crs_partitions(const stk::mesh::Selector &selector) const {
    std::cout << "inside get_or_create_crs_partitions" << std::endl;
    MUNDY_THROW_ASSERT(is_valid(), std::invalid_argument, "Link data is not valid.");

    // We only care about the intersection of the given selector and our universe link selector.
    std::cout << "about to create link_subset_selector" << std::endl;
    stk::mesh::Selector link_subset_selector = link_meta_data().universal_link_part() & selector;

    // Memoized return
    typename SelectorToPartitionsMap::iterator it = selector_to_partitions_map_.find(link_subset_selector);
    if (it != selector_to_partitions_map_.end()) {
      // Return the existing view
      return it->second;
    } else {
      // Create a new view
      // 1. Map selector to buckets
      std::cout << "about to get buckets" << std::endl;
      const stk::mesh::BucketVector &selected_buckets =
          bulk_data().get_buckets(link_meta_data().link_rank(), link_subset_selector);

      // 2. Sort and unique the keys for each buckets
      std::cout << "about to get unique keys" << std::endl;
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
        std::cout << "about to resize all_crs_partitions_" << std::endl;
        Kokkos::resize(Kokkos::WithoutInitializing, all_crs_partitions_, num_previous_partitions + num_new_partitions);

        // 4. Create a new LinkCRSPartition (for each unique new key) and store it within the all_crs_partitions_ view
        std::cout << "about to create new partitions" << std::endl;
        stk::mesh::Ordinal partition_id = static_cast<stk::mesh::Ordinal>(num_previous_partitions);
        for (const PartitionKey &key : new_keys) {
          std::cout << " Creating new partition with id: " << partition_id << std::endl;
          new (&all_crs_partitions_(partition_id))
              LinkCRSPartition(partition_id, key, link_rank(), get_linker_dimensionality(key),
                               bulk_data());
          partition_key_to_id_map_[key] = partition_id;
          ++partition_id;
        }
      }

      // 5. Create a new view of CRS partitions of size equal to the number of unique keys (both existing and new)
      LinkCRSPartitionView new_crs_partitions(Kokkos::view_alloc(Kokkos::WithoutInitializing, "LinkCRSPartitions"), num_new_partitions + num_old_partitions);

      // 6. Copy the corresponding LinkCRSPartition from the all_crs_partitions_ view to the new view using the key to
      // partition_id map
      std::cout << "about to copy old partitions to new view" << std::endl;
      unsigned count = 0;
      for (const PartitionKey &key : old_keys) {
        auto it = partition_key_to_id_map_.find(key);
        if (it != partition_key_to_id_map_.end()) {
          stk::mesh::Ordinal partition_id = it->second;
          new (&new_crs_partitions(count)) LinkCRSPartition(all_crs_partitions_(partition_id));  // Shallow copy
          ++count;
        } else {
          MUNDY_THROW_ASSERT(false, std::logic_error,
                             "Partition key not found in partition key to id map. This should never happen.");
        }
      }
      std::cout << "about to copy new partitions to new view" << std::endl;
      for (const PartitionKey &key : new_keys) {
        auto it = partition_key_to_id_map_.find(key);
        if (it != partition_key_to_id_map_.end()) {
          stk::mesh::Ordinal partition_id = it->second;
          new (&new_crs_partitions(count)) LinkCRSPartition(all_crs_partitions_(partition_id));  // Shallow copy
          ++count;
        } else {
          MUNDY_THROW_ASSERT(false, std::logic_error,
                             "Partition key not found in partition key to id map. This should never happen.");
        }
      }
      std::cout << "Count: " << count << ", new_crs_partitions.extent(0): " << new_crs_partitions.extent(0) << std::endl;

      // 7. Store the new view in the selector to partitions map
      std::cout << "about to store new view in map" << std::endl;
      selector_to_partitions_map_[link_subset_selector] = new_crs_partitions;

      // 8. Return a reference to the new view
      std::cout << "about to return new view" << std::endl;
      return selector_to_partitions_map_[link_subset_selector];
    }
  }

  /// \brief Get the map from link bucket id to partition id (memoized/host only/not thread safe).
  LinkBucketToPartitionIdMap get_updated_stk_link_bucket_to_partition_id_map() {
    // If the map is empty, populate it.  MOD MARK: There are so many better ways to do this.
    if (stk_link_bucket_to_partition_id_map_.size() == 0) {
      update_stk_link_bucket_to_partition_id_map();
    }

    return stk_link_bucket_to_partition_id_map_;
  }

  /// \brief Update the map from link bucket id to partition id (host only/not thread safe).
  void update_stk_link_bucket_to_partition_id_map() {
    // Get all link buckets that currently have selectors.
    stk::mesh::Selector our_all_selector = all_selector();
    const stk::mesh::BucketVector &all_link_buckets =
        bulk_data().get_buckets(link_meta_data().link_rank(), our_all_selector);
    const unsigned num_link_buckets = static_cast<unsigned>(all_link_buckets.size());

    // Resize the map if needed. (only grow)
    if (stk_link_bucket_to_partition_id_map_host_.capacity() < num_link_buckets) {
      stk_link_bucket_to_partition_id_map_host_.rehash(num_link_buckets);
    }

    // Loop over each bucket, get its partition key, map the key to an id, and store the id in the map.
    for (const stk::mesh::Bucket *bucket : all_link_buckets) {
      PartitionKey key = get_partition_key(*bucket);

      auto it = partition_key_to_id_map_.find(key);
      if (it != partition_key_to_id_map_.end()) {
        stk::mesh::Ordinal partition_id = it->second;
        bool insert_success =
            stk_link_bucket_to_partition_id_map_host_.insert(bucket->bucket_id(), partition_id).success();
        MUNDY_THROW_ASSERT(insert_success, std::runtime_error,
                            "Failed to insert bucket -> partition pair into the map. This is an internal error.");
      } else {
        MUNDY_THROW_ASSERT(false, std::logic_error,
                            "Partition key not found in partition key to id map. This should never happen.");
      }
    }

    // Copy to device
    Kokkos::deep_copy(stk_link_bucket_to_partition_id_map_, stk_link_bucket_to_partition_id_map_host_);
  }

  template<typename OtherMemSpace>
  void synchronize_with(LinkCRSDataT<OtherMemSpace> &src) {  // Shallow copy if same space, otherwise deep copy
    if constexpr (std::is_same_v<MemSpace, OtherMemSpace>) {
      // Shallow copy. They have the same template param as us, so we're friends.
      bulk_data_ptr_ = &src.bulk_data();
      link_meta_data_ptr_ = &src.link_meta_data();
      selector_to_partitions_map_ = src.selector_to_partitions_map_;
      partition_key_to_id_map_ = src.partition_key_to_id_map_;
      all_crs_partitions_ = src.all_crs_partitions_;
      stk_link_bucket_to_partition_id_map_host_ = src.stk_link_bucket_to_partition_id_map_host_;
      stk_link_bucket_to_partition_id_map_ = src.stk_link_bucket_to_partition_id_map_;
    } else {
      // For each partition in the source, loop over each of its buckets and deep copy them to the corresponding
      // desk bucket.
      //
      // At this point their number of partitions aren't guaranteed to be the same since get_or_create_crs_partitions
      // may have only been called on one memory space and not the other.
      stk::mesh::Selector src_all_selector = src.all_selector();
      auto &src_crs_partitions = src.get_all_crs_partitions();
      Kokkos::resize(Kokkos::WithoutInitializing, all_crs_partitions_, src_crs_partitions.extent(0));

      MUNDY_THROW_ASSERT(all_crs_partitions_.extent(0) == src_crs_partitions.extent(0), std::logic_error,
                         "Internal error, inform the devs. Number of partitions in our memory space somehow differs "
                         "from the src's number of partitions.");

      for (unsigned partition_id = 0u; partition_id < all_crs_partitions_.extent(0); ++partition_id) {
        // The only way the following is true, is if we sort the partition view by partition id.
        std::cout << "Synchronizing partition id: " << partition_id << std::endl;
        new (&all_crs_partitions_(partition_id)) LinkCRSPartition();
        auto &our_partition = all_crs_partitions_(partition_id);
        auto &src_partition = src_crs_partitions(partition_id);
        deep_copy(our_partition, src_partition);

        std::cout << "  Our partition has " << our_partition.num_buckets(link_meta_data().link_rank())
                  << " buckets." << std::endl;
      }    
    }
  }

  stk::mesh::Selector all_selector() const {
    stk::mesh::Selector our_all_selector;
    for (const auto &pair : selector_to_partitions_map_) {
      our_all_selector |= pair.first;
    }
    return our_all_selector;
  }

 protected:
  //! \name Internal methods
  //@{

  /// \brief Get the dimensionality of a linker
  /// \param linker [in] The linker (must be valid and of the correct rank).
  inline unsigned get_linker_dimensionality(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == bulk_data().entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data().is_valid(linker), std::invalid_argument, "Linker is not valid.");
    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    return stk::mesh::field_scalars_per_entity(linked_es_field, linker);
  }

  /// \brief Get the dimensionality of a linker bucket
  inline unsigned get_linker_dimensionality(const stk::mesh::Bucket &linker_bucket) const {
    MUNDY_THROW_ASSERT(link_meta_data().link_rank() == linker_bucket.entity_rank(), std::invalid_argument,
                       "Linker bucket is not of the correct rank.");
    MUNDY_THROW_ASSERT(linker_bucket.member(link_meta_data().universal_link_part()), std::invalid_argument,
                       "Linker bucket is not a subset of our universal link part.");

    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    return stk::mesh::field_scalars_per_entity(linked_es_field, linker_bucket);
  }

  /// \brief Get the dimensionality of a linker partition
  inline unsigned get_linker_dimensionality(const PartitionKey &partition_key) const {
    MUNDY_THROW_REQUIRE(partition_key.size() > 0, std::invalid_argument, "Partition key is empty.");

    // Fetch the parts
    stk::mesh::PartVector parts(partition_key.size());
    for (size_t i = 0; i < partition_key.size(); ++i) {
      parts[i] = &bulk_data().mesh_meta_data().get_part(partition_key[i]);
    }

    // FieldBase::restrictions
    auto &linked_es_field = impl::get_linked_entities_field(link_meta_data());
    const stk::mesh::FieldRestriction &restriction =
        stk::mesh::find_restriction(linked_es_field, link_meta_data().link_rank(), parts);
    return restriction.num_scalars_per_entity();
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

  void sort_partitions_by_id() {
    Kokkos::sort(all_crs_partitions_,
                 [](const LinkCRSPartition &a, const LinkCRSPartition &b) { return a.id() < b.id(); });
  }


  KOKKOS_FUNCTION
  void clear_partitions_and_views() {
    KOKKOS_IF_ON_HOST(
      // Kill all_partitions_ if we're the last reference to it.
      if (all_crs_partitions_.use_count() == 1) {
        for (unsigned i = 0; i < all_crs_partitions_.size(); ++i) {
          all_crs_partitions_[i].~LinkCRSPartition();
        }
      }

      // Kill selector_to_partitions_map_'s partitions if we're the last reference to them.
      // These are distinct copies of LinkCRSPartitions, so we need to destroy them too.
      // TODO(palmerb4): Does this double free their internal views or will their ref count prevent that?
      for (auto &pair : selector_to_partitions_map_) {
        LinkCRSPartitionView &view = pair.second;
        if (view.use_count() == 1) {
          for (unsigned i = 0; i < view.size(); ++i) {
            view[i].~LinkCRSPartition();
          }
        }
      }
    );
  }
  //@}


 private:
  //! \name Internal members (host only)
  //@{

  stk::mesh::BulkData *bulk_data_ptr_;
  LinkMetaData *link_meta_data_ptr_;

  using SelectorToPartitionsMap = std::map<stk::mesh::Selector, LinkCRSPartitionView>;
  using PartitionKeyToIdMap = std::map<PartitionKey, unsigned>;
  mutable SelectorToPartitionsMap selector_to_partitions_map_;  // NEEDS to be a VIEW data type. Right now, our copies may be modified without us knowing.
  mutable PartitionKeyToIdMap partition_key_to_id_map_;
  //@}

  //! \name Internal members (device compatible)
  //@{

  mutable LinkCRSPartitionView all_crs_partitions_;
  LinkBucketToPartitionIdMap stk_link_bucket_to_partition_id_map_;
  LinkBucketToPartitionIdMap::HostMirror stk_link_bucket_to_partition_id_map_host_;
  //@}
};  // LinkCRSDataT

// Following STK's default naming convention, to make return statements of our functions more readable.
using LinkCRSData = LinkCRSDataT<stk::ngp::HostMemSpace>;
template<typename NgpMemSpace>
using NgpLinkCRSDataT = LinkCRSDataT<NgpMemSpace>;
using NgpLinkCRSData = LinkCRSDataT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKCRSDATA_HPP_
