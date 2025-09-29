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

#ifndef MUNDY_MESH_NEW_NGPCRSPARTITION_HPP_
#define MUNDY_MESH_NEW_NGPCRSPARTITION_HPP_

/// \file NewNgpCRSPartition.hpp

/*
General comments as we go:

- Partitions need to store their selector
- We need to make sure that we are only updating the CRS for buckets that were marked as needing an update
- I'm not sure we should call these partitions partitions since all they really do is help manage the CRS connectivity
  It's just that the CRS connectivity can only be accessed for a given partition.

Can we hide this entire class from the user by making it internal? NgpPartitionedCRSConn
How would they access the CRS connectivity then? Can we could give them a PartitionOrdinal instead of giving them an
instance of this class? Well, if a user requests a partition, can we ever destroy it? No. I assume that's part of why
STK chose to make partitions internal details.

I guess we need to address the underlying question: when should we destroy a partition?
After much deliberation, I think the answer is never. We'll allow users to request CRS connectivity for a given
partition, and we will give them an ordinal that they can then use to access the CRS connectivity. In this fashion,
users will interact with the CRS connectivity via the NgpLinkData class calling get_connected_links(p_ordinal, rank,
entity_fmi), which will perform partitioned_crs_conn_vec_[p_ordinal].get_connected_links(rank, entity_fmi);

update_crs_from_coo should be done for all partitions all at once and should be managed by the NgpLinkData.

We need the vector of partitions to be accessible on the GPU. The only data contained in the NgpPartition is a key,
rank, dimensionality, and vector of linked buckets per rank. Of these rank and dim come from the owning link data.

What we actually need to store is:
- [rank][partition_id][linked_bucket_id][entity_offset] -> NgpLinkedBucket
  Array of Kokkos::View<NewNgpCRSBucketConnT<NgpMemSpace> **, stk::ngp::UVMMemSpace>
- [partition_id] -> PartitionKey
  std::vector<PartitionKey>
- [partition_key] -> partition_id
  std::unordered_map<PartitionKey, PartitionOrdinal>

Previously we used a map from key to partition. I don't think we really care about the partition key that much. It would
be better to use a contiguous vector of partitions indexed by contiguous i
*/

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
#include <mundy_core/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>        // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>        // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>    // for mundy::mesh::field_copy

namespace mundy {

namespace mesh {

using PartitionKey = std::vector<stk::mesh::PartOrdinal>;  // sorted view of part ordinals
using NgpPartitionKey = stk::mesh::PartOrdinalViewType;    // sorted view of part ordinals

template <typename NgpMemSpace>
class NewNgpCRSPartitionT;

template <typename NgpMemSpace>
class NewNgpCRSBucketConnT {
 public:
  using BucketConnectivityType = Kokkos::View<stk::mesh::Entity *, NgpMemSpace>;
  using UnsignedViewType = Kokkos::View<unsigned *, NgpMemSpace>;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

  NewNgpCRSBucketConnT()
      : dirty_(false),
        bucket_size_(0),
        bucket_capacity_(0),
        bucket_id_(0),
        bucket_rank_(stk::topology::INVALID_RANK),
        num_connected_links_("num_connected_links", 0),
        sparse_connectivity_offsets_("sparse_connectivity_offsets", 0),
        sparse_connectivity_("sparse_connectivity", 0) {
  }

  KOKKOS_INLINE_FUNCTION
  unsigned bucket_id() const noexcept {
    return bucket_id_;
  }

  KOKKOS_INLINE_FUNCTION
  size_t size() const noexcept {
    return bucket_size_;
  }

  KOKKOS_INLINE_FUNCTION
  size_t capacity() const noexcept {
    return bucket_capacity_;
  }

  KOKKOS_INLINE_FUNCTION
  ConnectedEntities get_connected_links(unsigned offset_into_bucket) const {
    const unsigned offset = sparse_connectivity_offsets_(offset_into_bucket);
    const unsigned length = sparse_connectivity_offsets_(offset_into_bucket + 1) - offset;
    return ConnectedEntities(&sparse_connectivity_(offset), length, 1);
  }

  KOKKOS_INLINE_FUNCTION
  unsigned num_connected_links(unsigned offset_into_bucket) const {
    return num_connected_links_(offset_into_bucket);
  }

  void initialize_bucket_attributes(const stk::mesh::Bucket &bucket) {
    dirty_ = false;
    bool bucket_capacity_changed = bucket.capacity() != bucket_capacity_;
    bucket_id_ = bucket.bucket_id();
    bucket_rank_ = bucket.entity_rank();
    bucket_capacity_ = bucket.capacity();
    bucket_size_ = bucket.size();

    if (bucket_capacity_changed) {
      // Note, we resize our bucket views to be the same size as the bucket *capacity*, not the bucket *size*,
      // since the bucket size changes more frequently than the capacity due to STK's dynamicly adapted capacity.
      // This also means that there is no need for us to invent our own capacity management scheme.
      Kokkos::realloc(num_connected_links_, bucket_capacity_);
      Kokkos::realloc(sparse_connectivity_offsets_, bucket_capacity_ + 1);
      Kokkos::deep_copy(num_connected_links_, 0);
      Kokkos::deep_copy(sparse_connectivity_offsets_, 0);
    }
  }

  void dump() const {
    std::cout << "Bucket ID: " << bucket_id_ << std::endl;
    std::cout << "Bucket rank: " << bucket_rank_ << std::endl;
    std::cout << "Bucket size: " << bucket_size_ << std::endl;
    std::cout << "Bucket capacity: " << bucket_capacity_ << std::endl;
    std::cout << "Bucket is dirty?: " << (dirty_ ? "true" : "false") << std::endl;

    std::cout << "Number of Connected Links: size " << num_connected_links_.extent(0) << " values: ";
    for (unsigned i = 0; i < bucket_size_; ++i) {
      std::cout << num_connected_links_(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "Sparse Connectivity Offsets: size " << sparse_connectivity_offsets_.extent(0) << " values: ";
    for (unsigned i = 0; i < bucket_size_ + 1; ++i) {
      std::cout << sparse_connectivity_offsets_(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "Sparse Connectivity: size " << sparse_connectivity_.extent(0) << " values: ";
    for (unsigned i = 0; i < sparse_connectivity_.extent(0); ++i) {
      std::cout << sparse_connectivity_(i) << " ";
    }
    std::cout << std::endl;
  }

  int dirty_;  ///< Whether this bucket's connectivity has been modified since the last update.
  unsigned bucket_size_;
  unsigned bucket_capacity_;
  unsigned bucket_id_;
  stk::mesh::EntityRank bucket_rank_;
  UnsignedViewType num_connected_links_;
  UnsignedViewType sparse_connectivity_offsets_;
  BucketConnectivityType sparse_connectivity_;
};  // NewNgpCRSBucketConnT

template<typename MemSpace1, typename MemSpace2>
void deep_copy(NewNgpCRSBucketConnT<MemSpace1> &dest, const NewNgpCRSBucketConnT<MemSpace2> &src) {
  dest.dirty_ = src.dirty_;
  dest.bucket_size_ = src.bucket_size_;
  dest.bucket_capacity_ = src.bucket_capacity_;
  dest.bucket_id_ = src.bucket_id_;
  dest.bucket_rank_ = src.bucket_rank_;
  Kokkos::resize(dest.num_connected_links_, src.num_connected_links_.extent(0));
  Kokkos::resize(dest.sparse_connectivity_offsets_, src.sparse_connectivity_offsets_.extent(0));
  Kokkos::resize(dest.sparse_connectivity_, src.sparse_connectivity_.extent(0));
  Kokkos::deep_copy(dest.num_connected_links_, src.num_connected_links_);
  Kokkos::deep_copy(dest.sparse_connectivity_offsets_, src.sparse_connectivity_offsets_);
  Kokkos::deep_copy(dest.sparse_connectivity_, src.sparse_connectivity_);
}

template <typename NgpMemSpace>
struct NewNgpCRSPartitionT {
  //! \name Aliases
  //@{

  static_assert(Kokkos::is_memory_space_v<NgpMemSpace>);
  using memory_space = NgpMemSpace;
  using execution_space = typename NgpMemSpace::execution_space;

  using LinkedBucket = NewNgpCRSBucketConnT<NgpMemSpace>;
  using LinkedBucketView = Kokkos::View<LinkedBucket *, stk::ngp::UVMMemSpace>;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  //@}

  //! \name Public constructors and destructor
  //@{

  NewNgpCRSPartitionT() = default;

  NewNgpCRSPartitionT(const stk::mesh::Ordinal &partition_id, const PartitionKey key,
                      const stk::mesh::EntityRank &link_rank, const unsigned link_dimensionality,
                      const stk::mesh::BulkData &bulk_data)
      : id_(partition_id), link_rank_(link_rank), link_dimensionality_(link_dimensionality) {
    // Map host key to ngp key
    ngp_key_ = NgpPartitionKey("NgpCRSPartitionKey", key.size());
    auto ngp_key_host = Kokkos::create_mirror_view(ngp_key_);
    for (size_t i = 0; i < key.size(); ++i) {
      ngp_key_host(i) = key[i];
    }
    Kokkos::deep_copy(ngp_key_, ngp_key_host);

    // Map key to selector
    stk::mesh::PartVector parts;
    for (const stk::mesh::PartOrdinal &part_ordinal : key) {
      parts.push_back(&bulk_data.mesh_meta_data().get_part(part_ordinal));
    }
    selector_ = stk::mesh::selectIntersection(parts);

    // Initialize the linked buckets for each rank
    for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      const stk::mesh::BucketVector &buckets = bulk_data.buckets(rank);
      size_t num_buckets = buckets.size();
      linked_buckets_[rank] = LinkedBucketView(
        "LinkedBuckets", num_buckets);
      for (size_t i = 0; i < num_buckets; ++i) {
        linked_buckets_[rank](i).initialize_bucket_attributes(*buckets[i]);
      }
    }
  }

  void initialize_attributes(const stk::mesh::Ordinal &partition_id, const PartitionKey key,
                             const stk::mesh::EntityRank &link_rank, const unsigned link_dimensionality,
                             const stk::mesh::BulkData &bulk_data) {
    id_ = partition_id;
    link_rank_ = link_rank;
    link_dimensionality_ = link_dimensionality;

    // Map host key to ngp key
    ngp_key_ = NgpPartitionKey("NgpCRSPartitionKey", key.size());
    auto ngp_key_host = Kokkos::create_mirror_view(ngp_key_);
    for (size_t i = 0; i < key.size(); ++i) {
      ngp_key_host(i) = key[i];
    }
    Kokkos::deep_copy(ngp_key_, ngp_key_host);

    // Map key to selector
    stk::mesh::PartVector parts;
    for (const stk::mesh::PartOrdinal &part_ordinal : key) {
      parts.push_back(&bulk_data.mesh_meta_data().get_part(part_ordinal));
    }
    selector_ = stk::mesh::selectIntersection(parts);

    // Initialize the linked buckets for each rank. One per stk bucket of a given rank.
    for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      const stk::mesh::BucketVector &buckets = bulk_data.buckets(rank);
      size_t num_buckets = buckets.size();
      linked_buckets_[rank] = LinkedBucketView("LinkedBuckets", num_buckets);
      for (size_t i = 0; i < num_buckets; ++i) {
        linked_buckets_[rank](i).initialize_bucket_attributes(*buckets[i]);
      }
    }
  }

  NewNgpCRSPartitionT(const NewNgpCRSPartitionT &other) = default;
  NewNgpCRSPartitionT(NewNgpCRSPartitionT &&other) = default;
  NewNgpCRSPartitionT &operator=(const NewNgpCRSPartitionT &other) = default;
  NewNgpCRSPartitionT &operator=(NewNgpCRSPartitionT &&other) = default;

  virtual ~NewNgpCRSPartitionT() {
    clear_buckets_and_views();
    
    std::cout << "DESTRUCTOR FOR NewNgpCRSPartitionT" << std::endl;
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the partition key.
  KOKKOS_INLINE_FUNCTION
  const NgpPartitionKey &ngp_key() const noexcept {
    return ngp_key_;
  }

  /// \brief Fetch the partition id.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Ordinal id() const noexcept {
    return id_;
  }

  /// \brief Fetch the link rank.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank link_rank() const noexcept {
    return link_rank_;
  }

  /// \brief Fetch the link dimensionality.
  KOKKOS_INLINE_FUNCTION
  unsigned link_dimensionality() const noexcept {
    return link_dimensionality_;
  }

  /// \brief Fetch the selector for this partition.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Selector selector() const noexcept {
    return selector_;
  }

  /// \brief Check if this partition contains a given part
  KOKKOS_INLINE_FUNCTION
  bool contains(stk::mesh::PartOrdinal part_ordinal) const {
    bool does_contain = false;
    for (unsigned i = 0u; i < ngp_key_.extent(0); ++i) {
      stk::mesh::PartOrdinal ordinal = ngp_key_(i);
      if (ordinal == part_ordinal) {
        does_contain = true;
        break;
      }
    }
    return does_contain;
  }
  //@}

  //! \name CRS connectivity
  //@{

  /// \brief If any of our linkers connect to an entity in the given bucket within the CRS connectivity.
  KOKKOS_INLINE_FUNCTION
  bool connects_to(stk::mesh::EntityRank rank, const unsigned &bucket_id) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    return linked_buckets_[rank](bucket_id).size() > 0;
  }

  /// \brief Get the number of linked buckets for a given rank.
  KOKKOS_INLINE_FUNCTION
  unsigned num_buckets(stk::mesh::EntityRank rank) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    return linked_buckets_[rank].size();
  }

  /// \brief Get the linked bucket for a given rank and bucket id.
  KOKKOS_INLINE_FUNCTION
  LinkedBucket &get_crs_bucket_conn(stk::mesh::EntityRank rank, unsigned bucket_id) {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    return linked_buckets_[rank](bucket_id);
  }
  KOKKOS_INLINE_FUNCTION
  const LinkedBucket &get_crs_bucket_conn(stk::mesh::EntityRank rank, unsigned bucket_id) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    return linked_buckets_[rank](bucket_id);
  }

  /// \brief Get all links in the current partition that connect to the given entity in the CRS connectivity.
  KOKKOS_INLINE_FUNCTION
  ConnectedEntities get_connected_links(stk::mesh::EntityRank rank,
                                        const stk::mesh::FastMeshIndex &entity_index) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(entity_index.bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(entity_index.bucket_ord < linked_buckets_[rank](entity_index.bucket_id).size(),
                       std::invalid_argument, "Bucket ordinal is out of bounds for this partition.");
    return linked_buckets_[rank](entity_index.bucket_id).get_connected_links(entity_index.bucket_ord);
  }

  /// \brief Get the number of links in the current partition that connect to the given entity in the CRS connectivity.
  KOKKOS_INLINE_FUNCTION
  unsigned num_connected_links(stk::mesh::EntityRank rank, const stk::mesh::FastMeshIndex &entity_index) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(entity_index.bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(entity_index.bucket_ord < linked_buckets_[rank](entity_index.bucket_id).size(),
                       std::invalid_argument, "Bucket ordinal is out of bounds for this partition.");
    return linked_buckets_[rank](entity_index.bucket_id).num_connected_links(entity_index.bucket_ord);
  }
  //@}

  private:
  
  //! \name Helpers
  //@{

  template<typename CRSPartition1, typename CRSPartition2>
  friend void deep_copy(CRSPartition1 &dest, const CRSPartition2 &src);

  KOKKOS_FUNCTION
  bool is_last_bucket_reference(stk::mesh::EntityRank rank = stk::topology::NODE_RANK) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Rank is out of bounds for this partition.");
    return (linked_buckets_[rank].use_count() == 1);
  }

  KOKKOS_FUNCTION
  void clear_buckets_and_views() {
    KOKKOS_IF_ON_HOST((if (is_last_bucket_reference()) {
      for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; rank++) {
        for (unsigned iBucket = 0; iBucket < linked_buckets_[rank].size(); ++iBucket) {
          linked_buckets_[rank][iBucket].~NewNgpCRSBucketConnT<NgpMemSpace>();
        }
      }
    }))
  }
  //@}

  //! \name Private data members
  //@{

  stk::mesh::Ordinal id_;    ///< Unique identifier for this partition.
  NgpPartitionKey ngp_key_;  ///< Sorted view of the part ordinals that this partition contains, in NGP memory space.
  stk::mesh::Selector selector_;     ///< Selector for this partition, derived from the ngp_key_.
  stk::mesh::EntityRank link_rank_;  ///< Rank of the linkers in this partition.
  unsigned link_dimensionality_;     ///< Maximum dimensionality of the parts contained in this partition.
  LinkedBucketView linked_buckets_[stk::topology::NUM_RANKS];  ///< Bucketized CRS connectivity for each rank.
  //@}
};

using NewNgpCRSPartition = NewNgpCRSPartitionT<stk::ngp::MemSpace>;

template<typename MemSpace1, typename MemSpace2>
void deep_copy(NewNgpCRSPartitionT<MemSpace1> &dest, const NewNgpCRSPartitionT<MemSpace2> &src) {
  dest.id_ = src.id_;
  dest.ngp_key_ = src.ngp_key_;
  dest.selector_ = src.selector_;
  dest.link_rank_ = src.link_rank_;
  dest.link_dimensionality_ = src.link_dimensionality_;

  if (dest.linked_buckets_.extent(0) != src.linked_buckets_.extent(0)) {
    Kokkos::resize(Kokkos::WithoutInitializing, dest.linked_buckets_, src.linked_buckets_.extent(0));
  }

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    deep_copy(dest.linked_buckets_[rank], src.linked_buckets_[rank]);
  }
}


}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NEW_NGPCRSPARTITION_HPP_
