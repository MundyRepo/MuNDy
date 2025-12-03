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

#ifndef MUNDY_MESH_LINKCSRPARTITION_HPP_
#define MUNDY_MESH_LINKCSRPARTITION_HPP_

/// \file LinkCSRPartition.hpp

/*
General comments as we go:

- Partitions need to store their selector
- We need to make sure that we are only updating the CSR for buckets that were marked as needing an update
- I'm not sure we should call these partitions partitions since all they really do is help manage the CSR connectivity
  It's just that the CSR connectivity can only be accessed for a given partition.

Can we hide this entire class from the user by making it internal? NgpPartitionedCSRConn
How would they access the CSR connectivity then? Can we could give them a PartitionOrdinal instead of giving them an
instance of this class? Well, if a user requests a partition, can we ever destroy it? No. I assume that's part of why
STK chose to make partitions internal details.

I guess we need to address the underlying question: when should we destroy a partition?
After much deliberation, I think the answer is never. We'll allow users to request CSR connectivity for a given
partition, and we will give them an ordinal that they can then use to access the CSR connectivity. In this fashion,
users will interact with the CSR connectivity via the NgpLinkData class calling get_connected_links(p_ordinal, rank,
entity_fmi), which will perform partitioned_crs_conn_vec_[p_ordinal].get_connected_links(rank, entity_fmi);

update_crs_from_coo should be done for all partitions all at once and should be managed by the NgpLinkData.

We need the vector of partitions to be accessible on the GPU. The only data contained in the NgpPartition is a key,
rank, dimensionality, and vector of linked buckets per rank. Of these rank and dim come from the owning link data.

What we actually need to store is:
- [rank][partition_id][linked_bucket_id][entity_offset] -> NgpLinkedBucket
  Array of Kokkos::View<LinkCSRBucketConnT<MemSpace> **, stk::ngp::UVMMemSpace>
- [partition_id] -> PartitionKey
  std::vector<PartitionKey>
- [partition_key] -> partition_id
  std::unordered_map<PartitionKey, PartitionOrdinal>

Previously we used a map from key to partition. I don't think we really care about the partition key that much. It would
be better to use a contiguous vector of partitions indexed by contiguous i
*/

// C++ core libs
#include <any>     // for std::any
#include <vector>  // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>             // for Kokkos::View, KOKKOS_INLINE_FUNCTION
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>  // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank
#include <stk_util/ngp/NgpSpaces.hpp>  // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy libs
#include <mundy_core/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_mesh/LinkCSRBucketConn.hpp>  // for mundy::mesh::LinkCSRBucketConn

namespace mundy {

namespace mesh {

using PartitionKey = std::vector<stk::mesh::PartOrdinal>;  // sorted view of part ordinals
using NgpPartitionKey = stk::mesh::PartOrdinalViewType;    // sorted view of part ordinals

template <typename MemSpace>
class LinkCSRPartitionT {  // Raw data in any space.
 public:
  //! \name Aliases
  //@{

  static_assert(Kokkos::is_memory_space_v<MemSpace>);
  using memory_space = MemSpace;
  using execution_space = typename MemSpace::execution_space;

  using LinkCSRBucketConn = LinkCSRBucketConnT<MemSpace>;
  using LinkCSRBucketConnView = Kokkos::View<LinkCSRBucketConn *, stk::ngp::UVMMemSpace>;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
  //@}

  //! \name Public constructors and destructor
  //@{

  KOKKOS_DEFAULTED_FUNCTION
  LinkCSRPartitionT() = default;

  LinkCSRPartitionT(const stk::mesh::Ordinal &partition_id, const PartitionKey key,
                    const stk::mesh::EntityRank &link_rank, const unsigned link_dimensionality,
                    const stk::mesh::BulkData &bulk_data)
      : id_(partition_id), link_rank_(link_rank), link_dimensionality_(link_dimensionality) {
    // Map host key to ngp key
    ngp_key_ = NgpPartitionKey("NgpCSRPartitionKey", key.size());
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
    // selector_ptr_ = new stk::mesh::Selector();
    // *selector_ptr_ = stk::mesh::selectIntersection(parts);
    selector_ = stk::mesh::selectIntersection(parts);

    // Initialize the linked buckets for each rank
    for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      const stk::mesh::BucketVector &buckets = bulk_data.buckets(rank);
      size_t num_buckets = buckets.size();

      linked_buckets_[rank] =
          LinkCSRBucketConnView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "LinkedBuckets"), num_buckets);
      for (size_t i = 0; i < num_buckets; ++i) {
        new (&linked_buckets_[rank][i]) LinkCSRBucketConn(*buckets[i]);
      }
    }
  }

  KOKKOS_DEFAULTED_FUNCTION LinkCSRPartitionT(const LinkCSRPartitionT &other) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCSRPartitionT(LinkCSRPartitionT &&other) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCSRPartitionT &operator=(const LinkCSRPartitionT &other) = default;
  KOKKOS_DEFAULTED_FUNCTION LinkCSRPartitionT &operator=(LinkCSRPartitionT &&other) = default;

  KOKKOS_FUNCTION virtual ~LinkCSRPartitionT() {
    clear_buckets_and_views();
    // delete selector_ptr_;
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
  stk::mesh::Selector selector() const {
    // MUNDY_THROW_REQUIRE(selector_ptr_ != nullptr, std::logic_error,
    //                     "Attempting to access a selector before it has been set.");
    // return *selector_ptr_;
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

  //! \name CSR connectivity
  //@{

  /// \brief If any of our linkers connect to an entity in the given bucket within the CSR connectivity.
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
  LinkCSRBucketConn &get_crs_bucket_conn(stk::mesh::EntityRank rank, unsigned bucket_id) {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    return linked_buckets_[rank](bucket_id);
  }
  KOKKOS_INLINE_FUNCTION
  const LinkCSRBucketConn &get_crs_bucket_conn(stk::mesh::EntityRank rank, unsigned bucket_id) const {
    MUNDY_THROW_ASSERT(rank < stk::topology::NUM_RANKS, std::invalid_argument,
                       "Bucket rank is out of bounds for this partition.");
    MUNDY_THROW_ASSERT(bucket_id < linked_buckets_[rank].size(), std::invalid_argument,
                       "Bucket id is out of bounds for this partition.");
    return linked_buckets_[rank](bucket_id);
  }

  /// \brief Get all links in the current partition that connect to the given entity in the CSR connectivity.
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

  /// \brief Get the number of links in the current partition that connect to the given entity in the CSR connectivity.
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

  template <typename MemSpace1, typename MemSpace2>
  friend void deep_copy(LinkCSRPartitionT<MemSpace1> &dest, const LinkCSRPartitionT<MemSpace2> &src);

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
          linked_buckets_[rank][iBucket].~LinkCSRBucketConnT<MemSpace>();
        }
      }
    }))
  }
  //@}

  //! \name Private data members
  //@{

  stk::mesh::Ordinal id_;    ///< Unique identifier for this partition.
  NgpPartitionKey ngp_key_;  ///< Sorted view of the part ordinals that this partition contains, in NGP memory space.
  stk::mesh::Selector selector_;     ///< Selector for this partition, derived from the ngp_key_. Must be default
                                     ///< constructable, copiable, movable on the device so we use a pointer.
  stk::mesh::EntityRank link_rank_;  ///< Rank of the linkers in this partition.
  unsigned link_dimensionality_;     ///< Maximum dimensionality of the parts contained in this partition.
  LinkCSRBucketConnView linked_buckets_[stk::topology::NUM_RANKS];  ///< Bucketized CSR connectivity for each rank.
  //@}
};

// Following STK's default naming convention, to make return statements of our functions more readable.
using LinkCSRPartition = LinkCSRPartitionT<stk::ngp::HostMemSpace>;
template <typename NgpMemSpace>
using NgpLinkCSRPartitionT = LinkCSRPartitionT<NgpMemSpace>;
using NgpLinkCSRPartition = LinkCSRPartitionT<stk::ngp::MemSpace>;

template <typename MemSpace1, typename MemSpace2>
void deep_copy(LinkCSRPartitionT<MemSpace1> &dest, const LinkCSRPartitionT<MemSpace2> &src) {
  // Destination must at least be default constructed.
  dest.id_ = src.id_;
  dest.ngp_key_ = src.ngp_key_;
  dest.selector_ = src.selector_;
  dest.link_rank_ = src.link_rank_;
  dest.link_dimensionality_ = src.link_dimensionality_;

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    if (dest.linked_buckets_[rank].extent(0) != src.linked_buckets_[rank].extent(0)) {
      Kokkos::resize(Kokkos::WithoutInitializing, dest.linked_buckets_[rank], src.linked_buckets_[rank].extent(0));
    }
    for (unsigned i = 0; i < src.linked_buckets_[rank].extent(0); ++i) {
      new (&dest.linked_buckets_[rank][i]) LinkCSRBucketConnT<MemSpace1>();
      deep_copy(dest.linked_buckets_[rank](i), src.linked_buckets_[rank](i));
    }
  }
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKCSRPARTITION_HPP_
