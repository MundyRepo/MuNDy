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

#ifndef MUNDY_MESH_LINKCRSBUCKETCONN_HPP_
#define MUNDY_MESH_LINKCRSBUCKETCONN_HPP_

/// \file LinkCRSBucketConn.hpp

// Trilinos libs
#include <Kokkos_Core.hpp>             // for Kokkos::View, KOKKOS_INLINE_FUNCTION
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::EntityRank
#include <stk_util/ngp/NgpSpaces.hpp>  // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/Types.hpp>         // for stk::ngp::HostMemSpace

namespace mundy {

namespace mesh {

template <typename MemSpace>
class LinkCRSBucketConnT;
namespace impl {
template <typename MemSpace>
KOKKOS_INLINE_FUNCTION int &get_dirty_flag(LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const int &get_dirty_flag(const LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<unsigned *, MemSpace> &get_num_connected_links(
    LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<unsigned *, MemSpace> &get_num_connected_links(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<unsigned *, MemSpace> &get_sparse_connectivity_offsets(
    LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<unsigned *, MemSpace> &get_sparse_connectivity_offsets(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<stk::mesh::Entity *, MemSpace> &get_sparse_connectivity(
    LinkCRSBucketConnT<MemSpace> &bucket_conn);

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<stk::mesh::Entity *, MemSpace> &get_sparse_connectivity(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn);
}  // namespace impl

template <typename MemSpace>
class LinkCRSBucketConnT {  // Raw data in any space.
 public:
  using BucketConnectivityType = Kokkos::View<stk::mesh::Entity *, MemSpace>;
  using UnsignedViewType = Kokkos::View<unsigned *, MemSpace>;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

  LinkCRSBucketConnT()
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
  unsigned size() const noexcept {
    return bucket_size_;
  }

  KOKKOS_INLINE_FUNCTION
  unsigned capacity() const noexcept {
    return bucket_capacity_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank bucket_rank() const noexcept {
    return bucket_rank_;
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

 private:
  // clang-format off
  template <typename MS1, typename MS2> friend void deep_copy(LinkCRSBucketConnT<MS1> &dest, const LinkCRSBucketConnT<MS2> &src);
  template <typename MS> friend       int &impl::get_dirty_flag(      LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend const int &impl::get_dirty_flag(const LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend       Kokkos::View<unsigned*, MS> &impl::get_num_connected_links(      LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend const Kokkos::View<unsigned*, MS> &impl::get_num_connected_links(const LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend       Kokkos::View<unsigned*, MS> &impl::get_sparse_connectivity_offsets(      LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend const Kokkos::View<unsigned*, MS> &impl::get_sparse_connectivity_offsets(const LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend       Kokkos::View<stk::mesh::Entity*, MS> &impl::get_sparse_connectivity(      LinkCRSBucketConnT<MS> &bucket_conn);
  template <typename MS> friend const Kokkos::View<stk::mesh::Entity*, MS> &impl::get_sparse_connectivity(const LinkCRSBucketConnT<MS> &bucket_conn);
  // clang-format on

  int dirty_;  ///< Whether this bucket's connectivity has been modified since the last update.
  unsigned bucket_size_;
  unsigned bucket_capacity_;
  unsigned bucket_id_;
  stk::mesh::EntityRank bucket_rank_;
  UnsignedViewType num_connected_links_;
  UnsignedViewType sparse_connectivity_offsets_;
  BucketConnectivityType sparse_connectivity_;
};  // LinkCRSBucketConnT

// Following STK's default naming convention, to make return statements of our functions more readable.
using LinkCRSBucketConn = LinkCRSBucketConnT<stk::ngp::HostMemSpace>;
using NgpLinkCRSBucketConn = LinkCRSBucketConnT<stk::ngp::MemSpace>;

template <typename MemSpace1, typename MemSpace2>
void deep_copy(LinkCRSBucketConnT<MemSpace1> &dest, const LinkCRSBucketConnT<MemSpace2> &src) {
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

namespace impl {
template <typename MemSpace>
KOKKOS_INLINE_FUNCTION int &get_dirty_flag(LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.dirty_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const int &get_dirty_flag(const LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.dirty_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<unsigned *, MemSpace> &get_num_connected_links(
    LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.num_connected_links_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<unsigned *, MemSpace> &get_num_connected_links(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.num_connected_links_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<unsigned *, MemSpace> &get_sparse_connectivity_offsets(
    LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.sparse_connectivity_offsets_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<unsigned *, MemSpace> &get_sparse_connectivity_offsets(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.sparse_connectivity_offsets_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION Kokkos::View<stk::mesh::Entity *, MemSpace> &get_sparse_connectivity(
    LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.sparse_connectivity_;
}

template <typename MemSpace>
KOKKOS_INLINE_FUNCTION const Kokkos::View<stk::mesh::Entity *, MemSpace> &get_sparse_connectivity(
    const LinkCRSBucketConnT<MemSpace> &bucket_conn) {
  return bucket_conn.sparse_connectivity_;
}
}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKCRSBUCKETCONN_HPP_
