// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of NTESS nor the names of its contributors
//       may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef MUNDY_MESH_IMPL_LINKEDBUCKETCONN_HPP_
#define MUNDY_MESH_IMPL_LINKEDBUCKETCONN_HPP_

// C++ core
#include <algorithm>  // for std::copy, std::lower_bound
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::pair

// Trilinos
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// STK
#include <stk_util/stk_config.h>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/util/ReportHandler.hpp>
#include <stk_util/util/SortAndUnique.hpp>
#include <stk_util/util/StridedArray.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/fmt_stk_types.hpp>

#define BUCKET_ORDINAL_ERROR_MESSAGE(function_name, bucket_ord, bucket_size)                                   \
  MUNDY_THROW_ASSERT(bucket_ord < bucket_size, std::invalid_argument,                                          \
                     fmt::format("LinkedBucketConn::{}: bucket_ordinal({}) must be less than bucket_size({})", \
                                 function_name, bucket_ord, bucket_size))

namespace mundy {

namespace mesh {

namespace impl {

/// \brief A class for storing the links connected to a entities within a bucket.
///
/// Because the number of links connected to a given entity is dynamic, we chose to use stk's BucketConnDynamic
/// as the basis for this class.
class LinkedBucketConn {
 public:
  //!\name Type aliases
  //@{

  using Entity = stk::mesh::Entity;
  using ConnectivityOrdinal = stk::mesh::ConnectivityOrdinal;
  using ConnectedEntities = stk::util::StridedArray<const Entity>;
  //@}

  //!\name Constructors and destructor
  //@{

  LinkedBucketConn() : LinkedBucketConn(stk::mesh::get_default_initial_bucket_capacity()) {
  }

  LinkedBucketConn(unsigned bucket_capacity)
      : bucket_capacity_(bucket_capacity),
        offsets_(bucket_capacity),
        connectivity_(),
        ordinals_(),
        num_unused_entities_(0),
        compression_threshold_(2) {
    MUNDY_THROW_REQUIRE(bucket_capacity > 0, std::invalid_argument,
                        "LinkedBucketConn must have bucket_capacity strictly greater than 0");
    for (unsigned i = 0; i < bucket_capacity; ++i) {
      offsets_[i] = IndexRange{0, 0};
    }
  }

  virtual ~LinkedBucketConn() = default;
  //@}

  //!\name Getters
  //@{

  size_t bucket_size() const {
    return offsets_.size();
  }

  size_t bucket_capacity() const {
    return bucket_capacity_;
  }

  size_t total_capacity() const {
    return connectivity_.capacity();
  }

  size_t total_num_connectivity() const {
    return connectivity_.size() - num_unused_entities_;
  }

  size_t num_unused_entries() const {
    return num_unused_entities_;
  }

  size_t heap_memory_in_bytes() const {
    return sizeof(IndexRange) * offsets_.capacity() + sizeof(Entity) * connectivity_.capacity() +
           sizeof(ConnectivityOrdinal) * ordinals_.capacity();
  }

  unsigned num_connectivity(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("num_connectivity", bucket_ord, bucket_size());
    auto [first, second] = offsets_[bucket_ord];
    return second - first;
  }

  const ConnectedEntities get_connected_entities(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("get_connected_entities", bucket_ord, bucket_size());
    auto [first, second] = connectivity_.empty() ? IndexRange{0, 0} : offsets_[bucket_ord];
    const unsigned len = second - first;
    const Entity* ptr = len == 0 ? nullptr : connectivity_.data() + first;
    return ConnectedEntities(ptr, len);
  }

  const Entity* begin(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("begin", bucket_ord, bucket_size());
    return connectivity_.empty() ? nullptr : connectivity_.data() + offsets_[bucket_ord].first;
  }
  Entity* begin(unsigned bucket_ord) {
    BUCKET_ORDINAL_ERROR_MESSAGE("begin", bucket_ord, bucket_size());
    return connectivity_.empty() ? nullptr : connectivity_.data() + offsets_[bucket_ord].first;
  }

  const Entity* end(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("end", bucket_ord, bucket_size());
    return connectivity_.empty() ? nullptr : connectivity_.data() + offsets_[bucket_ord].second;
  }
  Entity* end(unsigned bucket_ord) {
    BUCKET_ORDINAL_ERROR_MESSAGE("end", bucket_ord, bucket_size());
    return connectivity_.empty() ? nullptr : connectivity_.data() + offsets_[bucket_ord].second;
  }

  const ConnectivityOrdinal* begin_ordinals(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("begin_ordinals", bucket_ord, bucket_size());
    return ordinals_.empty() ? nullptr : ordinals_.data() + offsets_[bucket_ord].first;
  }
  const ConnectivityOrdinal* end_ordinals(unsigned bucket_ord) const {
    BUCKET_ORDINAL_ERROR_MESSAGE("end_ordinals", bucket_ord, bucket_size());
    return ordinals_.empty() ? nullptr : ordinals_.data() + offsets_[bucket_ord].second;
  }
  //@}

  //!\name Actions
  //@{

  bool add_connectivity(unsigned bucket_ord, Entity entity, ConnectivityOrdinal ordinal) {
    BUCKET_ORDINAL_ERROR_MESSAGE("add_connectivity", bucket_ord, bucket_size());
    return insert_connectivity(bucket_ord, entity, ordinal);
  }

  bool remove_connectivity(unsigned bucket_ord, Entity entity, ConnectivityOrdinal ordinal) {
    BUCKET_ORDINAL_ERROR_MESSAGE("remove_connectivity", bucket_ord, bucket_size());
    IndexRange& indices = offsets_[bucket_ord];
    UpwardConnIndexType idx = indices.second;
    for (UpwardConnIndexType i = indices.first; i < indices.second; ++i) {
      if (connectivity_[i] == entity && ordinals_[i] == ordinal) {
        idx = i;
        break;
      }
    }

    if (idx < indices.second) {
      for (UpwardConnIndexType i = idx; i < indices.second - 1; ++i) {
        connectivity_[i] = connectivity_[i + 1];
        ordinals_[i] = ordinals_[i + 1];
      }

      --indices.second;
      ordinals_[indices.second] = INVALID_CONNECTIVITY_ORDINAL;
      ++num_unused_entities_;
      return true;
    }

    return false;
  }

  bool remove_connectivity(unsigned bucket_ord) {
    BUCKET_ORDINAL_ERROR_MESSAGE("remove_connectivity", bucket_ord, bucket_size());
    IndexRange& indices = offsets_[bucket_ord];
    for (UpwardConnIndexType i = indices.first; i < indices.second; ++i) {
      ordinals_[i] = INVALID_CONNECTIVITY_ORDINAL;
    }

    UpwardConnIndexType num_removed = indices.second - indices.first;
    indices.second = indices.first;
    num_unused_entities_ += num_removed;
    return true;
  }

  bool replace_connectivity(unsigned bucket_ord, unsigned given_num_connectivity, const Entity* connectivity,
                            const ConnectivityOrdinal* ordinals) {
    BUCKET_ORDINAL_ERROR_MESSAGE("replace_connectivity", bucket_ord, bucket_size());
    IndexRange& indices = offsets_[bucket_ord];
    const unsigned num_existing = num_connectivity(bucket_ord);
    const unsigned num_to_replace = std::min(given_num_connectivity, num_existing);
    for (unsigned i = 0; i < num_to_replace; ++i) {
      connectivity_[i + indices.first] = connectivity[i];
      ordinals_[i + indices.first] = ordinals[i];
    }

    if (num_to_replace < num_existing) {
      for (unsigned i = num_to_replace; i < num_existing; ++i) {
        ordinals_[i + indices.first] = INVALID_CONNECTIVITY_ORDINAL;
      }
      indices.second = indices.first + num_to_replace;
      num_unused_entities_ += num_existing - num_to_replace;
      return true;
    } else {
      for (unsigned i = num_existing; i < given_num_connectivity; ++i) {
        add_connectivity(bucket_ord, connectivity[i], ordinals[i]);
      }
    }

    return true;
  }

  bool swap_connectivity(unsigned bktOrdinal1, unsigned bktOrdinal2) {
    BUCKET_ORDINAL_ERROR_MESSAGE("swap_connectivity", bktOrdinal1, bucket_size());
    BUCKET_ORDINAL_ERROR_MESSAGE("swap_connectivity", bktOrdinal2, bucket_size());
    IndexRange tmp = offsets_[bktOrdinal1];
    offsets_[bktOrdinal1] = offsets_[bktOrdinal2];
    offsets_[bktOrdinal2] = tmp;
    return true;
  }

  void compress_connectivity() {
    if (num_unused_entities_ == 0) {
      return;
    }

    unsigned offsets_size = offsets_.size();
    std::vector<std::pair<IndexRange, unsigned>> sorted_offsets(offsets_size);
    for (unsigned i = 0; i < offsets_size; ++i) {
      sorted_offsets[i].first = offsets_[i];
      sorted_offsets[i].second = i;
    }

    std::sort(sorted_offsets.begin(), sorted_offsets.end());

    if (sorted_offsets[0].first.first > 0) {
      IndexRange& sRange = sorted_offsets[0].first;
      const unsigned gap = sRange.first;
      slide_range_and_update(sRange, gap, sorted_offsets[0].second);
    }

    for (unsigned i = 0; i < offsets_size - 1; ++i) {
      const unsigned this_range_end = sorted_offsets[i].first.second;
      const unsigned next_range_begin = sorted_offsets[i + 1].first.first;
      const unsigned gap = next_range_begin - this_range_end;
      if (gap > 0) {
        slide_range_and_update(sorted_offsets[i + 1].first, gap, sorted_offsets[i + 1].second);
      }
    }

    const unsigned oldSize = connectivity_.size();
    connectivity_.resize(oldSize - num_unused_entities_);
    ordinals_.resize(oldSize - num_unused_entities_);
    num_unused_entities_ = 0;
    const unsigned lastIdx = offsets_size - 1;
    STK_ThrowRequireMsg(sorted_offsets[lastIdx].first.second == connectivity_.size(),
                        "Internal LinkedBucketConn::compress_connectivity ERROR, indices out of sync with data.");
  }

  void resize(unsigned new_size) {
    if (new_size != 0) {
      grow_if_necessary(new_size - 1);
    }
  }

  void grow_if_necessary(unsigned bucket_ord) {
    bucket_capacity_ = std::max(bucket_ord + 1, bucket_capacity_);
    if (bucket_ord >= offsets_.size()) {
      const unsigned candidate = offsets_.empty() ? bucket_ord + 1 : 2 * offsets_.size();
      const unsigned newSize = std::min(bucket_capacity_, candidate);
      offsets_.resize(newSize, IndexRange(0u, 0u));

      if (offsets_.capacity() > bucket_capacity_) {
        std::vector<IndexRange>(offsets_).swap(offsets_);
      }
    }
  }

  void increase_bucket_capacity(unsigned new_bucket_capacity) {
    STK_ThrowRequireMsg(new_bucket_capacity >= bucket_capacity_,
                        "BucketDynamicConn::increase_bucket_capacity, old capacity="
                            << bucket_capacity_ << " should be less than new capacity=" << new_bucket_capacity);
    bucket_capacity_ = new_bucket_capacity;
  }

 private:
  //!\name Private type aliases
  //@{

#if TRILINOS_MAJOR_MINOR_VERSION > 160000
  using UpwardConnIndexType = stk::mesh::UpwardConnIndexType;
  static constexpr stk::mesh::ConnectivityOrdinal INVALID_CONNECTIVITY_ORDINAL =
      stk::mesh::INVALID_CONNECTIVITY_ORDINAL;
  static constexpr UpwardConnIndexType INVALID_UPWARDCONN_INDEX = stk::mesh::INVALID_UPWARDCONN_INDEX;
#else
  using UpwardConnIndexType = uint32_t;
  static constexpr stk::mesh::ConnectivityOrdinal INVALID_CONNECTIVITY_ORDINAL =
      std::numeric_limits<stk::mesh::ConnectivityOrdinal>::max();
  static constexpr UpwardConnIndexType INVALID_UPWARDCONN_INDEX = std::numeric_limits<UpwardConnIndexType>::max();
#endif

  using IndexRange = std::pair<UpwardConnIndexType, UpwardConnIndexType>;
  //@}

  //!\name Private helpers
  //@{

  inline bool is_valid(ConnectivityOrdinal ordinal) {
    return ordinal != INVALID_CONNECTIVITY_ORDINAL;
  }

  void slide_range_and_update(IndexRange& range, unsigned gap, unsigned rangeOrd) {
    for (unsigned idx = range.first; idx < range.second; ++idx) {
      connectivity_[idx - gap] = connectivity_[idx];
      ordinals_[idx - gap] = ordinals_[idx];
    }
    range.first -= gap;
    range.second -= gap;
    offsets_[rangeOrd].first -= gap;
    offsets_[rangeOrd].second -= gap;
  }

  bool insert_connectivity(unsigned bucket_ord, Entity entity, ConnectivityOrdinal ordinal) {
    static constexpr unsigned min_size_heuristic = 256;
    if ((total_num_connectivity() > min_size_heuristic) &&
        (total_num_connectivity() < num_unused_entities_ * compression_threshold_)) {
      compress_connectivity();
    }

    grow_if_necessary(bucket_ord);

    IndexRange& indices = offsets_[bucket_ord];

    if (indices.second >= connectivity_.size()) {
      int insert_idx = find_sorted_insertion_index(indices, entity, ordinal);
      if (insert_idx < 0) {
        return false;
      }
      connectivity_.emplace_back();
      ordinals_.emplace_back();

      STK_ThrowRequireMsg(connectivity_.size() < INVALID_UPWARDCONN_INDEX,
                          "Internal error, LinkedBucketConn size exceeds limitation of index type");

      return insert_connectivity_at_idx(indices, insert_idx, entity, ordinal);
    }

    if (!is_valid(ordinals_[indices.second])) {
      bool didInsert = insert_connectivity_into_sorted_range(indices, entity, ordinal);
      if (didInsert) {
        --num_unused_entities_;
      }
      return didInsert;
    } else {
      const int insert_idx = find_sorted_insertion_index(indices, entity, ordinal);
      if (insert_idx < 0) {
        return false;
      }

      const UpwardConnIndexType distance_moved = move_connectivity_to_end(bucket_ord);
      connectivity_.emplace_back();
      ordinals_.emplace_back();

      STK_ThrowRequireMsg(connectivity_.size() < INVALID_UPWARDCONN_INDEX,
                          "Internal error, LinkedBucketConn size exceeds limitation of index type");

      return insert_connectivity_at_idx(indices, (insert_idx + distance_moved), entity, ordinal);
    }

    return false;
  }

  int find_sorted_insertion_index(const IndexRange& indices, Entity entity, ConnectivityOrdinal ordinal) {
    const ConnectivityOrdinal* beg = ordinals_.data() + indices.first;
    const ConnectivityOrdinal* end = ordinals_.data() + indices.second;
    const ConnectivityOrdinal* it = std::lower_bound(beg, end, ordinal);
    if (it != end) {
      int idx = indices.first + (it - beg);
      if (*it == ordinal) {
        for (; static_cast<unsigned>(idx) < indices.second && ordinals_[idx] == ordinal; ++idx) {
          if (entity < connectivity_[idx]) {
            return idx;
          }
          if (connectivity_[idx] == entity) {
            return -1;
          }
        }
      }
      return idx;
    }
    return indices.second;
  }

  bool insert_connectivity_at_idx(IndexRange& indices, int insert_idx, Entity entity, ConnectivityOrdinal ordinal) {
    if (insert_idx < 0) {
      return false;
    }
    unsigned uinsert_idx = static_cast<unsigned>(insert_idx);
    for (UpwardConnIndexType i = indices.second; i > uinsert_idx; --i) {
      connectivity_[i] = connectivity_[i - 1];
      ordinals_[i] = ordinals_[i - 1];
    }

    connectivity_[insert_idx] = entity;
    ordinals_[insert_idx] = ordinal;
    ++indices.second;
    return true;
  }

  bool insert_connectivity_into_sorted_range(IndexRange& indices, Entity entity, ConnectivityOrdinal ordinal) {
    return insert_connectivity_at_idx(indices, find_sorted_insertion_index(indices, entity, ordinal), entity, ordinal);
  }

  UpwardConnIndexType move_connectivity_to_end(unsigned bucket_ord) {
    IndexRange& indices = offsets_[bucket_ord];
    const UpwardConnIndexType new_start_idx = connectivity_.size();

    for (UpwardConnIndexType i = indices.first; i < indices.second; ++i) {
      connectivity_.push_back(connectivity_[i]);
      ordinals_.push_back(ordinals_[i]);
      ordinals_[i] = INVALID_CONNECTIVITY_ORDINAL;
    }

    STK_ThrowRequireMsg(connectivity_.size() < INVALID_UPWARDCONN_INDEX,
                        "Internal error, LinkedBucketConn size exceeds limitation of index type");

    num_unused_entities_ += indices.second - indices.first;
    const UpwardConnIndexType distance_moved = new_start_idx - indices.first;
    offsets_[bucket_ord] = IndexRange(new_start_idx, connectivity_.size());
    return distance_moved;
  }
  //@}

  //!\name Private members
  //@{

  unsigned bucket_capacity_;
  std::vector<IndexRange> offsets_;
  std::vector<Entity> connectivity_;
  std::vector<ConnectivityOrdinal> ordinals_;
  unsigned num_unused_entities_;
  int compression_threshold_;
  //@}
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#undef BUCKET_ORDINAL_ERROR_MESSAGE
#endif  // MUNDY_MESH_IMPL_LINKEDBUCKETCONN_HPP_
