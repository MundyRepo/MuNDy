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

/// \file UnitTestLinkCSRBucketConn.cpp
/// \brief Unit tests for LinkCSRBucketConnT — the leaf per-bucket CSR connectivity object.
///
/// These tests exercise the class in isolation, constructing directly from a stk::mesh::Bucket.
/// They do NOT run a CSR update (which would require the full NgpLinkData machinery); they only
/// verify construction metadata, the pre-update zero state, and deep_copy behaviour.
///
/// If a test fails, that is informative — it signals a divergence between the API contract
/// and the implementation.

// External libs
#include <gtest/gtest.h>

// C++ core libs
#include <memory>   // for std::shared_ptr
#include <sstream>  // for std::ostringstream (dump() smoke test)

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/LinkCSRBucketConn.hpp>  // for LinkCSRBucketConnT, deep_copy, impl::get_*
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>

namespace mundy {

namespace mesh {

namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Minimal mesh providing a single node bucket with a deterministic size and capacity.
///
/// A bucket capacity of 8 is forced so that the views allocated inside
/// LinkCSRBucketConn have a known, non-trivial extent.
struct BucketConnFixture {
  static constexpr unsigned kCapacity = 8;
  static constexpr unsigned kNumNodes = 4;

  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;

  BucketConnFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    builder.set_initial_bucket_capacity(kCapacity);
    builder.set_maximum_bucket_capacity(kCapacity);

    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);
    meta->commit();

    bulk->modification_begin();
    for (unsigned i = 1; i <= kNumNodes; ++i) {
      bulk->declare_node(i);
    }
    bulk->modification_end();
  }

  /// Returns a reference to the first non-empty node bucket.
  const stk::mesh::Bucket& node_bucket() const {
    for (const stk::mesh::Bucket* b : bulk->buckets(stk::topology::NODE_RANK)) {
      if (b->size() > 0) return *b;
    }
    ADD_FAILURE() << "No non-empty NODE bucket found — fixture setup incomplete.";
    return *bulk->buckets(stk::topology::NODE_RANK)[0];
  }
};

// Convenience alias: tests only target HostMemSpace to avoid device-sync complexity.
using BucketConn = LinkCSRBucketConnT<stk::ngp::HostMemSpace>;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, DefaultConstruction) {
  BucketConn conn;
  EXPECT_EQ(conn.bucket_id(), 0u);
  EXPECT_EQ(conn.size(), 0u);
  EXPECT_EQ(conn.capacity(), 0u);
  EXPECT_EQ(conn.bucket_rank(), stk::topology::INVALID_RANK);
  EXPECT_EQ(conn.total_num_connected_links(), 0u);
  EXPECT_EQ(impl::get_num_connected_links(conn).extent(0), 0u);
  EXPECT_EQ(impl::get_sparse_connectivity_offsets(conn).extent(0), 0u);
  EXPECT_EQ(impl::get_sparse_connectivity(conn).extent(0), 0u);
}

// ---------------------------------------------------------------------------
// Bucket-based construction: metadata and view extents must mirror the source bucket
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, BucketConstruction_MetadataMatchesBucket) {
  BucketConnFixture f;
  const stk::mesh::Bucket& bucket = f.node_bucket();
  BucketConn conn(bucket);
  EXPECT_EQ(conn.bucket_id(), bucket.bucket_id());
  EXPECT_EQ(conn.size(), static_cast<unsigned>(bucket.size()));
  EXPECT_EQ(conn.capacity(), static_cast<unsigned>(bucket.capacity()));
  EXPECT_EQ(conn.bucket_rank(), bucket.entity_rank());
  EXPECT_EQ(impl::get_num_connected_links(conn).extent(0), static_cast<size_t>(bucket.capacity()));
  EXPECT_EQ(impl::get_sparse_connectivity_offsets(conn).extent(0), static_cast<size_t>(bucket.capacity() + 1));
}

// ---------------------------------------------------------------------------
// Pre-CSR-update zero state
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, BeforeCsrUpdate_InitialState) {
  BucketConnFixture f;
  BucketConn conn(f.node_bucket());
  EXPECT_EQ(conn.total_num_connected_links(), 0u);
  for (unsigned i = 0; i < conn.size(); ++i) {
    EXPECT_EQ(conn.num_connected_links(i), 0u) << "slot " << i;
    EXPECT_EQ(conn.get_connected_links(i).size(), 0u) << "slot " << i;
  }
}

// ---------------------------------------------------------------------------
// deep_copy: metadata preserved; backing views are independent allocations
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, DeepCopy_DestReflectsSrc) {
  BucketConnFixture f;
  BucketConn src(f.node_bucket());
  BucketConn dest;
  deep_copy(dest, src);

  EXPECT_EQ(dest.bucket_id(), src.bucket_id());
  EXPECT_EQ(dest.size(), src.size());
  EXPECT_EQ(dest.capacity(), src.capacity());
  EXPECT_EQ(dest.bucket_rank(), src.bucket_rank());
  EXPECT_EQ(dest.total_num_connected_links(), src.total_num_connected_links());
  EXPECT_EQ(impl::get_num_connected_links(dest).extent(0), impl::get_num_connected_links(src).extent(0));
}

TEST(UnitTestLinkCSRBucketConn, DeepCopy_ViewsAreIndependent) {
  BucketConnFixture f;
  BucketConn src(f.node_bucket());
  BucketConn dest;
  deep_copy(dest, src);

  auto& src_ncl = impl::get_num_connected_links(src);
  auto& dest_ncl = impl::get_num_connected_links(dest);
  auto& src_off = impl::get_sparse_connectivity_offsets(src);
  auto& dest_off = impl::get_sparse_connectivity_offsets(dest);

  ASSERT_GT(src_ncl.extent(0), 0u);
  src_ncl(0) = 999u;
  EXPECT_NE(dest_ncl(0), 999u) << "num_connected_links views must be independent after deep_copy";

  ASSERT_GT(src_off.extent(0), 0u);
  src_off(0) = 777u;
  EXPECT_NE(dest_off(0), 777u) << "offset views must be independent after deep_copy";
}

// ---------------------------------------------------------------------------
// Dirty flag
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, DirtyFlag_InitiallyFalseAndMutable) {
  BucketConnFixture f;
  BucketConn conn(f.node_bucket());
  EXPECT_EQ(impl::get_dirty_flag(conn), 0);
  impl::get_dirty_flag(conn) = 1;
  EXPECT_EQ(impl::get_dirty_flag(conn), 1);
}

// ---------------------------------------------------------------------------
// dump() smoke test — must not crash in either construction state
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRBucketConn, Dump_DoesNotCrash) {
  std::ostringstream sink;
  EXPECT_NO_THROW(BucketConn{}.dump(sink));
  BucketConnFixture f;
  BucketConn conn(f.node_bucket());
  EXPECT_NO_THROW(conn.dump(sink));
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
