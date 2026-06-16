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

/// \file UnitTestLinkCSRPartition.cpp
/// \brief Unit tests for LinkCSRPartitionT — the per-partition bucket-level CSR container.
///
/// Priority targets:
///   - Correctness of the selector_view_ fix (C1 in LINK_ANALYSIS.md):
///       * default-constructed partition throws on selector() access
///       * copied partitions share the selector allocation without double-free
///       * deep_copy(dest, src) produces an independent selector value
///   - Metadata accessors: id(), link_rank(), link_dimensionality(), ngp_key(), contains()
///   - Pre-CSR-update state: get_connected_links returns empty for all entity offsets

// External libs
#include <gtest/gtest.h>

// C++ core libs
#include <memory>     // for std::shared_ptr
#include <stdexcept>  // for std::logic_error

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/LinkCSRPartition.hpp>  // for LinkCSRPartitionT, deep_copy
#include <mundy_mesh/LinkMetaData.hpp>      // for declare_link_meta_data
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/impl/PartitionKey.hpp>  // for impl::get_partition_key

namespace mundy {

namespace mesh {

namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Mesh with two constraint-rank link parts (dim2 and dim3) and one entity in each
/// so that non-empty, deterministic buckets exist for both partition keys.
struct LinkCSRPartitionFixture {
  static constexpr unsigned kCapacity = 8;

  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;
  LinkMetaData* link_meta{nullptr};
  stk::mesh::Part* link_part_dim2{nullptr};
  stk::mesh::Part* link_part_dim3{nullptr};
  stk::mesh::Part* unrelated_part{nullptr};

  stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;

  // Entities declared during setup
  stk::mesh::Entity link_dim2;
  stk::mesh::Entity link_dim3;

  LinkCSRPartitionFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    builder.set_initial_bucket_capacity(kCapacity);
    builder.set_maximum_bucket_capacity(kCapacity);

    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);

    link_meta = &declare_link_meta_data(*meta, "PART_TEST_LINKS", link_rank);
    link_part_dim2 = &link_meta->declare_link_part("PART_TEST_LINK_DIM2", 2u);
    link_part_dim3 = &link_meta->declare_link_part("PART_TEST_LINK_DIM3", 3u);
    unrelated_part = &meta->declare_part("UNRELATED_PART", stk::topology::NODE_RANK);

    meta->commit();

    bulk->modification_begin();
    link_dim2 = bulk->declare_entity(link_rank, 1u, stk::mesh::PartVector{link_part_dim2});
    link_dim3 = bulk->declare_entity(link_rank, 2u, stk::mesh::PartVector{link_part_dim3});
    // Also declare a node so NODE_RANK has at least one bucket
    bulk->declare_node(1u);
    bulk->modification_end();
  }

  /// Returns the partition key for the bucket containing link_dim2 (dim2-only bucket).
  impl::PartitionKey key_for_dim2() const {
    const stk::mesh::Bucket& b = bulk->bucket(link_dim2);
    return impl::get_partition_key(b);
  }

  /// Returns the partition key for the bucket containing link_dim3 (dim3-only bucket).
  impl::PartitionKey key_for_dim3() const {
    const stk::mesh::Bucket& b = bulk->bucket(link_dim3);
    return impl::get_partition_key(b);
  }
};

using LinkCSRPartition = LinkCSRPartitionT<stk::ngp::HostMemSpace>;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRPartition, DefaultConstruction) {
  LinkCSRPartition p;
  EXPECT_EQ(p.link_rank(), stk::topology::INVALID_RANK);
  EXPECT_EQ(p.link_dimensionality(), 0u);
  EXPECT_EQ(p.ngp_key().extent(0), 0u);
  EXPECT_THROW(p.selector(), std::logic_error)
      << "selector() on a default-constructed partition should throw because selector_view_ is empty";
}

// ---------------------------------------------------------------------------
// Full construction: metadata must reflect constructor arguments
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRPartition, FullConstruction_ScalarMetadataMatchesArguments) {
  LinkCSRPartitionFixture f;
  const impl::PartitionKey key = f.key_for_dim2();
  LinkCSRPartition p(42u, key, f.link_rank, 2u, *f.bulk);
  EXPECT_EQ(p.id(), 42u);
  EXPECT_EQ(p.link_rank(), f.link_rank);
  EXPECT_EQ(p.link_dimensionality(), 2u);
  EXPECT_EQ(p.ngp_key().extent(0), key.size());
}

TEST(UnitTestLinkCSRPartition, FullConstruction_NgpKeyReflectsPartitionKey) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition p(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  EXPECT_TRUE(p.contains(f.link_part_dim2->mesh_meta_data_ordinal()));
  EXPECT_FALSE(p.contains(f.link_part_dim3->mesh_meta_data_ordinal()));
  EXPECT_FALSE(p.contains(f.unrelated_part->mesh_meta_data_ordinal()));
}

TEST(UnitTestLinkCSRPartition, FullConstruction_SelectorMatchesPartitionKey) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition p(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  EXPECT_TRUE(p.selector()(f.bulk->bucket(f.link_dim2)));
  EXPECT_FALSE(p.selector()(f.bulk->bucket(f.link_dim3)));
}

// ---------------------------------------------------------------------------
// Pre-CSR-update state
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRPartition, BeforeCsrUpdate_InitialState) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition p(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  EXPECT_GT(p.num_buckets(stk::topology::NODE_RANK), 0u);
  const stk::mesh::Bucket& dim2_bucket = f.bulk->bucket(f.link_dim2);
  const stk::mesh::FastMeshIndex idx{dim2_bucket.bucket_id(), 0u};
  EXPECT_EQ(p.get_connected_links(f.link_rank, idx).size(), 0u);
}

// ---------------------------------------------------------------------------
// selector_view_ copy/lifetime semantics — the C1 fix (LINK_ANALYSIS.md)
// ---------------------------------------------------------------------------

// After a copy, the copy must select the correct buckets (not just compare equal to
// the original — they share the same selector object so that comparison is reflexive).
// Verify the selector VALUE by checking both a positive and a negative case.
TEST(UnitTestLinkCSRPartition, CopyConstructor_CopyHasSameSelectorValue) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition original(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  LinkCSRPartition copy = original;

  const stk::mesh::Bucket& dim2_bucket = f.bulk->bucket(f.link_dim2);
  const stk::mesh::Bucket& dim3_bucket = f.bulk->bucket(f.link_dim3);
  EXPECT_TRUE(copy.selector()(dim2_bucket)) << "copy selector must select the dim2 bucket it was built from";
  EXPECT_FALSE(copy.selector()(dim3_bucket))
      << "copy selector must not select the dim3 bucket — selector holds the wrong value";
}

// After the copy is destroyed, the original must still have a valid, accessible selector.
// This is the key regression test for the ref-counted selector_view_ fix:
// the previous raw-pointer approach caused a double-free here.
TEST(UnitTestLinkCSRPartition, CopyConstructor_OriginalSelectorValidAfterCopyDestruction) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition original(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  const stk::mesh::Bucket& dim2_bucket = f.bulk->bucket(f.link_dim2);

  {
    LinkCSRPartition copy = original;
    (void)copy;
    // copy destructs here — should decrement ref count, not delete the allocation
  }

  // original's selector should still be valid and accessible
  EXPECT_NO_THROW(original.selector())
      << "Original selector became inaccessible after copy was destroyed — likely a double-free";
  EXPECT_TRUE(original.selector()(dim2_bucket)) << "Original selector has wrong value after copy was destroyed";
}

// deep_copy(dest, src) must faithfully copy all metadata and the selector VALUE.
TEST(UnitTestLinkCSRPartition, DeepCopy_DestReflectsSrc) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition src(7u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  LinkCSRPartition dest;
  deep_copy(dest, src);

  EXPECT_EQ(dest.id(), src.id());
  EXPECT_EQ(dest.link_rank(), src.link_rank());
  EXPECT_EQ(dest.link_dimensionality(), src.link_dimensionality());
  EXPECT_EQ(dest.ngp_key().extent(0), src.ngp_key().extent(0));
  EXPECT_EQ(dest.selector()(f.bulk->bucket(f.link_dim2)), src.selector()(f.bulk->bucket(f.link_dim2)));
}

// After deep_copy, destroying dest must not corrupt src's selector allocation (independent lifetimes).
TEST(UnitTestLinkCSRPartition, DeepCopy_SrcSelectorValidAfterDestDestruction) {
  LinkCSRPartitionFixture f;
  LinkCSRPartition src(0u, f.key_for_dim2(), f.link_rank, 2u, *f.bulk);
  const stk::mesh::Bucket& dim2_bucket = f.bulk->bucket(f.link_dim2);

  {
    LinkCSRPartition dest;
    deep_copy(dest, src);
    (void)dest;
    // dest destructs here — must not touch src's selector allocation
  }

  EXPECT_NO_THROW(src.selector()) << "src selector became inaccessible after deep_copy dest was destroyed";
  EXPECT_TRUE(src.selector()(dim2_bucket));
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
