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

/// \file UnitTestLinkCSRData.cpp
/// \brief Unit tests for LinkCSRDataT — the memoized per-selector CSR partition registry.
///
/// Tested invariants:
///   - get_or_create_crs_partitions() returns a view whose partitions reflect the
///     current mesh bucket layout for the given selector.
///   - Repeated calls with the same selector are memoized (same underlying data).
///   - get_all_crs_partitions() grows monotonically as new partition keys are discovered.
///   - clear_structural_caches() resets all partition state; subsequent calls rebuild.
///   - mark_crs_bucket_conns_dirty() is selector-scoped: only matching partitions are dirtied.

// External libs
#include <gtest/gtest.h>

// C++ core libs
#include <memory>     // for std::shared_ptr
#include <stdexcept>  // for std::logic_error

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/LinkCSRBucketConn.hpp>  // for impl::get_dirty_flag
#include <mundy_mesh/LinkCSRData.hpp>        // for LinkCSRDataT
#include <mundy_mesh/LinkMetaData.hpp>       // for declare_link_meta_data
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>

namespace mundy {

namespace mesh {

namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Mesh with two constraint-rank link parts (dim2 and dim3), one entity in each,
/// and one node.  The node guarantees that at least one NODE_RANK bucket exists so
/// dirty-flag tests have concrete bucket conns to inspect.
struct LinkCSRDataFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;
  LinkMetaData* link_meta{nullptr};
  stk::mesh::Part* link_part_dim2{nullptr};
  stk::mesh::Part* link_part_dim3{nullptr};
  stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;
  stk::mesh::Entity link_dim2;
  stk::mesh::Entity link_dim3;
  stk::mesh::Selector selector_dim2;
  stk::mesh::Selector selector_dim3;

  LinkCSRDataFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);

    link_meta = &declare_link_meta_data(*meta, "CSR_TEST_LINKS", link_rank);
    link_part_dim2 = &link_meta->declare_link_part("CSR_TEST_LINK_DIM2", 2u);
    link_part_dim3 = &link_meta->declare_link_part("CSR_TEST_LINK_DIM3", 3u);

    meta->commit();

    bulk->modification_begin();
    link_dim2 = bulk->declare_entity(link_rank, 1u, stk::mesh::PartVector{link_part_dim2});
    link_dim3 = bulk->declare_entity(link_rank, 2u, stk::mesh::PartVector{link_part_dim3});
    bulk->declare_node(1u);
    bulk->modification_end();

    selector_dim2 = stk::mesh::Selector(*link_part_dim2);
    selector_dim3 = stk::mesh::Selector(*link_part_dim3);
  }
};

using LinkCSRData = LinkCSRDataT<stk::ngp::HostMemSpace>;
using LinkCSRPartition = LinkCSRPartitionT<stk::ngp::HostMemSpace>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count of bucket conns across all ranks.
unsigned total_bucket_conns(const LinkCSRPartition& p) {
  unsigned total = 0;
  for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank)
    total += p.num_buckets(rank);
  return total;
}

/// True when every bucket conn across all ranks has dirty == true (vacuously true if none exist).
/// Callers that need a non-vacuous result should assert total_bucket_conns() > 0 first.
bool all_bucket_conns_dirty(const LinkCSRPartition& p) {
  for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank)
    for (unsigned i = 0; i < p.num_buckets(rank); ++i)
      if (!impl::get_dirty_flag(p.get_crs_bucket_conn(rank, i))) return false;
  return true;
}

/// True when no bucket conn across all ranks has dirty == true.
bool no_bucket_conns_dirty(const LinkCSRPartition& p) {
  for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank)
    for (unsigned i = 0; i < p.num_buckets(rank); ++i)
      if (impl::get_dirty_flag(p.get_crs_bucket_conn(rank, i))) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCSRData, Construction) {
  LinkCSRData csr_default;
  EXPECT_FALSE(csr_default.is_valid());

  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);
  EXPECT_TRUE(csr.is_valid());
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 0u);
}

// get_or_create_crs_partitions returns one partition per distinct bucket group.
// Each partition's contains() reflects the parts of the bucket it was built from.
TEST(UnitTestLinkCSRData, GetOrCreatePartitions_ReturnsPartitionsMatchingBuckets) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  const auto& partitions_dim2 = csr.get_or_create_crs_partitions(f.selector_dim2);
  ASSERT_EQ(partitions_dim2.extent(0), 1u);
  EXPECT_TRUE(partitions_dim2(0).contains(f.link_part_dim2->mesh_meta_data_ordinal()));
  EXPECT_FALSE(partitions_dim2(0).contains(f.link_part_dim3->mesh_meta_data_ordinal()));

  const auto& partitions_dim3 = csr.get_or_create_crs_partitions(f.selector_dim3);
  ASSERT_EQ(partitions_dim3.extent(0), 1u);
  EXPECT_FALSE(partitions_dim3(0).contains(f.link_part_dim2->mesh_meta_data_ordinal()));
  EXPECT_TRUE(partitions_dim3(0).contains(f.link_part_dim3->mesh_meta_data_ordinal()));
}

// A second call with the same selector returns the same underlying data — no rebuild.
TEST(UnitTestLinkCSRData, GetOrCreatePartitions_Memoized) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  const auto& first = csr.get_or_create_crs_partitions(f.selector_dim2);
  const auto& second = csr.get_or_create_crs_partitions(f.selector_dim2);
  EXPECT_EQ(first.data(), second.data())
      << "second call with the same selector must return the memoized view, not a new allocation";
}

// get_all_crs_partitions grows by one each time a selector introduces a new bucket group,
// and does not grow on repeated calls with an already-registered selector.
TEST(UnitTestLinkCSRData, GetAllCrsPartitions_GrowsMonotonically) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 0u);

  csr.get_or_create_crs_partitions(f.selector_dim2);
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 1u);

  csr.get_or_create_crs_partitions(f.selector_dim3);
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 2u);

  csr.get_or_create_crs_partitions(f.selector_dim2);  // memoized — no new partition
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 2u);
}

// clear_structural_caches resets to zero partitions; a subsequent get_or_create rebuilds.
TEST(UnitTestLinkCSRData, ClearStructuralCaches) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  csr.get_or_create_crs_partitions(f.selector_dim2);
  csr.get_or_create_crs_partitions(f.selector_dim3);
  ASSERT_EQ(csr.get_all_crs_partitions().extent(0), 2u);

  csr.clear_structural_caches();
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 0u);

  // Rebuild works after clear
  const auto& rebuilt = csr.get_or_create_crs_partitions(f.selector_dim2);
  EXPECT_EQ(rebuilt.extent(0), 1u);
  EXPECT_EQ(csr.get_all_crs_partitions().extent(0), 1u);
}

// mark_crs_bucket_conns_dirty is selector-scoped: only bucket conns in matching
// partitions are dirtied; other partitions remain clean.
TEST(UnitTestLinkCSRData, MarkCrsBucketConnsDirty_SelectorScoped) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  csr.get_or_create_crs_partitions(f.selector_dim2);
  csr.get_or_create_crs_partitions(f.selector_dim3);

  const auto& all_parts = csr.get_all_crs_partitions();
  ASSERT_EQ(all_parts.extent(0), 2u);
  ASSERT_TRUE(no_bucket_conns_dirty(all_parts(0))) << "partition 0 must start clean";
  ASSERT_TRUE(no_bucket_conns_dirty(all_parts(1))) << "partition 1 must start clean";

  csr.mark_crs_bucket_conns_dirty(f.selector_dim2);

  const auto& dim2_parts = csr.get_or_create_crs_partitions(f.selector_dim2);
  ASSERT_EQ(dim2_parts.extent(0), 1u);
  ASSERT_GT(total_bucket_conns(dim2_parts(0)), 0u) << "dim2 partition must have bucket conns for a non-vacuous check";
  EXPECT_TRUE(all_bucket_conns_dirty(dim2_parts(0)));

  const auto& dim3_parts = csr.get_or_create_crs_partitions(f.selector_dim3);
  ASSERT_EQ(dim3_parts.extent(0), 1u);
  EXPECT_TRUE(no_bucket_conns_dirty(dim3_parts(0))) << "marking dim2 dirty must not affect the dim3 partition";
}

// mark_all_crs_bucket_conns_dirty reaches every partition regardless of selector.
TEST(UnitTestLinkCSRData, MarkAllCrsBucketConnsDirty) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  csr.get_or_create_crs_partitions(f.selector_dim2);
  csr.get_or_create_crs_partitions(f.selector_dim3);

  csr.mark_all_crs_bucket_conns_dirty();

  const auto& all_parts = csr.get_all_crs_partitions();
  ASSERT_EQ(all_parts.extent(0), 2u);
  ASSERT_GT(total_bucket_conns(all_parts(0)), 0u) << "partition 0 must have bucket conns for a non-vacuous check";
  ASSERT_GT(total_bucket_conns(all_parts(1)), 0u) << "partition 1 must have bucket conns for a non-vacuous check";
  EXPECT_TRUE(all_bucket_conns_dirty(all_parts(0))) << "partition 0 must be dirty";
  EXPECT_TRUE(all_bucket_conns_dirty(all_parts(1))) << "partition 1 must be dirty";
}

// all_selector() is empty before any get_or_create call; it grows to cover every
// registered selector after registration.
TEST(UnitTestLinkCSRData, AllSelector) {
  LinkCSRDataFixture f;
  LinkCSRData csr(*f.bulk, *f.link_meta);

  const stk::mesh::Bucket& dim2_bucket = f.bulk->bucket(f.link_dim2);
  const stk::mesh::Bucket& dim3_bucket = f.bulk->bucket(f.link_dim3);

  // Before any registration: all_selector() selects nothing
  EXPECT_FALSE(csr.all_selector()(dim2_bucket));
  EXPECT_FALSE(csr.all_selector()(dim3_bucket));

  csr.get_or_create_crs_partitions(f.selector_dim2);
  EXPECT_TRUE(csr.all_selector()(dim2_bucket))
      << "all_selector must select dim2 buckets after registering selector_dim2";
  EXPECT_FALSE(csr.all_selector()(dim3_bucket))
      << "all_selector must not select dim3 buckets before registering selector_dim3";

  csr.get_or_create_crs_partitions(f.selector_dim3);
  EXPECT_TRUE(csr.all_selector()(dim2_bucket));
  EXPECT_TRUE(csr.all_selector()(dim3_bucket))
      << "all_selector must select dim3 buckets after registering selector_dim3";
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
