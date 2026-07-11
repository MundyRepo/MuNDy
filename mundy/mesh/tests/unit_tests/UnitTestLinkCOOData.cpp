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

/// \file UnitTestLinkCOOData.cpp
/// \brief Unit tests for LinkCOOData — the host-side COO link connectivity.
///
/// These tests focus on the host-side `LinkCOOData` class directly (not through `LinkData`).
/// Internal state that is not part of the public API is inspected via the `impl::` free
/// functions declared in LinkCOOData.hpp (e.g. impl::get_linked_entity_crs,
/// impl::get_link_crs_needs_updated).
///
/// Notable invariant under test (from LINK_ANALYSIS.md S1 and the COO design):
///   destroy_relation() intentionally does NOT update the CRS-snapshot field
///   (linked_entities_crs_field_). This stale CRS snapshot is what the CSR synchronizer
///   uses to detect a change and know it must remove the old connection from the CSR
///   structure. Tests verify this behavior explicitly.

// External libs
#include <gtest/gtest.h>

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::parallel_for, RangePolicy, KOKKOS_LAMBDA, fence

// C++ core libs
#include <memory>     // for std::shared_ptr
#include <stdexcept>  // for std::invalid_argument

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Types.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/GetNgpLinkData.hpp>  // for get_updated_ngp_link_data
#include <mundy_mesh/LinkCOOData.hpp>     // for LinkCOOData, NgpLinkCOODataT
#include <mundy_mesh/LinkData.hpp>        // for declare_link_data (needed for CSR sync tests)
#include <mundy_mesh/LinkMetaData.hpp>    // for declare_link_meta_data, impl::get_* fields
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpLinkData.hpp>  // for NgpLinkData

namespace mundy {

namespace mesh {

namespace {

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Mesh with a constraint-rank link part (dimensionality 2), one link entity,
/// and two node-rank target entities.  The link entity and targets are declared
/// in a modification cycle; no COO relations are declared during setup so that
/// each test starts from a clean relation-free state.
struct LinkCOODataFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;
  LinkMetaData* link_meta{nullptr};
  stk::mesh::Part* link_part{nullptr};

  stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;

  stk::mesh::Entity link;
  stk::mesh::Entity target0;  // node
  stk::mesh::Entity target1;  // node

  LinkCOODataFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);

    link_meta = &declare_link_meta_data(*meta, "COO_TEST_LINKS", link_rank);
    link_part = &link_meta->declare_link_part("COO_TEST_LINK_PART", 2u);

    meta->commit();

    bulk->modification_begin();
    link = bulk->declare_entity(link_rank, 1u, stk::mesh::PartVector{link_part});
    target0 = bulk->declare_node(1u);
    target1 = bulk->declare_node(2u);
    bulk->modification_end();
  }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCOOData, Construction_ValidityReflectsSetup) {
  LinkCOOData coo_default;
  EXPECT_FALSE(coo_default.is_valid());

  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);
  EXPECT_TRUE(coo.is_valid());
}

// ---------------------------------------------------------------------------
// Precondition guards (MUNDY_THROW_ASSERT — debug builds only)
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCOOData, WrongRankLinkerThrows) {
#ifdef NDEBUG
  GTEST_SKIP() << "MUNDY_THROW_ASSERT is disabled in non-debug builds (NDEBUG defined).";
#endif
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);
  EXPECT_THROW(coo.declare_relation(f.target0, f.target1, 0u), std::exception);
  EXPECT_THROW(coo.destroy_relation(f.target0, 0u), std::exception);
}

// ---------------------------------------------------------------------------
// declare_relation: field writes and accessors
// ---------------------------------------------------------------------------

// All COO field accessors return correct values after declare_relation, across both ordinals.
TEST(UnitTestLinkCOOData, DeclareRelation_WritesAllFieldsCorrectly) {
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);
  coo.declare_relation(f.link, f.target0, 0u);
  coo.declare_relation(f.link, f.target1, 1u);

  EXPECT_EQ(coo.get_linked_entity(f.link, 0u), f.target0);
  EXPECT_EQ(coo.get_linked_entity(f.link, 1u), f.target1);
  EXPECT_EQ(coo.get_linked_entity_id(f.link, 0u), f.bulk->identifier(f.target0));
  EXPECT_EQ(coo.get_linked_entity_rank(f.link, 0u), stk::topology::NODE_RANK);
  const stk::mesh::FastMeshIndex idx = coo.get_linked_entity_index(f.link, 0u);
  EXPECT_EQ(idx.bucket_id, f.bulk->bucket(f.target0).bucket_id());
  EXPECT_EQ(idx.bucket_ord, f.bulk->bucket_ordinal(f.target0));
  EXPECT_TRUE(impl::get_link_crs_needs_updated(coo, f.link));
}

// After declare_relation, the CRS snapshot field should still hold the default-invalid
// value because no CSR sync has been performed.
TEST(UnitTestLinkCOOData, DeclareRelation_DoesNotTouchCrsSnapshotField) {
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);

  // Capture initial CRS snapshot value (should be default-invalid)
  const stk::mesh::Entity crs_before = impl::get_linked_entity_crs(coo, f.link, 0u);
  coo.declare_relation(f.link, f.target0, 0u);
  const stk::mesh::Entity crs_after = impl::get_linked_entity_crs(coo, f.link, 0u);

  EXPECT_EQ(crs_before, crs_after)
      << "declare_relation must not write the CRS snapshot field — that field is owned by the CSR synchronizer";
}

// A second declare at the same ordinal should overwrite, not append.
TEST(UnitTestLinkCOOData, DeclareRelation_OverwritesExistingRelation) {
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);
  coo.declare_relation(f.link, f.target0, 0u);
  EXPECT_EQ(coo.get_linked_entity(f.link, 0u), f.target0);

  coo.declare_relation(f.link, f.target1, 0u);
  EXPECT_EQ(coo.get_linked_entity(f.link, 0u), f.target1)
      << "A second declare_relation at the same ordinal should overwrite the first";
}

// ---------------------------------------------------------------------------
// destroy_relation: field writes and the intentional CRS-snapshot invariant
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCOOData, DestroyRelation_ClearsEntityAndSetsDirtyFlag) {
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);
  coo.declare_relation(f.link, f.target0, 0u);
  coo.destroy_relation(f.link, 0u);
  EXPECT_FALSE(f.bulk->is_valid(coo.get_linked_entity(f.link, 0u)));
  EXPECT_TRUE(impl::get_link_crs_needs_updated(coo, f.link));
}

// KEY INVARIANT: destroy_relation must NOT clear the CRS snapshot field.
// The CSR synchronizer uses a stale snapshot to detect removals: COO field goes
// invalid while CRS snapshot still holds the old entity.  If destroy cleared both,
// the synchronizer would see no change and the stale CSR entry would persist forever.
TEST(UnitTestLinkCOOData, DestroyRelation_PreservesCrsSnapshotField) {
  LinkCOODataFixture f;
  LinkCOOData coo(*f.bulk, *f.link_meta);

  // Snapshot the CRS field before any operation
  const stk::mesh::Entity crs_before_declare = impl::get_linked_entity_crs(coo, f.link, 0u);
  coo.declare_relation(f.link, f.target0, 0u);
  const stk::mesh::Entity crs_after_declare = impl::get_linked_entity_crs(coo, f.link, 0u);
  coo.destroy_relation(f.link, 0u);
  const stk::mesh::Entity crs_after_destroy = impl::get_linked_entity_crs(coo, f.link, 0u);

  // declare_relation must not touch the CRS snapshot
  EXPECT_EQ(crs_before_declare, crs_after_declare) << "declare_relation should not update the CRS snapshot field";

  // destroy_relation must also not touch the CRS snapshot
  EXPECT_EQ(crs_after_declare, crs_after_destroy)
      << "destroy_relation must preserve the CRS snapshot field (detection mechanism)";

  // The COO field should now be invalid (cleared by destroy)
  EXPECT_FALSE(f.bulk->is_valid(coo.get_linked_entity(f.link, 0u)));
}

// After running a full CSR sync, the CRS snapshot field reflects the declared relation.
// A subsequent destroy_relation clears the COO field but leaves the CRS snapshot
// pointing to the old target — confirming the snapshot is useful for change detection.
TEST(UnitTestLinkCOOData, CrsSnapshotReflectsRelationAfterSync_ThenPreservedAfterDestroy) {
  LinkCOODataFixture f;

  // Use the full LinkData machinery to perform a CSR sync
  LinkData& link_data = declare_link_data(*f.bulk, *f.link_meta);
  link_data.coo_data().declare_relation(f.link, f.target0, 0u);
  link_data.coo_modify_on_host();

  NgpLinkData& ngp = get_updated_ngp_link_data(link_data);
  ngp.coo_sync_to_device();
  ngp.update_crs_from_coo();
  link_data.crs_sync_to_host();

  // After CSR sync, the CRS snapshot should hold target0
  EXPECT_EQ(impl::get_linked_entity_crs(link_data.coo_data(), f.link, 0u), f.target0)
      << "After update_crs_from_coo the CRS snapshot field should record the current relation";

  // Now destroy the relation
  link_data.coo_data().destroy_relation(f.link, 0u);

  // COO field must be invalid; CRS snapshot must STILL hold target0
  EXPECT_FALSE(f.bulk->is_valid(link_data.coo_data().get_linked_entity(f.link, 0u)))
      << "COO field should be cleared by destroy_relation";
  EXPECT_EQ(impl::get_linked_entity_crs(link_data.coo_data(), f.link, 0u), f.target0)
      << "CRS snapshot must be preserved by destroy_relation to enable change detection";
}

// ---------------------------------------------------------------------------
// NgpLinkCOODataT: construction and basic reads
// ---------------------------------------------------------------------------

TEST(UnitTestLinkCOOData, NgpCOOData_Construction_IsValid) {
  LinkCOODataFixture f;
  EXPECT_TRUE(NgpLinkCOOData(*f.bulk, *f.link_meta).is_valid());
  LinkCOOData host_coo(*f.bulk, *f.link_meta);
  EXPECT_TRUE(NgpLinkCOOData(host_coo).is_valid());
}

// After declaring a relation on the host and syncing to device, a kernel must read
// back the correct entity and rank.

void cache_result(const NgpLinkCOOData& ngp_coo, stk::mesh::FastMeshIndex link_idx,
                  Kokkos::View<stk::mesh::Entity*, stk::ngp::MemSpace> entity_result,
                  Kokkos::View<stk::mesh::EntityRank*, stk::ngp::MemSpace> rank_result) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<stk::ngp::ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
        entity_result(0) = ngp_coo.get_linked_entity(link_idx, 0u);
        rank_result(0) = ngp_coo.get_linked_entity_rank(link_idx, 0u);
      });
  Kokkos::fence();
}

TEST(UnitTestLinkCOOData, NgpCOOData_ReflectsHostRelationsAfterSync) {
  LinkCOODataFixture f;

  LinkData& link_data = declare_link_data(*f.bulk, *f.link_meta);
  link_data.coo_data().declare_relation(f.link, f.target0, 0u);
  link_data.coo_modify_on_host();

  NgpLinkData& ngp = get_updated_ngp_link_data(link_data);
  ngp.coo_sync_to_device();

  NgpLinkCOOData ngp_coo = ngp.coo_data();
  const stk::mesh::FastMeshIndex link_idx = ngp.ngp_mesh().fast_mesh_index(f.link);
  Kokkos::View<stk::mesh::Entity*, stk::ngp::MemSpace> entity_result("entity_result", 1);
  Kokkos::View<stk::mesh::EntityRank*, stk::ngp::MemSpace> rank_result("rank_result", 1);
  cache_result(ngp_coo, link_idx, entity_result, rank_result);

  auto entity_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, entity_result);
  auto rank_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank_result);
  EXPECT_EQ(entity_host(0), f.target0);
  EXPECT_EQ(rank_host(0), link_data.coo_data().get_linked_entity_rank(f.link, 0u));
}

// A destroy_relation executed inside a device kernel must be visible on the host
// after coo_sync_to_host().

void destroy_relation_on_device(const NgpLinkCOOData& ngp_coo, stk::mesh::FastMeshIndex link_idx) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<stk::ngp::ExecSpace>(0, 1), KOKKOS_LAMBDA(int) { ngp_coo.destroy_relation(link_idx, 0u); });
  Kokkos::fence();
}

TEST(UnitTestLinkCOOData, NgpCOOData_DestroyRelationOnDevice_ReflectedAfterSyncBack) {
  LinkCOODataFixture f;

  LinkData& link_data = declare_link_data(*f.bulk, *f.link_meta);
  link_data.coo_data().declare_relation(f.link, f.target0, 0u);
  link_data.coo_modify_on_host();

  NgpLinkData& ngp = get_updated_ngp_link_data(link_data);
  ngp.coo_sync_to_device();

  NgpLinkCOOData ngp_coo = ngp.coo_data();
  const stk::mesh::FastMeshIndex link_idx = ngp.ngp_mesh().fast_mesh_index(f.link);

  destroy_relation_on_device(ngp_coo, link_idx);

  ngp.coo_modify_on_device();
  ngp.coo_sync_to_host();

  EXPECT_FALSE(f.bulk->is_valid(link_data.coo_data().get_linked_entity(f.link, 0u)))
      << "After device kernel destroy_relation synced back to host, the relation should be invalid";
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
