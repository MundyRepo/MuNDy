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

// External libs
#include <gtest/gtest.h>

// C++ core libs
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntitySorterBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Types.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/GetNgpLinkData.hpp>
#include <mundy_mesh/LinkData.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpLinkData.hpp>
#include <mundy_mesh/impl/LinkDataObserver.hpp>

namespace mundy {

namespace mesh {

namespace {

static constexpr unsigned default_bucket_capacity = 512;

struct LinkDataObserverFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta_data;
  std::shared_ptr<BulkData> bulk_data;
  LinkMetaData* link_meta_data{nullptr};
  LinkData* link_data{nullptr};
  stk::mesh::Part* link_part_dim2{nullptr};
  stk::mesh::Part* link_part_dim3{nullptr};
  stk::mesh::Part* linked_node_part{nullptr};
  std::vector<stk::mesh::EntityId> next_ids;

  explicit LinkDataObserverFixture(unsigned bucket_capacity = default_bucket_capacity, bool commit_mesh = true)
      : next_ids(stk::topology::NUM_RANKS, 0) {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    builder.set_initial_bucket_capacity(bucket_capacity);
    builder.set_maximum_bucket_capacity(bucket_capacity);

    meta_data = builder.create_meta_data();
    meta_data->use_simple_fields();
    bulk_data = builder.create_bulk_data(meta_data);

    link_meta_data = &declare_link_meta_data(*meta_data, "OBS_LINKS", stk::topology::CONSTRAINT_RANK);
    link_part_dim2 = &link_meta_data->declare_link_part("OBS_LINK_PART_DIM2", 2u);
    link_part_dim3 = &link_meta_data->declare_link_part("OBS_LINK_PART_DIM3", 3u);
    linked_node_part = &meta_data->declare_part("OBS_LINKED_NODE_PART", stk::topology::NODE_RANK);

    if (commit_mesh) {
      meta_data->commit();
    }

    link_data = &declare_link_data(*bulk_data, *link_meta_data);
  }

  stk::mesh::Entity declare_entity(stk::mesh::EntityRank rank, const stk::mesh::PartVector& parts = {}) {
    const size_t rank_index = static_cast<size_t>(rank);
    next_ids[rank_index] += 1;
    return bulk_data->declare_entity(rank, next_ids[rank_index], parts);
  }
};

using LinkTriple = std::array<stk::mesh::Entity, 3>;
using LinkQuad = std::array<stk::mesh::Entity, 4>;

LinkTriple create_connected_dim2_link(LinkDataObserverFixture& fixture,  //
                                      stk::mesh::EntityRank target_rank_0 = stk::topology::ELEM_RANK,
                                      stk::mesh::EntityRank target_rank_1 = stk::topology::ELEM_RANK,
                                      stk::mesh::Part* target_part_0 = nullptr,
                                      stk::mesh::Part* target_part_1 = nullptr) {
  fixture.bulk_data->modification_begin();

  stk::mesh::Entity link =
      fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim2});

  stk::mesh::PartVector target_parts_0;
  if (target_part_0 != nullptr) {
    target_parts_0.push_back(target_part_0);
  }

  stk::mesh::PartVector target_parts_1;
  if (target_part_1 != nullptr) {
    target_parts_1.push_back(target_part_1);
  }

  stk::mesh::Entity target_0 = fixture.declare_entity(target_rank_0, target_parts_0);
  stk::mesh::Entity target_1 = fixture.declare_entity(target_rank_1, target_parts_1);

  fixture.bulk_data->modification_end();

  fixture.link_data->coo_data().declare_relation(link, target_0, 0u);
  fixture.link_data->coo_data().declare_relation(link, target_1, 1u);
  fixture.link_data->coo_modify_on_host();

  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
  ngp_link_data.update_crs_from_coo();
  ngp_link_data.check_crs_coo_consistency();

  EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());

  return {link, target_0, target_1};
}

LinkQuad create_connected_dim3_link(LinkDataObserverFixture& fixture,  //
                                    stk::mesh::EntityRank target_rank_0 = stk::topology::ELEM_RANK,
                                    stk::mesh::EntityRank target_rank_1 = stk::topology::ELEM_RANK,
                                    stk::mesh::EntityRank target_rank_2 = stk::topology::ELEM_RANK) {
  fixture.bulk_data->modification_begin();

  stk::mesh::Entity link =
      fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim3});
  stk::mesh::Entity target_0 = fixture.declare_entity(target_rank_0, {});
  stk::mesh::Entity target_1 = fixture.declare_entity(target_rank_1, {});
  stk::mesh::Entity target_2 = fixture.declare_entity(target_rank_2, {});

  fixture.bulk_data->modification_end();

  fixture.link_data->coo_data().declare_relation(link, target_0, 0u);
  fixture.link_data->coo_data().declare_relation(link, target_1, 1u);
  fixture.link_data->coo_data().declare_relation(link, target_2, 2u);
  fixture.link_data->coo_modify_on_host();

  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
  ngp_link_data.update_crs_from_coo();
  ngp_link_data.check_crs_coo_consistency();

  EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());

  return {link, target_0, target_1, target_2};
}

NgpLinkData& expect_rebuild_then_update(LinkDataObserverFixture& fixture) {
  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
  EXPECT_FALSE(ngp_link_data.is_crs_up_to_date());
  EXPECT_NO_THROW(ngp_link_data.update_crs_from_coo());
  EXPECT_NO_THROW(ngp_link_data.check_crs_coo_consistency());

  EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());

  return ngp_link_data;
}

NgpLinkData& expect_rebuild_then_throw(LinkDataObserverFixture& fixture) {
  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
  EXPECT_FALSE(ngp_link_data.is_crs_up_to_date());
  EXPECT_THROW(ngp_link_data.update_crs_from_coo(), std::logic_error);
  EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));
  return ngp_link_data;
}

size_t count_links(const LinkDataObserverFixture& fixture) {
  stk::mesh::EntityVector links;
  fixture.bulk_data->get_entities(fixture.link_meta_data->link_rank(), fixture.link_meta_data->universal_link_class(),
                                  links);
  return links.size();
}

stk::mesh::Ordinal get_partition_id_for_dimensionality(const NgpLinkData& ngp_link_data, unsigned dimensionality) {
  const auto& partitions = ngp_link_data.crs_data().get_all_crs_partitions();
  for (size_t i = 0; i < partitions.extent(0); ++i) {
    const auto& partition = partitions(i);
    if (partition.link_dimensionality() == dimensionality) {
      return partition.id();
    }
  }
  ADD_FAILURE() << "Expected to find an active partition with dimensionality " << dimensionality;
  return stk::mesh::Ordinal(-1);
}

struct ReverseIdSorter : public stk::mesh::EntitySorterBase {
  void sort(stk::mesh::BulkData& bulk, stk::mesh::EntityVector& entity_vector) const override {
    std::sort(entity_vector.begin(), entity_vector.end(),
              [&bulk](const stk::mesh::Entity& lhs, const stk::mesh::Entity& rhs) {
                return bulk.identifier(lhs) > bulk.identifier(rhs);
              });
  }
};

TEST(UnitTestLinkDataObserver, LinkDeleted) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture);
    stk::mesh::Entity link = baseline[0];

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(link));
    fixture.bulk_data->modification_end();

    expect_rebuild_then_update(fixture);

    EXPECT_FALSE(fixture.bulk_data->is_valid(link));
    EXPECT_EQ(count_links(fixture), 0u);
  }
}

TEST(UnitTestLinkDataObserver, LinkedEntityDeleted) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture);
    stk::mesh::Entity link = baseline[0];
    stk::mesh::Entity deleted_linked_entity = baseline[1];
    stk::mesh::Entity surviving_linked_entity = baseline[2];

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(deleted_linked_entity));
    fixture.bulk_data->modification_end();

    expect_rebuild_then_throw(fixture);

    EXPECT_TRUE(fixture.bulk_data->is_valid(link));
    EXPECT_FALSE(fixture.bulk_data->is_valid(deleted_linked_entity));
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(link, 0u), deleted_linked_entity);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(link, 1u), surviving_linked_entity);
  }
}

TEST(UnitTestLinkDataObserver, EntireBucketDeleted) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture, stk::topology::NODE_RANK, stk::topology::ELEM_RANK,
                                                     fixture.linked_node_part, nullptr);
    stk::mesh::Entity link = baseline[0];
    stk::mesh::Entity bucket_entity_to_delete = baseline[1];

    const size_t num_node_buckets_before = fixture.bulk_data->buckets(stk::topology::NODE_RANK).size();
    ASSERT_GT(num_node_buckets_before, 0u);

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(bucket_entity_to_delete));
    fixture.bulk_data->modification_end();

    const size_t num_node_buckets_after = fixture.bulk_data->buckets(stk::topology::NODE_RANK).size();
    EXPECT_LT(num_node_buckets_after, num_node_buckets_before);

    expect_rebuild_then_throw(fixture);

    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(link, 0u), bucket_entity_to_delete);
  }
}

// When a linked entity changes parts it moves to a different STK bucket, invalidating the CRS
// structure.  The observer fires and marks the CRS dirty.  update_crs_from_coo() should then
// rebuild the CRS successfully: the entity handle and ID/rank are still valid, and
// get_linked_entity_index() always derives the current FastMeshIndex from the live entity rather
// than from a stale cached bucket position.
TEST(UnitTestLinkDataObserver, LinkedEntityChangedParts_CsrRebuildSucceeds) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture, stk::topology::NODE_RANK, stk::topology::NODE_RANK);
    stk::mesh::Entity linked_entity = baseline[1];

    fixture.bulk_data->modification_begin();
    fixture.bulk_data->change_entity_parts(linked_entity, stk::mesh::PartVector{fixture.linked_node_part});
    fixture.bulk_data->modification_end();

    expect_rebuild_then_update(fixture);
  }
}

TEST(UnitTestLinkDataObserver, LinkDeclared) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);
    const size_t num_links_before = count_links(fixture);

    fixture.bulk_data->modification_begin();
    stk::mesh::Entity new_link =
        fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim2});
    fixture.bulk_data->modification_end();

    expect_rebuild_then_update(fixture);

    EXPECT_TRUE(fixture.bulk_data->is_valid(new_link));
    EXPECT_EQ(count_links(fixture), num_links_before + 1u);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 0u), stk::mesh::Entity());
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 1u), stk::mesh::Entity());
  }
}

TEST(UnitTestLinkDataObserver, GetOrCreateCreatesPartitionsBeforeCsrUpdate) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);

    fixture.bulk_data->modification_begin();
    fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim2});
    fixture.bulk_data->modification_end();

    const auto& partitions = fixture.link_data->crs_data().get_or_create_crs_partitions(*fixture.link_part_dim2);
    EXPECT_EQ(partitions.extent(0), 1u);
    EXPECT_EQ(fixture.link_data->crs_data().get_all_crs_partitions().extent(0), 1u);
  }
}

TEST(UnitTestLinkDataObserver, LinkedEntityIdMismatchThrows) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture);
    stk::mesh::Entity link = baseline[0];

    auto& linked_entity_ids_field = impl::get_linked_entity_ids_field(*fixture.link_meta_data);
    stk::mesh::EntityId* linked_entity_ids = stk::mesh::field_data(linked_entity_ids_field, link);
    ASSERT_NE(linked_entity_ids, nullptr);
    linked_entity_ids[0] += 1u;

    impl::set_crs_structure_dirty(*fixture.link_data, true);
    expect_rebuild_then_throw(fixture);
  }
}

TEST(UnitTestLinkDataObserver, NonLinkPartChangeCallbackForcesRebuild) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture);
    stk::mesh::Entity non_link_entity = baseline[1];

    impl::LinkDataObserver observer(*fixture.bulk_data, *fixture.link_meta_data,
                                    impl::get_crs_structure_dirty_ref(*fixture.link_data));

    impl::set_crs_structure_dirty(*fixture.link_data, false);
    fixture.link_data->coo_sync_to_device();
    EXPECT_FALSE(fixture.link_data->coo_need_sync_to_device());

    observer.modification_begin_notification();
    observer.entity_parts_added(non_link_entity, stk::mesh::OrdinalVector{});
    observer.finished_modification_end_notification();

    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));
    EXPECT_FALSE(fixture.link_data->coo_need_sync_to_device());

    NgpLinkData& ngp_link_data = expect_rebuild_then_update(fixture);
    EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());
  }
}

TEST(UnitTestLinkDataObserver, MoveProcsCallbacksForceRebuild) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);

    impl::LinkDataObserver observer(*fixture.bulk_data, *fixture.link_meta_data,
                                    impl::get_crs_structure_dirty_ref(*fixture.link_data));

    impl::set_crs_structure_dirty(*fixture.link_data, false);
    observer.modification_begin_notification();
    observer.elements_about_to_move_procs_notification(stk::mesh::EntityProcVec{});
    observer.finished_modification_end_notification();
    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));

    expect_rebuild_then_update(fixture);
    EXPECT_FALSE(impl::get_crs_structure_dirty(*fixture.link_data));

    observer.modification_begin_notification();
    observer.elements_moved_procs_notification(stk::mesh::EntityProcVec{});
    observer.finished_modification_end_notification();
    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));

    expect_rebuild_then_update(fixture);
    EXPECT_FALSE(impl::get_crs_structure_dirty(*fixture.link_data));
  }
}

TEST(UnitTestLinkDataObserver, NewBucketDeclared) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(1u /*bucket capacity*/, commit_mesh);
    create_connected_dim2_link(fixture);

    const stk::mesh::EntityRank link_rank = fixture.link_meta_data->link_rank();
    const size_t num_link_buckets_before = fixture.bulk_data->buckets(link_rank).size();

    fixture.bulk_data->modification_begin();
    stk::mesh::Entity new_link =
        fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim2});
    fixture.bulk_data->modification_end();

    const size_t num_link_buckets_after = fixture.bulk_data->buckets(link_rank).size();
    EXPECT_GT(num_link_buckets_after, num_link_buckets_before);

    expect_rebuild_then_update(fixture);

    EXPECT_TRUE(fixture.bulk_data->is_valid(new_link));
  }
}

TEST(UnitTestLinkDataObserver, NewPartitionDeclared) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);

    NgpLinkData& baseline_ngp = get_updated_ngp_link_data(*fixture.link_data);
    const size_t num_partitions_before = baseline_ngp.crs_data().get_all_crs_partitions().extent(0);
    ASSERT_GT(num_partitions_before, 0u);

    fixture.bulk_data->modification_begin();
    stk::mesh::Entity new_link =
        fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim3});
    stk::mesh::Entity target_a = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    stk::mesh::Entity target_b = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    stk::mesh::Entity target_c = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    fixture.bulk_data->modification_end();

    fixture.link_data->coo_data().declare_relation(new_link, target_a, 0u);
    fixture.link_data->coo_data().declare_relation(new_link, target_b, 1u);
    fixture.link_data->coo_data().declare_relation(new_link, target_c, 2u);
    fixture.link_data->coo_modify_on_host();

    NgpLinkData& updated_ngp = expect_rebuild_then_update(fixture);
    const size_t num_partitions_after = updated_ngp.crs_data().get_all_crs_partitions().extent(0);

    EXPECT_GT(num_partitions_after, num_partitions_before);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 0u), target_a);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 1u), target_b);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 2u), target_c);
  }
}

TEST(UnitTestLinkDataObserver, NonLastLinkDeletionRebuildsRemainingPartition) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple first_link = create_connected_dim2_link(fixture);
    LinkTriple second_link = create_connected_dim2_link(fixture);

    NgpLinkData& baseline_ngp = get_updated_ngp_link_data(*fixture.link_data);
    ASSERT_EQ(baseline_ngp.crs_data().get_all_crs_partitions().extent(0), 1u);

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(first_link[0]));
    fixture.bulk_data->modification_end();

    NgpLinkData& updated_ngp = expect_rebuild_then_update(fixture);
    EXPECT_EQ(updated_ngp.crs_data().get_all_crs_partitions().extent(0), 1u);
    EXPECT_EQ(get_partition_id_for_dimensionality(updated_ngp, 2u), 0u);
    EXPECT_TRUE(fixture.bulk_data->is_valid(second_link[0]));
  }
}

TEST(UnitTestLinkDataObserver, RemovedPartitionCanBeRecreatedAfterFullRebuild) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);
    LinkQuad dim3_link = create_connected_dim3_link(fixture);

    NgpLinkData& baseline_ngp = get_updated_ngp_link_data(*fixture.link_data);
    ASSERT_EQ(baseline_ngp.crs_data().get_all_crs_partitions().extent(0), 2u);

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(dim3_link[0]));
    fixture.bulk_data->modification_end();

    NgpLinkData& after_delete_ngp = expect_rebuild_then_update(fixture);
    ASSERT_EQ(after_delete_ngp.crs_data().get_all_crs_partitions().extent(0), 1u);
    EXPECT_EQ(get_partition_id_for_dimensionality(after_delete_ngp, 2u), 0u);

    LinkQuad reactivated_dim3_link = create_connected_dim3_link(fixture);
    NgpLinkData& after_reactivate_ngp = get_updated_ngp_link_data(*fixture.link_data);
    EXPECT_EQ(after_reactivate_ngp.crs_data().get_all_crs_partitions().extent(0), 2u);
    EXPECT_NO_THROW(after_reactivate_ngp.check_crs_coo_consistency());
    EXPECT_TRUE(fixture.bulk_data->is_valid(reactivated_dim3_link[0]));
  }
}

TEST(UnitTestLinkDataObserver, ObserverDoesNotMutateHostCooSyncState) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    LinkTriple baseline = create_connected_dim2_link(fixture);
    stk::mesh::Entity deleted_linked_entity = baseline[1];

    fixture.link_data->coo_sync_to_device();
    EXPECT_FALSE(fixture.link_data->coo_need_sync_to_device());

    fixture.bulk_data->modification_begin();
    ASSERT_TRUE(fixture.bulk_data->destroy_entity(deleted_linked_entity));
    fixture.bulk_data->modification_end();

    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));
    EXPECT_FALSE(fixture.link_data->coo_need_sync_to_device());

    NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
    EXPECT_THROW(ngp_link_data.update_crs_from_coo(), std::logic_error);
    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));
  }
}

TEST(UnitTestLinkDataObserver, SelectorUpdateForcesUniversalRebuildAndClearsPending) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);

    NgpLinkData& baseline_ngp = get_updated_ngp_link_data(*fixture.link_data);
    const size_t num_partitions_before = baseline_ngp.crs_data().get_all_crs_partitions().extent(0);
    ASSERT_GT(num_partitions_before, 0u);

    fixture.bulk_data->modification_begin();
    stk::mesh::Entity new_link =
        fixture.declare_entity(fixture.link_meta_data->link_rank(), stk::mesh::PartVector{fixture.link_part_dim3});
    stk::mesh::Entity target_a = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    stk::mesh::Entity target_b = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    stk::mesh::Entity target_c = fixture.declare_entity(stk::topology::ELEM_RANK, {});
    fixture.bulk_data->modification_end();

    fixture.link_data->coo_data().declare_relation(new_link, target_a, 0u);
    fixture.link_data->coo_data().declare_relation(new_link, target_b, 1u);
    fixture.link_data->coo_data().declare_relation(new_link, target_c, 2u);
    fixture.link_data->coo_modify_on_host();

    EXPECT_TRUE(impl::get_crs_structure_dirty(*fixture.link_data));
    EXPECT_FALSE(baseline_ngp.is_crs_up_to_date(*fixture.link_part_dim2));

    EXPECT_NO_THROW(baseline_ngp.update_crs_from_coo(*fixture.link_part_dim2));
    EXPECT_FALSE(impl::get_crs_structure_dirty(*fixture.link_data));
    EXPECT_NO_THROW(baseline_ngp.check_crs_coo_consistency());
    EXPECT_TRUE(baseline_ngp.is_crs_up_to_date());

    const size_t num_partitions_after = baseline_ngp.crs_data().get_all_crs_partitions().extent(0);
    EXPECT_GT(num_partitions_after, num_partitions_before);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 0u), target_a);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 1u), target_b);
    EXPECT_EQ(fixture.link_data->coo_data().get_linked_entity(new_link, 2u), target_c);
  }
}

TEST(UnitTestLinkDataObserver, NoOpModificationCycleDoesNotRequestRebuild) {
  for (bool commit_mesh : {false, true}) {
    LinkDataObserverFixture fixture(default_bucket_capacity, commit_mesh);
    create_connected_dim2_link(fixture);

    fixture.bulk_data->modification_begin();
    fixture.bulk_data->modification_end();

    NgpLinkData& ngp_link_data = get_updated_ngp_link_data(*fixture.link_data);
    EXPECT_FALSE(impl::get_crs_structure_dirty(*fixture.link_data));
    EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());
  }
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
