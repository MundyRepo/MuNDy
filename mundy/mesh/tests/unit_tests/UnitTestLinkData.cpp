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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/LinkData.hpp>  // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>

namespace mundy {

namespace mesh {

namespace {

/// @brief Unit test basic usage of LinkData in Mundy.
///
/// This test covers the following:
/// - Setting up the mesh and metadata.
/// - Declaring link metadata and parts.
/// - Adding link support to parts.
/// - Declaring entities and links between them.
/// - Validating the links and their connected entities.
/// - Running parallel operations on links.
/// - Synchronizing link data between host and device.
void basic_usage_test() {
  using stk::mesh::Entity;
  using stk::mesh::EntityId;
  using stk::mesh::EntityRank;
  using stk::mesh::FastMeshIndex;
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::PartVector;
  using stk::topology::EDGE_RANK;
  using stk::topology::ELEM_RANK;
  using stk::topology::FACE_RANK;
  using stk::topology::NODE_RANK;

  // Setup
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = builder.create_meta_data();
  MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
  BulkData& bulk_data = *bulk_data_ptr;

  // Create the link meta data
  EntityRank linker_rank = NODE_RANK;
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", linker_rank);
  EXPECT_EQ(link_meta_data.link_rank(), linker_rank);
  EXPECT_TRUE(link_meta_data.name() == "ALL_LINKS");
  EXPECT_EQ(link_meta_data.universal_link_part().primary_entity_rank(), linker_rank);

  // Create a part and then add link support to it
  Part& link_part_a = meta_data.declare_part("LINK_PART_A", link_meta_data.link_rank());
  link_meta_data.add_link_support_to_part(link_part_a, 2 /*Link dimensionality within this part*/);

  // or declare a link part directly
  Part& link_part_b = link_meta_data.declare_link_part("LINK_PART_B", 3 /*Link dimensionality within this part*/);

  // Create a superset part and add the link parts to it
  Part& link_part_c = link_meta_data.declare_link_assembly_part("LINK_PART_C");
  meta_data.declare_part_subset(link_part_c, link_part_a);
  meta_data.declare_part_subset(link_part_c, link_part_b);
  meta_data.commit();

  // Create a link data manager (could be before or after commit. doesn't matter)
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);
  EXPECT_EQ(link_data.link_meta_data().link_rank(), linker_rank);

  // Declare some entities to connect and some links to place between them
  bulk_data.modification_begin();

  std::vector<unsigned> entity_counts(5, 0);
  std::vector<std::array<EntityRank, 2>> linked_entity_ranks_a = {{ELEM_RANK, ELEM_RANK}, {NODE_RANK, ELEM_RANK}};
  std::vector<std::array<EntityRank, 3>> linked_entity_ranks_b = {{ELEM_RANK, EDGE_RANK, NODE_RANK},
                                                                  {NODE_RANK, ELEM_RANK, EDGE_RANK}};

  std::vector<std::array<Entity, 3>> link_and_2_linked_entities;
  std::vector<std::array<Entity, 4>> link_and_3_linked_entities;
  PartVector empty_part_vector;
  for (const auto& [source_rank, target_rank] : linked_entity_ranks_a) {
    link_and_2_linked_entities.push_back(std::array<Entity, 3>{
        bulk_data.declare_entity(linker_rank, ++entity_counts[linker_rank], PartVector{&link_part_a}),
        bulk_data.declare_entity(source_rank, ++entity_counts[source_rank], empty_part_vector),
        bulk_data.declare_entity(target_rank, ++entity_counts[target_rank], empty_part_vector)});
  }
  for (const auto& [left_rank, middle_rank, right_rank] : linked_entity_ranks_b) {
    link_and_3_linked_entities.push_back(std::array<Entity, 4>{
        bulk_data.declare_entity(linker_rank, ++entity_counts[linker_rank], PartVector{&link_part_b}),
        bulk_data.declare_entity(left_rank, ++entity_counts[left_rank], empty_part_vector),
        bulk_data.declare_entity(middle_rank, ++entity_counts[middle_rank], empty_part_vector),
        bulk_data.declare_entity(right_rank, ++entity_counts[right_rank], empty_part_vector)});
  }
  bulk_data.modification_end();

  // Notice, we can declare link relations even outside of a modification block and between arbitrary ranks
  for (unsigned i = 0; i < link_and_2_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b] = link_and_2_linked_entities[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) && bulk_data.is_valid(linked_entity_b));
    link_data.declare_relation(link, linked_entity_a, 0);
    link_data.declare_relation(link, linked_entity_b, 1);
  }

  for (unsigned i = 0; i < link_and_3_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b, linked_entity_c] = link_and_3_linked_entities[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) &&
                bulk_data.is_valid(linked_entity_b) && bulk_data.is_valid(linked_entity_c));
    link_data.declare_relation(link, linked_entity_a, 0);
    link_data.declare_relation(link, linked_entity_b, 1);
    link_data.declare_relation(link, linked_entity_c, 2);
  }

  // Get the links
  for (unsigned i = 0; i < link_and_2_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b] = link_and_2_linked_entities[i];
    const auto& [entity_a_rank, entity_b_rank] = linked_entity_ranks_a[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) && bulk_data.is_valid(linked_entity_b));
    EXPECT_EQ(link_data.get_linked_entity(link, 0), linked_entity_a);
    EXPECT_EQ(link_data.get_linked_entity(link, 1), linked_entity_b);

    EXPECT_EQ(link_data.get_linked_entity_rank(link, 0), entity_a_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank(link, 1), entity_b_rank);

    EXPECT_EQ(link_data.get_linked_entity_id(link, 0), bulk_data.entity_key(linked_entity_a).id());
    EXPECT_EQ(link_data.get_linked_entity_id(link, 1), bulk_data.entity_key(linked_entity_b).id());
  }

  for (unsigned i = 0; i < link_and_3_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b, linked_entity_c] = link_and_3_linked_entities[i];
    const auto& [entity_a_rank, entity_b_rank, entity_c_rank] = linked_entity_ranks_b[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) &&
                bulk_data.is_valid(linked_entity_b) && bulk_data.is_valid(linked_entity_c));
    EXPECT_EQ(link_data.get_linked_entity(link, 0), linked_entity_a);
    EXPECT_EQ(link_data.get_linked_entity(link, 1), linked_entity_b);
    EXPECT_EQ(link_data.get_linked_entity(link, 2), linked_entity_c);

    EXPECT_EQ(link_data.get_linked_entity_rank(link, 0), entity_a_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank(link, 1), entity_b_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank(link, 2), entity_c_rank);

    EXPECT_EQ(link_data.get_linked_entity_id(link, 0), bulk_data.entity_key(linked_entity_a).id());
    EXPECT_EQ(link_data.get_linked_entity_id(link, 1), bulk_data.entity_key(linked_entity_b).id());
    EXPECT_EQ(link_data.get_linked_entity_id(link, 2), bulk_data.entity_key(linked_entity_c).id());
  }

  // Loop over the links in parallel
  for_each_link_run(
      link_data, link_part_b,
      [&link_part_a, &link_part_b, &link_part_c, &link_data](const stk::mesh::BulkData& bulk_data, const Entity& link) {
        // Check the link itself
        EXPECT_TRUE(bulk_data.is_valid(link));
        EXPECT_TRUE(bulk_data.entity_rank(link) == link_data.link_meta_data().link_rank());
        EXPECT_TRUE(bulk_data.bucket(link).member(link_data.link_meta_data().universal_link_part()));
        EXPECT_TRUE(bulk_data.bucket(link).member(link_part_b));
        EXPECT_TRUE(bulk_data.bucket(link).member(link_part_c));
        EXPECT_FALSE(bulk_data.bucket(link).member(link_part_a));

        // Check the connected entities
        for (unsigned link_ordinal = 0; link_ordinal < 3; ++link_ordinal) {
          EXPECT_TRUE(bulk_data.is_valid(link_data.get_linked_entity(link, link_ordinal)));
        }
      });

  // Get NGP_compatable link data
  NgpLinkData ngp_link_data = get_updated_ngp_data(link_data);

  // Declare link relations on the device
  ngp_link_data.modify_on_host();
  ngp_link_data.sync_to_device();
  auto run_ngp_test = [&link_part_a, &link_part_b, &link_part_c, &ngp_link_data]() {
    unsigned universal_link_ordinal =
        ngp_link_data.host_link_data().link_meta_data().universal_link_part().mesh_meta_data_ordinal();
    unsigned part_a_ordinal = link_part_a.mesh_meta_data_ordinal();
    unsigned part_b_ordinal = link_part_b.mesh_meta_data_ordinal();
    unsigned part_c_ordinal = link_part_c.mesh_meta_data_ordinal();

    // The lambda allows us to scope what is or is not copied by the KOKKOS_LAMBDA since bulk_data cannot be copied
    for_each_link_run(
        ngp_link_data, link_part_b, KOKKOS_LAMBDA(const FastMeshIndex& linker_index) {
          // Check the link itself
          EXPECT_TRUE(ngp_link_data.ngp_mesh()
                          .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                          .member(universal_link_ordinal));
          EXPECT_TRUE(ngp_link_data.ngp_mesh()
                          .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                          .member(part_b_ordinal));
          EXPECT_TRUE(ngp_link_data.ngp_mesh()
                          .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                          .member(part_c_ordinal));
          EXPECT_FALSE(ngp_link_data.ngp_mesh()
                           .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                           .member(part_a_ordinal));

          // Check the connected entities
          for (unsigned link_ordinal = 0; link_ordinal < 3; ++link_ordinal) {
            // TODO(palmerb4): Test that these are set via some device-compatable map.
            [[maybe_unused]] FastMeshIndex linked_entity = ngp_link_data.get_linked_entity_index(linker_index, link_ordinal);
            [[maybe_unused]] EntityId linked_entity_id = ngp_link_data.get_linked_entity_id(linker_index, link_ordinal);
            [[maybe_unused]] EntityRank linked_entity_rank =
                ngp_link_data.get_linked_entity_rank(linker_index, link_ordinal);
          }
        });

    // Not only can you fetch linked entities on the device, you can declare and delete relations in parallel and
    // without thread contention.
    for_each_link_run(
        ngp_link_data, link_part_b, KOKKOS_LAMBDA(const FastMeshIndex& linker_index) {
          // Get the linked entities and swap their order
          FastMeshIndex linked_entity_0 = ngp_link_data.get_linked_entity_index(linker_index, 0);
          FastMeshIndex linked_entity_1 = ngp_link_data.get_linked_entity_index(linker_index, 1);
          FastMeshIndex linked_entity_2 = ngp_link_data.get_linked_entity_index(linker_index, 2);

          EntityRank entity_0_rank = ngp_link_data.get_linked_entity_rank(linker_index, 0);
          EntityRank entity_1_rank = ngp_link_data.get_linked_entity_rank(linker_index, 1);
          EntityRank entity_2_rank = ngp_link_data.get_linked_entity_rank(linker_index, 2);

          ngp_link_data.delete_relation(linker_index, 0);
          ngp_link_data.delete_relation(linker_index, 1);
          ngp_link_data.delete_relation(linker_index, 2);

          ngp_link_data.declare_relation(linker_index, entity_2_rank, linked_entity_2, 0);
          ngp_link_data.declare_relation(linker_index, entity_1_rank, linked_entity_1, 1);
          ngp_link_data.declare_relation(linker_index, entity_0_rank, linked_entity_0, 2);
        });

    ngp_link_data.modify_on_device();
    ngp_link_data.sync_to_host();
  };
  run_ngp_test();

  // The host ordering should now be swapped
  for (unsigned i = 0; i < link_and_3_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b, linked_entity_c] = link_and_3_linked_entities[i];
    const auto& [entity_a_rank, entity_b_rank, entity_c_rank] = linked_entity_ranks_b[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) &&
                bulk_data.is_valid(linked_entity_b) && bulk_data.is_valid(linked_entity_c));
    EXPECT_EQ(bulk_data.entity_rank(link), linker_rank);
    EXPECT_EQ(link_data.get_linked_entity(link, 0), linked_entity_c);
    EXPECT_EQ(link_data.get_linked_entity(link, 1), linked_entity_b);
    EXPECT_EQ(link_data.get_linked_entity(link, 2), linked_entity_a);

    EXPECT_EQ(link_data.get_linked_entity_rank(link, 0), entity_c_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank(link, 1), entity_b_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank(link, 2), entity_a_rank);

    EXPECT_EQ(link_data.get_linked_entity_id(link, 0), bulk_data.entity_key(linked_entity_c).id());
    EXPECT_EQ(link_data.get_linked_entity_id(link, 1), bulk_data.entity_key(linked_entity_b).id());
    EXPECT_EQ(link_data.get_linked_entity_id(link, 2), bulk_data.entity_key(linked_entity_a).id());
  }

  // Ok, now, take these changes and propagate them to the crs connectivity
  link_data.propagate_updates();

  for_each_linked_entity_run(link_data,
                             [](const LinkData& link_data, const Entity& linked_entity, const Entity& linker) {
                               bool linked_entity_is_valid = link_data.bulk_data().is_valid(linked_entity);
                               bool linker_is_valid = link_data.bulk_data().is_valid(linker);
                               EXPECT_TRUE(linked_entity_is_valid);
                               EXPECT_TRUE(linker_is_valid);
                             });
}

TEST(UnitTestLinkData, BasicUsage) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  basic_usage_test();
}

void requests_test() {
  using stk::mesh::Entity;
  using stk::mesh::EntityId;
  using stk::mesh::EntityRank;
  using stk::mesh::FastMeshIndex;
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::PartVector;
  using stk::topology::EDGE_RANK;
  using stk::topology::ELEM_RANK;
  using stk::topology::FACE_RANK;
  using stk::topology::NODE_RANK;

  // Setup
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = builder.create_meta_data();
  MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link meta data and link data
  EntityRank linker_rank = NODE_RANK;
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", linker_rank);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);
  Part& link_part_a = link_meta_data.declare_link_part("LINK_PART_A", 2 /* our dimensionality */);
  Part& link_part_b = link_meta_data.declare_link_part("LINK_PART_B", 3 /* our dimensionality */);
  meta_data.commit();

  // Declare some pairs and triples of entities to connect
  std::vector<unsigned> entity_counts(5, 0);
  bulk_data.modification_begin();

  std::vector<std::array<Entity, 2>> two_linked_entities;
  for (unsigned i = 0; i < 6; ++i) {
    two_linked_entities.push_back(std::array<Entity, 2>{
        bulk_data.declare_entity(ELEM_RANK, ++entity_counts[ELEM_RANK], PartVector{}),
        bulk_data.declare_entity(NODE_RANK, ++entity_counts[NODE_RANK], PartVector{}),
    });
  }

  std::vector<std::array<Entity, 3>> three_linked_entities;
  for (unsigned i = 0; i < 6; ++i) {
    three_linked_entities.push_back(std::array<Entity, 3>{
        bulk_data.declare_entity(ELEM_RANK, ++entity_counts[ELEM_RANK], PartVector{}),
        bulk_data.declare_entity(EDGE_RANK, ++entity_counts[EDGE_RANK], PartVector{}),
        bulk_data.declare_entity(NODE_RANK, ++entity_counts[NODE_RANK], PartVector{}),
    });
  }
  bulk_data.modification_end();

  // Request links between the entities (potentially in parallel)
  //   Note, requesting a link must be done on a locally owned partition.
  auto& lo_link_partition_a =
      link_data.get_partition(stk::mesh::PartVector{&link_part_a, &meta_data.locally_owned_part()});
  auto& lo_link_partition_b =
      link_data.get_partition(stk::mesh::PartVector{&link_part_b, &meta_data.locally_owned_part()});

  ASSERT_EQ(lo_link_partition_a.link_dimensionality(), 2);
  ASSERT_EQ(lo_link_partition_b.link_dimensionality(), 3);

  lo_link_partition_a.increase_request_link_capacity(two_linked_entities.size());
  lo_link_partition_b.increase_request_link_capacity(three_linked_entities.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned i = 0; i < two_linked_entities.size(); ++i) {
    const auto& [entity_a, entity_b] = two_linked_entities[i];
    lo_link_partition_a.request_link(entity_a, entity_b);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned i = 0; i < three_linked_entities.size(); ++i) {
    const auto& [entity_a, entity_b, entity_c] = three_linked_entities[i];
    lo_link_partition_b.request_link(entity_a, entity_b, entity_c);
  }

  // Process the link requests
  link_data.process_requests();

  // Loop over each link in parallel and validate that it connects to one of the sets of requested entities
  stk::mesh::EntityVector a_links;
  stk::mesh::EntityVector b_links;
  stk::mesh::get_selected_entities(link_part_a, bulk_data.buckets(linker_rank), a_links);
  stk::mesh::get_selected_entities(link_part_b, bulk_data.buckets(linker_rank), b_links);
  ASSERT_EQ(a_links.size(), two_linked_entities.size());
  ASSERT_EQ(b_links.size(), three_linked_entities.size());

  for (const stk::mesh::Entity& a_link : a_links) {
    stk::mesh::Entity entity0 = link_data.get_linked_entity(a_link, 0);
    stk::mesh::Entity entity1 = link_data.get_linked_entity(a_link, 1);

    bool found = false;
    for (const auto& [entity_a, entity_b] : two_linked_entities) {
      if ((entity0 == entity_a && entity1 == entity_b) || (entity0 == entity_b && entity1 == entity_a)) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Link does not connect the correct entities.";
  }

  for (const stk::mesh::Entity& b_link : b_links) {
    stk::mesh::Entity entity0 = link_data.get_linked_entity(b_link, 0);
    stk::mesh::Entity entity1 = link_data.get_linked_entity(b_link, 1);
    stk::mesh::Entity entity2 = link_data.get_linked_entity(b_link, 2);

    bool found = false;
    for (const auto& [entity_a, entity_b, entity_c] : three_linked_entities) {
      if ((entity0 == entity_a && entity1 == entity_b && entity2 == entity_c) ||
          (entity0 == entity_a && entity1 == entity_c && entity2 == entity_b) ||
          (entity0 == entity_b && entity1 == entity_a && entity2 == entity_c) ||
          (entity0 == entity_b && entity1 == entity_c && entity2 == entity_a) ||
          (entity0 == entity_c && entity1 == entity_a && entity2 == entity_b) ||
          (entity0 == entity_c && entity1 == entity_b && entity2 == entity_a)) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Link does not connect the correct entities.";
  }

  // Now, let's destroy some links!
  for (unsigned i = 0; i < a_links.size(); ++i) {
    if (i >= a_links.size() / 2) {
      link_data.request_destruction(a_links[i]);
    }
  }
  link_data.process_requests();

  // Validate that the links were destroyed
  stk::mesh::EntityVector a_links_after_destruction;
  stk::mesh::get_selected_entities(link_part_a, bulk_data.buckets(linker_rank), a_links_after_destruction);
  ASSERT_EQ(a_links_after_destruction.size(), a_links.size() / 2);
  for (unsigned i = 0; i < a_links.size() / 2; ++i) {
    EXPECT_EQ(a_links[i], a_links_after_destruction[i]);
  }
}

TEST(UnitTestLinkData, Requests) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  requests_test();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
