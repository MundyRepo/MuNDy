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
#include <mundy_mesh/NewNgpLinkData.hpp>  // for mundy::mesh::NewNgpLinkData
#include <mundy_mesh/NewNgpCRSPartition.hpp>  // for mundy::mesh::NewNgpCRSPartition
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpForEachLink.hpp> 

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
  NewLinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", linker_rank);
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
  NewNgpLinkData link_data = declare_ngp_link_data(bulk_data, link_meta_data);
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
    link_data.declare_relation_host(link, linked_entity_a, 0);
    link_data.declare_relation_host(link, linked_entity_b, 1);
  }

  for (unsigned i = 0; i < link_and_3_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b, linked_entity_c] = link_and_3_linked_entities[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) &&
                bulk_data.is_valid(linked_entity_b) && bulk_data.is_valid(linked_entity_c));
    link_data.declare_relation_host(link, linked_entity_a, 0);
    link_data.declare_relation_host(link, linked_entity_b, 1);
    link_data.declare_relation_host(link, linked_entity_c, 2);
  }

  // Get the links
  for (unsigned i = 0; i < link_and_2_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b] = link_and_2_linked_entities[i];
    const auto& [entity_a_rank, entity_b_rank] = linked_entity_ranks_a[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) && bulk_data.is_valid(linked_entity_b));
    EXPECT_EQ(link_data.get_linked_entity_host(link, 0), linked_entity_a);
    EXPECT_EQ(link_data.get_linked_entity_host(link, 1), linked_entity_b);

    EXPECT_EQ(link_data.get_linked_entity_rank_host(link, 0), entity_a_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank_host(link, 1), entity_b_rank);

    EXPECT_EQ(link_data.get_linked_entity_id_host(link, 0), bulk_data.entity_key(linked_entity_a).id());
    EXPECT_EQ(link_data.get_linked_entity_id_host(link, 1), bulk_data.entity_key(linked_entity_b).id());
  }

  for (unsigned i = 0; i < link_and_3_linked_entities.size(); ++i) {
    const auto& [link, linked_entity_a, linked_entity_b, linked_entity_c] = link_and_3_linked_entities[i];
    const auto& [entity_a_rank, entity_b_rank, entity_c_rank] = linked_entity_ranks_b[i];
    ASSERT_TRUE(bulk_data.is_valid(link) && bulk_data.is_valid(linked_entity_a) &&
                bulk_data.is_valid(linked_entity_b) && bulk_data.is_valid(linked_entity_c));
    EXPECT_EQ(link_data.get_linked_entity_host(link, 0), linked_entity_a);
    EXPECT_EQ(link_data.get_linked_entity_host(link, 1), linked_entity_b);
    EXPECT_EQ(link_data.get_linked_entity_host(link, 2), linked_entity_c);

    EXPECT_EQ(link_data.get_linked_entity_rank_host(link, 0), entity_a_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank_host(link, 1), entity_b_rank);
    EXPECT_EQ(link_data.get_linked_entity_rank_host(link, 2), entity_c_rank);

    EXPECT_EQ(link_data.get_linked_entity_id_host(link, 0), bulk_data.entity_key(linked_entity_a).id());
    EXPECT_EQ(link_data.get_linked_entity_id_host(link, 1), bulk_data.entity_key(linked_entity_b).id());
    EXPECT_EQ(link_data.get_linked_entity_id_host(link, 2), bulk_data.entity_key(linked_entity_c).id());
  }

  // NGP stuff
  // The lambda allows us to scope what is or is not copied by the KOKKOS_LAMBDA since bulk_data cannot be copied
  link_data.modify_on_host();
  link_data.sync_to_device();
  auto run_ngp_test = [&link_part_a, &link_part_b, &link_part_c, &link_data]() {
    unsigned universal_link_ordinal =
        link_data.link_meta_data().universal_link_part().mesh_meta_data_ordinal();
    unsigned part_a_ordinal = link_part_a.mesh_meta_data_ordinal();
    unsigned part_b_ordinal = link_part_b.mesh_meta_data_ordinal();
    unsigned part_c_ordinal = link_part_c.mesh_meta_data_ordinal();

    for_each_link_run(
        link_data, link_part_b,
      KOKKOS_LAMBDA(const FastMeshIndex& linker_index) {
          // Check the link itself
          MUNDY_THROW_REQUIRE(link_data.ngp_mesh()
                          .get_bucket(link_data.link_rank(), linker_index.bucket_id)
                          .member(universal_link_ordinal), std::runtime_error, "Part membership error");
          MUNDY_THROW_REQUIRE(link_data.ngp_mesh()
                          .get_bucket(link_data.link_rank(), linker_index.bucket_id)
                          .member(part_b_ordinal), std::runtime_error, "Part membership error");
          MUNDY_THROW_REQUIRE(link_data.ngp_mesh()
                          .get_bucket(link_data.link_rank(), linker_index.bucket_id)
                          .member(part_c_ordinal), std::runtime_error, "Part membership error");
          MUNDY_THROW_REQUIRE(!link_data.ngp_mesh()
                           .get_bucket(link_data.link_rank(), linker_index.bucket_id)
                           .member(part_a_ordinal), std::runtime_error, "Part membership error");

          // Check that all downward linked entities are non-empty
          unsigned dimensionality_part_b = 3;
          for (unsigned d = 0; d < dimensionality_part_b; ++d) {
            stk::mesh::Entity linked_entity = link_data.get_linked_entity(linker_index, d);
            MUNDY_THROW_REQUIRE(linked_entity != stk::mesh::Entity(), std::runtime_error, "Fetching downward link failed.");
          }
        });

    // Not only can you fetch linked entities on the device, you can declare and delete relations in parallel and
    // without thread contention.
    for_each_link_run(
        link_data, link_part_b, KOKKOS_LAMBDA(const FastMeshIndex& linker_index) {
          // Get the linked entities and swap their order
          FastMeshIndex linked_entity_0 = link_data.get_linked_entity_index(linker_index, 0);
          FastMeshIndex linked_entity_1 = link_data.get_linked_entity_index(linker_index, 1);
          FastMeshIndex linked_entity_2 = link_data.get_linked_entity_index(linker_index, 2);

          EntityRank entity_0_rank = link_data.get_linked_entity_rank(linker_index, 0);
          EntityRank entity_1_rank = link_data.get_linked_entity_rank(linker_index, 1);
          EntityRank entity_2_rank = link_data.get_linked_entity_rank(linker_index, 2);

          link_data.delete_relation(linker_index, 0);
          link_data.delete_relation(linker_index, 1);
          link_data.delete_relation(linker_index, 2);

          link_data.declare_relation(linker_index, entity_2_rank, linked_entity_2, 0);
          link_data.declare_relation(linker_index, entity_1_rank, linked_entity_1, 1);
          link_data.declare_relation(linker_index, entity_0_rank, linked_entity_0, 2);
        });

    link_data.modify_on_device();
    link_data.sync_to_host();
  };
  run_ngp_test();

  // Check the CRS connectivity
  EXPECT_FALSE(link_data.is_crs_connectivity_up_to_date());
  link_data.update_crs_connectivity();
  EXPECT_TRUE(link_data.is_crs_connectivity_up_to_date());
  auto& crs_partition_view = link_data.get_all_crs_partitions();
}

TEST(UnitTestNgpLinkData, BasicUsage) {
  basic_usage_test();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
