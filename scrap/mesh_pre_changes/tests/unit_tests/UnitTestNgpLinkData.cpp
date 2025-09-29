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
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NewNgpLinkData.hpp>  // for mundy::mesh::NewNgpLinkData
#include <mundy_mesh/NgpForEachLink.hpp>
#include <mundy_mesh/GetNgpLinkData.hpp>  // for mundy::mesh::get_updated_ngp_link_data

namespace mundy {

namespace mesh {

namespace {

// Shared context for the test
struct TestContext {
  std::shared_ptr<MetaData> meta_data;
  std::shared_ptr<BulkData> bulk_data;
  stk::mesh::EntityRank link_rank;
  stk::mesh::Part* link_part_a = nullptr;
  stk::mesh::Part* link_part_b = nullptr;
  stk::mesh::Part* link_part_c = nullptr;
  size_t num_linked_entities = 0;                       ///< Number of linked entities created
  std::vector<size_t> entity_counts = {0, 0, 0, 0, 0};  ///< Counts of entities per rank
};

void setup_mesh_and_metadata(TestContext& context) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  context.meta_data = builder.create_meta_data();
  context.meta_data->use_simple_fields();

  context.bulk_data = builder.create_bulk_data(context.meta_data);
}

NewLinkMetaData declare_and_validate_link_metadata(TestContext& context, const std::string& name) {
  NewLinkMetaData link_meta_data = new_declare_link_meta_data(*context.meta_data, name, context.link_rank);
  EXPECT_EQ(link_meta_data.link_rank(), context.link_rank);
  EXPECT_TRUE(link_meta_data.name() == name);
  EXPECT_EQ(link_meta_data.universal_link_part().primary_entity_rank(), context.link_rank);
  return link_meta_data;
}

void setup_parts_and_links(TestContext& context, NewLinkMetaData& link_meta_data) {
  context.link_part_a = &context.meta_data->declare_part("LINK_PART_A", link_meta_data.link_rank());
  link_meta_data.add_link_support_to_part(*context.link_part_a, 2);

  context.link_part_b = &link_meta_data.declare_link_part("LINK_PART_B", 3);

  context.link_part_c = &link_meta_data.declare_link_assembly_part("LINK_PART_C");
  context.meta_data->declare_part_subset(*context.link_part_c, *context.link_part_a);
  context.meta_data->declare_part_subset(*context.link_part_c, *context.link_part_b);

  context.meta_data->commit();
}

NewLinkData declare_and_validate_link_data(TestContext& context, NewLinkMetaData& link_meta_data) {
  NewLinkData link_data = declare_link_data(*context.bulk_data, link_meta_data);
  EXPECT_EQ(link_data.link_meta_data().link_rank(), link_meta_data.link_rank());
  return link_data;
}

// Struct to organize link initialization data
template <size_t Dimensionality>
struct LinkInitializationData {
  using LinkAndLinkedEntitiesArray = std::array<stk::mesh::Entity, Dimensionality + 1>;
  using LinkedEntityRanksArray = std::array<stk::mesh::EntityRank, Dimensionality>;
  using LinkedEntityRanksVector = std::vector<LinkedEntityRanksArray>;
  using LinkAndLinkedEntitiesVector = std::vector<LinkAndLinkedEntitiesArray>;

  unsigned link_dimensionality = Dimensionality;         ///< Dimensionality of the link
  stk::mesh::Part* link_part;                            ///< Associated part
  LinkedEntityRanksVector linked_entity_ranks;           ///< Rank array
  LinkAndLinkedEntitiesVector link_and_linked_entities;  ///< Entities vector
};

// Function to initialize links using the struct
template <size_t Dimensionality>
void initialize_links(TestContext& context, LinkInitializationData<Dimensionality>& link_init_data) {
  stk::mesh::PartVector link_part_vector{link_init_data.link_part};
  stk::mesh::PartVector empty_part_vector;
  for (const auto& ranks : link_init_data.linked_entity_ranks) {
    std::array<stk::mesh::Entity, Dimensionality + 1> entities;
    entities[0] = context.bulk_data->declare_entity(context.link_rank, ++context.num_linked_entities, link_part_vector);
    for (size_t i = 0; i < ranks.size(); ++i) {
      entities[i + 1] =
          context.bulk_data->declare_entity(ranks[i], ++context.entity_counts[ranks[i]], empty_part_vector);
    }
    link_init_data.link_and_linked_entities.push_back(entities);
  }
}

template <size_t Dimensionality>
void declare_and_validate_relations(const TestContext& context,
                                    const LinkInitializationData<Dimensionality>& link_init_data,
                                    NewLinkData& link_data) {
  unsigned num_links_this_part = link_init_data.link_and_linked_entities.size();
  for (unsigned i = 0; i < num_links_this_part; ++i) {
    const auto& entities = link_init_data.link_and_linked_entities[i];
    const auto& entity_ranks = link_init_data.linked_entity_ranks[i];

    // Assert validity of all entities
    for (size_t j = 0; j < entities.size(); ++j) {
      ASSERT_TRUE(context.bulk_data->is_valid(entities[j]));
    }

    // Declare relations
    for (size_t j = 0; j < Dimensionality; ++j) {
      link_data.declare_relation(entities[0], entities[j + 1], j);

      // Validate linked entity, rank, and ID
      EXPECT_EQ(link_data.get_linked_entity(entities[0], j), entities[j + 1]);
      EXPECT_EQ(link_data.get_linked_entity_rank(entities[0], j), entity_ranks[j]);
      EXPECT_EQ(link_data.get_linked_entity_id(entities[0], j),
                context.bulk_data->entity_key(entities[j + 1]).id());
    }
  }

  link_data.modify_coo_on_host();

  EXPECT_FALSE(link_data.is_crs_up_to_date()) << "The modification should have dirtied the CRS connectivity.";
}

void validate_ngp_link_data(const TestContext& context, NewLinkData& link_data) {
  unsigned universal_link_ordinal = link_data.link_meta_data().universal_link_part().mesh_meta_data_ordinal();
  unsigned part_a_ordinal = context.link_part_a->mesh_meta_data_ordinal();
  unsigned part_b_ordinal = context.link_part_b->mesh_meta_data_ordinal();
  unsigned part_c_ordinal = context.link_part_c->mesh_meta_data_ordinal();

  auto ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.sync_coo_to_device();

  for_each_link_run(
      ngp_link_data, *context.link_part_b, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker_index) {
        // Check the link itself
        MUNDY_THROW_REQUIRE(ngp_link_data.ngp_mesh()
                                .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                                .member(universal_link_ordinal),
                            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(
            ngp_link_data.ngp_mesh().get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id).member(part_b_ordinal),
            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(
            ngp_link_data.ngp_mesh().get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id).member(part_c_ordinal),
            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(
            !ngp_link_data.ngp_mesh().get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id).member(part_a_ordinal),
            std::runtime_error, "Part membership error");

        // Check that all downward linked entities are non-empty
        unsigned dimensionality_part_b = 3;
        for (unsigned d = 0; d < dimensionality_part_b; ++d) {
          stk::mesh::Entity linked_entity = ngp_link_data.get_linked_entity(linker_index, d);
          MUNDY_THROW_REQUIRE(linked_entity != stk::mesh::Entity(), std::runtime_error,
                              "Fetching downward link failed.");
        }
      });
}

void modify_ngp_link_data(const TestContext& context, NewLinkData& link_data) {
  auto ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.sync_coo_to_device();

  // Not only can you fetch linked entities on the device, you can declare and delete relations in parallel and
  // without thread contention.
  for_each_link_run(
      ngp_link_data, *context.link_part_b, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker_index) {
        // Get the linked entities and swap their order
        stk::mesh::FastMeshIndex linked_entity_0 = ngp_link_data.get_linked_entity_index(linker_index, 0);
        stk::mesh::FastMeshIndex linked_entity_1 = ngp_link_data.get_linked_entity_index(linker_index, 1);
        stk::mesh::FastMeshIndex linked_entity_2 = ngp_link_data.get_linked_entity_index(linker_index, 2);

        stk::mesh::EntityRank entity_0_rank = ngp_link_data.get_linked_entity_rank(linker_index, 0);
        stk::mesh::EntityRank entity_1_rank = ngp_link_data.get_linked_entity_rank(linker_index, 1);
        stk::mesh::EntityRank entity_2_rank = ngp_link_data.get_linked_entity_rank(linker_index, 2);

        ngp_link_data.delete_relation(linker_index, 0);
        ngp_link_data.delete_relation(linker_index, 1);
        ngp_link_data.delete_relation(linker_index, 2);

        ngp_link_data.declare_relation(linker_index, entity_2_rank, linked_entity_2, 0);
        ngp_link_data.declare_relation(linker_index, entity_1_rank, linked_entity_1, 1);
        ngp_link_data.declare_relation(linker_index, entity_0_rank, linked_entity_0, 2);
      });

  ngp_link_data.modify_coo_on_device();
  
  EXPECT_FALSE(ngp_link_data.is_crs_up_to_date()) << "The modification should have dirtied the CRS connectivity.";
}

template <size_t Dimensionality>
void validate_crs_connectivity(const TestContext& context, LinkInitializationData<Dimensionality>& link_init_data,
                               NewLinkData& link_data) {
  auto ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.update_crs_from_coo();
  EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());

  // Invert the LinkInitializationData struct to store expected CRS connectivity
  std::map<stk::mesh::Entity, std::vector<stk::mesh::Entity>> expected_crs_conn;  // Entity to links map
  for (const auto& entities : link_init_data.link_and_linked_entities) {
    stk::mesh::Entity link = entities[0];
    for (size_t i = 1; i < entities.size(); ++i) {
      expected_crs_conn[entities[i]].push_back(link);
    }
  }

  // Perform test on host
  link_data.sync_crs_to_host();
  auto& crs_partition_view = ngp_link_data.get_or_create_crs_partitions(*link_init_data.link_part);
  EXPECT_EQ(crs_partition_view.extent(0), 1u) << "Expected one CRS partition for the part.";

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    stk::mesh::for_each_entity_run(
        ngp_link_data.bulk_data(), rank,
        [&expected_crs_conn, &crs_partition_view](const stk::mesh::BulkData& bulk_data,
                                                  const stk::mesh::Entity& entity) {
          auto it = expected_crs_conn.find(entity);
          if (it != expected_crs_conn.end()) {
            // Convert expected links to a set for comparison
            const std::vector<stk::mesh::Entity>& expected_links = it->second;
            std::set<stk::mesh::Entity> expected_link_set(expected_links.begin(), expected_links.end());

            // Fetch the actual connected links from the CRS connectivity
            std::set<stk::mesh::Entity> actual_link_set;
            for (unsigned partition_id = 0; partition_id < crs_partition_view.extent(0); ++partition_id) {
              const NewNgpCRSPartition& partition = crs_partition_view(partition_id);
              const stk::mesh::FastMeshIndex entity_index{bulk_data.bucket(entity).bucket_id(),
                                                          bulk_data.bucket_ordinal(entity)};

              auto connected_links = partition.get_connected_links(
                  bulk_data.entity_rank(entity),
                  entity_index);  // TODO(palmerb4): We currently only offer a device CRS connectivity interface. This
                                  // will fail for GPU builds.
              for (unsigned l = 0; l < connected_links.size(); ++l) {
                actual_link_set.insert(connected_links[l]);
              }
            }

            // Compare expected and actual sets
            EXPECT_EQ(expected_link_set, actual_link_set);
          }
        });
  }
}

/// @brief Unit test basic usage of LinkData in Mundy.
void basic_usage_test() {
  TestContext context;
  context.link_rank = stk::topology::NODE_RANK;

  // Setup mesh and metadata
  setup_mesh_and_metadata(context);

  // Declare and validate link metadata
  NewLinkMetaData link_meta_data = declare_and_validate_link_metadata(context, "ALL_LINKS");

  // Setup parts and links
  setup_parts_and_links(context, link_meta_data);

  // Declare and validate link data manager
  NewLinkData link_data = declare_and_validate_link_data(context, link_meta_data);

  // Declare some entities to connect and some links to place between them
  context.bulk_data->modification_begin();

  // Define link initialization data for 2-linked entities
  LinkInitializationData<2> link_init_data_a{
      .link_part = context.link_part_a,
      .linked_entity_ranks = {{stk::topology::ELEM_RANK, stk::topology::ELEM_RANK},
                              {stk::topology::NODE_RANK, stk::topology::ELEM_RANK}},
      .link_and_linked_entities = {}};

  // Define link initialization data for 3-linked entities
  LinkInitializationData<3> link_init_data_b{
      .link_part = context.link_part_b,
      .linked_entity_ranks = {{stk::topology::ELEM_RANK, stk::topology::EDGE_RANK, stk::topology::NODE_RANK},
                              {stk::topology::NODE_RANK, stk::topology::ELEM_RANK, stk::topology::EDGE_RANK}},
      .link_and_linked_entities = {}};

  // Initialize links using the helper function
  initialize_links(context, link_init_data_a);
  initialize_links(context, link_init_data_b);

  context.bulk_data->modification_end();

  // Declare and validate relations for 2-linked entities (works even though we are outside of a modification block)
  declare_and_validate_relations(context, link_init_data_a, link_data);

  // Declare and validate relations for 3-linked entities
  declare_and_validate_relations(context, link_init_data_b, link_data);

  // NGP stuff
  validate_ngp_link_data(context, link_data);
  modify_ngp_link_data(context, link_data);
  validate_ngp_link_data(context, link_data);

  // Check the CRS connectivity
  validate_crs_connectivity(context, link_init_data_a, link_data);
  validate_crs_connectivity(context, link_init_data_b, link_data);
}

TEST(UnitTestNgpLinkData, BasicUsage) {
  basic_usage_test();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
