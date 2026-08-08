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
#include <mpi.h>

// C++ core libs
#include <algorithm>    // for std::max
#include <filesystem>   // for std::filesystem::path
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <sstream>      // for std::ostringstream
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_io/FillMesh.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/FillMesh.hpp>        // for mundy::mesh::fill_mesh, fill_mesh_with_fields
#include <mundy_mesh/GetNgpLinkData.hpp>  // for mundy::mesh::get_updated_ngp_link_data
#include <mundy_mesh/LinkData.hpp>        // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpForEachLink.hpp>
#include <mundy_mesh/NgpLinkData.hpp>  // for mundy::mesh::NgpLinkData

namespace mundy {

namespace mesh {

namespace {

TEST(UnitTestNgpLinkData, LinkMetaDataUsesUniversalClassHierarchy) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data = builder.create_meta_data();
  meta_data->use_simple_fields();

  const stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;
  LinkMetaData& link_meta_data = declare_link_meta_data(*meta_data, "CLASS_LINKS", link_rank);
  Class& universal_link_class = link_meta_data.universal_link_class();

  std::ostringstream rank_name;
  rank_name << link_rank;
  EXPECT_EQ(universal_link_class.name(), std::string("MUNDY_UNIVERSAL_CLASS_LINKS_") + rank_name.str());
  EXPECT_EQ(universal_link_class.primary_entity_rank(), link_rank);
  EXPECT_EQ(universal_link_class.data_part().primary_entity_rank(), link_rank);

  Class& link_class = link_meta_data.declare_link_class("LINK_CLASS_0", 2);
  stk::mesh::Part& link_part = link_meta_data.declare_link_part("LINK_PART_0", 3);
  stk::mesh::Part& link_assembly_part = link_meta_data.declare_link_assembly_part("LINK_ASSEMBLY_0");

  EXPECT_TRUE(universal_link_class.contains(link_class));
  EXPECT_TRUE(universal_link_class.assembly_part().contains(link_class.assembly_part()));
  EXPECT_TRUE(universal_link_class.data_part().contains(link_class.data_part()));

  EXPECT_TRUE(universal_link_class.assembly_part().contains(link_part));
  EXPECT_TRUE(universal_link_class.data_part().contains(link_part));

  EXPECT_TRUE(universal_link_class.assembly_part().contains(link_assembly_part));
  EXPECT_FALSE(universal_link_class.data_part().contains(link_assembly_part));

  const auto& link_dirty_field = impl::get_link_crs_needs_updated_field(link_meta_data);
  const auto& linked_entity_ids_field = impl::get_linked_entity_ids_field(link_meta_data);

  EXPECT_TRUE(link_dirty_field.defined_on(universal_link_class.data_part()));
  EXPECT_FALSE(linked_entity_ids_field.defined_on(universal_link_class.data_part()));

  EXPECT_TRUE(linked_entity_ids_field.defined_on(link_class.data_part()));
  EXPECT_TRUE(linked_entity_ids_field.defined_on(link_part));
  EXPECT_FALSE(linked_entity_ids_field.defined_on(link_assembly_part));
}

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

LinkMetaData declare_and_validate_link_metadata(TestContext& context, const std::string& name) {
  LinkMetaData& link_meta_data = declare_link_meta_data(*context.meta_data, name, context.link_rank);
  EXPECT_EQ(link_meta_data.link_rank(), context.link_rank);
  EXPECT_TRUE(link_meta_data.name() == name);
  EXPECT_EQ(link_meta_data.universal_link_class().primary_entity_rank(), context.link_rank);
  return link_meta_data;
}

void setup_parts_and_links(TestContext& context, LinkMetaData& link_meta_data) {
  context.link_part_a = &context.meta_data->declare_part("LINK_PART_A", link_meta_data.link_rank());
  link_meta_data.add_link_support_to_part(*context.link_part_a, 2);

  context.link_part_b = &link_meta_data.declare_link_part("LINK_PART_B", 3);

  context.link_part_c = &link_meta_data.declare_link_assembly_part("LINK_PART_C");
  context.meta_data->declare_part_subset(*context.link_part_c, *context.link_part_a);
  context.meta_data->declare_part_subset(*context.link_part_c, *context.link_part_b);

  context.meta_data->commit();
}

// Struct to organize link initialization data
template <unsigned Dimensionality>
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
template <unsigned Dimensionality>
void initialize_links(TestContext& context, LinkInitializationData<Dimensionality>& link_init_data) {
  stk::mesh::PartVector link_part_vector{link_init_data.link_part};
  stk::mesh::PartVector empty_part_vector;
  for (const auto& ranks : link_init_data.linked_entity_ranks) {
    std::array<stk::mesh::Entity, Dimensionality + 1> entities;
    entities[0] = context.bulk_data->declare_entity(context.link_rank, ++context.num_linked_entities, link_part_vector);
    unsigned num_ranks = static_cast<unsigned>(ranks.size());
    for (unsigned i = 0; i < num_ranks; ++i) {
      entities[i + 1] =
          context.bulk_data->declare_entity(ranks[i], ++context.entity_counts[ranks[i]], empty_part_vector);
    }
    link_init_data.link_and_linked_entities.push_back(entities);
  }
}

template <unsigned Dimensionality>
void declare_and_validate_relations(const TestContext& context,
                                    const LinkInitializationData<Dimensionality>& link_init_data, LinkData& link_data) {
  size_t num_links_this_part = static_cast<size_t>(link_init_data.link_and_linked_entities.size());
  for (size_t i = 0; i < num_links_this_part; ++i) {
    const auto& entities = link_init_data.link_and_linked_entities[i];
    const auto& entity_ranks = link_init_data.linked_entity_ranks[i];

    // Assert validity of all entities
    for (size_t j = 0; j < entities.size(); ++j) {
      ASSERT_TRUE(context.bulk_data->is_valid(entities[j]));
    }

    // Declare relations
    for (unsigned j = 0; j < Dimensionality; ++j) {
      link_data.coo_data().declare_relation(entities[0], entities[j + static_cast<unsigned>(1)], j);

      // Validate linked entity, rank, and ID
      EXPECT_EQ(link_data.coo_data().get_linked_entity(entities[0], j), entities[j + 1]);
      EXPECT_EQ(link_data.coo_data().get_linked_entity_rank(entities[0], j), entity_ranks[j]);
      EXPECT_EQ(link_data.coo_data().get_linked_entity_id(entities[0], j),
                context.bulk_data->entity_key(entities[j + 1]).id());
    }
  }

  link_data.coo_modify_on_host();

  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(link_data);
  EXPECT_FALSE(ngp_link_data.is_crs_up_to_date())
      << "The modification should have dirtied the CSR connectivity and we should detect that, despite the host being "
         "modified and not the device.";
}

void validate_ngp_link_data(const TestContext& context, LinkData& link_data) {
  const stk::mesh::Part& universal_link_part = link_data.link_meta_data().universal_link_class();
  stk::mesh::Part& link_part_a = *context.link_part_a;
  stk::mesh::Part& link_part_b = *context.link_part_b;
  stk::mesh::Part& link_part_c = *context.link_part_c;
  unsigned universal_link_ordinal = universal_link_part.mesh_meta_data_ordinal();
  unsigned part_a_ordinal = link_part_a.mesh_meta_data_ordinal();
  unsigned part_b_ordinal = link_part_b.mesh_meta_data_ordinal();
  unsigned part_c_ordinal = link_part_c.mesh_meta_data_ordinal();

  // First check on the host
  for_each_link_run(
      link_data, *context.link_part_b, [&](const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& linker) {
        // Check the link itself
        const stk::mesh::Bucket& linker_bucket = bulk_data.bucket(linker);
        MUNDY_THROW_REQUIRE(bulk_data.is_valid(linker), std::runtime_error, "Linker is not valid.");
        MUNDY_THROW_REQUIRE(linker_bucket.member(universal_link_ordinal), std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(linker_bucket.member(part_b_ordinal), std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(linker_bucket.member(part_c_ordinal), std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(!linker_bucket.member(part_a_ordinal), std::runtime_error, "Part membership error");

        // Check that all downward linked entities are non-empty
        unsigned dimensionality_part_b = 3;
        for (unsigned d = 0; d < dimensionality_part_b; ++d) {
          stk::mesh::Entity linked_entity = link_data.coo_data().get_linked_entity(linker, d);
          MUNDY_THROW_REQUIRE(bulk_data.is_valid(linked_entity), std::runtime_error, "Linked entity is not valid.");
          MUNDY_THROW_REQUIRE(linked_entity != stk::mesh::Entity(), std::runtime_error,
                              "Fetching downward link failed.");
        }
      });

  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.coo_sync_to_device();

  stk::mesh::Entity::entity_value_type linked_entity_field_sum_host =
      ::mundy::mesh::field_sum<stk::mesh::Entity::entity_value_type>(
          impl::get_linked_entities_field(link_data.link_meta_data()), *context.link_part_b, stk::ngp::HostExecSpace());

  stk::mesh::Entity::entity_value_type linked_entity_field_sum_device =
      ::mundy::mesh::field_sum<stk::mesh::Entity::entity_value_type>(
          impl::get_linked_entities_field(link_data.link_meta_data()), *context.link_part_b, stk::ngp::ExecSpace());

  MUNDY_THROW_REQUIRE(linked_entity_field_sum_host != 0, std::runtime_error,
                      "host data was likely not properly initialized.");
  MUNDY_THROW_REQUIRE(linked_entity_field_sum_device != 0, std::runtime_error, "device data wasn't properly synced.");
  MUNDY_THROW_REQUIRE(linked_entity_field_sum_host == linked_entity_field_sum_device, std::runtime_error,
                      "device data wasn't properly synced.");

  for_each_link_run(
      ngp_link_data, *context.link_part_b, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker_index) {
        // Check the link itself
        MUNDY_THROW_REQUIRE(ngp_link_data.ngp_mesh()
                                .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                                .member(universal_link_ordinal),
                            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(ngp_link_data.ngp_mesh()
                                .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                                .member(part_b_ordinal),
                            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(ngp_link_data.ngp_mesh()
                                .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                                .member(part_c_ordinal),
                            std::runtime_error, "Part membership error");
        MUNDY_THROW_REQUIRE(!ngp_link_data.ngp_mesh()
                                 .get_bucket(ngp_link_data.link_rank(), linker_index.bucket_id)
                                 .member(part_a_ordinal),
                            std::runtime_error, "Part membership error");

        // Check that all downward linked entities are non-empty
        unsigned dimensionality_part_b = 3;
        for (unsigned d = 0; d < dimensionality_part_b; ++d) {
          stk::mesh::Entity linked_entity = ngp_link_data.coo_data().get_linked_entity(linker_index, d);
          MUNDY_THROW_REQUIRE(linked_entity != stk::mesh::Entity(), std::runtime_error,
                              "Fetching downward link failed.");
        }
      });
}

void modify_ngp_link_data(const TestContext& context, LinkData& link_data) {
  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.coo_sync_to_device();

  // Not only can you fetch linked entities on the device, you can declare and delete relations in parallel and
  // without thread contention.
  for_each_link_run(
      ngp_link_data, *context.link_part_b, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker_index) {
        const NgpLinkCOOData& ngp_coo_data = ngp_link_data.coo_data();

        // Get the linked entities and swap their order
        stk::mesh::FastMeshIndex linked_entity_0 = ngp_coo_data.get_linked_entity_index(linker_index, 0);
        stk::mesh::FastMeshIndex linked_entity_1 = ngp_coo_data.get_linked_entity_index(linker_index, 1);
        stk::mesh::FastMeshIndex linked_entity_2 = ngp_coo_data.get_linked_entity_index(linker_index, 2);

        stk::mesh::EntityRank entity_0_rank = ngp_coo_data.get_linked_entity_rank(linker_index, 0);
        stk::mesh::EntityRank entity_1_rank = ngp_coo_data.get_linked_entity_rank(linker_index, 1);
        stk::mesh::EntityRank entity_2_rank = ngp_coo_data.get_linked_entity_rank(linker_index, 2);

        ngp_coo_data.destroy_relation(linker_index, 0);
        ngp_coo_data.destroy_relation(linker_index, 1);
        ngp_coo_data.destroy_relation(linker_index, 2);

        ngp_coo_data.declare_relation(linker_index, entity_2_rank, linked_entity_2, 0);
        ngp_coo_data.declare_relation(linker_index, entity_1_rank, linked_entity_1, 1);
        ngp_coo_data.declare_relation(linker_index, entity_0_rank, linked_entity_0, 2);
      });

  ngp_link_data.coo_modify_on_device();

  EXPECT_FALSE(ngp_link_data.is_crs_up_to_date())
      << "The modification should have dirtied the CSR connectivity on the device.";
}

template <unsigned Dimensionality>
void validate_crs_connectivity(const TestContext& /*context*/, LinkInitializationData<Dimensionality>& link_init_data,
                               LinkData& link_data) {
  NgpLinkData& ngp_link_data = get_updated_ngp_link_data(link_data);
  ngp_link_data.update_crs_from_coo();
  EXPECT_TRUE(ngp_link_data.is_crs_up_to_date());

  // Invert the LinkInitializationData struct to store expected CSR connectivity
  std::map<stk::mesh::Entity, std::vector<stk::mesh::Entity>> expected_crs_conn;  // Entity to links map
  for (const auto& entities : link_init_data.link_and_linked_entities) {
    stk::mesh::Entity link = entities[0];
    for (size_t i = 1; i < entities.size(); ++i) {
      expected_crs_conn[entities[i]].push_back(link);
    }
  }

  // Perform test on host
  link_data.crs_sync_to_host();
  const auto& crs_partition_view = link_data.crs_data().get_all_crs_partitions();

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    ::mundy::mesh::for_each_entity_run(
        link_data.bulk_data(), rank,
        [&expected_crs_conn, &crs_partition_view, rank](const stk::mesh::BulkData& bulk_data,
                                                        const stk::mesh::Entity& entity) {
          auto it = expected_crs_conn.find(entity);
          if (it != expected_crs_conn.end()) {
            // Convert expected links to a set for comparison
            const std::vector<stk::mesh::Entity>& expected_links = it->second;
            std::set<stk::mesh::Entity> expected_link_set(expected_links.begin(), expected_links.end());

            // Fetch the actual connected links from the CSR connectivity
            std::set<stk::mesh::Entity> actual_link_set;
            for (unsigned partition_id = 0; partition_id < crs_partition_view.extent(0); ++partition_id) {
              const LinkCSRPartition& partition = crs_partition_view(partition_id);
              const stk::mesh::FastMeshIndex entity_index{bulk_data.bucket(entity).bucket_id(),
                                                          bulk_data.bucket_ordinal(entity)};

              auto connected_links = partition.get_connected_links(rank, entity_index);
              for (unsigned l = 0; l < connected_links.size(); ++l) {
                actual_link_set.insert(connected_links[l]);
              }
            }

            // Compare expected and actual sets
            // We are comparing to a set, so the fact that our modification on the device swapped the order of linked
            // entities won't affect the test.
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
  LinkMetaData link_meta_data = declare_and_validate_link_metadata(context, "ALL_LINKS");

  // Setup parts and links
  setup_parts_and_links(context, link_meta_data);

  // Declare and validate link data manager
  LinkData& link_data = declare_link_data(*context.bulk_data, link_meta_data);
  EXPECT_EQ(link_data.link_meta_data().link_rank(), link_meta_data.link_rank());

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

  // Check the CSR connectivity
  validate_crs_connectivity(context, link_init_data_a, link_data);
  validate_crs_connectivity(context, link_init_data_b, link_data);
}

TEST(UnitTestNgpLinkData, BasicUsage) {
  basic_usage_test();
}

TEST(UnitTestNgpLinkData, DeclareLinkClassSupportsNodeAndElementRankClasses) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data = builder.create_meta_data();
  meta_data->use_simple_fields();

  LinkMetaData& node_link_meta_data = declare_link_meta_data(*meta_data, "node_rank_links", stk::topology::NODE_RANK);
  LinkMetaData& elem_link_meta_data = declare_link_meta_data(*meta_data, "elem_rank_links", stk::topology::ELEM_RANK);

  Class& node_link_class = node_link_meta_data.declare_link_class("node_link_class", 2u);
  Class& elem_rank_only_link_class = elem_link_meta_data.declare_link_class("elem_link_class_without_topology", 3u);
  Class& elem_topological_link_class =
      elem_link_meta_data.declare_link_class("elem_link_class", stk::topology::PARTICLE, 3u);

  EXPECT_TRUE(node_link_meta_data.universal_link_class().contains(node_link_class));
  EXPECT_TRUE(elem_link_meta_data.universal_link_class().contains(elem_rank_only_link_class));
  EXPECT_TRUE(elem_link_meta_data.universal_link_class().contains(elem_topological_link_class));
  EXPECT_TRUE(node_link_meta_data.universal_link_class().has_io_support());
  EXPECT_FALSE(elem_link_meta_data.universal_link_class().has_io_support());
  EXPECT_FALSE(elem_rank_only_link_class.has_io_support());
  EXPECT_TRUE(elem_topological_link_class.has_io_support());
  EXPECT_EQ(node_link_class.primary_entity_rank(), stk::topology::NODE_RANK);
  EXPECT_EQ(elem_rank_only_link_class.primary_entity_rank(), stk::topology::ELEM_RANK);
  EXPECT_EQ(elem_topological_link_class.primary_entity_rank(), stk::topology::ELEM_RANK);

  const auto& node_link_ids = impl::get_linked_entity_ids_field(node_link_meta_data);
  const auto& node_link_ranks = impl::get_linked_entity_ranks_field(node_link_meta_data);
  const auto& elem_link_ids = impl::get_linked_entity_ids_field(elem_link_meta_data);
  const auto& elem_link_ranks = impl::get_linked_entity_ranks_field(elem_link_meta_data);

  EXPECT_TRUE(node_link_ids.defined_on(node_link_class.data_part()));
  EXPECT_TRUE(node_link_ids.defined_on(node_link_class.leaf_part()));
  EXPECT_TRUE(node_link_ranks.defined_on(node_link_class.data_part()));
  EXPECT_TRUE(node_link_ranks.defined_on(node_link_class.leaf_part()));

  EXPECT_TRUE(elem_link_ids.defined_on(elem_rank_only_link_class.data_part()));
  EXPECT_TRUE(elem_link_ids.defined_on(elem_rank_only_link_class.leaf_part()));
  EXPECT_TRUE(elem_link_ranks.defined_on(elem_rank_only_link_class.data_part()));
  EXPECT_TRUE(elem_link_ranks.defined_on(elem_rank_only_link_class.leaf_part()));

  EXPECT_TRUE(elem_link_ids.defined_on(elem_topological_link_class.data_part()));
  EXPECT_TRUE(elem_link_ids.defined_on(elem_topological_link_class.leaf_part()));
  EXPECT_TRUE(elem_link_ranks.defined_on(elem_topological_link_class.data_part()));
  EXPECT_TRUE(elem_link_ranks.defined_on(elem_topological_link_class.leaf_part()));
}

struct LinkRestartMeta {
  static constexpr unsigned link_dimensionality = 2u;

  std::shared_ptr<MetaData> meta_data;
  LinkMetaData* link_meta_data = nullptr;
  Class* link_class = nullptr;
  stk::mesh::Part* target_part = nullptr;
  stk::mesh::Field<double>* coords_field = nullptr;

  explicit LinkRestartMeta(MeshBuilder& mesh_builder, const bool commit_meta_data) {
    meta_data = mesh_builder.create_meta_data();
    meta_data->use_simple_fields();
    meta_data->set_coordinate_field_name("coords");

    coords_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
    stk::mesh::put_field_on_mesh(*coords_field, meta_data->universal_part(), 3, nullptr);

    link_meta_data = &declare_link_meta_data(*meta_data, "restart_links", stk::topology::ELEM_RANK);

    link_class = &link_meta_data->declare_link_class("link_class0", stk::topology::PARTICLE, link_dimensionality);

    target_part = &meta_data->declare_part_with_topology("target_part0", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(*target_part);

    if (commit_meta_data) {
      meta_data->commit();
    }
  }
};

struct LinkRestartIoContext {
  std::shared_ptr<BulkData> bulk_data;
  stk::io::StkMeshIoBroker io_broker;

  LinkRestartIoContext(MPI_Comm comm, MeshBuilder& mesh_builder, const std::shared_ptr<MetaData>& meta_data)
      : io_broker(comm) {
    bulk_data = mesh_builder.create_bulk_data(meta_data);
    io_broker.use_simple_fields();
    io_broker.set_bulk_data(*bulk_data);
  }
};

std::filesystem::path prepare_link_restart_output_dir(const std::string& directory_name) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::filesystem::path output_dir =
      std::filesystem::current_path() / ("mpi_size_" + std::to_string(size)) / directory_name;
  if (rank == 0) {
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return output_dir;
}

stk::mesh::Entity declare_particle(stk::mesh::BulkData& bulk_data, stk::mesh::Part& elem_part,
                                   stk::mesh::Field<double>& coords_field, const stk::mesh::EntityId elem_id,
                                   const stk::mesh::EntityId node_id, const double x_coord) {
  const stk::mesh::PartVector node_parts;
  const stk::mesh::PartVector elem_parts{&elem_part};

  const stk::mesh::Entity node = bulk_data.declare_node(node_id, node_parts);
  const stk::mesh::Entity elem = bulk_data.declare_element(elem_id, elem_parts);
  bulk_data.declare_relation(elem, node, 0u);

  double* coords = stk::mesh::field_data(coords_field, node);
  MUNDY_THROW_REQUIRE(coords != nullptr, std::runtime_error,
                      "Failed to fetch coordinates for restart-link test particle.");
  coords[0] = x_coord;
  coords[1] = 0.0;
  coords[2] = 0.0;

  return elem;
}

void expect_link_relation(const LinkData& link_data, const stk::mesh::Entity link, const unsigned ordinal,
                          const stk::mesh::Entity expected_linked_entity) {
  const stk::mesh::BulkData& bulk_data = link_data.bulk_data();
  ASSERT_TRUE(bulk_data.is_valid(link));
  ASSERT_TRUE(bulk_data.is_valid(expected_linked_entity));

  EXPECT_EQ(link_data.coo_data().get_linked_entity_id(link, ordinal), bulk_data.identifier(expected_linked_entity));
  EXPECT_EQ(link_data.coo_data().get_linked_entity_rank(link, ordinal), bulk_data.entity_rank(expected_linked_entity));
  EXPECT_EQ(link_data.coo_data().get_linked_entity(link, ordinal), expected_linked_entity);
}

struct NodeRankLinkClassRestartMeta {
  std::shared_ptr<MetaData> meta_data;
  LinkMetaData* link_meta_data = nullptr;
  Class* link_class = nullptr;
  stk::mesh::Part* target_part = nullptr;
  stk::mesh::Part* anchor_part = nullptr;
  stk::mesh::Field<double>* coords_field = nullptr;

  explicit NodeRankLinkClassRestartMeta(MeshBuilder& mesh_builder, const bool commit_meta_data) {
    meta_data = mesh_builder.create_meta_data();
    meta_data->use_simple_fields();
    meta_data->set_coordinate_field_name("coords");

    coords_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
    stk::mesh::put_field_on_mesh(*coords_field, meta_data->universal_part(), 3, nullptr);

    link_meta_data = &declare_link_meta_data(*meta_data, "node_rank_restart_links", stk::topology::NODE_RANK);
    link_class = &link_meta_data->declare_link_class("node_rank_restart_link_class", 2u);
    target_part = &meta_data->declare_part("node_rank_restart_target_part", stk::topology::NODE_RANK);
    anchor_part = &meta_data->declare_part_with_topology("node_rank_restart_anchor_part", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(*target_part);
    stk::io::put_io_part_attribute(*anchor_part);

    if (commit_meta_data) {
      meta_data->commit();
    }
  }
};

// Should the link data be declared before or after restart
enum class LinkDeclOrder { BeforeRestart, AfterRestart };

const char* link_decl_order_suffix(LinkDeclOrder order) {
  return order == LinkDeclOrder::BeforeRestart ? "BeforeRestart" : "AfterRestart";
}

// Perform the restart read and the declare_link_data call in the order under test, returning the (address-stable)
// LinkData. This relative ordering is the only step that differs between the two parameterizations; the rest of each
// round-trip is shared.
LinkData& read_restart_and_declare(LinkDeclOrder order, LinkRestartIoContext& reader,
                                   const std::filesystem::path& restart_file, LinkMetaData& reader_link_meta_data) {
  const auto read = [&] {
    fill_mesh_with_fields(restart_file.string(), reader.io_broker, *reader.bulk_data, stk::io::READ_RESTART);
  };
  if (order == LinkDeclOrder::BeforeRestart) {
    LinkData& link_data = declare_link_data(*reader.bulk_data, reader_link_meta_data);
    read();
    return link_data;
  }
  read();
  return declare_link_data(*reader.bulk_data, reader_link_meta_data);
}

class LinkRestartRoundTrip : public ::testing::TestWithParam<LinkDeclOrder> {};

TEST_P(LinkRestartRoundTrip, PreservesTopologicalLinkClassRelations) {
  const std::string order_suffix = link_decl_order_suffix(GetParam());
  const std::filesystem::path output_dir =
      prepare_link_restart_output_dir("unit_test_topological_link_class_restart_" + order_suffix);
  const std::filesystem::path restart_file = output_dir / "topological_link_class_restart.e-s.0";

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3);
  mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  std::shared_ptr<MetaData> writer_meta_data = mesh_builder.create_meta_data();
  writer_meta_data->use_simple_fields();
  writer_meta_data->set_coordinate_field_name("coords");

  stk::mesh::Field<double>& writer_coords = writer_meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
  stk::mesh::put_field_on_mesh(writer_coords, writer_meta_data->universal_part(), 3, nullptr);

  LinkMetaData& writer_link_meta_data =
      declare_link_meta_data(*writer_meta_data, "topological_restart_links", stk::topology::ELEM_RANK);
  Class& writer_link_class =
      writer_link_meta_data.declare_link_class("topological_restart_link_class", stk::topology::PARTICLE, 2u);
  stk::mesh::Part& writer_target_part =
      writer_meta_data->declare_part_with_topology("topological_restart_target_part", stk::topology::PARTICLE);
  stk::io::put_io_part_attribute(writer_target_part);

  writer_meta_data->commit();
  LinkRestartIoContext writer(MPI_COMM_WORLD, mesh_builder, writer_meta_data);
  LinkData& writer_link_data = declare_link_data(*writer.bulk_data, writer_link_meta_data);

  writer.bulk_data->modification_begin();
  const stk::mesh::Entity target0 =
      declare_particle(*writer.bulk_data, writer_target_part, writer_coords, 1u, 101u, 0.0);
  const stk::mesh::Entity target1 =
      declare_particle(*writer.bulk_data, writer_target_part, writer_coords, 2u, 102u, 1.0);
  const stk::mesh::Entity link =
      declare_particle(*writer.bulk_data, writer_link_class.leaf_part(), writer_coords, 100u, 200u, 0.5);
  writer.bulk_data->modification_end();

  writer_link_data.coo_data().declare_relation(link, target0, 0u);
  writer_link_data.coo_data().declare_relation(link, target1, 1u);
  writer_link_data.coo_modify_on_host();

  expect_link_relation(writer_link_data, link, 0u, target0);
  expect_link_relation(writer_link_data, link, 1u, target1);

  const size_t output_index = writer.io_broker.create_output_mesh(restart_file.string(), stk::io::WRITE_RESULTS);
  add_link_restart_fields(writer.io_broker, output_index, writer_link_meta_data);
  writer.io_broker.begin_output_step(output_index, 1.0);
  writer.io_broker.write_defined_output_fields(output_index);
  writer.io_broker.end_output_step(output_index);
  writer.io_broker.flush_output();
  writer.io_broker.close_output_mesh(output_index);

  MPI_Barrier(MPI_COMM_WORLD);

  std::shared_ptr<MetaData> reader_meta_data = mesh_builder.create_meta_data();
  reader_meta_data->use_simple_fields();
  reader_meta_data->set_coordinate_field_name("coords");

  stk::mesh::Field<double>& reader_coords = reader_meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
  stk::mesh::put_field_on_mesh(reader_coords, reader_meta_data->universal_part(), 3, nullptr);

  LinkMetaData& reader_link_meta_data =
      declare_link_meta_data(*reader_meta_data, "topological_restart_links", stk::topology::ELEM_RANK);
  Class& reader_link_class =
      reader_link_meta_data.declare_link_class("topological_restart_link_class", stk::topology::PARTICLE, 2u);
  stk::mesh::Part& reader_target_part =
      reader_meta_data->declare_part_with_topology("topological_restart_target_part", stk::topology::PARTICLE);
  stk::io::put_io_part_attribute(reader_target_part);
  (void)reader_target_part;

  ASSERT_FALSE(reader_meta_data->is_commit());
  LinkRestartIoContext reader(MPI_COMM_WORLD, mesh_builder, reader_meta_data);
  LinkData& reader_link_data = read_restart_and_declare(GetParam(), reader, restart_file, reader_link_meta_data);
  const stk::mesh::Entity reader_target0 = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 1u);
  const stk::mesh::Entity reader_target1 = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 2u);
  const stk::mesh::Entity reader_link = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 100u);

  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target0));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target1));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_link));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_link_class.leaf_part()));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_link_meta_data.universal_link_class()));

  expect_link_relation(reader_link_data, reader_link, 0u, reader_target0);
  expect_link_relation(reader_link_data, reader_link, 1u, reader_target1);

  NgpLinkData& reader_ngp_link_data = get_updated_ngp_link_data(reader_link_data);
  EXPECT_NO_THROW(reader_ngp_link_data.update_crs_from_coo());
  EXPECT_NO_THROW(reader_ngp_link_data.check_crs_coo_consistency());
  EXPECT_TRUE(reader_ngp_link_data.is_crs_up_to_date());
}

TEST_P(LinkRestartRoundTrip, PreservesNodeRankLinkClassRelations) {
  const std::string order_suffix = link_decl_order_suffix(GetParam());
  const std::filesystem::path output_dir =
      prepare_link_restart_output_dir("unit_test_node_rank_link_class_restart_" + order_suffix);
  const std::filesystem::path restart_file = output_dir / "node_rank_link_class_restart.e-s.0";

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3);
  mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  NodeRankLinkClassRestartMeta writer_meta(mesh_builder, true);
  LinkRestartIoContext writer(MPI_COMM_WORLD, mesh_builder, writer_meta.meta_data);
  LinkData& writer_link_data = declare_link_data(*writer.bulk_data, *writer_meta.link_meta_data);

  writer.bulk_data->modification_begin();
  const stk::mesh::PartVector target_parts{writer_meta.target_part};
  const stk::mesh::PartVector link_parts{&writer_meta.link_class->leaf_part()};

  const stk::mesh::Entity target0 = writer.bulk_data->declare_node(1u, target_parts);
  const stk::mesh::Entity target1 = writer.bulk_data->declare_node(2u, target_parts);
  const stk::mesh::Entity link = writer.bulk_data->declare_node(100u, link_parts);
  const stk::mesh::Entity anchor_elem =
      declare_particle(*writer.bulk_data, *writer_meta.anchor_part, *writer_meta.coords_field, 1000u, 1100u, 2.0);

  const auto& linked_entity_ids_field = impl::get_linked_entity_ids_field(*writer_meta.link_meta_data);
  EXPECT_TRUE(linked_entity_ids_field.defined_on(writer_meta.link_class->leaf_part()));
  EXPECT_TRUE(linked_entity_ids_field.defined_on(writer_meta.link_class->data_part()));

  double* target0_coords = stk::mesh::field_data(*writer_meta.coords_field, target0);
  double* target1_coords = stk::mesh::field_data(*writer_meta.coords_field, target1);
  double* link_coords = stk::mesh::field_data(*writer_meta.coords_field, link);
  ASSERT_NE(target0_coords, nullptr);
  ASSERT_NE(target1_coords, nullptr);
  ASSERT_NE(link_coords, nullptr);
  target0_coords[0] = 0.0;
  target0_coords[1] = 0.0;
  target0_coords[2] = 0.0;
  target1_coords[0] = 1.0;
  target1_coords[1] = 0.0;
  target1_coords[2] = 0.0;
  link_coords[0] = 0.5;
  link_coords[1] = 0.0;
  link_coords[2] = 0.0;
  ASSERT_TRUE(writer.bulk_data->is_valid(anchor_elem));
  writer.bulk_data->modification_end();

  writer_link_data.coo_data().declare_relation(link, target0, 0u);
  writer_link_data.coo_data().declare_relation(link, target1, 1u);
  writer_link_data.coo_modify_on_host();

  expect_link_relation(writer_link_data, link, 0u, target0);
  expect_link_relation(writer_link_data, link, 1u, target1);

  const size_t output_index = writer.io_broker.create_output_mesh(restart_file.string(), stk::io::WRITE_RESULTS);
  add_link_restart_fields(writer.io_broker, output_index, *writer_meta.link_meta_data);
  writer.io_broker.begin_output_step(output_index, 1.0);
  writer.io_broker.write_defined_output_fields(output_index);
  writer.io_broker.end_output_step(output_index);
  writer.io_broker.flush_output();
  writer.io_broker.close_output_mesh(output_index);

  MPI_Barrier(MPI_COMM_WORLD);

  NodeRankLinkClassRestartMeta reader_meta(mesh_builder, false);
  ASSERT_FALSE(reader_meta.meta_data->is_commit());
  LinkRestartIoContext reader(MPI_COMM_WORLD, mesh_builder, reader_meta.meta_data);
  LinkData& reader_link_data = read_restart_and_declare(GetParam(), reader, restart_file, *reader_meta.link_meta_data);
  const stk::mesh::Entity reader_target0 = reader.bulk_data->get_entity(stk::topology::NODE_RANK, 1u);
  const stk::mesh::Entity reader_target1 = reader.bulk_data->get_entity(stk::topology::NODE_RANK, 2u);
  const stk::mesh::Entity reader_link = reader.bulk_data->get_entity(stk::topology::NODE_RANK, 100u);
  const stk::mesh::Entity reader_anchor_elem = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 1000u);

  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target0));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target1));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_link));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_anchor_elem));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_meta.link_class->leaf_part()));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_meta.link_meta_data->universal_link_class()));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_anchor_elem).member(*reader_meta.anchor_part));

  expect_link_relation(reader_link_data, reader_link, 0u, reader_target0);
  expect_link_relation(reader_link_data, reader_link, 1u, reader_target1);

  NgpLinkData& reader_ngp_link_data = get_updated_ngp_link_data(reader_link_data);
  EXPECT_NO_THROW(reader_ngp_link_data.update_crs_from_coo());
  EXPECT_NO_THROW(reader_ngp_link_data.check_crs_coo_consistency());
  EXPECT_TRUE(reader_ngp_link_data.is_crs_up_to_date());
}

TEST_P(LinkRestartRoundTrip, PreservesLinkRelations) {
  const std::string order_suffix = link_decl_order_suffix(GetParam());
  const std::filesystem::path output_dir = prepare_link_restart_output_dir("unit_test_link_restart_" + order_suffix);
  const std::filesystem::path restart_file = output_dir / "link_restart.e-s.0";

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3);
  mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  LinkRestartMeta writer_meta(mesh_builder, true);
  LinkRestartIoContext writer(MPI_COMM_WORLD, mesh_builder, writer_meta.meta_data);
  LinkData& writer_link_data = declare_link_data(*writer.bulk_data, *writer_meta.link_meta_data);

  writer.bulk_data->modification_begin();
  const stk::mesh::Entity target0 =
      declare_particle(*writer.bulk_data, *writer_meta.target_part, *writer_meta.coords_field, 1u, 101u, 0.0);
  const stk::mesh::Entity target1 =
      declare_particle(*writer.bulk_data, *writer_meta.target_part, *writer_meta.coords_field, 2u, 102u, 1.0);
  const stk::mesh::Entity link = declare_particle(*writer.bulk_data, writer_meta.link_class->leaf_part(),
                                                  *writer_meta.coords_field, 100u, 200u, 0.5);
  writer.bulk_data->modification_end();

  writer_link_data.coo_data().declare_relation(link, target0, 0u);
  writer_link_data.coo_data().declare_relation(link, target1, 1u);
  writer_link_data.coo_modify_on_host();

  expect_link_relation(writer_link_data, link, 0u, target0);
  expect_link_relation(writer_link_data, link, 1u, target1);

  const size_t output_index = writer.io_broker.create_output_mesh(restart_file.string(), stk::io::WRITE_RESULTS);
  add_link_restart_fields(writer.io_broker, output_index, *writer_meta.link_meta_data);
  writer.io_broker.begin_output_step(output_index, 1.0);
  writer.io_broker.write_defined_output_fields(output_index);
  writer.io_broker.end_output_step(output_index);
  writer.io_broker.flush_output();
  writer.io_broker.close_output_mesh(output_index);

  MPI_Barrier(MPI_COMM_WORLD);

  LinkRestartMeta reader_meta(mesh_builder, false);
  ASSERT_FALSE(reader_meta.meta_data->is_commit());
  LinkRestartIoContext reader(MPI_COMM_WORLD, mesh_builder, reader_meta.meta_data);
  LinkData& reader_link_data = read_restart_and_declare(GetParam(), reader, restart_file, *reader_meta.link_meta_data);
  const stk::mesh::Entity reader_target0 = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 1u);
  const stk::mesh::Entity reader_target1 = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 2u);
  const stk::mesh::Entity reader_link = reader.bulk_data->get_entity(stk::topology::ELEM_RANK, 100u);

  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target0));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_target1));
  ASSERT_TRUE(reader.bulk_data->is_valid(reader_link));
  ASSERT_NE(reader_meta.link_class, nullptr);
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_meta.link_class->leaf_part()));
  EXPECT_TRUE(reader.bulk_data->bucket(reader_link).member(reader_meta.link_meta_data->universal_link_class()));

  expect_link_relation(reader_link_data, reader_link, 0u, reader_target0);
  expect_link_relation(reader_link_data, reader_link, 1u, reader_target1);

  NgpLinkData& reader_ngp_link_data = get_updated_ngp_link_data(reader_link_data);
  EXPECT_NO_THROW(reader_ngp_link_data.update_crs_from_coo());
  EXPECT_NO_THROW(reader_ngp_link_data.check_crs_coo_consistency());
  EXPECT_TRUE(reader_ngp_link_data.is_crs_up_to_date());
}

// Both declaration orders are exercised: a link data declared after the read reconciles on construction, and one
// declared before the read is reconciled by the link-aware fill_mesh_with_fields used in read_restart_and_declare.
INSTANTIATE_TEST_SUITE_P(DeclareOrder, LinkRestartRoundTrip,
                         ::testing::Values(LinkDeclOrder::BeforeRestart, LinkDeclOrder::AfterRestart),
                         [](const ::testing::TestParamInfo<LinkDeclOrder>& info) {
                           return std::string(link_decl_order_suffix(info.param));
                         });

// ---------------------------------------------------------------------------
// Focused invariant tests for LinkData
// ---------------------------------------------------------------------------

/// Minimal fixture for the focused LinkData API tests below.
struct LinkDataApiFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;
  LinkMetaData* link_meta{nullptr};
  static constexpr stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;

  LinkDataApiFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);
    link_meta = &declare_link_meta_data(*meta, "API_TEST_LINKS", link_rank);
    link_meta->declare_link_part("API_TEST_LINK_PART", 2u);
    meta->commit();
  }
};

// declare_link_data is idempotent: a second call with the same BulkData and
// LinkMetaData returns the same object, not a new one.
TEST(UnitTestNgpLinkData, DeclareLinkData_Idempotent) {
  LinkDataApiFixture f;
  LinkData& first  = declare_link_data(*f.bulk, *f.link_meta);
  LinkData& second = declare_link_data(*f.bulk, *f.link_meta);
  EXPECT_EQ(&first, &second);
}

// get_link_data returns nullptr when no declare_link_data call has been made yet.
TEST(UnitTestNgpLinkData, GetLinkData_ReturnsNullBeforeDeclaration) {
  LinkDataApiFixture f;
  EXPECT_EQ(get_link_data(*f.bulk, *f.link_meta), nullptr);
}

// The host-side CSR is a read-only snapshot of the device CSR; attempting to
// mark it modified always throws.
TEST(UnitTestNgpLinkData, CrsModifyOnHost_Throws) {
  LinkDataApiFixture f;
  LinkData& link_data = declare_link_data(*f.bulk, *f.link_meta);
  EXPECT_THROW(link_data.crs_modify_on_host(), std::exception);
}

// Host and device COO modification are mutually exclusive: marking the host as
// modified and then trying to mark the device as modified throws.
TEST(UnitTestNgpLinkData, CooModifyOnDevice_ThrowsWhenHostAlreadyModified) {
  LinkDataApiFixture f;
  LinkData& link_data = declare_link_data(*f.bulk, *f.link_meta);
  link_data.coo_modify_on_host();
  EXPECT_THROW(link_data.coo_modify_on_device(), std::exception);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
