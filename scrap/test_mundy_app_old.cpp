// C++ core libs
#include <gtest/gtest.h>  // for AssertHelper, EXPECT_EQ, etc

#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <mundy_utils/throw_assert.hpp>   // for MUNDY_THROW_ASSERT


void verify_global_entity_count(size_t expected_total_num_nodes, size_t expected_total_num_edges,
                                size_t expected_total_num_elements, const stk::mesh::BulkData &bulk_data) {
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(bulk_data, entity_counts);
  EXPECT_EQ(expected_total_num_nodes, entity_counts[stk::topology::NODE_RANK]);
  EXPECT_EQ(expected_total_num_edges, entity_counts[stk::topology::EDGE_RANK]);
  EXPECT_EQ(expected_total_num_elements, entity_counts[stk::topology::ELEMENT_RANK]);
}

int main(int argc, char **argv) {
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "MPI_Init failed." << std::endl;
    return -1;
  }

  /*
  The schematic for this test:
    PARTICLE1         PARTICLE2
    |        \       /        |
    |         SPRING1         |
    |        /       \        |
  NODE1   NODE2     NODE3   NODE4

  We divide this schematic over two processes as follows:
                      RANK0                   |                  RANK1
    PARTICLE1(LO)               PARTICLE2(G)  |   PARTICLE1(G)            PARTICLE2(LO)
    |            \             /              |               \          /            |
    |             SPRING1(LO,S)               |                SPRING1(S)             |
    |            /             \              |               /          \            |
  NODE1(LO)   NODE2(LO,S)     NODE3(LO,S)     |            NODE2(S)     NODE3(S)    NODE4(LO)

  LO: Locally owned
  G: Ghosted
  S: Shared
  */

  // Construct metaData and bulk data.
  const size_t spatial_dimension = 3;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"node", "edge", "face", "elem"});
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create();

  if (bulk_data_ptr->parallel_size() == 2) {
    // Setup the meta mesh.
    stk::mesh::MetaData &metaData = bulk_data_ptr->mesh_meta_data();
    metaData.use_simple_fields();

    stk::mesh::Part &particle_part = metaData.declare_part_with_topology("particle", stk::topology::PARTICLE);
    stk::mesh::Part &spring_part = metaData.declare_part_with_topology("spring", stk::topology::LINE_2);
    metaData.commit();

    bulk_data_ptr->modification_begin();
    const int my_rank = bulk_data_ptr->parallel_rank();
    if (my_rank == 0) {
      stk::mesh::Entity particle1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{&particle_part});
      stk::mesh::Entity spring1 = bulk_data_ptr->declare_edge(1, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity node1 = bulk_data_ptr->declare_node(1);
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      stk::mesh::Entity node3 = bulk_data_ptr->declare_node(3);
      bulk_data_ptr->declare_relation(particle1, node1, 0);
      bulk_data_ptr->declare_relation(spring1, node2, 0);
      bulk_data_ptr->declare_relation(spring1, node3, 1);
      // bulk_data_ptr->declare_relation(particle1, spring1, 0);

      bulk_data_ptr->add_node_sharing(node2, 1);
      bulk_data_ptr->add_node_sharing(node3, 1);
      bulk_data_ptr->modification_end();

      // size_t expected_total_num_nodes_prior_to_end = 6;
      // size_t expected_total_num_edges_prior_to_end = 2;
      // size_t expected_total_num_elements_prior_to_end = 2;
      // verify_global_entity_count(expected_total_num_nodes_prior_to_end, expected_total_num_edges_prior_to_end,
      //                            expected_total_num_elements_prior_to_end, *bulk_data_ptr);

      // size_t expected_total_num_nodes_after_end = 4;
      // size_t expected_total_num_edges_after_end = 1;
      // size_t expected_total_num_elements_after_end = 2;
      // verify_global_entity_count(expected_total_num_nodes_after_end, expected_total_num_edges_after_end,
      //                            expected_total_num_elements_after_end, *bulk_data_ptr);

      // EXPECT_EQ(bulk_data_ptr->num_nodes(particle1), 1u);
      // EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      // EXPECT_EQ(bulk_data_ptr->num_edges(particle1), 1u);
      // EXPECT_EQ(bulk_data_ptr->num_elements(spring1), 2u);
      // EXPECT_EQ(particle1, bulk_data_ptr->begin_elements(spring1)[0]);
      // EXPECT_EQ(spring1, bulk_data_ptr->begin_edges(particle1)[0]);
    } else {
      stk::mesh::Entity particle2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{&particle_part});
      stk::mesh::Entity spring1 = bulk_data_ptr->declare_edge(1, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      stk::mesh::Entity node3 = bulk_data_ptr->declare_node(3);
      stk::mesh::Entity node4 = bulk_data_ptr->declare_node(4);
      bulk_data_ptr->declare_relation(particle2, node4, 0);
      bulk_data_ptr->declare_relation(spring1, node2, 0);
      bulk_data_ptr->declare_relation(spring1, node3, 1);
      // bulk_data_ptr->declare_relation(particle2, spring1, 0);

      bulk_data_ptr->add_node_sharing(node2, 0);
      bulk_data_ptr->add_node_sharing(node3, 0);
      bulk_data_ptr->modification_end();

      // size_t expected_total_num_nodes_prior_to_end = 6;
      // size_t expected_total_num_edges_prior_to_end = 2;
      // size_t expected_total_num_elements_prior_to_end = 2;
      // verify_global_entity_count(expected_total_num_nodes_prior_to_end, expected_total_num_edges_prior_to_end,
      //                            expected_total_num_elements_prior_to_end, *bulk_data_ptr);

      // size_t expected_total_num_nodes_after_end = 4;
      // size_t expected_total_num_edges_after_end = 1;
      // size_t expected_total_num_elements_after_end = 2;
      // verify_global_entity_count(expected_total_num_nodes_after_end, expected_total_num_edges_after_end,
      //                            expected_total_num_elements_after_end, *bulk_data_ptr);

      // EXPECT_EQ(bulk_data_ptr->num_nodes(particle2), 1u);
      // EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      // EXPECT_EQ(bulk_data_ptr->num_edges(particle2), 1u);
      // EXPECT_EQ(bulk_data_ptr->num_elements(spring1), 2u);
      // EXPECT_EQ(particle2, bulk_data_ptr->begin_elements(spring1)[1]);
      // EXPECT_EQ(spring1, bulk_data_ptr->begin_edges(particle2)[0]);
    }
  }

  MPI_Finalize();
  return 0;
}