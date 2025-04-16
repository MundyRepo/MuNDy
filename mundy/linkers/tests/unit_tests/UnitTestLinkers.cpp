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
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_io/FillMesh.hpp>
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>  // for stk::mesh::communicate_field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/MeshUtils.hpp>      // for stk::mesh::fixup_ghosted_to_shared_nodes
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <mundy_linkers/Linkers.hpp>   // for mundy::linkers::Linker and  mundy::linkers::declare_family_tree_relation
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace linkers {

namespace {

//! \name Linkers functionality unit tests
//@{

TEST(Linkers, LinkersInSTKWorkAsExpectedSymmetric) {
  /* This test checks the core functionality of STK that is necessary for MundyLinker to work properly.

  The schematic for this test:
                        CONSTRAINT1
                       /         |  \
       SPRING1        / SPHERE1  |   \       SPRING2
      /       \      /       |  /     \     /       \
  NODE1         NODE2       NODE3      NODE4         NODE5

  We divide this schematic over two processes as follows:
                            RANK0
                        CONSTRAINT1(A)
                       /           |  \
       SPRING1(LO)    / SPHERE1(G) |   \       SPRING2(G)
      /       \      /       |    /     \     /         \
  NODE1(LO)  NODE2(LO,S)    NODE3(A)    NODE4(A)         NODE5(G)


                            RANK1
                        CONSTRAINT1(LO)
                      /            |  \
       SPRING1(G)    / SPHERE1(LO) |   \        SPRING2(LO)
      /        \    /       |     /     \      /         \
  NODE1(G)    NODE2(S)     NODE3(LO)  NODE4(LO)       NODE5(LO)

  LO: Locally owned
  G: Ghosted
  A: In aura
  S: Shared
  */

  // Construct metaData and bulk data.
  const size_t spatial_dimension = 3;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create();

  if (bulk_data_ptr->parallel_size() == 2) {
    // Setup the meta mesh.
    stk::mesh::MetaData &metaData = bulk_data_ptr->mesh_meta_data();
    metaData.use_simple_fields();

    stk::mesh::Part &linker_part = metaData.declare_part("linker", stk::topology::CONSTRAINT_RANK);
    stk::mesh::Part &sphere_part = metaData.declare_part_with_topology("sphere", stk::topology::PARTICLE);
    stk::mesh::Part &spring_part = metaData.declare_part_with_topology("spring", stk::topology::BEAM_2);
    LinkedEntitiesFieldType &linked_entities_field =
        metaData.declare_field<LinkedEntitiesFieldType::value_type>(stk::topology::CONSTRAINT_RANK, "linked_entities");
    stk::mesh::Field<int> &linked_entity_owners_field =
        metaData.declare_field<int>(stk::topology::CONSTRAINT_RANK, "linked_entity_owners");
    stk::mesh::put_field_on_mesh(linked_entities_field, linker_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(linked_entity_owners_field, linker_part, 3, nullptr);

    metaData.commit();

    const int my_rank = bulk_data_ptr->parallel_rank();
    if (my_rank == 0) {
      bulk_data_ptr->modification_begin();
      stk::mesh::Entity spring1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity node1 = bulk_data_ptr->declare_node(1);
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      bulk_data_ptr->declare_relation(spring1, node1, 0);
      bulk_data_ptr->declare_relation(spring1, node2, 1);

      stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
      bulk_data_ptr->change_ghosting(custom_ghosting, {std::make_pair(spring1, 1), std::make_pair(node2, 1)});
      bulk_data_ptr->modification_end();

      // At this point, all we (rank 0) know about is our spring, its nodes, and the linker sent to us by rank 1.
      // In the following modification cycle, rank 1 declares the linker and connects it to our node2,
      // promoting it from ghosted to shared.
      stk::mesh::Entity linker1 = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 1);
      stk::mesh::Entity sphere1 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
      stk::mesh::Entity spring2 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
      stk::mesh::Entity node3 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 3);
      stk::mesh::Entity node4 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 4);
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker1));
      ASSERT_TRUE(bulk_data_ptr->is_valid(node3));
      ASSERT_TRUE(bulk_data_ptr->is_valid(node4));
      bulk_data_ptr->modification_begin();
      bulk_data_ptr->declare_relation(linker1, node2, 0);
      bulk_data_ptr->declare_relation(linker1, node3, 1);
      bulk_data_ptr->declare_relation(linker1, node4, 2);

      stk::mesh::field_data(linked_entities_field, linker1)[0] = bulk_data_ptr->entity_key(spring1);
      stk::mesh::field_data(linked_entities_field, linker1)[1] = bulk_data_ptr->entity_key(sphere1);
      stk::mesh::field_data(linked_entities_field, linker1)[2] = bulk_data_ptr->entity_key(spring2);

      stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
      mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr, linked_entities_field, linked_entity_owners_field,
                                                   linker_part);
      bulk_data_ptr->modification_end();

      // At this point, the mesh should be in a consistent state and we (rank 0) should know about
      // the linker and all of our new ghosts. As well, none of our entities should have become invalid.
      stk::mesh::Entity node5 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 5);
      EXPECT_TRUE(bulk_data_ptr->is_valid(linker1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(sphere1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node3));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node4));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node5));

      // Check buckets.
      // Note, custom ghosted objects are not in the aura.
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(linker1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(spring1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node2).owned());
      EXPECT_TRUE(bulk_data_ptr->bucket(node2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node2).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node3).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node3).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node4).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node4).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node5).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).in_aura());

      // Check the node connectivity.
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 3u);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node2, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node3, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node4, stk::topology::CONSTRAINT_RANK)[0]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(sphere1)[0]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(linker1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring2), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node3), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node4), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node5), 1);
    } else {
      bulk_data_ptr->modification_begin();
      stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{&linker_part});
      stk::mesh::Entity spring2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity sphere1 = bulk_data_ptr->declare_element(3, stk::mesh::ConstPartVector{&sphere_part});
      stk::mesh::Entity node3 = bulk_data_ptr->declare_node(3);
      stk::mesh::Entity node4 = bulk_data_ptr->declare_node(4);
      stk::mesh::Entity node5 = bulk_data_ptr->declare_node(5);
      bulk_data_ptr->declare_relation(spring2, node4, 0);
      bulk_data_ptr->declare_relation(spring2, node5, 1);
      bulk_data_ptr->declare_relation(sphere1, node3, 0);

      stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
      bulk_data_ptr->change_ghosting(
          custom_ghosting, {std::make_pair(linker1, 0), std::make_pair(spring2, 0), std::make_pair(sphere1, 0),
                            std::make_pair(node3, 0), std::make_pair(node4, 0)});
      bulk_data_ptr->modification_end();

      // At this point, all we (rank 1) know about is our spring and its nodes. As well as node2 sent to us by rank 0.
      stk::mesh::Entity spring1 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
      stk::mesh::Entity node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
      ASSERT_TRUE(bulk_data_ptr->is_valid(node2));

      // Declare the linker and connect it to the nodes. Use fixup_ghosted_to_shared_nodes to promote node2 from ghosted
      // to shared.
      bulk_data_ptr->modification_begin();
      bulk_data_ptr->declare_relation(linker1, node2, 0);
      bulk_data_ptr->declare_relation(linker1, node3, 1);
      bulk_data_ptr->declare_relation(linker1, node4, 2);

      stk::mesh::field_data(linked_entities_field, linker1)[0] = bulk_data_ptr->entity_key(spring1);
      stk::mesh::field_data(linked_entities_field, linker1)[1] = bulk_data_ptr->entity_key(sphere1);
      stk::mesh::field_data(linked_entities_field, linker1)[2] = bulk_data_ptr->entity_key(spring2);

      stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
      mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr, linked_entities_field, linked_entity_owners_field,
                                                   linker_part);
      bulk_data_ptr->modification_end();

      // At this point, the mesh should be in a consistent state and we (rank 1) should know about
      // the linker and all of our new ghosts. As well, none of our entities should have become invalid.
      stk::mesh::Entity node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
      EXPECT_TRUE(bulk_data_ptr->is_valid(linker1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(sphere1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node3));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node4));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node5));

      // Check buckets.
      EXPECT_TRUE(bulk_data_ptr->bucket(linker1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(spring1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(spring2).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(sphere1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node2).owned());
      EXPECT_TRUE(bulk_data_ptr->bucket(node2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node2).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node3).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node4).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node5).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).in_aura());

      // Check the connectivity.
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 3u);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node2, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node3, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node4, stk::topology::CONSTRAINT_RANK)[0]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(sphere1)[0]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(linker1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring2), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node3), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node4), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node5), 1);
    }
  }
}

TEST(Linkers, LinkersInSTKWorkAsExpectedOneSided) {
  /* This test checks the core functionality of STK that is necessary for MundyLinker to work properly.

  The schematic for this test:
                        CONSTRAINT1
                       /         |  \
       SPRING1        / SPHERE1  |   \       SPRING2
      /       \      /       |  /     \     /       \
  NODE1         NODE2       NODE3      NODE4         NODE5

  We divide this schematic over two processes as follows:
                            RANK0
                        CONSTRAINT1(A)
                       /           |  \
       SPRING1(LO)    / SPHERE1(G) |   \       SPRING2(G)
      /       \      /       |    /     \     /         \
  NODE1(LO)  NODE2(LO,S)    NODE3(A)    NODE4(A)         NODE5(G)


                            RANK1
                        CONSTRAINT1(LO)
                      /            |  \
       SPRING1(G)    / SPHERE1(LO) |   \        SPRING2(LO)
      /        \    /       |     /     \      /         \
  NODE1(G)    NODE2(S)     NODE3(LO)  NODE4(LO)       NODE5(LO)

  LO: Locally owned
  G: Ghosted
  A: In aura
  S: Shared
  */

  // Construct metaData and bulk data.
  const size_t spatial_dimension = 3;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create();

  if (bulk_data_ptr->parallel_size() == 2) {
    // Setup the meta mesh.
    stk::mesh::MetaData &metaData = bulk_data_ptr->mesh_meta_data();
    metaData.use_simple_fields();

    stk::mesh::Part &linker_part = metaData.declare_part("linker", stk::topology::CONSTRAINT_RANK);
    stk::mesh::Part &sphere_part = metaData.declare_part_with_topology("sphere", stk::topology::PARTICLE);
    stk::mesh::Part &spring_part = metaData.declare_part_with_topology("spring", stk::topology::BEAM_2);
    LinkedEntitiesFieldType &linked_entities_field =
        metaData.declare_field<LinkedEntitiesFieldType::value_type>(stk::topology::CONSTRAINT_RANK, "linked_entities");
    stk::mesh::Field<int> &linked_entity_owners_field =
        metaData.declare_field<int>(stk::topology::CONSTRAINT_RANK, "linked_entity_owners");
    stk::mesh::put_field_on_mesh(linked_entities_field, linker_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(linked_entity_owners_field, linker_part, 3, nullptr);

    metaData.commit();

    const int my_rank = bulk_data_ptr->parallel_rank();
    if (my_rank == 0) {
      bulk_data_ptr->modification_begin();
      stk::mesh::Entity spring1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity node1 = bulk_data_ptr->declare_node(1);
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      bulk_data_ptr->declare_relation(spring1, node1, 0);
      bulk_data_ptr->declare_relation(spring1, node2, 1);

      stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
      bulk_data_ptr->change_ghosting(custom_ghosting, {std::make_pair(spring1, 1), std::make_pair(node2, 1)});
      bulk_data_ptr->modification_end();

      // At this point, all we (rank 0) know about is our spring, its nodes, and the linker sent to us by rank 1.
      // In the following modification cycle, rank 1 declares the linker and connects it to our node2,
      // promoting it from ghosted to shared.
      bulk_data_ptr->modification_begin();
      stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
      bulk_data_ptr->modification_end();

      // At this point, the linker has been declared and node2 should be shared but the linked entities are not yet
      // ghosted
      bulk_data_ptr->modification_begin();
      mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr, linked_entities_field, linked_entity_owners_field,
                                                   linker_part);
      bulk_data_ptr->modification_end();

      // Now, everything should be up to date.
      stk::mesh::Entity linker1 = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 1);
      stk::mesh::Entity sphere1 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
      stk::mesh::Entity spring2 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
      stk::mesh::Entity node3 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 3);
      stk::mesh::Entity node4 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 4);
      stk::mesh::Entity node5 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 5);
      EXPECT_TRUE(bulk_data_ptr->is_valid(linker1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node3));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node4));
      EXPECT_TRUE(bulk_data_ptr->is_valid(linker1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(sphere1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node3));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node4));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node5));

      // We used expects above to get more information but throw here to prevent segfaults later on.
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker1) && bulk_data_ptr->is_valid(spring1) &&
                  bulk_data_ptr->is_valid(spring2) && bulk_data_ptr->is_valid(sphere1) &&
                  bulk_data_ptr->is_valid(node1) && bulk_data_ptr->is_valid(node2) && bulk_data_ptr->is_valid(node3) &&
                  bulk_data_ptr->is_valid(node4) && bulk_data_ptr->is_valid(node5));

      // Check buckets.
      // Note, custom ghosted objects are not in the aura.
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(linker1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(spring1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node2).owned());
      EXPECT_TRUE(bulk_data_ptr->bucket(node2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node2).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node3).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node3).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node4).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node4).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node5).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).in_aura());

      // Check the node connectivity.
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 3u);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node2, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node3, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node4, stk::topology::CONSTRAINT_RANK)[0]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(sphere1)[0]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(linker1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring2), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node3), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node4), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node5), 1);
    } else {
      bulk_data_ptr->modification_begin();
      stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{&linker_part});
      stk::mesh::Entity spring2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity sphere1 = bulk_data_ptr->declare_element(3, stk::mesh::ConstPartVector{&sphere_part});
      stk::mesh::Entity node3 = bulk_data_ptr->declare_node(3);
      stk::mesh::Entity node4 = bulk_data_ptr->declare_node(4);
      stk::mesh::Entity node5 = bulk_data_ptr->declare_node(5);
      bulk_data_ptr->declare_relation(spring2, node4, 0);
      bulk_data_ptr->declare_relation(spring2, node5, 1);
      bulk_data_ptr->declare_relation(sphere1, node3, 0);

      stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
      bulk_data_ptr->change_ghosting(custom_ghosting, {});
      bulk_data_ptr->modification_end();

      // As well as node2 sent to us their spring and node.
      stk::mesh::Entity spring1 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
      stk::mesh::Entity node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
      ASSERT_TRUE(bulk_data_ptr->is_valid(node2));

      // Declare the linker and connect it to the nodes. Use fixup_ghosted_to_shared_nodes to promote node2 from ghosted
      // to shared.
      bulk_data_ptr->modification_begin();
      bulk_data_ptr->declare_relation(linker1, node2, 0);
      bulk_data_ptr->declare_relation(linker1, node3, 1);
      bulk_data_ptr->declare_relation(linker1, node4, 2);

      stk::mesh::field_data(linked_entities_field, linker1)[0] = bulk_data_ptr->entity_key(spring1);
      stk::mesh::field_data(linked_entities_field, linker1)[1] = bulk_data_ptr->entity_key(sphere1);
      stk::mesh::field_data(linked_entities_field, linker1)[2] = bulk_data_ptr->entity_key(spring2);

      stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
      bulk_data_ptr->modification_end();

      // At this point, the linker has been declared and node2 should be shared but the linked entities are not yet
      // ghosted
      bulk_data_ptr->modification_begin();
      mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr, linked_entities_field, linked_entity_owners_field,
                                                   linker_part);
      bulk_data_ptr->modification_end();

      // Now, everything should be up to date.
      stk::mesh::Entity node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
      EXPECT_TRUE(bulk_data_ptr->is_valid(linker1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(spring2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(sphere1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node1));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node2));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node3));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node4));
      EXPECT_TRUE(bulk_data_ptr->is_valid(node5));

      // Check buckets.
      EXPECT_TRUE(bulk_data_ptr->bucket(linker1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(linker1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(spring1).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(spring2).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(spring2).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(sphere1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(sphere1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node1).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node1).shared());
      EXPECT_TRUE(bulk_data_ptr->bucket(node1).in_aura());

      EXPECT_FALSE(bulk_data_ptr->bucket(node2).owned());
      EXPECT_TRUE(bulk_data_ptr->bucket(node2).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node2).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node3).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node3).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node4).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node4).in_aura());

      EXPECT_TRUE(bulk_data_ptr->bucket(node5).owned());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).shared());
      EXPECT_FALSE(bulk_data_ptr->bucket(node5).in_aura());

      // Check the connectivity.
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 3u);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node2, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node3, stk::topology::CONSTRAINT_RANK)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(node4, stk::topology::CONSTRAINT_RANK)[0]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);

      EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(sphere1)[0]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(linker1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring2), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node3), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node4), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node5), 1);
    }
  }
}
//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
