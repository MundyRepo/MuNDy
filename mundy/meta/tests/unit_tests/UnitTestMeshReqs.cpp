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
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>    // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase
#include <mundy_meta/MeshReqs.hpp>       // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>       // for mundy::meta::PartReqs

namespace mundy {

namespace meta {

namespace {

/*
MetaMeshs are rather complex, with an assortment of requirements that must be met before a MetaMesh can be created.
These requirements are explicitly listed in the MeshBuilder class, but they are also implicitly listed in the
MeshReqs class. The MeshReqs class is a helper class that is used to collect and sync requirements
for a MetaMesh. The MeshReqs class can also used to generate a MetaMesh, given that the MeshReqs object
is fully specified; here, fully specified only requires that the mesh communicator is set. In the meantime,
MeshReqs can be merged together to create a new MeshReqs object that is the union of the two
MeshReqs objects.

The following tests check that the MeshReqs object is working as expected. The tests are organized into
sections based on the functionality being tested. The sections are as follows:
  -# MeshReqs object construction
  -# MeshReqs object setting
  -# MeshReqs object getting
  -# MeshReqs object deleting
  -# MeshReqs object merging
  -# MeshReqs object declaring
*/

//! \name MeshReqs object construction tests
//@{

TEST(MeshReqsConstruction, IsDefaultConstructible) {
  // Check that MeshReqs is default constructible
  ASSERT_NO_THROW(MeshReqs mesh_reqs);
}

TEST(MeshReqsConstruction, IsConstructibleWithComm) {
  // Check that MeshReqs is constructible with a communicator
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  ASSERT_NO_THROW(MeshReqs mesh_reqs(comm));
}
//@}

//! \name MeshReqs object setting tests
//@{

TEST(MeshReqsSetters, IsSettable) {
  // Check that the setters work.
  MeshReqs mesh_reqs;
  EXPECT_FALSE(mesh_reqs.constrains_spatial_dimension());
  EXPECT_FALSE(mesh_reqs.constrains_entity_rank_names());
  EXPECT_FALSE(mesh_reqs.constrains_communicator());
  EXPECT_FALSE(mesh_reqs.constrains_aura_option());
  EXPECT_FALSE(mesh_reqs.constrains_field_data_manager());
  EXPECT_FALSE(mesh_reqs.constrains_bucket_capacity());
  EXPECT_FALSE(mesh_reqs.constrains_upward_connectivity_flag());
  mesh_reqs.set_spatial_dimension(3);
  EXPECT_TRUE(mesh_reqs.constrains_spatial_dimension());
  mesh_reqs.set_entity_rank_names(std::vector<std::string>());
  EXPECT_TRUE(mesh_reqs.constrains_entity_rank_names());
  mesh_reqs.set_communicator(MPI_COMM_WORLD);
  EXPECT_TRUE(mesh_reqs.constrains_communicator());
  mesh_reqs.set_aura_option(stk::mesh::BulkData::AUTO_AURA);
  EXPECT_TRUE(mesh_reqs.constrains_aura_option());
  mesh_reqs.set_field_data_manager(nullptr);
  EXPECT_TRUE(mesh_reqs.constrains_field_data_manager());
  mesh_reqs.set_bucket_capacity(1024);
  EXPECT_TRUE(mesh_reqs.constrains_bucket_capacity());
  mesh_reqs.set_upward_connectivity_flag(true);
  EXPECT_TRUE(mesh_reqs.constrains_upward_connectivity_flag());
}

TEST(MeshRequiremesntsSetters, AddFieldReqs) {
  // Check that the add_and_sync_field_reqs method works.

  // Create a dummy field requirements object.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs_ptr = std::make_shared<mundy::meta::FieldReqs<ExampleFieldType>>(
      field_name, field_rank, field_dimension, field_min_number_of_states);

  // Create a mesh requirements object and add the field requirements to it.
  MeshReqs mesh_reqs(MPI_COMM_WORLD);
  ASSERT_NO_THROW(mesh_reqs.add_and_sync_field_reqs(field_reqs_ptr));

  // Check that the field requirements were added correctly.
  // TODO(palmerb4): Add a field requirements getter method so we can perform this check.
}

TEST(MeshReqsSetters, AddPartReqs) {
  // Check that the add_and_sync_part_reqs method works.

  // Create a dummy part requirements object.
  const std::string part_name = "part_name";
  auto part_reqs_ptr = std::make_shared<mundy::meta::PartReqs>(part_name);

  // Create a mesh requirements object and add the part requirements to it.
  MeshReqs mesh_reqs(MPI_COMM_WORLD);
  ASSERT_NO_THROW(mesh_reqs.add_and_sync_part_reqs(part_reqs_ptr));

  // Check that the part requirements were added correctly.
  // TODO(palmerb4): Add a part requirements getter method so we can perform this check.
}

TEST(MeshReqsSetters, AddMeshAttributes) {
  // Check that the add_mesh_attribute method works.

  // Create a dummy mesh attribute object.
  std::string attribute_name = "attribute_name";

  // Create a mesh requirements object and add the mesh attribute to it.
  MeshReqs mesh_reqs(MPI_COMM_WORLD);
  ASSERT_NO_THROW(mesh_reqs.add_mesh_attribute(attribute_name));

  // Check that the mesh attribute was added correctly.
  ASSERT_EQ(mesh_reqs.get_mesh_attribute_names().size(), 1);
  EXPECT_EQ(mesh_reqs.get_mesh_attribute_names()[0], attribute_name);
}
//@}

//! \name MeshReqs object getting tests
//@{

TEST(MeshReqsGetters, IsGettable) {
  // Check that the getters work.
  MeshReqs mesh_reqs;
  EXPECT_THROW(mesh_reqs.get_spatial_dimension(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_entity_rank_names(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_communicator(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_aura_option(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_field_data_manager(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_bucket_capacity(), std::logic_error);
  EXPECT_THROW(mesh_reqs.get_upward_connectivity_flag(), std::logic_error);
  mesh_reqs.set_spatial_dimension(3);
  mesh_reqs.set_entity_rank_names(std::vector<std::string>());
  mesh_reqs.set_communicator(MPI_COMM_WORLD);
  mesh_reqs.set_aura_option(stk::mesh::BulkData::AUTO_AURA);
  mesh_reqs.set_field_data_manager(nullptr);
  mesh_reqs.set_bucket_capacity(1024);
  mesh_reqs.set_upward_connectivity_flag(true);
  EXPECT_EQ(mesh_reqs.get_spatial_dimension(), 3);
  EXPECT_EQ(mesh_reqs.get_entity_rank_names(), std::vector<std::string>());
  EXPECT_EQ(mesh_reqs.get_communicator(), MPI_COMM_WORLD);
  EXPECT_EQ(mesh_reqs.get_aura_option(), stk::mesh::BulkData::AUTO_AURA);
  EXPECT_EQ(mesh_reqs.get_field_data_manager(), nullptr);
  EXPECT_EQ(mesh_reqs.get_bucket_capacity(), 1024);
  EXPECT_EQ(mesh_reqs.get_upward_connectivity_flag(), true);
}
//@}

//! \name MeshReqs object deleting tests
//@{

TEST(MeshReqsDeleters, IsDeletable) {
  // Check that the deleters work.
  MeshReqs mesh_reqs;
  EXPECT_FALSE(mesh_reqs.constrains_spatial_dimension());
  EXPECT_FALSE(mesh_reqs.constrains_entity_rank_names());
  EXPECT_FALSE(mesh_reqs.constrains_communicator());
  EXPECT_FALSE(mesh_reqs.constrains_aura_option());
  EXPECT_FALSE(mesh_reqs.constrains_field_data_manager());
  EXPECT_FALSE(mesh_reqs.constrains_bucket_capacity());
  EXPECT_FALSE(mesh_reqs.constrains_upward_connectivity_flag());
  mesh_reqs.set_spatial_dimension(3);
  mesh_reqs.set_entity_rank_names(std::vector<std::string>());
  mesh_reqs.set_communicator(MPI_COMM_WORLD);
  mesh_reqs.set_aura_option(stk::mesh::BulkData::AUTO_AURA);
  mesh_reqs.set_field_data_manager(nullptr);
  mesh_reqs.set_bucket_capacity(1024);
  mesh_reqs.set_upward_connectivity_flag(true);
  EXPECT_TRUE(mesh_reqs.constrains_spatial_dimension());
  EXPECT_TRUE(mesh_reqs.constrains_entity_rank_names());
  EXPECT_TRUE(mesh_reqs.constrains_communicator());
  EXPECT_TRUE(mesh_reqs.constrains_aura_option());
  EXPECT_TRUE(mesh_reqs.constrains_field_data_manager());
  EXPECT_TRUE(mesh_reqs.constrains_bucket_capacity());
  EXPECT_TRUE(mesh_reqs.constrains_upward_connectivity_flag());
  mesh_reqs.delete_spatial_dimension();
  mesh_reqs.delete_entity_rank_names();
  mesh_reqs.delete_communicator();
  mesh_reqs.delete_aura_option();
  mesh_reqs.delete_field_data_manager();
  mesh_reqs.delete_bucket_capacity();
  mesh_reqs.delete_upward_connectivity_flag();
  EXPECT_FALSE(mesh_reqs.constrains_spatial_dimension());
  EXPECT_FALSE(mesh_reqs.constrains_entity_rank_names());
  EXPECT_FALSE(mesh_reqs.constrains_communicator());
  EXPECT_FALSE(mesh_reqs.constrains_aura_option());
  EXPECT_FALSE(mesh_reqs.constrains_field_data_manager());
  EXPECT_FALSE(mesh_reqs.constrains_bucket_capacity());
  EXPECT_FALSE(mesh_reqs.constrains_upward_connectivity_flag());
}
//@}

//! \name MeshReqs object merging tests
//@{

TEST(MeshReqsMerge, IsMergeable) {
  // Check that the sync method works.
  auto mesh_reqs1_ptr = std::make_shared<MeshReqs>();
  auto mesh_reqs2_ptr = std::make_shared<MeshReqs>();
  mesh_reqs1_ptr->set_spatial_dimension(3);
  mesh_reqs1_ptr->set_entity_rank_names(std::vector<std::string>());
  mesh_reqs1_ptr->set_communicator(MPI_COMM_WORLD);
  mesh_reqs1_ptr->set_aura_option(stk::mesh::BulkData::AUTO_AURA);
  mesh_reqs2_ptr->set_field_data_manager(nullptr);
  mesh_reqs2_ptr->set_bucket_capacity(1024);
  mesh_reqs2_ptr->set_upward_connectivity_flag(true);
  mesh_reqs1_ptr->sync(mesh_reqs2_ptr);
  EXPECT_EQ(mesh_reqs1_ptr->get_spatial_dimension(), 3);
  EXPECT_EQ(mesh_reqs1_ptr->get_entity_rank_names(), std::vector<std::string>());
  EXPECT_EQ(mesh_reqs1_ptr->get_communicator(), MPI_COMM_WORLD);
  EXPECT_EQ(mesh_reqs1_ptr->get_aura_option(), stk::mesh::BulkData::AUTO_AURA);
  EXPECT_EQ(mesh_reqs1_ptr->get_field_data_manager(), nullptr);
  EXPECT_EQ(mesh_reqs1_ptr->get_bucket_capacity(), 1024);
  EXPECT_EQ(mesh_reqs1_ptr->get_upward_connectivity_flag(), true);
}

TEST(MeshReqsMerge, AreFieldsMergable) {
  /* Check that the sync properly merges fields.
  The setup for this test is as follows:
  mesh1
    field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
    field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
  mesh2:
    field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
    field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
  merged12
    field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    field2 merged w/ field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
    field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
    field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)

  Note, only fields with the same name AND rank will be merged. As a result, field2 will be merged with field 4 (this
  should increase it's minimim number of states to 4), but field3 will not be merged with field 5 because they don't
  have the same rank.
  */

  // Setup the dummy fields.
  using ExampleFieldType = double;
  auto field_reqs1_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("field1", stk::topology::NODE_RANK, 3, 1);
  auto field_reqs2_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("field2", stk::topology::NODE_RANK, 3, 2);
  auto field_reqs3_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("field3", stk::topology::ELEMENT_RANK, 3, 3);
  auto field_reqs4_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("field4", stk::topology::NODE_RANK, 3, 4);
  auto field_reqs5_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("field5", stk::topology::NODE_RANK, 3, 5);

  // Setup the mesh requirements according to the diagram above.
  auto mesh_reqs1_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  auto mesh_reqs2_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs1_ptr->add_and_sync_field_reqs(field_reqs1_ptr);
  mesh_reqs1_ptr->add_and_sync_field_reqs(field_reqs2_ptr);
  mesh_reqs1_ptr->add_and_sync_field_reqs(field_reqs3_ptr);
  mesh_reqs2_ptr->add_and_sync_field_reqs(field_reqs4_ptr);
  mesh_reqs2_ptr->add_and_sync_field_reqs(field_reqs5_ptr);

  // Synchronize (merge and rectify differences) the mesh requirements and check that the fields were merged correctly.
  ASSERT_NO_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr));
  // TODO(palmerb4): Use the field getters to check that the fields were merged correctly.
}

TEST(MeshReqsMerge, AreMeshAttributesMergable) {
  /* Check that the sync function properly merges mesh attributes.
  The setup for this test is as follows:
  mesh1
    attribute1 (name="attribute1")
    attribute2 (name="attribute23")
  mesh2:
    attribute3 (name="attribute23")
    attribute4 (name="attribute4")
  merged12
    attribute1 (name="attribute1")
    attribute2 merged w/ attribute3 (name="attribute23")
    attribute4 (name="attribute4")

  Note, only attributes with the same name will be merged. As a result, attribute2 will be merged with attribute 3.
  */

  // Setup the dummy attributes.
  std::string attribute1_name = "attribute1";
  std::string attribute2_name = "attribute23";
  std::string attribute3_name = "attribute23";
  std::string attribute4_name = "attribute4";

  // Setup the mesh requirements according to the diagram above.
  auto mesh_reqs1_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  auto mesh_reqs2_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs1_ptr->add_mesh_attribute(attribute1_name);
  mesh_reqs1_ptr->add_mesh_attribute(attribute2_name);
  mesh_reqs2_ptr->add_mesh_attribute(attribute3_name);
  mesh_reqs2_ptr->add_mesh_attribute(attribute4_name);

  // Synchronize (merge and rectify differences) the mesh requirements and check that the attributes were merged
  // correctly.
  ASSERT_NO_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr));

  // Check that the attributes were merged correctly.
  ASSERT_EQ(mesh_reqs1_ptr->get_mesh_attribute_names().size(), 3);
  const auto &merged_attribute_names = mesh_reqs1_ptr->get_mesh_attribute_names();
  const bool attribute1_exists =
      std::count(merged_attribute_names.begin(), merged_attribute_names.end(), attribute1_name) > 0;
  const bool attribute2_exists =
      std::count(merged_attribute_names.begin(), merged_attribute_names.end(), attribute2_name) > 0;
  const bool attribute4_exists =
      std::count(merged_attribute_names.begin(), merged_attribute_names.end(), attribute4_name) > 0;

  EXPECT_TRUE(attribute1_exists);
  EXPECT_TRUE(attribute2_exists);
  EXPECT_TRUE(attribute4_exists);
}

TEST(MeshReqsMerge, AreMeshPartsAndTheirFieldsMergable) {
  /* Check that the sync function properly merges mesh parts and their fields/subparts/attributes.
  The setup for this test is as follows:
  mesh1
    part1 (name=A)
      field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
      field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
      subpart1 (name=C)
      attribute1 (name="attribute12")
  mesh2:
    part2 (name=A)
      field3 (name=a, rank=NODE, dimension=3, min_number_of_states=3)
      field4 (name=b, rank=ELEMENT, dimension=3, min_number_of_states=4)
      field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
      attribute2 (name="attribute12")
      attribute3 (name="attribute3")
    part3 (name=B)
  merged12:
    part1 merged w/ part2 (name=A)
      field1 merged w/ field3 (name=a, rank=NODE, dimension=3, min_number_of_states=3)
      field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
      field4 (name=b, rank=ELEMENT, dimension=3, min_number_of_states=4)
      field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
      attribute1 (name="attribute12")
      attribute3 (name="attribute3")
    part2 (name=B)
  */

  // Setup the dummy fields.
  using ExampleFieldType = double;
  auto field_reqs1_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("a", stk::topology::NODE_RANK, 3, 1);
  auto field_reqs2_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("b", stk::topology::NODE_RANK, 3, 2);
  auto field_reqs3_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("a", stk::topology::NODE_RANK, 3, 3);
  auto field_reqs4_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("b", stk::topology::ELEMENT_RANK, 3, 4);
  auto field_reqs5_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("c", stk::topology::NODE_RANK, 3, 5);

  // Setup the dummy attributes.
  std::string attribute1_name = "attribute12";
  std::string attribute2_name = "attribute12";
  std::string attribute3_name = "attribute3";

  // Setup the dummy parts.
  auto part_reqs1_ptr = std::make_shared<PartReqs>("A");
  auto part_reqs2_ptr = std::make_shared<PartReqs>("B");
  auto subpart_reqs1_ptr = std::make_shared<PartReqs>("C");
  part_reqs1_ptr->add_and_sync_field_reqs(field_reqs1_ptr);
  part_reqs1_ptr->add_and_sync_field_reqs(field_reqs2_ptr);
  part_reqs2_ptr->add_and_sync_field_reqs(field_reqs3_ptr);
  part_reqs2_ptr->add_and_sync_field_reqs(field_reqs4_ptr);
  part_reqs2_ptr->add_and_sync_field_reqs(field_reqs5_ptr);
  part_reqs1_ptr->add_and_sync_subpart_reqs(subpart_reqs1_ptr);
  part_reqs1_ptr->add_part_attribute(attribute1_name);
  part_reqs2_ptr->add_part_attribute(attribute2_name);
  part_reqs2_ptr->add_part_attribute(attribute3_name);

  // Setup the mesh requirements according to the diagram above.
  auto mesh_reqs1_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  auto mesh_reqs2_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs1_ptr->add_and_sync_part_reqs(part_reqs1_ptr);
  mesh_reqs2_ptr->add_and_sync_part_reqs(part_reqs2_ptr);

  // Synchronize (merge and rectify differences) the mesh requirements and check that the parts were merged correctly.
  ASSERT_NO_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr));
  // TODO(palmerb4): Add attribute/field/part getters so we can perform this check.
}

TEST(MeshReqsMerge, MergePropertlyHandlesNullptr) {
  // Check that the sync function properly handles nullptrs. It should throw an invalid argument error.

  // Setup the Mesh requirements.
  MeshReqs mesh_reqs(MPI_COMM_WORLD);

  // Perform the sync.
  ASSERT_THROW(mesh_reqs.sync(nullptr), std::invalid_argument);
}

TEST(MeshReqsMerge, MergePropertlyHandlesConflicts) {
  // Check that the sync function throws a logic error if any of the constrained quantities differ.
  auto mesh_reqs1_ptr = std::make_shared<MeshReqs>();
  auto mesh_reqs2_ptr = std::make_shared<MeshReqs>();

  mesh_reqs1_ptr->set_spatial_dimension(3);
  mesh_reqs2_ptr->set_spatial_dimension(2);
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_spatial_dimension();
  mesh_reqs2_ptr->delete_spatial_dimension();

  mesh_reqs1_ptr->set_entity_rank_names(std::vector<std::string>());
  mesh_reqs2_ptr->set_entity_rank_names({"NODE", "ELEMENT"});
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_entity_rank_names();
  mesh_reqs2_ptr->delete_entity_rank_names();

  mesh_reqs1_ptr->set_communicator(MPI_COMM_WORLD);
  mesh_reqs2_ptr->set_communicator(MPI_COMM_NULL);
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_communicator();
  mesh_reqs2_ptr->delete_communicator();

  mesh_reqs1_ptr->set_aura_option(stk::mesh::BulkData::AUTO_AURA);
  mesh_reqs2_ptr->set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_aura_option();
  mesh_reqs2_ptr->delete_aura_option();

  auto field_data_manager_ptr = std::make_unique<stk::mesh::DefaultFieldDataManager>(1, 4);
  mesh_reqs1_ptr->set_field_data_manager(nullptr);
  mesh_reqs2_ptr->set_field_data_manager(field_data_manager_ptr.get());
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_field_data_manager();
  mesh_reqs2_ptr->delete_field_data_manager();

  mesh_reqs1_ptr->set_bucket_capacity(1024);
  mesh_reqs2_ptr->set_bucket_capacity(512);
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_bucket_capacity();
  mesh_reqs2_ptr->delete_bucket_capacity();

  mesh_reqs1_ptr->set_upward_connectivity_flag(true);
  mesh_reqs2_ptr->set_upward_connectivity_flag(false);
  EXPECT_THROW(mesh_reqs1_ptr->sync(mesh_reqs2_ptr), std::logic_error);
  mesh_reqs1_ptr->delete_upward_connectivity_flag();
  mesh_reqs2_ptr->delete_upward_connectivity_flag();
}
//@}

//! \name MeshReqs object declaring tests
//@{

TEST(MeshReqsDeclare, DeclareMeshWithComm) {
  // Check that the declare function properly declares the mesh requirements.

  // Setup the Mesh requirements.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  ASSERT_TRUE(mesh_reqs_ptr->is_fully_specified());

  // Declare the mesh requirements.
  ASSERT_NO_THROW(mesh_reqs_ptr->declare_mesh());
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  EXPECT_NE(bulk_data_ptr, nullptr);
}

TEST(MeshReqsDeclare, DeclareMeshWithoutComm) {
  // Check that the mesh requirements throw an error if the communicator is not set.

  // Setup the Mesh requirements.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>();
  ASSERT_FALSE(mesh_reqs_ptr->is_fully_specified());

  // Declare the mesh requirements.
  ASSERT_THROW(mesh_reqs_ptr->declare_mesh(), std::logic_error);
}

TEST(MeshReqsDeclare, DeclareMeshWithCommAndFields) {
  // Check that the declare function properly declares a mesh with fields.

  // Create a dummy field requirements object.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs_ptr = std::make_shared<FieldReqs<ExampleFieldType>>(field_name, field_rank, field_dimension,
                                                                      field_min_number_of_states);
  ASSERT_TRUE(field_reqs_ptr->is_fully_specified());

  // Setup the Mesh requirements.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->add_and_sync_field_reqs(field_reqs_ptr);
  ASSERT_TRUE(mesh_reqs_ptr->is_fully_specified());

  // Declare the mesh requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  ASSERT_NE(bulk_data_ptr, nullptr);
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  ASSERT_NO_THROW(meta_data.get_field<ExampleFieldType>(field_rank, field_name));
}

TEST(MeshReqsDeclare, DeclareMeshWithCommAndDimAndParts) {
  // Check that the declare function properly declares a mesh with parts.

  // Create a dummy part requirements object.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  auto part_reqs_ptr = std::make_shared<PartReqs>(part_name, part_rank);
  ASSERT_TRUE(part_reqs_ptr->is_fully_specified());

  // Setup the Mesh requirements.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs_ptr);
  mesh_reqs_ptr->set_spatial_dimension(3);
  ASSERT_TRUE(mesh_reqs_ptr->is_fully_specified());

  // Declare the mesh requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  ASSERT_NE(bulk_data_ptr, nullptr);
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  auto part_ptr = meta_data.get_part(part_name);
  ASSERT_NE(part_ptr, nullptr);
}

TEST(MeshReqsDeclare, DeclareMeshWithCommAndAttributes) {
  // Check that the declare function properly declares a mesh with attributes.

  // Create a dummy attributes.
  std::string attribute_name = "attribute";

  // Setup the Mesh requirements.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->add_mesh_attribute(attribute_name);
  ASSERT_TRUE(mesh_reqs_ptr->is_fully_specified());

  // Declare the mesh requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  ASSERT_NE(bulk_data_ptr, nullptr);
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  ASSERT_NE(meta_data.get_attribute(attribute_name), nullptr);
}

TEST(MeshReqsDeclare, DeclareComplexMesh) {
  // Check that the declare function works properly for a full realistic example.

  /* Check that the sync function properly merges mesh parts and their fields/subparts/attributes.
  The setup for this test is as follows:
  mesh:
    part1 (name=A)
      field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
      field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
      attribute1 (type=int)
      attribute2 (type=double)
      subpart: (name=C)
    part2 (name=B)
      field3 (name=b, rank=ELEMENT, dimension=3, min_number_of_states=3)
      field4 (name=c, rank=NODE, dimension=3, min_number_of_states=4)
      attribute1 (type=int)
  */

  // Setup the dummy fields.
  using ExampleFieldType = double;
  auto field_reqs1_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("a", stk::topology::NODE_RANK, 3, 1);
  auto field_reqs2_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("b", stk::topology::NODE_RANK, 3, 2);
  auto field_reqs3_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("b", stk::topology::ELEMENT_RANK, 3, 3);
  auto field_reqs4_ptr = std::make_shared<FieldReqs<ExampleFieldType>>("c", stk::topology::NODE_RANK, 3, 4);

  // Setup the dummy attributes.
  std::string attribute1_name = "attribute1";
  std::string attribute2_name = "attribute2";

  // Setup the dummy parts.
  auto part_reqs1_ptr = std::make_shared<PartReqs>("A");
  auto part_reqs2_ptr = std::make_shared<PartReqs>("B");
  auto subpart_reqs1_ptr = std::make_shared<PartReqs>("C");
  part_reqs1_ptr->add_and_sync_field_reqs(field_reqs1_ptr);
  part_reqs1_ptr->add_and_sync_field_reqs(field_reqs2_ptr);
  part_reqs1_ptr->add_and_sync_subpart_reqs(subpart_reqs1_ptr);
  part_reqs1_ptr->add_part_attribute(attribute1_name);
  part_reqs1_ptr->add_part_attribute(attribute2_name);
  part_reqs2_ptr->add_and_sync_field_reqs(field_reqs3_ptr);
  part_reqs2_ptr->add_and_sync_field_reqs(field_reqs4_ptr);
  part_reqs2_ptr->add_part_attribute(attribute1_name);

  // Setup the mesh requirements according to the diagram above.
  auto mesh_reqs_ptr = std::make_shared<MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs1_ptr);
  mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs2_ptr);

  // Declare the mesh requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();

  // Check that the mesh was declared properly.
  ASSERT_NE(bulk_data_ptr, nullptr);
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  auto part1_ptr = meta_data.get_part("A");
  ASSERT_NE(part1_ptr, nullptr);
  EXPECT_EQ(part1_ptr->name(), "A");
  auto part2_ptr = meta_data.get_part("B");
  ASSERT_NE(part2_ptr, nullptr);
  EXPECT_EQ(part2_ptr->name(), "B");
  auto subpart_ptr = meta_data.get_part("C");
  ASSERT_NE(subpart_ptr, nullptr);
  EXPECT_EQ(subpart_ptr->name(), "C");
  EXPECT_NE(part1_ptr->mesh_meta_data_ordinal(), part2_ptr->mesh_meta_data_ordinal());
  EXPECT_NE(part1_ptr->mesh_meta_data_ordinal(), subpart_ptr->mesh_meta_data_ordinal());
  EXPECT_NE(part2_ptr->mesh_meta_data_ordinal(), subpart_ptr->mesh_meta_data_ordinal());
  ASSERT_NE(meta_data.get_field<ExampleFieldType>(stk::topology::NODE_RANK, "a"), nullptr);
  ASSERT_NE(meta_data.get_field<ExampleFieldType>(stk::topology::NODE_RANK, "b"), nullptr);
  ASSERT_NE(meta_data.get_field<ExampleFieldType>(stk::topology::ELEMENT_RANK, "b"), nullptr);
  ASSERT_NE(meta_data.get_field<ExampleFieldType>(stk::topology::NODE_RANK, "c"), nullptr);
  ASSERT_NE(meta_data.get_attribute(*part1_ptr, attribute1_name), nullptr);
  ASSERT_NE(meta_data.get_attribute(*part1_ptr, attribute2_name), nullptr);
  ASSERT_NE(meta_data.get_attribute(*part2_ptr, attribute1_name), nullptr);
  EXPECT_EQ(part1_ptr->subsets().size(), 1);
}

}  // namespace

}  // namespace meta

}  // namespace mundy
