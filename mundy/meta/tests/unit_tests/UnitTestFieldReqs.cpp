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

namespace mundy {

namespace meta {

namespace {

/*
Fields are defined by their name, rank, dimension, number of states, type, and attributes. By default, FieldReqs
only constrains the field's type, but partial requirements are allows; for example, a FieldReqs object can
constrain field name and rank, but not number of components (aka. dimension). A FieldReqs object can also be
used to generate a Field, given that the FieldReqs object is fully specified. In the meantime, FieldReqs
can be merged together to create a new FieldReqs object that is the union of the two FieldReqs objects.

The following tests check that the FieldReqs object is working as expected. The tests are organized into
sections based on the functionality being tested. The sections are as follows:
  -# FieldReqs object construction
  -# FieldReqs object setting
  -# FieldReqs object getting
  -# FieldReqs object deleting
  -# FieldReqs object merging
  -# FieldReqs object declaring
*/

//! \name FieldReqs object construction tests
//@{

TEST(FieldReqsConstruction, IsDefaultConstructable) {
  // Check that the default constructor works.
  using ExampleFieldType = double;
  ASSERT_NO_THROW(mundy::meta::FieldReqs<ExampleFieldType>());
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  EXPECT_FALSE(field_reqs.is_fully_specified());
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
}
//@}

//! \name FieldReqs object setting tests
//@{

TEST(FieldReqsSetters, IsSettable) {
  // Check that the setters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_name(field_name);
  EXPECT_TRUE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_rank(field_rank);
  EXPECT_TRUE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_dimension(field_dimension);
  EXPECT_TRUE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  EXPECT_TRUE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_TRUE(field_reqs.is_fully_specified());
}

TEST(FieldReqsAttributes, AddAttributes) {
  // Check that the attribute adders work.

  // Add the attribute to the FieldReqs object.
  using ExampleFieldType = double;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  const std::string attribute_name = "attribute_name";
  ASSERT_NO_THROW(field_reqs.add_field_attribute(attribute_name));

  // Check that the attribute is in the FieldReqs object.
  ASSERT_EQ(field_reqs.get_field_attribute_names().size(), 1);
  EXPECT_EQ(field_reqs.get_field_attribute_names()[0], attribute_name);
}
//@}

//! \name FieldReqs object getting tests
//@{

TEST(FieldReqsGetters, IsGettable) {
  // Check that the getters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  EXPECT_THROW(field_reqs.get_field_name(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_rank(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_dimension(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_min_num_states(), std::logic_error);
  std::cout << "Threw as expected" << std::endl;

  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  std::cout << "Set without error" << std::endl;

  EXPECT_EQ(field_reqs.get_field_name(), field_name);
  std::cout << "get_field_name finished" << std::endl;

  EXPECT_EQ(field_reqs.get_field_rank(), field_rank);
  std::cout << "get_field_rank finished" << std::endl;

  EXPECT_EQ(field_reqs.get_field_dimension(), field_dimension);
  std::cout << "get_field_dimension finished" << std::endl;

  EXPECT_EQ(field_reqs.get_field_min_num_states(), field_min_number_of_states);
  std::cout << "get_field_min_num_states finished" << std::endl;

  EXPECT_EQ(field_reqs.get_field_type_info(), typeid(ExampleFieldType));
  std::cout << "get_field_type_info finished" << std::endl;
  std::cout << "Set set correctly and got got correctly" << std::endl;
}
//@}

//! \name FieldReqs object deleting tests
//@{

TEST(FieldReqsDeleters, DeletersWorkProperly) {
  // Check that the getters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  field_reqs.delete_field_name();
  field_reqs.delete_field_rank();
  field_reqs.delete_field_dimension();
  field_reqs.delete_field_min_number_of_states();
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_FALSE(field_reqs.is_fully_specified());
}
//@}

//! \name FieldReqs object merging tests
//@{

TEST(FieldReqsSync, IsSyncable) {
  // Check that the sync function works.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto main_field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldReqs<ExampleFieldType>>();
  main_field_reqs.set_field_name(field_name);
  main_field_reqs.set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states);
  main_field_reqs.sync(other_field_reqs_ptr);

  // Both field requirements should be synced (aka. merged and had their differences rectified)
  EXPECT_TRUE(main_field_reqs.is_fully_specified());
  EXPECT_EQ(main_field_reqs.get_field_name(), field_name);
  EXPECT_EQ(main_field_reqs.get_field_rank(), field_rank);
  EXPECT_EQ(main_field_reqs.get_field_dimension(), field_dimension);
  EXPECT_EQ(main_field_reqs.get_field_min_num_states(), field_min_number_of_states);
  EXPECT_EQ(main_field_reqs.get_field_type_info(), typeid(ExampleFieldType));

  EXPECT_TRUE(other_field_reqs_ptr->is_fully_specified());
  EXPECT_EQ(other_field_reqs_ptr->get_field_name(), field_name);
  EXPECT_EQ(other_field_reqs_ptr->get_field_rank(), field_rank);
  EXPECT_EQ(other_field_reqs_ptr->get_field_dimension(), field_dimension);
  EXPECT_EQ(other_field_reqs_ptr->get_field_min_num_states(), field_min_number_of_states);
  EXPECT_EQ(other_field_reqs_ptr->get_field_type_info(), typeid(ExampleFieldType));
}

TEST(FieldReqsSync, SyncProperlyHandlesNullptr) {
  // Check that the sync function throws when given a nullptr.
  using ExampleFieldType = double;
  auto main_field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  std::shared_ptr<mundy::meta::FieldReqsBase> other_field_reqs_ptr = nullptr;
  EXPECT_THROW(main_field_reqs.sync(other_field_reqs_ptr), std::invalid_argument);
}

TEST(FieldReqsSync, SyncProperlyHandlesConflicts) {
  // Check that the sync function works with a conflict. It should throw a logic error.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Check that the sync function throws a logic error if the field name, rank, or dimension are different.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldReqs<ExampleFieldType>>();
  other_field_reqs_ptr->set_field_name("other_field_name");
  EXPECT_THROW(field_reqs.sync(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_name();

  other_field_reqs_ptr->set_field_rank(stk::topology::EDGE_RANK);
  EXPECT_THROW(field_reqs.sync(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_rank();

  other_field_reqs_ptr->set_field_dimension(4);
  EXPECT_THROW(field_reqs.sync(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_dimension();

  // Notice that the field min number of states is always valid to sync.
  other_field_reqs_ptr->set_field_min_number_of_states(3);
  EXPECT_NO_THROW(field_reqs.sync(other_field_reqs_ptr));
  other_field_reqs_ptr->delete_field_min_number_of_states();
}

TEST(FieldReqsSync, SyncProperlyHandlesDifferentTypes) {
  // Check that the sync function works with a conflict. It should throw a logic error.
  using ExampleFieldType1 = double;
  using ExampleFieldType2 = int;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType1>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Notice that the field type is different. This should result in a logic error during the sync.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldReqs<ExampleFieldType2>>();
  other_field_reqs_ptr->set_field_name(field_name);
  other_field_reqs_ptr->set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states);
  EXPECT_THROW(field_reqs.sync(other_field_reqs_ptr), std::logic_error);
}

TEST(FieldReqsSync, SyncProperlyHandlesMinNumStates) {
  // Check that the sync function works with a conflict. It should throw a logic error.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Create a new field requirements with a larger min number of states.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldReqs<ExampleFieldType>>();
  other_field_reqs_ptr->set_field_name(field_name);
  other_field_reqs_ptr->set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states + 1);

  // The sync should increase the min number of states to the maximum of the two values.
  field_reqs.sync(other_field_reqs_ptr);
  EXPECT_EQ(field_reqs.get_field_min_num_states(), field_min_number_of_states + 1);
  EXPECT_EQ(other_field_reqs_ptr->get_field_min_num_states(), field_min_number_of_states + 1);

  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states - 1);
  EXPECT_EQ(field_reqs.get_field_min_num_states(), field_min_number_of_states - 1);
  EXPECT_EQ(other_field_reqs_ptr->get_field_min_num_states(), field_min_number_of_states - 1);
}
//@}

//! \name FieldReqs object declaring tests
//@{

TEST(FieldReqsDeclare, DeclareOnPart) {
  // Check that the field requirements match the declared field.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  ASSERT_TRUE(field_reqs.is_fully_specified());

  // Create a dummy mesh with an example part.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  stk::mesh::Part &example_part = meta_data.declare_part("example_part", field_rank);

  // Declare the field on the example part and validate that the resulting field matches the requirements.
  field_reqs.declare_field_on_part(&bulk_data_ptr->mesh_meta_data(), example_part);
  stk::mesh::Field<ExampleFieldType> *my_field_ptr = meta_data.get_field<ExampleFieldType>(field_rank, field_name);
  ASSERT_NE(my_field_ptr, nullptr);
  EXPECT_EQ(my_field_ptr->name(), field_name);
  EXPECT_EQ(my_field_ptr->entity_rank(), field_rank);
  EXPECT_EQ(my_field_ptr->max_size(), field_dimension);
  EXPECT_EQ(my_field_ptr->number_of_states(), field_min_number_of_states);
}

TEST(FieldReqsDeclare, DeclareOnEntireMesh) {
  // Check that the field requirements match the declared field.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldReqs<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  ASSERT_TRUE(field_reqs.is_fully_specified());

  // Create a dummy mesh with an example part.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare the field on the example part and validate that the resulting field matches the requirements.
  field_reqs.declare_field_on_entire_mesh(&bulk_data_ptr->mesh_meta_data());
  stk::mesh::Field<ExampleFieldType> *my_field_ptr = meta_data.get_field<ExampleFieldType>(field_rank, field_name);
  ASSERT_NE(my_field_ptr, nullptr);
  EXPECT_EQ(my_field_ptr->name(), field_name);
  EXPECT_EQ(my_field_ptr->entity_rank(), field_rank);
  EXPECT_EQ(my_field_ptr->max_size(), field_dimension);
  EXPECT_EQ(my_field_ptr->number_of_states(), field_min_number_of_states);
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy
