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
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Ioss_ElementBlock.h>  // for Ioss::ElementBlock
#include <Ioss_IOFactory.h>     // for Ioss::IOFactory
#include <Ioss_NodeBlock.h>     // for Ioss::NodeBlock

#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>  // for stk:::mesh::communicate_field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <MundyIo_config.hpp>          // for HAVE_MUNDYIO_MUNDYSHAPES
#include <mundy_io/IOBroker.hpp>       // for mundy::io::IoBroker
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>    // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshReqsAndIsRegisterable
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB

namespace mundy {

namespace io {

namespace {

/* What tests should we run?

IoBroker is the main IO class for mundy, and we want to make sure it works in both serial and parallel environments,
as well as for the accuracy of what it is reading and writing. Based loosely off of the BrownianIORestart.cpp
example, as well as the MundyLinker examples.

This class is a priviledged class, as it needs to be able to modify the mesh, and in the RESTART capacity, can
actually commit the meta data.
*/

//! \name IOBroker static interface implementations unit tests
//@{
TEST(IOBroker, FixedParameterDefaults) {
  // Check the expected default values.
  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  // Check that all of the default parameter values are present and set correctly
  ASSERT_TRUE(fixed_params.isParameter("exodus_database_output_filename"));
  ASSERT_TRUE(fixed_params.isParameter("coordinate_field_name"));
  ASSERT_TRUE(fixed_params.isParameter("transient_coordinate_field_name"));

  // Check for the enabled parts and fields
  ASSERT_TRUE(fixed_params.isParameter("enabled_io_parts"));
}
//@}

// Check if the field role is what we expect
bool check_field_role(std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr, const std::string &field_name,
                      const stk::topology::rank_t &field_rank, const Ioss::Field::RoleType &expected_role) {
  auto field_ptr = meta_data_ptr->get_field(field_rank, field_name);
  const Ioss::Field::RoleType *field_role = stk::io::get_field_role(*field_ptr);
  return (*field_role == expected_role);
}

//! \name IOBroker functional tests
//@{

// Test if we can create a new instance of the IOBroker
TEST(IOBroker, CreateNewInstanceIOAABB) {
  // Attempt to get the mesh requirements using the default parameters of ComputeAABB
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Set up a ComputeAABB function
  Teuchos::ParameterList fixed_params_sphere;
  fixed_params_sphere.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(fixed_params_sphere));

  // Add the TRANSIENT node coordinate field to the requirements so that we have it later
  mesh_reqs_ptr->add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);

  // Get fixed parameters for the IOBroker
  Teuchos::ParameterList fixed_params_iobroker;

  // Set the IO Parts
  std::vector<std::string> default_io_part_names{"SPHERES", "SPHEROCYLINDERS"};
  Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
  fixed_params_iobroker.set("enabled_io_parts", default_array_of_io_part_names, "PARTS with enabled IO.");

  // Set the IO fields
  // Do this for just element rank (for now, each rank is done separately)
  std::vector<std::string> default_io_field_element_names{"ELEMENT_AABB"};
  Teuchos::Array<std::string> default_array_of_io_field_element_names(default_io_field_element_names);
  fixed_params_iobroker.set("enabled_io_fields_element_rank", default_array_of_io_field_element_names,
                            "ELEMENT_RANK fields with enabled IO.");

  // Set custom values for the ComputeAABB methods to work with IO
  fixed_params_iobroker.set("coordinate_field_name", "NODE_COORDS");
  fixed_params_iobroker.set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES");

  // Validate and set
  fixed_params_iobroker.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  auto io_broker_ptr = IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);

  // Print the mesh roles
  io_broker_ptr->print_field_roles();

  // Print out the mesh
  // stk::log_with_time_and_memory(MPI_COMM_WORLD, "Mesh dump");
  // stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr, std::cout);

  // Check that the ELEMENT_AABB is set to transient, TRANSIENT_NODE_COORDINATES is set to transient, and that
  // NODE_COORDS is set to MESH
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  EXPECT_TRUE(check_field_role(meta_data_ptr, "ELEMENT_AABB", stk::topology::ELEMENT_RANK, Ioss::Field::TRANSIENT));
  EXPECT_TRUE(
      check_field_role(meta_data_ptr, "TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, Ioss::Field::TRANSIENT));
  EXPECT_TRUE(check_field_role(meta_data_ptr, "NODE_COORDS", stk::topology::NODE_RANK, Ioss::Field::MESH));
}

// Test if we can write some initial configuration with the iobroker based on ComputeAABB
TEST(IOBroker, WriteInitialConfigAABB) {
  std::string restart_filename = "exodus_mesh_initial.exo";

  // Attempt to get the mesh requirements using the default parameters of ComputeAABB
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Set up a ComputeAABB function
  Teuchos::ParameterList fixed_params_sphere;
  fixed_params_sphere.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(fixed_params_sphere));

  // Add the TRANSIENT node coordinate field to the requirements so that we have it later
  mesh_reqs_ptr->add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);

  // Get fixed parameters for the IOBroker
  Teuchos::ParameterList fixed_params_iobroker;

  // Set the IO Parts
  std::vector<std::string> default_io_part_names{"SPHERES", "SPHEROCYLINDERS"};
  Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
  fixed_params_iobroker.set("enabled_io_parts", default_array_of_io_part_names, "PARTS with enabled IO.");

  // Set the IO fields
  // ELEMENT_RANK
  std::vector<std::string> default_io_field_element_names{"ELEMENT_AABB"};
  Teuchos::Array<std::string> default_array_of_io_field_element_names(default_io_field_element_names);
  fixed_params_iobroker.set("enabled_io_fields_element_rank", default_array_of_io_field_element_names,
                            "ELEMENT_RANK fields with enabled IO.");

  // Set custom values for the ComputeAABB methods to work with IO
  fixed_params_iobroker.set("coordinate_field_name", "NODE_COORDS");
  fixed_params_iobroker.set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES");

  // Set the output filename and file type
  fixed_params_iobroker.set("exodus_database_output_filename", restart_filename);
  fixed_params_iobroker.set("parallel_io_mode", "hdf5");
  fixed_params_iobroker.set("database_purpose", "restart");

  // Validate and set
  fixed_params_iobroker.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  auto io_broker_ptr = IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);
  auto compute_aabb_ptr = mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params_sphere);

  // Hit the button on committing the metadata
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->use_simple_fields();
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
  meta_data_ptr->commit();

  // Get the sphere part
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERES");
  ASSERT_TRUE(sphere_part_ptr != nullptr);

  // Create some fake data (based on UnitTestComputeAABB)
  bulk_data_ptr->modification_begin();
  stk::mesh::EntityId sphere_id = 1;
  stk::mesh::Entity sphere_element =
      bulk_data_ptr->declare_element(sphere_id, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_node = bulk_data_ptr->declare_node(sphere_id);
  bulk_data_ptr->declare_relation(sphere_element, sphere_node, 0);
  bulk_data_ptr->modification_end();

  // Fetch the required fields to set (note that we are only going to write out some of them)
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<double> *radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(radius_field_ptr != nullptr);

  // Set the sphere's position.
  double sphere_position[3] = {1.0, 2.0, 3.0};
  double *node_coords = stk::mesh::field_data(*node_coord_field_ptr, sphere_node);
  node_coords[0] = sphere_position[0];
  node_coords[1] = sphere_position[1];
  node_coords[2] = sphere_position[2];

  // Set the sphere's radius.
  double sphere_radius = 1.5;
  double *radius = stk::mesh::field_data(*radius_field_ptr, sphere_element);
  radius[0] = sphere_radius;

  // Compute the AABB.
  compute_aabb_ptr->execute(*sphere_part_ptr);

  // Print out the mesh
  // stk::log_with_time_and_memory(MPI_COMM_WORLD, "Mesh dump after ComputeAABB");
  // stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr, std::cout);

  // Print the entire IOBroker
  io_broker_ptr->print_io_broker();

  // Setup the output for RESTART
  io_broker_ptr->setup_io_broker();

  // Write the initial config
  io_broker_ptr->write_io_broker(0.0);

  // Close up everything
  io_broker_ptr->finalize_io_broker();
  // Verification of the database contents via direct IOSS access
  {
    Ioss::DatabaseIO *resultsDb = Ioss::IOFactory::create("exodus", restart_filename, Ioss::READ_MODEL, MPI_COMM_WORLD);
    ASSERT_TRUE(resultsDb != nullptr);
    Ioss::Region results(resultsDb);
    // Should have a single step in the database
    EXPECT_EQ(results.get_property("state_count").get_int(), 1);
    // Get the node block (transient coordinates)
    Ioss::NodeBlock *nb = results.get_node_blocks()[0];
    EXPECT_EQ(1u, nb->field_count(Ioss::Field::TRANSIENT));
    EXPECT_TRUE(nb->field_exists("TRANSIENT_NODE_COORDINATES"));

    // Try to grab the data from inside of this
    for (size_t step = 0; step < 1; step++) {
      results.begin_state(static_cast<int>(step + 1));
      std::vector<double> field_data;
      nb->get_field_data("TRANSIENT_NODE_COORDINATES", field_data);
      EXPECT_DOUBLE_EQ(field_data[0], 1.0);
      EXPECT_DOUBLE_EQ(field_data[1], 2.0);
      EXPECT_DOUBLE_EQ(field_data[2], 3.0);
    }

    // Get the element_aabb information (element_block)
    Ioss::ElementBlock *eb = results.get_element_blocks()[0];
    ASSERT_TRUE(eb != nullptr);
    EXPECT_EQ(1u, nb->field_count(Ioss::Field::TRANSIENT));
    EXPECT_TRUE(eb->field_exists("ELEMENT_AABB"));

    // Try to grab the data from inside of the elements
    for (size_t step = 0; step < 1; step++) {
      results.begin_state(static_cast<int>(step + 1));
      std::vector<double> field_data;
      eb->get_field_data("ELEMENT_AABB", field_data);
      EXPECT_DOUBLE_EQ(field_data[0], -0.5);
      EXPECT_DOUBLE_EQ(field_data[1], 0.5);
      EXPECT_DOUBLE_EQ(field_data[2], 1.5);
      EXPECT_DOUBLE_EQ(field_data[3], 2.5);
      EXPECT_DOUBLE_EQ(field_data[4], 3.5);
      EXPECT_DOUBLE_EQ(field_data[5], 4.5);
    }
  }
}

// Test if we can write a results file with ComputeAABB (include integers)
TEST(IOBroker, WriteResultsAABBInteger) {
  std::string results_filename = "exodus_mesh_results.exo";

  // Attempt to get the mesh requirements using the default parameters of ComputeAABB
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Set up a ComputeAABB function
  Teuchos::ParameterList fixed_params_sphere;
  fixed_params_sphere.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(fixed_params_sphere));

  // Add the TRANSIENT node coordinate field to the requirements so that we have it later
  mesh_reqs_ptr->add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);

  // Directly set an unsigned field on everybody
  mesh_reqs_ptr->add_and_sync_field_reqs(
      std::make_shared<mundy::meta::FieldReqs<unsigned>>("NODE_RNG_COUNTER", stk::topology::NODE_RANK, 1, 1));

  // Get fixed parameters for the IOBroker
  Teuchos::ParameterList fixed_params_iobroker;

  // Set the IO Parts
  std::vector<std::string> default_io_part_names{"SPHERES", "SPHEROCYLINDERS"};
  Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
  fixed_params_iobroker.set("enabled_io_parts", default_array_of_io_part_names, "PARTS with enabled IO.");

  // Set the IO fields
  // ELEMENT_RANK
  std::vector<std::string> default_io_field_element_names{"ELEMENT_AABB"};
  Teuchos::Array<std::string> default_array_of_io_field_element_names(default_io_field_element_names);
  fixed_params_iobroker.set("enabled_io_fields_element_rank", default_array_of_io_field_element_names,
                            "ELEMENT_RANK fields with enabled IO.");
  // NODE_RANK
  std::vector<std::string> default_io_field_node_names{"NODE_RNG_COUNTER"};
  Teuchos::Array<std::string> default_array_of_io_field_node_names(default_io_field_node_names);
  fixed_params_iobroker.set("enabled_io_fields_node_rank", default_array_of_io_field_node_names,
                            "NODE_RANK fields with enabled IO.");

  // Set custom values for the ComputeAABB methods to work with IO
  fixed_params_iobroker.set("coordinate_field_name", "NODE_COORDS");
  fixed_params_iobroker.set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES");

  // Set the output filename and file type
  fixed_params_iobroker.set("exodus_database_output_filename", results_filename);
  fixed_params_iobroker.set("parallel_io_mode", "hdf5");
  fixed_params_iobroker.set("database_purpose", "results");

  // Validate and set
  fixed_params_iobroker.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  auto io_broker_ptr = IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);
  auto compute_aabb_ptr = mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params_sphere);

  // Hit the button on committing the metadata
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->commit();

  // Get the sphere part
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_REQUIRE(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");

  // Create multiple spheres (based on SphereBrownianMotionWithContact)
  int num_spheres_local = 10 / bulk_data_ptr->parallel_size();
  const int remaining_spheres = 10 - num_spheres_local * bulk_data_ptr->parallel_size();
  if (bulk_data_ptr->parallel_rank() < remaining_spheres) {
    num_spheres_local += 1;
  }
  int num_nodes_local = num_spheres_local;

  bulk_data_ptr->modification_begin();
  std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_nodes_local;
  requests[stk::topology::ELEMENT_RANK] = num_spheres_local;

  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  // Associate each segments with the sphere part and connect them to their nodes.
  std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_nodes_local + i];
    stk::mesh::Entity node_i = requested_entities[i];
    bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
    bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
  }
  bulk_data_ptr->modification_end();

  // Fetch the required fields to set (note that we are only going to write out some of them)
  stk::mesh::Field<double> *node_coordinates_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<unsigned> *node_rng_counter_field_ptr =
      meta_data_ptr->get_field<unsigned>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");

  // Initialize the spheres position
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity node_i = requested_entities[i];
    stk::mesh::Entity sphere_i = requested_entities[num_nodes_local + i];

    // Get the GID for this sphere
    const stk::mesh::EntityId sphere_node_gid = bulk_data_ptr->identifier(node_i);

    // Set the coordinates, should be a 5x2 grid
    double *node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node_i);
    node_coords[0] = (i % 5);
    node_coords[1] = (i / 5);
    node_coords[2] = 0.0;

    // Set the RNG to the GID of the object
    unsigned *node_rng_counter = stk::mesh::field_data(*node_rng_counter_field_ptr, node_i);
    node_rng_counter[0] = static_cast<unsigned>(sphere_node_gid);

    // Set the radius
    stk::mesh::field_data(*element_radius_field_ptr, sphere_i)[0] = 1.2;
  }

  // Setup the output for RESULTS
  io_broker_ptr->setup_io_broker();

  // Loop over some fictitious times
  for (int i = 0; i < 2; i++) {
    // Now loop over the spheres and do something with the coordinates and write out
    for (int j = 0; j < num_spheres_local; j++) {
      stk::mesh::Entity node_j = requested_entities[j];

      // Set the coordinates, should be a 5x2 grid, just march somewhere else
      double *node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node_j);
      node_coords[0] = node_coords[0] + 1.0;
      node_coords[1] = node_coords[1] + 1.0;
      node_coords[2] = 0.0;
    }

    // ComputeAABB with the adjusted coordinates
    compute_aabb_ptr->execute(*spheres_part_ptr);

    // Write the initial config
    io_broker_ptr->write_io_broker(i);
  }

  // Close up everything
  io_broker_ptr->finalize_io_broker();

  // Verification of the database contents via direct IOSS access
  {
    Ioss::DatabaseIO *resultsDb = Ioss::IOFactory::create("exodus", results_filename, Ioss::READ_MODEL, MPI_COMM_WORLD);
    Ioss::Region results(resultsDb);
    // Should have a single step in the database
    EXPECT_EQ(results.get_property("state_count").get_int(), 2);
    // Get the node block (transient coordinates)
    Ioss::NodeBlock *nb = results.get_node_blocks()[0];
    EXPECT_EQ(2u, nb->field_count(Ioss::Field::TRANSIENT));
    EXPECT_TRUE(nb->field_exists("TRANSIENT_NODE_COORDINATES"));

    // Try to grab the data from inside of this
    for (size_t step = 0; step < 1; step++) {
      results.begin_state(static_cast<int>(step + 1));
      std::vector<double> field_data;
      nb->get_field_data("TRANSIENT_NODE_COORDINATES", field_data);

      // Loop over local spheres, the comparison is complicated because of how it's packed on disk
      for (int j = 0; j < num_spheres_local; j++) {
        EXPECT_DOUBLE_EQ(field_data[3 * j], (j % 5) + static_cast<double>(step) + 1.0);
        EXPECT_DOUBLE_EQ(field_data[3 * j + 1], (j / 5) + static_cast<double>(step) + 1.0);
        EXPECT_DOUBLE_EQ(field_data[3 * j + 2], 0.0);
      }
    }
  }
}

// Test if we can write a restart file and then read it back in with integers
TEST(IOBroker, WriteReadRestartAABBIntegerPart1) {
  std::string restart_filename = "exodus_mesh_restart.exo";

  ////////////////
  // Configure MuNDy and ComputeAABB
  ////////////////

  // Attempt to get the mesh requirements using the default parameters of ComputeAABB
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Set up a ComputeAABB function
  Teuchos::ParameterList fixed_params_sphere;
  fixed_params_sphere.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(fixed_params_sphere));

  // Add the TRANSIENT node coordinate field to the requirements so that we have it later
  mesh_reqs_ptr->add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);

  // Directly set an unsigned field on everybody
  mesh_reqs_ptr->add_and_sync_field_reqs(
      std::make_shared<mundy::meta::FieldReqs<unsigned>>("NODE_RNG_COUNTER", stk::topology::NODE_RANK, 1, 1));

  ////////////////
  // Configure IOBroker
  ////////////////

  // Get fixed parameters for the IOBroker
  Teuchos::ParameterList fixed_params_iobroker;

  // Set the IO Parts
  std::vector<std::string> default_io_part_names{"SPHERES", "SPHEROCYLINDERS"};
  Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
  fixed_params_iobroker.set("enabled_io_parts", default_array_of_io_part_names, "PARTS with enabled IO.");

  // Set the IO fields
  // ELEMENT_RANK
  std::vector<std::string> default_io_field_element_names{"ELEMENT_AABB", "ELEMENT_RADIUS"};
  Teuchos::Array<std::string> default_array_of_io_field_element_names(default_io_field_element_names);
  fixed_params_iobroker.set("enabled_io_fields_element_rank", default_array_of_io_field_element_names,
                            "ELEMENT_RANK fields with enabled IO.");
  // NODE_RANK
  std::vector<std::string> default_io_field_node_names{"NODE_RNG_COUNTER"};
  Teuchos::Array<std::string> default_array_of_io_field_node_names(default_io_field_node_names);
  fixed_params_iobroker.set("enabled_io_fields_node_rank", default_array_of_io_field_node_names,
                            "NODE_RANK fields with enabled IO.");

  // Set custom values for the ComputeAABB methods to work with IO
  fixed_params_iobroker.set("coordinate_field_name", "NODE_COORDS");
  fixed_params_iobroker.set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES");

  // Set the output filename and file type
  fixed_params_iobroker.set("exodus_database_output_filename", restart_filename);
  fixed_params_iobroker.set("parallel_io_mode", "hdf5");
  fixed_params_iobroker.set("database_purpose", "restart");

  // Validate and set
  fixed_params_iobroker.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  ////////////////
  // Build the mesh
  ////////////////

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();

  // Set the simple fields
  bulk_data_ptr->mesh_meta_data_ptr()->use_simple_fields();

  ////////////////
  // Create instances of our objects
  ////////////////
  auto io_broker_ptr = IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);
  auto compute_aabb_ptr = mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params_sphere);

  // Hit the button on committing the metadata
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->commit();

  // Get the sphere part
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_REQUIRE(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");

  // Create multiple spheres (based on SphereBrownianMotionWithContact)
  int num_spheres_local = 10 / bulk_data_ptr->parallel_size();
  const int remaining_spheres = 10 - num_spheres_local * bulk_data_ptr->parallel_size();
  if (bulk_data_ptr->parallel_rank() < remaining_spheres) {
    num_spheres_local += 1;
  }
  int num_nodes_local = num_spheres_local;

  bulk_data_ptr->modification_begin();
  std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_nodes_local;
  requests[stk::topology::ELEMENT_RANK] = num_spheres_local;

  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  // Associate each segments with the sphere part and connect them to their nodes.
  std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_nodes_local + i];
    stk::mesh::Entity node_i = requested_entities[i];
    bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
    bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
  }
  bulk_data_ptr->modification_end();

  // Fetch the required fields to set (note that we are only going to write out some of them)
  stk::mesh::Field<double> *node_coordinates_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<unsigned> *node_rng_counter_field_ptr =
      meta_data_ptr->get_field<unsigned>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");

  // Initialize the spheres position
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity node_i = requested_entities[i];
    stk::mesh::Entity sphere_i = requested_entities[num_nodes_local + i];

    // Get the GID for this sphere
    const stk::mesh::EntityId sphere_node_gid = bulk_data_ptr->identifier(node_i);

    // Set the coordinates, should be a 5x2 grid
    double *node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node_i);
    node_coords[0] = (i % 5);
    node_coords[1] = (i / 5);
    node_coords[2] = 0.0;

    // Set the RNG to the GID of the object
    unsigned *node_rng_counter = stk::mesh::field_data(*node_rng_counter_field_ptr, node_i);
    node_rng_counter[0] = static_cast<unsigned>(sphere_node_gid);

    // Set the radius
    stk::mesh::field_data(*element_radius_field_ptr, sphere_i)[0] = 1.2;
  }

  // ComputeAABB on everything
  compute_aabb_ptr->execute(*spheres_part_ptr);

  // Setup the output for RESULTS
  io_broker_ptr->setup_io_broker();

  // Write the initial config
  io_broker_ptr->write_io_broker(0.0);

  // Close up everything
  io_broker_ptr->finalize_io_broker();

  // Note, this test doesn't do anything, but write out the restart filename
  // stk::log_with_time_and_memory(MPI_COMM_WORLD, "Final mesh state for Part1 of restart duology.");
  // stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr, std::cout);
}

// Test if we can read back in the written mesh
TEST(IOBroker, WriteReadRestartAABBIntegerPart2) {
  std::string restart_filename = "exodus_mesh_restart.exo";
  std::string results_filename = "exodus_mesh_restart_results.exo";

  // Attempt to get the mesh requirements using the default parameters of ComputeAABB
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Set up a ComputeAABB function
  Teuchos::ParameterList fixed_params_sphere;
  fixed_params_sphere.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(fixed_params_sphere));

  // Add the TRANSIENT node coordinate field to the requirements so that we have it later
  mesh_reqs_ptr->add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);

  // Directly set an unsigned field on everybody
  mesh_reqs_ptr->add_and_sync_field_reqs(
      std::make_shared<mundy::meta::FieldReqs<unsigned>>("NODE_RNG_COUNTER", stk::topology::NODE_RANK, 1, 1));

  // Get fixed parameters for the IOBroker
  Teuchos::ParameterList fixed_params_iobroker;

  // Set the IO Parts
  std::vector<std::string> default_io_part_names{"SPHERES", "SPHEROCYLINDERS"};
  Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
  fixed_params_iobroker.set("enabled_io_parts", default_array_of_io_part_names, "PARTS with enabled IO.");

  // Set the IO fields
  // ELEMENT_RANK
  std::vector<std::string> default_io_field_element_names{"ELEMENT_AABB", "ELEMENT_RADIUS"};
  Teuchos::Array<std::string> default_array_of_io_field_element_names(default_io_field_element_names);
  fixed_params_iobroker.set("enabled_io_fields_element_rank", default_array_of_io_field_element_names,
                            "ELEMENT_RANK fields with enabled IO.");
  // NODE_RANK
  std::vector<std::string> default_io_field_node_names{"NODE_RNG_COUNTER"};
  Teuchos::Array<std::string> default_array_of_io_field_node_names(default_io_field_node_names);
  fixed_params_iobroker.set("enabled_io_fields_node_rank", default_array_of_io_field_node_names,
                            "NODE_RANK fields with enabled IO.");

  // Set custom values for the ComputeAABB methods to work with IO
  fixed_params_iobroker.set("coordinate_field_name", "NODE_COORDS");
  fixed_params_iobroker.set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES");

  // Set the output filename and file type
  fixed_params_iobroker.set("exodus_database_output_filename", results_filename);
  fixed_params_iobroker.set("exodus_database_input_filename", restart_filename);
  fixed_params_iobroker.set("enable_restart", "true");
  fixed_params_iobroker.set("parallel_io_mode", "hdf5");
  fixed_params_iobroker.set("database_purpose", "results");

  // Validate and set
  fixed_params_iobroker.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();

  // Set the simple fields
  bulk_data_ptr->mesh_meta_data_ptr()->use_simple_fields();

  auto io_broker_ptr = IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);
  auto compute_aabb_ptr = mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params_sphere);

  // Hit the button on committing the metadata
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  // Print the io broker
  io_broker_ptr->print_io_broker();

  // Get the sphere part
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_REQUIRE(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");

  // Fetch the required fields to set (note that we are only going to write out some of them)
  stk::mesh::Field<double> *node_coordinates_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");

  // Get the local number of entitites of all ranks (going to use to loop over and check values)
  // std::vector<size_t> entity_counts;
  stk::mesh::Selector locally_owned = meta_data_ptr->locally_owned_part();
  // stk::mesh::count_entities(locally_owned, *bulk_data_ptr, entity_counts);
  std::vector<stk::mesh::Entity> local_node_entities;
  stk::mesh::get_entities(*bulk_data_ptr, stk::topology::NODE_RANK, locally_owned, local_node_entities);
  for (size_t i = 0; i < local_node_entities.size(); ++i) {
    stk::mesh::Entity node_i = local_node_entities[i];

    // Check the coordinates and if they were read properly
    double *node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node_i);
    EXPECT_DOUBLE_EQ(node_coords[0], (static_cast<double>(i % 5)));
    EXPECT_DOUBLE_EQ(node_coords[1], (static_cast<double>(i / 5)));
    EXPECT_DOUBLE_EQ(node_coords[2], 0.0);
  }
}

//@}

}  // namespace

}  // namespace io

}  // namespace mundy
