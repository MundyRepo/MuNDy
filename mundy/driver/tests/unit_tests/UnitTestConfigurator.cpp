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
#include <gmock/gmock.h>  // for EXPECT_THAT
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <string>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <Teuchos_YamlParser_decl.hpp>

// Mundy libs
#include <mundy_driver/Configurator.hpp>

namespace mundy {

namespace driver {

namespace {

/* What tests should we run?

The Configurator is responsible for pushing the information from the configuration file into the driver, including the
configuration of the MetaMethods and the ordering of operations.

*/

//! \name Configurator unit tests
//@{

TEST(Configurator, Constructor) {
  // Construct a configurator from the YAML file
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  EXPECT_NO_THROW({ Configurator configurator(comm); });
}

TEST(Configurator, ParseParametersYAML) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./unit_test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator(MPI_COMM_WORLD);

  // Set things that need to be setup before calling the main parse command
  std::vector<std::string> expected_rank_names = {"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"};
  configurator.set_entity_rank_names(expected_rank_names);
  configurator.set_input_file(yaml_file, "yaml");

  // Try to fire off a parse
  configurator.parse_parameters();

  // Test if we can see inside the configurator
  std::stringstream buffer;
  buffer << configurator;
  EXPECT_THAT(buffer.str(), ::testing::ContainsRegex("GENERATE_NEIGHBOR_LINKERS"));
  EXPECT_THAT(buffer.str(), ::testing::ContainsRegex("buffer_distance = 0"));
  std::cout << "DEBUG YAML parameters\n";
  std::cout << configurator;
}

TEST(Configurator, CreateMeshReqs) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./unit_test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator(MPI_COMM_WORLD);

  // Set things that need to be setup before calling the main parse command
  std::vector<std::string> expected_rank_names = {"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"};
  configurator.set_entity_rank_names(expected_rank_names);
  configurator.set_input_file(yaml_file, "yaml");

  // Parse the parameters
  configurator.parse_parameters();

  // Create mesh requirements from the parameters
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs;
  EXPECT_NO_THROW({ mesh_reqs = configurator.create_mesh_requirements(); });

  // Print this out to make sure it's good (can check logs for this)
  mesh_reqs->print(std::cout, 0);

  // Check that we have a fully valid set of requirements
  ASSERT_TRUE(mesh_reqs->is_fully_specified());
}

TEST(Configurator, ConstructDriver) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./unit_test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator(MPI_COMM_WORLD);

  // Set things that need to be setup before calling the main parse command
  std::vector<std::string> expected_rank_names = {"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"};
  configurator.set_entity_rank_names(expected_rank_names);
  configurator.set_input_file(yaml_file, "yaml");

  // Parse the configuration
  configurator.parse_parameters();

  // Generate mesh requirements
  configurator.create_mesh_requirements();

  // Construct a driver
  std::shared_ptr<Driver> driver_ptr;

  EXPECT_NO_THROW({ driver_ptr = configurator.generate_driver(); });
}

//@}

}  // namespace

}  // namespace driver

}  // namespace mundy
