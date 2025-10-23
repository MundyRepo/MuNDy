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
#include <mundy_driver/Driver.hpp>

namespace mundy {

namespace driver {

namespace {

/* What tests should we run?

This is an integration test for the configurator being able to push information into the driver.

*/

//! \name Combined configurator and driver integration tests
//@{
// TEST(ConfigureDriver, ConfigureBasicYAML) {
//   // This should be in the local test directory if things were set up correctly
//   const std::string yaml_file = "./integration_test_configuredriver_basic.yaml";

//   // Construct a Driver with the correct communicator
//   std::shared_ptr<Driver> driver_ptr = std::make_shared<Driver>(MPI_COMM_WORLD);

//   // Construct a configurator from the YAML file
//   Configurator configurator("yaml", yaml_file);

//   // Associate the Driver with the Configurator
//   configurator.set_driver(driver_ptr);

//   // Run the parse command
//   configurator.parse_parameters();

//   // Print the Configurator to see what's inside
//   configurator.print_enabled_meta_methods();

//   // Ask the configurator to set up the driver for us
//   configurator.generate_driver();
// }

//@}

}  // namespace

}  // namespace driver

}  // namespace mundy
