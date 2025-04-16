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
#include <mundy_driver/Driver.hpp>

namespace mundy {

namespace driver {

namespace {

/* What tests should we run?

Driver is our main 'simulation' class, and as such, holds all of the simulation data. This includes the state of the
system.

*/

//! \name Driver unit tests
//@{

//@}

//! \name Driver exposure tests
//@{
TEST(Driver, ExposeRegisteredClasses) {
  // Create a dummy configurator, this will still trigger the registration procedure
  Driver driver;

  // Ask it to expose the registered MetaMethodExecutionInterface
  std::string registered_mmei = driver.get_registered_meta_method_execution_interface();
  // Ask it to expose the registered MetaMethodSubsetExecutionInterface
  std::string registered_mmsei = driver.get_registered_meta_method_subset_execution_interface();
  // Ask it to expose the registered MetaMethodPairwiseSubsetExecutionInterface
  std::string registered_mmpsei = driver.get_registered_meta_method_pairwise_subset_execution_interface();
  // Ask for all the registered classes
  std::string registered_classes = driver.get_registered_classes();

  EXPECT_THAT(registered_mmei, ::testing::ContainsRegex("DECLARE_AND_INIT_SHAPES"));
  EXPECT_THAT(registered_mmsei, ::testing::ContainsRegex("COMPUTE_OBB"));
  EXPECT_THAT(registered_mmpsei, ::testing::ContainsRegex("GENERATE_NEIGHBOR_LINKERS"));
  EXPECT_THAT(registered_classes, ::testing::ContainsRegex("COMPUTE_AABB"));
}

//@}

}  // namespace

}  // namespace driver

}  // namespace mundy
