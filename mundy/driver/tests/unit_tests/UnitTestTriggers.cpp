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
#include <mundy_driver/PeriodicTrigger.hpp>
#include <mundy_driver/TriggerBase.hpp>

namespace mundy {

namespace driver {

namespace {

/* What tests should we run?

The Configurator is responsible for pushing the information from the configuration file into the driver, including the
configuration of the MetaMethods and the ordering of operations.

*/

//! \name Configurator unit tests
//@{

TEST(Triggers, PeriodicTrigger) {
  // Set up a convenient using alias
  using PeriodicTrigger = mundy::driver::PeriodicTrigger;

  // Build a parameter list for the trigger
  Teuchos::ParameterList master_list;
  master_list.set("n_periodic", 1);
  // Create the trigger object
  std::shared_ptr<PeriodicTrigger> periodic_trigger_1 = std::make_shared<PeriodicTrigger>(master_list);

  // Check the trigger on steps 1,9
  EXPECT_EQ(periodic_trigger_1->trigger_check(1), mundy::driver::TRIGGERSTATUS::FIRED);
  EXPECT_EQ(periodic_trigger_1->trigger_check(9), mundy::driver::TRIGGERSTATUS::FIRED);

  // Try a trigger with a different periodicity
  master_list.set("n_periodic", 5);
  std::shared_ptr<PeriodicTrigger> periodic_trigger_5 = std::make_shared<PeriodicTrigger>(master_list);

  EXPECT_EQ(periodic_trigger_5->trigger_check(1), mundy::driver::TRIGGERSTATUS::SKIP);
  EXPECT_EQ(periodic_trigger_5->trigger_check(5), mundy::driver::TRIGGERSTATUS::FIRED);
  EXPECT_EQ(periodic_trigger_5->trigger_check(9), mundy::driver::TRIGGERSTATUS::SKIP);
  EXPECT_EQ(periodic_trigger_5->trigger_check(10), mundy::driver::TRIGGERSTATUS::FIRED);
}

//@}

}  // namespace

}  // namespace driver

}  // namespace mundy
