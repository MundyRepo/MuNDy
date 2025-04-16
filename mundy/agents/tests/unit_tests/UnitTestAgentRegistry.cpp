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

// Mundy libs
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents

// Mundy test libs
#include "utils/ExampleAgent.hpp"  // for mundy::agents::ExampleAgent

namespace mundy {

namespace agents {

namespace {

//! \name Registration tests
//@{

struct DummyRegistrationIdentifier {};  // Dummy registration identifier;

TEST(HierarchyOfAgents, AutoRegistration) {
  // Test that the MUNDY_REGISTER_METACLASS macro performed the registration with the HierarchyOfAgents
  HierarchyOfAgents::register_new_class<mundy::agents::utils::ExampleAgent<1>>();
  HierarchyOfAgents::register_new_class<mundy::agents::utils::ExampleAgent<2>>();

  EXPECT_GT(HierarchyOfAgents::get_number_of_registered_types(), 0);
  EXPECT_TRUE(HierarchyOfAgents::is_valid(mundy::agents::utils::ExampleAgent<1>::get_name()));
  EXPECT_TRUE(HierarchyOfAgents::is_valid(mundy::agents::utils::ExampleAgent<2>::get_name()));
  HierarchyOfAgents::print_hierarchy();
}
//@}

}  // namespace

}  // namespace agents

}  // namespace mundy
