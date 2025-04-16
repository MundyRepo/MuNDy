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
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_mesh/BulkData.hpp>             // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>          // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>             // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>            // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>        // for mundy::meta::FieldReqsBase

// Mundy test libs
#include "utils/ExampleAgent.hpp"  // for mundy::agents::utils::ExampleAgent

namespace mundy {

namespace agents {

namespace {

//! \name HierarchyOfAgents object registration tests
//@{

TEST(HierarchyOfAgentsRegistration, RegistrationWorksProperly) {
  // Registration of a class with HierarchyOfAgents should allow access to the class's internal static methods.
  // This test checks that the registration was successful.
  HierarchyOfAgents::register_new_class<mundy::agents::utils::ExampleAgent<3>>();
}
//@}

}  // namespace

}  // namespace agents

}  // namespace mundy
