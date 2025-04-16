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
#include <mundy_meta/MetaFactory.hpp>    // for mundy::meta::MetaFactory
#include <mundy_meta/MetaRegistry.hpp>   // for mundy::meta::MetaRegistry

// Mundy test libs
#include "utils/ExampleMetaMethod.hpp"  // for mundy::meta::utils::ExampleMetaMethod

namespace mundy {

namespace meta {

namespace {

//! \name MetaRegistry automatic registration tests
//@{

TEST(MetaRegistry, AutoRegistration) {
  // Test that the MUNDY_REGISTER_METACLASS macro performed the registration with the MetaMethodFactory with a given
  // RegistrationIdentifier
  using OurTestMetaMethodFactory =
      mundy::meta::MetaFactory<mundy::meta::utils::DefaultExampleMetaMethod::PolymorphicBaseType,
                               mundy::meta::make_registration_string("TEST_FACTORY")>;
  EXPECT_EQ(OurTestMetaMethodFactory::num_registered_classes(), 2);
  EXPECT_TRUE(OurTestMetaMethodFactory::is_valid_key("KEY1"));
  EXPECT_TRUE(OurTestMetaMethodFactory::is_valid_key("KEY2"));
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy

// Register a class with the MetaMethodFactory with a given RegistrationIdentifier.
MUNDY_REGISTER_METACLASS("KEY1", mundy::meta::utils::ExampleMetaMethod<mundy::meta::make_registration_string("CLASS1")>,
                         mundy::meta::MetaFactory<mundy::meta::utils::DefaultExampleMetaMethod::PolymorphicBaseType,
                                                  mundy::meta::make_registration_string("TEST_FACTORY")>)

// Register a different class with the same MetaMethodFactory with a given RegistrationIdentifier.
MUNDY_REGISTER_METACLASS("KEY2", mundy::meta::utils::ExampleMetaMethod<mundy::meta::make_registration_string("CLASS2")>,
                         mundy::meta::MetaFactory<mundy::meta::utils::DefaultExampleMetaMethod::PolymorphicBaseType,
                                                  mundy::meta::make_registration_string("TEST_FACTORY")>)
