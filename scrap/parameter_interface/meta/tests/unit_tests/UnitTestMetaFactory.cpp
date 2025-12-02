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

// Mundy test libs
#include "utils/ExampleMetaMethod.hpp"  // for mundy::meta::utils::ExampleMetaMethod

namespace mundy {

namespace meta {

namespace {

//! \name MetaFactory object registration tests
//@{

// To avoid contaminating other tests and mundy itself, we'll use a unique registration identifier for all test
// factories and reset the factory before and after each test.
TEST(MetaFactoryRegistration, RegistrationWorksProperly) {
  // Registration of a class derived from \c HasMeshReqsAndIsRegisterable with \c MetaFactory should store the
  // class's identifier, instance generator, requirements generator, fixed parameters validator, and mutable parameters
  // validator.

  // Create our example class to register.
  // Reset the test counter within our classes to register.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  using ClassToRegister = utils::DefaultExampleMetaMethod;
  using PolymorphicBaseType = ClassToRegister::PolymorphicBaseType;
  ClassToRegister::reset_counters();

  // Create our example meta factory.
  using ExampleMetaFactory = MetaFactory<PolymorphicBaseType, mundy::meta::make_registration_string("TEST_FACTORY")>;
  ExampleMetaFactory::reset();

  // Perform the registration.
  std::string class_registration_string = "EXAMPLE_REG_KEY";
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 0);
  EXPECT_FALSE(ExampleMetaFactory::is_valid_key(class_registration_string));
  ExampleMetaFactory::register_new_class<ClassToRegister>(class_registration_string);
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 1);
  EXPECT_TRUE(ExampleMetaFactory::is_valid_key(class_registration_string));

  // Try to use the factory to access the internal member functions of our registered class.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  mundy::mesh::BulkData* bulk_data_ptr = nullptr;
  Teuchos::ParameterList fixed_params;
  Teuchos::ParameterList mutable_params;

  EXPECT_EQ(ClassToRegister::num_get_mesh_requirements_calls(), 0);
  ExampleMetaFactory::get_mesh_requirements(class_registration_string, fixed_params);
  EXPECT_EQ(ClassToRegister::num_get_mesh_requirements_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_get_valid_fixed_params_calls(), 0);
  ExampleMetaFactory::get_valid_fixed_params(class_registration_string);
  EXPECT_EQ(ClassToRegister::num_get_valid_fixed_params_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_get_valid_mutable_params_calls(), 0);
  ExampleMetaFactory::get_valid_mutable_params(class_registration_string);
  EXPECT_EQ(ClassToRegister::num_get_valid_mutable_params_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_create_new_instance_calls(), 0);
  ExampleMetaFactory::create_new_instance(class_registration_string, bulk_data_ptr, fixed_params);
  EXPECT_EQ(ClassToRegister::num_create_new_instance_calls(), 1);

  ExampleMetaFactory::reset();
}

TEST(MetaFactoryRegistration, Reregistration) {
  // Ensure that attempting to register a class with a key that already exists should throw an exception.

  // Create our example classes to register.
  // Reset the test counter within our classes to register.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  using ClassToRegister1 = utils::ExampleMetaMethod<mundy::meta::make_registration_string("CLASS1")>;
  using ClassToRegister2 = utils::ExampleMetaMethod<mundy::meta::make_registration_string("CLASS2")>;
  using PolymorphicBaseType = ClassToRegister1::PolymorphicBaseType;
  bool classes_are_different = !std::is_same_v<ClassToRegister1, ClassToRegister2>;
  ASSERT_TRUE(classes_are_different);
  ClassToRegister1::reset_counters();
  ClassToRegister2::reset_counters();

  // Create our example meta factory.
  using ExampleMetaFactory = MetaFactory<PolymorphicBaseType, mundy::meta::make_registration_string("TEST_FACTORY")>;
  ExampleMetaFactory::reset();

  // Register the first class.
  const std::string shared_class_registration_string = "A_SHARED_KEY";
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 0);
  ExampleMetaFactory::register_new_class<ClassToRegister1>(shared_class_registration_string);
  ASSERT_EQ(ExampleMetaFactory::num_registered_classes(), 1);

  // Attempting to register a class with a key that already exists should throw an exception
  EXPECT_THROW(ExampleMetaFactory::register_new_class<ClassToRegister1>(shared_class_registration_string),
               std::logic_error);
  EXPECT_THROW(ExampleMetaFactory::register_new_class<ClassToRegister2>(shared_class_registration_string),
               std::logic_error);

  ExampleMetaFactory::reset();
}

TEST(MetaFactoryRegistration, RegistrationWithDifferentRegistrationIdentifier) {
  /* Ensure that MetaFactories with different registration identifiers do not share registered classes.
  The setup for this test is as follows:
    Register ClassToRegister with MetaFactory1 with registration identifier RegistrationIdentifier1.
    Register ClassToRegister with MetaFactory2 with registration identifier RegistrationIdentifier2.
  */

  // Create our example class to register.
  // Reset the test counter within our classes to register.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  using ClassToRegister = utils::DefaultExampleMetaMethod;
  using PolymorphicBaseType = ClassToRegister::PolymorphicBaseType;
  ClassToRegister::reset_counters();

  // Create our example meta factories.
  constexpr auto factory_registration_string_wrapper1 = mundy::meta::make_registration_string("TEST_FACTORY1");
  constexpr auto factory_registration_string_wrapper2 = mundy::meta::make_registration_string("TEST_FACTORY2");
  using ExampleMetaFactory1 = MetaFactory<PolymorphicBaseType, factory_registration_string_wrapper1>;
  using ExampleMetaFactory2 = MetaFactory<PolymorphicBaseType, factory_registration_string_wrapper2>;
  ExampleMetaFactory1::reset();

  // Register with the first factory.
  std::string class_registration_string = "EXAMPLE_REG_KEY";
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 0);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 0);
  EXPECT_FALSE(ExampleMetaFactory1::is_valid_key(class_registration_string));
  ExampleMetaFactory1::register_new_class<ClassToRegister>(class_registration_string);
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 1);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 0);

  // Register with the second factory.
  EXPECT_FALSE(ExampleMetaFactory2::is_valid_key(class_registration_string));
  ExampleMetaFactory2::register_new_class<ClassToRegister>(class_registration_string);
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 1);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 1);

  ExampleMetaFactory1::reset();
  ExampleMetaFactory2::reset();
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy
