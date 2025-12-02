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
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>    // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace shapes {

namespace {

//! \name ComputeAABB static interface implementations unit tests
//@{

TEST(ComputeAABBStaticInterface, IsRegisterable) {
  // Check if ComputeAABB has the correct static interface to be compatible with MetaFactory.
  ASSERT_TRUE(mundy::meta::HasMeshReqsAndIsRegisterable<ComputeAABB>::value);
}

TEST(ComputeAABBStaticInterface, FixedParameterDefaults) {
  // Check the expected default values.
  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(ComputeAABB::get_valid_fixed_params());

  // Check that all the enabled kernels are in the list of registered kernels.
  ASSERT_TRUE(fixed_params.isParameter("enabled_kernel_names"));
  Teuchos::Array<std::string> enabled_kernel_names =
      fixed_params.get<Teuchos::Array<std::string>>("enabled_kernel_names");
  ASSERT_EQ(enabled_kernel_names.size(), ComputeAABB::OurKernelFactory::num_registered_classes());
  for (const std::string &key : ComputeAABB::OurKernelFactory::get_keys()) {
    ASSERT_TRUE(std::find(enabled_kernel_names.begin(), enabled_kernel_names.end(), key) != enabled_kernel_names.end());
  }

  // Check that the fixed parameters for each kernel are present.
  for (const std::string &key : ComputeAABB::OurKernelFactory::get_keys()) {
    ASSERT_TRUE(fixed_params.isSublist(key));
    Teuchos::ParameterList &kernel_params = fixed_params.sublist(key, true);
    ASSERT_TRUE(kernel_params.isParameter("valid_entity_part_names"));
    ASSERT_TRUE(kernel_params.get<Teuchos::Array<std::string>>("valid_entity_part_names").size() > 0);
  }

  // TODO(palmerb4): Check that the parameters are forwarded correctly.
}

TEST(ComputeAABBStaticInterface, MutableParameterDefaults) {
  // Check the expected default values.
  Teuchos::ParameterList mutable_params;
  mutable_params.validateParametersAndSetDefaults(ComputeAABB::get_valid_mutable_params());

  // Check that the mutable parameters for each kernel are present.
  for (const std::string &key : ComputeAABB::OurKernelFactory::get_keys()) {
    ASSERT_TRUE(mutable_params.isSublist(key));
  }
}

TEST(ComputeAABBStaticInterface, FixedParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
}

TEST(ComputeAABBStaticInterface, MutableParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
}

TEST(ComputeAABBStaticInterface, GetMeshReqsFromDefaultParameters) {
  // Attempt to get the mesh requirements using the default parameters.
  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(ComputeAABB::get_valid_fixed_params());
  ASSERT_NO_THROW(ComputeAABB::get_mesh_requirements(fixed_params));
}

TEST(ComputeAABBStaticInterface, CreateNewInstanceFromDefaultParameters) {
  // Attempt to get the mesh requirements using the default parameters.
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(ComputeAABB::get_valid_fixed_params());
  mesh_reqs_ptr->sync(ComputeAABB::get_mesh_requirements(fixed_params));

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  EXPECT_NO_THROW(ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params));
}
//@}

//! \name ComputeAABB functionality unit tests
//@{

TEST(ComputeAABB, PerformsAABBCalculationCorrectlyForSphere) {
  /* Check that ComputeAABB works correctly for spheres.
  For a sphere at any arbitrary position, the AABB should be a cube with side length equal to the diameter of the sphere
  and center at the sphere's position.
  */

  // Create an instance of ComputeAABB based on committed mesh that meets the requirements for ComputeAABB.
  auto [compute_aabb_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<ComputeAABB>();
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  ComputeAABB::get_mesh_requirements(Teuchos::ParameterList())->print();

  // Fetch the multibody sphere part and add a single sphere to it.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERES");
  ASSERT_TRUE(sphere_part_ptr != nullptr);

  bulk_data_ptr->modification_begin();
  stk::mesh::EntityId sphere_id = 1;
  stk::mesh::Entity sphere_element =
      bulk_data_ptr->declare_element(sphere_id, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_node = bulk_data_ptr->declare_node(sphere_id);
  bulk_data_ptr->declare_relation(sphere_element, sphere_node, 0);
  bulk_data_ptr->modification_end();

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  stk::mesh::Field<double> *radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  ASSERT_TRUE(radius_field_ptr != nullptr);
  stk::mesh::Field<double> *aabb_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
  ASSERT_TRUE(aabb_field_ptr != nullptr);

  // Set the sphere's position.
  double sphere_position[3] = {0.0, 0.0, 0.0};
  double *node_coords = stk::mesh::field_data(*node_coord_field_ptr, sphere_node);
  node_coords[0] = sphere_position[0];
  node_coords[1] = sphere_position[1];
  node_coords[2] = sphere_position[2];

  // Set the sphere's radius.
  double sphere_radius = 1.0;
  double *radius = stk::mesh::field_data(*radius_field_ptr, sphere_element);
  radius[0] = sphere_radius;

  // Compute the AABB.
  compute_aabb_ptr->execute(*sphere_part_ptr);

  // Check that the computed aabb is as expected.
  double *aabb = stk::mesh::field_data(*aabb_field_ptr, sphere_element);
  double expected_aabb[6] = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
  for (int i = 0; i < 6; i++) {
    EXPECT_DOUBLE_EQ(aabb[i], expected_aabb[i]);
  }
}
//@}

}  // namespace

}  // namespace shapes

}  // namespace mundy
