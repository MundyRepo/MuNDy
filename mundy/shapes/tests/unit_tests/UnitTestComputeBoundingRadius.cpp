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
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>              // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>                 // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>                // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>            // for mundy::meta::FieldReqsBase
#include <mundy_shapes/ComputeBoundingRadius.hpp>  // for mundy::shapes::ComputeBoundingRadius

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace shapes {

namespace {

//! \name ComputeBoundingRadius functionality unit tests
//@{

TEST(ComputeBoundingRadius, PerformsBoundingRadiusCalculationCorrectlyForSphere) {
  /* Check that ComputeBoundingRadius works correctly for spheres.
  For a sphere at any arbitrary position, the bounding radius is just the sphere's radius.
  */

  // Create an instance of ComputeBoundingRadius based on committed mesh that meets the requirements for
  // ComputeBoundingRadius.
  auto [compute_bounding_radius_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<ComputeBoundingRadius>();
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

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
  stk::mesh::Field<double> *radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  ASSERT_TRUE(radius_field_ptr != nullptr);
  stk::mesh::Field<double> *bounding_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_BOUNDING_RADIUS");
  ASSERT_TRUE(bounding_radius_field_ptr != nullptr);

  // Set the sphere's radius.
  double sphere_radius = 1.0;
  double *radius = stk::mesh::field_data(*radius_field_ptr, sphere_element);
  radius[0] = sphere_radius;

  // Compute the bounding radius.
  compute_bounding_radius_ptr->execute(*sphere_part_ptr);

  // Check that the computed bounding radius is as expected.
  double *bounding_radius = stk::mesh::field_data(*bounding_radius_field_ptr, sphere_element);
  double expected_bounding_radius = sphere_radius;
  EXPECT_DOUBLE_EQ(bounding_radius[0], expected_bounding_radius);
}
//@}

}  // namespace

}  // namespace shapes

}  // namespace mundy
