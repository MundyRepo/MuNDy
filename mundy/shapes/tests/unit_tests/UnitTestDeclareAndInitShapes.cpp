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
#include <stk_io/StkMeshIoBroker.hpp>      // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>             // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>                // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>               // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>           // for mundy::meta::FieldReqsBase
#include <mundy_shapes/DeclareAndInitShapes.hpp>  // for mundy::shapes::DeclareAndInitShapes
#include <mundy_shapes/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp>  // for mundy::shapes::declare_and_initialize_shapes::techniques::GridCoordinateMapping

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace shapes {

namespace {

//! \name DeclareAndInitShapes functionality unit tests
//@{

TEST(DeclareAndInitShapes, GridOfSpheresVisualInspection) {
  /* Check that DeclareAndInitShapes works correctly for spheres. It's difficult to check the correctness of the
   * sphere placement algorithm, so we will visually inspect the output.
   */

  // Create an instance of DeclareAndInitShapes based on committed mesh that meets the requirements for
  // DeclareAndInitShapes.
  std::cout << "Creating DeclareAndInitShapes instance and mesh." << std::endl;

  Teuchos::ParameterList declare_and_init_shapes_fixed_params;
  declare_and_init_shapes_fixed_params.set("enabled_technique_name", "GRID_OF_SPHERES");

  auto [declare_and_init_shapes_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<DeclareAndInitShapes>(
          {declare_and_init_shapes_fixed_params});
  ASSERT_TRUE(declare_and_init_shapes_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");

  std::cout << "Successfully created DeclareAndInitShapes instance and mesh." << std::endl;

  // Fetch the sphere part.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERES");
  ASSERT_TRUE(sphere_part_ptr != nullptr);
  stk::io::put_io_part_attribute(*sphere_part_ptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  ASSERT_TRUE(element_radius_field_ptr != nullptr);

  // Setup a helper function for executing and then dumping the mesh to file given a set of mutable params.
  auto set_mutable_params_execute_and_dump_mesh_to_file =
      [&](const std::string &file_name, const Teuchos::ParameterList &declare_and_init_shapes_mutable_params) {
        declare_and_init_shapes_ptr->set_mutable_params(declare_and_init_shapes_mutable_params);
        declare_and_init_shapes_ptr->execute();

        // setup stk io
        stk::io::StkMeshIoBroker stk_io_broker;
        stk_io_broker.use_simple_fields();
        stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

        size_t output_file_index = stk_io_broker.create_output_mesh(file_name, stk::io::WRITE_RESULTS);
        stk_io_broker.add_field(output_file_index, *node_coord_field_ptr);
        stk_io_broker.add_field(output_file_index, *element_radius_field_ptr);

        // Declare and initialize the shapes.
        declare_and_init_shapes_ptr->execute();

        // Write the mesh to file.
        stk_io_broker.begin_output_step(output_file_index, 0.0);
        stk_io_broker.write_defined_output_fields(output_file_index);
        stk_io_broker.end_output_step(output_file_index);
      };  // set_mutable_params_execute_and_dump_mesh_to_file

  // Setup our mutable params
  Teuchos::ParameterList declare_and_init_shapes_mutable_params;
  declare_and_init_shapes_mutable_params.sublist("GRID_OF_SPHERES")
      .set<size_t>("num_spheres_x", 4)
      .set<size_t>("num_spheres_y", 5)
      .set<size_t>("num_spheres_z", 6)
      .set("sphere_radius_lower_bound", 0.1)
      .set("sphere_radius_upper_bound", 0.1)
      .set("zmorton", false)
      .set("shuffle", false);
  set_mutable_params_execute_and_dump_mesh_to_file("test_grid_of_spheres_4x5x6_r0.1.exo",
                                                   declare_and_init_shapes_mutable_params);

  // Delete the spheres and rerun with zmorton sorting.
  bulk_data_ptr->modification_begin();
  bulk_data_ptr->destroy_elements_of_topology(stk::topology::PARTICLE);
  bulk_data_ptr->modification_end();
  declare_and_init_shapes_mutable_params.sublist("GRID_OF_SPHERES").set("zmorton", true);
  set_mutable_params_execute_and_dump_mesh_to_file("test_grid_of_spheres_4x5x6_r0.1_zmorton.exo",
                                                   declare_and_init_shapes_mutable_params);

  // Delete the spheres and rerun with shuffling.
  bulk_data_ptr->modification_begin();
  bulk_data_ptr->destroy_elements_of_topology(stk::topology::PARTICLE);
  bulk_data_ptr->modification_end();
  declare_and_init_shapes_mutable_params.sublist("GRID_OF_SPHERES").set("zmorton", false).set("shuffle", true);
  set_mutable_params_execute_and_dump_mesh_to_file("test_grid_of_spheres_4x5x6_r0.1_shuffle.exo",
                                                   declare_and_init_shapes_mutable_params);

  // Delete the spheres and rerun with zmorton sorting and a Levi's function coordinate mapping
  bulk_data_ptr->modification_begin();
  bulk_data_ptr->destroy_elements_of_topology(stk::topology::PARTICLE);
  bulk_data_ptr->modification_end();
  using CoordinateMappingType = mundy::shapes::declare_and_initialize_shapes::techniques::GridCoordinateMapping;
  using OurCoordinateMappingType = mundy::shapes::declare_and_initialize_shapes::techniques::LevisFunction2DTo3D;
  auto levis_function_mapping_ptr = std::make_shared<OurCoordinateMappingType>(40, 50);
  declare_and_init_shapes_mutable_params.sublist("GRID_OF_SPHERES")
      .set<size_t>("num_spheres_x", 40)
      .set<size_t>("num_spheres_y", 50)
      .set<size_t>("num_spheres_z", 1)
      .set("zmorton", true)
      .set("shuffle", false)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);
  set_mutable_params_execute_and_dump_mesh_to_file("test_grid_of_spheres_40x50_r0.1_zmorton_levis_function.exo",
                                                   declare_and_init_shapes_mutable_params);
}
//@}

}  // namespace

}  // namespace shapes

}  // namespace mundy
