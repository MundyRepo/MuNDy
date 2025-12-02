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
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp>  // for mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>    // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace constraints {

namespace {

//! \name DeclareAndInitConstraints functionality unit tests
//@{

void declare_linear_and_angular_springs_and_dump_mesh(const Teuchos::ParameterList &fixed_params,
                                                      const Teuchos::ParameterList &mutable_params,
                                                      const std::string &output_file_name,
                                                      const bool generate_hookean_springs = true,
                                                      const bool generate_angular_springs = true,
                                                      const bool generate_spheres_at_nodes = true,
                                                      const bool generate_spherocylinder_segments_along_edges = true) {
  std::cout << "#######################################################" << std::endl;
  std::cout << "Generating " << output_file_name << "..." << std::endl;
  std::cout << "#######################################################" << std::endl;

  // Create an instance of DeclareAndInitConstraints based on committed mesh that meets the requirements for
  // DeclareAndInitConstraints.
  auto [declare_and_init_constraints_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<DeclareAndInitConstraints>(
          {fixed_params});
  ASSERT_TRUE(declare_and_init_constraints_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");

  // Fetch the parts
  if (generate_hookean_springs) {
    stk::mesh::Part *linear_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::HookeanSprings::get_name());
    ASSERT_TRUE(linear_spring_part_ptr != nullptr);
    stk::io::put_io_part_attribute(*linear_spring_part_ptr);
  }
  if (generate_angular_springs) {
    stk::mesh::Part *angular_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::AngularSprings::get_name());
    ASSERT_TRUE(angular_spring_part_ptr != nullptr);
    stk::io::put_io_part_attribute(*angular_spring_part_ptr);
  }
  if (generate_spheres_at_nodes) {
    stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spheres::get_name());
    ASSERT_TRUE(sphere_part_ptr != nullptr);
    stk::io::put_io_part_attribute(*sphere_part_ptr);
  }
  if (generate_spherocylinder_segments_along_edges) {
    stk::mesh::Part *spherocylinder_segment_part_ptr =
        meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());
    ASSERT_TRUE(spherocylinder_segment_part_ptr != nullptr);
    stk::io::put_io_part_attribute(*spherocylinder_segment_part_ptr);
  }

  // Execute and then dumping the mesh to file given a set of mutable params.
  declare_and_init_constraints_ptr->set_mutable_params(mutable_params);
  declare_and_init_constraints_ptr->execute();

  // setup stk io
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh(output_file_name, stk::io::WRITE_RESULTS);

  // Fetch the required fields and add them as output fields
  auto validate_field_ptr = [](const stk::mesh::FieldBase *const field_ptr, const std::string &field_name) {
    ASSERT_TRUE(field_ptr != nullptr) << "Expected a field with name '" << field_name << "' but field does not exist.";
  };

  const std::string node_coord_field_name = mundy::constraints::HookeanSprings::get_node_coord_field_name();
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  validate_field_ptr(node_coord_field_ptr, node_coord_field_name);
  stk_io_broker.add_field(output_file_index, *node_coord_field_ptr);

  if (generate_hookean_springs) {
    const std::string element_hookean_spring_constant_field_name =
        mundy::constraints::HookeanSprings::get_element_spring_constant_field_name();
    const std::string element_hookean_spring_rest_length_field_name =
        mundy::constraints::HookeanSprings::get_element_rest_length_field_name();
    stk::mesh::Field<double> *element_hookean_spring_constant_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_hookean_spring_constant_field_name);
    stk::mesh::Field<double> *element_hookean_spring_rest_length_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_hookean_spring_rest_length_field_name);
    validate_field_ptr(element_hookean_spring_constant_field_ptr, element_hookean_spring_constant_field_name);
    validate_field_ptr(element_hookean_spring_rest_length_field_ptr, element_hookean_spring_rest_length_field_name);
    stk_io_broker.add_field(output_file_index, *element_hookean_spring_constant_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_hookean_spring_rest_length_field_ptr);
  }

  if (generate_angular_springs) {
    const std::string element_angular_spring_constant_field_name =
        mundy::constraints::AngularSprings::get_element_spring_constant_field_name();
    const std::string element_angular_spring_rest_angle_field_name =
        mundy::constraints::AngularSprings::get_element_rest_angle_field_name();
    stk::mesh::Field<double> *element_angular_spring_constant_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_angular_spring_constant_field_name);
    stk::mesh::Field<double> *element_angular_spring_rest_angle_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_angular_spring_rest_angle_field_name);
    validate_field_ptr(element_angular_spring_constant_field_ptr, element_angular_spring_constant_field_name);
    validate_field_ptr(element_angular_spring_rest_angle_field_ptr, element_angular_spring_rest_angle_field_name);
    stk_io_broker.add_field(output_file_index, *element_angular_spring_constant_field_ptr);
    stk_io_broker.add_field(output_file_index, *element_angular_spring_rest_angle_field_ptr);
  }

  if (generate_spheres_at_nodes) {
    const std::string element_sphere_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
    stk::mesh::Field<double> *element_sphere_radius_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_sphere_radius_field_name);
    validate_field_ptr(element_sphere_radius_field_ptr, element_sphere_radius_field_name);
    stk_io_broker.add_field(output_file_index, *element_sphere_radius_field_ptr);
  }

  if (generate_spherocylinder_segments_along_edges) {
    const std::string element_spherocylinder_segment_radius_field_name =
        mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
    stk::mesh::Field<double> *element_spherocylinder_segment_radius_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, element_spherocylinder_segment_radius_field_name);
    validate_field_ptr(element_spherocylinder_segment_radius_field_ptr,
                       element_spherocylinder_segment_radius_field_name);
    stk_io_broker.add_field(output_file_index, *element_spherocylinder_segment_radius_field_ptr);
  }

  // Write the mesh to file.
  stk_io_broker.begin_output_step(output_file_index, 0.0);
  stk_io_broker.write_defined_output_fields(output_file_index);
  stk_io_broker.end_output_step(output_file_index);
}

TEST(DeclareAndInitConstraints, ChainOfLinearAndAngularSpringsVisualInspection) {
  /* Check that DeclareAndInitConstraints works correctly for angular and linear springs. It's difficult to check the
   * correctness of the spring placement algorithm, so we will visually inspect the output.
   */
  // Setup mutable params for a default chain of springs
  Teuchos::ParameterList mutable_params_for_default_chain_of_springs;
  mutable_params_for_default_chain_of_springs.sublist("CHAIN_OF_SPRINGS").set<size_t>("num_nodes", 13);

  // Setup mutable params for a chain of linear springs with a helix coordinate mapping
  Teuchos::ParameterList mutable_params_for_helix_chain_of_springs;
  using CoordinateMappingType =
      mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
  using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::Helix;
  auto levis_function_mapping_ptr =
      std::make_shared<OurCoordinateMappingType>(100, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  mutable_params_for_helix_chain_of_springs.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", 100)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);

  // Dump the mesh for each combination of fixed and mutable params
  std::vector<std::pair<std::string, Teuchos::ParameterList>> named_mutable_params_list = {
      std::make_pair("n12_default_chain", mutable_params_for_default_chain_of_springs),
      std::make_pair("n100_helix_chain", mutable_params_for_helix_chain_of_springs)};
  for (int i = 0; i < 2; ++i) {
    const bool generate_hookean_springs = static_cast<bool>(i);
    for (int j = 0; j < 2; ++j) {
      const bool generate_angular_springs = static_cast<bool>(j);
      for (int k = 0; k < 2; ++k) {
        const bool generate_spheres_at_nodes = static_cast<bool>(k);
        for (int l = 0; l < 2; ++l) {
          const bool generate_spherocylinder_segments_along_edges = static_cast<bool>(l);
          if (!generate_hookean_springs && !generate_angular_springs && !generate_spheres_at_nodes &&
              !generate_spherocylinder_segments_along_edges) {
            // Skip the case where objects are not generated
            continue;
          }

          for (const auto &[name, mutable_params] : named_mutable_params_list) {
            Teuchos::ParameterList fixed_params;
            fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
                .sublist("CHAIN_OF_SPRINGS")
                .set<bool>("generate_hookean_springs", generate_hookean_springs)
                .set<bool>("generate_angular_springs", generate_angular_springs)
                .set<bool>("generate_spheres_at_nodes", generate_spheres_at_nodes)
                .set<bool>("generate_spherocylinder_segments_along_edges",
                           generate_spherocylinder_segments_along_edges);
            std::string output_file_name = name + "_h" + std::to_string(generate_hookean_springs) + "_a" +
                                           std::to_string(generate_angular_springs) + "_s" +
                                           std::to_string(generate_spheres_at_nodes) + "_c" +
                                           std::to_string(generate_spherocylinder_segments_along_edges) + ".exo";
            declare_linear_and_angular_springs_and_dump_mesh(
                fixed_params, mutable_params, output_file_name, generate_hookean_springs, generate_angular_springs,
                generate_spheres_at_nodes, generate_spherocylinder_segments_along_edges);
          }
        }
      }
    }
  }
}

//@}

}  // namespace

}  // namespace constraints

}  // namespace mundy
