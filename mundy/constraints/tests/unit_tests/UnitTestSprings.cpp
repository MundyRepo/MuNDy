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
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/MeshUtils.hpp>     // for stk::mesh::fixup_ghosted_to_shared_nodes
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_constraints/AngularSprings.hpp>            // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>  // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/FENESprings.hpp>               // for mundy::constraints::FENESprings
#include <mundy_constraints/FENEWCASprings.hpp>            // for mundy::constraints::FENEWCASprings
#include <mundy_constraints/HookeanSprings.hpp>            // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                  // for mundy::core::make_string_array
#include <mundy_mesh/BulkData.hpp>                         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>                       // for mundy::mesh::vector3_field_data
#include <mundy_mesh/MeshBuilder.hpp>                      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>                         // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>                        // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>                    // for mundy::meta::FieldReqsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace constraints {

namespace {

//! \name FENESprings functionality unit tests
//@{

TEST(FENESprings, FENESpringsKernel) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 2) {
    GTEST_SKIP() << "This test is designed for 1 or 2 ranks.";
  }
  // Create an instance of FENESprings based on committed mesh that meets the
  // default requirements for FENESprings.
  auto fene_springs_fixed_params =
      Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("FENE_SPRINGS"));
  fene_springs_fixed_params.sublist("FENE_SPRINGS")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"));

  auto [compute_constraint_forcing_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<ComputeConstraintForcing>(
          {fene_springs_fixed_params});
  ASSERT_TRUE(compute_constraint_forcing_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // This test is designed for either 1 or 2 ranks.
  const int rank = bulk_data_ptr->parallel_rank();

  // Fetch the parts.
  stk::mesh::Part *spherocylinder_segment_part_ptr =
      meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());
  stk::mesh::Part *fene_springs_part_ptr = meta_data_ptr->get_part(mundy::constraints::FENESprings::get_name());

  ASSERT_TRUE(spherocylinder_segment_part_ptr != nullptr);
  ASSERT_TRUE(fene_springs_part_ptr != nullptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<double> *node_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");
  stk::mesh::Field<double> *element_fene_spring_constant_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_SPRING_CONSTANT");
  stk::mesh::Field<double> *element_fene_spring_rmax_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_SPRING_R0");

  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(node_force_field_ptr != nullptr);
  ASSERT_TRUE(element_fene_spring_constant_field_ptr != nullptr);
  ASSERT_TRUE(element_fene_spring_rmax_field_ptr != nullptr);

  // Declare a single spherocylinder segment.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_shapes_to_rank0;

  // Declare the spherocylinder segment on rank 0.
  if (rank == 0) {
    stk::mesh::Entity seg1_element =
        bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->declare_node(1);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->declare_node(2);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node1, 0);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node2, 1);
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_SHAPES_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_shapes_to_rank0);
  bulk_data_ptr->modification_end();

  bulk_data_ptr->modification_begin();
  if (rank == 0) {
    // Fetch the declared entities and check them
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);

    for (auto &entity : {seg1_element}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(entity));
    }
  }

  // We're using one-sided linker creation, so we need to fixup the ghosted to shared nodes.
  stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
  bulk_data_ptr->modification_end();

  // Initialize the spherocylinder segments. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_element));
    stk::mesh::field_data(*element_fene_spring_constant_field_ptr, seg1_element)[0] = 3.0;
    stk::mesh::field_data(*element_fene_spring_rmax_field_ptr, seg1_element)[0] = 2.5;
  }

  // Initialize the node coordinates
  if (rank == 0) {
    // Fetch the linkers
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node1).set(0.0, 0.0, 0.0);  // Node 1
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node2).set(0.0, 0.0, 1.0);  // Node 2
  }

  // Execute the ComputeConstraintForcing for all FENE springs.
  ASSERT_NO_THROW(compute_constraint_forcing_ptr->execute(*fene_springs_part_ptr)) << "Failed to evaluate FENE spring.";

  auto check_potential_force_magnitude = [&node_force_field_ptr](stk::mesh::Entity sphere_node, const double &dr,
                                                                 const double &k, const double &rmax,
                                                                 const std::string &message) {
    // Check that the result is as expected.
    const auto potential_force = mundy::mesh::vector3_field_data(*node_force_field_ptr, sphere_node);
    const double potential_force_magnitude = mundy::math::norm(potential_force);

    // The expected potential force is computed using FENE bonds without a repulsive term.
    const double expected_potential_force_magnitude = k * dr / (1.0 - dr * dr / (rmax * rmax));
    std::cout << "FENE: Combined potential dr: " << dr << ", force: " << potential_force
              << ", force_magnitude: " << potential_force_magnitude
              << ", expected_force_magnitude: " << expected_potential_force_magnitude << std::endl;
    EXPECT_DOUBLE_EQ(potential_force_magnitude, expected_potential_force_magnitude) << message;
  };

  // Only rank 0 will check the results since it owns the springs.
  if (rank == 0) {
    check_potential_force_magnitude(bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1), 1.0, 3.0, 2.5,
                                    "Failed to evaluate FENE spring.");
    check_potential_force_magnitude(bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2), 1.0, 3.0, 2.5,
                                    "Failed to evaluate FENE spring.");
  }
}

//@}

//! \name FENEWCASprings functionality unit tests
//@{

TEST(FENEWCASprings, FENEWCASpringsKernel) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 2) {
    GTEST_SKIP() << "This test is designed for 1 or 2 ranks.";
  }
  // Create an instance of FENESprings based on committed mesh that meets the
  // default requirements for FENESprings.
  auto fenewca_springs_fixed_params =
      Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("FENEWCA_SPRINGS"));
  fenewca_springs_fixed_params.sublist("FENEWCA_SPRINGS")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"));

  auto [compute_constraint_forcing_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<ComputeConstraintForcing>(
          {fenewca_springs_fixed_params});
  ASSERT_TRUE(compute_constraint_forcing_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // This test is designed for either 1 or 2 ranks.
  const int rank = bulk_data_ptr->parallel_rank();

  // Fetch the parts.
  stk::mesh::Part *spherocylinder_segment_part_ptr =
      meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());
  stk::mesh::Part *fenewca_springs_part_ptr = meta_data_ptr->get_part(mundy::constraints::FENEWCASprings::get_name());

  ASSERT_TRUE(spherocylinder_segment_part_ptr != nullptr);
  ASSERT_TRUE(fenewca_springs_part_ptr != nullptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<double> *node_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");
  stk::mesh::Field<double> *element_fenewca_spring_constant_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_SPRING_CONSTANT");
  stk::mesh::Field<double> *element_fenewca_spring_rmax_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_SPRING_R0");

  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(node_force_field_ptr != nullptr);
  ASSERT_TRUE(element_fenewca_spring_constant_field_ptr != nullptr);
  ASSERT_TRUE(element_fenewca_spring_rmax_field_ptr != nullptr);

  // Declare a single spherocylinder segment.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_shapes_to_rank0;

  // Declare the spherocylinder segment on rank 0.
  if (rank == 0) {
    stk::mesh::Entity seg1_element =
        bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->declare_node(1);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->declare_node(2);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node1, 0);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node2, 1);
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_SHAPES_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_shapes_to_rank0);
  bulk_data_ptr->modification_end();

  bulk_data_ptr->modification_begin();
  if (rank == 0) {
    // Fetch the declared entities and check them
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);

    for (auto &entity : {seg1_element}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(entity));
    }
  }

  // We're using one-sided linker creation, so we need to fixup the ghosted to shared nodes.
  stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
  bulk_data_ptr->modification_end();

  // Initialize the spherocylinder segments. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_element));
    stk::mesh::field_data(*element_fenewca_spring_constant_field_ptr, seg1_element)[0] = 3.0;
    stk::mesh::field_data(*element_fenewca_spring_rmax_field_ptr, seg1_element)[0] = 2.5;
  }

  // Initialize the node coordinates
  if (rank == 0) {
    // Fetch the linkers
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node1).set(0.0, 0.0, 0.0);  // Node 1
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node2).set(0.0, 0.0, 1.0);  // Node 2
  }

  // Execute the ComputeConstraintForcing for all FENE springs.
  ASSERT_NO_THROW(compute_constraint_forcing_ptr->execute(*fenewca_springs_part_ptr))
      << "Failed to evaluate FENEWCA spring.";

  auto check_potential_force_magnitude =
      [&node_force_field_ptr](stk::mesh::Entity sphere_node, const double &dr, const double &k, const double &rmax,
                              const double &epsilon, const double &sigma, const std::string &message) {
        // Check that the result is as expected.
        const auto potential_force = mundy::mesh::vector3_field_data(*node_force_field_ptr, sphere_node);
        const double potential_force_magnitude = mundy::math::norm(potential_force);

        // The expected potential force is computed using FENE bonds without a repulsive term.
        const double expected_potential_force_magnitude_spring = k * dr / (1.0 - dr * dr / (rmax * rmax));
        double expected_potential_force_magnitude_wca = 0.0;
        if (dr < std::pow(2.0, 1.0 / 6.0) * sigma) {
          expected_potential_force_magnitude_wca =
              6.0 * 4.0 * epsilon * (2.0 * std::pow(sigma / dr, 12) - std::pow(sigma / dr, 6)) / dr;
        }
        // For FENE+WCA springs the magnitude is the opposite for the FENE springs and WCA contributions
        const double expected_potential_force_magnitude =
            -expected_potential_force_magnitude_spring + expected_potential_force_magnitude_wca;
        std::cout << "FENE: Combined potential dr: " << dr << ", force: " << potential_force
                  << ", force_magnitude: " << potential_force_magnitude
                  << ", expected_force_magnitude: " << expected_potential_force_magnitude << std::endl;
        EXPECT_DOUBLE_EQ(potential_force_magnitude, expected_potential_force_magnitude) << message;
      };

  // Only rank 0 will check the results since it owns the springs.
  if (rank == 0) {
    check_potential_force_magnitude(bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1), 1.0, 3.0, 2.5, 1.0, 1.0,
                                    "Failed to evaluate FENE spring.");
    check_potential_force_magnitude(bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2), 1.0, 3.0, 2.5, 1.0, 1.0,
                                    "Failed to evaluate FENE spring.");
  }
}

//@}

}  // namespace

}  // namespace constraints

}  // namespace mundy
