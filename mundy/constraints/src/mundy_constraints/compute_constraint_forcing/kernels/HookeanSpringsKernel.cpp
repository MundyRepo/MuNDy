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

/// \file HookeanSpringsKernel.cpp
/// \brief Definition of the ComputeCOnstraintForcing's HookeanSpringsKernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/FieldParallel.hpp>  // for stk::mesh::communicate_field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

// Mundy libs
#include <mundy_constraints/HookeanSprings.hpp>  // for mundy::constraints::HookeanSprings
#include <mundy_constraints/compute_constraint_forcing/kernels/HookeanSpringsKernel.hpp>  // for mundy::constraints::compute_constraint_forcing::kernels::HookeanSpringsKernel
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData

namespace mundy {

namespace constraints {

namespace compute_constraint_forcing {

namespace kernels {

// \name Constructors and destructor
//{

HookeanSpringsKernel::HookeanSpringsKernel(mundy::mesh::BulkData *const bulk_data_ptr,
                                           const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "HookeanSpringsKernel: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(HookeanSpringsKernel::get_valid_fixed_params());

  // Store the valid entity parts for the kernel.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  for (const std::string &part_name : valid_entity_part_names) {
    valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
    MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                        std::string("HookeanSpringsKernel: Part '") + part_name +
                            "' from the valid_entity_part_names does not exist in the meta data.");
  }

  // Fetch the fields.
  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
  const std::string node_coord_field_name = HookeanSprings::get_node_coord_field_name();
  const std::string element_rest_length_field_name = HookeanSprings::get_element_rest_length_field_name();
  const std::string element_spring_constant_field_name = HookeanSprings::get_element_spring_constant_field_name();

  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  node_coordinates_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_rest_length_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_rest_length_field_name);
  element_spring_constant_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_spring_constant_field_name);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> HookeanSpringsKernel::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void HookeanSpringsKernel::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(HookeanSpringsKernel::get_valid_mutable_params());
}
//}

// \name Actions
//{

void HookeanSpringsKernel::execute(const stk::mesh::Selector &spring_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coordinates_field_ptr_, element_rest_length_field_ptr_,
                                                      element_spring_constant_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coord_field = *node_coordinates_field_ptr_;
  const stk::mesh::Field<double> &element_rest_length_field = *element_rest_length_field_ptr_;
  const stk::mesh::Field<double> &element_spring_constant_field = *element_spring_constant_field_ptr_;
  stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

  // At the end of this loop, all locally owned nodes will be up-to-date. Shared nodes will need to be summed.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & spring_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, intersection_with_valid_entity_parts,
      [&node_force_field, &node_coord_field, &element_rest_length_field, &element_spring_constant_field](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring_element) {
        // Fetch the connected nodes.
        const stk::mesh::Entity *nodes = bulk_data.begin_nodes(spring_element);
        const stk::mesh::Entity &node1 = nodes[0];
        const stk::mesh::Entity &node2 = nodes[1];

        // Fetch the required node and element field data.
        const double *node1_coord = stk::mesh::field_data(node_coord_field, node1);
        const double *node2_coord = stk::mesh::field_data(node_coord_field, node2);
        const double *element_rest_length = stk::mesh::field_data(element_rest_length_field, spring_element);
        const double *element_spring_constant = stk::mesh::field_data(element_spring_constant_field, spring_element);

        // Compute the separation distance and the unit vector from node1 to node2.
        double edge_tangent_left_to_right[3] = {node2_coord[0] - node1_coord[0], node2_coord[1] - node1_coord[1],
                                                node2_coord[2] - node1_coord[2]};
        const double edge_length = std::sqrt(edge_tangent_left_to_right[0] * edge_tangent_left_to_right[0] +
                                             edge_tangent_left_to_right[1] * edge_tangent_left_to_right[1] +
                                             edge_tangent_left_to_right[2] * edge_tangent_left_to_right[2]);
        const double inv_edge_length = 1.0 / edge_length;
        edge_tangent_left_to_right[0] *= inv_edge_length;
        edge_tangent_left_to_right[1] *= inv_edge_length;
        edge_tangent_left_to_right[2] *= inv_edge_length;

        // Compute the spring force.
        const double spring_force = element_spring_constant[0] * (edge_length - element_rest_length[0]);
        const double right_node_force[3] = {-spring_force * edge_tangent_left_to_right[0],
                                            -spring_force * edge_tangent_left_to_right[1],
                                            -spring_force * edge_tangent_left_to_right[2]};

        // Add the spring force to the nodes.
        double *node1_force = stk::mesh::field_data(node_force_field, node1);
        double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
        node1_force[0] -= right_node_force[0];
#pragma omp atomic
        node1_force[1] -= right_node_force[1];
#pragma omp atomic
        node1_force[2] -= right_node_force[2];
#pragma omp atomic
        node2_force[0] += right_node_force[0];
#pragma omp atomic
        node2_force[1] += right_node_force[1];
#pragma omp atomic
        node2_force[2] += right_node_force[2];
      });

  // Sum the forces on shared nodes.
  stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_});
}
//}

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace constraints

}  // namespace mundy
