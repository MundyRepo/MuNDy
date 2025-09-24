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

/// \file AngularSpringsKernel.cpp
/// \brief Definition of the ComputeConstraintForcing's AngularSpringsKernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/FieldParallel.hpp>  // for stk::mesh::communicate_field_data

// Mundy libs
#include <mundy_constraints/AngularSprings.hpp>  // for mundy::constraints::AngularSprings
#include <mundy_constraints/compute_constraint_forcing/kernels/AngularSpringsKernel.hpp>  // for mundy::constraints::compute_constraint_forcing::kernels::AngularSpringsKernel
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

namespace mundy {

namespace constraints {

namespace compute_constraint_forcing {

namespace kernels {

// \name Constructors and destructor
//{

AngularSpringsKernel::AngularSpringsKernel(mundy::mesh::BulkData *const bulk_data_ptr,
                                           const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "AngularSpringsKernel: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(AngularSpringsKernel::get_valid_fixed_params());

  // Store the valid entity parts for the kernel.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  for (const std::string &part_name : valid_entity_part_names) {
    valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
    MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                        std::string("AngularSpringsKernel: Part '") + part_name +
                            "' from the valid_entity_part_names does not exist in the meta data.");
  }

  // Fetch the fields.
  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
  const std::string node_coord_field_name = AngularSprings::get_node_coord_field_name();
  const std::string element_rest_angle_field_name = AngularSprings::get_element_rest_angle_field_name();
  const std::string element_spring_constant_field_name = AngularSprings::get_element_spring_constant_field_name();

  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  node_coordinates_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_rest_angle_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_rest_angle_field_name);
  element_spring_constant_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_spring_constant_field_name);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> AngularSpringsKernel::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void AngularSpringsKernel::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(AngularSpringsKernel::get_valid_mutable_params());
}
//}

// \name Actions
//{

void AngularSpringsKernel::execute(const stk::mesh::Selector &spring_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coordinates_field_ptr_, element_rest_angle_field_ptr_,
                                                      element_spring_constant_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coord_field = *node_coordinates_field_ptr_;
  const stk::mesh::Field<double> &element_rest_angle_field = *element_rest_angle_field_ptr_;
  const stk::mesh::Field<double> &element_spring_constant_field = *element_spring_constant_field_ptr_;
  stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

  // At the end of this loop, all locally owned nodes will be up-to-date. Shared nodes will need to be summed.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & spring_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, intersection_with_valid_entity_parts,
      [&node_force_field, &node_coord_field, &element_rest_angle_field, &element_spring_constant_field](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &angular_spring_element) {
        // Fetch the connected nodes. Note, for a BEAM_3 topology the nodes are numbered
        // 1     2
        //  \   /
        //   \ /
        //    3
        const stk::mesh::Entity *nodes = bulk_data.begin_nodes(angular_spring_element);
        const stk::mesh::Entity &node1 = nodes[0];
        const stk::mesh::Entity &node2 = nodes[1];
        const stk::mesh::Entity &node3 = nodes[2];

        // Fetch the required node and element field data.
        const auto node1_coord = mundy::mesh::vector3_field_data(node_coord_field, node1);
        const auto node2_coord = mundy::mesh::vector3_field_data(node_coord_field, node2);
        const auto node3_coord = mundy::mesh::vector3_field_data(node_coord_field, node3);
        const double element_rest_angle = stk::mesh::field_data(element_rest_angle_field, angular_spring_element)[0];
        const double element_spring_constant =
            stk::mesh::field_data(element_spring_constant_field, angular_spring_element)[0];

        // Get the vectors between the nodes.
        const auto vec_from_3_to_1 = node1_coord - node3_coord;
        const auto vec_from_3_to_2 = node2_coord - node3_coord;

        // Get the necessary magnitudes (and square magnitudes) of the vectors.
        const double dist_sq_from_3_to_1 = mundy::math::dot(vec_from_3_to_1, vec_from_3_to_1);
        const double dist_sq_from_3_to_2 = mundy::math::dot(vec_from_3_to_2, vec_from_3_to_2);
        const double dist_from_3_to_1 = std::sqrt(dist_sq_from_3_to_1);
        const double dist_from_3_to_2 = std::sqrt(dist_sq_from_3_to_2);

        // Get the minor angle between lines 31 and 32.
        const double cos_of_angle =
            mundy::math::dot(vec_from_3_to_1, vec_from_3_to_2) / (dist_from_3_to_1 * dist_from_3_to_2);

        // Compute the spring torque.
        const double torque_about_3 = element_spring_constant * (cos_of_angle - std::cos(element_rest_angle));

        // Now, we follow HOOMD's convention for computing the node forces given the torque.
        // While the torque is computed about node 3, it will induce forces on all three nodes.
        const double a11 = torque_about_3 * cos_of_angle / dist_sq_from_3_to_1;
        const double a13 = -torque_about_3 / (dist_from_3_to_1 * dist_from_3_to_2);
        const double a33 = torque_about_3 * cos_of_angle / dist_sq_from_3_to_2;

        const auto force_on_1 = a11 * vec_from_3_to_1 + a13 * vec_from_3_to_2;
        const auto force_on_2 = a33 * vec_from_3_to_2 + a13 * vec_from_3_to_1;
        const auto force_on_3 = -force_on_1 - force_on_2;

        // Add the spring force to the nodes.
        auto node1_force = mundy::mesh::vector3_field_data(node_force_field, node1);
        auto node2_force = mundy::mesh::vector3_field_data(node_force_field, node2);
        auto node3_force = mundy::mesh::vector3_field_data(node_force_field, node3);

#pragma omp atomic
        node1_force[0] += force_on_1[0];
#pragma omp atomic
        node1_force[1] += force_on_1[1];
#pragma omp atomic
        node1_force[2] += force_on_1[2];
#pragma omp atomic
        node2_force[0] += force_on_2[0];
#pragma omp atomic
        node2_force[1] += force_on_2[1];
#pragma omp atomic
        node2_force[2] += force_on_2[2];
#pragma omp atomic
        node3_force[0] += force_on_3[0];
#pragma omp atomic
        node3_force[1] += force_on_3[1];
#pragma omp atomic
        node3_force[2] += force_on_3[2];
      });

  // Sum the forces on shared nodes.
  stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_});
}
//}

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace constraints

}  // namespace mundy
