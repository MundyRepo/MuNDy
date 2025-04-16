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

/// \file Sphere.cpp
/// \brief Definition of the MapSurfaceForceToRigidBodyForce's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_surface_force_to_rigid_body_force/kernels/Sphere.hpp>  // for mundy::motion::...::kernels::Sphere.hpp

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_surface_force_to_rigid_body_force {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument, "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the given parameter list.
  node_coord_field_name_ = valid_fixed_params.get<std::string>("node_coord_field_name");
  node_force_field_name_ = valid_fixed_params.get<std::string>("node_force_field_name");
  node_torque_field_name_ = valid_fixed_params.get<std::string>("node_torque_field_name");

  // Get the field pointers.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name_);
  node_torque_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_torque_field_name_);

  // Check that the fields exist.
  MUNDY_THROW_REQUIRE(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(node_force_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_force_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(node_torque_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_torque_field_ptr cannot be a nullptr. Check that the field exists.");
}
//}

// \name MetaKernel interface implementation
//{

void Sphere::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the validated parameter list.
  alpha_ = valid_mutable_params.get<double>("alpha");
  beta_ = valid_mutable_params.get<double>("beta");
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &sphere_element) const {
  // TODO(palmerb4): Populate the ghosted nodes and spheres information.
  stk::mesh::Entity const linker = bulk_data_ptr_->begin(sphere_element, stk::topology::CONSTRAINT_RANK)[0];
  stk::mesh::Entity const *surface_nodes = bulk_data_ptr_->begin_nodes(linker);
  stk::mesh::Entity const body_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];

  double *body_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, body_node);
  double *body_node_force = stk::mesh::field_data(*node_force_field_ptr_, body_node);
  double *body_node_torque = stk::mesh::field_data(*node_torque_field_ptr_, body_node);
  body_node_force[0] *= beta_;
  body_node_force[1] *= beta_;
  body_node_force[2] *= beta_;
  body_node_torque[0] *= beta_;
  body_node_torque[1] *= beta_;
  body_node_torque[2] *= beta_;

  unsigned num_surface_nodes = bulk_data_ptr_->num_nodes(linker);
  for (unsigned i = 0; i < num_surface_nodes; i++) {
    double *surface_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, surface_nodes[i]);
    double *surface_node_force = stk::mesh::field_data(*node_force_field_ptr_, surface_nodes[i]);
    double *surface_node_torque = stk::mesh::field_data(*node_torque_field_ptr_, surface_nodes[i]);

    double relative_pos[3] = {surface_node_coords[0] - body_node_coords[0],
                              surface_node_coords[1] - body_node_coords[1],
                              surface_node_coords[2] - body_node_coords[2]};

    body_node_force[0] += alpha_ * surface_node_force[0];
    body_node_force[1] += alpha_ * surface_node_force[1];
    body_node_force[2] += alpha_ * surface_node_force[2];
    body_node_torque[0] += alpha_ * (surface_node_torque[0] + relative_pos[1] * surface_node_force[2] -
                                     relative_pos[2] * surface_node_force[1]);
    body_node_torque[1] += alpha_ * (surface_node_torque[1] + relative_pos[2] * surface_node_force[0] -
                                     relative_pos[0] * surface_node_force[2]);
    body_node_torque[2] += alpha_ * (surface_node_torque[2] + relative_pos[0] * surface_node_force[1] -
                                     relative_pos[1] * surface_node_force[0]);
  }
  // TODO(palmerb4): Communicate the ghosted information. Overwrite the information in the non-ghosted spheres.
}
//}

}  // namespace kernels

}  // namespace map_surface_force_to_rigid_body_force

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy
