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
/// \brief Definition of LocalDrag's Sphere kernel.

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
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/local_drag/kernels/Sphere.hpp>  // for mundy::motion::...::kernels::Sphere.hpp
#include <mundy_motion/utils/Quaternion.hpp>  // for mundy::utils::Quaternion

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_rigid_body_force_to_rigid_body_velocity {

namespace techniques {

namespace local_drag {

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
  node_force_field_name_ = valid_fixed_params.get<std::string>("node_force_field_name");
  node_torque_field_name_ = valid_fixed_params.get<std::string>("node_torque_field_name");
  node_velocity_field_name_ = valid_fixed_params.get<std::string>("node_velocity_field_name");
  node_omega_field_name_ = valid_fixed_params.get<std::string>("node_omega_field_name");
  element_radius_field_name_ = valid_fixed_params.get<std::string>("element_radius_field_name");

  // Get the field pointers.
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name_);
  node_torque_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_torque_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
  node_omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_omega_field_name_);
  element_radius_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name_);

  // Check that the fields exist.
  MUNDY_THROW_REQUIRE(node_force_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_force_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(node_torque_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_torque_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(node_velocity_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_velocity_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(node_omega_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_omega_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: element_radius_field_ptr cannot be a nullptr. Check that the field exists.");
}
//}

// \name MetaKernel interface implementation
//{

void Sphere::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the given parameter list.
  viscosity_ = valid_mutable_params.get<double>("viscosity");
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &sphere_element) const {
  // Fetch the sphere's fields.
  const stk::mesh::Entity center_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];

  double *node_force = stk::mesh::field_data(*node_force_field_ptr_, center_node);
  double *node_torque = stk::mesh::field_data(*node_torque_field_ptr_, center_node);
  double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, center_node);
  double *node_omega = stk::mesh::field_data(*node_omega_field_ptr_, center_node);
  double *radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);

  // Compute the mobility matrix for the sphere using local drag.
  static constexpr const double pi = 3.14159265358979323846;
  const double drag_trans = 6 * pi * radius[0] * viscosity_;
  const double drag_rot = 8 * pi * radius[0] * radius[0] * radius[0] * viscosity_;
  const double drag_trans_inv = 1.0 / drag_trans;
  const double drag_rot_inv = 1.0 / drag_rot;

  // solve for the induced node_velocity and node_omega
  node_velocity[0] = drag_trans_inv * node_force[0];
  node_velocity[1] = drag_trans_inv * node_force[1];
  node_velocity[2] = drag_trans_inv * node_force[2];
  node_omega[0] = drag_rot_inv * node_torque[0];
  node_omega[1] = drag_rot_inv * node_torque[1];
  node_omega[2] = drag_rot_inv * node_torque[2];
}
//}

}  // namespace kernels

}  // namespace local_drag

}  // namespace techniques

}  // namespace map_rigid_body_force_to_rigid_body_velocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy
