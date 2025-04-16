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

/// \file RigidBodyMotion.cpp
/// \brief Definition of the RigidBodyMotion class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MeshReqs.hpp>                            // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_motion/compute_mobility/techniques/RigidBodyMotion.hpp>  // for mundy::motion::...::RigidBodyMotion

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

// \name Constructors and destructor
//{

RigidBodyMotion::RigidBodyMotion(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "RigidBodyMotion: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fetch the parameters for each sub-method.
  Teuchos::ParameterList &technique_params = valid_fixed_params.sublist("technique");
  const std::string technique_name = technique_params.get<std::string>("name");
  Teuchos::ParameterList &map_rbf_to_rbv_params =
      valid_fixed_params.sublist("methods").sublist("map_rigid_body_force_to_rigid_body_velocity");
  Teuchos::ParameterList &map_rbv_to_sv_params =
      valid_fixed_params.sublist("methods").sublist("map_rigid_body_velocity_to_surface_velocity");
  Teuchos::ParameterList &map_sf_to_rbf_params =
      valid_fixed_params.sublist("methods").sublist("map_surface_force_to_rigid_body_force");

  // Initialize and store the sub-methods.
  const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
  const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
  const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
  map_rigid_body_force_to_rigid_body_velocity_method_ptr_ =
      OurMethodFactory::create_new_instance(rbf_to_rbv_name, bulk_data_ptr_, map_rbf_to_rbv_params);
  map_rigid_body_velocity_to_surface_velocity_method_ptr_ =
      OurMethodFactory::create_new_instance(rbv_to_sv_name, bulk_data_ptr_, map_rbv_to_sv_params);
  map_surface_force_to_rigid_body_force_method_ptr_ =
      OurMethodFactory::create_new_instance(sf_to_rbf_name, bulk_data_ptr_, map_sf_to_rbf_params);
}
//}

// \name MetaFactory static interface implementation
//{

void RigidBodyMotion::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fetch the parameters for each sub-method.
  Teuchos::ParameterList &technique_params = valid_mutable_params.sublist("technique");
  const std::string technique_name = technique_params.get<std::string>("name");
  Teuchos::ParameterList &map_rbf_to_rbv_params =
      valid_mutable_params.sublist("methods").sublist("map_rigid_body_force_to_rigid_body_velocity");
  Teuchos::ParameterList &map_rbv_to_sv_params =
      valid_mutable_params.sublist("methods").sublist("map_rigid_body_velocity_to_surface_velocity");
  Teuchos::ParameterList &map_sf_to_rbf_params =
      valid_mutable_params.sublist("methods").sublist("map_surface_force_to_rigid_body_force");

  // Set the mutable params for each of our sub-methods.
  map_rigid_body_force_to_rigid_body_velocity_method_ptr_->set_mutable_params(map_rbf_to_rbv_params);
  map_rigid_body_velocity_to_surface_velocity_method_ptr_->set_mutable_params(map_rbv_to_sv_params);
  map_surface_force_to_rigid_body_force_method_ptr_->set_mutable_params(map_sf_to_rbf_params);
}
//}

// \name Actions
//{

void RigidBodyMotion::execute(const stk::mesh::Selector &input_selector) {
  map_surface_force_to_rigid_body_force_method_ptr_->execute(input_selector);
  map_rigid_body_force_to_rigid_body_velocity_method_ptr_->execute(input_selector);
  map_rigid_body_velocity_to_surface_velocity_method_ptr_->execute(input_selector);
}
//}

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy
