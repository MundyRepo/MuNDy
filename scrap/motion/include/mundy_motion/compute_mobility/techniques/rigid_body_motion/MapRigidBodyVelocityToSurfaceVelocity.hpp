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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_

/// \file MapRigidBodyVelocityToSurfaceVelocity.hpp
/// \brief Declaration of the MapRigidBodyVelocityToSurfaceVelocity class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_velocity_to_surface_velocity/kernels/Sphere.hpp>  // for mundy::motion::...::kernels::Sphere

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

/// \class MapRigidBodyVelocityToSurfaceVelocity
/// \brief Method for using rigid body motion about a known body point to compute the velocity at all surface points.
class MapRigidBodyVelocityToSurfaceVelocity
    : public mundy::meta::MetaKernelDispatcher<MapRigidBodyVelocityToSurfaceVelocity,
                                               mundy::meta::make_registration_string(
                                                   "MAP_RIGID_BODY_VELOCITY_TO_SURFACE_VELOCITY")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MapRigidBodyVelocityToSurfaceVelocity() = delete;

  /// \brief Constructor
  MapRigidBodyVelocityToSurfaceVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                                        const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<MapRigidBodyVelocityToSurfaceVelocity,
                                          mundy::meta::make_registration_string(
                                              "MAP_RIGID_BODY_VELOCITY_TO_SURFACE_VELOCITY")>(bulk_data_ptr,
                                                                                              fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node field containing the velocity.");
    default_parameter_list.set("node_omega_field_name", std::string(default_node_omega_field_name_),
                               "Name of the node field containing the angular velocity.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  //@}
};  // MapRigidBodyVelocityToSurfaceVelocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default kernels
MUNDY_REGISTER_METACLASS("SPHERE",
                         mundy::motion::compute_mobility::techniques::rigid_body_motion::
                             map_rigid_body_velocity_to_surface_velocity::kernels::Sphere,
                         mundy::motion::compute_mobility::techniques::rigid_body_motion::
                             MapRigidBodyVelocityToSurfaceVelocity::OurKernelFactory)
//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_
