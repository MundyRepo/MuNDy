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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_

/// \file LocalDrag.hpp
/// \brief Declaration of the LocalDrag class

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                            // for mundy::meta::PartReqs
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/local_drag/kernels/Sphere.hpp>  // for mundy::motion::...::kernels::Sphere

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_rigid_body_force_to_rigid_body_velocity {

namespace techniques {

/// \class LocalDrag
/// \brief Method for computing the rigid body force to rigid body velocity using the local drag of different parts.
class LocalDrag
    : public mundy::meta::MetaKernelDispatcher<LocalDrag, mundy::meta::make_registration_string("LOCAL_DRAG")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  LocalDrag() = delete;

  /// \brief Constructor
  LocalDrag(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<LocalDrag, mundy::meta::make_registration_string("LOCAL_DRAG")>(
            bulk_data_ptr, fixed_params) {
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
    default_parameter_list.set("node_force_field_name", std::string(node_force_field_name),
                               "Name of the node field containing the force.");
    default_parameter_list.set("node_torque_field_name", std::string(node_torque_field_name),
                               "Name of the node field containing the torque.");
    default_parameter_list.set("node_velocity_field_name", std::string(node_velocity_field_name),
                               "Name of the node field containing the velocity.");
    default_parameter_list.set("node_omega_field_name", std::string(node_omega_field_name),
                               "Name of the node field containing the angular velocity.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("viscosity", default_viscosity_, "The viscosity of the suspending fluid.");
    return default_parameter_list;
  }
  //@}
 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_torque_field_name_ = "NODE_TORQUE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  static constexpr double default_viscosity_ = 1.0;
  //@}
};  // LocalDrag

}  // namespace techniques

}  // namespace map_rigid_body_force_to_rigid_body_velocity

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
                             map_rigid_body_force_to_rigid_body_velocity::techniques::local_drag::kernels::Sphere,
                         mundy::motion::compute_mobility::techniques::rigid_body_motion::
                             map_rigid_body_force_to_rigid_body_velocity::techniques::LocalDrag::OurKernelFactory)

//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_
