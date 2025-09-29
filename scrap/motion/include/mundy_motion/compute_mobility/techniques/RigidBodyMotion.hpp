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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGIDBODYMOTION_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGIDBODYMOTION_HPP_

/// \file RigidBodyMotion.hpp
/// \brief Declaration of the RigidBodyMotion class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

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
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                            // for mundy::meta::PartReqs
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyForceToRigidBodyVelocity.hpp>  // for mundy::motion::...::MapRigidBodyForceToRigidBodyVelocity
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyVelocityToSurfaceVelocity.hpp>  // for mundy::motion::...::MapRigidBodyVelocityToSurfaceVelocity
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.hpp>  // for mundy::motion::...::MapSurfaceForceToRigidBodyForce

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

/// \class RigidBodyMotion
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class RigidBodyMotion : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  // TODO(palmerb4): OurMethodFactory needs to be broken into a different factory for each group of methods.
  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurMethodFactory = mundy::meta::MetaMethodFactory<void, RigidBodyMotion>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  RigidBodyMotion() = delete;

  /// \brief Constructor
  RigidBodyMotion(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fetch the parameters for this part's sub-methods.
    Teuchos::ParameterList &map_rbf_to_rbv_params =
        valid_fixed_params.sublist("submethods").sublist("map_rigid_body_force_to_rigid_body_velocity");
    Teuchos::ParameterList &map_rbv_to_sv_params =
        valid_fixed_params.sublist("submethods").sublist("map_rigid_body_velocity_to_surface_velocity");
    Teuchos::ParameterList &map_sf_to_rbf_params =
        valid_fixed_params.sublist("submethods").sublist("map_surface_force_to_rigid_body_force");

    // Collect and sync the submethod requirements.
    auto mesh_reqs = std::make_shared<mundy::meta::MeshReqs>();
    const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
    const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
    const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
    mesh_reqs->sync(OurMethodFactory::get_mesh_requirements(rbf_to_rbv_name, map_rbf_to_rbv_params));
    mesh_reqs->sync(OurMethodFactory::get_mesh_requirements(rbv_to_sv_name, map_rbv_to_sv_params));
    mesh_reqs->sync(OurMethodFactory::get_mesh_requirements(sf_to_rbf_name, map_sf_to_rbf_params));

    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    Teuchos::ParameterList &map_rbf_to_rbv_params =
        fixed_params_ptr->sublist("submethods", false).sublist("map_rigid_body_force_to_rigid_body_velocity", false);
    Teuchos::ParameterList &map_rbv_to_sv_params =
        fixed_params_ptr->sublist("submethods", false).sublist("map_rigid_body_velocity_to_surface_velocity", false);
    Teuchos::ParameterList &map_sf_to_rbf_params =
        fixed_params_ptr->sublist("submethods", false).sublist("map_surface_force_to_rigid_body_force", false);

    if (map_rbf_to_rbv_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         std::string("RigidBodyMotion: Type error. Given a map_rigid_body_force_to_rigid_body_velocity ")
                         + "parameter with name 'name' but "
                             + "with a type other than std::string");
    } else {
      map_rbf_to_rbv_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                                "Name of the method for mapping from rigid body force to rigid body velocity.");
    }

    if (map_rbv_to_sv_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         std::string("RigidBodyMotion: Type error. Given a map_rigid_body_velocity_to_surface_velocity ")
                         + "parameter with name 'name' but "
                             + "with a type other than std::string");
    } else {
      map_rbv_to_sv_params.set("name", std::string(default_map_rbv_to_sv_name_),
                               "Name of the method for mapping from rigid body velocity to surface velocity.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          std::string("RigidBodyMotion: Type error. Given a map_surface_force_to_rigid_body_force parameter with name 'name' but ")
              + "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set("name", std::string(default_map_sf_to_rbf_name_),
                               "Name of the method for mapping from surface force to rigid body force.");
    }

    const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
    const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
    const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(rbf_to_rbv_name, &map_rbf_to_rbv_params);
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(rbv_to_sv_name, &map_rbv_to_sv_params);
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(sf_to_rbf_name, &map_sf_to_rbf_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    Teuchos::ParameterList &map_rbf_to_rbv_params =
        mutable_params_ptr->sublist("submethods", false).sublist("map_rigid_body_force_to_rigid_body_velocity", false);
    Teuchos::ParameterList &map_rbv_to_sv_params =
        mutable_params_ptr->sublist("submethods", false).sublist("map_rigid_body_velocity_to_surface_velocity", false);
    Teuchos::ParameterList &map_sf_to_rbf_params =
        mutable_params_ptr->sublist("submethods", false).sublist("map_surface_force_to_rigid_body_force", false);

    if (map_rbf_to_rbv_params.isParameter("name")) {
      const bool valid_type = map_rbf_to_rbv_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         std::string("RigidBodyMotion: Type error. Given a map_rigid_body_force_to_rigid_body_velocity ")
                         + "parameter with name 'name' but "
                             + "with a type other than std::string");
    } else {
      map_rbf_to_rbv_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                                "Name of the method for mapping from rigid body force to rigid body velocity.");
    }

    if (map_rbv_to_sv_params.isParameter("name")) {
      const bool valid_type = map_rbv_to_sv_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         std::string("RigidBodyMotion: Type error. Given a map_rigid_body_velocity_to_surface_velocity ")
                         + "parameter with name 'name' but "
                             + "with a type other than std::string");
    } else {
      map_rbv_to_sv_params.set("name", std::string(default_map_rbv_to_sv_name_),
                               "Name of the method for mapping from rigid body velocity to surface velocity.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = map_sf_to_rbf_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          std::string("RigidBodyMotion: Type error. Given a map_surface_force_to_rigid_body_force parameter with name 'name' but ")
              + "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set("name", std::string(default_map_sf_to_rbf_name_),
                               "Name of the method for mapping from surface force to rigid body force.");
    }

    const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
    const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
    const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(rbf_to_rbv_name, &map_rbf_to_rbv_params);
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(rbv_to_sv_name, &map_rbv_to_sv_params);
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(sf_to_rbf_name, &map_sf_to_rbf_params);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<RigidBodyMotion>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_map_rbf_to_rbv_name_ = "MapRigidBodyForceToRigidBodyVelocity";
  static constexpr std::string_view default_map_rbv_to_sv_name_ = "MapRigidBodyVelocityToSurfaceVelocity";
  static constexpr std::string_view default_map_sf_to_rbf_name_ = "MapSurfaceForceToRigidBodyForce";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "RIGID_BODY_MOTION";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Method for mapping from rigid body force to rigid body velocity.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>
      map_rigid_body_force_to_rigid_body_velocity_method_ptr_;

  /// \brief Method for mapping from rigid body velocity to surface velocity.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>
      map_rigid_body_velocity_to_surface_velocity_method_ptr_;

  /// \brief Method for mapping from surface force to rigid body force.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>
      map_surface_force_to_rigid_body_force_method_ptr_;
  //@}
};  // RigidBodyMotion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default method with our method factory.
MUNDY_REGISTER_METACLASS(
    "MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY",
    mundy::motion::compute_mobility::techniques::rigid_body_motion::MapRigidBodyForceToRigidBodyVelocity,
    mundy::motion::compute_mobility::techniques::RigidBodyMotion::OurMethodFactory)
MUNDY_REGISTER_METACLASS(
    "MAP_RIGID_BODY_VELOCITY_TO_SURFACE_VELOCITY",
    mundy::motion::compute_mobility::techniques::rigid_body_motion::MapRigidBodyVelocityToSurfaceVelocity,
    mundy::motion::compute_mobility::techniques::RigidBodyMotion::OurMethodFactory)
MUNDY_REGISTER_METACLASS(
    "MAP_SURFACE_FORCE_TO_RIGID_BODY_FORCE",
    mundy::motion::compute_mobility::techniques::rigid_body_motion::MapSurfaceForceToRigidBodyForce,
    mundy::motion::compute_mobility::techniques::RigidBodyMotion::OurMethodFactory)
//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGIDBODYMOTION_HPP_
