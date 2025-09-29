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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYFORCETORIGIDBODYVELOCITY_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYFORCETORIGIDBODYVELOCITY_HPP_

/// \file MapRigidBodyForceToRigidBodyVelocity.hpp
/// \brief Declaration of the MapRigidBodyForceToRigidBodyVelocity class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/MetaTechniqueDispatcher.hpp>             // for mundy::meta::MetaMethodSubsetExecutionDispatcher
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/LocalDrag.hpp>  // for mundy::motion::...::techniques::LocalDrag

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

/// \class MapRigidBodyForceToRigidBodyVelocity
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class MapRigidBodyForceToRigidBodyVelocity
    : public mundy::meta::MetaMethodSubsetExecutionDispatcher<MapRigidBodyForceToRigidBodyVelocity,
                                                              mundy::core::make_string_literal(
                                                                  "MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MapRigidBodyForceToRigidBodyVelocity() = delete;

  /// \brief Constructor
  MapRigidBodyForceToRigidBodyVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<MapRigidBodyForceToRigidBodyVelocity>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY";
  //@}
};  // MapRigidBodyForceToRigidBodyVelocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default techniques
MUNDY_REGISTER_METACLASS("LOCAL_DRAG",
                         mundy::motion::compute_mobility::techniques::rigid_body_motion::
                             map_rigid_body_force_to_rigid_body_velocity::techniques::LocalDrag,
                         mundy::motion::compute_mobility::techniques::rigid_body_motion::
                             MapRigidBodyForceToRigidBodyVelocity::OurMethodFactory)
//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYFORCETORIGIDBODYVELOCITY_HPP_
