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

#ifndef MUNDY_METHODS_COMPUTEMOBILITY_HPP_
#define MUNDY_METHODS_COMPUTEMOBILITY_HPP_

/// \file ComputeMobility.hpp
/// \brief Declaration of the ComputeMobility class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>             // for mundy::meta::MetaMethodSubsetExecutionDispatcher
#include <mundy_motion/compute_mobility/techniques/RigidBodyMotion.hpp>  // for mundy::motion::compute_mobility::techniques::RigidBodyMotion

namespace mundy {

namespace motion {

/// \class ComputeMobility
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class ComputeMobility : public mundy::meta::MetaMethodSubsetExecutionDispatcher<
                            ComputeMobility, void, mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
                            mundy::meta::make_registration_string("RIGID_BODY_MOTION")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeMobility() = delete;

  /// \brief Constructor
  ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaMethodSubsetExecutionDispatcher<ComputeMobility, void,
                                                         mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
                                                         mundy::meta::make_registration_string("RIGID_BODY_MOTION")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaMethodSubsetExecutionDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require all kernels registered with our kernel factory to have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    // TODO(palmerb4): Add fixed parameters here
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require all kernels registered with our kernel factory to have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    // TODO(palmerb4): Add fixed parameters here
    return default_parameter_list;
  }
  //@}
};  // ComputeMobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default techniques
MUNDY_REGISTER_METACLASS("RIGID_BODY_MOTION", mundy::motion::compute_mobility::techniques::RigidBodyMotion,
                         mundy::motion::ComputeMobility::OurMethodFactory)

//@}

#endif  // MUNDY_METHODS_COMPUTEMOBILITY_HPP_
