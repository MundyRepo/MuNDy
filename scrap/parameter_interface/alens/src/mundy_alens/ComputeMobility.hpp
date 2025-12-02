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

#ifndef MUNDY_ALENS_COMPUTEMOBILITY_HPP_
#define MUNDY_ALENS_COMPUTEMOBILITY_HPP_

/// \file ComputeMobility.hpp
/// \brief Declaration of the ComputeMobility class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_alens/compute_mobility/LocalDragNonOrientableSpheres.hpp>  // for mundy::alens::compute_mobility::LocalDragNonOrientableSpheres
#include <mundy_alens/compute_mobility/RPYSpheres.hpp>  // for mundy::alens::compute_mobility::RPYSpheres
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace alens {

/// \class ComputeMobility
/// \brief Method for computing the mobility problem.
class ComputeMobility : public mundy::meta::MetaMethodSubsetExecutionDispatcher<
                            ComputeMobility, void, mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
                            mundy::meta::make_registration_string("LOCAL_DRAG_NONORIENTABLE_SPHERES")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeMobility() = delete;

  /// \brief Constructor
  ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr,
                  const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaMethodSubsetExecutionDispatcher<
            ComputeMobility, void, mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
            mundy::meta::make_registration_string("LOCAL_DRAG_NONORIENTABLE_SPHERES")>(bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaTechniqueDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field.");
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}
};  // ComputeMobility

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_mobility_kernels_ = []() {
  // Register our default kernels
  mundy::alens::ComputeMobility::OurTechniqueFactory::register_new_class<
      mundy::alens::compute_mobility::LocalDragNonOrientableSpheres>("LOCAL_DRAG_NONORIENTABLE_SPHERES");
  mundy::alens::ComputeMobility::OurTechniqueFactory::register_new_class<mundy::alens::compute_mobility::RPYSpheres>(
      "RPY_SPHERES");
  return true;
}();

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_COMPUTEMOBILITY_HPP_
