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

#ifndef MUNDY_LINKERS_EVALUATELINKERPOTENTIALS_HPP_
#define MUNDY_LINKERS_EVALUATELINKERPOTENTIALS_HPP_

/// \file EvaluateLinkerPotentials.hpp
/// \brief Declaration of the EvaluateLinkerPotentials class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_linkers/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.hpp>  // for mundy::linkers::...::kernels::SphereSphereHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderHertzianContact.hpp>  // for mundy::linkers::...::kernels::SphereSpherocylinderHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderSegmentHertzianContact.hpp>  // for mundy::linkers::...::kernels::SphereSpherocylinderSegmentHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentWCA.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentWCA
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSpherocylinderHertzianContact
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderSegmentHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSpherocylinderSegmentHertzianContacts
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace linkers {

/// \class EvaluateLinkerPotentials
/// \brief Method for compute linker potentials.
class EvaluateLinkerPotentials
    : public mundy::meta::MetaKernelDispatcher<EvaluateLinkerPotentials,
                                               mundy::meta::make_registration_string("EVALUATE_LINKER_POTENTIALS")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  EvaluateLinkerPotentials() = delete;

  /// \brief Constructor
  EvaluateLinkerPotentials(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<EvaluateLinkerPotentials,
                                          mundy::meta::make_registration_string("EVALUATE_LINKER_POTENTIALS")>(
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
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}
};  // EvaluateLinkerPotentials

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_evaluate_linker_potentials_kernels_ = []() {
  // Register our default kernels
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SphereSphereHertzianContact>(
      "SPHERE_SPHERE_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SphereSpherocylinderHertzianContact>(
      "SPHERE_SPHEROCYLINDER_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SphereSpherocylinderSegmentHertzianContact>(
      "SPHERE_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSegmentSpherocylinderSegmentHertzianContact>(
      "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSegmentSpherocylinderSegmentWCA>(
      "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_WCA");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSpherocylinderHertzianContact>(
      "SPHEROCYLINDER_SPHEROCYLINDER_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSpherocylinderSegmentHertzianContact>(
      "SPHEROCYLINDER_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT");
  mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory::register_new_class<
      mundy::linkers::evaluate_linker_potentials::kernels::
          SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact>(
      "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_FRICTIONAL_HERTZIAN_CONTACT");

  return true;
}();

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_EVALUATELINKERPOTENTIALS_HPP_
