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

#ifndef MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_
#define MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_

/// \file LinkerPotentialForceReduction.hpp
/// \brief Declaration of the LinkerPotentialForceReduction class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_linkers/linker_potential_force_reduction/kernels/Sphere.hpp>  // for mundy::linkers::...::kernels::Sphere
#include <mundy_linkers/linker_potential_force_reduction/kernels/Spherocylinder.hpp>  // for mundy::linkers::...::kernels::Spherocylinder
#include <mundy_linkers/linker_potential_force_reduction/kernels/SpherocylinderSegment.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegment
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace linkers {

/// \class LinkerPotentialForceReduction
/// \brief Method for summing the effect of multiple linker potentials.
class LinkerPotentialForceReduction
    : public mundy::meta::MetaKernelDispatcher<LinkerPotentialForceReduction, mundy::meta::make_registration_string(
                                                                                  "LINKER_POTENTIAL_FORCE_REDUCTION")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  LinkerPotentialForceReduction() = delete;

  /// \brief Constructor
  LinkerPotentialForceReduction(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<LinkerPotentialForceReduction,
                                          mundy::meta::make_registration_string("LINKER_POTENTIAL_FORCE_REDUCTION")>(
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
    const static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("name_of_linker_part_to_reduce_over", std::string(default_name_of_linker_part_to_reduce_over_),
                 "The name of the linker part that we will reduce over.")
            .set<std::string>("linker_potential_force_field_name",
                              std::string(default_linker_potential_force_field_name_),
                              "The field name of the linker potential force field that we will use for the reduction.");
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

  static constexpr std::string_view default_name_of_linker_part_to_reduce_over_ = "NEIGHBOR_LINKERS";
  static constexpr std::string_view default_linker_potential_force_field_name_ = "LINKER_POTENTIAL_FORCE";
  //@}

};  // LinkerPotentialForceReduction

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_linker_potential_reduction_kernels_ = []() {
  // Register our default kernels
  mundy::linkers::LinkerPotentialForceReduction::OurKernelFactory::register_new_class<
      mundy::linkers::linker_potential_force_reduction::kernels::Sphere>("SPHERE");
  mundy::linkers::LinkerPotentialForceReduction::OurKernelFactory::register_new_class<
      mundy::linkers::linker_potential_force_reduction::kernels::Spherocylinder>("SPHEROCYLINDER");
  mundy::linkers::LinkerPotentialForceReduction::OurKernelFactory::register_new_class<
      mundy::linkers::linker_potential_force_reduction::kernels::SpherocylinderSegment>("SPHEROCYLINDER_SEGMENT");
  return true;
}();

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_LINKERPOTENTIALFORCEMAGNITUDEREDUCTION_HPP_
