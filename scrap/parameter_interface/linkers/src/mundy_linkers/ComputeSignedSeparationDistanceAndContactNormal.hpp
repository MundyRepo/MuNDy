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

#ifndef MUNDY_LINKERS_COMPUTESIGNEDSEPARATIONDISTANCEANDCONTACTNORMAL_HPP_
#define MUNDY_LINKERS_COMPUTESIGNEDSEPARATIONDISTANCEANDCONTACTNORMAL_HPP_

/// \file ComputeSignedSeparationDistanceAndContactNormal.hpp
/// \brief Declaration of the ComputeSignedSeparationDistanceAndContactNormal class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSphereLinker.hpp>  // for mundy::...::SphereSphereLinker
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderLinker.hpp>  // for mundy::...::SphereSpherocylinderLinker
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderSegmentLinker.hpp>  // for mundy::...::SphereSpherocylinderSegmentLinker
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSegmentSpherocylinderSegmentLinker.hpp>  // for mundy::...::SpherocylinderSegmentSpherocylinderSegmentLinker
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderLinker.hpp>  // for mundy::...::SpherocylinderSpherocylinderLinker
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderSegmentLinker.hpp>  // for mundy::...::SpherocylinderSpherocylinderSegmentLinker
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace linkers {

/// \class ComputeSignedSeparationDistanceAndContactNormal
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeSignedSeparationDistanceAndContactNormal
    : public mundy::meta::MetaKernelDispatcher<ComputeSignedSeparationDistanceAndContactNormal,
                                               mundy::meta::make_registration_string(
                                                   "COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeSignedSeparationDistanceAndContactNormal() = delete;

  /// \brief Constructor
  ComputeSignedSeparationDistanceAndContactNormal(mundy::mesh::BulkData *const bulk_data_ptr,
                                                  const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeSignedSeparationDistanceAndContactNormal,
                                          mundy::meta::make_registration_string(
                                              "COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL")>(bulk_data_ptr,
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
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("linker_signed_separation_distance_field_name",
                 std::string(default_linker_signed_separation_distance_field_name_),
                 "Name of the constraint-rank field within which the signed separation distance will be written.")
            .set("linker_contact_normal_field_name", std::string(default_linker_contact_normal_field_name_),
                 "Name of the constraint-rank field within which the contact normal (pointing from left "
                 "entity to right entity) will be written.");
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

  static constexpr std::string_view default_linker_signed_separation_distance_field_name_ =
      "LINKER_SIGNED_SEPARATION_DISTANCE";
  static constexpr std::string_view default_linker_contact_normal_field_name_ = "LINKER_CONTACT_NORMAL";
  //@}
};  // ComputeSignedSeparationDistanceAndContactNormal

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_brownian_velocity_kernels_ = []() {
  // Register our default kernels
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SphereSphereLinker>("SPHERE_SPHERE_LINKER");
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SphereSpherocylinderLinker>("SPHERE_SPHEROCYLINDER_LINKER");
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SphereSpherocylinderSegmentLinker>("SPHERE_SPHEROCYLINDER_SEGMENT_LINKER");
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SpherocylinderSegmentSpherocylinderSegmentLinker>("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER");
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SpherocylinderSpherocylinderLinker>("SPHEROCYLINDER_SPHEROCYLINDER_LINKER");
  mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::OurKernelFactory::register_new_class<
      mundy::linkers::compute_signed_separation_distance_contact_normal_and_contact_points::kernels::
          SpherocylinderSpherocylinderSegmentLinker>("SPHEROCYLINDER_SPHEROCYLINDER_SEGMENT_LINKER");

  return true;
}();

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_COMPUTESIGNEDSEPARATIONDISTANCEANDCONTACTNORMAL_HPP_
