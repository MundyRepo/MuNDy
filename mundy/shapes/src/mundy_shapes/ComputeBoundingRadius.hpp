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

#ifndef MUNDY_SHAPES_COMPUTEBOUNDINGRADIUS_HPP_
#define MUNDY_SHAPES_COMPUTEBOUNDINGRADIUS_HPP_

/// \file ComputeBoundingRadius.hpp
/// \brief Declaration of the ComputeBoundingRadius class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>           // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_shapes/compute_bounding_radius/kernels/Sphere.hpp>  // for mundy::shapes::compute_bounding_radius::kernels::Sphere
#include <mundy_shapes/compute_bounding_radius/kernels/Spherocylinder.hpp>  // for mundy::shapes::compute_bounding_radius::kernels::Spherocylinder
#include <mundy_shapes/compute_bounding_radius/kernels/SpherocylinderSegment.hpp>  // for mundy::shapes::compute_bounding_radius::kernels::SpherocylinderSegment

namespace mundy {

namespace shapes {

/// \class ComputeBoundingRadius
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeBoundingRadius
    : public mundy::meta::MetaKernelDispatcher<ComputeBoundingRadius,
                                               mundy::meta::make_registration_string("COMPUTE_BOUNDING_SPHERE")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeBoundingRadius() = delete;

  /// \brief Constructor
  ComputeBoundingRadius(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeBoundingRadius,
                                          mundy::meta::make_registration_string("COMPUTE_BOUNDING_SPHERE")>(
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
    static Teuchos::ParameterList default_parameter_list = Teuchos::ParameterList().set(
        "element_bounding_radius_field_name", std::string(default_element_bounding_radius_field_name_),
        "Name of the element field within which the output bounding radius will be written.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list = Teuchos::ParameterList().set(
        "buffer_distance", default_buffer_distance_, "Buffer distance to be added to the bounding radius.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_element_bounding_radius_field_name_ = "ELEMENT_BOUNDING_RADIUS";
  //@}
};  // ComputeBoundingRadius

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_bounding_radius_kernels_ = []() {
  // Register our default kernels
  mundy::shapes::ComputeBoundingRadius::OurKernelFactory::register_new_class<
      mundy::shapes::compute_bounding_radius::kernels::Sphere>("SPHERE");
  mundy::shapes::ComputeBoundingRadius::OurKernelFactory::register_new_class<
      mundy::shapes::compute_bounding_radius::kernels::Spherocylinder>("SPHEROCYLINDER");
  mundy::shapes::ComputeBoundingRadius::OurKernelFactory::register_new_class<
      mundy::shapes::compute_bounding_radius::kernels::SpherocylinderSegment>("SPHEROCYLINDER_SEGMENT");
  return true;
}();

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_COMPUTEBOUNDINGRADIUS_HPP_
