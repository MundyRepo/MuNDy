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

#ifndef MUNDY_CONSTRAINTS_COMPUTECONSTRAINTPROJECTION_HPP_
#define MUNDY_CONSTRAINTS_COMPUTECONSTRAINTPROJECTION_HPP_

/// \file ComputeConstraintProjection.hpp
/// \brief Declaration of the ComputeConstraintProjection class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_constraints/compute_constraint_projection/kernels/Collision.hpp>  // for mundy::constraint::...::Collision
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for mundy::meta::MetaMethodRegistry

namespace mundy {

namespace constraints {

/// \class ComputeConstraintProjection
/// \brief Method for computing the projection of the constraint Lagrange multiplier onto its feasible set.
class ComputeConstraintProjection
    : public mundy::meta::MetaKernelDispatcher<ComputeConstraintProjection,
                                               mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_PROJECTION")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintProjection() = delete;

  /// \brief Constructor
  ComputeConstraintProjection(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeConstraintProjection,
                                          mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_PROJECTION")>(
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
};  // ComputeConstraintProjection

}  // namespace constraints

}  // namespace mundy

//! \name Registration
//@{

MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_PROJECTION",
                         mundy::constraint::compute_constraint_projection::kernels::Collision,
                         mundy::constraint::ComputeConstraintProjection::OurKernelFactory)
//@}

#endif  // MUNDY_CONSTRAINTS_COMPUTECONSTRAINTPROJECTION_HPP_
