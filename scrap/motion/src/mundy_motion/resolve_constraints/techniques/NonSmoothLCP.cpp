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

/// \file NonSmoothLCP.cpp
/// \brief Definition of the NonSmoothLCP class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MeshReqs.hpp>                            // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_motion/ResolveConstraints.hpp>                // for mundy::motion::ResolveConstraints
#include <mundy_motion/resolve_constraints/techniques/NonSmoothLCP.hpp>  // for mundy::motion::...::NonSmoothLCP

namespace mundy {

namespace motion {

namespace resolve_constraints {

namespace techniques {

// \name Constructors and destructor
//{

NonSmoothLCP::NonSmoothLCP(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "NonSmoothLCP: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fetch the parameters for this part's sub-methods.
  Teuchos::ParameterList &compute_constraint_forcing_params =
      valid_fixed_params.sublist("submethods").sublist("compute_constraint_forcing");
  Teuchos::ParameterList &compute_constraint_projection_params =
      valid_fixed_params.sublist("submethods").sublist("compute_constraint_projection");
  Teuchos::ParameterList &compute_constraint_residual_params =
      valid_fixed_params.sublist("submethods").sublist("compute_constraint_residual");
  Teuchos::ParameterList &compute_constraint_violation_params =
      valid_fixed_params.sublist("submethods").sublist("compute_constraint_violation");

  // Initialize and store the sub-methods.
  const std::string compute_constraint_forcing_name = compute_constraint_forcing_params.get<std::string>("name");
  const std::string compute_constraint_projection_name = compute_constraint_projection_params.get<std::string>("name");
  const std::string compute_constraint_residual_name = compute_constraint_residual_params.get<std::string>("name");
  const std::string compute_constraint_violation_name = compute_constraint_violation_params.get<std::string>("name");
  compute_constraint_forcing_method_ptr_ = OurConstraintForcingMethodFactory::create_new_instance(
      compute_constraint_forcing_name, bulk_data_ptr_, compute_constraint_forcing_params);
  compute_constraint_projection_method_ptr_ = OurConstraintProjectionMethodFactory::create_new_instance(
      compute_constraint_projection_name, bulk_data_ptr_, compute_constraint_projection_params);
  compute_constraint_residual_method_ptr_ = OurConstraintResidualMethodFactory::create_new_instance(
      compute_constraint_residual_name, bulk_data_ptr_, compute_constraint_residual_params);
  compute_constraint_violation_method_ptr_ = OurConstraintViolationMethodFactory::create_new_instance(
      compute_constraint_violation_name, bulk_data_ptr_, compute_constraint_violation_params);

  // Fill the internal members using the valid parameter list.
  element_constraint_violation_field_name_ =
      valid_fixed_params.get<std::string>("element_constraint_violation_field_name");
  element_constraint_violation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_constraint_violation_field_name_);
}
//}

// \name MetaFactory static interface implementation
//{

void NonSmoothLCP::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fetch the parameters for this part's sub-methods.
  Teuchos::ParameterList &compute_constraint_forcing_params =
      valid_mutable_params.sublist("submethods").sublist("compute_constraint_forcing");
  Teuchos::ParameterList &compute_constraint_projection_params =
      valid_mutable_params.sublist("submethods").sublist("compute_constraint_projection");
  Teuchos::ParameterList &compute_constraint_residual_params =
      valid_mutable_params.sublist("submethods").sublist("compute_constraint_residual");
  Teuchos::ParameterList &compute_constraint_violation_params =
      valid_mutable_params.sublist("submethods").sublist("compute_constraint_violation");

  // Set the mutable params for each of our sub-methods.
  compute_constraint_forcing_method_ptr_->set_mutable_params(compute_constraint_forcing_params);
  compute_constraint_projection_method_ptr_->set_mutable_params(compute_constraint_projection_params);
  compute_constraint_residual_method_ptr_->set_mutable_params(compute_constraint_residual_params);
  compute_constraint_violation_method_ptr_->set_mutable_params(compute_constraint_violation_params);

  // Fill the internal members using the valid parameter list.
  constraint_residual_tolerance_ = valid_mutable_params.get<double>("constraint_residual_tolerance");
  max_num_iterations_ = valid_mutable_params.get<int>("max_num_iterations");
}
//}

// \name Actions
//{

void NonSmoothLCP::execute(const stk::mesh::Selector &input_selector) {
  // The following is the BBPGD solution to the linear complementarity problem.

  // Fill the Lagrange multipliers xkm1 with our initial guess. Our choice of initial guess is zero.
  mundy::mesh::for_each_entity_run(*bulk_data_ptr_, stk::topology::ELEMENT_RANK, input_selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
                                   stk::mesh::field_data(*element_constraint_violation_field_ptr_, element)[0] = 0.0;
                                 });

  // Iterate until the residual is below the threshold or until we surpass the maximum number of iterations.
  double alpha;
  int ite_count = 0;
  while (ite_count < max_num_iterations_) {
    if (ite_count > 0) {
      // Take a projected gradient step.
      compute_gradient_step_method_ptr_->execute(input_selector);
      compute_constraint_projection_method_ptr_->execute(input_selector);
    }

    // Compute the new gradient using gk = D^T M D xk. This involves three steps.
    // Step 1: Compute the force induced by each constraint on its nodes.
    compute_constraint_forcing_method_ptr_->execute(input_selector);

    // Step 2: Compute the the velocity of each particle's nodes.
    compute_mobility_method_ptr_->execute(input_selector);

    // Step 3: Map the velocity of each constraint's nodes to that constraint's linearized rate of change of constraint
    // violation.
    compute_linearized_rate_of_change_of_constraint_violation->execute(input_selector);

    // Compute the global constraint residual.
    compute_constraint_violation_method_ptr_->execute(input_selector);
    double constraint_residual = compute_constraint_residual_method_ptr_->execute(input_selector);

    // Check for early termination.
    if (constraint_residual < constraint_residual_tolerance_) {
      // Success, the current set of Lagrange multipliers is correct.
      break;
    }

    // Compute the new Barzilai-Borwein step size.
    if (ite_count == 0) {
      // Initial guess for Barzilai-Borwein step size.
      alpha = 1.0 / constraint_residual;
    } else if (ite_count % 2) {
      // Barzilai-Borwein step size Choice 1.
      alpha = xkdiff_dot_xkdiff / xkdiff_dot_gkdiff;
    } else {
      // Barzilai-Borwein step size Choice 2.
      alpha = xkdiff_dot_gkdiff / gkdiff_dot_gkdiff;
    }
    Teuchos::ParameterList constraint_projection_mutable_params;
    constraint_projection_mutable_params->set("step_size", alpha);
    compute_constraint_projection_method_ptr_->set_mutable_params(constraint_projection_mutable_params);

    // Rotate the state of the xk and gk.
    bulk_data_ptr_->update_field_data_states(element_lagrange_multiplier_field_ptr_);
    bulk_data_ptr_->update_field_data_states(element_constraint_violation_gradient_field_ptr_);
    ite_count++;
  }

  // If the maximum number of iterations is surpassed, we optionally throw an error.
  if (ite_count >= con_ite_max && throw_on_failed_to_converge_) {
    ThrowRequireMsg(false, "NonSmoothLCP: Failed to converge in " << max_num_iterations_
                                                                  << " iterations. \n Current constraint residual is "
                                                                  << constraint_residual);
  }
}
//}

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace motion

}  // namespace mundy
