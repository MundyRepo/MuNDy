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

#ifndef MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_
#define MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_

/// \file NonSmoothLCP.hpp
/// \brief Declaration of the NonSmoothLCP class

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
#include <mundy_constraints/ComputeConstraintForcing.hpp>     // for mundy::constraint::ComputeConstraintForcing
#include <mundy_constraints/ComputeConstraintProjection.hpp>  // for mundy::constraint::ComputeConstraintProjection
#include <mundy_constraints/ComputeConstraintResidual.hpp>    // for mundy::constraint::ComputeConstraintResidual
#include <mundy_constraints/ComputeConstraintViolation.hpp>   // for mundy::constraint::ComputeConstraintViolation
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                            // for mundy::meta::PartReqs

namespace mundy {

namespace motion {

namespace resolve_constraints {

namespace techniques {

/// \class NonSmoothLCP
/// \brief Method for resolving constraints using a non-smooth linear complementarity problem (LCP) formulation.
class NonSmoothLCP : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  // TODO(palmerb4): MetaMethodFactory should be able to spawn multiple factories for a single class.
  // Below, we see why this is necessary. Even though we have different aliases for the various factories,
  // three of them are actually the same type. To me, this tells me that RegistryIdentifier_t should be replaced with
  // compile-time strings via mundy::core::Stringliteral.

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurConstraintForcingMethodFactory = mundy::meta::MetaMethodFactory<void, NonSmoothLCP>;
  using OurConstraintProjectionMethodFactory = mundy::meta::MetaMethodFactory<void, NonSmoothLCP>;
  using OurConstraintResidualMethodFactory = mundy::meta::MetaMethodFactory<double, NonSmoothLCP>;
  using OurConstraintViolationMethodFactory = mundy::meta::MetaMethodFactory<void, NonSmoothLCP>;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NonSmoothLCP() = delete;

  /// \brief Constructor
  NonSmoothLCP(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    Teuchos::ParameterList &compute_constraint_forcing_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_forcing");
    Teuchos::ParameterList &compute_constraint_projection_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_projection");
    Teuchos::ParameterList &compute_constraint_residual_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_residual");
    Teuchos::ParameterList &compute_constraint_violation_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_violation");

    // Collect and sync the submethod requirements.
    auto mesh_reqs = std::make_shared<mundy::meta::MeshReqs>();
    const std::string compute_constraint_forcing_name = compute_constraint_forcing_params.get<std::string>("name");
    const std::string compute_constraint_projection_name =
        compute_constraint_projection_params.get<std::string>("name");
    const std::string compute_constraint_residual_name = compute_constraint_residual_params.get<std::string>("name");
    const std::string compute_constraint_violation_name = compute_constraint_violation_params.get<std::string>("name");
    mesh_reqs->sync(OurConstraintForcingMethodFactory::get_mesh_requirements(compute_constraint_forcing_name,
                                                                             compute_constraint_forcing_params));
    mesh_reqs->sync(OurConstraintProjectionMethodFactory::get_mesh_requirements(compute_constraint_projection_name,
                                                                                compute_constraint_projection_params));
    mesh_reqs->sync(OurConstraintResidualMethodFactory::get_mesh_requirements(compute_constraint_residual_name,
                                                                              compute_constraint_residual_params));
    mesh_reqs->sync(OurConstraintViolationMethodFactory::get_mesh_requirements(compute_constraint_violation_name,
                                                                               compute_constraint_violation_params));
    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    Teuchos::ParameterList &compute_constraint_forcing_params =
        fixed_params_ptr->sublist("submethods", false).sublist("compute_constraint_forcing", false);
    Teuchos::ParameterList &compute_constraint_projection_params =
        fixed_params_ptr->sublist("submethods", false).sublist("compute_constraint_projection", false);
    Teuchos::ParameterList &compute_constraint_residual_params =
        fixed_params_ptr->sublist("submethods", false).sublist("compute_constraint_residual", false);
    Teuchos::ParameterList &compute_constraint_violation_params =
        fixed_params_ptr->sublist("submethods", false).sublist("compute_constraint_violation", false);

    if (compute_constraint_forcing_params.isParameter("name")) {
      const bool valid_type = compute_constraint_forcing_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
                             "with a type other than std::string");
    } else {
      compute_constraint_forcing_params.set(
          "name", std::string(default_compute_constraint_forcing_name_),
          "Name of the method for computing the force induced by the constraints on their nodes.");
    }

    if (compute_constraint_projection_params.isParameter("name")) {
      const bool valid_type =
          compute_constraint_projection_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_projection parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_constraint_projection_params.set(
          "name", std::string(default_compute_constraint_projection_name_),
          "Name of the method for projecting the constraints onto the feasible solution space.");
    }

    if (compute_constraint_residual_params.isParameter("name")) {
      const bool valid_type = compute_constraint_residual_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a compute_constraint_residual parameter with name 'name' but "
                             "with a type other than std::string");
    } else {
      compute_constraint_residual_params.set(
          "name", std::string(default_compute_constraint_residual_name_),
          "Name of the method for computing the global residual quantifying the constraint violation.");
    }

    if (compute_constraint_violation_params.isParameter("name")) {
      const bool valid_type =
          compute_constraint_violation_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_violation parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_constraint_violation_params.set(
          "name", std::string(default_compute_constraint_violation_name_),
          "Name of the method for computing the amount of violation of all constraints.");
    }

    const std::string compute_constraint_forcing_name = compute_constraint_forcing_params.get<std::string>("name");
    const std::string compute_constraint_projection_name =
        compute_constraint_projection_params.get<std::string>("name");
    const std::string compute_constraint_residual_name = compute_constraint_residual_params.get<std::string>("name");
    const std::string compute_constraint_violation_name = compute_constraint_violation_params.get<std::string>("name");

    // Validate the fixed parameters of the submethods.
    OurConstraintForcingMethodFactory::validate_fixed_parameters_and_set_defaults(compute_constraint_forcing_name,
                                                                                  &compute_constraint_forcing_params);
    OurConstraintProjectionMethodFactory::validate_fixed_parameters_and_set_defaults(
        compute_constraint_projection_name, &compute_constraint_projection_params);
    OurConstraintResidualMethodFactory::validate_fixed_parameters_and_set_defaults(compute_constraint_residual_name,
                                                                                   &compute_constraint_residual_params);
    OurConstraintViolationMethodFactory::validate_fixed_parameters_and_set_defaults(
        compute_constraint_violation_name, &compute_constraint_violation_params);

    // Validate the fixed parameters of this method itself.
    if (fixed_params_ptr->isParameter("element_constraint_violation_field")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_constraint_violation_field");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a parameter with name 'element_constraint_violation_field' but "
              "with a type other than std::string");
    } else {
      fixed_params_ptr->set(
          "element_constraint_violation_field", std::string(default_element_constraint_violation_field_name_),
          "Name of the element field containing the constraint violation measure for each constraint.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    Teuchos::ParameterList &compute_constraint_forcing_params =
        mutable_params_ptr->sublist("submethods", false).sublist("compute_constraint_forcing", false);
    Teuchos::ParameterList &compute_constraint_projection_params =
        mutable_params_ptr->sublist("submethods", false).sublist("compute_constraint_projection", false);
    Teuchos::ParameterList &compute_constraint_residual_params =
        mutable_params_ptr->sublist("submethods", false).sublist("compute_constraint_residual", false);
    Teuchos::ParameterList &compute_constraint_violation_params =
        mutable_params_ptr->sublist("submethods", false).sublist("compute_constraint_violation", false);

    if (compute_constraint_forcing_params.isParameter("name")) {
      const bool valid_type = compute_constraint_forcing_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
                             "with a type other than std::string");
    } else {
      compute_constraint_forcing_params.set(
          "name", std::string(default_compute_constraint_forcing_name_),
          "Name of the method for computing the force induced by the constraints on their nodes.");
    }

    if (compute_constraint_projection_params.isParameter("name")) {
      const bool valid_type =
          compute_constraint_projection_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_projection parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_constraint_projection_params.set(
          "name", std::string(default_compute_constraint_projection_name_),
          "Name of the method for projecting the constraints onto the feasible solution space.");
    }

    if (compute_constraint_residual_params.isParameter("name")) {
      const bool valid_type = compute_constraint_residual_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a compute_constraint_residual parameter with name 'name' but "
                             "with a type other than std::string");
    } else {
      compute_constraint_residual_params.set(
          "name", std::string(default_compute_constraint_residual_name_),
          "Name of the method for computing the global residual quantifying the constraint violation.");
    }

    if (compute_constraint_violation_params.isParameter("name")) {
      const bool valid_type =
          compute_constraint_violation_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_violation parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_constraint_violation_params.set(
          "name", std::string(default_compute_constraint_violation_name_),
          "Name of the method for computing the amount of violation of all constraints.");
    }

    const std::string compute_constraint_forcing_name = compute_constraint_forcing_params.get<std::string>("name");
    const std::string compute_constraint_projection_name =
        compute_constraint_projection_params.get<std::string>("name");
    const std::string compute_constraint_residual_name = compute_constraint_residual_params.get<std::string>("name");
    const std::string compute_constraint_violation_name = compute_constraint_violation_params.get<std::string>("name");

    // Validate the mutable parameters of the submethods.
    OurConstraintForcingMethodFactory::validate_mutable_parameters_and_set_defaults(compute_constraint_forcing_name,
                                                                                    &compute_constraint_forcing_params);
    OurConstraintProjectionMethodFactory::validate_mutable_parameters_and_set_defaults(
        compute_constraint_projection_name, &compute_constraint_projection_params);
    OurConstraintResidualMethodFactory::validate_mutable_parameters_and_set_defaults(
        compute_constraint_residual_name, &compute_constraint_residual_params);
    OurConstraintViolationMethodFactory::validate_mutable_parameters_and_set_defaults(
        compute_constraint_violation_name, &compute_constraint_violation_params);

    // Validate the mutable parameters of this method itself.
    if (mutable_params_ptr->isParameter("max_num_iterations")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("max_num_iterations");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a parameter with name 'max_num_iterations' but "
                             "with a type other than unsigned");
    } else {
      mutable_params_ptr->set("max_num_iterations", default_max_num_iterations_,
                              "The maximum number of BBPGD iterations to take.");
    }

    if (mutable_params_ptr->isParameter("constraint_residual_tolerance")) {
      const bool valid_type =
          mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("constraint_residual_tolerance");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NonSmoothLCP: Type error. Given a parameter with name 'constraint_residual_tolerance' but "
                             "with a type other than double");
      const bool is_constraint_residual_tolerance_positive = mutable_params_ptr->get<double>("viscosity") > 0;
      MUNDY_THROW_REQUIRE(
          is_constraint_residual_tolerance_positive, std::invalid_argument,
          "NodeEuler: Invalid parameter. Given a parameter with name 'constraint_residual_tolerance' but "
              "with a value less than or equal to zero.");
    } else {
      mutable_params_ptr->set("constraint_residual_tolerance", default_constraint_residual_tolerance_,
                              "The desired tolerance for the constraint violation residual..");
    }
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<NonSmoothLCP>(bulk_data_ptr, fixed_params);
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

  static constexpr std::string_view default_compute_constraint_forcing_name_ = "COMPUTE_CONSTRAINT_FORCING";
  static constexpr std::string_view default_compute_constraint_projection_name_ = "COMPUTE_CONSTRAINT_PROJECTION";
  static constexpr std::string_view default_compute_constraint_residual_name_ = "COMPUTE_CONSTRAINT_RESIDUAL";
  static constexpr std::string_view default_compute_constraint_violation_name_ = "COMPUTE_CONSTRAINT_VIOLATION";
  static constexpr std::string_view default_element_constraint_violation_field_name_ = "ELEMENT_CONSTRAINT_VIOLATION";
  static constexpr int default_max_num_iterations_ = 10000;
  static constexpr double default_constraint_residual_tolerance_ = 1.0e-6;
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "NON_SMOOTH_LCP";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The maximum number of BBPGD iterations to take.
  int max_num_iterations_;

  /// \brief The desired tolerance for the constraint violation residual.
  double constraint_residual_tolerance_;

  /// \brief The name of the element field containing the constraint violation measure for each constraint.
  std::string element_constraint_violation_field_name_;

  /// \brief The element field containing the constraint violation measure for each constraint.
  stk::mesh::Field<double> *element_constraint_violation_field_ptr_ = nullptr;

  /// \brief Method for computing the force induced by the constraints on their nodes.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_forcing_method_ptr_;

  /// \brief Method for projecting the constraints onto the feasible solution space.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_projection_method_ptr_;

  /// \brief Method for computing the global residual quantifying the constraint violation.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<double>> compute_constraint_residual_method_ptr_;

  /// \brief Method for computing the amount of violation of all constraints.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_violation_method_ptr_;
  //@}
};  // NonSmoothLCP

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default constraint forcing method with our method factory.
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_FORCING", mundy::constraint::ComputeConstraintForcing,
    mundy::motion::resolve_constraints::techniques::NonSmoothLCP::OurConstraintForcingMethodFactory)

/// \brief Register our default constraint projection method with our method factory.
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_PROJECTION", mundy::constraint::ComputeConstraintProjection,
    mundy::motion::resolve_constraints::techniques::NonSmoothLCP::OurConstraintProjectionMethodFactory)

/// \brief Register our default constraint residual method with our method factory.
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_RESIDUAL", mundy::constraint::ComputeConstraintResidual,
    mundy::motion::resolve_constraints::techniques::NonSmoothLCP::OurConstraintResidualMethodFactory)

/// \brief Register our default constraint violation method with our method factory.
MUNDY_REGISTER_METACLASS(
    "COMPUTE_CONSTRAINT_VIOLATION", mundy::constraint::ComputeConstraintViolation,
    mundy::motion::resolve_constraints::techniques::NonSmoothLCP::OurConstraintViolationMethodFactory)
//}

#endif  // MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_
