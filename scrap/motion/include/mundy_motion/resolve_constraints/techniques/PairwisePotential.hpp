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

#ifndef MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_PAIRWISEPOTENTIAL_HPP_
#define MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_PAIRWISEPOTENTIAL_HPP_

/// \file PairwisePotential.hpp
/// \brief Declaration of the PairwisePotential class

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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>    // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                            // for mundy::meta::PartReqs

namespace mundy {

namespace motion {

namespace resolve_constraints {

namespace techniques {

/// \class PairwisePotential
/// \brief Method for resolving constraints using pairwise potentials.
///
/// The methodology behind the design choices in this class is as follows:
/// The \c PairwisePotential class is a \c MetaTwoWayMethod that is responsible for resolving constraints using pairwise
/// potentials, where each potential is a \c MetaTwoWayKernel with a well-defined source and target set of particles.
///
/// A simple example would be a system consisting of two sphere parts (named "A" and "B") with a no-overlap constraint
/// imposed by a repulsive potential. Additionally, spheres in part A are "sticky" and are attracted to spheres in part
/// B via an attractive potential. That is,
///  - A-A interactions are repulsive
///  - B-B interactions are repulsive
///  - A-B (and B-A) interactions are repulsive + attractive
/// This can be accomplished by defining two symmetric \c MetaTwoWayKernels:
/// - \c SPHERE_SPHERE_SYMMETRIC_REPULSIVE_POTENTIAL: A-A and B-B source-target pairs
/// - \c SPHERE_SPHERE_SYMMETRIC_ATTRACTIVE_POTENTIAL: A-B and B-A source-target pairs
/// Once these kernels are defined and registered with \c OurTwoWayKernelFactory, the \c PairwisePotential class can
/// be used to evaluate the potentials using the following fixed_params \c ParameterList:
/// \code{.yaml}
/// source_target_pairs:
///   count: 3
///   source_target_pair_0:
///     source_part: "A"
///     target_part: "A"
///     symmetric: true
///     kernel_name: "SPHERE_SPHERE_SYMMETRIC_REPULSIVE_POTENTIAL"
///   source_target_pair_1:
///     source_part: "B"
///     target_part: "B"
///     symmetric: true
///     kernel_name: "SPHERE_SPHERE_SYMMETRIC_REPULSIVE_POTENTIAL"
///   source_target_pair_2:
///     source_part: "A"
///     target_part: "B"
///     symmetric: true
///     kernel_name: "SPHERE_SPHERE_SYMMETRIC_ATTRACTIVE_POTENTIAL"
/// \endcode
/// Note, the \c symmetric flag is used to indicate that the kernels should use a symmetric neighbor list. Accounting
/// for symmetric interactions reducing the number of kernel evaluations by a factor of two, but needs to be done with
/// care. This is because writing to the target particles can induce race conditions. As a result, if the \c symmetric
/// flag is set to true, the kernel should employ atomic operations when writing to the target particles. Even if the
/// \c symmetric flag is set to false, use caution when the source and target particles are the same. In this case, the
/// kernel should not rely on fields that are being written to. If this becomes an issue, consider using multiple field
/// states such that the field being read and written to are different.
///
/// At runtime, the \c execute function should be passed a vector of source-target pairs containing subsets of the
/// desired parts. In the above example, the vector should contain three pairs of subsets (subA1-subA2, subB1-subB2, and
/// subA3-subB3).
class PairwisePotential : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurTwoWayKernelFactory = mundy::meta::MetaTwoWayKernelFactory<void, PairwisePotential>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  PairwisePotential() = delete;

  /// \brief Constructor
  PairwisePotential(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    Teuchos::ParameterList &compute_pairwise_potential_params =
        valid_fixed_params.sublist("subkernels").sublist("compute_pairwise_potential");
    const std::string compute_pairwise_potential_name = compute_pairwise_potential_params.get<std::string>("name");

    return OurTwoWayKernelFactory::get_mesh_requirements(compute_pairwise_potential_name,
                                                         compute_pairwise_potential_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    Teuchos::ParameterList &compute_pairwise_potential_params =
        fixed_params_ptr->sublist("subkernels", false).sublist("compute_pairwise_potential", false);

    if (compute_pairwise_potential_params.isParameter("name")) {
      const bool valid_type = compute_pairwise_potential_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "PairwisePotential: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_pairwise_potential_params.set("name", std::string(default_compute_pairwise_potential_name_),
                                            "Name of the method for computing the pairwise potential.");
    }
    const std::string compute_pairwise_potential_name = compute_pairwise_potential_params.get<std::string>("name");

    // Validate the fixed parameters of the subkernels.
    OurTwoWayKernelFactory::validate_fixed_parameters_and_set_defaults(compute_pairwise_potential_name,
                                                                       &compute_pairwise_potential_params);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    Teuchos::ParameterList &compute_pairwise_potential_params =
        mutable_params_ptr->sublist("subkernels", false).sublist("compute_pairwise_potential", false);

    if (compute_pairwise_potential_params.isParameter("name")) {
      const bool valid_type = compute_pairwise_potential_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "PairwisePotential: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
              "with a type other than std::string");
    } else {
      compute_pairwise_potential_params.set("name", std::string(default_compute_pairwise_potential_name_),
                                            "Name of the method for computing the pairwise potential.");
    }
    const std::string compute_pairwise_potential_name = compute_pairwise_potential_params.get<std::string>("name");

    // Validate the fixed parameters of the subkernels.
    OurTwoWayKernelFactory::validate_mutable_parameters_and_set_defaults(compute_pairwise_potential_name,
                                                                         &compute_pairwise_potential_params);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<PairwisePotential>(bulk_data_ptr, fixed_params);
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

  static constexpr std::string_view default_compute_pairwise_potential_name_ = "COMPUTE_PAIRWISE_POTENTIAL";
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

  /// \brief Method for computing the actual pairwise potential.
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_pairwise_potential_method_ptr_;
  //@}
};  // PairwisePotential

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace motion

}  // namespace mundy

#endif  // MUNDY_MOTION_RESOLVE_CONSTRAINTS_TECHNIQUES_PAIRWISEPOTENTIAL_HPP_
