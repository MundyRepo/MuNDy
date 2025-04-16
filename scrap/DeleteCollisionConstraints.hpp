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

#ifndef MUNDY_CONSTRAINTS_DELETECOLLISIONCONSTRAINTS_HPP_
#define MUNDY_CONSTRAINTS_DELETECOLLISIONCONSTRAINTS_HPP_

/// \file DeleteCollisionConstraints.hpp
/// \brief Declaration of the DeleteCollisionConstraints class

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
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>  // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>        // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace constraints {

/// \class DeleteCollisionConstraints
/// \brief Method for deleting unnecessary collision constraints.
class DeleteCollisionConstraints : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, DeleteCollisionConstraints>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  DeleteCollisionConstraints() = delete;

  /// \brief Constructor
  DeleteCollisionConstraints(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels");
    const int num_specified_kernels = kernels_sublist.get<int>("count");

    auto mesh_requirements_ptr = std::make_shared<mundy::meta::MeshReqs>();
    for (int i = 0; i < num_specified_kernels; i++) {
      Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
      const std::string kernel_name = kernel_params.get<std::string>("name");
      mesh_requirements_ptr->sync(OurKernelFactory::get_mesh_requirements(kernel_name, kernel_params));
    }

    return mesh_requirements_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    Teuchos::ParameterList params = *fixed_params_ptr;

    if (params.isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernels", true);
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernels", false);
      const unsigned num_specified_kernels = OurKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", true);
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurKernelFactory::validate_mutable_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", false);
      const unsigned num_specified_kernels = OurKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurKernelFactory::validate_mutable_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DeleteCollisionConstraints>(bulk_data_ptr, fixed_params);
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
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "GENERATE_COLLISION_CONSTRAINTS";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of active multibody types.
  size_t num_multibody_types_ = 0;

  /// \brief Vector of pointers to the active multibody parts this class acts upon.
  std::vector<stk::mesh::Part *> multibody_part_ptr_vector_;

  /// \brief Vector of kernels, one for each active multibody part.
  std::vector<std::shared_ptr<mundy::meta::MetaKernel<>>> multibody_kernel_ptrs_;

  /// \brief The set of neighbor pairs
  std::shared_ptr<IdentProcPairVector> old_neighbor_pairs_ptr_;

  std::shared_ptr<IdentProcPairVector> current_neighbor_pairs_ptr_;

  //@}
};  // DeleteCollisionConstraints

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_DELETECOLLISIONCONSTRAINTS_HPP_
