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

#ifndef UNIT_TESTS_MUNDY_META_UTILS_EXAMPLEMETAMETHOD_HPP_
#define UNIT_TESTS_MUNDY_META_UTILS_EXAMPLEMETAMETHOD_HPP_

/// \file ExampleMetaMethod.hpp
/// \brief Declaration of the ExampleMetaMethod class

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
#include <mundy_meta/MeshReqs.hpp>      // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>    // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>  // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace meta {

namespace utils {

/// \class ExampleMetaMethod
/// \brief Method for computing the axis aligned boundary box of different parts.
///
/// \tparam registration_string_wrapper [in] A unique registration identifier associated with our kernel and method
/// factories.
template <mundy::meta::RegistrationStringValueWrapper registration_string_wrapper>
class ExampleMetaMethod : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurKernelFactory = mundy::meta::StringBasedMetaFactory<mundy::meta::MetaKernel<>, registration_string_wrapper>;
  using OurMethodFactory = mundy::meta::StringBasedMetaFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>,
                                                               registration_string_wrapper>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ExampleMetaMethod() = delete;

  /// \brief Constructor
  ExampleMetaMethod([[maybe_unused]] mundy::mesh::BulkData *const bulk_data_ptr,
                    [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Typically, we would use the following line to check that the bulk data pointer is not a nullptr.
    // However, in this case, we want to be able to allow a nullptr bulk data to simplify testing.
    // MUNDY_THROW_REQUIRE(bulk_data_ptr != nullptr, "ExampleMetaMethod: bulk_data_ptr must not be a nullptr.");
  }
  //@}

  //! \name Testing counters
  //@{

  /// \brief Get the number of times that \c get_mesh_requirements has been called.
  static int num_get_mesh_requirements_calls() {
    return get_mesh_requirements_counter_;
  }

  /// \brief Get the number of times that \c get_valid_fixed_params has been called.
  static int num_get_valid_fixed_params_calls() {
    return get_valid_fixed_params_counter_;
  }

  /// \brief Get the number of times that \c get_valid_mutable_params has been called.
  static int num_get_valid_mutable_params_calls() {
    return get_valid_mutable_params_counter_;
  }

  /// \brief Get the number of times that \c create_new_instance has been called.
  static int num_create_new_instance_calls() {
    return create_new_instance_counter_;
  }

  /// \brief Get the number of times that \c set_mutable_params has been called.
  static int num_set_mutable_params_calls() {
    return set_mutable_params_counter_;
  }

  /// \brief Get the number of times that \c execute has been called.
  static int num_execute_calls() {
    return execute_counter_;
  }

  /// \brief Reset all counters to zero.
  static void reset_counters() {
    get_mesh_requirements_counter_ = 0;
    get_valid_fixed_params_counter_ = 0;
    get_valid_mutable_params_counter_ = 0;
    create_new_instance_counter_ = 0;
    set_mutable_params_counter_ = 0;
    execute_counter_ = 0;
  }

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
    get_mesh_requirements_counter_++;
    std::shared_ptr<mundy::meta::MeshReqs> mesh_requirements_ptr;
    return mesh_requirements_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static Teuchos::ParameterList get_valid_fixed_params() {
    get_valid_fixed_params_counter_++;
    return Teuchos::ParameterList();
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static Teuchos::ParameterList get_valid_mutable_params() {
    get_valid_mutable_params_counter_++;
    return Teuchos::ParameterList();
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    create_new_instance_counter_++;
    return std::make_shared<ExampleMetaMethod<registration_string_wrapper>>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) override {
    set_mutable_params_counter_++;
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute([[maybe_unused]] const stk::mesh::Selector &input_selector) override {
    execute_counter_++;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The number of times \c get_mesh_requirements has been called.
  static inline int get_mesh_requirements_counter_ = 0;

  /// \brief The number of times \c get_valid_fixed_params has been called.
  static inline int get_valid_fixed_params_counter_ = 0;

  /// \brief The number of times \c get_valid_mutable_params has been called.
  static inline int get_valid_mutable_params_counter_ = 0;

  /// \brief The number of times \c create_new_instance has been called.
  static inline int create_new_instance_counter_ = 0;

  /// \brief The number of times \c set_mutable_params has been called.
  static inline int set_mutable_params_counter_ = 0;

  /// \brief The number of times \c execute has been called.
  static inline int execute_counter_ = 0;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;
  //@}
};  // ExampleMetaMethod

using DefaultExampleMetaMethod = ExampleMetaMethod<mundy::meta::make_registration_string("DEFAULT")>;

}  // namespace utils

}  // namespace meta

}  // namespace mundy

#endif  // UNIT_TESTS_MUNDY_META_UTILS_EXAMPLEMETAMETHOD_HPP_
