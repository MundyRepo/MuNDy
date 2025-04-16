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

#ifndef MUNDY_META_KERNELDISPATCHER_HPP_
#define MUNDY_META_KERNELDISPATCHER_HPP_

/// \file MetaKernelDispatcher.hpp
/// \brief Declaration of the MetaKernelDispatcher class

// C++ core lib
#include <algorithm>  // for std::transform
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>       // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>    // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>     // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace meta {

template <typename T>
concept HasGetValidForwardedKernelFixedParams = requires(T t) {
  { T::get_valid_forwarded_kernel_fixed_params() } -> std::same_as<Teuchos::ParameterList>;
};  // HasGetValidForwardedKernelFixedParams

template <typename T>
concept HasGetValidForwardedKernelMutableParams = requires(T t) {
  { T::get_valid_forwarded_kernel_mutable_params() } -> std::same_as<Teuchos::ParameterList>;
};  // HasGetValidForwardedKernelMutableParams

template <typename T>
concept HasGetValidRequiredKernelFixedParams = requires(T t) {
  { T::get_valid_required_kernel_fixed_params() } -> std::same_as<Teuchos::ParameterList>;
};  // HasGetValidRequiredKernelFixedParams

template <typename T>
concept HasGetValidRequiredKernelMutableParams = requires(T t) {
  { T::get_valid_required_kernel_mutable_params() } -> std::same_as<Teuchos::ParameterList>;
};  // HasGetValidRequiredKernelMutableParams

/// \brief A helper class for defining MetaMethods that dispatch multiple MetaKernels and constrain their fixed/mutable
/// params.
///
/// Use this class to create a collection of kernels that act on the same entity rank, constrain the fixed/mutable
/// params of those kernels, and then execute them all in a single call.
///
/// This class is a concrete implementation of the \c MetaMethodSubsetExecutionInterface interface. It provides a means
/// of dispatching a collection of kernels (each associated with potentially different parts) on a subset of the
/// entities in the mesh. In a sense, it's role is to streamline the registration of collections of kernels and to
/// provide a means of executing them in a single call. It uses subsetting and part unions to properly dispatch the
/// kernels on the correct entities AND (importantly) to guarantee that a kernel only acts on an entity once even if
/// that entity is in multiple valid parts.
///
/// \note All kernels within a MetaKernelDispatcher must act on entities of the same rank. Otherwise, it's not clear how
/// to properly dispatch the kernels in a way that guarantees that each entity is only acted upon once. For example, a
/// user might erroneously create a kernel that acts on a ELEMENT_RANKED entities in a PARTICLE topology part and
/// another that acts on NODE_RANK entities in the same part. If the NODE_RANK entity kernel changes the state of the
/// elements, then the order of the kernels will matter.
///
/// The DerivedType only has four required methods. It must contain a get_valid_forwarded_kernel_fixed_params() and a
/// get_valid_forwarded_kernel_mutable_params(); these methods specify the fixed and mutable parameters that
/// the method will accept and forward to all kernels. It must also contain a get_valid_required_kernel_fixed_params()
/// and a get_valid_required_kernel_mutable_params(); these methods specify the fixed and mutable parameters that the
/// method requires all kernels to have. As such, we throw if the kernels don't have a forwarded or required parameter
/// in its valid params
template <typename DerivedType,
          mundy::meta::RegistrationStringValueWrapper kernel_factory_registration_string_value_wrapper>
// requires HasGetValidForwardedKernelFixedParams<DerivedType> && HasGetValidForwardedKernelMutableParams<DerivedType>
// && HasGetValidRequiredKernelFixedParams<DerivedType> && HasGetValidRequiredKernelMutableParams<DerivedType>
class MetaKernelDispatcher : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurKernelFactory =
      mundy::meta::StringBasedMetaFactory<mundy::meta::MetaKernel<>, kernel_factory_registration_string_value_wrapper>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaKernelDispatcher() = delete;

  /// \brief Constructor
  MetaKernelDispatcher(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(
        MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::get_valid_fixed_params());

    // At this point, the only parameters are the set of enabled kernels, the forwarded parameters for the kernels, and
    // the non-forwarded kernel params within the kernel sublists. We'll loop over all parameters that aren't in the
    // kernel sublists and forward them to the kernels.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_kernel_names") {
        for (int j = 0; j < OurKernelFactory::num_registered_classes(); j++) {
          const std::string kernel_name = std::string(OurKernelFactory::get_keys()[j]);
          Teuchos::ParameterList &kernel_params = valid_fixed_params.sublist(kernel_name);
          kernel_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Get the requirements for each kernel and sync them.
    Teuchos::Array<std::string> enabled_kernel_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("enabled_kernel_names");
    auto mesh_requirements_ptr = std::make_shared<mundy::meta::MeshReqs>();
    for (const std::string &kernel_name : enabled_kernel_names) {
      Teuchos::ParameterList &kernel_params = valid_fixed_params.sublist(kernel_name);
      mesh_requirements_ptr->sync(OurKernelFactory::get_mesh_requirements(kernel_name, kernel_params));
    }

    return mesh_requirements_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList parameter_list;

    // Teuchos expects an array of strings for the enabled_kernel_names parameter.
    std::vector<std::string> default_kernel_names = OurKernelFactory::get_keys();
    Teuchos::Array<std::string> default_array_of_kernel_names(default_kernel_names);
    std::string valid_kernel_names = OurKernelFactory::get_keys_as_string();
    parameter_list.set("enabled_kernel_names", default_array_of_kernel_names,
                       "The names of all kernels to enable. Valid kernel names are: " + valid_kernel_names);

    const Teuchos::ParameterList valid_required_kernel_fixed_params =
        DerivedType::get_valid_required_kernel_fixed_params();

    static Teuchos::ParameterList valid_forwarded_kernel_fixed_params =
        DerivedType::get_valid_forwarded_kernel_fixed_params();

    return add_valid_enabled_kernels_and_kernel_params_to_parameter_list(
        "fixed", parameter_list, valid_required_kernel_fixed_params, valid_forwarded_kernel_fixed_params,
        [](const std::string &kernel_name) { return OurKernelFactory::get_valid_fixed_params(kernel_name); });
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList parameter_list;

    const Teuchos::ParameterList valid_required_kernel_mutable_params =
        DerivedType::get_valid_required_kernel_mutable_params();

    static Teuchos::ParameterList valid_forwarded_kernel_mutable_params =
        DerivedType::get_valid_forwarded_kernel_mutable_params();

    return add_valid_enabled_kernels_and_kernel_params_to_parameter_list(
        "mutable", parameter_list, valid_required_kernel_mutable_params, valid_forwarded_kernel_mutable_params,
        [](const std::string &kernel_name) { return OurKernelFactory::get_valid_mutable_params(kernel_name); });
  }

  /// \brief Get the unique registration identifier associated with our kernel factory.
  static std::string get_kernel_factory_registration_id() {
    return kernel_factory_registration_string_value_wrapper.to_string();
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DerivedType>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of active kernels.
  int num_active_kernels_ = 0;

  /// \brief The names of the enabled kernels.
  Teuchos::Array<std::string> enabled_kernel_names_;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief Vector of kernels, one for each active kernel.
  std::vector<std::shared_ptr<mundy::meta::MetaKernel<>>> kernel_ptrs_;
  //@}

  //! \name Internal methods
  //@{

  /// \brief Get the valid enabled kernels and their parameters.
  /// \param get_kernel_params_func [in] A function that returns the valid parameters for a kernel given its name
  static Teuchos::ParameterList add_valid_enabled_kernels_and_kernel_params_to_parameter_list(
      const std::string &parameter_list_name, Teuchos::ParameterList &parameter_list_to_add_to,
      const Teuchos::ParameterList &required_parameter_list, const Teuchos::ParameterList &forwarded_parameter_list,
      const std::function<Teuchos::ParameterList(const std::string &)> &get_kernel_params_func) {
    parameter_list_to_add_to.setParameters(forwarded_parameter_list);

    // Because this is the valid params we list ALL possible parameters. We expect the parameters for
    // each kernel to be a sublist of this list with the same name as the kernel.
    for (auto &key : OurKernelFactory::get_keys()) {
      std::string valid_kernel_name = key;
      Teuchos::ParameterList &kernel_params = parameter_list_to_add_to.sublist(valid_kernel_name);
      kernel_params.setParameters(get_kernel_params_func(valid_kernel_name));

      // Check that the forwarded params exist and then remove them from the valid params since the
      // user isn't responsible for setting them.
      for (Teuchos::ParameterList::ConstIterator i = forwarded_parameter_list.begin();
           i != forwarded_parameter_list.end(); i++) {
        const std::string &forwarded_parameter_name = forwarded_parameter_list.name(i);

        MUNDY_THROW_REQUIRE(kernel_params.isParameter(forwarded_parameter_name), std::logic_error,
                            std::string("MetaKernelDispatcher: The kernel ") + valid_kernel_name +
                                " does not have the required (forwarded) parameter '" + forwarded_parameter_name +
                                "' in its valid " + parameter_list_name + " params.");
        kernel_params.remove(forwarded_parameter_name);
      }

      // Check that the required params exist but do not remove them, as the user is responsible for setting them.
      for (Teuchos::ParameterList::ConstIterator i = required_parameter_list.begin();
           i != required_parameter_list.end(); i++) {
        const std::string &required_parameter_name = required_parameter_list.name(i);

        MUNDY_THROW_REQUIRE(kernel_params.isParameter(required_parameter_name), std::logic_error,
                            std::string("MetaKernelDispatcher: The kernel ") + valid_kernel_name +
                                " does not have the required (required) parameter '" + required_parameter_name +
                                "' in its valid " + parameter_list_name + " params.");
      }
    }

    return parameter_list_to_add_to;
  }
  //@}
};  // MetaKernelDispatcher

//! \name Template specializations
//@{

// \name Constructors and destructor
//{

template <typename DerivedType,
          mundy::meta::RegistrationStringValueWrapper kernel_factory_registration_string_value_wrapper>
MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::MetaKernelDispatcher(
    mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "MetaKernelDispatcher: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(
      MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::get_valid_fixed_params());

  // Populate our internal members.
  enabled_kernel_names_ = valid_fixed_params.get<Teuchos::Array<std::string>>("enabled_kernel_names");
  num_active_kernels_ = static_cast<int>(enabled_kernel_names_.size());
  if (num_active_kernels_ > 0) {
    kernel_ptrs_.reserve(num_active_kernels_);
    valid_entity_parts_.reserve(num_active_kernels_);
    for (int i = 0; i < num_active_kernels_; i++) {
      // Create the kernel and store it in the kernel_ptrs_ vector.
      const std::string kernel_name = enabled_kernel_names_[i];
      Teuchos::ParameterList &kernel_params = valid_fixed_params.sublist(kernel_name);

      // At this point, the only parameters are the set of enabled kernels, the forwarded parameters for the kernels,
      // and the non-forwarded kernel params within the kernel sublists. We'll loop over all parameters that aren't in
      // the kernel sublists and forward them to the current kernel.
      for (Teuchos::ParameterList::ConstIterator j = valid_fixed_params.begin(); j != valid_fixed_params.end(); j++) {
        const std::string &param_name = valid_fixed_params.name(j);
        const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
        if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_kernel_names") {
          kernel_params.setEntry(param_name, param_entry);
        }
      }

      kernel_ptrs_.push_back(OurKernelFactory::create_new_instance(kernel_name, bulk_data_ptr_, kernel_params));

      // Get the valid entity parts for the kernel.
      auto valid_entity_parts_i = kernel_ptrs_[i]->get_valid_entity_parts();
      valid_entity_parts_.insert(valid_entity_parts_.end(), valid_entity_parts_i.begin(), valid_entity_parts_i.end());
    }
  }
}
//}

// \name MetaFactory static interface implementation
//{

template <typename DerivedType,
          mundy::meta::RegistrationStringValueWrapper kernel_factory_registration_string_value_wrapper>
void MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::set_mutable_params(
    const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(
      MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::get_valid_mutable_params());

  // Populate our internal members.
  for (int i = 0; i < num_active_kernels_; i++) {
    // Create the kernel and store it in the kernel_ptrs_ vector.
    const std::string kernel_name = enabled_kernel_names_[i];
    Teuchos::ParameterList &kernel_params = valid_mutable_params.sublist(kernel_name);

    // At this point, the only parameters are the set of enabled kernels, the forwarded parameters for the kernels,
    // and the non-forwarded kernel params within the kernel sublists. We'll loop over all parameters that aren't in
    // the kernel sublists and forward them to the current kernel.
    for (Teuchos::ParameterList::ConstIterator j = valid_mutable_params.begin(); j != valid_mutable_params.end(); j++) {
      const std::string &param_name = valid_mutable_params.name(j);
      const Teuchos::ParameterEntry &param_entry = valid_mutable_params.getEntry(param_name);
      if (!valid_mutable_params.isSublist(param_name) && param_name != "enabled_kernel_names") {
        kernel_params.setEntry(param_name, param_entry);
      }
    }

    kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
}
//}

// \name Getters
//{

template <typename DerivedType,
          mundy::meta::RegistrationStringValueWrapper kernel_factory_registration_string_value_wrapper>
std::vector<stk::mesh::Part *>
MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::get_valid_entity_parts() const {
  return valid_entity_parts_;
}
//}

// \name Actions
//{

template <typename DerivedType,
          mundy::meta::RegistrationStringValueWrapper kernel_factory_registration_string_value_wrapper>
void MetaKernelDispatcher<DerivedType, kernel_factory_registration_string_value_wrapper>::execute(
    const stk::mesh::Selector &input_selector) {
  const bool is_safe_to_loop_over_entities = bulk_data_ptr_->in_synchronized_state();
  MUNDY_THROW_REQUIRE(
      is_safe_to_loop_over_entities, std::logic_error,
      "MetaKernelDispatcher: The bulk data must NOT be in a modification cycle to execute kernels. "
      "Kernel dispatcher loops over entities in a way that assumes bucket stability, but that need not "
      "be the case during a modification cycle. For example, changing entity parts within a kernel "
      "could cause the entity to change buckets. This is a potentially dangerous operation, so we forbid it.");

  for (int i = 0; i < num_active_kernels_; i++) {
    kernel_ptrs_[i]->execute(input_selector);
  }
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_KERNELDISPATCHER_HPP_
