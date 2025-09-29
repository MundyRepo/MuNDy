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

#ifndef MUNDY_META_METATECHNIQUEDISPATCHER_HPP_
#define MUNDY_META_METATECHNIQUEDISPATCHER_HPP_

/// \file MetaTechniqueDispatcher.hpp
/// \brief Declaration of the MetaTechniqueDispatcher class

// C++ core
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_topology/topology.hpp>        // for stk::topology

// Mundy
#include <mundy_core/StringLiteral.hpp>                               // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                                // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                    // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                                    // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>                                    // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                                  // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace meta {

// TODO(palmerb4): If string lookups turn egregiously slow, we need to modify this class to allow for arbitrary
// registration types, just like MetaFactory. For now, we'll default to using strings as the registration type.

/// \brief A helper class for defining MetaMethods whose only job is to dispatch different techniques for carrying out a
/// task with optional constrains on their fixed/mutable params.
///
/// Use this class to create a collection of techniques (each with their own specialized fixed/mutable params and mesh
/// requirements), constrain the fixed/mutable params of those techniques, and then execute the desired technique.
///
/// The DerivedType has four required methods. It must contain a get_valid_forwarded_technique_fixed_params() and a
/// get_valid_required_forwarded_technique_mutable_params() method; these methods specify the fixed and mutable
/// parameters that the method will accept and forward to all techniques. It must also contain a
/// get_valid_required_technique_fixed_params() and a get_valid_required_technique_mutable_params() method; these
/// methods specify the fixed and mutable parameters that the method will require for each technique. As such, we throw
/// if the technique don't have a forwarded or required parameter in its valid params
template <typename DerivedType, typename DerivedTypePolymorphicBaseType, typename TechniquePolymorphicBaseType,
          mundy::meta::RegistrationStringValueWrapper technique_factory_registration_string_value_wrapper,
          mundy::meta::RegistrationStringValueWrapper default_technique_name_wrapper>
class MetaTechniqueDispatcher {
 public:
  //! \name Typedefs
  //@{

  using OurTechniqueFactory = mundy::meta::StringBasedMetaFactory<TechniquePolymorphicBaseType,
                                                                  technique_factory_registration_string_value_wrapper>;
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
    valid_fixed_params.validateParametersAndSetDefaults(get_valid_fixed_params());

    // The enabled technique name must be in our registry.
    std::string enabled_technique_name = valid_fixed_params.get<std::string>("enabled_technique_name");
    MUNDY_THROW_REQUIRE(
        OurTechniqueFactory::is_valid_key(enabled_technique_name), std::runtime_error,
        std::string("MetaTechniqueDispatcher: The enabled technique name '") + enabled_technique_name +
            "' is not a valid technique name. Valid names are: " + OurTechniqueFactory::get_keys_as_string());

    // At this point, the only parameters are the enabled technique name, the forwarded parameters for the enabled
    // technique, and the non-forwarded technique params within the technique sublists. We'll loop over all parameters
    // that aren't in the technique sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_technique_name") {
        for (int j = 0; j < OurTechniqueFactory::num_registered_classes(); j++) {
          const std::string technique_name = std::string(OurTechniqueFactory::get_keys()[j]);
          Teuchos::ParameterList &technique_params = valid_fixed_params.sublist(technique_name);
          technique_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Fetch the technique sublist and return its parameters.
    const Teuchos::ParameterList &enabled_technique_params = valid_fixed_params.sublist(enabled_technique_name);
    return OurTechniqueFactory::get_mesh_requirements(enabled_technique_name, enabled_technique_params);
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList parameter_list;
    parameter_list.set(
        "enabled_technique_name", default_technique_name_wrapper.value(),
        "The name of the technique to use. Valid names are: " + OurTechniqueFactory::get_keys_as_string());

    static const Teuchos::ParameterList valid_required_technique_fixed_params =
        DerivedType::get_valid_required_technique_fixed_params();

    static const Teuchos::ParameterList valid_forwarded_technique_fixed_params =
        DerivedType::get_valid_forwarded_technique_fixed_params();

    return add_valid_enabled_techniques_and_technique_params_to_parameter_list(
        "fixed", parameter_list, valid_required_technique_fixed_params, valid_forwarded_technique_fixed_params,
        [](const std::string &technique_name) { return OurTechniqueFactory::get_valid_fixed_params(technique_name); });
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList parameter_list;

    static const Teuchos::ParameterList valid_required_technique_mutable_params =
        DerivedType::get_valid_required_technique_mutable_params();

    static const Teuchos::ParameterList valid_forwarded_technique_mutable_params =
        DerivedType::get_valid_forwarded_technique_mutable_params();

    return add_valid_enabled_techniques_and_technique_params_to_parameter_list(
        "mutable", parameter_list, valid_required_technique_mutable_params, valid_forwarded_technique_mutable_params,
        [](const std::string &technique_name) {
          return OurTechniqueFactory::get_valid_mutable_params(technique_name);
        });
  }

  /// \brief Get the unique registration identifier associated with our technique factory.
  static std::string get_technique_factory_registration_id() {
    return technique_factory_registration_string_value_wrapper.to_string();
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<DerivedTypePolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DerivedType>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal methods
  //@{

  /// \brief Get the valid enabled techniques and their parameters.
  /// \param parameter_list_name [in] The name of the parameter list
  /// \param parameter_list_to_add_to [in/out] The parameter list to add the valid enabled techniques and their
  /// parameters to
  /// \param required_parameter_list [in] The required parameters for the techniques
  /// \param forwarded_parameter_list [in] The parameters that are forwarded to the techniques
  /// \param get_technique_params_func [in] A function that returns the valid parameters for a technique given its name
  static Teuchos::ParameterList add_valid_enabled_techniques_and_technique_params_to_parameter_list(
      const std::string &parameter_list_name, Teuchos::ParameterList &parameter_list_to_add_to,
      const Teuchos::ParameterList &required_parameter_list, const Teuchos::ParameterList &forwarded_parameter_list,
      const std::function<Teuchos::ParameterList(const std::string &)> &get_technique_params_func) {
    parameter_list_to_add_to.setParameters(forwarded_parameter_list);

    // Because this is the valid params we list ALL possible parameters. We expect the parameters for
    // each technique to be a sublist of this list with the same name as the technique.
    for (auto &key : OurTechniqueFactory::get_keys()) {
      std::string valid_technique_name = key;
      Teuchos::ParameterList &technique_params = parameter_list_to_add_to.sublist(valid_technique_name);
      technique_params.setParameters(get_technique_params_func(valid_technique_name));

      // Check that the forwarded params exist and then remove them from the valid params since the
      // user isn't responsible for setting them.
      for (Teuchos::ParameterList::ConstIterator i = forwarded_parameter_list.begin();
           i != forwarded_parameter_list.end(); i++) {
        const std::string &forwarded_parameter_name = forwarded_parameter_list.name(i);

        MUNDY_THROW_REQUIRE(technique_params.isParameter(forwarded_parameter_name), std::runtime_error,
                            std::string("MetaTechniqueDispatcher: The technique ") + valid_technique_name +
                                " does not have the required (forwarded) parameter '" + forwarded_parameter_name +
                                "' in its " + parameter_list_name + "params.");
        technique_params.remove(forwarded_parameter_name);
      }

      // Check that the required params exist but do not remove them, as the user is responsible for setting them.
      for (Teuchos::ParameterList::ConstIterator i = required_parameter_list.begin();
           i != required_parameter_list.end(); i++) {
        const std::string &required_parameter_name = required_parameter_list.name(i);

        MUNDY_THROW_REQUIRE(technique_params.isParameter(required_parameter_name), std::runtime_error,
                            std::string("MetaTechniqueDispatcher: The technique ") + valid_technique_name +
                                " does not have the required (required) parameter '" + required_parameter_name +
                                "' in its " + parameter_list_name + "params.");
      }
    }

    return parameter_list_to_add_to;
  }
  //@}

};  // MetaTechniqueDispatcher

//! \brief Type specializations for different polymorphic base types.
//@{

template <typename DerivedType, typename ReturnType,
          mundy::meta::RegistrationStringValueWrapper technique_factory_registration_string_value_wrapper,
          mundy::meta::RegistrationStringValueWrapper default_technique_name_wrapper>
class MetaMethodExecutionDispatcher
    : public mundy::meta::MetaMethodExecutionInterface<ReturnType>,
      public MetaTechniqueDispatcher<DerivedType, mundy::meta::MetaMethodExecutionInterface<ReturnType>,
                                     mundy::meta::MetaMethodExecutionInterface<ReturnType>,
                                     technique_factory_registration_string_value_wrapper,
                                     default_technique_name_wrapper> {
 public:
  using PolymorphicBaseType = mundy::meta::MetaMethodExecutionInterface<ReturnType>;
  using OurMetaTechniqueDispatcher =
      MetaTechniqueDispatcher<DerivedType, PolymorphicBaseType, PolymorphicBaseType,
                              technique_factory_registration_string_value_wrapper, default_technique_name_wrapper>;

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaMethodExecutionDispatcher() = delete;

  /// \brief Constructor
  MetaMethodExecutionDispatcher(mundy::mesh::BulkData *const bulk_data_ptr,
                                const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_fixed_params());

    // The enabled technique name must be in our registry.
    enabled_technique_name_ = valid_fixed_params.get<std::string>("enabled_technique_name");
    MUNDY_THROW_REQUIRE(OurMetaTechniqueDispatcher::OurTechniqueFactory::is_valid_key(enabled_technique_name_),
                        std::logic_error,
                        std::string("MetaTechniqueDispatcher: The enabled technique name '") + enabled_technique_name_ +
                            "' is not a valid technique name. Valid names are: " +
                            OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys_as_string());

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_fixed_params.sublist(enabled_technique_name_);

    // At this point, the only parameters are the enabled technique name, the forwarded parameters for the enabled
    // technique, and the non-required technique params within the technique sublists. We'll loop over all parameters
    // that aren't in the technique sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_technique_name") {
        technique_params.setEntry(param_name, param_entry);
      }
    }

    technique_ptr_ = OurMetaTechniqueDispatcher::OurTechniqueFactory::create_new_instance(
        enabled_technique_name_, bulk_data_ptr, technique_params);
  }
  //@}

  //! \name MetaMethodExecutionInterface's interface implementation
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_mutable_params());

    // At this point, the only parameters are the forwarded parameters for the enabled technique and the non-forwarded
    // technique params within the technique sublists. We'll loop over all parameters that aren't in the technique
    // sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_mutable_params.begin(); i != valid_mutable_params.end(); i++) {
      const std::string &param_name = valid_mutable_params.name(i);

      const Teuchos::ParameterEntry &param_entry = valid_mutable_params.getEntry(param_name);
      if (!valid_mutable_params.isSublist(param_name)) {
        for (int j = 0; j < OurMetaTechniqueDispatcher::OurTechniqueFactory::num_registered_classes(); j++) {
          const std::string technique_name =
              std::string(OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys()[j]);
          Teuchos::ParameterList &technique_params = valid_mutable_params.sublist(technique_name);
          technique_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_mutable_params.sublist(enabled_technique_name_);
    technique_ptr_->set_mutable_params(technique_params);
  }

  /// \brief Run the method's core calculation.
  ReturnType execute() override {
    // Forward the inputs to the technique.
    technique_ptr_->execute();
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The enabled technique name.
  std::string enabled_technique_name_;

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<PolymorphicBaseType> technique_ptr_;
  //@}
};  // MetaMethodExecutionDispatcher

template <typename DerivedType, typename ReturnType,
          mundy::meta::RegistrationStringValueWrapper technique_factory_registration_string_value_wrapper,
          mundy::meta::RegistrationStringValueWrapper default_technique_name_wrapper>
class MetaMethodSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>,
      public MetaTechniqueDispatcher<DerivedType, mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>,
                                     mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>,
                                     technique_factory_registration_string_value_wrapper,
                                     default_technique_name_wrapper> {
 public:
  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>;
  using OurMetaTechniqueDispatcher =
      MetaTechniqueDispatcher<DerivedType, PolymorphicBaseType, PolymorphicBaseType,
                              technique_factory_registration_string_value_wrapper, default_technique_name_wrapper>;

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaMethodSubsetExecutionDispatcher() = delete;

  /// \brief Constructor
  MetaMethodSubsetExecutionDispatcher(mundy::mesh::BulkData *const bulk_data_ptr,
                                      const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_fixed_params());

    // The enabled technique name must be in our registry.
    enabled_technique_name_ = valid_fixed_params.get<std::string>("enabled_technique_name");
    MUNDY_THROW_REQUIRE(OurMetaTechniqueDispatcher::OurTechniqueFactory::is_valid_key(enabled_technique_name_),
                        std::logic_error,
                        std::string("MetaTechniqueDispatcher: The enabled technique name '") + enabled_technique_name_ +
                            "' is not a valid technique name. Valid names are: " +
                            OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys_as_string());

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_fixed_params.sublist(enabled_technique_name_);

    // At this point, the only parameters are the enabled technique name, the forwarded parameters for the enabled
    // technique, and the non-required technique params within the technique sublists. We'll loop over all parameters
    // that aren't in the technique sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_technique_name") {
        technique_params.setEntry(param_name, param_entry);
      }
    }

    technique_ptr_ = OurMetaTechniqueDispatcher::OurTechniqueFactory::create_new_instance(
        enabled_technique_name_, bulk_data_ptr, technique_params);
  }
  //@}

  //! \name MetaMethodSubsetExecutionInterface's interface implementation
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_mutable_params());

    // At this point, the only parameters are the forwarded parameters for the enabled technique and the non-forwarded
    // technique params within the technique sublists. We'll loop over all parameters that aren't in the technique
    // sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_mutable_params.begin(); i != valid_mutable_params.end(); i++) {
      const std::string &param_name = valid_mutable_params.name(i);

      const Teuchos::ParameterEntry &param_entry = valid_mutable_params.getEntry(param_name);
      if (!valid_mutable_params.isSublist(param_name)) {
        for (int j = 0; j < OurMetaTechniqueDispatcher::OurTechniqueFactory::num_registered_classes(); j++) {
          const std::string technique_name =
              std::string(OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys()[j]);
          Teuchos::ParameterList &technique_params = valid_mutable_params.sublist(technique_name);
          technique_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_mutable_params.sublist(enabled_technique_name_);
    technique_ptr_->set_mutable_params(technique_params);
  }

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return technique_ptr_->get_valid_entity_parts();
  }

  /// \brief Run the method's core calculation.
  ReturnType execute(const stk::mesh::Selector &input_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(input_selector);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The enabled technique name.
  std::string enabled_technique_name_;

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<PolymorphicBaseType> technique_ptr_;
  //@}
};  // MetaMethodSubsetExecutionDispatcher

template <typename DerivedType, typename ReturnType,
          mundy::meta::RegistrationStringValueWrapper technique_factory_registration_string_value_wrapper,
          mundy::meta::RegistrationStringValueWrapper default_technique_name_wrapper>
class MetaMethodPairwiseSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>,
      public MetaTechniqueDispatcher<DerivedType, mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>,
                                     mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>,
                                     technique_factory_registration_string_value_wrapper,
                                     default_technique_name_wrapper> {
 public:
  using PolymorphicBaseType = mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>;
  using OurMetaTechniqueDispatcher =
      MetaTechniqueDispatcher<DerivedType, PolymorphicBaseType, PolymorphicBaseType,
                              technique_factory_registration_string_value_wrapper, default_technique_name_wrapper>;
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaMethodPairwiseSubsetExecutionDispatcher() = delete;

  /// \brief Constructor
  MetaMethodPairwiseSubsetExecutionDispatcher(mundy::mesh::BulkData *const bulk_data_ptr,
                                              const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_fixed_params());

    // The enabled technique name must be in our registry.
    enabled_technique_name_ = valid_fixed_params.get<std::string>("enabled_technique_name");
    MUNDY_THROW_REQUIRE(OurMetaTechniqueDispatcher::OurTechniqueFactory::is_valid_key(enabled_technique_name_),
                        std::logic_error,
                        std::string("MetaTechniqueDispatcher: The enabled technique name '") + enabled_technique_name_ +
                            "' is not a valid technique name. Valid names are: " +
                            OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys_as_string());

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_fixed_params.sublist(enabled_technique_name_);

    // At this point, the only parameters are the enabled technique name, the forwarded parameters for the enabled
    // technique, and the non-required technique params within the technique sublists. We'll loop over all parameters
    // that aren't in the technique sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "enabled_technique_name") {
        technique_params.setEntry(param_name, param_entry);
      }
    }

    technique_ptr_ = OurMetaTechniqueDispatcher::OurTechniqueFactory::create_new_instance(
        enabled_technique_name_, bulk_data_ptr, technique_params);
  }
  //@}

  //! \name MetaMethodPairwiseSubsetExecutionInterface's interface implementation
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(OurMetaTechniqueDispatcher::get_valid_mutable_params());

    // At this point, the only parameters are the forwarded parameters for the enabled technique and the non-forwarded
    // technique params within the technique sublists. We'll loop over all parameters that aren't in the technique
    // sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_mutable_params.begin(); i != valid_mutable_params.end(); i++) {
      const std::string &param_name = valid_mutable_params.name(i);

      const Teuchos::ParameterEntry &param_entry = valid_mutable_params.getEntry(param_name);
      if (!valid_mutable_params.isSublist(param_name)) {
        for (int j = 0; j < OurMetaTechniqueDispatcher::OurTechniqueFactory::num_registered_classes(); j++) {
          const std::string technique_name =
              std::string(OurMetaTechniqueDispatcher::OurTechniqueFactory::get_keys()[j]);
          Teuchos::ParameterList &technique_params = valid_mutable_params.sublist(technique_name);
          technique_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Forward the inputs to the technique.
    Teuchos::ParameterList technique_params = valid_mutable_params.sublist(enabled_technique_name_);
    technique_ptr_->set_mutable_params(technique_params);
  }

  /// \brief Get valid source entity parts for the method.
  /// By "valid source entity parts," we mean the parts whose entities this method can act on as source entities.
  std::vector<stk::mesh::Part *> get_valid_source_entity_parts() const override {
    return technique_ptr_->get_valid_source_entity_parts();
  }

  /// \brief Get valid target entity parts for the method.
  /// By "valid target entity parts," we mean the parts whose entities this method can act on as target entities.
  std::vector<stk::mesh::Part *> get_valid_target_entity_parts() const override {
    return technique_ptr_->get_valid_target_entity_parts();
  }

  /// \brief Run the method's core calculation.
  ReturnType execute(const stk::mesh::Selector &source_selector, const stk::mesh::Selector &target_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(source_selector, target_selector);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The enabled technique name.
  std::string enabled_technique_name_;

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<PolymorphicBaseType> technique_ptr_;
  //@}
};  // MetaMethodSubsetExecutionDispatcher
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METATECHNIQUEDISPATCHER_HPP_
