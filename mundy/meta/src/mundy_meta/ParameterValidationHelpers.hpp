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

#ifndef MUNDY_META_PARAMETERVALIDATIONHELPERS_HPP_
#define MUNDY_META_PARAMETERVALIDATIONHELPERS_HPP_

/// \file ParameterValidationHelpers.hpp
/// \brief Declaration of the parameter validation helper functions.

// C++ core libs
#include <stdexcept>  // for std::invalid_argument
#include <string>     // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

namespace mundy {

namespace meta {

/// \brief A struct that holds the configuration of a parameter.
/// @tparam ValueType The type of the parameter's value.
template <typename ValueType>
struct ParamConfig {
  /// \brief The name of the parameter.
  std::string name;

  /// \brief The default value of the parameter.
  ValueType default_value;

  /// \brief The documentation string of the parameter.
  std::string doc_string = "";
};  // struct ParamConfig

/// \brief A function that checks if a parameter is in the parameter list (with the correct type) and sets its default
/// value if it is not.
/// @tparam ValueType The type of the parameter's value.
/// \param params_ptr The pointer to the parameter list.
/// \param config The configuration of the parameter.
template <typename ValueType>
void check_parameter_and_set_default(Teuchos::ParameterList *const params_ptr, const ParamConfig<ValueType> &config) {
  if (params_ptr->isParameter(config.name)) {
    // Check if the parameter is of the expected type
    const bool valid_type = params_ptr->isType<ValueType>(config.name);
    if (!valid_type) {
      throw std::invalid_argument("Type error in parameter '" + config.name + "'");
    }
  } else {
    // Set the parameter with its default value and description
    params_ptr->set(config.name, config.default_value, config.doc_string);
  }
}

/// \brief A function that checks if a parameter is in the parameter list and has the correct type. Throws an exception
/// if it is not.
/// @tparam ValueType The type of the parameter's value.
/// \param params_ptr The pointer to the parameter list.
/// \param name The name of the parameter.
template <typename ValueType>
void check_required_parameter(Teuchos::ParameterList *const params_ptr, const std::string &name) {
  MUNDY_THROW_REQUIRE(params_ptr->isParameter(name), std::invalid_argument,
                      std::string("Missing parameter '") + name + "' in the parameter list.");
  const bool valid_type = params_ptr->INVALID_TEMPLATE_QUALIFIER isType<ValueType>(name);
  MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                      std::string("Type error. Given a parameter with name '") + name +
                          "' but with a type other than " + typeid(ValueType).name());
}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_PARAMETERVALIDATIONHELPERS_HPP_