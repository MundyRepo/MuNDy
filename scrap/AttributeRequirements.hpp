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

#ifndef MUNDY_META_ATTRIBUTREQUIREMENTS_HPP_
#define MUNDY_META_ATTRIBUTREQUIREMENTS_HPP_

/// \file AttributeRequirements.hpp
/// \brief Declaration of the AttributeRequirements class

// C++ core libs
#include <algorithm>    // for std::max
#include <any>          // for std::any
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData

namespace mundy {

namespace meta {

/// \class AttributeRequirements
/// \brief A set of necessary parameters for declaring an attribute to be stored on a attribute, part, or mesh.
class AttributeRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  AttributeRequirements() = default;

  /// \brief Constructor with full fill.
  ///
  /// \param attribute_name [in] Name of the attribute.
  AttributeRequirements(const std::string &attribute_name);

  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] List of parameters for specifying the part requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit AttributeRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required attribute name.
  /// \param attribute_name [in] Required name of the attribute.
  void set_attribute_name(const std::string &attribute_name) final;

  /// \brief Get if the attribute name is constrained or not.
  bool constrains_attribute_name() const final;

  /// \brief Get if the attribute is fully specified (aka, the name is set).
  bool is_fully_specified() const final;

  /// \brief Return the attribute name.
  /// Will throw an error if the attribute name is not constrained.
  std::string get_attribute_name() const final;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the attribute that this class defines and assign it to a field.
  void declare_attribute_on_field(mundy::mesh::MetaData *const meta_data_ptr,
                                  const stk::mesh::Field &field) const final;

  /// \brief Declare/create the attribute that this class defines and assign it to a part.
  void declare_attribute_on_part(mundy::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const final;

  /// \brief Declare/create the attribute that this class defines and assign it to the entire mesh.
  void declare_attribute_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const final;

  /// \brief Delete the attribute name constraint (if it exists).
  void delete_attribute_name() final;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// TODO(palmerb4): Is it even possible for an AttributeRequirements object to be invalid?
  void check_if_valid() final;

  /// \brief Generate new instance of this class, constructed using the given parameter list.
  std::shared_ptr<AttributesBase> create_new_instance(const Teuchos::ParameterList &parameter_list) const final;

  /// \brief Generate new instance of this class, constructed using the given parameter list.
  static std::shared_ptr<AttributesBase> static_create_new_instance(const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<AttributeRequirements>(parameter_list);
  }

  /// \brief Dump the contents of \c AttributeRequirements to the given stream (defaults to std::cout).
  void print(std::ostream &os = std::cout, int indent_level = 0) const final;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const final;
  //@}

 private:
  /// \brief Name of the attribute.
  std::string attribute_name_;

  /// \brief If the name of the attribute is set or not.
  bool attribute_name_is_set_
};  // AttributeRequirements

//! \name template implementations
//@{

// \name Constructors and destructor
//{

AttributeRequirements::AttributeRequirements(const std::string &attribute_name) : attribute_name_(attribute_name) {
  attribute_name_is_set_ = true;
  this->check_if_valid();
}

AttributeRequirements::AttributeRequirements(const Teuchos::ParameterList &parameter_list) {
  // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
  // This helps catch misspellings.
  Teuchos::ParameterList valid_params = parameter_list;
  validate_parameters_and_set_defaults(&valid_params);

  // Store the given parameters.
  this->set_attribute_name(valid_params.get<std::string>("name"));
}
//}

// \name Setters and Getters
//{

void AttributeRequirements::set_attribute_name(const std::string &attribute_name) {
  attribute_name_ = attribute_name;
  attribute_name_is_set_ = true;
  this->check_if_valid();
}

bool AttributeRequirements::constrains_attribute_name() const {
  return attribute_name_is_set_;
}

bool AttributeRequirements::is_fully_specified() const {
  return this->constrains_attribute_name();
}

std::string AttributeRequirements::get_attribute_name() const {
  MUNDY_THROW_REQUIRE(this->constrains_attribute_name(), std::logic_error,
                     std::string("AttributeRequirements: Attempting to access the attribute name requirement even though attribute ")
                     + "name is unconstrained.\n"
                         + "The current set of requirements is:\n"
                         + get_reqs_as_a_string());

  return attribute_name_;
}
//}

// \name Actions
//{

void AttributeRequirements::declare_attribute_on_field(mundy::mesh::MetaData *const meta_data_ptr,
                                                       const stk::mesh::Field &field) const {
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument,
                     "AttributeRequirements: MetaData pointer cannot be null).");
  MUNDY_THROW_REQUIRE(this->constrains_attribute_name(), std::logic_error,
                     std::string("AttributeRequirements: Attribute name must be set before calling declare_attribute*.\n")
                         + "The current set of requirements is:\n"
                         + get_reqs_as_a_string());

  std::any empty_attribute;
  meta_data_ptr->declare_attribute(field, attribute_name_, empty_attribute);
}

void AttributeRequirements::declare_attribute_on_part(mundy::mesh::MetaData *const meta_data_ptr,
                                                      const stk::mesh::Part &part) const {
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument,
                     "AttributeRequirements: MetaData pointer cannot be null).");
  MUNDY_THROW_REQUIRE(this->constrains_attribute_name(), std::logic_error,
                     std::string("AttributeRequirements: Attribute name must be set before calling declare_attribute*.\n")
                         + "The current set of requirements is:\n"
                         + get_reqs_as_a_string());

  std::any empty_attribute;
  meta_data_ptr->declare_attribute(part, attribute_name_, empty_attribute);
}

void AttributeRequirements::declare_attribute_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument,
                     "AttributeRequirements: MetaData pointer cannot be null).");
  MUNDY_THROW_REQUIRE(this->constrains_attribute_name(), std::logic_error,
                     std::string("AttributeRequirements: Attribute name must be set before calling declare_attribute*.\n")
                         + "The current set of requirements is:\n"
                         + get_reqs_as_a_string());

  std::any empty_attribute;
  meta_data_ptr->declare_attribute(attribute_name_, empty_attribute);
}

void AttributeRequirements::delete_attribute_name() {
  attribute_name_is_set_ = false;
}

std::shared_ptr<AttributesBase> AttributeRequirements::create_new_instance(
    const Teuchos::ParameterList &parameter_list) const {
  return create_new_instance(parameter_list);
}

void AttributeRequirements::print(std::ostream &os, int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  os << indent << "AttributeRequirements: " << std::endl;

  if (this->constrains_attribute_name()) {
    os << indent << "  Attribute name is set." << std::endl;
    os << indent << "  Attribute name: " << this->get_attribute_name() << std::endl;
  } else {
    os << indent << "  Attribute name is not set." << std::endl;
  }

  os << indent << "End of AttributeRequirements" << std::endl;
}

std::string AttributeRequirements::get_reqs_as_a_string() const {
  std::stringstream ss;
  this->print(ss);
  return ss.str();
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_ATTRIBUTREQUIREMENTS_HPP_
