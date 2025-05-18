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

#ifndef MUNDY_MESH_METADATA_HPP_
#define MUNDY_MESH_METADATA_HPP_

/// \file MetaData.hpp
/// \brief Declaration of the MetaData class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part

namespace mundy {

namespace mesh {

/// \class MetaData
/// \brief A extension of STK's MetaData, better suited for some of Mundy's requirements.
///
/// For now, this extension modifies how attributes are created and stored.
class MetaData : public stk::mesh::MetaData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Construct a meta data manager to own parts and fields.
  MetaData();

  /// \brief Construct a meta data manager to own parts and fields.
  explicit MetaData(size_t spatial_dimension, const std::vector<std::string> &rank_names = std::vector<std::string>());

  /// \brief Destructor.
  virtual ~MetaData();
  //@}

  //! \name Actions
  //@{

  /// \brief Declare an attribute on the given field.
  /// \param field The field which should contain the given attribute.
  /// \param attribute_name The given attribute's name. Must have a unique name not shared by other attributes on the
  /// field.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name,
                         const std::any &attribute_data);

  /// \brief Declare an attribute on the given field.
  /// \param field The field which should contain the given attribute.
  /// \param attribute_name The given attribute's name. Must have a unique name not shared by other attributes on the
  /// field.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name,
                         const std::any &&attribute_data);

  /// \brief Declare an attribute on the given part.
  /// \param part The part which should contain the given attribute.
  /// \param attribute_name The given attribute's name. Must have a unique name not shared by other attributes on the
  /// part.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const stk::mesh::Part &part, const std::string &attribute_name,
                         const std::any &attribute_data);

  /// \brief Declare an attribute on the given part.
  /// \param part The part which should contain the given attribute.
  /// \param attribute_name The given attribute's name. Must have a unique name not shared by other attributes on the
  /// part.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const stk::mesh::Part &part, const std::string &attribute_name,
                         const std::any &&attribute_data);

  /// \brief Declare an attribute on the mesh itself.
  /// \param attribute_name The name of the attribute to declare. Must have a unique name not shared by other attributes
  /// on the mesh.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const std::string &attribute_name, const std::any &attribute_data);

  /// \brief Declare an attribute on the mesh itself.
  /// \param attribute_name The name of the attribute to declare. Must have a unique name not shared by other attributes
  /// on the mesh.
  /// \param attribute_data The given attribute's data.
  void declare_attribute(const std::string &attribute_name, const std::any &&attribute_data);

  /// \brief Attempt to remove an attribute from the provided field.
  /// \param field The given field whose attribute we are trying to remove.
  /// \param attribute_name The name of the attribute to remove.
  /// \return A flag indicating if the attribute existed on the given field or not.
  bool remove_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name);

  /// \brief Attempt to remove an attribute from the provided part.
  /// \param part The given part whose attribute we are trying to remove.
  /// \param attribute_name The name of the attribute to remove.
  /// \return A flag indicating if the attribute existed on the given part or not.
  bool remove_attribute(const stk::mesh::Part &part, const std::string &attribute_name);

  /// \brief Attempt to remove an attribute from this mesh.
  /// \param attribute_name The name of the attribute to remove.
  /// \return A flag indicating if the attribute existed on this mesh or not.
  bool remove_attribute(const std::string &attribute_name);

  /// \brief Attempt to fetch a field attribute with the provided name from the given field.
  /// \param field The given field whose attribute we are trying to fetch.
  /// \param attribute_name The name of the attribute to fetch.
  /// \return A pointer to the internally maintained attribute data if it exists, nullptr otherwise.
  /// TODO(palmerb4): This should return a formal stk::mesh::Attribute which wraps the std::any and stores the attribute
  /// mesh ordinal and name.
  std::any *get_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name);

  /// \brief Attempt to fetch a part attribute with the provided name from the given part.
  /// \param part The given part whose attribute we are trying to fetch.
  /// \param attribute_name The name of the attribute to fetch.
  /// \return A pointer to the internally maintained attribute data if it exists, nullptr otherwise.
  /// TODO(palmerb4): This should return a formal stk::mesh::Attribute which wraps the std::any and stores the attribute
  /// mesh ordinal and name.
  std::any *get_attribute(const stk::mesh::Part &part, const std::string &attribute_name);

  /// \brief Attempt to fetch an attribute with the provided name from the current mesh.
  /// \param attribute_name The name of the attribute to fetch.
  /// \return A pointer to the internally maintained attribute data if it exists, nullptr otherwise.
  /// TODO(palmerb4): This should return a formal stk::mesh::Attribute which wraps the std::any and stores the attribute
  /// mesh ordinal and name.
  std::any *get_attribute(const std::string &attribute_name);
  //@}

 private:
  //! \name Internal members
  //@{

  std::map<unsigned, std::map<std::string, std::any>> field_to_field_attributes_map_;
  std::map<unsigned, std::map<std::string, std::any>> part_to_part_attributes_map_;
  std::map<std::string, std::any> mesh_attributes_map_;
  //@}
};  // MetaData

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_METADATA_HPP_
