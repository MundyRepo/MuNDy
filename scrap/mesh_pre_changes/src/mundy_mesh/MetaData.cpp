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

/// \file MetaData.cpp
/// \brief Definition of the MetaData class

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <algorithm>    // for std::max
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <typeindex>    // for std::type_index
#include <utility>      // for std::make_pair

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types

namespace mundy {

namespace mesh {

// \name Constructors and destructor
//{

MetaData::MetaData() : stk::mesh::MetaData() {
}

MetaData::MetaData(size_t spatial_dimension, const std::vector<std::string> &rank_names)
    : stk::mesh::MetaData(spatial_dimension, rank_names) {
}

MetaData::~MetaData() {
}

//}

// \name Actions
//{

void MetaData::declare_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name,
                                 const std::any &attribute_data) {
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_is_unique = (field_to_field_attributes_map_[field_id].count(attribute_name) == 0);
    MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                        fmt::format("MetaData: An attribute with the same name as the provided attribute already "
                                    "exists on the given field.\n"
                                    "  Field name: {}\n"
                                    "  Attribute name: {}\n",
                                    field.name(), attribute_name));
  } else {
    field_to_field_attributes_map_.insert(std::make_pair(field_id, std::map<std::string, std::any>()));
  }

  field_to_field_attributes_map_[field_id].insert(std::make_pair(attribute_name, attribute_data));
}

void MetaData::declare_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name,
                                 const std::any &&attribute_data) {
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_is_unique = (field_to_field_attributes_map_[field_id].count(attribute_name) == 0);
    MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                        fmt::format("MetaData: An attribute with the same name as the provided attribute already "
                                    "exists on the given field.\n"
                                    "  Field name: {}\n"
                                    "  Attribute name: {}\n",
                                    field.name(), attribute_name));
  } else {
    field_to_field_attributes_map_.insert(std::make_pair(field_id, std::map<std::string, std::any>()));
  }

  field_to_field_attributes_map_[field_id].insert(std::make_pair(attribute_name, std::move(attribute_data)));
}

void MetaData::declare_attribute(const stk::mesh::Part &part, const std::string &attribute_name,
                                 const std::any &attribute_data) {
  const unsigned part_id = part.mesh_meta_data_ordinal();

  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_is_unique = (part_to_part_attributes_map_[part_id].count(attribute_name) == 0);
    MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                        fmt::format("MetaData: An attribute with the same type as the provided attribute already "
                                    "exists on the given part.\n"
                                    "  Part id: {}\n"
                                    "  Part name: {}\n"
                                    "  Attribute name: {}\n",
                                    part_id, part.name(), attribute_name));
  } else {
    part_to_part_attributes_map_.insert(std::make_pair(part_id, std::map<std::string, std::any>()));
  }

  part_to_part_attributes_map_[part_id].insert(std::make_pair(attribute_name, attribute_data));
}

void MetaData::declare_attribute(const stk::mesh::Part &part, const std::string &attribute_name,
                                 const std::any &&attribute_data) {
  const unsigned part_id = part.mesh_meta_data_ordinal();

  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_is_unique = (part_to_part_attributes_map_[part_id].count(attribute_name) == 0);
    MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                        fmt::format("MetaData: An attribute with the same type as the provided attribute already "
                                    "exists on the given part.\n"
                                    "  Part id: {}\n"
                                    "  Part name: {}\n"
                                    "  Attribute name: {}\n",
                                    part_id, part.name(), attribute_name));
  } else {
    part_to_part_attributes_map_.insert(std::make_pair(part_id, std::map<std::string, std::any>()));
  }

  part_to_part_attributes_map_[part_id].insert(std::make_pair(attribute_name, std::move(attribute_data)));
}

void MetaData::declare_attribute(const std::string &attribute_name, const std::any &attribute_data) {
  const bool attribute_is_unique = (mesh_attributes_map_.count(attribute_name) == 0);
  MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                      fmt::format("MetaData: An attribute with the same type as the provided attribute already "
                                  "exists on this mesh.\n"
                                  "  Attribute name: {}\n",
                                  attribute_name));

  mesh_attributes_map_.insert(std::make_pair(attribute_name, attribute_data));
}

void MetaData::declare_attribute(const std::string &attribute_name, const std::any &&attribute_data) {
  const bool attribute_is_unique = (mesh_attributes_map_.count(attribute_name) == 0);
  MUNDY_THROW_REQUIRE(attribute_is_unique, std::invalid_argument,
                      fmt::format("MetaData: An attribute with the same type as the provided attribute already "
                                  "exists on this mesh.\n"
                                  "  Attribute name: {}\n",
                                  attribute_name));

  mesh_attributes_map_.insert(std::make_pair(attribute_name, std::move(attribute_data)));
}

// \name Actions
//{

bool MetaData::remove_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name) {
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_exists = (field_to_field_attributes_map_[field_id].count(attribute_name) != 0);
    if (attribute_exists) {
      field_to_field_attributes_map_[field_id].erase(attribute_name);
      return true;
    } else {
      // The attribute doesn't exist, nothing to remove.
      return false;
    }
  } else {
    // The field has no attributes, so I'm pretty certain the provided attribute doesn't exist.
    return false;
  }
}

bool MetaData::remove_attribute(const stk::mesh::Part &part, const std::string &attribute_name) {
  const unsigned part_id = part.mesh_meta_data_ordinal();

  // TODO(palmerb4): Attributes should be inherited. Check if any of our parents are in the list.
  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_exists = (part_to_part_attributes_map_[part_id].count(attribute_name) != 0);
    if (attribute_exists) {
      part_to_part_attributes_map_[part_id].erase(attribute_name);
      return true;
    } else {
      // The attribute doesn't exist, nothing to remove.
      return false;
    }
  } else {
    // The part has no attributes, so I'm pretty certain the provided attribute doesn't exist.
    return false;
  }
}

bool MetaData::remove_attribute(const std::string &attribute_name) {
  const bool attribute_exists = (mesh_attributes_map_.count(attribute_name) != 0);
  if (attribute_exists) {
    mesh_attributes_map_.erase(attribute_name);
    return true;
  } else {
    // The attribute doesn't exist, nothing to remove.
    return false;
  }
}

std::any *MetaData::get_attribute(const stk::mesh::FieldBase &field, const std::string &attribute_name) {
  const unsigned field_id = field.mesh_meta_data_ordinal();
  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_exists = (field_to_field_attributes_map_[field_id].count(attribute_name) != 0);
    if (attribute_exists) {
      // Return a pointer to the attribute.
      return &field_to_field_attributes_map_[field_id][attribute_name];
    }
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}

std::any *MetaData::get_attribute(const stk::mesh::Part &part, const std::string &attribute_name) {
  const unsigned part_id = part.mesh_meta_data_ordinal();
  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_exists = (part_to_part_attributes_map_[part_id].count(attribute_name) != 0);
    if (attribute_exists) {
      // Return a pointer to the attribute.
      return &part_to_part_attributes_map_[part_id][attribute_name];
    }
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}

std::any *MetaData::get_attribute(const std::string &attribute_name) {
  const bool attribute_exists = (mesh_attributes_map_.count(attribute_name) != 0);
  if (attribute_exists) {
    // Return a pointer to the attribute.
    return &mesh_attributes_map_[attribute_name];
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}
//}

//@}

}  // namespace mesh

}  // namespace mundy
