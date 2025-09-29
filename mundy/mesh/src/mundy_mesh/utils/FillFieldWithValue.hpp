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

#ifndef MUNDY_MESH_UTILS_FILLFIELDWITHVALUE_HPP_
#define MUNDY_MESH_UTILS_FILLFIELDWITHVALUE_HPP_

/// \file FillFieldWithValue.hpp
/// \brief A set of helper methods for filling a field (or a subset of a field) with a given value.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <tuple>    // for std::tuple, std::make_tuple
#include <utility>  // for std::pair, std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy libs
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace utils {

/// \brief Helper function for filling a subset of a field with a given value.
/// \param selector The selector for the entities to fill (we'll only fill entities that are in the selector and have
/// the field)
/// \param field The field to fill
/// \param value The value to fill the field with
///
/// \tparam value_t The type of the field
/// \tparam value_size The size of the field value
template <typename value_t, std::size_t value_size>
void fill_field_with_value(const stk::mesh::Selector &selector, const stk::mesh::Field<value_t> &field,
                           const std::array<value_t, value_size> &value) {
  stk::mesh::BulkData &bulk_data = field.get_mesh();
  ::mundy::mesh::for_each_entity_run(
      bulk_data, field.entity_rank(), selector,
      [&field, &value]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
        for (std::size_t i = 0; i < value_size; ++i) {
          stk::mesh::field_data(field, entity)[i] = value[i];
        }
      });
}

/// \brief Helper function for filling a field with a given value.
/// \param field The field to fill
/// \param value The value to fill the field with
///
/// \tparam value_t The type of the field
/// \tparam value_size The size of the field value
template <typename value_t, std::size_t value_size>
void fill_field_with_value(const stk::mesh::Field<value_t> &field, const std::array<value_t, value_size> &value) {
  stk::mesh::Selector field_selector = stk::mesh::selectField(field);
  fill_field_with_value(field_selector, field, value);
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_UTILS_FILLFIELDWITHVALUE_HPP_
