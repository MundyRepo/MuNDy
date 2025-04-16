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

/// \file DestroyFlaggedEntities.cpp
/// \brief Definition of the destroy_flagged_entities helper functions.

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <tuple>    // for std::tuple, std::make_tuple
#include <utility>  // for std::pair, std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::get_selected_entities

// Mundy libs
#include <mundy_core/throw_assert.hpp>                  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                      // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>                 // adds fmt::format for stk types
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>  // for mundy::mesh::utils::destroy_flagged_entities

namespace mundy {

namespace mesh {

namespace utils {

void destroy_flagged_entities(BulkData &bulk_data, const stk::mesh::EntityVector &entities_to_maybe_destroy,
                              const stk::mesh::Field<int> &flag_field, const int &deletion_flag_value) {
  // Assert that the bulk data is in a modification cycle
  MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error,
                      "Bulk data must be in a modification cycle to destroy entities.");

  // Iterate over the entities to destroy and destroy them
  for (const stk::mesh::Entity &entity : entities_to_maybe_destroy) {
    const bool should_destroy_entity = stk::mesh::field_data(flag_field, entity)[0] == deletion_flag_value;
    if (should_destroy_entity) {
      bool success = bulk_data.destroy_entity(entity);
      MUNDY_THROW_ASSERT(success, std::runtime_error,
                         fmt::format("Failed to destroy entity. Entity rank: {}, entity id: {}",
                                     bulk_data.entity_rank(entity), bulk_data.identifier(entity)));
    }
  }
}

void destroy_flagged_entities(BulkData &bulk_data, const stk::topology::rank_t &entity_rank,
                              const stk::mesh::Selector &selector, const stk::mesh::Field<int> &flag_field,
                              const int &deletion_flag_value) {
  // Get the entities to destroy from the selector
  stk::mesh::EntityVector entities_to_maybe_destroy;
  stk::mesh::get_selected_entities(selector, bulk_data.buckets(entity_rank), entities_to_maybe_destroy);
  destroy_flagged_entities(bulk_data, entities_to_maybe_destroy, flag_field, deletion_flag_value);
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy
