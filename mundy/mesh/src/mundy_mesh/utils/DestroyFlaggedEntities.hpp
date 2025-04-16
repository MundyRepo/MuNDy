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

#ifndef MUNDY_MESH_UTILS_DESTROYFLAGGEDENTITIES_HPP_
#define MUNDY_MESH_UTILS_DESTROYFLAGGEDENTITIES_HPP_

/// \file DestroyFlaggedEntities.hpp
/// \brief A set of helper methods for destroying entities flagged for destruction by a boolean or integer field.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <tuple>    // for std::tuple, std::make_tuple
#include <utility>  // for std::pair, std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace utils {

/* Methodology:

Destroying entities must occur within a modification cycle. We'll assert this.
Destroying entities may lead to the bucket structure changing. This means that we cannot rely on the buckets for
iterating over entities. Instead, we fetch all entities that are in the given Selector or EntityVector, iterate over
them in serial, and destroy them.
*/

/// \brief Helper function for destroying entities flagged for destruction by an integer flag field.
/// \param bulk_data The bulk data
/// \param entities_to_maybe_destroy The entities to maybe destroy
/// \param flag_field The field that flags entities for destruction
/// \param deletion_flag_value The value of the flag field that indicates an entity should be destroyed
void destroy_flagged_entities(BulkData &bulk_data, const stk::mesh::EntityVector &entities_to_maybe_destroy,
                              const stk::mesh::Field<int> &flag_field, const int &deletion_flag_value);

/// \brief Helper function for destroying entities flagged for destruction by a boolean flag field.
/// \param bulk_data The bulk data
/// \param entity_rank The rank of the entities to destroy
/// \param selector The selector for the entities to destroy
/// \param flag_field The field that flags entities for destruction
/// \param deletion_flag_value The value of the flag field that indicates an entity should be destroyed
void destroy_flagged_entities(BulkData &bulk_data, const stk::topology::rank_t &entity_rank,
                              const stk::mesh::Selector &selector, const stk::mesh::Field<int> &flag_field,
                              const int &deletion_flag_value);

}  // namespace utils

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_UTILS_DESTROYFLAGGEDENTITIES_HPP_
