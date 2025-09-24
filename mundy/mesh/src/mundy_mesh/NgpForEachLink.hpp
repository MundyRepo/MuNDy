// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                       Copyright 2025 Michigan State University
//                                                 Author: Bryce Palmer
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

#ifndef MUNDY_MESH_NGPFOREACHLINK_HPP_
#define MUNDY_MESH_NGPFOREACHLINK_HPP_

/// \file NgpForEachLink.hpp

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/NgpMesh.hpp>              // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>                 // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>             // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>                // for stk::mesh::EntityRank
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets

// Mundy libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_mesh/NgpLinkData.hpp>    // for mundy::mesh::NgpLinkData

namespace mundy {

namespace mesh {

//! \name Link iteration
//@{

/// \brief Run an ngp-compatible function over each link in the ngp_link_data that falls in the given selector in
/// parallel.
///
/// Thread parallel over each link.
///
/// The functor must have the following signature:
///   operator()(const FastMeshIndex &) -> void
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(
      ngp_link_data.ngp_mesh(), ngp_link_data.link_rank(),
      ngp_link_data.link_meta_data().universal_link_part() & linker_subset_selector, functor);
}

/// \brief Run an ngp-compatible function over each link in the ngp_link_data in parallel.
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const FunctionToRunPerLink &functor) {
  for_each_link_run(ngp_link_data, ngp_link_data.link_meta_data().universal_link_part(), functor);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPFOREACHLINK_HPP_
