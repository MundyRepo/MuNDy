// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is empty software: you can redistribute it and/or modify it under the terms of the GNU General Public License
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

#ifndef MUNDY_MESH_IMPL_PARTITIONKEY_HPP_
#define MUNDY_MESH_IMPL_PARTITIONKEY_HPP_


// C++ core
#include <stdexcept>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>


namespace mundy {

namespace mesh {

namespace impl {

using PartitionKey = std::vector<stk::mesh::PartOrdinal>;  // sorted view of part ordinals
using NgpPartitionKey = stk::mesh::PartOrdinalViewType;    // sorted view of part ordinals

/// \brief Get the partition key for a given set of link parts (independent of their order, host only)
PartitionKey get_partition_key(const stk::mesh::PartVector &parts) const {
stk::mesh::OrdinalVector parts_and_supersets;
stk::mesh::impl::fill_add_parts_and_supersets(parts, parts_and_supersets);
return parts_and_supersets;
}

/// \brief Get the partition key for a given link bucket (host only)
PartitionKey get_partition_key(const stk::mesh::Bucket &link_bucket) const {
return get_partition_key(link_bucket.supersets());
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_PARTITIONKEY_HPP_