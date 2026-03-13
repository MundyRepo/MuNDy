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

// C++ core
#include <stdexcept>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_mesh/impl/PartitionKey.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace mesh {

namespace impl {

PartitionKey get_partition_key(const stk::mesh::PartVector& parts) {
  stk::mesh::OrdinalVector parts_and_supersets;
  stk::mesh::impl::fill_add_parts_and_supersets(parts, parts_and_supersets);
  return parts_and_supersets;
}

PartitionKey get_partition_key(const stk::mesh::Bucket& link_bucket) {
  return get_partition_key(link_bucket.supersets());
}

stk::mesh::PartVector get_parts_for_partition_key(const PartitionKey& key, const stk::mesh::MetaData& meta_data) {
  size_t num_parts = key.size();
  stk::mesh::PartVector parts(num_parts);
  for (size_t i = 0; i < num_parts; ++i) {
    stk::mesh::PartOrdinal part_ordinal = key[i];
    stk::mesh::Part& part = meta_data.get_part(part_ordinal);
    parts[i] = &part;
  }
  return parts;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy
