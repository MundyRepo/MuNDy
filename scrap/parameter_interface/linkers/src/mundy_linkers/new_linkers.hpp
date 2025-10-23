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

#ifndef MUNDY_NEW_LINKERS_HPP_
#define MUNDY_NEW_LINKERS_HPP_

// C++ core
#include <map>
#include <utility>
#include <vector>

// STK
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/MetaData.hpp>

/// \file NewLinkers.hpp

namespace mundy {

namespace linkers {

namespace impl {

/// \brief Decompose a map<pair<Selector, Selector>, PartVector> into a map<SortedMasterPartVector,
/// vector<pair<Partition, Partition>>>.
///
/// The goal here is to take a user defined map from a pair of source/target selectors for a neighbor search and the
/// PartVector for the linker that spans objects in these selectors and invert this into a map from each unique linker
/// part vector to the vector of pairs of linked entity partitions for which need to perform the neighbor search. If we
/// perform a neighbor search between all source/target buckets within the vector of pairs of linked entity partitions,
/// then the resulting neighbor list will require linkers within the set of parts in the corresponding part vector.
/// Hence, this map represents the minimal number of neighbor searches that we need to perform to cover all of the user
/// defined selectors, while producing neighbor lists that can have their neighbor linkers created in bulk.
///
/// The procedure is as follows:
///   For all pairs of Partitions of a given rank (num partitions^2 loop), loop over each of the given selector pairs
///   and see if the first bucket in each Partition is within said selector, if so, append the corresponding PartVector
///   to a master PartVector for this pair of Partitions. After the loop, sort the master part vector to form a key and
///   add it to a map<SortedMasterPartVector, vector<pair<Partition, Partition>>>. We will perform one neighbor search
///   per key within this map between all buckets within the source and target partitions.
///
/// The total cost here is num_partitions^2 * num_input_selectors, which is at worst like 100^2 * 10 = 100,000 but will
/// typically be like 6^2 * 2 = 64.
std::map<stk::mesh::PartVector, std::vector<std::pair<stk::mesh::impl::Partition, stk::mesh::impl::Partition>>>
get_neighbor_linker_specializations_to_search_map(
    const mundy::mesh::BulkData &bulk_data, stk::topology::rank_t source_rank, stk::topology::rank_t target_rank,
    const std::map<std::pair<stk::mesh::Selector, stk::mesh::Selector>, stk::mesh::PartVector>
        &neighbor_linker_specializations) {
  using stk::mesh::PartVector;
  using stk::mesh::impl::Partition;
  using PartitionPair = std::pair<Partition, Partition>;
  std::map<PartVector, std::vector<PartitionPair>> neighbor_linker_specializations_to_search_map;

  // Need BucketRepository's get_partitions function, which, in turn requires that BulkData expose it's
  // m_bucket_repository in some way. Our BulkData exposes it within a get_bucket_repository function.
  const auto &bucket_repository = bulk_data.get_bucket_repository();
  const auto &source_partitions = bucket_repository.get_partitions(source_rank);
  const auto &target_partitions = bucket_repository.get_partitions(target_rank);

  // Loop over all pairs of partitions of the source and target ranks.
  for (const Partition &source_partition : source_partitions) {
    for (const Partition &target_partition : target_partitions) {
      // For each pair of partitions, loop over the selector pairs and see if the first bucket in each partition is
      // within the selector.
      for (const auto &[selector_pair, part_vector] : neighbor_linker_specializations) {
        const stk::mesh::Selector &source_selector = selector_pair.first;
        const stk::mesh::Selector &target_selector = selector_pair.second;
        if (source_selector(source_partition.get_all_mesh_buckets()) &&
            target_selector(target_partition.get_all_mesh_buckets())) {
          // If so, append the part vector to a master part vector for this pair of partitions.
          PartVector &master_part_vector = neighbor_linker_specializations_to_search_map[part_vector];
          master_part_vector.push_back(source_partition);
          master_part_vector.push_back(target_partition);
        }
      }
    }
  }

  // For each pair of partitions, loop over the selector pairs and see if the first bucket in each partition is within
  // the selector.
}

/* Notes:

Our neighbor linkers need to go


*/
class NeighborAggregate {};

}  // namespace impl

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_NEW_LINKERS_HPP_
