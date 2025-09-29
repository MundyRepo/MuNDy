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

#ifndef MUNDY_MESH_IMPL_NGPCOOTOCRSSYNCHRONIZER_HPP_
#define MUNDY_MESH_IMPL_NGPCOOTOCRSSYNCHRONIZER_HPP_

/// \file NgpCOOToCRSSynchronizerT.hpp
/// \brief Declaration of the NgpCOOToCRSSynchronizerT class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <Kokkos_Sort.hpp>                        // for Kokkos::sort
#include <Kokkos_UnorderedMap.hpp>                // for Kokkos::UnorderedMap
#include <stk_mesh/base/Entity.hpp>               // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>                // for stk::mesh::Field
#include <stk_mesh/base/FindRestriction.hpp>      // for stk::mesh::find_restriction
#include <stk_mesh/base/GetEntities.hpp>          // for stk::mesh::get_selected_entities
#include <stk_mesh/base/GetNgpField.hpp>          // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>           // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>             // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>              // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>                 // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>             // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>                // for stk::mesh::EntityRank
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets
#include <stk_util/ngp/NgpSpaces.hpp>             // for stk::ngp::HostMemSpace, stk::ngp::UVMMemSpace

// Mundy libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/LinkCRSPartition.hpp>  // for mundy::mesh::LinkCRSPartition
#include <mundy_mesh/LinkMetaData.hpp>      // for mundy::mesh::LinkMetaData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>      // for mundy::mesh::field_copy

namespace mundy {

namespace mesh {

namespace impl {

// This class is really more like a namespace with similar methods, which exists because we want the CRS/COO
// to only be friends with this and not each individual method.
template <typename NgpMemSpace>
class NgpCOOToCRSSynchronizerT {
 public:
  //! \name Aliases
  //@{

  static_assert(Kokkos::is_memory_space_v<NgpMemSpace> &&
                    Kokkos::SpaceAccessibility<stk::ngp::ExecSpace, NgpMemSpace>::accessible,
                "NgpMemSpace must be a Kokkos memory space accessible from the device execution space.");

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

  template <typename T>
  using LinkCRSPartitionViewT = Kokkos::View<LinkCRSPartitionT<T> *, stk::ngp::UVMMemSpace>;

  template <typename T>
  using LinkBucketToPartitionIdMapT = Kokkos::UnorderedMap<unsigned, unsigned, T>;

  using NgpLinkCRSPartitionView = Kokkos::View<LinkCRSPartitionT<NgpMemSpace> *, stk::ngp::UVMMemSpace>;
  using LinkBucketToPartitionIdMap = Kokkos::UnorderedMap<unsigned, unsigned, NgpMemSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  NgpCOOToCRSSynchronizerT() = default;

  /// \brief Default copy or move constructors/operators.
  NgpCOOToCRSSynchronizerT(const NgpCOOToCRSSynchronizerT &) = default;
  NgpCOOToCRSSynchronizerT(NgpCOOToCRSSynchronizerT &&) = default;
  NgpCOOToCRSSynchronizerT &operator=(const NgpCOOToCRSSynchronizerT &) = default;
  NgpCOOToCRSSynchronizerT &operator=(NgpCOOToCRSSynchronizerT &&) = default;

  /// \brief Destructor.
  virtual ~NgpCOOToCRSSynchronizerT() = default;
  //@}

  //! \name Methods
  //@{

  /// \brief Check if the CRS connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CRS buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  static bool is_crs_up_to_date(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpCOOToCRSSynchronizerT::is_crs_up_to_date");

    // Dereference just once
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Two types of out-of-date:
    //  1. The CRS connectivity of a selected partition is dirty.
    //    - Team loop over each selected partition and thread loop over each bucket in the partition. If any bucket is
    //    dirty, atomically set the needs updated flag to true.
    const NgpLinkCRSPartitionView &partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    unsigned num_partitions = partitions.extent(0);
    bool crs_buckets_up_to_date = true;
    for (unsigned i = 0; i < num_partitions; ++i) {
      const NgpLinkCRSPartitionT<NgpMemSpace> &partition = partitions(i);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        const unsigned num_buckets = partition.num_buckets(rank);
        for (unsigned bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
          const auto &crs_bucket_conn = partition.get_crs_bucket_conn(rank, bucket_index);
          if (impl::get_dirty_flag(crs_bucket_conn)) {
            crs_buckets_up_to_date = false;
            goto done_checking_crs_buckets;
          }
        }
      }
    }
  done_checking_crs_buckets:

    // TODO(palmerb4): It appears as though counting the number of dirty buckets in a parallel_for is slower than doing
    // it serially (at least for a CPU build). Is this true for GPU builds too?

    // int num_dirty_buckets = 0;
    // typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    // const auto &team_policy =
    //     stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(partitions.extent(0), Kokkos::AUTO);
    // Kokkos::parallel_reduce(
    //     "NgpCOOToCRSSynchronizerT::is_crs_up_to_date", team_policy,
    //     KOKKOS_LAMBDA(const TeamHandleType &team, int &team_local_count) {
    //       const stk::mesh::Ordinal partition_id = team.league_rank();
    //       const NgpLinkCRSPartitionT<NgpMemSpace> &partition = partitions(partition_id);

    //       int tmp_team_local_count = 0;

    //       for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    //         const unsigned num_buckets = partition.num_buckets(rank);
    //         int rank_local_count = 0;
    //         Kokkos::parallel_reduce(
    //             Kokkos::TeamThreadRange(team, num_buckets),
    //             [&](const unsigned bucket_index, int &count) {
    //               const auto &crs_bucket_conn = partition.get_crs_bucket_conn(rank, bucket_index);
    //               count += impl::get_dirty_flag(crs_bucket_conn);
    //             },
    //             Kokkos::Sum<int>(rank_local_count));
    //         tmp_team_local_count += rank_local_count;
    //       }

    //       team_local_count += tmp_team_local_count;
    //     },
    //     Kokkos::Sum<int>(num_dirty_buckets));
    // bool crs_buckets_up_to_date = num_dirty_buckets == 0;

    if (crs_buckets_up_to_date) {  // No need to perform the second check if the first fails.
      //  2. A selected link is out-of-date.
      int link_needs_updated_count =
          ::mundy::mesh::field_sum<int>(impl::get_link_crs_needs_updated_field(crs_data.link_meta_data()),
                                        link_subset_selector, stk::ngp::ExecSpace());
      bool links_up_to_date = (link_needs_updated_count == 0);
      return links_up_to_date;
    }

    Kokkos::Profiling::popRegion();
    return crs_buckets_up_to_date;
  }

  /// \brief Check if the CRS connectivity is up-to-date for all links.
  static bool is_crs_up_to_date(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data) {
    return is_crs_up_to_date(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity for the given link subset selector.
  /// This takes changes made via the declare/delete_relation functions or request/destroy links and updates
  /// the CRS connectivity to reflect these changes.
  static void update_crs_from_coo(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                  const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    if (is_crs_up_to_date(crs_data, coo_data, link_subset_selector)) {
      return;
    }

    Kokkos::Profiling::pushRegion("NgpCOOToCRSSynchronizerT::update_crs_from_coo");

    // There are a couple options here for the types of state that we need to address:
    //  1. Nothing happened: All STK buckets are up-to-date (technically we only care about link buckets and the buckets
    //  of entities they link but we'll ignore that for now), all selected links are up-to-date, and all CRS buckets of
    //  selected partitions are up-to-date.
    //  2. A link was added or removed: Some STK buckets are out-of-date but all selected links are up-to-date.
    //  3. A link relation was added or removed: All STK buckets are up-to-date but some selected links are out-of-date.
    //  4. A combination of 2 and 3: Some STK buckets are out-of-date and some selected links are out-of-date.
    //
    // If some links are out-of-date, we need to mark the CRS connectivity of each downward linked entity as dirty. Upon
    // update, all dirty buckets will be updated.
    //
    // If link is created (or becomes a member of a link part), this has no effect on the CRS connectivity until it is
    // connected to linked entities, at which point, the regular update procedure would handle propagating changes.
    //
    // If a link is deleted (or loses its link part membership), then we need to update the CRS connectivity of the
    // downward linked CRS entities (regardless of the currently linked entities).
    //
    // If a the ownership of a link changes, the process losing ownership must mark the buckets of downward linked
    // entities are dirty and the receiving process must mark the link as needing an update to its CRS connectivity.
    //
    // If a link ever enters a state where it is linked to non-empty entities and none of those entities are owned by
    // its owning process, the link will transfer ownership to the process that owns the first non-empty linked entity.
    //
    // An observer will detect deletions, loss of link part membership, and changes of ownership. It will properly flag
    // the buckets of the linked entities as dirty or the link itself as being out-of-date, which will then be processed
    // in the next update.
    //
    // This function is independent of said observer and is only responsible for updating the CRS connectivity given
    // that some links are out-of-date or some crs buckets are dirty. As such, it simply loops over the links in the
    // given selector, flags the buckets of linked entities as dirty if the link is out-of-date, and then updates all
    // dirty crs buckets.
    //
    // stk_link_bucket_to_partition_id_map_ is a weird animal in that it must have the same size as the number of
    // buckets but the number of buckets may change during a modification cycle. We need to be certain if buckets may
    // even change their IDs during a modification cycle or not. If not, then we need to delete and shift this map each
    // time a bucket is destroyed.
    //
    // Each link bucket needs to be able to access its link partition. Buckets may change parts, but the observer isn't
    // informed of this change. Every tome that STK send a signal for local_buckets_changed_notification(link_rank), we
    // need to rebuilt this map by looping over all selected link buckets, fetching their partition key, and using
    // partition_key_to_id_map_ to get the corresponding id. We'll then store this in the bucket to id map.
    //
    // This tells us that the LinkDataObserver is in charge of deciding when to rebuild this map but not when to build
    // it in the first place. We could use a memoized getter that sees if the list is empty or not. If it's empty, it
    // calls rebuild_stk_link_bucket_to_partition_id_map.
    flag_dirty_linked_buckets_of_modified_links(crs_data, coo_data, link_subset_selector);

    reset_dirty_linked_buckets(crs_data, coo_data, link_subset_selector);

    gather_part_1_count(crs_data, coo_data, link_subset_selector);

    gather_part_2_partial_sum(crs_data, coo_data, link_subset_selector);

    scatter_part_1_setup(crs_data, coo_data, link_subset_selector);

    scatter_part_2_fill(crs_data, coo_data, link_subset_selector);

    finalize_crs_update(crs_data, coo_data, link_subset_selector);
    Kokkos::Profiling::popRegion();

// If in debug, check consistency
#ifndef NDEBUG
    check_crs_coo_consistency(crs_data, coo_data, link_subset_selector);
#endif
  }

  /// \brief Propagate changes made to the COO connectivity to the CRS connectivity.
  static void update_crs_from_coo(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data) {
    update_crs_from_coo(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part());
  }

  static void flag_dirty_linked_buckets_of_modified_links(NgpLinkCRSDataT<NgpMemSpace> &crs_data,
                                                          NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                                          const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::flag_dirty_linked_buckets");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Flag dirty buckets: Team loop over selected link buckets, fetch their partition, thread loop over links,
    // determine if any of those links are flagged as modified. If so, determine if their links were created or
    // destroyed. Flag the linked bucket of new or deleted entities as dirty.

    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "flag_dirty_linked_buckets_of_modified_links", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");
          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);

          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            if (coo_data.get_link_crs_needs_updated(link_index)) {
              // Loop over the linked entities of this link
              for (unsigned d = 0; d < dimensionality; ++d) {
                stk::mesh::Entity linked_entity_crs = coo_data.get_linked_entity_crs(link_index, d);
                stk::mesh::Entity linked_entity = coo_data.get_linked_entity(link_index, d);
                bool things_changed = linked_entity_crs != linked_entity;
                if (things_changed) {
                  bool old_entity_is_valid = (linked_entity_crs != stk::mesh::Entity());
                  if (old_entity_is_valid) {
                    // Mark the old linked entity's crs bucket conn as dirty
                    const stk::mesh::FastMeshIndex linked_entity_crs_index =
                        ngp_mesh.fast_mesh_index(linked_entity_crs);
                    const stk::mesh::EntityRank linked_entity_crs_rank = ngp_mesh.entity_rank(linked_entity_crs);
                    auto &crs_bucket_conn =
                        crs_partition.get_crs_bucket_conn(linked_entity_crs_rank, linked_entity_crs_index.bucket_id);
                    Kokkos::atomic_store(&impl::get_dirty_flag(crs_bucket_conn),
                                         true);  // TODO: This should be a protected function (flag_as_dirty_atomically)
                  }

                  bool new_entity_is_valid = (linked_entity != stk::mesh::Entity());
                  if (new_entity_is_valid) {
                    // Mark the new linked entity's crs bucket conn as dirty
                    const stk::mesh::FastMeshIndex new_linked_entity_index = ngp_mesh.fast_mesh_index(linked_entity);
                    const stk::mesh::EntityRank linked_entity_rank = ngp_mesh.entity_rank(linked_entity);
                    auto &crs_bucket_conn =
                        crs_partition.get_crs_bucket_conn(linked_entity_rank, new_linked_entity_index.bucket_id);
                    Kokkos::atomic_store(&impl::get_dirty_flag(crs_bucket_conn),
                                         true);  // TODO: This should be a protected function (flag_as_dirty_atomically)
                  }
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  static void reset_dirty_linked_buckets(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                         const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::reset_dirty_linked_buckets");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    //  Reset dirty buckets: Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over
    //  the partitions, if its corresponding linked bucket has been modified, thread loop over the linked entities and
    //  reset the connectivity counts.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          "reset_dirty_linked_buckets", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, team.league_rank());
            unsigned bucket_size = bucket.size();

            // Serial loop over the partitions
            for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
              NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);

              // Fetch the crs bucket conn for this rank and bucket
              auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

              // If the bucket is dirty, reset the connectivity counts
              if (impl::get_dirty_flag(crs_bucket_conn)) {
                // Reset the connectivity counts for each entity in the bucket
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                     [&](const int &i) { impl::get_num_connected_links(crs_bucket_conn)(i) = 0; });
              }
            }
          });
    }

    Kokkos::Profiling::popRegion();
  }

  static void gather_part_1_count(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                  const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::gather_part_1_count");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Gather part 1 (count): Team loop over selected link buckets, fetch their partition, team loop over the links,
    // serial loop over the downward linked entities, if their bucket is dirty, atomically increment the connectivity
    // counts of the downward connected entities.

    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "gather_part_1_count", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            // Loop over the linked entities of this link
            for (unsigned d = 0; d < dimensionality; ++d) {
              // Only consider non-empty links
              if (coo_data.get_linked_entity(link_index, d) != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(link_index, d);
                auto &crs_bucket_conn =
                    crs_partition.get_crs_bucket_conn(linked_entity_rank, linked_entity_index.bucket_id);

                if (impl::get_dirty_flag(crs_bucket_conn)) {
                  // Atomically increment the connectivity count
                  Kokkos::atomic_add(&impl::get_num_connected_links(crs_bucket_conn)(linked_entity_index.bucket_ord),
                                     1u);
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  static void gather_part_2_partial_sum(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                        const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::gather_part_2_partial_sum");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Gather part 2 (partial sum): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop
    // over the partitions, if its corresponding linked bucket has been modified, thread loop over the linked bucket to
    // partial sum the connectivity counts into the connectivity offsets.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          "gather_part_2_partial_sum", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, team.league_rank());
            unsigned bucket_size = bucket.size();

            // Serial loop over the partitions
            for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
              NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);

              // Fetch the crs bucket conn for this rank and bucket
              auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

              // If the bucket is dirty, partial sum the connectivity counts into the connectivity offsets.
              if (impl::get_dirty_flag(crs_bucket_conn)) {
                // Use a parallel_scan to compute the offsets
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                      [&](unsigned i, unsigned &partial_sum, bool final_pass) {
                                        const unsigned num_connected_links =
                                            impl::get_num_connected_links(crs_bucket_conn)(i);
                                        if (final_pass) {
                                          // exclusive offset
                                          impl::get_sparse_connectivity_offsets(crs_bucket_conn)(i) = partial_sum;

                                          if (i == bucket_size - 1) {
                                            // Store the total number of connected links at the end of the offsets array
                                            impl::get_sparse_connectivity_offsets(crs_bucket_conn)(bucket_size) =
                                                partial_sum + num_connected_links;
                                          }
                                        }
                                        partial_sum += num_connected_links;
                                      });
                // Stash the total for access on the host
                impl::get_total_num_connected_links(crs_bucket_conn) =
                    impl::get_sparse_connectivity_offsets(crs_bucket_conn)(bucket_size);
              }
            }
          });
    }

    Kokkos::Profiling::popRegion();
  }

  static void scatter_part_1_setup(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                   const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_1_setup");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Scatter part 1 (setup): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop over
    // the partitions, if its corresponding linked bucket has been modified, reset the connectivity counts to zero.
    //
    //
    reset_dirty_linked_buckets(crs_data, coo_data, link_subset_selector);

    // Resize the bucket sparse connectivity arrays
    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
#pragma omp parallel for
    for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
      NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        // Only attempt to resize dirty buckets that have non-zero connections
        for (unsigned bucket_id = 0; bucket_id < crs_partition.num_buckets(rank); ++bucket_id) {
          auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
          if (impl::get_dirty_flag(crs_bucket_conn)) {
            // Only resize if needed
            unsigned new_size = impl::get_total_num_connected_links(crs_bucket_conn);
            if (new_size > impl::get_sparse_connectivity(crs_bucket_conn).extent(0)) {  // Only grow
              Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                             impl::get_sparse_connectivity(crs_bucket_conn), new_size);
            }
          }
        }
      }
    }

    Kokkos::Profiling::popRegion();
  }

  static void scatter_part_2_fill(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                  const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_2_fill");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Scatter part 2 (fill): Team loop over each selected link buckets, fetch
    // their partition ID, thread loop over the links, serial loop over their downward linked entities, and if their
    // bucket is dirty, scatter the link. Copy the link into the old field. Update the count as each entity is inserted.

    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto &team_policy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        "scatter_part_2_fill", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = bucket.size();

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          unsigned map_index = stk_link_bucket_to_partition_id_map.find(bucket_id);
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CRS partitions.");

          NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_links), [&](const int &i) {
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            // Loop over the linked entities of this link
            for (unsigned d = 0; d < dimensionality; ++d) {
              // Only consider non-empty links
              stk::mesh::Entity linked_entity = coo_data.get_linked_entity(link_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(link_index, d);
                stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(link_index, d);
                auto &crs_bucket_conn =
                    crs_partition.get_crs_bucket_conn(linked_entity_rank, linked_entity_index.bucket_id);

                if (impl::get_dirty_flag(crs_bucket_conn)) {
                  // Atomically increment the connectivity count
                  const unsigned offset =
                      impl::get_sparse_connectivity_offsets(crs_bucket_conn)(linked_entity_index.bucket_ord);
                  const unsigned num_inserted_old = Kokkos::atomic_fetch_add(
                      &impl::get_num_connected_links(crs_bucket_conn)(linked_entity_index.bucket_ord), 1);
                  impl::get_sparse_connectivity(crs_bucket_conn)(offset + num_inserted_old) = link;
                }
              }
            }
          });
        });

    Kokkos::Profiling::popRegion();
  }

  static void finalize_crs_update(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                  const stk::mesh::Selector &selector) {
    Kokkos::Profiling::pushRegion("NgpLinkPartitionT::scatter_part_3_finalize");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Finalize CRS update: Mark all buckets as no longer dirty, mark all selected links are up-to-date, and copy the
    // old COO connectivity to the new COO connectivity (for the given selector)

    // Serial loop over each rank, parallel loop over the stk buckets of said rank, serial loop over the partitions,
    // if its corresponding linked bucket has been modified, reset the dirty flag.
    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCRSPartitionView &crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Regular for loop over each stk bucket of said rank
      for (unsigned bucket_id = 0; bucket_id < ngp_mesh.num_buckets(rank); ++bucket_id) {
        // Serial loop over the partitions
        for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
          NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);

          // Fetch the crs bucket conn for this rank and bucket
          auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
          impl::get_dirty_flag(crs_bucket_conn) = false;  // Reset the dirty flag
        }
      }

      // TODO(palmerb4): It appears as though resetting the flag in a parallel_for is slower than doing it
      // serially (at least for a CPU build). Is this true for GPU builds too?

      // Regular parallel_for over each stk bucket of said rank
      // Kokkos::parallel_for("finalize_crs_update_reset_dirty_flag",
      //     Kokkos::RangePolicy<stk::mesh::NgpMesh::MeshExecSpace>(0, ngp_mesh.num_buckets(rank)),
      //     KOKKOS_LAMBDA(const int &bucket_id) {
      //       // Serial loop over the partitions
      //       for (unsigned partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
      //         NgpLinkCRSPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);

      //         // Fetch the crs bucket conn for this rank and bucket
      //         auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
      //         impl::get_dirty_flag(crs_bucket_conn) = false;  // Reset the dirty flag
      //       }
      //     });
    }

    // Mark all selected links as up-to-date
    auto &link_needs_updated_field = impl::get_link_crs_needs_updated_field(crs_data.link_meta_data());
    ::mundy::mesh::field_fill(0, link_needs_updated_field, link_subset_selector, stk::ngp::ExecSpace());

    // Copy the old COO connectivity to the new COO connectivity
    ::mundy::mesh::field_copy<entity_value_t>(impl::get_linked_entities_field(crs_data.link_meta_data()),
                                              impl::get_linked_entities_crs_field(crs_data.link_meta_data()),
                                              link_subset_selector, stk::ngp::ExecSpace());

    Kokkos::Profiling::popRegion();
  }

  /// \brief Check consistency between the COO and CRS connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CRS and CRS -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  static void check_crs_coo_consistency(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                        const stk::mesh::Selector &selector) {
    MUNDY_THROW_REQUIRE(crs_data.is_valid() && coo_data.is_valid(), std::invalid_argument,
                        "CRS and COO data must be valid to check consistency.");
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;
    check_all_links_in_sync(crs_data, coo_data, link_subset_selector);
    check_linked_bucket_conn_size(crs_data, coo_data, link_subset_selector);
    check_coo_to_crs_conn(crs_data, coo_data, link_subset_selector);
    check_crs_to_coo_conn(crs_data, coo_data, link_subset_selector);
  }

  /// \brief Check consistency between the COO and CRS connectivity for all links
  static void check_crs_coo_consistency(NgpLinkCRSDataT<NgpMemSpace> &crs_data,
                                        NgpLinkCOODataT<NgpMemSpace> &coo_data) {
    MUNDY_THROW_REQUIRE(crs_data.is_valid() && coo_data.is_valid(), std::invalid_argument,
                        "CRS and COO data must be valid to check consistency.");
    check_crs_coo_consistency(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part());
  }

  static void check_all_links_in_sync(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                      const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;
    int needs_updated_count = field_sum<int>(impl::get_link_crs_needs_updated_field(crs_data.link_meta_data()),
                                             link_subset_selector, stk::ngp::ExecSpace());
    MUNDY_THROW_REQUIRE(needs_updated_count == 0, std::logic_error, "There are still links that are out of sync.");
  }

  static void check_linked_bucket_conn_size(NgpLinkCRSDataT<NgpMemSpace> &crs_data,
                                            NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                            const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Serial loop over each selected partition. Serial loop over each rank.
    // Assert that the size of the bucket conn is the same as the number of STK buckets of the given rank.
    const NgpLinkCRSPartitionView &partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NgpLinkCRSPartitionT<NgpMemSpace> &partition = partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        unsigned num_buckets = partition.num_buckets(rank);
        unsigned num_stk_buckets = crs_data.bulk_data().buckets(rank).size();
        MUNDY_THROW_REQUIRE(num_buckets == num_stk_buckets, std::logic_error,
                            "The number of linked buckets does not match the number of STK buckets.");
      }
    }
  }

  static void check_coo_to_crs_conn(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                    const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Serial loop over each partial, hierarchical parallelism over each link in said selector,
    // serial loop over each of its downward connections, if it is non-empty, fetch their CRS conn,
    // serial loop over each link in the CRS conn, and check if it is the same as the source link.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCRSPartitionView &partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NgpLinkCRSPartitionT<NgpMemSpace> &partition = partitions(partition_id);
      const unsigned dimensionality = partition.link_dimensionality();
      stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();

      stk::mesh::for_each_entity_run(
          ngp_mesh, link_rank, partition.selector(), KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &linker_index) {
            // Loop over each linked entity in the linker
            for (unsigned d = 0; d < dimensionality; ++d) {
              stk::mesh::Entity linked_entity = coo_data.get_linked_entity(linker_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                // Fetch the CRS connectivity of the linked entity
                stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(linker_index, d);
                stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(linker_index, d);
                ConnectedEntities connected_links =
                    partition.get_connected_links(linked_entity_rank, linked_entity_index);

                MUNDY_THROW_REQUIRE(partition.num_connected_links(linked_entity_rank, linked_entity_index) > 0,
                                    std::logic_error,
                                    "A linked entity in the CRS connectivity is not connected to any links.");
                MUNDY_THROW_REQUIRE(
                    partition.num_connected_links(linked_entity_rank, linked_entity_index) == connected_links.size(),
                    std::logic_error,
                    "The number of connected links in the CRS connectivity does not match the size of the connected "
                    "links array.");

                // Loop over each connected link in the CRS connectivity
                bool found_link = false;
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::FastMeshIndex connected_link_index =
                      ngp_mesh.fast_mesh_index(connected_links[connected_link_ord]);
                  if (fma_equal(connected_link_index, linker_index)) {
                    found_link = true;
                    break;
                  }
                }

                MUNDY_THROW_REQUIRE(found_link, std::logic_error,
                                    "A linker in the CRS connectivity is missing from the COO connectivity.");
              }
            }
          });
    }
  }

  static void check_crs_to_coo_conn(NgpLinkCRSDataT<NgpMemSpace> &crs_data, NgpLinkCOODataT<NgpMemSpace> &coo_data,
                                    const stk::mesh::Selector &selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_part() & selector;

    // Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over each CRS partition,
    // fetch the corresponding CRS bucket conn, thread loop over the entities in said bucket, serial loop over their
    // connected links, and check if the source entity is linked to the link.

    const stk::mesh::NgpMesh &ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCRSPartitionView &partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      stk::NgpVector<unsigned> bucket_ids =
          ngp_mesh.get_bucket_ids(rank, crs_data.bulk_data().mesh_meta_data().universal_part());

      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto &team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(bucket_ids.size(), Kokkos::AUTO);

      Kokkos::parallel_for(
          "check_crs_to_coo_conn", team_policy, KOKKOS_LAMBDA(const TeamHandleType &team) {
            // Fetch our bucket
            const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
            const stk::mesh::NgpMesh::BucketType &bucket = ngp_mesh.get_bucket(rank, bucket_id);
            unsigned num_entities = bucket.size();

            // Serial loop over each partition
            unsigned num_partitions = partitions.extent(0);
            for (unsigned partition_id = 0; partition_id < num_partitions; ++partition_id) {
              const NgpLinkCRSPartitionT<NgpMemSpace> &partition = partitions(partition_id);
              const unsigned dimensionality = partition.link_dimensionality();

              // Thread loop over each entity in the bucket
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_entities), [&](const int &i) {
                stk::mesh::Entity entity = bucket[i];
                stk::mesh::FastMeshIndex entity_index = ngp_mesh.fast_mesh_index(entity);

                // Each connected link better be attached to us
                ConnectedEntities connected_links = partition.get_connected_links(rank, entity_index);
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::Entity connected_link = connected_links[connected_link_ord];
                  stk::mesh::FastMeshIndex connected_link_index = ngp_mesh.fast_mesh_index(connected_link);

                  MUNDY_THROW_REQUIRE(connected_link != stk::mesh::Entity(), std::logic_error,
                                      "A connected link in the CRS connectivity is empty.");

                  // Serial loop over each linked entity in the connected link
                  bool found_entity = false;
                  for (unsigned d = 0; d < dimensionality; ++d) {
                    stk::mesh::Entity linked_entity = coo_data.get_linked_entity(connected_link_index, d);
                    if (linked_entity == entity) {
                      found_entity = true;
                      break;
                    }
                  }

                  MUNDY_THROW_REQUIRE(found_entity, std::logic_error,
                                      "A linked entity in the COO connectivity is missing from the CRS connectivity.");
                }
              });
            }
          });
    }
  }

  KOKKOS_INLINE_FUNCTION
  static bool fma_equal(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
    return (lhs.bucket_id == rhs.bucket_id) && (lhs.bucket_ord == rhs.bucket_ord);
  }
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPCOOTOCRSSYNCHRONIZER_HPP_
