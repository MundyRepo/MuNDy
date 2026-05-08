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

#ifndef MUNDY_MESH_IMPL_NGPCOOTOCSRSYNCHRONIZER_HPP_
#define MUNDY_MESH_IMPL_NGPCOOTOCSRSYNCHRONIZER_HPP_

/// \file NgpCOOToCSRSynchronizerT.hpp
/// \brief Declaration of the NgpCOOToCSRSynchronizerT class

// C++ core libs
#include <chrono>       // for std::chrono profiling (temporary)
#include <cstdlib>      // for std::getenv (temporary)
#include <iostream>     // for std::cout profiling output (temporary)
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
#include <mundy_mesh/ForEachEntity.hpp>                    // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/LinkCSRPartition.hpp>                 // for mundy::mesh::LinkCSRPartition
#include <mundy_mesh/LinkMetaData.hpp>                     // for mundy::mesh::LinkMetaData
#include <mundy_mesh/MetaData.hpp>                         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>   // for mundy::mesh::field_copy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

namespace impl {

inline bool is_link_sync_profiling_enabled() {
  const char* env = std::getenv("MUNDY_LINKDATA_SYNC_PROFILE");
  return env != nullptr && std::string(env) == "1";
}

inline bool use_team_flat_for_gather_part_2_partial_sum() {
  const char* env = std::getenv("MUNDY_LINKDATA_GATHER2_TEAM_FLAT");
  return env != nullptr && std::string(env) == "1";
}

inline double elapsed_sync_profile_seconds(const std::chrono::steady_clock::time_point& begin,
                                           const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double>(end - begin).count();
}

class ScopedLinkSyncProfileTimer {
 public:
  explicit ScopedLinkSyncProfileTimer(const char* label)
      : label_(label), enabled_(is_link_sync_profiling_enabled()), begin_(std::chrono::steady_clock::now()) {
  }

  ~ScopedLinkSyncProfileTimer() {
    if (!enabled_) {
      return;
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "[LinkDataSyncProfile] " << label_ << "=" << elapsed_sync_profile_seconds(begin_, end) << " s\n";
  }

 private:
  const char* label_;
  bool enabled_;
  std::chrono::steady_clock::time_point begin_;
};

#define MUNDY_LINK_SYNC_PROFILE_SCOPE(label_literal) \
  ::mundy::mesh::impl::ScopedLinkSyncProfileTimer scoped_link_sync_profile_timer(label_literal)

// This class is really more like a namespace with similar methods, which exists because we want the CSR/COO
// to only be friends with this and not each individual method.
template <typename NgpMemSpace>
class NgpCOOToCSRSynchronizerT {
 public:
  //! \name Aliases
  //@{

  static_assert(Kokkos::is_memory_space_v<NgpMemSpace> &&
                    Kokkos::SpaceAccessibility<stk::ngp::ExecSpace, NgpMemSpace>::accessible,
                "NgpMemSpace must be a Kokkos memory space accessible from the device execution space.");

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

  template <typename T>
  using LinkCSRPartitionViewT = Kokkos::View<LinkCSRPartitionT<T>*, stk::ngp::UVMMemSpace>;

  template <typename T>
  using LinkBucketToPartitionIdMapT = Kokkos::UnorderedMap<unsigned, unsigned, T>;

  using NgpLinkCSRPartitionView = Kokkos::View<LinkCSRPartitionT<NgpMemSpace>*, stk::ngp::UVMMemSpace>;
  using LinkBucketToPartitionIdMap = Kokkos::UnorderedMap<unsigned, unsigned, NgpMemSpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.
  NgpCOOToCSRSynchronizerT() = default;

  /// \brief Default copy or move constructors/operators.
  NgpCOOToCSRSynchronizerT(const NgpCOOToCSRSynchronizerT&) = default;
  NgpCOOToCSRSynchronizerT(NgpCOOToCSRSynchronizerT&&) = default;
  NgpCOOToCSRSynchronizerT& operator=(const NgpCOOToCSRSynchronizerT&) = default;
  NgpCOOToCSRSynchronizerT& operator=(NgpCOOToCSRSynchronizerT&&) = default;

  /// \brief Destructor.
  virtual ~NgpCOOToCSRSynchronizerT() = default;
  //@}

  //! \name Methods
  //@{

  static bool has_stale_or_invalid_coo_relations(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                                 const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("has_stale_or_invalid_coo_relations");
    const stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;
    const stk::mesh::BulkData& bulk_data = crs_data.bulk_data();
    const LinkMetaData& link_meta_data = crs_data.link_meta_data();
    const auto& linked_entities_field = impl::get_linked_entities_field(link_meta_data);
    const auto& linked_entity_ids_field = impl::get_linked_entity_ids_field(link_meta_data);
    const auto& linked_entity_ranks_field = impl::get_linked_entity_ranks_field(link_meta_data);
    const auto& linked_entity_bucket_ids_field = impl::get_linked_entity_bucket_ids_field(link_meta_data);
    const auto& linked_entity_bucket_ords_field = impl::get_linked_entity_bucket_ords_field(link_meta_data);

    const stk::mesh::BucketVector& link_buckets =
        bulk_data.get_buckets(link_meta_data.link_rank(), link_subset_selector);
    for (const stk::mesh::Bucket* bucket : link_buckets) {
      const unsigned link_dimensionality = stk::mesh::field_scalars_per_entity(linked_entities_field, *bucket);
      for (size_t i = 0; i < bucket->size(); ++i) {
        const stk::mesh::Entity link = (*bucket)[i];
        const auto* linked_entities_data = stk::mesh::field_data(linked_entities_field, link);
        const auto* linked_entity_ids_data = stk::mesh::field_data(linked_entity_ids_field, link);
        const auto* linked_entity_ranks_data = stk::mesh::field_data(linked_entity_ranks_field, link);
        const auto* linked_entity_bucket_ids_data = stk::mesh::field_data(linked_entity_bucket_ids_field, link);
        const auto* linked_entity_bucket_ords_data = stk::mesh::field_data(linked_entity_bucket_ords_field, link);

        for (unsigned d = 0; d < link_dimensionality; ++d) {
          const stk::mesh::Entity linked_entity(linked_entities_data[d]);
          const stk::mesh::EntityId stored_id = linked_entity_ids_data[d];
          const LinkMetaData::entity_rank_value_t stored_rank_value = linked_entity_ranks_data[d];
          const bool empty_handle = linked_entity == stk::mesh::Entity();
          const bool empty_id_rank =
              stored_id == stk::mesh::EntityId() &&
              stored_rank_value == static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);

          if (empty_handle || empty_id_rank) {
            if (!empty_handle || !empty_id_rank || linked_entity_bucket_ids_data[d] != 0u ||
                linked_entity_bucket_ords_data[d] != 0u) {
              return true;
            }
            continue;
          }

          const bool stored_rank_valid =
              stored_rank_value >= static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::BEGIN_RANK) &&
              stored_rank_value < static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::NUM_RANKS);
          if (!stored_rank_valid) {
            return true;
          }

          if (!bulk_data.is_valid(linked_entity)) {
            return true;
          }

          const stk::mesh::EntityRank stored_rank = static_cast<stk::mesh::EntityRank>(stored_rank_value);
          if (bulk_data.identifier(linked_entity) != stored_id || bulk_data.entity_rank(linked_entity) != stored_rank) {
            return true;
          }

          if (bulk_data.get_entity(stored_rank, stored_id) != linked_entity) {
            return true;
          }

          const unsigned current_bucket_id = bulk_data.bucket(linked_entity).bucket_id();
          const unsigned current_bucket_ord = bulk_data.bucket_ordinal(linked_entity);
          if (linked_entity_bucket_ids_data[d] != current_bucket_id ||
              linked_entity_bucket_ords_data[d] != current_bucket_ord) {
            return true;
          }
        }
      }
    }

    return false;
  }

  /// \brief Check if the CSR connectivity is up-to-date for the given link subset selector.
  ///
  /// \note This check is more than just a lookup of a flag. Instead, it performs two operations
  ///  1. A reduction over all selected partitions to check if any of the CSR buckets are dirty.
  ///  2. A reduction over all selected links to check if any of the links are dirty.
  /// These aren't expensive operations and they're designed to be fast/GPU-compatible, but they aren't free.
  static bool is_crs_up_to_date(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& /*coo_data*/,
                                const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("is_crs_up_to_date(selector)");

    // Dereference just once
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;
    const stk::mesh::BucketVector& selected_link_buckets =
        crs_data.bulk_data().get_buckets(crs_data.link_meta_data().link_rank(), link_subset_selector);

    if (crs_data.get_all_crs_partitions().extent(0) == 0 && !selected_link_buckets.empty()) {
      return false;
    }

    // Two types of out-of-date:
    //  1. The CSR connectivity of a selected partition is dirty.
    //    - Team loop over each selected partition and thread loop over each bucket in the partition. If any bucket is
    //    dirty, atomically set the needs updated flag to true.
    const NgpLinkCSRPartitionView& partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    size_t num_partitions = partitions.extent(0);
    bool crs_buckets_up_to_date = true;
    for (size_t i = 0; i < num_partitions; ++i) {
      const NgpLinkCSRPartitionT<NgpMemSpace>& partition = partitions(i);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        const size_t num_buckets = partition.num_buckets(rank);
        for (size_t bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
          const auto& crs_bucket_conn = partition.get_crs_bucket_conn(rank, bucket_index);
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
    //     "NgpCOOToCSRSynchronizerT::is_crs_up_to_date", team_policy,
    //     KOKKOS_LAMBDA(const TeamHandleType &team, int &team_local_count) {
    //       const stk::mesh::Ordinal partition_id = team.league_rank();
    //       const NgpLinkCSRPartitionT<NgpMemSpace> &partition = partitions(partition_id);

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

    bool is_up_to_date = crs_buckets_up_to_date;
    if (is_up_to_date) {  // No need to perform the second check if the first fails.
      //  2. A selected link is out-of-date.
      int link_needs_updated_count =
          ::mundy::mesh::field_sum<int>(impl::get_link_crs_needs_updated_field(crs_data.link_meta_data()),
                                        link_subset_selector, stk::ngp::ExecSpace());
      const bool links_up_to_date = (link_needs_updated_count == 0);
      is_up_to_date = links_up_to_date;

    }

    return is_up_to_date;
  }

  /// \brief Check if the CSR connectivity is up-to-date for all links.
  static bool is_crs_up_to_date(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("is_crs_up_to_date(universal)");
    return is_crs_up_to_date(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part());
  }

  /// \brief Propagate changes made to the COO connectivity to the CSR connectivity for the given link subset selector.
  /// This takes changes made via the declare/destroy_relation functions or request/destroy links and updates
  /// the CSR connectivity to reflect these changes.
  static void update_crs_from_coo(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                  const stk::mesh::Selector& selector, bool force_full_rebuild = false) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("update_crs_from_coo(selector)");
    const bool sync_profile_enabled = is_link_sync_profiling_enabled();
    const auto sync_total_begin = std::chrono::steady_clock::now();

    double is_up_to_date_time = 0.0;
    double validation_time = 0.0;
    double prepare_rebuild_time = 0.0;
    double flag_dirty_time = 0.0;
    double reset_dirty_time = 0.0;
    double gather_count_time = 0.0;
    double gather_prefix_time = 0.0;
    double scatter_setup_time = 0.0;
    double scatter_fill_time = 0.0;
    double finalize_time = 0.0;

    stk::mesh::Selector universal_link_selector = crs_data.link_meta_data().universal_link_class();
    stk::mesh::Selector link_subset_selector = universal_link_selector & selector;
    if (force_full_rebuild) {
      link_subset_selector = universal_link_selector;
    }

    if (!force_full_rebuild) {
      const auto up_to_date_begin = std::chrono::steady_clock::now();
      if (is_crs_up_to_date(crs_data, coo_data, link_subset_selector)) {
        if (sync_profile_enabled) {
          const auto sync_total_end = std::chrono::steady_clock::now();
          std::cout << "[LinkDataSyncProfile] update_crs_from_coo early-return up-to-date in "
                    << elapsed_sync_profile_seconds(sync_total_begin, sync_total_end) << " s\n";
        }
        return;
      }
      const auto up_to_date_end = std::chrono::steady_clock::now();
      is_up_to_date_time += elapsed_sync_profile_seconds(up_to_date_begin, up_to_date_end);

    }

    if (force_full_rebuild) {
      const auto validation_begin = std::chrono::steady_clock::now();
      MUNDY_THROW_REQUIRE(!has_stale_or_invalid_coo_relations(crs_data, link_subset_selector), std::logic_error,
                          "Cannot rebuild link CSR data because one or more COO relations reference an invalid entity, "
                          "an entity with mismatched id/rank metadata, or stale bucket id/ordinal metadata.");
      const auto validation_end = std::chrono::steady_clock::now();
      validation_time += elapsed_sync_profile_seconds(validation_begin, validation_end);
    }

    if (force_full_rebuild) {
      const auto rebuild_begin = std::chrono::steady_clock::now();
      prepare_full_rebuild_state(crs_data);
      const auto rebuild_end = std::chrono::steady_clock::now();
      prepare_rebuild_time += elapsed_sync_profile_seconds(rebuild_begin, rebuild_end);
    }

    // Incremental updates only need to flag buckets touched by dirty COO links. Full rebuilds have already recreated
    // the CSR partition structure and marked every CSR bucket connection dirty.
    if (!force_full_rebuild) {
      const auto flag_begin = std::chrono::steady_clock::now();
      flag_dirty_linked_buckets_of_modified_links(crs_data, coo_data, link_subset_selector);
      const auto flag_end = std::chrono::steady_clock::now();
      flag_dirty_time += elapsed_sync_profile_seconds(flag_begin, flag_end);
    }

    const auto reset_begin = std::chrono::steady_clock::now();
    reset_dirty_linked_buckets(crs_data, coo_data, link_subset_selector);
    const auto reset_end = std::chrono::steady_clock::now();
    reset_dirty_time += elapsed_sync_profile_seconds(reset_begin, reset_end);

    const auto gather_count_begin = std::chrono::steady_clock::now();
    gather_part_1_count(crs_data, coo_data, link_subset_selector);
    const auto gather_count_end = std::chrono::steady_clock::now();
    gather_count_time += elapsed_sync_profile_seconds(gather_count_begin, gather_count_end);

    const auto gather_prefix_begin = std::chrono::steady_clock::now();
    gather_part_2_partial_sum(crs_data, coo_data, link_subset_selector);
    const auto gather_prefix_end = std::chrono::steady_clock::now();
    gather_prefix_time += elapsed_sync_profile_seconds(gather_prefix_begin, gather_prefix_end);

    const auto scatter_setup_begin = std::chrono::steady_clock::now();
    scatter_part_1_setup(crs_data, coo_data, link_subset_selector);
    const auto scatter_setup_end = std::chrono::steady_clock::now();
    scatter_setup_time += elapsed_sync_profile_seconds(scatter_setup_begin, scatter_setup_end);

    const auto scatter_fill_begin = std::chrono::steady_clock::now();
    scatter_part_2_fill(crs_data, coo_data, link_subset_selector);
    const auto scatter_fill_end = std::chrono::steady_clock::now();
    scatter_fill_time += elapsed_sync_profile_seconds(scatter_fill_begin, scatter_fill_end);

    const auto finalize_begin = std::chrono::steady_clock::now();
    finalize_crs_update(crs_data, coo_data, link_subset_selector);
    const auto finalize_end = std::chrono::steady_clock::now();
    finalize_time += elapsed_sync_profile_seconds(finalize_begin, finalize_end);

    if (sync_profile_enabled) {
      const auto sync_total_end = std::chrono::steady_clock::now();
      std::cout << "[LinkDataSyncProfile] update_crs_from_coo total="
                << elapsed_sync_profile_seconds(sync_total_begin, sync_total_end) << " s"
                << " (force_full_rebuild=" << (force_full_rebuild ? "true" : "false") << ")\n"
                << "[LinkDataSyncProfile]   is_crs_up_to_date=" << is_up_to_date_time << " s\n"
                << "[LinkDataSyncProfile]   validate_coo_relations=" << validation_time << " s\n"
                << "[LinkDataSyncProfile]   prepare_full_rebuild_state=" << prepare_rebuild_time << " s\n"
                << "[LinkDataSyncProfile]   flag_dirty_linked_buckets_of_modified_links=" << flag_dirty_time << " s\n"
                << "[LinkDataSyncProfile]   reset_dirty_linked_buckets=" << reset_dirty_time << " s\n"
                << "[LinkDataSyncProfile]   gather_part_1_count=" << gather_count_time << " s\n"
                << "[LinkDataSyncProfile]   gather_part_2_partial_sum=" << gather_prefix_time << " s\n"
                << "[LinkDataSyncProfile]   scatter_part_1_setup=" << scatter_setup_time << " s\n"
                << "[LinkDataSyncProfile]   scatter_part_2_fill=" << scatter_fill_time << " s\n"
                << "[LinkDataSyncProfile]   finalize_crs_update=" << finalize_time << " s\n";
    }

// If in debug, check consistency
#ifndef NDEBUG
    check_crs_coo_consistency(crs_data, coo_data, link_subset_selector);
#endif
  }

  /// \brief Propagate changes made to the COO connectivity to the CSR connectivity.
  static void update_crs_from_coo(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                  bool force_full_rebuild = false) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("update_crs_from_coo(universal)");
    update_crs_from_coo(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part(), force_full_rebuild);
  }

  static void prepare_full_rebuild_state(NgpLinkCSRDataT<NgpMemSpace>& crs_data) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("prepare_full_rebuild_state");
    const stk::mesh::Selector universal_link_selector = crs_data.link_meta_data().universal_link_class();
    crs_data.clear_structural_caches();
    crs_data.get_or_create_crs_partitions(universal_link_selector);
    crs_data.update_stk_link_bucket_to_partition_id_map();
    crs_data.mark_all_crs_bucket_conns_dirty();
  }

  static void flag_dirty_linked_buckets_of_modified_links(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                                          NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                                          const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("flag_dirty_linked_buckets_of_modified_links");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Flag dirty buckets: Team loop over selected link buckets, fetch their partition, thread loop over links,
    // determine if any of those links are flagged as modified. If so, determine if their links were created or
    // destroyed. Flag the linked bucket of new or deleted entities as dirty.

    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto& team_policy =
        stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(static_cast<int>(bucket_ids.size()), Kokkos::AUTO);

    Kokkos::parallel_for(
        "flag_dirty_linked_buckets_of_modified_links", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          const unsigned num_links = static_cast<unsigned>(bucket.size());

          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");
          const unsigned map_index = static_cast<unsigned>(stk_link_bucket_to_partition_id_map.find(bucket_id));
          const stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CSR partitions.");

          NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);
          const unsigned dimensionality = crs_partition.link_dimensionality();

          // TEAM_FLAT path: flatten (link,ordinal) work into a single TeamThreadRange to improve utilization.
          const unsigned work_items = num_links * dimensionality;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, work_items), [&](const unsigned work) {
            const unsigned i = work / dimensionality;
            const unsigned d = work % dimensionality;
            const stk::mesh::Entity link = bucket[i];
            const stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            if (!coo_data.get_link_crs_needs_updated(link_index)) {
              return;
            }

            const stk::mesh::Entity linked_entity_crs = coo_data.get_linked_entity_crs(link_index, d);
            const stk::mesh::Entity linked_entity = coo_data.get_linked_entity(link_index, d);
            if (linked_entity_crs == linked_entity) {
              return;
            }

            if (linked_entity_crs != stk::mesh::Entity()) {
              const stk::mesh::FastMeshIndex linked_entity_crs_index = ngp_mesh.fast_mesh_index(linked_entity_crs);
              const stk::mesh::EntityRank linked_entity_crs_rank = ngp_mesh.entity_rank(linked_entity_crs);
              auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(
                  linked_entity_crs_rank, static_cast<unsigned>(linked_entity_crs_index.bucket_id));
              Kokkos::atomic_store(&impl::get_dirty_flag(crs_bucket_conn), true);
            }

            if (linked_entity != stk::mesh::Entity()) {
              const stk::mesh::FastMeshIndex linked_entity_index = ngp_mesh.fast_mesh_index(linked_entity);
              const stk::mesh::EntityRank linked_entity_rank = ngp_mesh.entity_rank(linked_entity);
              auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(
                  linked_entity_rank, static_cast<unsigned>(linked_entity_index.bucket_id));
              Kokkos::atomic_store(&impl::get_dirty_flag(crs_bucket_conn), true);
            }
          });
        });
  }

  static void reset_dirty_linked_buckets(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                         NgpLinkCOODataT<NgpMemSpace>& /*coo_data*/,
                                         const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("reset_dirty_linked_buckets");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    //  Reset dirty buckets: Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over
    //  the partitions, if its corresponding linked bucket has been modified, thread loop over the linked entities and
    //  reset the connectivity counts.

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Team loop over each stk bucket of said rank
      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto& team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngp_mesh.num_buckets(rank), Kokkos::AUTO);
      Kokkos::parallel_for(
          "reset_dirty_linked_buckets", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
            // Fetch our bucket
            const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(rank, team.league_rank());
            unsigned bucket_size = static_cast<unsigned>(bucket.size());

            // Serial loop over the partitions
            for (size_t partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
              NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);

              // Fetch the crs bucket conn for this rank and bucket
              auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

              // If the bucket is dirty, reset the connectivity counts
              if (impl::get_dirty_flag(crs_bucket_conn)) {
                // Reset the connectivity counts for each entity in the bucket
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                     [&](const int& i) { impl::get_num_connected_links(crs_bucket_conn)(i) = 0; });
              }
            }
          });
    }
  }

  static void gather_part_1_count(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                  const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("gather_part_1_count");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Gather part 1 (count): Team loop over selected link buckets, fetch their partition, team loop over the links,
    // serial loop over the downward linked entities, if their bucket is dirty, atomically increment the connectivity
    // counts of the downward connected entities.

    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto& team_policy =
        stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(static_cast<int>(bucket_ids.size()), Kokkos::AUTO);

    Kokkos::parallel_for(
        "gather_part_1_count", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          const unsigned num_links = static_cast<unsigned>(bucket.size());

          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          const unsigned map_index = static_cast<unsigned>(stk_link_bucket_to_partition_id_map.find(bucket_id));
          const stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CSR partitions.");

          NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);
          const unsigned dimensionality = crs_partition.link_dimensionality();

          // TEAM_FLAT path: flatten (link,ordinal) work into a single TeamThreadRange to improve utilization.
          const unsigned work_items = num_links * dimensionality;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, work_items), [&](const unsigned work) {
            const unsigned i = work / dimensionality;
            const unsigned d = work % dimensionality;
            const stk::mesh::Entity link = bucket[i];
            const stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);
            if (coo_data.get_linked_entity(link_index, d) == stk::mesh::Entity()) {
              return;
            }
            const stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(link_index, d);
            const stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(link_index, d);
            auto& crs_bucket_conn =
                crs_partition.get_crs_bucket_conn(linked_entity_rank, linked_entity_index.bucket_id);

            if (impl::get_dirty_flag(crs_bucket_conn)) {
              Kokkos::atomic_add(&impl::get_num_connected_links(crs_bucket_conn)(linked_entity_index.bucket_ord), 1u);
            }
          });
        });
  }

  static void gather_part_2_partial_sum(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                        NgpLinkCOODataT<NgpMemSpace>& /*coo_data*/,
                                        const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("gather_part_2_partial_sum");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Gather part 2 (partial sum): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop
    // over the partitions, if its corresponding linked bucket has been modified, thread loop over the linked bucket to
    // partial sum the connectivity counts into the connectivity offsets.

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // TEAM_FLAT: Team loop over each stk bucket/partition pair
      using TeamHandleType = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type;
      const unsigned num_buckets = static_cast<unsigned>(ngp_mesh.num_buckets(rank));
      const unsigned num_partitions = static_cast<unsigned>(crs_partitions.extent(0));
      const unsigned work_items = num_buckets * num_partitions;
      const auto& team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(static_cast<int>(work_items), Kokkos::AUTO);
      Kokkos::parallel_for(
          "gather_part_2_partial_sum", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
            const unsigned work = static_cast<unsigned>(team.league_rank());
            const unsigned bucket_id = work / num_partitions;
            const unsigned partition_id = work % num_partitions;

            const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(rank, bucket_id);
            const unsigned bucket_size = static_cast<unsigned>(bucket.size());
            NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);
            auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket.bucket_id());

            if (impl::get_dirty_flag(crs_bucket_conn)) {
              Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, 0u, bucket_size),
                                    [&](unsigned i, unsigned& partial_sum, bool final_pass) {
                                      const unsigned num_connected_links =
                                          impl::get_num_connected_links(crs_bucket_conn)(i);
                                      if (final_pass) {
                                        impl::get_sparse_connectivity_offsets(crs_bucket_conn)(i) = partial_sum;
                                        if (i == bucket_size - 1) {
                                          impl::get_sparse_connectivity_offsets(crs_bucket_conn)(bucket_size) =
                                              partial_sum + num_connected_links;
                                        }
                                      }
                                      partial_sum += num_connected_links;
                                    });
              impl::get_total_num_connected_links(crs_bucket_conn) =
                  impl::get_sparse_connectivity_offsets(crs_bucket_conn)(bucket_size);
            }
          });
    }
  }

  static void scatter_part_1_setup(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                   const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("scatter_part_1_setup");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Scatter part 1 (setup): Serial loop over each rank, team loop over the stk buckets of said rank, serial loop over
    // the partitions, if its corresponding linked bucket has been modified, reset the connectivity counts to zero so
    // scatter_part_2_fill() can reuse num_connected_links as an insertion cursor.
    //
    // Resize the sparse connectivity arrays: OpenMP loop over the partitions, serial loop over each rank, and serial
    // loop over each bucket. If its corresponding linked bucket has been modified, grow the sparse connectivity array
    // to the required size.
    //

    reset_dirty_linked_buckets(crs_data, coo_data, link_subset_selector);

    // Resize the bucket sparse connectivity arrays
    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
#pragma omp parallel for
    for (size_t partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
      NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        // Only attempt to resize dirty buckets that have non-zero connections
        for (unsigned bucket_id = 0; bucket_id < crs_partition.num_buckets(rank); ++bucket_id) {
          auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
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
  }

  static void scatter_part_2_fill(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                  const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("scatter_part_2_fill");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Scatter part 2 (fill): Team loop over each selected link buckets, fetch
    // their partition ID, thread loop over the links, serial loop over their downward linked entities, and if their
    // bucket is dirty, scatter the link. Copy the link into the old field. Update the count as each entity is inserted.

    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    auto stk_link_bucket_to_partition_id_map = crs_data.get_updated_stk_link_bucket_to_partition_id_map();

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();
    stk::NgpVector<unsigned> bucket_ids = ngp_mesh.get_bucket_ids(link_rank, link_subset_selector);

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto& team_policy =
        stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(static_cast<int>(bucket_ids.size()), Kokkos::AUTO);

    Kokkos::parallel_for(
        "scatter_part_2_fill", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
          // Fetch our bucket
          const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
          const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(link_rank, bucket_id);
          unsigned num_links = static_cast<unsigned>(bucket.size());

          // Fetch the partition for this bucket
          MUNDY_THROW_ASSERT(stk_link_bucket_to_partition_id_map.exists(bucket_id), std::out_of_range,
                             "Bucket ID not found in the link bucket to partition ID map.");

          unsigned map_index = static_cast<unsigned>(stk_link_bucket_to_partition_id_map.find(bucket_id));
          stk::mesh::Ordinal partition_id = stk_link_bucket_to_partition_id_map.value_at(map_index);
          MUNDY_THROW_ASSERT(partition_id < crs_partitions.extent(0), std::out_of_range,
                             "Partition ID is out of range for the number of CSR partitions.");

          NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);
          unsigned dimensionality = crs_partition.link_dimensionality();

          // TEAM_FLAT path: flatten (link,ordinal) work into a single TeamThreadRange to improve utilization.
          const unsigned work_items = num_links * dimensionality;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, work_items), [&](const unsigned work) {
            const unsigned i = work / dimensionality;
            const unsigned d = work % dimensionality;
            stk::mesh::Entity link = bucket[i];
            stk::mesh::FastMeshIndex link_index = ngp_mesh.fast_mesh_index(link);

            // Loop over the linked entities of this link + Only consider non-empty links
            stk::mesh::Entity linked_entity = coo_data.get_linked_entity(link_index, d);
            if (linked_entity != stk::mesh::Entity()) {
              stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(link_index, d);
              stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(link_index, d);
              auto& crs_bucket_conn =
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
          });
        });
  }

  static void finalize_crs_update(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& /*coo_data*/,
                                  const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("finalize_crs_update");

    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Finalize CSR update: Mark all buckets as no longer dirty, mark all selected links are up-to-date, and copy the
    // old COO connectivity to the new COO connectivity (for the given selector)

    // Serial loop over each rank, parallel loop over the stk buckets of said rank, serial loop over the partitions,
    // if its corresponding linked bucket has been modified, reset the dirty flag.
    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCSRPartitionView& crs_partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    // Serial loop over each rank
    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      // Regular for loop over each stk bucket of said rank
      for (unsigned bucket_id = 0; bucket_id < ngp_mesh.num_buckets(rank); ++bucket_id) {
        // Serial loop over the partitions
        for (size_t partition_id = 0; partition_id < crs_partitions.extent(0); ++partition_id) {
          NgpLinkCSRPartitionT<NgpMemSpace>& crs_partition = crs_partitions(partition_id);

          // Fetch the crs bucket conn for this rank and bucket
          auto& crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
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
      //         NgpLinkCSRPartitionT<NgpMemSpace> &crs_partition = crs_partitions(partition_id);

      //         // Fetch the crs bucket conn for this rank and bucket
      //         auto &crs_bucket_conn = crs_partition.get_crs_bucket_conn(rank, bucket_id);
      //         impl::get_dirty_flag(crs_bucket_conn) = false;  // Reset the dirty flag
      //       }
      //     });
    }

    // Mark all selected links as up-to-date
    auto& link_needs_updated_field = impl::get_link_crs_needs_updated_field(crs_data.link_meta_data());
    ::mundy::mesh::field_fill(0, link_needs_updated_field, link_subset_selector, stk::ngp::ExecSpace());

    // Copy the old COO connectivity to the new COO connectivity
    ::mundy::mesh::field_copy<entity_value_t>(impl::get_linked_entities_field(crs_data.link_meta_data()),
                                              impl::get_linked_entities_crs_field(crs_data.link_meta_data()),
                                              link_subset_selector, stk::ngp::ExecSpace());
  }

  /// \brief Check consistency between the COO and CSR connectivity for the given selector
  ///
  /// Relatively expensive check that verifies COO -> CSR and CSR -> COO consistency.
  ///
  /// \note The checks performed in this function are performed even in RELEASE mode.
  static void check_crs_coo_consistency(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                        const stk::mesh::Selector& selector) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("check_crs_coo_consistency(selector)");
    MUNDY_THROW_REQUIRE(crs_data.is_valid() && coo_data.is_valid(), std::invalid_argument,
                        "CSR and COO data must be valid to check consistency.");
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;
    check_all_links_in_sync(crs_data, coo_data, link_subset_selector);
    check_linked_bucket_conn_size(crs_data, coo_data, link_subset_selector);
    check_coo_to_crs_conn(crs_data, coo_data, link_subset_selector);
    check_crs_to_coo_conn(crs_data, coo_data, link_subset_selector);
  }

  /// \brief Check consistency between the COO and CSR connectivity for all links
  static void check_crs_coo_consistency(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                        NgpLinkCOODataT<NgpMemSpace>& coo_data) {
    MUNDY_LINK_SYNC_PROFILE_SCOPE("check_crs_coo_consistency(universal)");
    MUNDY_THROW_REQUIRE(crs_data.is_valid() && coo_data.is_valid(), std::invalid_argument,
                        "CSR and COO data must be valid to check consistency.");
    check_crs_coo_consistency(crs_data, coo_data, crs_data.bulk_data().mesh_meta_data().universal_part());
  }

  static void check_all_links_in_sync(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                      const stk::mesh::Selector& selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;
    int needs_updated_count = field_sum<int>(impl::get_link_crs_needs_updated_field(crs_data.link_meta_data()),
                                             link_subset_selector, stk::ngp::ExecSpace());
    MUNDY_THROW_REQUIRE(needs_updated_count == 0, std::logic_error, "There are still links that are out of sync.");
  }

  static void check_linked_bucket_conn_size(NgpLinkCSRDataT<NgpMemSpace>& crs_data,
                                            NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                            const stk::mesh::Selector& selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Serial loop over each selected partition. Serial loop over each rank.
    // Assert that the size of the bucket conn is the same as the number of STK buckets of the given rank.
    const NgpLinkCSRPartitionView& partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NgpLinkCSRPartitionT<NgpMemSpace>& partition = partitions(partition_id);
      for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        unsigned num_buckets = partition.num_buckets(rank);
        unsigned num_stk_buckets = crs_data.bulk_data().buckets(rank).size();
        MUNDY_THROW_REQUIRE(num_buckets == num_stk_buckets, std::logic_error,
                            "The number of linked buckets does not match the number of STK buckets.");
      }
    }
  }

  static void check_coo_to_crs_conn(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                    const stk::mesh::Selector& selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Serial loop over each partial, hierarchical parallelism over each link in said selector,
    // serial loop over each of its downward connections, if it is non-empty, fetch their CSR conn,
    // serial loop over each link in the CSR conn, and check if it is the same as the source link.

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCSRPartitionView& partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);
    for (unsigned partition_id = 0; partition_id < partitions.extent(0); ++partition_id) {
      const NgpLinkCSRPartitionT<NgpMemSpace>& partition = partitions(partition_id);
      const unsigned dimensionality = partition.link_dimensionality();
      stk::mesh::EntityRank link_rank = crs_data.link_meta_data().link_rank();

      stk::mesh::for_each_entity_run(
          ngp_mesh, link_rank, partition.selector(), KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker_index) {
            // Loop over each linked entity in the linker
            for (unsigned d = 0; d < dimensionality; ++d) {
              stk::mesh::Entity linked_entity = coo_data.get_linked_entity(linker_index, d);
              if (linked_entity != stk::mesh::Entity()) {
                // Fetch the CSR connectivity of the linked entity
                stk::mesh::EntityRank linked_entity_rank = coo_data.get_linked_entity_rank(linker_index, d);
                stk::mesh::FastMeshIndex linked_entity_index = coo_data.get_linked_entity_index(linker_index, d);
                ConnectedEntities connected_links =
                    partition.get_connected_links(linked_entity_rank, linked_entity_index);

                MUNDY_THROW_REQUIRE(partition.num_connected_links(linked_entity_rank, linked_entity_index) > 0,
                                    std::logic_error,
                                    "A linked entity in the CSR connectivity is not connected to any links.");
                MUNDY_THROW_REQUIRE(
                    partition.num_connected_links(linked_entity_rank, linked_entity_index) == connected_links.size(),
                    std::logic_error,
                    "The number of connected links in the CSR connectivity does not match the size of the connected "
                    "links array.");

                // Loop over each connected link in the CSR connectivity
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
                                    "A linker in the CSR connectivity is missing from the COO connectivity.");
              }
            }
          });
    }
  }

  static void check_crs_to_coo_conn(NgpLinkCSRDataT<NgpMemSpace>& crs_data, NgpLinkCOODataT<NgpMemSpace>& coo_data,
                                    const stk::mesh::Selector& selector) {
    stk::mesh::Selector link_subset_selector = crs_data.link_meta_data().universal_link_class() & selector;

    // Serial loop over each rank, team loop over each stk bucket of said rank, serial loop over each CSR partition,
    // fetch the corresponding CSR bucket conn, thread loop over the entities in said bucket, serial loop over their
    // connected links, and check if the source entity is linked to the link.

    const stk::mesh::NgpMesh& ngp_mesh = stk::mesh::get_updated_ngp_mesh(crs_data.bulk_data());
    const NgpLinkCSRPartitionView& partitions = crs_data.get_or_create_crs_partitions(link_subset_selector);

    for (stk::topology::rank_t rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      stk::NgpVector<unsigned> bucket_ids =
          ngp_mesh.get_bucket_ids(rank, crs_data.bulk_data().mesh_meta_data().universal_part());

      typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
      const auto& team_policy =
          stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(static_cast<int>(bucket_ids.size()), Kokkos::AUTO);

      Kokkos::parallel_for(
          "check_crs_to_coo_conn", team_policy, KOKKOS_LAMBDA(const TeamHandleType& team) {
            // Fetch our bucket
            const unsigned bucket_id = bucket_ids.get<stk::mesh::NgpMesh::MeshExecSpace>(team.league_rank());
            const stk::mesh::NgpMesh::BucketType& bucket = ngp_mesh.get_bucket(rank, bucket_id);
            unsigned num_entities = static_cast<unsigned>(bucket.size());

            // Serial loop over each partition
            size_t num_partitions = partitions.extent(0);
            for (unsigned partition_id = 0; partition_id < num_partitions; ++partition_id) {
              const NgpLinkCSRPartitionT<NgpMemSpace>& partition = partitions(partition_id);
              const unsigned dimensionality = partition.link_dimensionality();

              // Thread loop over each entity in the bucket
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_entities), [&](const int& i) {
                stk::mesh::Entity entity = bucket[i];
                stk::mesh::FastMeshIndex entity_index = ngp_mesh.fast_mesh_index(entity);

                // Each connected link better be attached to us
                ConnectedEntities connected_links = partition.get_connected_links(rank, entity_index);
                for (unsigned connected_link_ord = 0; connected_link_ord < connected_links.size();
                     ++connected_link_ord) {
                  stk::mesh::Entity connected_link = connected_links[connected_link_ord];
                  stk::mesh::FastMeshIndex connected_link_index = ngp_mesh.fast_mesh_index(connected_link);

                  MUNDY_THROW_REQUIRE(connected_link != stk::mesh::Entity(), std::logic_error,
                                      "A connected link in the CSR connectivity is empty.");

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
                                      "A linked entity in the COO connectivity is missing from the CSR connectivity.");
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

#endif  // MUNDY_MESH_IMPL_NGPCOOTOCSRSYNCHRONIZER_HPP_
