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

#ifndef MUNDY_MESH_FOREACHENTITY_HPP_
#define MUNDY_MESH_FOREACHENTITY_HPP_

/// \file ForEachEntity.hpp
/// \brief Wrappers for STK's for_each_entity_run function that do a better job of detecting NGP vs non-ngp runs.

// C++ core
#include <type_traits>  // for std::is_base_of

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>          // for stk::mesh::BulkData
#include <stk_mesh/base/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/NgpForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_util/ngp/NgpSpaces.hpp>          // for stk::ngp::TeamPolicy

// Mundy
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/EntityIndices.hpp>  // for mundy::mesh::get_local_bucket_ids
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename T>
inline constexpr bool always_false_v = false;

}  // namespace impl

/// \brief NGP for_each_entity_run over the entities of a (rank, selector) chunk.
///
/// Equivalent to stk::mesh::for_each_entity_run, except that the selector's bucket ids come from the memoized
/// get_local_bucket_ids rather than being rebuilt (allocated, filled, and copied to device) on every call.
template <typename Mesh, typename AlgorithmPerEntity, typename EXEC_SPACE>
MUNDY_REQUIRES(!std::is_base_of_v<stk::mesh::BulkData, Mesh>)
inline void for_each_entity_run(Mesh& mesh, stk::topology::rank_t rank, const stk::mesh::Selector& selector,
                                const AlgorithmPerEntity& functor, const EXEC_SPACE& exec_space) {
  auto ngp_bucket_ids = get_local_bucket_ids(mesh.get_bulk_on_host(), rank, selector, exec_space);
  ngp_bucket_ids.sync_to_device();
  const auto bucket_ids = ngp_bucket_ids.view_device();
  const unsigned num_buckets = static_cast<unsigned>(bucket_ids.extent(0));

  using team_handle_t = typename stk::ngp::TeamPolicy<EXEC_SPACE>::member_type;
  Kokkos::parallel_for(
      stk::ngp::TeamPolicy<EXEC_SPACE>(exec_space, num_buckets, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_handle_t& team) {
        const typename Mesh::BucketType& bucket = mesh.get_bucket(rank, bucket_ids(team.league_rank()));
        const unsigned num_entities = bucket.size();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, num_entities),
                             [&](const unsigned& i) { functor(stk::mesh::FastMeshIndex{bucket.bucket_id(), i}); });
      });
}

template <typename Mesh, typename AlgorithmPerEntity>
MUNDY_REQUIRES(!std::is_base_of_v<stk::mesh::BulkData, Mesh> && !std::is_base_of_v<::mundy::mesh::BulkData, Mesh>)
inline void for_each_entity_run(Mesh& mesh, stk::topology::rank_t rank, const stk::mesh::Selector& selector,
                                const AlgorithmPerEntity& functor) {
  for_each_entity_run(mesh, rank, selector, functor, typename Mesh::MeshExecSpace{});
}

template <typename Mesh, typename AlgorithmPerEntity>
MUNDY_REQUIRES(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
struct TeamFunctor {
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
  using team_member_t = typename team_policy_t::member_type;

  TeamFunctor(const Mesh& m, const stk::mesh::BucketVector& bs, const AlgorithmPerEntity& f)
      : mesh(m), buckets(bs), functor(f) {
  }

  void operator()(const team_member_t& team_member) const {
    stk::mesh::Bucket* bucket = buckets[team_member.league_rank()];
    const int bucket_size = static_cast<int>(bucket->size());
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, bucket_size), [&](const int i) {
      if constexpr (std::is_invocable_v<AlgorithmPerEntity, const Mesh&, const stk::mesh::MeshIndex&>) {
        functor(mesh, stk::mesh::MeshIndex({bucket, i}));
      } else if constexpr (std::is_invocable_v<AlgorithmPerEntity, const Mesh&, const stk::mesh::Entity&>) {
        functor(mesh, (*bucket)[i]);
      } else if constexpr (std::is_invocable_v<AlgorithmPerEntity, const stk::mesh::Entity&>) {
        functor((*bucket)[i]);
      } else {
        static_assert(impl::always_false_v<AlgorithmPerEntity>,
                      "Host for_each_entity_run functors must accept (mesh, mesh_index), (mesh, entity), or (entity).");
      }
    });
  }

  const Mesh& mesh;
  const stk::mesh::BucketVector& buckets;
  const AlgorithmPerEntity& functor;
};

template <typename Mesh, typename AlgorithmPerEntity>
MUNDY_REQUIRES(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
inline void for_each_entity_run(const Mesh& mesh, stk::topology::rank_t rank, const stk::mesh::Selector& selector,
                                const AlgorithmPerEntity& functor) {
  const stk::mesh::BucketVector& buckets = mesh.get_buckets(rank, selector);
  using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
  const unsigned n_buckets = static_cast<unsigned>(buckets.size());
  TeamFunctor<Mesh, AlgorithmPerEntity> team_functor(mesh, buckets, functor);

  Kokkos::parallel_for("for_each_entity_run", team_policy(n_buckets, Kokkos::AUTO), team_functor);
}

template <typename Mesh, typename AlgorithmPerEntity>
MUNDY_REQUIRES(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
inline void for_each_entity_run(const Mesh& mesh, stk::topology::rank_t rank, const AlgorithmPerEntity& functor) {
  stk::mesh::Selector selectAll = mesh.mesh_meta_data().universal_part();
  for_each_entity_run(mesh, rank, selectAll, functor);
}

// template <typename AlgorithmPerEntity>
// inline void for_each_entity_run_no_threads(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
//                                     const stk::mesh::Selector &selector, const AlgorithmPerEntity &functor)
//                                     {
//   stk::mesh::for_each_entity_run_no_threads(mesh, rank, selector, functor);
// }

// template <typename AlgorithmPerEntity>
// inline void for_each_entity_run_no_threads(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
//                                     const AlgorithmPerEntity &functor) {
//   stk::mesh::for_each_entity_run_no_threads(mesh, rank, functor);
// }

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FOREACHENTITY_HPP_
