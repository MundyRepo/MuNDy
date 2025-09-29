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

// Mundy
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData

namespace mundy {

namespace mesh {

template <typename Mesh, typename AlgorithmPerEntity>
  requires(!std::is_base_of_v<stk::mesh::BulkData, Mesh> && !std::is_base_of_v<::mundy::mesh::BulkData, Mesh>)
inline void for_each_entity_run(Mesh &mesh, stk::topology::rank_t rank, const stk::mesh::Selector &selector,
                                const AlgorithmPerEntity &functor) {
  stk::mesh::for_each_entity_run(mesh, rank, selector, functor);
}

template <typename Mesh, typename AlgorithmPerEntity, typename EXEC_SPACE>
  requires(!std::is_base_of_v<stk::mesh::BulkData, Mesh>)
inline void for_each_entity_run(Mesh &mesh, stk::topology::rank_t rank, const stk::mesh::Selector &selector,
                                const AlgorithmPerEntity &functor, const EXEC_SPACE &exec_space) {
  stk::mesh::for_each_entity_run(mesh, rank, selector, functor, exec_space);
}

template <typename Mesh, typename AlgorithmPerEntity>
  requires(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
struct TeamFunctor {
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
  using team_member_t = typename team_policy_t::member_type;

  TeamFunctor(const Mesh &m, const stk::mesh::BucketVector &bs, const AlgorithmPerEntity &f)
      : mesh(m), buckets(bs), functor(f) {
  }

  void operator()(const team_member_t &team_member) const {
    stk::mesh::Bucket *bucket = buckets[team_member.league_rank()];
    const int bucket_size = static_cast<int>(bucket->size());
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, bucket_size), [&](const int i) {
      if constexpr (std::is_invocable_v<AlgorithmPerEntity, const Mesh &, const stk::mesh::MeshIndex &>) {
        functor(mesh, stk::mesh::MeshIndex({bucket, i}));
      } else {
        functor(mesh, (*bucket)[i]);
      }
    });
  }

  const Mesh &mesh;
  const stk::mesh::BucketVector &buckets;
  const AlgorithmPerEntity &functor;
};

template <typename Mesh, typename AlgorithmPerEntity>
  requires(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
inline void for_each_entity_run(const Mesh &mesh, stk::topology::rank_t rank, const stk::mesh::Selector &selector,
                                const AlgorithmPerEntity &functor) {
  const stk::mesh::BucketVector &buckets = mesh.get_buckets(rank, selector);
  using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
  const size_t n_buckets = buckets.size();
  TeamFunctor<Mesh, AlgorithmPerEntity> team_functor(mesh, buckets, functor);

  Kokkos::parallel_for("for_each_entity_run", team_policy(n_buckets, Kokkos::AUTO), team_functor);
}

template <typename Mesh, typename AlgorithmPerEntity>
  requires(std::is_base_of_v<stk::mesh::BulkData, Mesh> || std::is_base_of_v<BulkData, Mesh>)
inline void for_each_entity_run(const Mesh &mesh, stk::topology::rank_t rank, const AlgorithmPerEntity &functor) {
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
