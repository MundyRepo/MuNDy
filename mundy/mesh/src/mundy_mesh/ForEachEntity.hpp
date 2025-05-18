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
#include <stk_mesh/base/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/NgpForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

namespace mundy {

namespace mesh {

template <typename Mesh, typename AlgorithmPerEntity>
  requires(!std::is_base_of_v<stk::mesh::BulkData, Mesh>)
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

template <typename ALGORITHM_TO_RUN_PER_ENTITY>
inline void for_each_entity_run(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
                                const stk::mesh::Selector &selector, const ALGORITHM_TO_RUN_PER_ENTITY &functor) {
  stk::mesh::for_each_entity_run(mesh, rank, selector, functor);
}

template <typename ALGORITHM_TO_RUN_PER_ENTITY>
inline void for_each_entity_run(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
                                const ALGORITHM_TO_RUN_PER_ENTITY &functor) {
  stk::mesh::for_each_entity_run(mesh, rank, functor);
}

// template <typename ALGORITHM_TO_RUN_PER_ENTITY>
// inline void for_each_entity_run_no_threads(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
//                                     const stk::mesh::Selector &selector, const ALGORITHM_TO_RUN_PER_ENTITY &functor)
//                                     {
//   stk::mesh::for_each_entity_run_no_threads(mesh, rank, selector, functor);
// }

// template <typename ALGORITHM_TO_RUN_PER_ENTITY>
// inline void for_each_entity_run_no_threads(const stk::mesh::BulkData &mesh, stk::topology::rank_t rank,
//                                     const ALGORITHM_TO_RUN_PER_ENTITY &functor) {
//   stk::mesh::for_each_entity_run_no_threads(mesh, rank, functor);
// }

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FOREACHENTITY_HPP_
