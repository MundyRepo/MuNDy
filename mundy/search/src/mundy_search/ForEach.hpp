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

#ifndef MUNDY_SEARCH_FOREACH_HPP_
#define MUNDY_SEARCH_FOREACH_HPP_

/// \file ForEach.hpp
/// \brief Parallel iteration and reduction entry points over neighbor pairs and per-target neighbor ranges.
///
/// \section Reduction functions
/// `for_each_neighbor_pair_reduce` and `for_each_target_with_neighbors_reduce` mirror STK's
/// `for_each_entity_reduce` pattern: the caller passes a Kokkos built-in reducer
/// (e.g. `Kokkos::Sum<double>`, `Kokkos::Min<int>`) together with a functor whose second argument
/// is `value_type&`.  Example usage:
/// \code{.cpp}
///   size_t pair_count = 0;
///   Kokkos::Sum<size_t> reducer(pair_count);
///   mundy::search::for_each_neighbor_pair_reduce(
///       exec, list,
///       KOKKOS_LAMBDA(const mundy::search::NeighborPair<List1d>& /*p*/, size_t& n) { ++n; },
///       reducer);
/// \endcode

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Neighbors.hpp>                  // for NeighborListType concept
#include <mundy_search/impl/NeighborListFunctors.hpp>  // for DeployFunctorOnNeighborPairs, DeployFunctorOnTargetNeighbors

namespace mundy {

namespace search {

/// \brief Run a callback for every stored neighbor pair using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable with NeighborPair<ListType>.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
template <NeighborListType ListType, typename Functor>
void for_each_neighbor_pair(const ListType& list, const Functor& functor) {
  typename ListType::execution_space exec_space{};
  for_each_neighbor_pair(exec_space, list, functor);
}

/// \brief Run a callback for every stored neighbor pair using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable with NeighborPair<ListType>.
/// \param exec_space [in] Execution space for the outer target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor>
void for_each_neighbor_pair(const ExecutionSpace& exec_space, const ListType& list, const Functor& functor) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployFunctorOnNeighborPairs<ListType, Functor> deploy_functor(list, functor);
  Kokkos::parallel_for("mundy::search::for_each_neighbor_pair", range_policy_t(exec_space, 0, list.num_targets()),
                       deploy_functor);
}

/// \brief Run a callback for every target and its neighbors using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable with Neighbors<ListType>.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
template <NeighborListType ListType, typename Functor>
void for_each_target_with_neighbors(const ListType& list, const Functor& functor) {
  typename ListType::execution_space exec_space{};
  for_each_target_with_neighbors(exec_space, list, functor);
}

/// \brief Run a callback for every target and its neighbors using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable with Neighbors<ListType>.
/// \param exec_space [in] Execution space for the target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor>
void for_each_target_with_neighbors(const ExecutionSpace& exec_space, const ListType& list, const Functor& functor) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployFunctorOnTargetNeighbors<ListType, Functor> deploy_functor(list, functor);
  Kokkos::parallel_for("mundy::search::for_each_target_with_neighbors",
                       range_policy_t(exec_space, 0, list.num_targets()), deploy_functor);
}

/// \brief Run a Kokkos reduction over every stored neighbor pair using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable as `functor(NeighborPair<ListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
/// \param reducer [in,out] Kokkos reducer that owns the result and defines join/init.
template <NeighborListType ListType, typename Functor, typename ReducerType>
void for_each_neighbor_pair_reduce(const ListType& list, const Functor& functor, ReducerType& reducer) {
  typename ListType::execution_space exec_space{};
  for_each_neighbor_pair_reduce(exec_space, list, functor, reducer);
}

/// \brief Run a Kokkos reduction over every stored neighbor pair using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable as `functor(NeighborPair<ListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
/// \param exec_space [in] Execution space for the outer target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
/// \param reducer [in,out] Kokkos reducer that owns the result and defines join/init.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor, typename ReducerType>
void for_each_neighbor_pair_reduce(const ExecutionSpace& exec_space, const ListType& list,
                                   const Functor& functor, ReducerType& reducer) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployReduceFunctorOnNeighborPairs<ListType, Functor, ReducerType> deploy_functor(list, functor);
  Kokkos::parallel_reduce("mundy::search::for_each_neighbor_pair_reduce",
                          range_policy_t(exec_space, 0, list.num_targets()),
                          deploy_functor, reducer);
}

/// \brief Run a Kokkos reduction over every target and its neighbors using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable as `functor(Neighbors<ListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
/// \param reducer [in,out] Kokkos reducer that owns the result and defines join/init.
template <NeighborListType ListType, typename Functor, typename ReducerType>
void for_each_target_with_neighbors_reduce(const ListType& list, const Functor& functor, ReducerType& reducer) {
  typename ListType::execution_space exec_space{};
  for_each_target_with_neighbors_reduce(exec_space, list, functor, reducer);
}

/// \brief Run a Kokkos reduction over every target and its neighbors using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable as `functor(Neighbors<ListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
/// \param exec_space [in] Execution space for the target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
/// \param reducer [in,out] Kokkos reducer that owns the result and defines join/init.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor, typename ReducerType>
void for_each_target_with_neighbors_reduce(const ExecutionSpace& exec_space, const ListType& list,
                                           const Functor& functor, ReducerType& reducer) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployReduceFunctorOnTargetNeighbors<ListType, Functor, ReducerType> deploy_functor(list, functor);
  Kokkos::parallel_reduce("mundy::search::for_each_target_with_neighbors_reduce",
                          range_policy_t(exec_space, 0, list.num_targets()),
                          deploy_functor, reducer);
}

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_FOREACH_HPP_
