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

#ifndef MUNDY_SEARCH_NEIGHBORLISTITERATIONTRAITS_HPP_
#define MUNDY_SEARCH_NEIGHBORLISTITERATIONTRAITS_HPP_

/// \file NeighborListIterationTraits.hpp
/// \brief NeighborListIterationTraits primary template and dispatch API.
///
/// The primary template provides the default target-parallel dispatch strategy: a
/// RangePolicy over `[0, num_targets)` with an inner serial walk over each target's
/// neighbor row. Concrete list types that expose a better decomposition can specialize 
/// this struct to override `dispatch_pair` and `dispatch_pair_reduce` without touching 
/// `ForEach.hpp` or any call site.

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/impl/NeighborListFunctors.hpp>  // Deploy* and FlatDeploy* functors

namespace mundy {

namespace search {

/// \struct NeighborListIterationTraits
/// \brief Traits class coupling a concrete neighbor-list type to its parallel dispatch strategy.
///
/// The primary template implements the default target-parallel strategy.
/// Specialize for list types that support a more efficient decomposition.
template <typename ListType>
struct NeighborListIterationTraits {
  //! Size type used by the list.
  using size_type = typename ListType::size_type;

  /// \brief Dispatch a pair callback over all stored pairs.
  ///
  /// \param exec [in] Execution space instance.
  /// \param list [in] Concrete neighbor list.
  /// \param f [in] User callback: f(NeighborPair<ListType>)
  template <typename ExecutionSpace, typename Functor>
  static void dispatch_pair(const ExecutionSpace& exec, const ListType& list, const Functor& f) {
    using range_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    Kokkos::parallel_for("mundy::search::for_each_neighbor_pair", range_t(exec, 0, list.num_targets()),
                         impl::DeployFunctorOnNeighborPairs<ListType, Functor>(list, f));
  }

  /// \brief Dispatch a pair reduction over all stored pairs.
  ///
  /// \param exec [in] Execution space instance.
  /// \param list [in] Concrete neighbor list.
  /// \param f [in] User callback: f(NeighborPair<ListType>, value_type&)
  /// \param r [in,out] Kokkos reducer (e.g. `Kokkos::Sum<double>`) that owns the result and defines join/init.
  template <typename ExecutionSpace, typename Functor, typename ReducerType>
  static void dispatch_pair_reduce(const ExecutionSpace& exec, const ListType& list, const Functor& f, ReducerType& r) {
    using range_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    Kokkos::parallel_reduce("mundy::search::for_each_neighbor_pair_reduce", range_t(exec, 0, list.num_targets()),
                            impl::DeployReduceFunctorOnNeighborPairs<ListType, Functor, ReducerType>(list, f), r);
  }

  /// \brief Dispatch a target-neighbors callback over all targets.
  ///
  /// \param exec [in] Execution space instance.
  /// \param list [in] Concrete neighbor list.
  /// \param f [in] User callback: f(Neighbors<ListType>)
  template <typename ExecutionSpace, typename Functor>
  static void dispatch_target(const ExecutionSpace& exec, const ListType& list, const Functor& f) {
    using range_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    Kokkos::parallel_for("mundy::search::for_each_target_with_neighbors", range_t(exec, 0, list.num_targets()),
                         impl::DeployFunctorOnTargetNeighbors<ListType, Functor>(list, f));
  }

  /// \brief Dispatch a target-neighbors reduction over all targets.
  /// \param exec [in] Execution space instance.
  /// \param list [in] Concrete neighbor list.
  /// \param f [in] User callback: f(Neighbors<ListType>, value_type&)
  /// \param r [in,out] Kokkos reducer (e.g. `Kokkos::Sum<double>`) that owns the result and defines join/init.
  template <typename ExecutionSpace, typename Functor, typename ReducerType>
  static void dispatch_target_reduce(const ExecutionSpace& exec, const ListType& list, const Functor& f,
                                     ReducerType& r) {
    using range_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
    Kokkos::parallel_reduce("mundy::search::for_each_target_with_neighbors_reduce",
                            range_t(exec, 0, list.num_targets()),
                            impl::DeployReduceFunctorOnTargetNeighbors<ListType, Functor, ReducerType>(list, f), r);
  }
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTITERATIONTRAITS_HPP_
