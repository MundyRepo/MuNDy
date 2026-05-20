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

#ifndef MUNDY_SEARCH_IMPL_NEIGHBORLISTFUNCTORS_HPP_
#define MUNDY_SEARCH_IMPL_NEIGHBORLISTFUNCTORS_HPP_

/// \file impl/NeighborListFunctors.hpp
/// \brief Kokkos functors used internally by for_each_neighbor_pair and for_each_target_with_neighbors.

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Neighbors.hpp>  // for Neighbors, NeighborPair

namespace mundy {

namespace search {

namespace impl {

/// \class DeployFunctorOnNeighborPairs
/// \brief Kokkos functor that expands target-parallel work into neighbor-pair callbacks.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `NeighborPair<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every neighbor pair.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor) : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for every neighbor of one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    const size_type num_neighbors = list_.num_neighbors(target_index);
    for (size_type neighbor_ordinal = 0; neighbor_ordinal < num_neighbors; ++neighbor_ordinal) {
      functor_(mundy::search::NeighborPair<NeighborListType>(list_, target_index, neighbor_ordinal));
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each stored neighbor pair.
  Functor functor_;
  //@}
};

/// \class DeployFunctorOnTargetNeighbors
/// \brief Kokkos functor that invokes a target-neighbors callback for each target ordinal.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `Neighbors<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnTargetNeighbors {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every target.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnTargetNeighbors(const NeighborListType& list, const Functor& functor)
      : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    functor_(mundy::search::Neighbors<NeighborListType>(list_, target_index));
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each target-neighbors payload.
  Functor functor_;
  //@}
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_NEIGHBORLISTFUNCTORS_HPP_
