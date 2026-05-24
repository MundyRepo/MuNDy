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
  DeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor) : list_(&list), functor_(functor) {
  }

  /// \brief Run the user callback for every neighbor of one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    const size_type num_neighbors = list_->num_neighbors(target_index);
    for (size_type neighbor_ordinal = 0; neighbor_ordinal < num_neighbors; ++neighbor_ordinal) {
      functor_(mundy::search::NeighborPair<NeighborListType>(*list_, target_index, neighbor_ordinal));
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
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
      : list_(&list), functor_(functor) {
  }

  /// \brief Run the user callback for one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    functor_(mundy::search::Neighbors<NeighborListType>(*list_, target_index));
  }

 private:
  //! \name Internal members
  //@{

  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
  //! User callback invoked once for each target-neighbors payload.
  Functor functor_;
  //@}
};

/// \class DeployReduceFunctorOnNeighborPairs
/// \brief Kokkos reduction functor that expands target-parallel work into per-pair reduction callbacks.
///
/// The outer `parallel_reduce` iterates over targets; for each target this functor walks the neighbor
/// row serially and calls the user callback with a `NeighborPair` payload and the thread-local
/// reduction accumulator.  The `value_type` is taken from `ReducerType::value_type` so it matches
/// whatever Kokkos built-in reducer (Sum, Min, Max, …) the caller passes.
///
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable as `functor(NeighborPair<NeighborListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
template <typename NeighborListType, typename Functor, typename ReducerType>
class DeployReduceFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;
  //! Reduction accumulator type required by Kokkos parallel_reduce.
  using value_type = typename ReducerType::value_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every neighbor pair.
  KOKKOS_INLINE_FUNCTION
  DeployReduceFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor)
      : list_(&list), functor_(functor) {}

  /// \brief Accumulate over every neighbor of one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  /// \param update [in,out] Thread-local reduction accumulator.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index, value_type& update) const {
    const size_type num_neighbors = list_->num_neighbors(target_index);
    for (size_type k = 0; k < num_neighbors; ++k) {
      functor_(mundy::search::NeighborPair<NeighborListType>(*list_, target_index, k), update);
    }
  }

 private:
  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
  //! User callback invoked once for each stored neighbor pair.
  Functor functor_;
};

/// \class DeployReduceFunctorOnTargetNeighbors
/// \brief Kokkos reduction functor that invokes a target-neighbors reduction callback for each target ordinal.
///
/// The outer `parallel_reduce` iterates over targets; this functor constructs a `Neighbors` facade for
/// each target and calls the user callback with it and the thread-local accumulator.
///
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable as `functor(Neighbors<NeighborListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
template <typename NeighborListType, typename Functor, typename ReducerType>
class DeployReduceFunctorOnTargetNeighbors {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;
  //! Reduction accumulator type required by Kokkos parallel_reduce.
  using value_type = typename ReducerType::value_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every target.
  KOKKOS_INLINE_FUNCTION
  DeployReduceFunctorOnTargetNeighbors(const NeighborListType& list, const Functor& functor)
      : list_(&list), functor_(functor) {}

  /// \brief Accumulate over one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  /// \param update [in,out] Thread-local reduction accumulator.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index, value_type& update) const {
    functor_(mundy::search::Neighbors<NeighborListType>(*list_, target_index), update);
  }

 private:
  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
  //! User callback invoked once for each target-neighbors payload.
  Functor functor_;
};

/// \class FlatDeployFunctorOnNeighborPairs
/// \brief Kokkos functor for use with MDRangePolicy<Rank<2>> over a dense 2D neighbor layout.
///
/// The outer `parallel_for` iterates over a rectangular `(num_targets, max_neighbors_per_target)`
/// grid.  Each cell `(t, k)` is a potential pair slot; the functor guards on
/// `k < list->num_neighbors(t)` and skips padding slots silently.  This exposes
/// `num_targets * max_neighbors_per_target` parallel work items, removing the serial inner
/// loop and load-imbalance of the target-parallel strategy when neighbor counts vary.
///
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `NeighborPair<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class FlatDeployFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the flat deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every valid neighbor pair.
  KOKKOS_INLINE_FUNCTION
  FlatDeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor)
      : list_(&list), functor_(functor) {}

  /// \brief Invoke the user callback for cell `(target_index, neighbor_ordinal)` if valid.
  /// \param target_index [in] Dense target ordinal (outer MDRange dimension).
  /// \param neighbor_ordinal [in] Neighbor slot index (inner MDRange dimension).
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index, const size_type neighbor_ordinal) const {
    if (neighbor_ordinal < list_->num_neighbors(target_index)) {
      functor_(mundy::search::NeighborPair<NeighborListType>(*list_, target_index, neighbor_ordinal));
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
  //! User callback invoked once for each valid pair cell.
  Functor functor_;
  //@}
};

/// \class FlatDeployReduceFunctorOnNeighborPairs
/// \brief Kokkos reduction functor for use with MDRangePolicy<Rank<2>> over a dense 2D neighbor layout.
///
/// Same cell-guard logic as `FlatDeployFunctorOnNeighborPairs`, but with an accumulator argument
/// for use inside `parallel_reduce`.
///
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable as `functor(NeighborPair<NeighborListType>, value_type&)`.
/// \tparam ReducerType Kokkos reducer type (e.g. `Kokkos::Sum<double>`); supplies `value_type`.
template <typename NeighborListType, typename Functor, typename ReducerType>
class FlatDeployReduceFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;
  //! Reduction accumulator type required by Kokkos parallel_reduce.
  using value_type = typename ReducerType::value_type;

  /// \brief Construct the flat reduction functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every valid neighbor pair.
  KOKKOS_INLINE_FUNCTION
  FlatDeployReduceFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor)
      : list_(&list), functor_(functor) {}

  /// \brief Accumulate over cell `(target_index, neighbor_ordinal)` if valid.
  /// \param target_index [in] Dense target ordinal (outer MDRange dimension).
  /// \param neighbor_ordinal [in] Neighbor slot index (inner MDRange dimension).
  /// \param update [in,out] Thread-local reduction accumulator.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index, const size_type neighbor_ordinal, value_type& update) const {
    if (neighbor_ordinal < list_->num_neighbors(target_index)) {
      functor_(mundy::search::NeighborPair<NeighborListType>(*list_, target_index, neighbor_ordinal), update);
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Pointer to the neighbor list (never null after construction; outlives this functor).
  const NeighborListType* list_;
  //! User callback invoked once for each valid pair cell.
  Functor functor_;
  //@}
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_NEIGHBORLISTFUNCTORS_HPP_
