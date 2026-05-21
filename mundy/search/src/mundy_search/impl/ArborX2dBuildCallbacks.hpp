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

#ifndef MUNDY_SEARCH_IMPL_ARBORX2DBUILDCALLBACKS_HPP_
#define MUNDY_SEARCH_IMPL_ARBORX2DBUILDCALLBACKS_HPP_

/// \file impl/ArborX2dBuildCallbacks.hpp
/// \brief No-output ArborX callbacks for non-periodic 2D neighbor-list construction.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>
#include <mundy_search/impl/ArborXCallback.hpp>  // for ArborXSearchCandidateFactory

namespace mundy {

namespace search {

namespace impl {

/// \class ArborX2dCountCallback
/// \brief Count non-excluded neighbors per target (no-output, side-effect only).
///
/// Used as the first pass in the two-pass non-periodic 2D build. Atomically increments a per-target count for every
/// ArborX hit that survives the excluder. No managed CSR output is produced.
/// \tparam TargetBoxes Search-box wrapper for target boxes.
/// \tparam SourceBoxes Search-box wrapper for source boxes.
/// \tparam Excluder Excluder type applied to each candidate.
template <typename TargetBoxes, typename SourceBoxes, ExcluderType Excluder>
class ArborX2dCountCallback {
 public:
  //! \name Aliases
  //@{

  using memory_space = typename TargetBoxes::memory_space;
  using size_type = typename TargetBoxes::size_type;
  using factory_type = ArborXSearchCandidateFactory<TargetBoxes, SourceBoxes>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION ArborX2dCountCallback() = default;

  KOKKOS_INLINE_FUNCTION
  ArborX2dCountCallback(const factory_type& factory, const Excluder& excluder, const count_view_t& counts)
      : factory_(factory), excluder_(excluder), counts_(counts) {
  }
  //@}

  //! \name ArborX callback interface (no OutputFunctor — pure side-effect)
  //@{

#if ARBORX_VERSION >= 10799
  template <typename Predicate, typename Geometry>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred,
                                         const ArborX::PairValueIndex<Geometry, int>& val) const {
    const size_type source_idx = static_cast<size_type>(val.index);
    const auto candidate = factory_(pred, source_idx);
    if (!excluder_(candidate)) {
      Kokkos::atomic_inc(&counts_(static_cast<size_type>(ArborX::getData(pred))));
    }
  }
#else
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred, int source_idx_raw) const {
    const size_type source_idx = static_cast<size_type>(source_idx_raw);
    const auto candidate = factory_(pred, source_idx);
    if (!excluder_(candidate)) {
      Kokkos::atomic_inc(&counts_(static_cast<size_type>(ArborX::getData(pred))));
    }
  }
#endif
  //@}

 private:
  factory_type factory_;
  Excluder excluder_;
  count_view_t counts_;
};

/// \class ArborX2dFillCallback
/// \brief Fill source-index rows for non-periodic 2D builds (no-output, side-effect only).
///
/// Used as the second pass after counts are known and the 2D view is allocated. Uses `atomic_fetch_add` on a per-target
/// write-position cursor to claim a row slot for each surviving hit.
/// \tparam TargetBoxes Search-box wrapper for target boxes.
/// \tparam SourceBoxes Search-box wrapper for source boxes.
/// \tparam Excluder Excluder type applied to each candidate.
template <typename TargetBoxes, typename SourceBoxes, ExcluderType Excluder>
class ArborX2dFillCallback {
 public:
  //! \name Aliases
  //@{

  using memory_space = typename TargetBoxes::memory_space;
  using size_type = typename TargetBoxes::size_type;
  using factory_type = ArborXSearchCandidateFactory<TargetBoxes, SourceBoxes>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<size_type**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION ArborX2dFillCallback() = default;

  KOKKOS_INLINE_FUNCTION
  ArborX2dFillCallback(const factory_type& factory, const Excluder& excluder, const count_view_t& write_positions,
                       const source_index_view_t& source_indices)
      : factory_(factory), excluder_(excluder), write_positions_(write_positions), source_indices_(source_indices) {
  }
  //@}

  //! \name ArborX callback interface (no OutputFunctor — pure side-effect)
  //@{

#if ARBORX_VERSION >= 10799
  template <typename Predicate, typename Geometry>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred,
                                         const ArborX::PairValueIndex<Geometry, int>& val) const {
    const size_type source_idx = static_cast<size_type>(val.index);
    const auto candidate = factory_(pred, source_idx);
    if (!excluder_(candidate)) {
      const size_type target_idx = static_cast<size_type>(ArborX::getData(pred));
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_idx), size_type(1));
      source_indices_(target_idx, pos) = source_idx;
    }
  }
#else
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred, int source_idx_raw) const {
    const size_type source_idx = static_cast<size_type>(source_idx_raw);
    const auto candidate = factory_(pred, source_idx);
    if (!excluder_(candidate)) {
      const size_type target_idx = static_cast<size_type>(ArborX::getData(pred));
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_idx), size_type(1));
      source_indices_(target_idx, pos) = source_idx;
    }
  }
#endif
  //@}

 private:
  factory_type factory_;
  Excluder excluder_;
  count_view_t write_positions_;
  source_index_view_t source_indices_;
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // MUNDY_SEARCH_IMPL_ARBORX2DBUILDCALLBACKS_HPP_
