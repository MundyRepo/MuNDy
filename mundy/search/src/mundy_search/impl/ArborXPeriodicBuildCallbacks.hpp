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

#ifndef MUNDY_SEARCH_IMPL_ARBORXPERIODICBUILDCALLBACKS_HPP_
#define MUNDY_SEARCH_IMPL_ARBORXPERIODICBUILDCALLBACKS_HPP_

/// \file impl/ArborXPeriodicBuildCallbacks.hpp
/// \brief No-output ArborX callbacks for periodic neighbor-list construction (shared between 1D and 2D builds).
///
/// These callbacks receive ArborX image ordinals, map them to owner ordinals and relative shifts via the periodic
/// candidate factory, apply the excluder, and atomically write to owner-indexed output views.

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>
#include <mundy_search/impl/ArborXCallback.hpp>  // for PeriodicArborXSearchCandidateFactory

namespace mundy {

namespace search {

namespace impl {

/// \class ArborXPeriodicCountCallback
/// \brief Count non-excluded periodic pairs per target owner (no-output, side-effect only).
///
/// Shared between the periodic 1D and 2D builds. On each surviving hit, atomically increments the count for the
/// target owner ordinal (not the image ordinal) so the prefix scan and output allocation use owner-indexed storage.
/// \tparam TargetBoxes Periodic search-box wrapper for target image boxes.
/// \tparam SourceBoxes Periodic search-box wrapper for source image boxes.
/// \tparam Excluder Excluder type applied to each candidate.
template <typename TargetBoxes, typename SourceBoxes, ExcluderType Excluder>
class ArborXPeriodicCountCallback {
 public:
  //! \name Aliases
  //@{

  using memory_space = typename TargetBoxes::memory_space;
  using size_type = typename TargetBoxes::size_type;
  using factory_type = PeriodicArborXSearchCandidateFactory<TargetBoxes, SourceBoxes>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION ArborXPeriodicCountCallback() = default;

  KOKKOS_INLINE_FUNCTION
  ArborXPeriodicCountCallback(const factory_type& factory, const Excluder& excluder,
                               const count_view_t& owner_counts)
      : factory_(factory), excluder_(excluder), owner_counts_(owner_counts) {
  }
  //@}

  //! \name ArborX callback interface (no OutputFunctor — pure side-effect)
  //@{

#if ARBORX_VERSION >= 10799
  template <typename Predicate, typename Geometry>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred,
                                         const ArborX::PairValueIndex<Geometry, int>& val) const {
    const size_type source_image_idx = static_cast<size_type>(val.index);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      Kokkos::atomic_increment(&owner_counts_(candidate.target_index()));
    }
  }
#else
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred, int source_image_raw) const {
    const size_type source_image_idx = static_cast<size_type>(source_image_raw);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      Kokkos::atomic_increment(&owner_counts_(candidate.target_index()));
    }
  }
#endif
  //@}

 private:
  factory_type factory_;
  Excluder excluder_;
  count_view_t owner_counts_;
};

/// \class ArborXPeriodic1dFillCallback
/// \brief Fill flat 1D owner-indexed pair arrays for periodic builds (no-output, side-effect only).
///
/// Writes source owner ordinals and relative image shifts into flat CSR storage indexed by the prefix-scan `offsets`
/// and per-target `write_positions` cursors. `atomic_fetch_add` on `write_positions` serializes writes within each
/// target owner's row.
/// \tparam TargetBoxes Periodic search-box wrapper for target image boxes.
/// \tparam SourceBoxes Periodic search-box wrapper for source image boxes.
/// \tparam Excluder Excluder type applied to each candidate.
/// \tparam ImageShiftType Type used for image-shift vectors (e.g., mundy::Vector3<float>).
template <typename TargetBoxes, typename SourceBoxes, ExcluderType Excluder, typename ImageShiftType>
class ArborXPeriodic1dFillCallback {
 public:
  //! \name Aliases
  //@{

  using memory_space = typename TargetBoxes::memory_space;
  using size_type = typename TargetBoxes::size_type;
  using factory_type = PeriodicArborXSearchCandidateFactory<TargetBoxes, SourceBoxes>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<ImageShiftType*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION ArborXPeriodic1dFillCallback() = default;

  KOKKOS_INLINE_FUNCTION
  ArborXPeriodic1dFillCallback(const factory_type& factory, const Excluder& excluder,
                                const count_view_t& write_positions, const count_view_t& offsets,
                                const source_index_view_t& source_owner_indices,
                                const image_shift_view_t& relative_image_shifts)
      : factory_(factory),
        excluder_(excluder),
        write_positions_(write_positions),
        offsets_(offsets),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts) {
  }
  //@}

  //! \name ArborX callback interface (no OutputFunctor — pure side-effect)
  //@{

#if ARBORX_VERSION >= 10799
  template <typename Predicate, typename Geometry>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred,
                                         const ArborX::PairValueIndex<Geometry, int>& val) const {
    const size_type source_image_idx = static_cast<size_type>(val.index);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      const size_type target_owner_idx = candidate.target_index();
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_owner_idx), size_type(1));
      const size_type flat_idx = offsets_(target_owner_idx) + pos;
      source_owner_indices_(flat_idx) = candidate.source_index();
      relative_image_shifts_(flat_idx) = candidate.relative_image_shift();
    }
  }
#else
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred, int source_image_raw) const {
    const size_type source_image_idx = static_cast<size_type>(source_image_raw);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      const size_type target_owner_idx = candidate.target_index();
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_owner_idx), size_type(1));
      const size_type flat_idx = offsets_(target_owner_idx) + pos;
      source_owner_indices_(flat_idx) = candidate.source_index();
      relative_image_shifts_(flat_idx) = candidate.relative_image_shift();
    }
  }
#endif
  //@}

 private:
  factory_type factory_;
  Excluder excluder_;
  count_view_t write_positions_;
  count_view_t offsets_;
  source_index_view_t source_owner_indices_;
  image_shift_view_t relative_image_shifts_;
};

/// \class ArborXPeriodic2dFillCallback
/// \brief Fill 2D owner-indexed pair arrays for periodic builds (no-output, side-effect only).
///
/// Writes source owner ordinals and relative image shifts into dense 2D storage. Uses `atomic_fetch_add` on a
/// per-target write-position cursor to assign a column slot within each target owner's dense row.
/// \tparam TargetBoxes Periodic search-box wrapper for target image boxes.
/// \tparam SourceBoxes Periodic search-box wrapper for source image boxes.
/// \tparam Excluder Excluder type applied to each candidate.
/// \tparam ImageShiftType Type used for image-shift vectors (e.g., mundy::Vector3<float>).
template <typename TargetBoxes, typename SourceBoxes, ExcluderType Excluder, typename ImageShiftType>
class ArborXPeriodic2dFillCallback {
 public:
  //! \name Aliases
  //@{

  using memory_space = typename TargetBoxes::memory_space;
  using size_type = typename TargetBoxes::size_type;
  using factory_type = PeriodicArborXSearchCandidateFactory<TargetBoxes, SourceBoxes>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<size_type**, memory_space>;
  using image_shift_view_t = Kokkos::View<ImageShiftType**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION ArborXPeriodic2dFillCallback() = default;

  KOKKOS_INLINE_FUNCTION
  ArborXPeriodic2dFillCallback(const factory_type& factory, const Excluder& excluder,
                                const count_view_t& write_positions,
                                const source_index_view_t& source_owner_indices,
                                const image_shift_view_t& relative_image_shifts)
      : factory_(factory),
        excluder_(excluder),
        write_positions_(write_positions),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts) {
  }
  //@}

  //! \name ArborX callback interface (no OutputFunctor — pure side-effect)
  //@{

#if ARBORX_VERSION >= 10799
  template <typename Predicate, typename Geometry>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred,
                                         const ArborX::PairValueIndex<Geometry, int>& val) const {
    const size_type source_image_idx = static_cast<size_type>(val.index);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      const size_type target_owner_idx = candidate.target_index();
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_owner_idx), size_type(1));
      source_owner_indices_(target_owner_idx, pos) = candidate.source_index();
      relative_image_shifts_(target_owner_idx, pos) = candidate.relative_image_shift();
    }
  }
#else
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& pred, int source_image_raw) const {
    const size_type source_image_idx = static_cast<size_type>(source_image_raw);
    const auto candidate = factory_(pred, source_image_idx);
    if (!excluder_(candidate)) {
      const size_type target_owner_idx = candidate.target_index();
      const size_type pos = Kokkos::atomic_fetch_add(&write_positions_(target_owner_idx), size_type(1));
      source_owner_indices_(target_owner_idx, pos) = candidate.source_index();
      relative_image_shifts_(target_owner_idx, pos) = candidate.relative_image_shift();
    }
  }
#endif
  //@}

 private:
  factory_type factory_;
  Excluder excluder_;
  count_view_t write_positions_;
  source_index_view_t source_owner_indices_;
  image_shift_view_t relative_image_shifts_;
};

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_ARBORXPERIODICBUILDCALLBACKS_HPP_
