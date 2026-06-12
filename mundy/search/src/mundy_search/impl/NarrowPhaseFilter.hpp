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

#ifndef MUNDY_SEARCH_IMPL_NARROWPHASEFILTER_HPP_
#define MUNDY_SEARCH_IMPL_NARROWPHASEFILTER_HPP_

/// \file impl/NarrowPhaseFilter.hpp
/// \brief Narrow-phase L passes that filter a broad-phase CSR neighbor list down to a final CSR neighbor list.

// C++ core
#include <cstddef>  // for size_t
#include <tuple>    // for std::tuple
#include <utility>  // for std::pair

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>         // for ExcluderType
#include <mundy_search/SearchCandidate.hpp>  // for NeighborSearchCandidate, PeriodicNeighborSearchCandidate

namespace mundy {

namespace search {

namespace impl {

/// \brief Apply a narrow-phase excluder to a broad-phase CSR neighbor list (non-periodic).
///
/// Runs three Kokkos passes over the broad-phase CSR to produce a filtered CSR with the same
/// target-indexed structure:
///
///  - L0 (count): For each target, count broad-phase source candidates that survive the excluder.
///  - L1 (scan):  Exclusive prefix scan of L0 counts to form new CSR offsets.
///  - L2 (fill):  Re-iterate each target's broad row and write surviving source ordinals.
///
/// Iteration is **by target** (outer `parallel_for` over `num_targets`), with a serial inner loop
/// over each target's CSR row `[broad_offsets(t), broad_offsets(t+1))`.  Each thread owns its row
/// exclusively, so no atomics are needed in the fill pass.
///
/// \par Known trade-off: double excluder evaluation
/// `narrow_excluder` is called once per candidate in L0 and once again in L2 — two evaluations
/// per broad-phase pair.  This is structurally required by the count-then-fill pattern: L1 needs
/// the L0 counts before L2 can size its output views.  For cheap excluders the overhead is
/// negligible.  If an expensive excluder (e.g. OBB overlap at O(degree) per call) proves a
/// bottleneck, a flag-caching variant can be introduced: L0 writes a 0/1 validity array, L1 sums
/// it, and L2 scatters survivors using the flags — one evaluation at the cost of O(pairs) extra
/// device memory.  That optimisation is not implemented here.
///
/// \tparam ExecutionSpace Kokkos execution space for all three passes.
/// \tparam NarrowExcluder Excluder type satisfying `ExcluderType`; must be callable with
///                        `NeighborSearchCandidate<size_t>`.
/// \tparam EntityView     Kokkos view of `stk::mesh::Entity*` (rank 1).
/// \tparam IndexView      Kokkos view of integral source-ordinal values (rank 1).
/// \tparam OffsetView     Kokkos view of integral offset values (rank 1).
///
/// \param exec                 [in] Execution space instance.
/// \param narrow_excluder      [in] Excluder applied to every broad-phase candidate.
/// \param target_entities      [in] Dense target entity view of extent `num_targets`.
/// \param source_entities      [in] Dense source entity view of extent `num_sources`.
/// \param broad_source_indices [in] Flat CSR data: source ordinals, extent `total_broad_pairs`.
/// \param broad_offsets        [in] CSR offsets of extent `num_targets + 1`.
///
/// \returns `{narrow_source_indices, narrow_offsets}` — the filtered CSR data and offset views.
template <typename ExecutionSpace, typename NarrowExcluder, typename EntityView, typename IndexView,
          typename OffsetView>
  requires ExcluderType<NarrowExcluder>
std::pair<IndexView, OffsetView> apply_narrow_phase(ExecutionSpace exec, const NarrowExcluder& narrow_excluder,
                                                    const EntityView& target_entities,
                                                    const EntityView& source_entities,
                                                    const IndexView& broad_source_indices,
                                                    const OffsetView& broad_offsets) {
  using size_type = typename OffsetView::value_type;

  const size_type num_targets = static_cast<size_type>(target_entities.extent(0));

  // -------------------------------------------------------------------------
  // Phase L0: count surviving pairs per target.
  // Each thread owns one target row and loops serially — no atomics needed.
  // -------------------------------------------------------------------------
  OffsetView narrow_counts("mundy_narrow_L0_counts", num_targets);

  Kokkos::parallel_for(
      "mundy_narrow_L0_count", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        const size_type beg = broad_offsets(t);
        const size_type end = broad_offsets(t + 1);
        size_type count = 0;
        for (size_type k = beg; k < end; ++k) {
          const size_type s = static_cast<size_type>(broad_source_indices(k));
          if (!narrow_excluder(NeighborSearchCandidate<size_t>(static_cast<size_t>(t), static_cast<size_t>(s),
                                                               target_entities(t), source_entities(s)))) {
            ++count;
          }
        }
        narrow_counts(t) = count;
      });

  // -------------------------------------------------------------------------
  // Phase L1: exclusive prefix scan → narrow CSR offsets.
  // -------------------------------------------------------------------------
  OffsetView narrow_offsets("mundy_narrow_L1_offsets", num_targets + 1);

  Kokkos::parallel_scan(
      "mundy_narrow_L1_scan", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets + 1),
      KOKKOS_LAMBDA(size_type i, size_type & update, bool final_pass) {
        if (final_pass) narrow_offsets(i) = update;
        update += (i < num_targets) ? narrow_counts(i) : size_type(0);
      });

  size_type total_narrow = 0;
  Kokkos::deep_copy(total_narrow, Kokkos::subview(narrow_offsets, num_targets));

  // -------------------------------------------------------------------------
  // Phase L2: fill surviving source ordinals into the output view.
  // Same per-target serial inner loop; thread-local write cursor starts at
  // narrow_offsets(t) and advances for each surviving candidate.
  // -------------------------------------------------------------------------
  IndexView narrow_source_indices("mundy_narrow_L2_source_indices", total_narrow);

  Kokkos::parallel_for(
      "mundy_narrow_L2_fill", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        const size_type beg = broad_offsets(t);
        const size_type end = broad_offsets(t + 1);
        size_type write_pos = narrow_offsets(t);
        for (size_type k = beg; k < end; ++k) {
          const size_type s = static_cast<size_type>(broad_source_indices(k));
          if (!narrow_excluder(NeighborSearchCandidate<size_t>(static_cast<size_t>(t), static_cast<size_t>(s),
                                                               target_entities(t), source_entities(s)))) {
            narrow_source_indices(write_pos) = s;
            ++write_pos;
          }
        }
      });

  return {narrow_source_indices, narrow_offsets};
}

/// \brief Apply a narrow-phase excluder to a broad-phase CSR neighbor list (periodic).
///
/// Periodic overload that carries per-object image shifts through the filter: a per-target-owner target shift and a
/// per-pair source shift. It constructs `PeriodicNeighborSearchCandidate` in L0/L2 from both, and in L2 writes the
/// surviving source owner ordinal and source shift to the output views.
///
/// \tparam ExecutionSpace Kokkos execution space for all three passes.
/// \tparam NarrowExcluder Excluder type satisfying `ExcluderType`; must be callable with
///                        `PeriodicNeighborSearchCandidate<shift_type, size_t>`.
/// \tparam EntityView     Kokkos view of `stk::mesh::Entity*` (rank 1).
/// \tparam IndexView      Kokkos view of integral source-owner-ordinal values (rank 1).
/// \tparam ShiftView      Kokkos view of image-shift values (rank 1).
/// \tparam OffsetView     Kokkos view of integral offset values (rank 1).
///
/// \param exec                       [in] Execution space instance.
/// \param narrow_excluder            [in] Excluder applied to every broad-phase candidate.
/// \param target_entities            [in] Dense target owner entity view of extent `num_targets`.
/// \param source_entities            [in] Dense source owner entity view of extent `num_sources`.
/// \param target_image_shifts        [in] Per-target-owner image shift, extent `num_targets`.
/// \param broad_source_owner_indices [in] Flat CSR data: source owner ordinals, extent `total_broad_pairs`.
/// \param broad_source_image_shifts  [in] Flat CSR data: per-pair source image shifts, extent `total_broad_pairs`.
/// \param broad_offsets              [in] CSR offsets of extent `num_targets + 1`.
///
/// \returns `{narrow_source_owner_indices, narrow_source_image_shifts, narrow_offsets}` — the filtered CSR data
///          views and offset view.
template <typename ExecutionSpace, typename NarrowExcluder, typename EntityView, typename IndexView, typename ShiftView,
          typename OffsetView>
  requires ExcluderType<NarrowExcluder>
std::tuple<IndexView, ShiftView, OffsetView> apply_narrow_phase(
    ExecutionSpace exec, const NarrowExcluder& narrow_excluder, const EntityView& target_entities,
    const EntityView& source_entities, const ShiftView& target_image_shifts,
    const IndexView& broad_source_owner_indices, const ShiftView& broad_source_image_shifts,
    const OffsetView& broad_offsets) {
  using size_type = typename OffsetView::value_type;
  using shift_type = typename ShiftView::value_type;

  const size_type num_targets = static_cast<size_type>(target_entities.extent(0));

  // -------------------------------------------------------------------------
  // Phase L0: count surviving pairs per target.
  // -------------------------------------------------------------------------
  OffsetView narrow_counts("mundy_narrow_L0_counts", num_targets);

  Kokkos::parallel_for(
      "mundy_narrow_L0_count", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        const size_type beg = broad_offsets(t);
        const size_type end = broad_offsets(t + 1);
        size_type count = 0;
        const shift_type target_shift = target_image_shifts(t);
        for (size_type k = beg; k < end; ++k) {
          const size_type s = static_cast<size_type>(broad_source_owner_indices(k));
          const shift_type source_shift = broad_source_image_shifts(k);
          if (!narrow_excluder(PeriodicNeighborSearchCandidate<shift_type, size_t>(
                  static_cast<size_t>(t), static_cast<size_t>(s), target_entities(t), source_entities(s), target_shift,
                  source_shift))) {
            ++count;
          }
        }
        narrow_counts(t) = count;
      });

  // -------------------------------------------------------------------------
  // Phase L1: exclusive prefix scan → narrow CSR offsets.
  // -------------------------------------------------------------------------
  OffsetView narrow_offsets("mundy_narrow_L1_offsets", num_targets + 1);

  Kokkos::parallel_scan(
      "mundy_narrow_L1_scan", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets + 1),
      KOKKOS_LAMBDA(size_type i, size_type & update, bool final_pass) {
        if (final_pass) narrow_offsets(i) = update;
        update += (i < num_targets) ? narrow_counts(i) : size_type(0);
      });

  size_type total_narrow = 0;
  Kokkos::deep_copy(total_narrow, Kokkos::subview(narrow_offsets, num_targets));

  // -------------------------------------------------------------------------
  // Phase L2: fill surviving source owner ordinals and relative image shifts.
  // -------------------------------------------------------------------------
  IndexView narrow_source_owner_indices("mundy_narrow_L2_source_owner_indices", total_narrow);
  ShiftView narrow_source_image_shifts("mundy_narrow_L2_source_image_shifts", total_narrow);

  Kokkos::parallel_for(
      "mundy_narrow_L2_fill", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        const size_type beg = broad_offsets(t);
        const size_type end = broad_offsets(t + 1);
        size_type write_pos = narrow_offsets(t);
        const shift_type target_shift = target_image_shifts(t);
        for (size_type k = beg; k < end; ++k) {
          const size_type s = static_cast<size_type>(broad_source_owner_indices(k));
          const shift_type source_shift = broad_source_image_shifts(k);
          if (!narrow_excluder(PeriodicNeighborSearchCandidate<shift_type, size_t>(
                  static_cast<size_t>(t), static_cast<size_t>(s), target_entities(t), source_entities(s), target_shift,
                  source_shift))) {
            narrow_source_owner_indices(write_pos) = s;
            narrow_source_image_shifts(write_pos) = source_shift;
            ++write_pos;
          }
        }
      });

  return {narrow_source_owner_indices, narrow_source_image_shifts, narrow_offsets};
}

/// \brief Apply a narrow-phase excluder to a broad-phase dense-2D neighbor list (non-periodic).
///
/// The broad-phase 2D list is represented as a dense `source_indices(t, k)` view with a separate
/// `neighbor_counts(t)` view giving the valid column count per target.  Three passes:
///
///  - L0 (count): For each target row, count candidates that survive the excluder.
///  - L1 (max):   Find the new maximum column width to size the output 2D view.
///  - L2 (fill):  Re-iterate each row and write surviving source ordinals, left-compacted.
///
/// \tparam ExecutionSpace  Kokkos execution space.
/// \tparam NarrowExcluder  Excluder satisfying `ExcluderType`.
/// \tparam EntityView      Kokkos view of `stk::mesh::Entity*` (rank 1).
/// \tparam CountView       Kokkos view of per-target count values (rank 1).
/// \tparam IndexView2D     Kokkos view of source-ordinal values (rank 2, `(target, col)`).
///
/// \returns `{narrow_counts, narrow_src}` — the filtered per-target counts and dense-2D source index view.
template <typename ExecutionSpace, typename NarrowExcluder, typename EntityView, typename CountView,
          typename IndexView2D>
  requires ExcluderType<NarrowExcluder>
std::pair<CountView, IndexView2D> apply_narrow_phase_2d(ExecutionSpace exec, const NarrowExcluder& narrow_excluder,
                                                        const EntityView& target_entities,
                                                        const EntityView& source_entities,
                                                        const IndexView2D& broad_source_indices,
                                                        const CountView& neighbor_counts) {
  using size_type = typename CountView::value_type;
  using memory_space = typename CountView::memory_space;

  const size_type num_targets = static_cast<size_type>(target_entities.extent(0));

  // L0: count narrow survivors per target row.
  CountView narrow_counts("mundy_2d_narrow_counts", num_targets);
  Kokkos::parallel_for(
      "mundy_2d_narrow_L0", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        size_type count = 0;
        for (size_type k = 0; k < neighbor_counts(t); ++k) {
          const size_type s = broad_source_indices(t, k);
          if (!narrow_excluder(NeighborSearchCandidate<size_t>(static_cast<size_t>(t), static_cast<size_t>(s),
                                                               target_entities(t), source_entities(s)))) {
            ++count;
          }
        }
        narrow_counts(t) = count;
      });

  // L1: new max column width.
  size_type new_max = 0;
  Kokkos::parallel_reduce(
      "mundy_2d_narrow_L1_max", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets),
      KOKKOS_LAMBDA(size_type t, size_type & lmax) { lmax = lmax > narrow_counts(t) ? lmax : narrow_counts(t); },
      Kokkos::Max<size_type>(new_max));
  Kokkos::fence();

  // L2: fill compacted 2D grid.
  IndexView2D narrow_src("mundy_2d_narrow_src", num_targets, new_max);
  Kokkos::parallel_for(
      "mundy_2d_narrow_L2", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, num_targets), KOKKOS_LAMBDA(size_type t) {
        size_type write_col = 0;
        for (size_type k = 0; k < neighbor_counts(t); ++k) {
          const size_type s = broad_source_indices(t, k);
          if (!narrow_excluder(NeighborSearchCandidate<size_t>(static_cast<size_t>(t), static_cast<size_t>(s),
                                                               target_entities(t), source_entities(s)))) {
            narrow_src(t, write_col++) = s;
          }
        }
      });

  return {narrow_counts, narrow_src};
}

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_NARROWPHASEFILTER_HPP_
