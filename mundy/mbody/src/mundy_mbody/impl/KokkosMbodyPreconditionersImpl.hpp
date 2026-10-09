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

#ifndef MUNDY_MBODY_IMPL_KOKKOSMBODYPRECONDITIONERSIMPL_HPP_
#define MUNDY_MBODY_IMPL_KOKKOSMBODYPRECONDITIONERSIMPL_HPP_

/// \file
/// \brief The sparse matrix dt B^T M_self B + K^{-1} of a linearization's bilateral block, assembled on the device.

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS

// C++ core
#include <cstddef>  // for size_t

// Kokkos
#include <KokkosSparse_SortCrs.hpp>  // for KokkosSparse::sort_and_merge_graph
#include <Kokkos_Core.hpp>

namespace mundy {

namespace mbody {

namespace impl {

/// \brief Each row's rods, in the row's order: row i's are rods(offsets(i)), ..., rods(offsets(i + 1) - 1).
template <typename ExecSpace>
struct RowRods {
  Kokkos::View<size_t*, typename ExecSpace::memory_space> offsets;
  Kokkos::View<int*, typename ExecSpace::memory_space> rods;
};

/// \brief The rods of each of jacobian's num_rows rows.
template <typename ExecSpace, typename Jacobian>
RowRods<ExecSpace> make_row_rods(const Jacobian& jacobian, size_t num_rows) {
  using memory_space = typename ExecSpace::memory_space;
  const Kokkos::View<size_t*, memory_space> offsets("row_rod_offsets", num_rows + 1);
  size_t num_incidences = 0;
  Kokkos::parallel_scan(
      "make_row_rods::offsets", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const size_t row, size_t& partial, const bool final) {
        partial += static_cast<size_t>(jacobian.num_bodies(row));
        if (final) {
          offsets(row + 1) = partial;
        }
      },
      num_incidences);
  const Kokkos::View<int*, memory_space> rods(Kokkos::view_alloc(Kokkos::WithoutInitializing, "row_rods"),
                                              num_incidences);
  Kokkos::parallel_for(
      "make_row_rods::rods", Kokkos::RangePolicy<ExecSpace>(0, num_rows), KOKKOS_LAMBDA(const size_t row) {
        for (int k = 0; k < jacobian.num_bodies(row); ++k) {
          rods(offsets(row) + k) = jacobian.body(row, k);
        }
      });
  return RowRods<ExecSpace>{offsets, rods};
}

/// \brief Whether a and b give every row the same rods in the same order.
template <typename ExecSpace>
bool same_row_rods(const RowRods<ExecSpace>& a, const RowRods<ExecSpace>& b) {
  if (a.offsets.extent(0) != b.offsets.extent(0) || a.rods.extent(0) != b.rods.extent(0)) {
    return false;
  }
  int num_differing = 0;
  Kokkos::parallel_reduce(
      "same_row_rods", Kokkos::RangePolicy<ExecSpace>(0, a.offsets.extent(0) - 1),
      KOKKOS_LAMBDA(const size_t row, int& differing) {
        // offsets(0) is 0 in both, so equal ends give equal ranges.
        if (a.offsets(row + 1) != b.offsets(row + 1)) {
          ++differing;
          return;
        }
        for (size_t e = a.offsets(row); e < a.offsets(row + 1); ++e) {
          differing += (a.rods(e) != b.rods(e)) ? 1 : 0;
        }
      },
      num_differing);
  return num_differing == 0;
}

/// \brief A square SparseMatrix with row_rods' rows, in which row i couples row j when they share a rod; its values are
/// zero.
template <typename SparseMatrix, typename ExecSpace>
SparseMatrix make_shared_rod_pattern(const RowRods<ExecSpace>& row_rods) {
  using memory_space = typename ExecSpace::memory_space;
  using row_map_t = typename SparseMatrix::row_map_type::non_const_type;
  using entries_t = typename SparseMatrix::index_type::non_const_type;
  using values_t = typename SparseMatrix::values_type::non_const_type;
  using ordinal_t = typename SparseMatrix::non_const_ordinal_type;
  using offset_t = typename SparseMatrix::non_const_size_type;
  const size_t num_rows = row_rods.offsets.extent(0) - 1;
  const auto row_offsets = row_rods.offsets;
  const auto rods = row_rods.rods;

  // Each rod's rows: the transpose of row_rods, counted one slot ahead and then prefix-summed into offsets.
  int max_rod = -1;
  Kokkos::parallel_reduce(
      "make_shared_rod_pattern::max_rod", Kokkos::RangePolicy<ExecSpace>(0, rods.extent(0)),
      KOKKOS_LAMBDA(const size_t e, int& largest) { largest = rods(e) > largest ? rods(e) : largest; },
      Kokkos::Max<int>(max_rod));
  const size_t num_rods = rods.extent(0) == 0 ? 0 : static_cast<size_t>(max_rod + 1);
  const Kokkos::View<size_t*, memory_space> rod_offsets("rod_row_offsets", num_rods + 1);
  Kokkos::parallel_for(
      "make_shared_rod_pattern::rod_counts", Kokkos::RangePolicy<ExecSpace>(0, rods.extent(0)),
      KOKKOS_LAMBDA(const size_t e) { Kokkos::atomic_inc(&rod_offsets(rods(e) + 1)); });
  Kokkos::parallel_scan(
      "make_shared_rod_pattern::rod_offsets", Kokkos::RangePolicy<ExecSpace>(0, num_rods),
      KOKKOS_LAMBDA(const size_t rod, size_t& partial, const bool final) {
        partial += rod_offsets(rod + 1);
        if (final) {
          rod_offsets(rod + 1) = partial;
        }
      });
  const Kokkos::View<size_t*, memory_space> rod_cursor(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rod_cursor"),
                                                       num_rods);
  Kokkos::deep_copy(rod_cursor, Kokkos::subview(rod_offsets, Kokkos::make_pair(size_t{0}, num_rods)));
  const Kokkos::View<int*, memory_space> rod_rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rod_rows"),
                                                  rods.extent(0));
  Kokkos::parallel_for(
      "make_shared_rod_pattern::rod_rows", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const size_t row) {
        for (size_t e = row_offsets(row); e < row_offsets(row + 1); ++e) {
          rod_rows(Kokkos::atomic_fetch_inc(&rod_cursor(rods(e)))) = static_cast<int>(row);
        }
      });

  // Each row's candidates, every row of each of its rods, then sorted with duplicates merged.
  const row_map_t candidate_offsets("candidate_offsets", num_rows + 1);
  size_t num_candidates = 0;
  Kokkos::parallel_scan(
      "make_shared_rod_pattern::candidate_offsets", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const size_t row, size_t& partial, const bool final) {
        for (size_t e = row_offsets(row); e < row_offsets(row + 1); ++e) {
          partial += rod_offsets(rods(e) + 1) - rod_offsets(rods(e));
        }
        if (final) {
          candidate_offsets(row + 1) = static_cast<offset_t>(partial);
        }
      },
      num_candidates);
  const entries_t candidates(Kokkos::view_alloc(Kokkos::WithoutInitializing, "candidates"), num_candidates);
  Kokkos::parallel_for(
      "make_shared_rod_pattern::candidates", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const size_t row) {
        offset_t slot = candidate_offsets(row);
        for (size_t e = row_offsets(row); e < row_offsets(row + 1); ++e) {
          for (size_t r = rod_offsets(rods(e)); r < rod_offsets(rods(e) + 1); ++r) {
            candidates(slot++) = static_cast<ordinal_t>(rod_rows(r));
          }
        }
      });
  row_map_t row_map;
  entries_t entries;
  KokkosSparse::sort_and_merge_graph(ExecSpace{}, candidate_offsets, candidates, row_map, entries,
                                     static_cast<ordinal_t>(num_rows));

  const values_t values("shared_rod_values", entries.extent(0));
  return SparseMatrix("shared_rod_pattern", static_cast<ordinal_t>(num_rows), static_cast<ordinal_t>(num_rows),
                      static_cast<offset_t>(entries.extent(0)), values, row_map, entries);
}

/// \brief Write dt B^T M_self B + K^{-1} into matrix, whose pattern couples the rows that share a rod.
///
/// Entry (i, j) is dt sum v_ik^T M_b v_jl over the pairs (k, l) with rod(i, k) = rod(j, l) = b, plus K^{-1}_i when
/// i = j, for v_ik = [force(i, k); torque(i, k)] and M_b rod b's self block. It is evaluated from the lower-indexed
/// row, so (i, j) and (j, i) hold the same value.
template <typename ExecSpace, typename Jacobian, typename Mobility, typename SparseMatrix>
void fill_self_mobility_values(double dt, const Jacobian& jacobian, const Mobility& mobility,
                               const Kokkos::View<double*, typename ExecSpace::memory_space>& compliance,
                               const SparseMatrix& matrix) {
  const auto row_map = matrix.graph.row_map;
  const auto entries = matrix.graph.entries;
  const auto values = matrix.values;
  Kokkos::parallel_for(
      "fill_self_mobility_values", Kokkos::RangePolicy<ExecSpace>(0, matrix.numRows()), KOKKOS_LAMBDA(const size_t i) {
        for (auto e = row_map(i); e < row_map(i + 1); ++e) {
          const size_t j = static_cast<size_t>(entries(e));
          const size_t p = i < j ? i : j;
          const size_t q = i < j ? j : i;
          double v_m_v = 0.0;
          for (int k = 0; k < jacobian.num_bodies(p); ++k) {
            const int rod = jacobian.body(p, k);
            for (int l = 0; l < jacobian.num_bodies(q); ++l) {
              if (jacobian.body(q, l) != rod) {
                continue;
              }
              const auto self_block = mobility.self_mobility(rod);
              const auto force_p = jacobian.force(p, k);
              const auto torque_p = jacobian.torque(p, k);
              const auto force_q = jacobian.force(q, l);
              const auto torque_q = jacobian.torque(q, l);
              const double v_p[6] = {force_p[0], force_p[1], force_p[2], torque_p[0], torque_p[1], torque_p[2]};
              const double v_q[6] = {force_q[0], force_q[1], force_q[2], torque_q[0], torque_q[1], torque_q[2]};
              for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                  v_m_v += v_p[a] * self_block(a, b) * v_q[b];
                }
              }
            }
          }
          values(e) = dt * v_m_v + (i == j ? compliance(i) : 0.0);
        }
      });
}

}  // namespace impl

}  // namespace mbody

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

#endif  // MUNDY_MBODY_IMPL_KOKKOSMBODYPRECONDITIONERSIMPL_HPP_
