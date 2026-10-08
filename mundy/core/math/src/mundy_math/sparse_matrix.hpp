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

#ifndef MUNDY_MATH_SPARSE_MATRIX_HPP_
#define MUNDY_MATH_SPARSE_MATRIX_HPP_

/// \file sparse_matrix.hpp
/// \brief Sparse matrices: KokkosSparse::CrsMatrix, the compressed-row peer of a dense rank-2 view.
///
/// The Kokkos backend applies either kind of matrix view, so every operator algebra and solver accepts both.
/// make_sparse_matrix stores a dense matrix's nonzero entries in a sparse one.
///
/// This header is a no-op unless the KokkosKernels TPL is enabled (HAVE_MUNDYMATH_KOKKOSKERNELS).

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS

// External
#include <KokkosSparse_CrsMatrix.hpp>  // for KokkosSparse::CrsMatrix
#include <Kokkos_Core.hpp>             // for Kokkos::View, Kokkos::create_mirror_view, Kokkos::deep_copy

// C++ core
#include <cstddef>  // for size_t
#include <vector>   // for std::vector

// Mundy
#include <mundy_math/impl/solver_backends_impl.hpp>  // for mundy::impl::{DenseMatView, SparseMatView}
#include <mundy_utils/requires.hpp>                  // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>              // for MUNDY_THROW_ASSERT

namespace mundy {

namespace impl {

/// \brief One entry of a sparse matrix: value at (row, col), zero-based.
template <class Scalar>
struct SparseEntry {
  size_t row;
  size_t col;
  Scalar value;
};

/// \brief A rows x cols SparseMatrix holding entries, which are ordered by row and, within a row, by column.
template <class SparseMatrix, class Scalar>
SparseMatrix make_sparse_matrix(size_t rows, size_t cols, const std::vector<SparseEntry<Scalar>>& entries) {
  using ordinal_t = typename SparseMatrix::non_const_ordinal_type;
  using offset_t = typename SparseMatrix::non_const_size_type;
  using value_t = typename SparseMatrix::non_const_value_type;
  const size_t nnz = entries.size();

  typename SparseMatrix::row_map_type::non_const_type row_map("row_map", rows + 1);
  typename SparseMatrix::index_type::non_const_type cols_of(Kokkos::view_alloc(Kokkos::WithoutInitializing, "entries"),
                                                            nnz);
  typename SparseMatrix::values_type::non_const_type values(Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"),
                                                            nnz);
  const auto row_map_host = Kokkos::create_mirror_view(row_map);
  const auto cols_of_host = Kokkos::create_mirror_view(cols_of);
  const auto values_host = Kokkos::create_mirror_view(values);

  // Count each row's entries one slot ahead, then prefix-sum the counts into row offsets.
  Kokkos::deep_copy(row_map_host, offset_t(0));
  for (size_t k = 0; k < nnz; ++k) {
    MUNDY_THROW_ASSERT(entries[k].row < rows && entries[k].col < cols, std::invalid_argument,
                       "make_sparse_matrix: an entry lies outside the matrix.");
    MUNDY_THROW_ASSERT(k == 0 || entries[k - 1].row < entries[k].row ||
                           (entries[k - 1].row == entries[k].row && entries[k - 1].col < entries[k].col),
                       std::invalid_argument, "make_sparse_matrix: entries must be ordered by row, then column.");
    ++row_map_host(entries[k].row + 1);
    cols_of_host(k) = static_cast<ordinal_t>(entries[k].col);
    values_host(k) = static_cast<value_t>(entries[k].value);
  }
  for (size_t i = 0; i < rows; ++i) {
    row_map_host(i + 1) += row_map_host(i);
  }
  Kokkos::deep_copy(row_map, row_map_host);
  Kokkos::deep_copy(cols_of, cols_of_host);
  Kokkos::deep_copy(values, values_host);
  return SparseMatrix("sparse_matrix", static_cast<ordinal_t>(rows), static_cast<ordinal_t>(cols),
                      static_cast<offset_t>(nnz), values, row_map, cols_of);
}

}  // namespace impl

/// \brief The SparseMatrix holding every nonzero entry of dense, a rank-2 view, in SparseMatrix's memory space.
///
/// Each row's entries are ordered by column. Host-side.
template <class SparseMatrix, class DenseMatrix>
MUNDY_REQUIRES(impl::SparseMatView<SparseMatrix>&& impl::DenseMatView<DenseMatrix>)
SparseMatrix make_sparse_matrix(const DenseMatrix& dense) {
  using value_t = typename SparseMatrix::non_const_value_type;
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dense);
  std::vector<impl::SparseEntry<value_t>> entries;
  for (size_t i = 0; i < host.extent(0); ++i) {
    for (size_t j = 0; j < host.extent(1); ++j) {
      if (host(i, j) != 0) {
        entries.push_back({i, j, static_cast<value_t>(host(i, j))});
      }
    }
  }
  return impl::make_sparse_matrix<SparseMatrix>(host.extent(0), host.extent(1), entries);
}

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

#endif  // MUNDY_MATH_SPARSE_MATRIX_HPP_
