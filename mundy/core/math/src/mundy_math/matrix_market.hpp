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

#ifndef MUNDY_MATH_MATRIX_MARKET_HPP_
#define MUNDY_MATH_MATRIX_MARKET_HPP_

/// \file matrix_market.hpp
/// \brief Write and read vectors and matrices as Matrix Market array files, exactly.
///
/// The Matrix Market array format is the dense text format of numpy and scipy (mmread, mmwrite), MATLAB, Tpetra, and
/// Eigen: the banner "%%MatrixMarket matrix array real general", the extents "rows cols", then the entries in
/// column-major order, one per line. A vector is a rows x 1 column. Each entry is written as the shortest decimal that
/// reads back to the same bits, so writing then reading reproduces every value exactly, including -0, infinities, and
/// NaN.
///
/// Both solver backends are supported. Kokkos views (rank 1 or 2, any memory space and layout) are resized to the
/// file's extents. AVector and AMatrix have compile-time sizes, so the file's extents must equal them.
///
/// These functions are serial: every process that calls them opens the file itself, so under MPI only one process
/// should write a given file.

// External
#include <Kokkos_Core.hpp>  // for Kokkos::View, Kokkos::resize, Kokkos::create_mirror_view, ...

// C++ core
#include <string>       // for std::string
#include <type_traits>  // for std::is_same_v, std::remove_cv_t

// Mundy
#include <mundy_math/Accessor.hpp>                 // for mundy::ValidAccessor
#include <mundy_math/Matrix.hpp>                   // for mundy::AMatrix
#include <mundy_math/Vector.hpp>                   // for mundy::AVector
#include <mundy_math/impl/matrix_market_impl.hpp>  // for mundy::impl::{write_matrix_market, MatrixMarketReader}
#include <mundy_utils/requires.hpp>                // for MUNDY_REQUIRES
#include <mundy_utils/throw_assert.hpp>            // for MUNDY_THROW_REQUIRE

namespace mundy {

//! \name Kokkos views
//@{

/// \brief Write a rank-1 or rank-2 view to filename as a Matrix Market array file.
template <class View>
MUNDY_REQUIRES(Kokkos::is_view_v<View>)
void write_matrix_market(const std::string& filename, const View& view) {
  using scalar_t = typename View::non_const_value_type;
  static_assert(View::rank() == 1 || View::rank() == 2, "write_matrix_market: views must be rank 1 or 2.");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  if constexpr (View::rank() == 1) {
    impl::write_matrix_market<scalar_t>(filename, host.extent(0), 1, [&](size_t i, size_t) { return host(i); });
  } else {
    impl::write_matrix_market<scalar_t>(filename, host.extent(0), host.extent(1),
                                        [&](size_t i, size_t j) { return host(i, j); });
  }
}

/// \brief Read a Matrix Market array file into a rank-1 or rank-2 view, resized to the file's extents.
///
/// A rank-1 view reads a rows x 1 file. Resizing keeps the view's allocation when its extents already match.
template <class View>
MUNDY_REQUIRES(Kokkos::is_view_v<View>)
void read_matrix_market(const std::string& filename, View& view) {
  using scalar_t = typename View::value_type;
  static_assert(View::rank() == 1 || View::rank() == 2, "read_matrix_market: views must be rank 1 or 2.");
  static_assert(!std::is_const_v<scalar_t>, "read_matrix_market: views must be writable.");
  impl::MatrixMarketReader reader(filename);
  if constexpr (View::rank() == 1) {
    reader.require_extents(reader.rows(), 1);
    Kokkos::resize(view, reader.rows());
  } else {
    Kokkos::resize(view, reader.rows(), reader.cols());
  }
  const auto host = Kokkos::create_mirror_view(view);
  reader.read<scalar_t>([&](size_t i, size_t j, scalar_t value) {
    if constexpr (View::rank() == 1) {
      host(i) = value;
    } else {
      host(i, j) = value;
    }
  });
  Kokkos::deep_copy(view, host);
}
//@}

//! \name AVector and AMatrix
//@{

/// \brief Write an N-vector to filename as an N x 1 Matrix Market array file.
template <typename T, size_t N, ValidAccessor<T> Accessor>
void write_matrix_market(const std::string& filename, const AVector<T, N, Accessor>& vec) {
  impl::write_matrix_market<std::remove_cv_t<T>>(filename, N, 1, [&](size_t i, size_t) { return vec[i]; });
}

/// \brief Write an N x M matrix to filename as a Matrix Market array file.
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
void write_matrix_market(const std::string& filename, const AMatrix<T, N, M, Accessor>& mat) {
  impl::write_matrix_market<std::remove_cv_t<T>>(filename, N, M, [&](size_t i, size_t j) { return mat(i, j); });
}

/// \brief Read an N x 1 Matrix Market array file into an N-vector.
template <typename T, size_t N, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
void read_matrix_market(const std::string& filename, AVector<T, N, Accessor>& vec) {
  impl::MatrixMarketReader reader(filename);
  reader.require_extents(N, 1);
  reader.read<T>([&](size_t i, size_t, T value) { vec[i] = value; });
}

/// \brief Read an N x M Matrix Market array file into an N x M matrix.
template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
void read_matrix_market(const std::string& filename, AMatrix<T, N, M, Accessor>& mat) {
  impl::MatrixMarketReader reader(filename);
  reader.require_extents(N, M);
  reader.read<T>([&](size_t i, size_t j, T value) { mat(i, j) = value; });
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX_MARKET_HPP_
