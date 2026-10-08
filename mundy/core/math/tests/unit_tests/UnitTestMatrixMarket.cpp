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

/// \file UnitTestMatrixMarket.cpp
/// \brief Matrix Market array and coordinate files (matrix_market.hpp): exact round trips, the file text, and rejected
/// files.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <array>        // for std::array
#include <cstdint>      // for std::uint32_t, std::uint64_t
#include <cstdio>       // for std::remove
#include <fstream>      // for std::ifstream, std::ofstream
#include <limits>       // for std::numeric_limits
#include <sstream>      // for std::stringstream
#include <stdexcept>    // for std::runtime_error
#include <string>       // for std::string
#include <type_traits>  // for std::conditional_t
#include <vector>       // for std::vector

// Mundy
#include <MundyMath_config.hpp>            // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix, mundy::get_matrix
#include <mundy_math/Vector.hpp>           // for mundy::Vector, mundy::get_vector
#include <mundy_math/cmath.hpp>            // for mundy::bit_cast
#include <mundy_math/matrix_market.hpp>    // for mundy::write_matrix_market, mundy::read_matrix_market
#include <mundy_math/solver_backends.hpp>  // for mundy::{KokkosBackend, MundyMathBackend}
#include <mundy_math/sparse_matrix.hpp>    // for mundy::impl::{SparseEntry, make_sparse_matrix}

namespace mundy {

namespace {

//! \name Helpers
//@{

/// \brief Values a fixed-point or truncating writer gets wrong, plus the special values.
template <class Scalar>
std::vector<Scalar> hard_values() {
  using limits = std::numeric_limits<Scalar>;
  return {Scalar(0.1),     Scalar(-1.2345678901234566e-07), Scalar(3e-17),        Scalar(1) / Scalar(3),
          limits::min(),   limits::denorm_min(),            Scalar(-0.0),         limits::max(),
          limits::lowest(), limits::infinity(),             -limits::infinity(), limits::quiet_NaN()};
}

/// \brief Whether a and b have the same bits; every NaN matches every NaN.
template <class Scalar>
bool same_bits(Scalar a, Scalar b) {
  using bits_t = std::conditional_t<sizeof(Scalar) == 8, std::uint64_t, std::uint32_t>;
  return (a != a && b != b) || bit_cast<bits_t>(a) == bit_cast<bits_t>(b);
}

/// \brief Write text to filename, standing in for a file from another writer.
void write_text(const std::string& filename, const std::string& text) {
  std::ofstream(filename) << text;
}

/// \brief The whole text of filename.
std::string read_text(const std::string& filename) {
  std::stringstream text;
  text << std::ifstream(filename).rdbuf();
  return text.str();
}
//@}

//! \name Kokkos views
//@{

template <class Scalar>
void expect_vector_round_trip(const std::string& filename) {
  const std::vector<Scalar> values = hard_values<Scalar>();
  Kokkos::View<Scalar*, Kokkos::HostSpace> written("written", values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    written(i) = values[i];
  }
  write_matrix_market(filename, written);
  Kokkos::View<Scalar*, Kokkos::HostSpace> read("read", 0);
  read_matrix_market(filename, read);
  ASSERT_EQ(read.extent(0), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_TRUE(same_bits(read(i), values[i])) << "entry " << i << ": wrote " << values[i] << ", read " << read(i);
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, VectorsRoundTripExactly) {
  expect_vector_round_trip<double>("MatrixMarket_double_vector.mtx");
  expect_vector_round_trip<float>("MatrixMarket_float_vector.mtx");
}

template <class Layout>
void expect_matrix_round_trip(const std::string& filename) {
  const std::vector<double> values = hard_values<double>();
  Kokkos::View<double**, Layout, Kokkos::HostSpace> written("written", 3, 4);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      written(i, j) = values[(4 * i + j) % values.size()];
    }
  }
  write_matrix_market(filename, written);
  Kokkos::View<double**, Layout, Kokkos::HostSpace> read("read", 0, 0);
  read_matrix_market(filename, read);
  ASSERT_EQ(read.extent(0), 3u);
  ASSERT_EQ(read.extent(1), 4u);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      EXPECT_TRUE(same_bits(read(i, j), written(i, j))) << "entry (" << i << ", " << j << ")";
    }
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, MatricesRoundTripExactlyInEitherLayout) {
  expect_matrix_round_trip<Kokkos::LayoutLeft>("MatrixMarket_layout_left.mtx");
  expect_matrix_round_trip<Kokkos::LayoutRight>("MatrixMarket_layout_right.mtx");
}

TEST(MatrixMarket, WritesTheStandardArrayFormat) {
  // The banner, the extents, then the entries in column-major order, one per line.
  const std::string filename = "MatrixMarket_format.mtx";
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> matrix("matrix", 2, 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix(i, j) = 1.0 + static_cast<double>(3 * i + j);  // [[1, 2, 3], [4, 5, 6]]
    }
  }
  write_matrix_market(filename, matrix);
  EXPECT_EQ(read_text(filename), "%%MatrixMarket matrix array real general\n2 3\n1\n4\n2\n5\n3\n6\n");

  Kokkos::View<double*, Kokkos::HostSpace> vector("vector", 2);
  vector(0) = 0.5;
  vector(1) = -2.0;
  write_matrix_market(filename, vector);
  EXPECT_EQ(read_text(filename), "%%MatrixMarket matrix array real general\n2 1\n0.5\n-2\n");
  std::remove(filename.c_str());
}

TEST(MatrixMarket, ReadsFilesFromOtherWriters) {
  // Any case in the banner, comments, blank lines, integer entries, a leading '+', and several entries per line.
  const std::string filename = "MatrixMarket_other_writer.mtx";
  write_text(filename,
             "%%matrixmarket MATRIX Array Integer General\n% written by hand\n\n2 2\n+1 3\n% interlude\n2\n   4  \n");
  Kokkos::View<double**, Kokkos::HostSpace> integers("integers", 0, 0);
  read_matrix_market(filename, integers);
  ASSERT_EQ(integers.extent(0), 2u);
  EXPECT_EQ(integers(0, 0), 1.0);
  EXPECT_EQ(integers(1, 0), 3.0);
  EXPECT_EQ(integers(0, 1), 2.0);
  EXPECT_EQ(integers(1, 1), 4.0);

  write_text(filename, "%%MatrixMarket matrix array real general\n3 1\n1.500000000000000000e+00\n+2.5E-1\n-inf\n");
  Kokkos::View<double*, Kokkos::HostSpace> reals("reals", 0);
  read_matrix_market(filename, reals);
  ASSERT_EQ(reals.extent(0), 3u);
  EXPECT_EQ(reals(0), 1.5);
  EXPECT_EQ(reals(1), 0.25);
  EXPECT_EQ(reals(2), -std::numeric_limits<double>::infinity());
  std::remove(filename.c_str());
}

TEST(MatrixMarket, ResizesToTheFileAndKeepsMatchingAllocations) {
  const std::string filename = "MatrixMarket_resize.mtx";
  Kokkos::View<double*, Kokkos::HostSpace> five("five", 5);
  write_matrix_market(filename, five);

  Kokkos::View<double*, Kokkos::HostSpace> longer("longer", 7);
  read_matrix_market(filename, longer);
  EXPECT_EQ(longer.extent(0), 5u);

  Kokkos::View<double*, Kokkos::HostSpace> same("same", 5);
  const double* const allocation = same.data();
  read_matrix_market(filename, same);
  EXPECT_EQ(same.data(), allocation) << "a view that already matches the file keeps its allocation";
  std::remove(filename.c_str());
}

TEST(MatrixMarket, RejectsInvalidFiles) {
  const std::string filename = "MatrixMarket_invalid.mtx";
  Kokkos::View<double*, Kokkos::HostSpace> vector("vector", 0);
  Kokkos::View<double**, Kokkos::HostSpace> matrix("matrix", 0, 0);
  const auto expect_rejected = [&](const std::string& text, const std::string& why) {
    write_text(filename, text);
    EXPECT_THROW(read_matrix_market(filename, matrix), std::runtime_error) << why;
  };
  expect_rejected("3\n1\n2\n3\n", "a legacy file without a banner");
  expect_rejected("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 5\n", "the sparse coordinate format");
  expect_rejected("%%MatrixMarket matrix array complex general\n1 1\n1 2\n", "complex entries");
  expect_rejected("%%MatrixMarket matrix array real symmetric\n2 2\n1\n2\n3\n", "a symmetric file");
  expect_rejected("%%MatrixMarket matrix array real general\n2\n1\n2\n", "an extents line without cols");
  expect_rejected("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n", "too few entries");
  expect_rejected("%%MatrixMarket matrix array real general\n2 1\n1\n2\n3\n", "too many entries");
  expect_rejected("%%MatrixMarket matrix array real general\n2 1\n1\none\n", "an entry that is not a number");
  expect_rejected("%%MatrixMarket matrix array real general\n1 1\n1e999\n", "an entry beyond double's range");
  EXPECT_THROW(read_matrix_market("MatrixMarket_missing.mtx", matrix), std::runtime_error) << "a missing file";

  write_text(filename, "%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n");
  EXPECT_THROW(read_matrix_market(filename, vector), std::runtime_error) << "a 2 x 2 file into a vector";
  EXPECT_THROW(write_matrix_market("MatrixMarket_missing_directory/file.mtx", vector), std::runtime_error);
  std::remove(filename.c_str());
}

/// \brief Fill matrix(i, j) = 1 / (1 + i + 7 j) on the default execution space.
///
/// A free function, since CUDA forbids KOKKOS_LAMBDA in a test body (a private member function).
template <class View>
void fill_on_device(const View& matrix) {
  Kokkos::parallel_for(
      "MatrixMarket::fill", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {matrix.extent(0), matrix.extent(1)}),
      KOKKOS_LAMBDA(const int i, const int j) { matrix(i, j) = 1.0 / (1.0 + i + 7 * j); });
}

TEST(MatrixMarket, DeviceViewsRoundTrip) {
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  const std::string filename = "MatrixMarket_device.mtx";
  Kokkos::View<double**, Kokkos::LayoutLeft, memory_space> written("written", 4, 3);
  fill_on_device(written);
  write_matrix_market(filename, written);
  Kokkos::View<double**, Kokkos::LayoutLeft, memory_space> read("read", 0, 0);
  read_matrix_market(filename, read);
  const auto written_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, written);
  const auto read_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, read);
  ASSERT_EQ(read_host.extent(0), 4u);
  ASSERT_EQ(read_host.extent(1), 3u);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_TRUE(same_bits(read_host(i, j), written_host(i, j))) << "entry (" << i << ", " << j << ")";
    }
  }
  std::remove(filename.c_str());
}
//@}

//! \name AVector and AMatrix
//@{

TEST(MatrixMarket, MundyVectorsAndMatricesRoundTripExactly) {
  const std::string filename = "MatrixMarket_mundy.mtx";
  const std::vector<double> values = hard_values<double>();
  const Vector<double, 4> vector{values[0], values[1], values[5], values[9]};
  write_matrix_market(filename, vector);
  Vector<double, 4> vector_read;
  read_matrix_market(filename, vector_read);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(same_bits(vector_read[i], vector[i])) << "vector entry " << i;
  }

  const Matrix<double, 2, 3> matrix{values[0], values[1], values[2], values[3], values[4], values[6]};
  write_matrix_market(filename, matrix);
  Matrix<double, 2, 3> matrix_read;
  read_matrix_market(filename, matrix_read);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_TRUE(same_bits(matrix_read(i, j), matrix(i, j))) << "matrix entry (" << i << ", " << j << ")";
    }
  }

  const std::vector<float> floats = hard_values<float>();
  const Matrix<float, 3, 2> float_matrix{floats[0], floats[1], floats[2], floats[3], floats[5], floats[11]};
  write_matrix_market(filename, float_matrix);
  Matrix<float, 3, 2> float_matrix_read;
  read_matrix_market(filename, float_matrix_read);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_TRUE(same_bits(float_matrix_read(i, j), float_matrix(i, j))) << "float entry (" << i << ", " << j << ")";
    }
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, BothBackendsShareOneFormat) {
  const std::string filename = "MatrixMarket_both_backends.mtx";
  Kokkos::View<double**, Kokkos::HostSpace> view("view", 2, 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      view(i, j) = 0.1 * static_cast<double>(3 * i + j);
    }
  }
  write_matrix_market(filename, view);
  Matrix<double, 2, 3> matrix;
  read_matrix_market(filename, matrix);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(matrix(i, j), view(i, j)) << "view to matrix, entry (" << i << ", " << j << ")";
    }
  }

  const Vector<double, 3> vector{0.1, 0.2, 0.3};
  write_matrix_market(filename, vector);
  Kokkos::View<double*, Kokkos::HostSpace> vector_view("vector_view", 0);
  read_matrix_market(filename, vector_view);
  ASSERT_EQ(vector_view.extent(0), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(vector_view(i), vector[i]) << "vector to view, entry " << i;
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, ReadsThroughWritableAccessors) {
  // A Mundy vector or matrix over other storage reads straight into that storage.
  const std::string filename = "MatrixMarket_accessor.mtx";
  write_text(filename, "%%MatrixMarket matrix array real general\n2 2\n1\n3\n2\n4\n");
  std::array<double, 4> storage{};
  double* data = storage.data();
  auto matrix = get_matrix<double, 2, 2>(data);
  read_matrix_market(filename, matrix);
  EXPECT_EQ(storage, (std::array<double, 4>{1.0, 2.0, 3.0, 4.0})) << "row-major storage of [[1, 2], [3, 4]]";

  write_text(filename, "%%MatrixMarket matrix array real general\n4 1\n5\n6\n7\n8\n");
  auto vector = get_vector<double, 4>(data);
  read_matrix_market(filename, vector);
  EXPECT_EQ(storage, (std::array<double, 4>{5.0, 6.0, 7.0, 8.0}));
  std::remove(filename.c_str());
}

TEST(MatrixMarket, MundyTypesRequireTheFileExtents) {
  const std::string filename = "MatrixMarket_extents.mtx";
  write_text(filename, "%%MatrixMarket matrix array real general\n3 1\n1\n2\n3\n");
  Vector<double, 4> vector;
  EXPECT_THROW(read_matrix_market(filename, vector), std::runtime_error) << "a 3 x 1 file into a 4-vector";

  write_text(filename, "%%MatrixMarket matrix array real general\n2 3\n1\n2\n3\n4\n5\n6\n");
  Matrix<double, 3, 2> matrix;
  EXPECT_THROW(read_matrix_market(filename, matrix), std::runtime_error) << "a 2 x 3 file into a 3 x 2 matrix";
  std::remove(filename.c_str());
}
//@}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
//! \name Sparse matrices
//@{

/// \brief A sparse matrix of doubles in Space's memory.
template <class Space>
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Space, typename Space::memory_space>, void, size_t>;

/// \brief A rows x cols sparse matrix in Space's memory storing entries, ordered by row and then column.
template <class Space>
sparse_matrix_t<Space> make_sparse(size_t rows, size_t cols, const std::vector<impl::SparseEntry<double>>& entries) {
  return impl::make_sparse_matrix<sparse_matrix_t<Space>>(rows, cols, entries);
}

// A 5 x 4 matrix storing every hard value, -0 included, with row 2 empty.
template <class Space>
void expect_sparse_round_trip(const std::string& filename) {
  const std::vector<double> values = hard_values<double>();
  const size_t positions[12][2] = {{0, 0}, {0, 2}, {0, 3}, {1, 1}, {1, 3}, {3, 0},
                                   {3, 1}, {3, 2}, {4, 0}, {4, 1}, {4, 2}, {4, 3}};
  std::vector<impl::SparseEntry<double>> entries;
  for (size_t k = 0; k < values.size(); ++k) {
    entries.push_back({positions[k][0], positions[k][1], values[k]});
  }
  const sparse_matrix_t<Space> written = make_sparse<Space>(5, 4, entries);
  write_matrix_market(filename, written);
  sparse_matrix_t<Space> read;
  read_matrix_market(filename, read);

  ASSERT_EQ(read.numRows(), 5);
  ASSERT_EQ(read.numCols(), 4);
  ASSERT_EQ(read.nnz(), values.size());
  const auto row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, read.graph.row_map);
  const auto cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, read.graph.entries);
  const auto read_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, read.values);
  const size_t expected_row_map[6] = {0, 3, 5, 5, 8, 12};
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(row_map(i), expected_row_map[i]) << "row offset " << i;
  }
  for (size_t k = 0; k < values.size(); ++k) {
    EXPECT_EQ(static_cast<size_t>(cols(k)), positions[k][1]) << "entry " << k;
    EXPECT_TRUE(same_bits(read_values(k), values[k]))
        << "entry " << k << ": wrote " << values[k] << ", read " << read_values(k);
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, SparseMatricesRoundTripExactly) {
  expect_sparse_round_trip<Kokkos::DefaultHostExecutionSpace>("MatrixMarket_sparse_host.mtx");
  expect_sparse_round_trip<Kokkos::DefaultExecutionSpace>("MatrixMarket_sparse_device.mtx");
}

TEST(MatrixMarket, WritesTheStandardCoordinateFormat) {
  // The banner, the extents and stored-entry count, then one "row col value" line per stored entry, one-based.
  const std::string filename = "MatrixMarket_coordinate_format.mtx";
  write_matrix_market(filename, make_sparse<Kokkos::DefaultHostExecutionSpace>(2, 3, {{0, 0, 1.5}, {1, 2, -2.0}}));
  EXPECT_EQ(read_text(filename), "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 1.5\n2 3 -2\n");
  std::remove(filename.c_str());
}

TEST(MatrixMarket, ReadsCoordinateFilesFromOtherWriters) {
  // Any case in the banner, comments, blank lines, integer entries, a leading '+', and entries in any order.
  const std::string filename = "MatrixMarket_coordinate_other_writer.mtx";
  write_text(filename,
             "%%MATRIXMARKET Matrix Coordinate Integer General\n% written by hand\n\n3 3 4\n 3 1   +5\n1 2 -7\n"
             "% interlude\n2 2 1\n1 1 2\n");
  sparse_matrix_t<Kokkos::DefaultHostExecutionSpace> matrix;
  read_matrix_market(filename, matrix);
  ASSERT_EQ(matrix.numRows(), 3);
  ASSERT_EQ(matrix.numCols(), 3);
  ASSERT_EQ(matrix.nnz(), 4u);
  const size_t expected_row_map[4] = {0, 2, 3, 4};
  const int expected_cols[4] = {0, 1, 1, 0};
  const double expected_values[4] = {2.0, -7.0, 1.0, 5.0};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(matrix.graph.row_map(i), expected_row_map[i]) << "row offset " << i;
    EXPECT_EQ(matrix.graph.entries(i), expected_cols[i]) << "entry " << i;
    EXPECT_EQ(matrix.values(i), expected_values[i]) << "entry " << i;
  }
  std::remove(filename.c_str());
}

TEST(MatrixMarket, RejectsInvalidCoordinateFiles) {
  const std::string filename = "MatrixMarket_invalid_coordinate.mtx";
  sparse_matrix_t<Kokkos::DefaultHostExecutionSpace> matrix;
  const auto expect_rejected = [&](const std::string& text, const std::string& why) {
    write_text(filename, text);
    EXPECT_THROW(read_matrix_market(filename, matrix), std::runtime_error) << why;
  };
  const std::string banner = "%%MatrixMarket matrix coordinate real general\n";
  expect_rejected("%%MatrixMarket matrix array real general\n1 1\n1\n", "the dense array format");
  expect_rejected("%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 1 1\n", "a symmetric file");
  expect_rejected("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 1\n", "a pattern file");
  expect_rejected(banner + "2 2\n1 1 1\n", "an extents line without nnz");
  expect_rejected(banner + "2 2 2\n1 1 1\n", "too few entries");
  expect_rejected(banner + "2 2 1\n1 1 1\n2 2 1\n", "too many entries");
  expect_rejected(banner + "2 2 2\n1 1 1\n1 1 2\n", "an entry listed twice");
  expect_rejected(banner + "2 2 1\n0 1 1\n", "a row index below one");
  expect_rejected(banner + "2 2 1\n1 3 1\n", "a column index beyond the columns");
  expect_rejected(banner + "2 2 1\n1 1 one\n", "an entry that is not a number");
  std::remove(filename.c_str());
}
//@}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

//! \name Linear operators
//@{

/// \brief y_i = x_{i+1} - x_i, an (n - 1) x n operator with no stored matrix, applied on the default execution space.
struct ForwardDifference {
  using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

  size_t n;

  size_t domain_size() const {
    return n;
  }
  size_t range_size() const {
    return n - 1;
  }
  view_t make_domain_vector() const {
    return view_t("x", n);
  }
  view_t make_range_vector() const {
    return view_t("y", n - 1);
  }
  void apply(const view_t& x, view_t& y) const {
    Kokkos::parallel_for(
        "ForwardDifference", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n - 1),
        KOKKOS_LAMBDA(const int i) { y(i) = x(i + 1) - x(i); });
  }
};

// An operator is written as its matrix: one with a stored matrix writes exactly that matrix, and one without writes
// the matrix of its action, here an exact (n - 1) x n difference stencil with the range size as its rows.
TEST(MatrixMarket, OperatorsWriteTheirMatrix) {
  const std::string filename = "MatrixMarket_operator.mtx";

  // A stored matrix, through the MundyMath backend
  const std::vector<double> values = hard_values<double>();
  const Matrix<double, 3, 2> matrix{values[0], values[1], values[2], values[3], values[4], values[5]};
  write_matrix_market<MundyMathBackend>(filename, matrix);
  Matrix<double, 3, 2> matrix_read;
  read_matrix_market(filename, matrix_read);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_TRUE(same_bits(matrix_read(i, j), matrix(i, j))) << "matrix entry (" << i << ", " << j << ")";
    }
  }

  // No stored matrix, through the Kokkos backend on the default execution space
  const size_t n = 5;
  write_matrix_market<KokkosBackend<Kokkos::DefaultExecutionSpace>>(filename, ForwardDifference{n});
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> stencil("stencil", 0, 0);
  read_matrix_market(filename, stencil);
  ASSERT_EQ(stencil.extent(0), n - 1);
  ASSERT_EQ(stencil.extent(1), n);
  for (size_t i = 0; i + 1 < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      EXPECT_EQ(stencil(i, j), j == i ? -1.0 : (j == i + 1 ? 1.0 : 0.0)) << "entry (" << i << ", " << j << ")";
    }
  }

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // A sparse matrix, through the Kokkos backend on the default execution space
  write_matrix_market<KokkosBackend<Kokkos::DefaultExecutionSpace>>(
      filename, make_sparse<Kokkos::DefaultExecutionSpace>(2, 3, {{0, 0, 1.5}, {1, 2, -2.0}}));
  Kokkos::View<double**, Kokkos::HostSpace> dense("dense", 0, 0);
  read_matrix_market(filename, dense);
  ASSERT_EQ(dense.extent(0), 2u);
  ASSERT_EQ(dense.extent(1), 3u);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(dense(i, j), i == 0 && j == 0 ? 1.5 : (i == 1 && j == 2 ? -2.0 : 0.0))
          << "entry (" << i << ", " << j << ")";
    }
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
  std::remove(filename.c_str());
}
//@}

}  // namespace

}  // namespace mundy
