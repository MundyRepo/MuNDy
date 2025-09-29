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

#ifndef MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_
#define MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_

/* This class evaluates the fluid flow on some points within the domain induced by the no-slip confition on the
periphery. We do so via storing the precomputed inverse of the self-interaction matrix. Ths class is in charge
of writing out that inverse if it doesn't already exist, reading it into memory if it does, evaluating the surface
forces induced by external flow on the surface (by evaluating f = M^{-1}u), and then evaluating the fluid flow at points
within the domain induced by these surface forces.

The periphery is described by a collection of nodes with inward-pointing surface normals and predefined quadrature
weights. We are not in charge of computing these quantities.

We will write this entire class with raw Kokkos.

Because the Periphery itself is only designed for smallish-scale problems (N<5000), we won't worry about distributing
the periphery across multiple devices. After all, a 5000x5000 matrix is only 200MB, which is small enough to fit on
a single GPU.


*/

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <KokkosBlas_gesv.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_alens/periphery/Gauss_Legendre_Nodes_and_Weights.hpp>  // for Gauss_Legendre_Nodes_and_Weights
#include <mundy_core/throw_assert.hpp>                                 // for MUNDY_THROW_ASSERT
#include <mundy_math/Vector3.hpp>                                      // for mundy::math::Vector3
#define DOUBLE_ZERO 1.0e-12

namespace mundy {

namespace alens {

namespace periphery {

KOKKOS_INLINE_FUNCTION
double quake_inv_sqrt(double number) {
  double y = number;
  double x2 = y * 0.5;
  std::int64_t i = *(std::int64_t *)&y;
  // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
  i = 0x5fe6eb50c7b537a9 - (i >> 1);
  y = *(double *)&i;
  y = y * (1.5 - (x2 * y * y));  // 1st iteration
  //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed (left out of respect for Quake)
  return y;
}

/// \brief Get the Gauss Legrandre-based quadrature weights, nodes, and normals for a sphere
///
/// Point order: 0 at northpole, then 2p+2 points per circle. the last at south pole
/// The north and south pole are not included in the nodesGL of Gauss-Legendre nodes.
/// We add those two points with weight = 0 manually.
/// total point = (p+1)(2p+2) + north/south pole = 2p^2+4p+4
void gen_sphere_quadrature(const int &order, const double &radius, std::vector<double> *const points_ptr,
                           std::vector<double> *const weights_ptr, std::vector<double> *const normals_ptr,
                           const bool include_poles = false, const bool invert = false) {
  MUNDY_THROW_REQUIRE(order >= 0, std::invalid_argument, "gen_sphere_quadrature: order must be non-negative.");
  MUNDY_THROW_REQUIRE(radius > 0, std::invalid_argument,
                      fmt::format("gen_sphere_quadrature: radius must be positive. The current value is {}", radius));
  MUNDY_THROW_REQUIRE(points_ptr != nullptr, std::invalid_argument,
                      "gen_sphere_quadrature: points_ptr must be non-null.");
  MUNDY_THROW_REQUIRE(weights_ptr != nullptr, std::invalid_argument,
                      "gen_sphere_quadrature: weights_ptr must be non-null.");
  MUNDY_THROW_REQUIRE(normals_ptr != nullptr, std::invalid_argument,
                      "gen_sphere_quadrature: normals_ptr must be non-null.");

  // Get references to the vectors
  std::vector<double> &points = *points_ptr;
  std::vector<double> &weights = *weights_ptr;
  std::vector<double> &normals = *normals_ptr;

  // Resize the vectors
  const int num_points = (order + 1) * (2 * order + 2) + (include_poles ? 2 : 0);
  points.resize(3 * num_points);
  weights.resize(num_points);
  normals.resize(3 * num_points);

  // Compute the Gauss-Legendre nodes and weights
  std::vector<double> nodes_gl;  // cos thetaj = tj
  std::vector<double> weights_gl;
  Gauss_Legendre_Nodes_and_Weights(order + 1u, nodes_gl, weights_gl);  // order+1 points, excluding the two poles

  // Calculate the grid cordinates with the [0, 0, 1] at the north pole and [0, 0, -1] at the south pole.
  if (include_poles) {
    // North pole:
    points[0] = 0;
    points[1] = 0;
    points[2] = 1;
    weights[0] = 0;
  }

  // Between north and south pole:
  // from north pole (1) to south pole (-1), picking the points from nodes_gl in reversed order
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  const double weightfactor = radius * radius * 2 * pi / (2 * order + 2);
  for (int j = 0; j < order + 1; j++) {
    for (int k = 0; k < 2 * order + 2; k++) {
      const double costhetaj = nodes_gl[order - j];
      const double phik = 2 * pi * k / (2 * order + 2);
      const double sinthetaj = Kokkos::sqrt(1 - costhetaj * costhetaj);
      const int index = (j * (2 * order + 2)) + k + (include_poles ? 1 : 0);
      points[3 * index] = sinthetaj * Kokkos::cos(phik);
      points[3 * index + 1] = sinthetaj * Kokkos::sin(phik);
      points[3 * index + 2] = costhetaj;
      weights[index] = weightfactor * weights_gl[order - j];  // area element = sin thetaj
    }
  }

  if (include_poles) {
    // South pole:
    points[3 * (num_points - 1)] = 0;
    points[3 * (num_points - 1) + 1] = 0;
    points[3 * (num_points - 1) + 2] = -1;
    weights[num_points - 1] = 0;
  }

  // On the unit sphere, grid norms equal grid coordinates
  for (int i = 0; i < num_points; i++) {
    const double sign = invert ? -1 : 1;
    normals[3 * i] = sign * points[3 * i];
    normals[3 * i + 1] = sign * points[3 * i + 1];
    normals[3 * i + 2] = sign * points[3 * i + 2];
  }

  // Scale the points by the radius
  for (int i = 0; i < num_points; i++) {
    points[3 * i] *= radius;
    points[3 * i + 1] *= radius;
    points[3 * i + 2] *= radius;
  }
}

/// \brief Invert and LU decompose a dense square matrix of size n x n
///
/// \param space The execution space
/// \param[in & out] matrix The matrix to invert. On exit, the matrix is replaced with its LU decomposition
/// \param[out] M_inv The inverse of the matrix.
template <class ExecutionSpace, typename MatrixType1, typename MatrixType2>
void invert_matrix([[maybe_unused]] const ExecutionSpace &space, const MatrixType1 &matrix,
                   const MatrixType2 &matrix_inv) {
  static_assert(Kokkos::is_view<MatrixType1>::value && Kokkos::is_view<MatrixType2>::value,
                "The matrices must be a Kokkos::View");
  static_assert(std::is_same_v<typename MatrixType1::value_type, double> &&
                    std::is_same_v<typename MatrixType2::value_type, double>,
                "invert_matrix: The view must have 'double' as its value "
                "type");
  static_assert(MatrixType1::rank == 2 && MatrixType2::rank == 2,
                "invert_matrix: The view must have rank 2 (i.e., double**)");
  static_assert(std::is_same_v<typename MatrixType1::memory_space, typename MatrixType2::memory_space>,
                "invert_matrix: The matrices must have the same memory space");
  static_assert(std::is_same_v<typename MatrixType1::array_layout, typename MatrixType2::array_layout>,
                "invert_matrix: The matrices must have the same layout");

  // Check the input sizes
  const size_t matrix_size = matrix.extent(0);
  MUNDY_THROW_ASSERT(matrix.extent(1) == matrix_size, std::invalid_argument, "invert_matrix: matrix must be square.");
  MUNDY_THROW_ASSERT((matrix_inv.extent(0) == matrix_size) && (matrix_inv.extent(1) == matrix_size),
                     std::invalid_argument, "invert_matrix: matrix_inv must be the same size as the matrix to invert.");

  // Create a view to store the pivots
  Kokkos::View<int *, typename MatrixType1::array_layout, typename MatrixType1::memory_space> pivots("pivots",
                                                                                                     matrix_size);

  // Fill matrix_inv with the identity matrix
  Kokkos::deep_copy(matrix_inv, 0.0);
  Kokkos::parallel_for(
      "FillIdentity", Kokkos::RangePolicy<ExecutionSpace>(0, matrix_size),
      KOKKOS_LAMBDA(const size_t i) { matrix_inv(i, i) = 1.0; });

  // Solve the dense linear equation system M*X = I, which results in X = M^{-1}
  // On exist, M is replaced with its LU decomposition
  //           M_inv is replaced with the solution X = M^{-1}
  KokkosBlas::gesv(matrix, matrix_inv, pivots);
}

/// \brief Write a matrix to a human-readable text file
///
/// \param[in] filename The filename
/// \param[in] matrix_host The matrix to write (host)
template <typename MatrixType>
void write_matrix_to_file(const std::string &filename, const MatrixType &matrix_host) {
  static_assert(Kokkos::is_view<MatrixType>::value, "The matrix must be a Kokkos::View");
  static_assert(MatrixType::rank == 2, "write_matrix_to_file: The view must have rank 2 (i.e., double**)");
  static_assert(std::is_same_v<typename MatrixType::memory_space, Kokkos::HostSpace>,
                "write_matrix_to_file: The matrix must be in host memory");

  // Perform the write
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write the matrix to the file (space separated)
  // The first two lines are the number of rows and columns
  const size_t num_rows = matrix_host.extent(0);
  const size_t num_columns = matrix_host.extent(1);
  outfile << num_rows << std::endl;
  outfile << num_columns << std::endl;

  // Write matrix data with appropriate precision
  using ValueType = typename MatrixType::value_type;
  if constexpr (std::is_floating_point_v<ValueType>) {
    outfile << std::fixed << std::setprecision(std::numeric_limits<ValueType>::digits10 + 1);
  }
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      outfile << matrix_host(i, j) << " ";
    }
    outfile << std::endl;
  }

  // Close the file
  outfile.close();
}

/// \brief Read a matrix from a human-readable text file
///
/// \param[in] filename The filename
/// \param[out] matrix_host The matrix to read (host)
template <typename MatrixType>
void read_matrix_from_file(const std::string &filename, const size_t expected_num_rows,
                           const size_t expected_num_columns, const MatrixType &matrix_host) {
  static_assert(Kokkos::is_view<MatrixType>::value, "The matrix must be a Kokkos::View");
  static_assert(MatrixType::rank == 2, "read_matrix_from_file: The view must have rank 2 (i.e., double**)");
  static_assert(std::is_same_v<typename MatrixType::memory_space, Kokkos::HostSpace>,
                "read_matrix_from_file: The matrix must be in host memory");

  // Read the matrix from a file
  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  size_t num_rows;
  size_t num_columns;
  if (!(infile >> num_rows) || !(infile >> num_columns)) {
    std::cerr << "Error reading matrix dimensions from file: " + filename << std::endl;
    return;
  }

  if ((num_rows != expected_num_rows) || (num_columns != expected_num_columns)) {
    std::cerr << "Matrix size mismatch: expected (" << expected_num_rows << ", " << expected_num_columns << "), got ("
              << num_rows << ", " << num_columns << ")" << std::endl;
    return;
  }
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      if (!(infile >> matrix_host(i, j))) {
        std::cerr << "Failed to read matrix element at (" << i << ", " << j << ")" << std::endl;
        return;
      }
    }
  }

  // Close the file
  infile.close();
}

/// \brief Write a vector to a human readable text file
///
/// \param[in] filename The filename
/// \param[in] vector_host The vector to write (host)
template <typename VectorType>
void write_vector_to_file(const std::string &filename, const VectorType &vector_host) {
  static_assert(Kokkos::is_view<VectorType>::value, "The vector must be a Kokkos::View");
  static_assert(VectorType::rank == 1, "write_vector_to_file: The view must have rank 1 (i.e., double*)");
  static_assert(std::is_same_v<typename VectorType::memory_space, Kokkos::HostSpace>,
                "write_vector_to_file: The vector must be in host memory");

  // Perform the write
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write the vector to the file (space separated)
  // The first line is the number of elements
  const size_t num_elements = vector_host.extent(0);
  outfile << num_elements << std::endl;

  // Write vector data with appropriate precision
  using ValueType = typename VectorType::value_type;
  if constexpr (std::is_floating_point_v<ValueType>) {
    outfile << std::fixed << std::setprecision(std::numeric_limits<ValueType>::digits10 + 1);
  }

  for (size_t i = 0; i < num_elements; ++i) {
    outfile << vector_host(i) << std::endl;
  }

  // Close the file
  outfile.close();
}

/// \brief Read a vector from a file
///
/// \param[in] filename The filename
/// \param[out] vector_host The vector to read (host)
template <typename VectorType>
void read_vector_from_file(const std::string &filename, const size_t expected_num_elements,
                           const VectorType &vector_host) {
  static_assert(Kokkos::is_view<VectorType>::value, "The vector must be a Kokkos::View");
  static_assert(VectorType::rank == 1, "read_vector_from_file: The view must have rank 1 (i.e., double*)");
  static_assert(std::is_same_v<typename VectorType::memory_space, Kokkos::HostSpace>,
                "read_vector_from_file: The vector must be in host memory");

  // Read the vector from a file
  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  size_t num_elements;
  if (!(infile >> num_elements)) {
    std::cerr << "Error reading vector size from file: " + filename << std::endl;
    return;
  }

  if (num_elements != expected_num_elements) {
    std::cerr << "Vector size mismatch: expected " << expected_num_elements << ", got " << num_elements << std::endl;
    return;
  }

  for (size_t i = 0; i < num_elements; ++i) {
    if (!(infile >> vector_host(i))) {
      std::cerr << "Failed to read vector element at index " << i << std::endl;
      return;
    }
  }

  // Close the file
  infile.close();
}

// /// \brief Write a matrix to a file
// ///
// /// \param[in] filename The filename
// /// \param[in] matrix_host The matrix to write (host)
// template <typename ValueType, class Layout>
// void write_matrix_to_file(const std::string &filename,
//                           const Kokkos::View<ValueType **, Layout, Kokkos::HostSpace> &matrix_host) {
//   // Perform the write
//   std::ofstream outfile(filename, std::ios::binary);
//   if (!outfile) {
//     std::cerr << "Failed to open file: " << filename << std::endl;
//     return;
//   }

//   // Write the matrix to the file (using reinterpret_cast to map directly to binary data)
//   const size_t num_rows = matrix_host.extent(0);
//   const size_t num_columns = matrix_host.extent(1);
//   outfile.write(reinterpret_cast<const char *>(&num_rows), sizeof(size_t));
//   outfile.write(reinterpret_cast<const char *>(&num_columns), sizeof(size_t));
//   for (size_t i = 0; i < num_rows; ++i) {
//     for (size_t j = 0; j < num_columns; ++j) {
//       outfile.write(reinterpret_cast<const char *>(&matrix_host(i, j)), sizeof(ValueType));
//     }
//   }

//   // Close the file
//   outfile.close();
// }

// /// \brief Read a matrix from a file
// ///
// /// \param[in] filename The filename
// /// \param[out] matrix_host The matrix to read (host)
// template <typename ValueType, class Layout>
// void read_matrix_from_file(const std::string &filename, const size_t expected_num_rows,
//                            const size_t expected_num_columns,
//                            const Kokkos::View<ValueType **, Layout, Kokkos::HostSpace> &matrix_host) {
//   // Read the matrix from a file
//   std::ifstream infile(filename, std::ios::binary);
//   if (!infile) {
//     std::cerr << "Failed to open file: " << filename << std::endl;
//     return;
//   }

//   // Parse the input
//   size_t num_rows;
//   size_t num_columns;
//   infile.read(reinterpret_cast<char *>(&num_rows), sizeof(size_t));
//   infile.read(reinterpret_cast<char *>(&num_columns), sizeof(size_t));
//   if ((num_rows != expected_num_rows) || (num_columns != expected_num_columns)) {
//     std::cerr << "Matrix size mismatch: expected (" << expected_num_rows << ", " << expected_num_columns << "), got
//     ("
//               << num_rows << ", " << num_columns << ")" << std::endl;
//     return;
//   }
//   for (size_t i = 0; i < num_rows; ++i) {
//     for (size_t j = 0; j < num_columns; ++j) {
//       infile.read(reinterpret_cast<char *>(&matrix_host(i, j)), sizeof(ValueType));
//     }
//   }
// }

// /// \brief Write a vector to a file
// ///
// /// \param[in] filename The filename
// /// \param[in] vector_host The vector to write (host)
// template <typename VectorDataType, class Layout>
// void write_vector_to_file(const std::string &filename,
//                           const Kokkos::View<VectorDataType *, Layout, Kokkos::HostSpace> &vector_host) {
//   // Perform the write
//   std::ofstream outfile(filename, std::ios::binary);
//   if (!outfile) {
//     std::cerr << "Failed to open file: " << filename << std::endl;
//     return;
//   }

//   // Write the vector to the file (using reinterpret_cast to map directly to binary data)
//   const size_t num_elements = vector_host.extent(0);
//   outfile.write(reinterpret_cast<const char *>(&num_elements), sizeof(size_t));
//   for (size_t i = 0; i < num_elements; ++i) {
//     outfile.write(reinterpret_cast<const char *>(&vector_host(i)), sizeof(VectorDataType));
//   }

//   // Close the file
//   outfile.close();
// }

// /// \brief Read a vector from a file
// ///
// /// \param[in] filename The filename
// /// \param[out] vector_host The vector to read (host)
// template <typename VectorDataType, class Layout>
// void read_vector_from_file(const std::string &filename, const size_t expected_num_elements,
//                            const Kokkos::View<VectorDataType *, Layout, Kokkos::HostSpace> &vector_host) {
//   // Read the vector from a file
//   std::ifstream infile(filename, std::ios::binary);
//   if (!infile) {
//     std::cerr << "Failed to open file: " << filename << std::endl;
//     return;
//   }

//   // Parse the input
//   size_t num_elements;
//   infile.read(reinterpret_cast<char *>(&num_elements), sizeof(size_t));
//   if (num_elements != expected_num_elements) {
//     std::cerr << "Vector size mismatch: expected " << expected_num_elements << ", got " << num_elements << std::endl;
//     return;
//   }
//   for (size_t i = 0; i < num_elements; ++i) {
//     infile.read(reinterpret_cast<char *>(&vector_host(i)), sizeof(VectorDataType));
//   }

//   // Close the file
//   infile.close();
// }

template <class Space>
struct VelocityReducer {
 public:
  // Required
  typedef VelocityReducer reducer;
  typedef mundy::math::Vector3d value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

 private:
  value_type &value;

 public:
  KOKKOS_INLINE_FUNCTION
  VelocityReducer(value_type &value_) : value(value_) {
  }

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const {
    val.set(0.0, 0.0, 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return result_view_type(&value, 1);
  }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const {
    return true;
  }
};

template <typename Func>
struct VelocityKernelThreadReductionFunctor {
  KOKKOS_INLINE_FUNCTION
  VelocityKernelThreadReductionFunctor(const Func &compute_velocity_contribution, const int t)
      : compute_velocity_contribution_(compute_velocity_contribution), t_(t) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int s, mundy::math::Vector3d &v_accum) const {
    // Call the custom operation to compute the contribution
    compute_velocity_contribution_(t_, s, v_accum[0], v_accum[1], v_accum[2]);
  }

  const Func compute_velocity_contribution_;
  const int t_;
};

template <int panel_size, typename ExecutionSpace, typename Func>
struct VelocityKernelTeamFunctor {
  using TeamMemberType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

  KOKKOS_INLINE_FUNCTION
  VelocityKernelTeamFunctor(const TeamMemberType &team_member, const Func &compute_velocity_contribution,
                            const int panel_start, const int num_source_points,
                            Kokkos::Array<double, panel_size> &local_vx, Kokkos::Array<double, panel_size> &local_vy,
                            Kokkos::Array<double, panel_size> &local_vz)
      : team_member_(team_member),
        compute_velocity_contribution_(compute_velocity_contribution),
        panel_start_(panel_start),
        num_source_points_(num_source_points),
        local_vx_(local_vx),
        local_vy_(local_vy),
        local_vz_(local_vz) {
  }

  KOKKOS_FUNCTION
  void operator()(const int t) const {
    mundy::math::Vector3d v_sum = {0.0, 0.0, 0.0};

    // Loop over all source points
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member_, num_source_points_),
                            VelocityKernelThreadReductionFunctor<Func>(compute_velocity_contribution_, t),
                            VelocityReducer<ExecutionSpace>(v_sum));

    // Store the results in the local arrays
    local_vx_[t - panel_start_] = v_sum[0];
    local_vy_[t - panel_start_] = v_sum[1];
    local_vz_[t - panel_start_] = v_sum[2];
  }

  const TeamMemberType &team_member_;
  const Func compute_velocity_contribution_;
  const int panel_start_;
  const int num_source_points_;
  Kokkos::Array<double, panel_size> &local_vx_;
  Kokkos::Array<double, panel_size> &local_vy_;
  Kokkos::Array<double, panel_size> &local_vz_;
};

template <int panel_size, typename VectorType>
struct VelocityKernelTeamTeamAccumulator {
  KOKKOS_INLINE_FUNCTION
  VelocityKernelTeamTeamAccumulator(const VectorType &target_velocities,                //
                                    const Kokkos::Array<double, panel_size> &local_vx,  //
                                    const Kokkos::Array<double, panel_size> &local_vy,  //
                                    const Kokkos::Array<double, panel_size> &local_vz,  //
                                    const int panel_start,                              //
                                    const int panel_end)
      : target_velocities_(target_velocities),
        local_vx_(local_vx),
        local_vy_(local_vy),
        local_vz_(local_vz),
        panel_start_(panel_start),
        panel_end_(panel_end) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()() const {
    for (int t = panel_start_; t < panel_end_; ++t) {
      Kokkos::atomic_add(&target_velocities_(3 * t + 0), local_vx_[t - panel_start_]);
      Kokkos::atomic_add(&target_velocities_(3 * t + 1), local_vy_[t - panel_start_]);
      Kokkos::atomic_add(&target_velocities_(3 * t + 2), local_vz_[t - panel_start_]);
    }
  }

  static_assert(Kokkos::is_view<VectorType>::value,
                "VelocityKernelTeamTeamAccumulator: target_velocities must be a "
                "Kokkos::View.");
  static_assert(VectorType::rank == 1, "VelocityKernelTeamTeamAccumulator: target_velocities must be rank 1.");
  static_assert(std::is_same_v<typename VectorType::value_type, double>,
                "VelocityKernelTeamTeamAccumulator: target_velocities must have double as its value type.");

  const VectorType target_velocities_;
  const Kokkos::Array<double, panel_size> &local_vx_;
  const Kokkos::Array<double, panel_size> &local_vy_;
  const Kokkos::Array<double, panel_size> &local_vz_;
  const int panel_start_;
  const int panel_end_;
};

template <int panel_size, class ExecutionSpace, typename VectorType, typename Func>
void panelize_velocity_kernel_over_target_points([[maybe_unused]] const ExecutionSpace &space,  //
                                                 int num_target_points,                         //
                                                 int num_source_points,                         //
                                                 const VectorType target_velocities,            //
                                                 const Func &compute_velocity_contribution) {
  static_assert(Kokkos::is_view<VectorType>::value,
                "panelize_velocity_kernel_over_target_points: target_velocities must be a "
                "Kokkos::View.");
  static_assert(VectorType::rank == 1,
                "panelize_velocity_kernel_over_target_points: target_velocities must be rank 1.");
  static_assert(std::is_same_v<typename VectorType::value_type, double>,
                "panelize_velocity_kernel_over_target_points: target_velocities must have double as its value type.");

  int num_panels = (num_target_points + panel_size - 1) / panel_size;

  // Define the team policy with the number of panels
  using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
  Kokkos::parallel_for(
      team_policy(num_panels, Kokkos::AUTO), KOKKOS_LAMBDA(const team_policy::member_type &team_member) {
        const int panel_start = team_member.league_rank() * panel_size;
        const int panel_end =
            (panel_start + panel_size) > num_target_points ? num_target_points : (panel_start + panel_size);

        // Local accumulation arrays for each target point in the panel
        Kokkos::Array<double, panel_size> local_vx = {0.0};
        Kokkos::Array<double, panel_size> local_vy = {0.0};
        Kokkos::Array<double, panel_size> local_vz = {0.0};

        // Loop over each target point in the panel
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, panel_start, panel_end),
                             VelocityKernelTeamFunctor<panel_size, ExecutionSpace, Func>(
                                 team_member, compute_velocity_contribution, panel_start, num_source_points, local_vx,
                                 local_vy, local_vz));

        // After processing, update the global output using a single thread per team
        Kokkos::single(Kokkos::PerTeam(team_member),
                       VelocityKernelTeamTeamAccumulator<panel_size, VectorType>(target_velocities, local_vx, local_vy,
                                                                                 local_vz, panel_start, panel_end));
      });
}

/// \brief Apply the stokes kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceForceVectorType, typename TargetVelocityVectorType>
void apply_stokes_kernel([[maybe_unused]] const ExecutionSpace &space,  //
                         const double viscosity,                        //
                         const SourcePosVectorType &source_positions,   //
                         const TargetPosVectorType &target_positions,   //
                         const SourceForceVectorType &source_forces,    //
                         const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_stokes_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "apply_stokes_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_stokes_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_stokes_kernel: The input vectors must have the same memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_kernel: target_velocities must have size 3 * num_target_points.");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);

  auto stokes_computation =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;

    const double f_dot_r = fx * dx + fy * dy + fz * dz;
    const double scale_factor_rinv3 = scale_factor * rinv3;

    // Accumulate velocity contribution to local variables
    vx_accum += scale_factor_rinv3 * (r2 * fx + dx * f_dot_r);
    vy_accum += scale_factor_rinv3 * (r2 * fy + dy * f_dot_r);
    vz_accum += scale_factor_rinv3 * (r2 * fz + dz * f_dot_r);
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  stokes_computation);
}

/// \brief Apply the stokes kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceForceVectorType, typename SourceWeightVectorType, typename TargetVelocityVectorType>
void apply_weighted_stokes_kernel([[maybe_unused]] const ExecutionSpace &space,  //
                                  const double viscosity,                        //
                                  const SourcePosVectorType &source_positions,   //
                                  const TargetPosVectorType &target_positions,   //
                                  const SourceForceVectorType &source_forces,    //
                                  const SourceWeightVectorType &source_weights,  //
                                  const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<SourceWeightVectorType>::value &&
                    Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_stokes_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    SourceWeightVectorType::rank == 1 && TargetVelocityVectorType::rank == 1,
                "apply_stokes_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename SourceWeightVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_stokes_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceWeightVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_stokes_kernel: The input vectors must have the same memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_kernel: target_velocities must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_weights.extent(0) == num_source_points, std::invalid_argument,
                     "apply_stokes_kernel: source_weights must have size num_source_points.");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);
  auto weighted_stokes_computation =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0) * source_weights(s);
    const double fy = source_forces(3 * s + 1) * source_weights(s);
    const double fz = source_forces(3 * s + 2) * source_weights(s);

    // const double r2 = dx * dx + dy * dy + dz * dz;
    // const double rinv = r2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(r2);
    // const double rinv3 = rinv * rinv * rinv;

    // const double inner_prod = fx * dx + fy * dy + fz * dz;
    // const double scale_factor_rinv3 = scale_factor * rinv3;

    // // Accumulate velocity contribution to local variables
    // vx_accum += scale_factor_rinv3 * (r2 * fx + dx * inner_prod);
    // vy_accum += scale_factor_rinv3 * (r2 * fy + dy * inner_prod);
    // vz_accum += scale_factor_rinv3 * (r2 * fz + dz * inner_prod);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(r2);
    const double rinv2 = rinv * rinv;

    const double f_dot_r_rinv2 = (fx * dx + fy * dy + fz * dz) * rinv2;
    const double scale_factor_rinv = scale_factor * rinv;

    // Accumulate velocity contribution to local variables
    vx_accum += scale_factor_rinv * (fx + dx * f_dot_r_rinv2);
    vy_accum += scale_factor_rinv * (fy + dy * f_dot_r_rinv2);
    vz_accum += scale_factor_rinv * (fz + dz * f_dot_r_rinv2);
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  weighted_stokes_computation);
}

/// \brief Apply the RPY kernel to map source forces to target velocities: u_target += M f_source
///
/// Note, this does not include self-interaction. If that is desired simply add 1/(6 pi mu) * f to u
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceRadiusVectorType, typename TargetRadiusVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_rpy_kernel([[maybe_unused]] const ExecutionSpace &space,  //
                      const double viscosity,                        //
                      const SourcePosVectorType &source_positions,   //
                      const TargetPosVectorType &target_positions,   //
                      const SourceRadiusVectorType &source_radii,    //
                      const TargetRadiusVectorType &target_radii,    //
                      const SourceForceVectorType &source_forces,    //
                      const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceRadiusVectorType>::value && Kokkos::is_view<TargetRadiusVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_rpy_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceRadiusVectorType::rank == 1 &&
                    TargetRadiusVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "apply_rpy_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceRadiusVectorType::value_type, double> &&
                    std::is_same_v<typename TargetRadiusVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_rpy_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceRadiusVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetRadiusVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_rpy_kernel: The input vectors must have the same memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_rpy_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_rpy_kernel: target_velocities must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_radii.extent(0) == num_source_points, std::invalid_argument,
                     "apply_rpy_kernel: source_radii must have size num_source_points.");
  MUNDY_THROW_ASSERT(target_radii.extent(0) == num_target_points, std::invalid_argument,
                     "apply_rpy_kernel: target_radii must have size num_target_points.");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);
  auto rpy_computation =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);
    const double a = source_radii(s);

    constexpr double one_over_three = 1.0 / 3.0;
    constexpr double one_over_six = 1.0 / 6.0;

    const double a2_over_three = one_over_three * a * a;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;
    const double rinv5 = rinv * rinv * rinv3;
    const double fdotr = fx * dx + fy * dy + fz * dz;

    const double three_fdotr_rinv5 = 3.0 * fdotr * rinv5;
    const double cx = fx * rinv3 - three_fdotr_rinv5 * dx;
    const double cy = fy * rinv3 - three_fdotr_rinv5 * dy;
    const double cz = fz * rinv3 - three_fdotr_rinv5 * dz;

    const double fdotr_rinv3 = fdotr * rinv3;

    // Velocity
    const double v0 = scale_factor * (fx * rinv + dx * fdotr_rinv3 + a2_over_three * cx);
    const double v1 = scale_factor * (fy * rinv + dy * fdotr_rinv3 + a2_over_three * cy);
    const double v2 = scale_factor * (fz * rinv + dz * fdotr_rinv3 + a2_over_three * cz);

    // Laplacian
    const double lap0 = 2.0 * scale_factor * cx;
    const double lap1 = 2.0 * scale_factor * cy;
    const double lap2 = 2.0 * scale_factor * cz;

    // Apply the result
    const double lap_coeff = one_over_six * target_radii(t) * target_radii(t);
    vx_accum += v0 + lap_coeff * lap0;
    vy_accum += v1 + lap_coeff * lap1;
    vz_accum += v2 + lap_coeff * lap2;
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  rpy_computation);
}

/// \brief Apply the corrected RPY kernel to map source forces to target velocities: u_target += M f_source
///
/// Note, this does not include self-interaction. If that is desired simply add 1/(6 pi mu) * f to u
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceRadiusVectorType, typename TargetRadiusVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_rpyc_kernel([[maybe_unused]] const ExecutionSpace &space,  //
                       const double viscosity,                        //
                       const SourcePosVectorType &source_positions,   //
                       const TargetPosVectorType &target_positions,   //
                       const SourceRadiusVectorType &source_radii,    //
                       const TargetRadiusVectorType &target_radii,    //
                       const SourceForceVectorType &source_forces,    //
                       const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceRadiusVectorType>::value && Kokkos::is_view<TargetRadiusVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_rpyc_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceRadiusVectorType::rank == 1 &&
                    TargetRadiusVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "apply_rpyc_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceRadiusVectorType::value_type, double> &&
                    std::is_same_v<typename TargetRadiusVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_rpyc_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceRadiusVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetRadiusVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_rpyc_kernel: The input vectors must have the same memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_rpyc_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_rpyc_kernel: target_velocities must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_radii.extent(0) == num_source_points, std::invalid_argument,
                     "apply_rpyc_kernel: source_radii must have size num_source_points.");
  MUNDY_THROW_ASSERT(target_radii.extent(0) == num_target_points, std::invalid_argument,
                     "apply_rpyc_kernel: target_radii must have size num_target_points.");

  // Launch the parallel kernel
  constexpr double one_over_eight = 1.0 / 8.0;
  constexpr double one_over_six = 1.0 / 6.0;
  constexpr double one_over_three = 1.0 / 3.0;
  constexpr double one_over_32 = 1.0 / 32.0;
  constexpr double inv_pi = 1.0 / Kokkos::numbers::pi_v<double>;
  const double inv_viscosity = 1.0 / viscosity;
  auto rpyc_computation =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);
    const double a = source_radii(s);
    const double b = target_radii(t);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double r = Kokkos::sqrt(r2);
    const double r3 = r * r2;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / r;
    const double rinv2 = rinv * rinv;
    const double rinv3 = rinv2 * rinv;

    const double dx_hat = dx * rinv;
    const double dy_hat = dy * rinv;
    const double dz_hat = dz * rinv;

    const double f_dot_rhat = fx * dx_hat + fy * dy_hat + fz * dz_hat;

    if (a + b < r) {
      // If a + b < r, regular RPY
      // M = coeff * (tmp1 * I + tmp2 * r_hat outer r_hat)
      //   coeff = 1 / (8 pi mu r_norm)
      //   tmp1 = 1 + (a**2 + b**2) / (3 * r_norm**2)
      //   tmp2 = 1 - (a**2 + b**2) / (r_norm**2)
      //   r_hat = r / r_norm
      const double a2 = a * a;
      const double b2 = b * b;
      const double a2_plus_b2_rinv2 = (a2 + b2) * rinv2;
      const double scale_factor = one_over_eight * inv_pi * inv_viscosity * rinv;
      const double tmp1_scaled = scale_factor * (1. + a2_plus_b2_rinv2 * one_over_three);
      const double tmp2_scaled = scale_factor * (1. - a2_plus_b2_rinv2);
      vx_accum += tmp1_scaled * fx + tmp2_scaled * (f_dot_rhat * dx_hat);
      vy_accum += tmp1_scaled * fy + tmp2_scaled * (f_dot_rhat * dy_hat);
      vz_accum += tmp1_scaled * fz + tmp2_scaled * (f_dot_rhat * dz_hat);
    } else if (Kokkos::abs(a - b) < r && a > DOUBLE_ZERO && b > DOUBLE_ZERO) {
      // If neither radius is zero and if abs(a - b) < r < a + b, corrected RPY
      // M = 1/(6 pi mu a b) * (tmp1 I + tmp2 r_hat outer r_hat) f
      //  tmp1 = (16 r^3 (a + b) - ((a - b)^2 + 3 r^2)^2) / (32 r^3)
      //  tmp2 = 3 ((a - b)^2 - r^2)^2 / (32 r^3)
      //  r_hat = r / r_norm
      const double a_plus_b = a + b;
      const double a_minus_b = a - b;
      const double a_minus_b2 = a_minus_b * a_minus_b;
      const double tmp3 = a_minus_b2 + 3 * r2;
      const double tmp4 = a_minus_b2 - r2;

      const double scale_factor = one_over_six * inv_pi * inv_viscosity / (a * b);
      const double tmp1_scaled = scale_factor * (16.0 * r3 * a_plus_b - tmp3 * tmp3) * one_over_32 * rinv3;
      const double tmp2_scaled = scale_factor * 3.0 * tmp4 * tmp4 * one_over_32 * rinv3;

      vx_accum += tmp1_scaled * fx + tmp2_scaled * (f_dot_rhat * dx_hat);
      vy_accum += tmp1_scaled * fy + tmp2_scaled * (f_dot_rhat * dy_hat);
      vz_accum += tmp1_scaled * fz + tmp2_scaled * (f_dot_rhat * dz_hat);
    } else {
      //  if r < abs(a - b), Local drag
      // v = 1 / (6 pi mu max(a, b)) * f
      if (r2 < DOUBLE_ZERO) {
        // Skip self interaction
        return;
      }

      const double max_a_b = Kokkos::max(a, b);
      const double scale_factor = one_over_six * inv_pi * inv_viscosity / max_a_b;
      vx_accum += scale_factor * fx;
      vy_accum += scale_factor * fy;
      vz_accum += scale_factor * fz;
    }
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  rpyc_computation);
}

/// \brief Apply the stokes double layer kernel with singularity subtraction) to map source forces to target velocities:
/// u_target += M (f_source + f_target)
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] source_forces The vector to apply the self-interaction matrix to (size num_nodes * 3)
/// \param[out] target_velocities The result of applying the self-interaction matrix to f (size num_nodes * 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_stokes_double_layer_kernel_ss([[maybe_unused]] const ExecutionSpace &space,          //
                                         const double viscosity,                                //
                                         const size_t num_source_points,                        //
                                         const size_t num_target_points,                        //
                                         const SourcePosVectorType &source_positions,           //
                                         const TargetPosVectorType &target_positions,           //
                                         const SourceNormalVectorType &source_normals,          //
                                         const QuadratureWeightVectorType &quadrature_weights,  //
                                         const SourceForceVectorType &source_forces,            //
                                         const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_stokes_double_layer_kernel_ss: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceNormalVectorType::rank == 1 &&
                    QuadratureWeightVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "apply_stokes_double_layer_kernel_ss: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_stokes_double_layer_kernel_ss: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceNormalVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space,
                         typename QuadratureWeightVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_stokes_double_layer_kernel_ss: The input vectors must have the same memory space.");

  MUNDY_THROW_ASSERT(source_positions.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: source_positions must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_positions.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: target_positions must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_normals.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: source_normals must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: quadrature_weights must have size num_source_points.");
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel_ss: target_velocities must have size 3 * num_target_points.");

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto stokes_double_layer_computation =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Skip self-interaction
    if (t == s) {
      return;
    }

    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // Compute the double layer potential
    const double sxx =
        source_normals(3 * s + 0) * (source_forces(3 * s + 0) + source_forces(3 * t + 0)) * quadrature_weights(s);
    const double sxy =
        source_normals(3 * s + 0) * (source_forces(3 * s + 1) + source_forces(3 * t + 1)) * quadrature_weights(s);
    const double sxz =
        source_normals(3 * s + 0) * (source_forces(3 * s + 2) + source_forces(3 * t + 2)) * quadrature_weights(s);
    const double syx =
        source_normals(3 * s + 1) * (source_forces(3 * s + 0) + source_forces(3 * t + 0)) * quadrature_weights(s);
    const double syy =
        source_normals(3 * s + 1) * (source_forces(3 * s + 1) + source_forces(3 * t + 1)) * quadrature_weights(s);
    const double syz =
        source_normals(3 * s + 1) * (source_forces(3 * s + 2) + source_forces(3 * t + 2)) * quadrature_weights(s);
    const double szx =
        source_normals(3 * s + 2) * (source_forces(3 * s + 0) + source_forces(3 * t + 0)) * quadrature_weights(s);
    const double szy =
        source_normals(3 * s + 2) * (source_forces(3 * s + 1) + source_forces(3 * t + 1)) * quadrature_weights(s);
    const double szz =
        source_normals(3 * s + 2) * (source_forces(3 * s + 2) + source_forces(3 * t + 2)) * quadrature_weights(s);

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    vx_accum += dx * coeff;
    vy_accum += dy * coeff;
    vz_accum += dz * coeff;
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  stokes_double_layer_computation);
}

/// \brief Apply the stokes double layer kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] source_forces The vector to apply the self-interaction matrix to (size num_nodes * 3)
/// \param[out] target_velocities The result of applying the self-interaction matrix to f (size num_nodes * 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_stokes_double_layer_kernel([[maybe_unused]] const ExecutionSpace &space,          //
                                      const double viscosity,                                //
                                      const size_t num_source_points,                        //
                                      const size_t num_target_points,                        //
                                      const SourcePosVectorType &source_positions,           //
                                      const TargetPosVectorType &target_positions,           //
                                      const SourceNormalVectorType &source_normals,          //
                                      const QuadratureWeightVectorType &quadrature_weights,  //
                                      const SourceForceVectorType &source_forces,            //
                                      const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_stokes_double_layer_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceNormalVectorType::rank == 1 &&
                    QuadratureWeightVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "apply_stokes_double_layer_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_stokes_double_layer_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceNormalVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space,
                         typename QuadratureWeightVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_stokes_double_layer_kernel: The input vectors must have the same memory space.");

  MUNDY_THROW_ASSERT(source_positions.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: source_positions must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_positions.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: target_positions must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_normals.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: source_normals must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: quadrature_weights must have size num_source_points.");
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_stokes_double_layer_kernel: target_velocities must have size 3 * num_target_points.");

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto stokes_double_layer_contribution =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Skip self-interaction
    if (t == s) {
      return;
    }

    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // Compute the double layer potential
    const double sxx = source_normals(3 * s + 0) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double sxy = source_normals(3 * s + 0) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double sxz = source_normals(3 * s + 0) * source_forces(3 * s + 2) * quadrature_weights(s);
    const double syx = source_normals(3 * s + 1) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double syy = source_normals(3 * s + 1) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double syz = source_normals(3 * s + 1) * source_forces(3 * s + 2) * quadrature_weights(s);
    const double szx = source_normals(3 * s + 2) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double szy = source_normals(3 * s + 2) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double szz = source_normals(3 * s + 2) * source_forces(3 * s + 2) * quadrature_weights(s);

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    vx_accum += dx * coeff;
    vy_accum += dy * coeff;
    vz_accum += dz * coeff;
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  stokes_double_layer_contribution);
}

/// \brief Apply local drag to the sphere velocities v += 1/(6 pi mu r) f
template <class ExecutionSpace, typename SphereVelocityVectorType, typename SphereForceVectorType,
          typename SphereRadiusVectorType>
void apply_local_drag([[maybe_unused]] const ExecutionSpace &space,       //
                      const double viscosity,                             //
                      const SphereVelocityVectorType &sphere_velocities,  //
                      const SphereForceVectorType &sphere_forces,         //
                      const SphereRadiusVectorType &sphere_radii) {
  static_assert(Kokkos::is_view<SphereVelocityVectorType>::value && Kokkos::is_view<SphereForceVectorType>::value &&
                    Kokkos::is_view<SphereRadiusVectorType>::value,
                "apply_local_drag: The input vectors must be a Kokkos::View.");
  static_assert(
      SphereVelocityVectorType::rank == 1 && SphereForceVectorType::rank == 1 && SphereRadiusVectorType::rank == 1,
      "apply_local_drag: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SphereVelocityVectorType::value_type, double> &&
                    std::is_same_v<typename SphereForceVectorType::value_type, double> &&
                    std::is_same_v<typename SphereRadiusVectorType::value_type, double>,
                "apply_local_drag: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SphereVelocityVectorType::memory_space, typename SphereForceVectorType::memory_space> &&
          std::is_same_v<typename SphereVelocityVectorType::memory_space,
                         typename SphereRadiusVectorType::memory_space>,
      "apply_local_drag: The input vectors must have the same memory space.");

  MUNDY_THROW_ASSERT(sphere_velocities.extent(0) == 3 * sphere_radii.extent(0), std::invalid_argument,
                     "apply_local_drag: sphere_velocities must have size 3 * num_spheres.");
  MUNDY_THROW_ASSERT(sphere_forces.extent(0) == 3 * sphere_radii.extent(0), std::invalid_argument,
                     "apply_local_drag: sphere_forces must have size 3 * num_spheres.");

  const size_t num_spheres = sphere_radii.extent(0);
  const double scale = 1.0 / (6.0 * M_PI * viscosity);
  Kokkos::parallel_for(
      "apply_local_drag", Kokkos::RangePolicy<ExecutionSpace>(0, num_spheres), KOKKOS_LAMBDA(const size_t i) {
        const double r = sphere_radii(i);
        const double inv_drag_coeff = scale / r;
        sphere_velocities(3 * i) += inv_drag_coeff * sphere_forces(3 * i);
        sphere_velocities(3 * i + 1) += inv_drag_coeff * sphere_forces(3 * i + 1);
        sphere_velocities(3 * i + 2) += inv_drag_coeff * sphere_forces(3 * i + 2);
      });
}

/// \brief Fill the stokes double layer matrix times the surface normal
///
///   T is the stokes double layer kernel T_{ij} = -3 viscosity / (4*pi) * r_i * r_j * r_k * normal_k / r**5
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename MatrixType>
void fill_stokes_double_layer_matrix([[maybe_unused]] const ExecutionSpace &space,          //
                                     const double viscosity,                                //
                                     const size_t num_source_points,                        //
                                     const size_t num_target_points,                        //
                                     const SourcePosVectorType &source_positions,           //
                                     const TargetPosVectorType &target_positions,           //
                                     const SourceNormalVectorType &source_normals,          //
                                     const QuadratureWeightVectorType &quadrature_weights,  //
                                     const MatrixType &T) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value && Kokkos::is_view<MatrixType>::value,
                "fill_stokes_double_layer_matrix: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceNormalVectorType::rank == 1 &&
                    QuadratureWeightVectorType::rank == 1 && MatrixType::rank == 2,
                "fill_stokes_double_layer_matrix: The input vectors must be rank 1 and the matrix must be rank 2.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename MatrixType::value_type, double>,
                "fill_stokes_double_layer_matrix: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename MatrixType::memory_space, typename SourcePosVectorType::memory_space> &&
          std::is_same_v<typename MatrixType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename MatrixType::memory_space, typename SourceNormalVectorType::memory_space> &&
          std::is_same_v<typename MatrixType::memory_space, typename QuadratureWeightVectorType::memory_space>,
      "fill_stokes_double_layer_matrix: The input vectors must have the same memory space.");

  MUNDY_THROW_ASSERT(source_positions.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: source_positions must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_positions.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: target_positions must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_normals.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: source_normals must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: quadrature_weights must have size num_source_points.");
  MUNDY_THROW_ASSERT(T.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: T must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(T.extent(1) == 3 * num_source_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: T must have size 3 * num_source_points.");

  // Compute the scale factor
  const double scale_factor = -3.0 / (4.0 * M_PI * viscosity);
  Kokkos::parallel_for(
      "DoubleLayerMatrixFill", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Compute the distance vector
        const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
        const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
        const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

        // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
        const double dr2 = dx * dx + dy * dy + dz * dz;
        const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(dr2);
        const double rinv2 = rinv * rinv;
        const double rinv5 = rinv * rinv2 * rinv2;

        // Read in the surface normal at the source
        const double quadrature_weight = quadrature_weights(s);
        const double scaled_normal_s0 = source_normals(3 * s + 0) * quadrature_weight;
        const double scaled_normal_s1 = source_normals(3 * s + 1) * quadrature_weight;
        const double scaled_normal_s2 = source_normals(3 * s + 2) * quadrature_weight;

        // tmp_vec = (r dot normal) r, scaled by -3 / (4 pi r^5 viscosity)
        const double r_dot_scaled_normal = dx * scaled_normal_s0 + dy * scaled_normal_s1 + dz * scaled_normal_s2;
        const double tmp_vec0 = scale_factor * rinv5 * r_dot_scaled_normal * dx;
        const double tmp_vec1 = scale_factor * rinv5 * r_dot_scaled_normal * dy;
        const double tmp_vec2 = scale_factor * rinv5 * r_dot_scaled_normal * dz;

        // T (local) = r outer tmp_vec = r outer r (r dot normal) * -3 / (4 pi r^5 viscosity)
        // clang-format off
        T(t * 3 + 0, s * 3 + 0) = dx * tmp_vec0; T(t * 3 + 0, s * 3 + 1) = dx * tmp_vec1; T(t * 3 + 0, s * 3 + 2) = dx * tmp_vec2;
        T(t * 3 + 1, s * 3 + 0) = dy * tmp_vec0; T(t * 3 + 1, s * 3 + 1) = dy * tmp_vec1; T(t * 3 + 1, s * 3 + 2) = dy * tmp_vec2;
        T(t * 3 + 2, s * 3 + 0) = dz * tmp_vec0; T(t * 3 + 2, s * 3 + 1) = dz * tmp_vec1; T(t * 3 + 2, s * 3 + 2) = dz * tmp_vec2;
        // clang-format on
      });
}

/// \brief Fill the stokes double layer matrix times the surface normal with singularity subtraction
///
///   T is the stokes double layer kernel T_{ij} = -3 viscosity / (4*pi) * r_i * r_j * r_k * normal_k / r**5
///
/// \param space The execution space
/// \param[in] T The stokes double layer times weighted normal matrix to apply singularity subtraction to (size
/// num_target_points * 3 x num_source_points * 3)
template <class ExecutionSpace, typename MatrixType>
void add_singularity_subtraction([[maybe_unused]] const ExecutionSpace &space, const MatrixType &T) {
  static_assert(Kokkos::is_view<MatrixType>::value,
                "add_singularity_subtraction: The input matrix must be a Kokkos::View.");
  static_assert(MatrixType::rank == 2, "add_singularity_subtraction: The input matrix must be rank 2.");
  static_assert(std::is_same_v<typename MatrixType::value_type, double>,
                "add_singularity_subtraction: The input matrix must have 'double' as its value type.");

  const size_t num_source_points = T.extent(1) / 3;
  const size_t num_target_points = T.extent(0) / 3;

  // Add the singularity subtraction to T
  // Create a vector that has 1 in the x-component and 0 in the y and z components for each source point
  using Layout = typename MatrixType::array_layout;
  using MemorySpace = typename MatrixType::memory_space;
  Kokkos::View<double *, Layout, MemorySpace> e1("e1", 3 * num_source_points);
  Kokkos::View<double *, Layout, MemorySpace> e2("e2", 3 * num_source_points);
  Kokkos::View<double *, Layout, MemorySpace> e3("e3", 3 * num_source_points);
  Kokkos::parallel_for(
      "E1/2/3", Kokkos::RangePolicy<ExecutionSpace>(0, num_source_points), KOKKOS_LAMBDA(const size_t s) {
        e1(3 * s + 0) = 1.0;
        e1(3 * s + 1) = 0.0;
        e1(3 * s + 2) = 0.0;

        e2(3 * s + 0) = 0.0;
        e2(3 * s + 1) = 1.0;
        e2(3 * s + 2) = 0.0;

        e3(3 * s + 0) = 0.0;
        e3(3 * s + 1) = 0.0;
        e3(3 * s + 2) = 1.0;
      });

  // Compute w1 = T * e1, w2 = T * e2, w3 = T * e3
  Kokkos::View<double *, Layout, MemorySpace> w1("w1", 3 * num_target_points);
  Kokkos::View<double *, Layout, MemorySpace> w2("w2", 3 * num_target_points);
  Kokkos::View<double *, Layout, MemorySpace> w3("w3", 3 * num_target_points);
  KokkosBlas::gemv("N", 1.0, T, e1, 0.0, w1);
  KokkosBlas::gemv("N", 1.0, T, e2, 0.0, w2);
  KokkosBlas::gemv("N", 1.0, T, e3, 0.0, w3);

  // Apply singularity subtraction.
  // sum_s T_{s,t} [q_s + q_t]
  //    = sum_s T_{s,t} q_s + sum_s T_{s,t} q_t
  //    = sum_s T_{s,t} q_s + q1_t sum_s T_{s,t} e1_t + q2_t sum_s T_{s,t} e2_t + q3_t sum_s T_{s,t} e3_t
  //    = sum_s T_{s,t} q_s + q1_t w1_t + q2_t w2_t + q3_t w3_t
  //
  // Note, T [q(y) + q(x)](x) is a vector of size 3.
  // w1, w2, w3 are vectors of size 3 * num_target_points
  // sum_s T_{s,t} e1_t = T[e1](x) = w1_t where e1 is a vector of size 3 * num_source_points with 1 in the x-component
  // and 0 in the y and z components of each source point.
  //
  // q1_t w1_t + q2_t w2_t - q3_t w3_t is a vector of size 3 and equal [w1_t w2_t w3_t] [q1_t q2_t q3_t]^T,
  // which is the multiplication of a 3x3 matrix with a 3x1 vector.
  //
  // Hence, the matrix form of T with singularity subtraction is
  //   T + [w1_0 w2_0 w3_0                              ]
  //       [               w1_1 w2_1 w3_1               ]
  //       [                              w2_2 w2_2 w3_2]
  Kokkos::parallel_for(
      "SingularitySubtraction", Kokkos::RangePolicy<ExecutionSpace>(0, num_source_points),
      KOKKOS_LAMBDA(const size_t ts) {
        T(ts * 3 + 0, ts * 3 + 0) += w1(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 0) += w1(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 0) += w1(3 * ts + 2);

        T(ts * 3 + 0, ts * 3 + 1) += w2(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 1) += w2(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 1) += w2(3 * ts + 2);

        T(ts * 3 + 0, ts * 3 + 2) += w3(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 2) += w3(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 2) += w3(3 * ts + 2);
      });
}

/// \brief Add the complementary matrix to the stokes double layer matrix
///
/// \param space The execution space
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] T The stokes double layer matrix (size num_target_points * 3 x num_source_points * 3)
template <class ExecutionSpace, typename SourceNormalVectorType, typename QuadratureWeightVectorType,
          typename MatrixType>
void add_complementary_matrix([[maybe_unused]] const ExecutionSpace &space,          //
                              const SourceNormalVectorType &source_normals,          //
                              const QuadratureWeightVectorType &quadrature_weights,  //
                              const MatrixType &T) {
  static_assert(Kokkos::is_view<SourceNormalVectorType>::value && Kokkos::is_view<QuadratureWeightVectorType>::value &&
                    Kokkos::is_view<MatrixType>::value,
                "add_complementary_matrix: The input vectors must be a Kokkos::View.");
  static_assert(SourceNormalVectorType::rank == 1 && QuadratureWeightVectorType::rank == 1 && MatrixType::rank == 2,
                "add_complementary_matrix: The input vectors must be rank 1 and the matrix must be rank 2.");
  static_assert(std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename MatrixType::value_type, double>,
                "add_complementary_matrix: The input vectors must have 'double' as their value type.");
  static_assert(std::is_same_v<typename SourceNormalVectorType::memory_space,
                               typename QuadratureWeightVectorType::memory_space> &&
                    std::is_same_v<typename SourceNormalVectorType::memory_space, typename MatrixType::memory_space>,
                "add_complementary_matrix: The input vectors must have the same memory space.");

  const size_t num_source_points = T.extent(1) / 3;
  const size_t num_target_points = T.extent(0) / 3;
  MUNDY_THROW_ASSERT(num_source_points == num_target_points, std::invalid_argument,
                     "add_complementary_matrix: The number of source and target points must be the same.");
  MUNDY_THROW_ASSERT(source_normals.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "add_complementary_matrix: source_normals must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "add_complementary_matrix: quadrature_weights must have size num_source_points.");

  // Add the complementary matrix
  Kokkos::parallel_for(
      "ComplementaryMatrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        const double normal_s0 = source_normals(3 * s + 0);
        const double normal_s1 = source_normals(3 * s + 1);
        const double normal_s2 = source_normals(3 * s + 2);

        const double normal_t0 = source_normals(3 * t + 0);
        const double normal_t1 = source_normals(3 * t + 1);
        const double normal_t2 = source_normals(3 * t + 2);

        const double weighted_normal_s0 = normal_s0 * quadrature_weights(s);
        const double weighted_normal_s1 = normal_s1 * quadrature_weights(s);
        const double weighted_normal_s2 = normal_s2 * quadrature_weights(s);

        T(t * 3 + 0, s * 3 + 0) += normal_t0 * weighted_normal_s0;
        T(t * 3 + 0, s * 3 + 1) += normal_t0 * weighted_normal_s1;
        T(t * 3 + 0, s * 3 + 2) += normal_t0 * weighted_normal_s2;

        T(t * 3 + 1, s * 3 + 0) += normal_t1 * weighted_normal_s0;
        T(t * 3 + 1, s * 3 + 1) += normal_t1 * weighted_normal_s1;
        T(t * 3 + 1, s * 3 + 2) += normal_t1 * weighted_normal_s2;

        T(t * 3 + 2, s * 3 + 0) += normal_t2 * weighted_normal_s0;
        T(t * 3 + 2, s * 3 + 1) += normal_t2 * weighted_normal_s1;
        T(t * 3 + 2, s * 3 + 2) += normal_t2 * weighted_normal_s2;
      });
}

/// \brief Add the complementary kernel v(t) += integral_{S} (normal(t) * normal(s) dot f(s) ds)
///
/// \param space The execution space
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] T The stokes double layer matrix (size num_target_points * 3 x num_source_points * 3)
template <class ExecutionSpace, typename SourceNormalVectorType, typename TargetNormalVectorType,
          typename QuadratureWeightVectorType, typename SourceForceVectorType, typename TargetVelocityVectorType>
void add_complementary_kernel([[maybe_unused]] const ExecutionSpace &space,          //
                              const SourceNormalVectorType &source_normals,          //
                              const TargetNormalVectorType &target_normals,          //
                              const QuadratureWeightVectorType &quadrature_weights,  //
                              const SourceForceVectorType &source_forces,            //
                              const TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourceNormalVectorType>::value && Kokkos::is_view<TargetNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "add_complementary_kernel: The input vectors must be a Kokkos::View.");
  static_assert(SourceNormalVectorType::rank == 1 && TargetNormalVectorType::rank == 1 &&
                    QuadratureWeightVectorType::rank == 1 && SourceForceVectorType::rank == 1 &&
                    TargetVelocityVectorType::rank == 1,
                "add_complementary_kernel: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename TargetNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "add_complementary_kernel: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourceNormalVectorType::memory_space, typename TargetNormalVectorType::memory_space> &&
          std::is_same_v<typename SourceNormalVectorType::memory_space,
                         typename QuadratureWeightVectorType::memory_space> &&
          std::is_same_v<typename SourceNormalVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourceNormalVectorType::memory_space,
                         typename TargetVelocityVectorType::memory_space>,
      "add_complementary_kernel: The input vectors must have the same memory space.");

  const size_t num_source_points = source_normals.extent(0) / 3;
  const size_t num_target_points = target_normals.extent(0) / 3;
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "add_complementary_kernel: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "add_complementary_kernel: quadrature_weights must have size num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "add_complementary_kernel: target_velocities must have size 3 * num_target_points.");

  // Add the complementary kernel
  auto complementary_contribution =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    const double normal_t0 = target_normals(3 * t + 0);
    const double normal_t1 = target_normals(3 * t + 1);
    const double normal_t2 = target_normals(3 * t + 2);

    const double normal_s0 = source_normals(3 * s + 0);
    const double normal_s1 = source_normals(3 * s + 1);
    const double normal_s2 = source_normals(3 * s + 2);

    const double scaled_normal_dot_force =
        (normal_s0 * source_forces(3 * s + 0) + normal_s1 * source_forces(3 * s + 1) +
         normal_s2 * source_forces(3 * s + 2)) *
        quadrature_weights(s);

    vx_accum += normal_t0 * scaled_normal_dot_force;
    vy_accum += normal_t1 * scaled_normal_dot_force;
    vz_accum += normal_t2 * scaled_normal_dot_force;
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  complementary_contribution);
}

/// \brief Fill the second kind Fredholm integral equation matrix for Stokes flow induced by a boundary due to
/// satisfaction of some induced surface velocity.
///
/// M * f = (1/2 I + T + N) * f = u
///   where
///   - I is the identity matrix
///   - T is the stokes double layer kernel T_{ij} = -3 viscosity / (4*pi) * r_i * r_j * r_k * normal_k / r**5 *
///   quadrature_weight_j
///   - N is the null-space correction matrix N_{ij} = normal_i * normal_j * quadrature_weight_j
///
/// For no-slip, u = -u_slip.
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename MatrixType>
void fill_skfie_matrix([[maybe_unused]] const ExecutionSpace &space,          //
                       const double viscosity,                                //
                       const size_t num_source_points,                        //
                       const size_t num_target_points,                        //
                       const SourcePosVectorType &source_positions,           //
                       const TargetPosVectorType &target_positions,           //
                       const SourceNormalVectorType &source_normals,          //
                       const QuadratureWeightVectorType &quadrature_weights,  //
                       MatrixType &M) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value && Kokkos::is_view<MatrixType>::value,
                "fill_skfie_matrix: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceNormalVectorType::rank == 1 &&
                    QuadratureWeightVectorType::rank == 1 && MatrixType::rank == 2,
                "fill_skfie_matrix: The input vectors must be rank 1 and the matrix must be rank 2.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename MatrixType::value_type, double>,
                "fill_skfie_matrix: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceNormalVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space,
                         typename QuadratureWeightVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename MatrixType::memory_space>,
      "fill_skfie_matrix: The input vectors must have the same memory space.");

  // Fill the stokes double layer matrix
  fill_stokes_double_layer_matrix(space, viscosity, num_source_points, num_target_points, source_positions,
                                  target_positions, source_normals, quadrature_weights, M);

  // Add singularity subtraction
  add_singularity_subtraction(space, M);

  // Add the complementary matrix
  add_complementary_matrix(space, source_normals, quadrature_weights, M);
}

/// \brief Apply the second kind Fredholm integral equation matrix for Stokes flow induced by a boundary due to
/// satisfaction of some induced surface velocity.
///
/// M * f = (1/2 I + T + N) * f = u
///   where
///   - I is the identity matrix
///   - T is the stokes double layer kernel T_{ij} = -3 viscosity / (4*pi) * r_i * r_j * r_k * normal_k / r**5 *
///   quadrature_weight_j
///   - N is the null-space correction matrix N_{ij} = normal_i * normal_j * quadrature_weight_j
///
/// For no-slip, u = -u_slip.
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename TargetNormalVectorType, typename QuadratureWeightVectorType,
          typename SourceForceVectorType, typename TargetVelocityVectorType>
void apply_skfie([[maybe_unused]] const ExecutionSpace &space,          //
                 const double viscosity,                                //
                 const size_t num_source_points,                        //
                 const size_t num_target_points,                        //
                 const SourcePosVectorType &source_positions,           //
                 const TargetPosVectorType &target_positions,           //
                 const SourceNormalVectorType &source_normals,          //
                 const TargetNormalVectorType &target_normals,          //
                 const QuadratureWeightVectorType &quadrature_weights,  //
                 const SourceForceVectorType &source_forces,            //
                 TargetVelocityVectorType &target_velocities) {
  static_assert(Kokkos::is_view<SourcePosVectorType>::value && Kokkos::is_view<TargetPosVectorType>::value &&
                    Kokkos::is_view<SourceNormalVectorType>::value && Kokkos::is_view<TargetNormalVectorType>::value &&
                    Kokkos::is_view<QuadratureWeightVectorType>::value &&
                    Kokkos::is_view<SourceForceVectorType>::value && Kokkos::is_view<TargetVelocityVectorType>::value,
                "apply_skfie: The input vectors must be a Kokkos::View.");
  static_assert(SourcePosVectorType::rank == 1 && TargetPosVectorType::rank == 1 && SourceNormalVectorType::rank == 1 &&
                    TargetNormalVectorType::rank == 1 && QuadratureWeightVectorType::rank == 1 &&
                    SourceForceVectorType::rank == 1 && TargetVelocityVectorType::rank == 1,
                "apply_skfie: The input vectors must be rank 1.");
  static_assert(std::is_same_v<typename SourcePosVectorType::value_type, double> &&
                    std::is_same_v<typename TargetPosVectorType::value_type, double> &&
                    std::is_same_v<typename SourceNormalVectorType::value_type, double> &&
                    std::is_same_v<typename TargetNormalVectorType::value_type, double> &&
                    std::is_same_v<typename QuadratureWeightVectorType::value_type, double> &&
                    std::is_same_v<typename SourceForceVectorType::value_type, double> &&
                    std::is_same_v<typename TargetVelocityVectorType::value_type, double>,
                "apply_skfie: The input vectors must have 'double' as their value type.");
  static_assert(
      std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetPosVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceNormalVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetNormalVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space,
                         typename QuadratureWeightVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename SourceForceVectorType::memory_space> &&
          std::is_same_v<typename SourcePosVectorType::memory_space, typename TargetVelocityVectorType::memory_space>,
      "apply_skfie: The input vectors must have the same memory space.");

  MUNDY_THROW_ASSERT(num_source_points == num_target_points, std::invalid_argument,
                     "apply_skfie: The number of source and target points must be the same.");
  MUNDY_THROW_ASSERT(source_positions.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_skfie: source_positions must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_positions.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_skfie: target_positions must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(source_normals.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_skfie: source_normals must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_normals.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_skfie: target_normals must have size 3 * num_target_points.");
  MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_source_points, std::invalid_argument,
                     "apply_skfie: quadrature_weights must have size num_source_points.");
  MUNDY_THROW_ASSERT(source_forces.extent(0) == 3 * num_source_points, std::invalid_argument,
                     "apply_skfie: source_forces must have size 3 * num_source_points.");
  MUNDY_THROW_ASSERT(target_velocities.extent(0) == 3 * num_target_points, std::invalid_argument,
                     "apply_skfie: target_velocities must have size 3 * num_target_points.");

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto skfie_contribution =
      KOKKOS_LAMBDA(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    // Read in the necessary values
    const double normal_s0 = source_normals(3 * s + 0);
    const double normal_s1 = source_normals(3 * s + 1);
    const double normal_s2 = source_normals(3 * s + 2);
    const double normal_t0 = target_normals(3 * t + 0);
    const double normal_t1 = target_normals(3 * t + 1);
    const double normal_t2 = target_normals(3 * t + 2);
    const double quadrature_weight_s = quadrature_weights(s);
    const double force_s0 = source_forces(3 * s + 0);
    const double force_s1 = source_forces(3 * s + 1);
    const double force_s2 = source_forces(3 * s + 2);
    const double force_t0 = source_forces(3 * t + 0);  // Assumes that we have the same number of sources and targets
    const double force_t1 = source_forces(3 * t + 1);
    const double force_t2 = source_forces(3 * t + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : quake_inv_sqrt(dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // Compute the double layer potential
    const double sxx = normal_s0 * (force_s0 + force_t0) * quadrature_weight_s;
    const double sxy = normal_s0 * (force_s1 + force_t1) * quadrature_weight_s;
    const double sxz = normal_s0 * (force_s2 + force_t2) * quadrature_weight_s;
    const double syx = normal_s1 * (force_s0 + force_t0) * quadrature_weight_s;
    const double syy = normal_s1 * (force_s1 + force_t1) * quadrature_weight_s;
    const double syz = normal_s1 * (force_s2 + force_t2) * quadrature_weight_s;
    const double szx = normal_s2 * (force_s0 + force_t0) * quadrature_weight_s;
    const double szy = normal_s2 * (force_s1 + force_t1) * quadrature_weight_s;
    const double szz = normal_s2 * (force_s2 + force_t2) * quadrature_weight_s;

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    // Compute the complementarity term v += normal(t) * normal(s) dot f(s) w(s)
    const double scaled_normal_dot_force =
        (normal_s0 * force_s0 + normal_s1 * force_s1 + normal_s2 * force_s2) * quadrature_weight_s;

    vx_accum += dx * coeff + scaled_normal_dot_force * normal_t0;
    vy_accum += dy * coeff + scaled_normal_dot_force * normal_t1;
    vz_accum += dz * coeff + scaled_normal_dot_force * normal_t2;
  };

  panelize_velocity_kernel_over_target_points<32>(space, num_target_points, num_source_points, target_velocities,
                                                  skfie_contribution);
}

template <typename ExecSpace>
class PeripheryT {
 public:
  //! \name Types
  //@{

  using DeviceExecutionSpace = ExecSpace;
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  PeripheryT() = delete;

  /// \brief No copy constructor
  PeripheryT(const PeripheryT &) = delete;

  /// \brief No copy assignment
  PeripheryT &operator=(const PeripheryT &) = delete;

  /// \brief Default move constructor
  PeripheryT(PeripheryT &&) = default;

  /// \brief Default move assignment
  PeripheryT &operator=(PeripheryT &&) = default;

  /// \brief Destructor
  ~PeripheryT() = default;

  /// \brief Constructor
  PeripheryT(const size_t num_surface_nodes, const double viscosity)
      : num_surface_nodes_(num_surface_nodes),
        viscosity_(viscosity),
        is_surface_positions_set_(false),
        is_surface_normals_set_(false),
        is_quadrature_weights_set_(false),
        is_inverse_self_interaction_matrix_set_(false),
        surface_positions_("surface_positions", 3 * num_surface_nodes_),
        surface_normals_("surface_normals", 3 * num_surface_nodes_),
        quadrature_weights_("quadrature_weights", num_surface_nodes_),
        M_inv_("M_inv", 3 * num_surface_nodes_, 3 * num_surface_nodes_) {
    // Initialize host Kokkos mirrors
    surface_positions_host_ = Kokkos::create_mirror_view(surface_positions_);
    surface_normals_host_ = Kokkos::create_mirror_view(surface_normals_);
    quadrature_weights_host_ = Kokkos::create_mirror_view(quadrature_weights_);
    M_inv_host_ = Kokkos::create_mirror_view(M_inv_);
  }
  //@}

  //! \name Setters
  //@{

  /// Instead of complex constructors, we offer a variety of setters to initialize the periphery
  /// Each array can be passed in as a Kokkos view, a raw pointer, or a filename to be read from disk

  /// \brief Set the surface positions
  ///
  /// \param surface_positions The surface positions (size num_nodes * 3)
  template <class MemorySpace, class Layout>
  PeripheryT &set_surface_positions(const Kokkos::View<double *, Layout, MemorySpace> &surface_positions) {
    MUNDY_THROW_ASSERT(surface_positions.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                       "set_surface_positions: surface_positions must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_positions_, surface_positions);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface positions
  ///
  /// \param surface_positions The surface positions (size num_nodes * 3)
  PeripheryT &set_surface_positions(const double *surface_positions) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        surface_positions_host_(idx) = surface_positions[idx];
      }
    }
    Kokkos::deep_copy(surface_positions_, surface_positions_host_);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface positions
  ///
  /// \param surface_positions_filename The filename to read the surface positions from
  PeripheryT &set_surface_positions(const std::string &surface_positions_filename) {
    read_vector_from_file(surface_positions_filename, 3 * num_surface_nodes_, surface_positions_host_);
    Kokkos::deep_copy(surface_positions_, surface_positions_host_);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  template <class MemorySpace, class Layout>
  PeripheryT &set_surface_normals(const Kokkos::View<double *, Layout, MemorySpace> &surface_normals) {
    MUNDY_THROW_ASSERT(surface_normals.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                       "set_surface_normals: surface_normals must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_normals_, surface_normals);
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  PeripheryT &set_surface_normals(const double *surface_normals) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        surface_normals_host_(idx) = surface_normals[idx];
      }
    }
    Kokkos::deep_copy(surface_normals_, surface_normals_host_);
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals
  ///
  /// \param surface_normals_filename The filename to read the surface normals from
  PeripheryT &set_surface_normals(const std::string &surface_normals_filename) {
    read_vector_from_file(surface_normals_filename, 3 * num_surface_nodes_, surface_normals_host_);
    Kokkos::deep_copy(surface_normals_, surface_normals_host_);
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  template <class MemorySpace, class Layout>
  PeripheryT &set_quadrature_weights(const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights) {
    MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_surface_nodes_, std::invalid_argument,
                       "set_quadrature_weights: quadrature_weights must have size num_surface_nodes.");
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  PeripheryT &set_quadrature_weights(const double *quadrature_weights) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      quadrature_weights_host_(i) = quadrature_weights[i];
    }
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights_host_);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights_filename The filename to read the quadrature weights from
  PeripheryT &set_quadrature_weights(const std::string &quadrature_weights_filename) {
    read_vector_from_file(quadrature_weights_filename, num_surface_nodes_, quadrature_weights_host_);
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights_host_);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param M_inv The precomputed matrix (size 3 * num_nodes x 3 * num_nodes)
  template <class MemorySpace>
  PeripheryT &set_inverse_self_interaction_matrix(
      const Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> &M_inv) {
    MUNDY_THROW_ASSERT(
        (M_inv.extent(0) == 3 * num_surface_nodes_) && (M_inv.extent(1) == 3 * num_surface_nodes_),
        std::invalid_argument,
        "set_inverse_self_interaction_matrix: M_inv must have size 3 * num_surface_nodes x 3 * num_surface_nodes.");
    Kokkos::deep_copy(M_inv_, M_inv);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param M_inv The precomputed matrix (size 3 * num_nodes x 3 * num_nodes)
  PeripheryT &set_inverse_self_interaction_matrix(const double *M_inv_flat) {
    for (size_t i = 0; i < 3 * num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3 * num_surface_nodes_; ++j) {
        const size_t idx = i * num_surface_nodes_ + j;
        M_inv_host_(i, j) = M_inv_flat[idx];
      }
    }
    Kokkos::deep_copy(M_inv_, M_inv_host_);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param inverse_self_interaction_matrix_filename The filename to read the precomputed matrix from
  PeripheryT &set_inverse_self_interaction_matrix(const std::string &inverse_self_interaction_matrix_filename) {
    read_matrix_from_file(inverse_self_interaction_matrix_filename, 3 * num_surface_nodes_, 3 * num_surface_nodes_,
                          M_inv_host_);
    Kokkos::deep_copy(M_inv_, M_inv_host_);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }
  //@}

  //! \name Public member functions
  //@{

  // TODO(palmerb4): A better method would be read_from_file and write_to_file, which would be more general
  PeripheryT &build_inverse_self_interaction_matrix(
      const bool &write_to_file = true,
      const std::string &inverse_self_interaction_matrix_filename = "inverse_self_interaction_matrix.dat") {
    MUNDY_THROW_REQUIRE(is_surface_positions_set_ && is_surface_normals_set_ && is_quadrature_weights_set_,
                        std::runtime_error,
                        "build_inverse_self_interaction_matrix: surface_positions, surface_normals, and "
                        "quadrature_weights must be set before calling this function.");

    // Fill the self-interaction matrix using temporary storage
    Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("M", 3 * num_surface_nodes_,
                                                                     3 * num_surface_nodes_);
    fill_skfie_matrix(DeviceExecutionSpace(), viscosity_, num_surface_nodes_, num_surface_nodes_, surface_positions_,
                      surface_positions_, surface_normals_, quadrature_weights_, M);

    // Now invert the matrix and store the result
    invert_matrix(DeviceExecutionSpace(), M, M_inv_);

    if (write_to_file) {
      // Write the precomputed matrix to a file
      Kokkos::deep_copy(M_inv_host_, M_inv_);
      write_matrix_to_file(inverse_self_interaction_matrix_filename, M_inv_host_);
    }
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Compute the surface forces induced by external flow on the surface
  ///
  /// \param[in] external_flow_velocity The external flow velocity (size num_nodes x 3)
  /// \param[out] surface_forces The surface forces induced by enforcing no-slip on the surface (size num_nodes x 3)
  PeripheryT &compute_surface_forces(
      const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &external_flow_velocity,
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &surface_forces) {
    // Check if the periphery is in a valid state
    if (!is_inverse_self_interaction_matrix_set_) {
      std::cerr << "compute_surface_forces: inverse self-interaction matrix must be set before calling this function."
                << std::endl;
      return *this;
    }

    // Apply the inverse of the self-interaction matrix to the external flow velocity
    // Notice the negative one in the gemv call, which accounts for the fact that our force should balance the u_slip
    KokkosBlas::gemv(DeviceExecutionSpace(), "N", -1.0, M_inv_, external_flow_velocity, 1.0, surface_forces);

    return *this;
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of nodes
  size_t get_num_nodes() const {
    return num_surface_nodes_;
  }

  /// \brief Get the viscosity
  ///
  /// \return The viscosity
  double get_viscosity() const {
    return viscosity_;
  }

  /// \brief Get the surface positions
  ///
  /// \return The surface positions
  const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &get_surface_positions() const {
    return surface_positions_;
  }

  /// \brief Get the surface normals
  const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &get_surface_normals() const {
    return surface_normals_;
  }

  /// \brief Get the quadrature weights
  const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &get_quadrature_weights() const {
    return quadrature_weights_;
  }

  /// \brief Get the inverse of the self-interaction matrix
  const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &get_M_inv() const {
    return M_inv_;
  }
  //@}

 private:
  //! \name Private member variables
  //@{

  size_t num_surface_nodes_;  //!< The number of nodes
  double viscosity_;          //!< The viscosity

  bool is_surface_positions_set_;                //!< Whether the surface positions have been set
  bool is_surface_normals_set_;                  //!< Whether the surface normals have been set
  bool is_quadrature_weights_set_;               //!< Whether the quadrature weights have been set
  bool is_inverse_self_interaction_matrix_set_;  //!< Whether the inverse of the self-interaction matrix has been set

  // Host Kokkos views
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>
      surface_positions_host_;  //!< The surface positions (host)
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_normals_host_;  //!< The surface normals (host)
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>
      quadrature_weights_host_;  //!< The quadrature weights (host)
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      M_inv_host_;  //!< The inverse of the self-interaction matrix (host)

  // Device Kokkos views
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_positions_;  //!< The node positions (device)
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_normals_;    //!< The surface normals (device)
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace>
      quadrature_weights_;  //!< The quadrature weights (device)
  Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace>
      M_inv_;  //!< The inverse of the self-interaction matrix (device)
  //@}
};  // class PeripheryT

using Periphery = PeripheryT<Kokkos::DefaultExecutionSpace>;  //!< Default periphery type

}  // namespace periphery

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_
