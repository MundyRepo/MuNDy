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

#ifndef MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_FASTDIRECTPERIPHERY_HPP_
#define MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_FASTDIRECTPERIPHERY_HPP_

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

// C++ core
#include <algorithm>
#include <fstream>
#include <iostream>
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
#define DOUBLE_ZERO 1.0e-12

namespace mundy {

namespace alens {

namespace periphery {

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
                      std::string("gen_sphere_quadrature: radius must be positive. The current value is ") + radius);
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
  const double weightfactor = radius * radius * 2 * M_PI / (2 * order + 2);
  for (int j = 0; j < order + 1; j++) {
    for (int k = 0; k < 2 * order + 2; k++) {
      const double costhetaj = nodes_gl[order - j];
      const double phik = 2 * M_PI * k / (2 * order + 2);
      const double sinthetaj = std::sqrt(1 - costhetaj * costhetaj);
      const int index = (j * (2 * order + 2)) + k + (include_poles ? 1 : 0);
      points[3 * index] = sinthetaj * std::cos(phik);
      points[3 * index + 1] = sinthetaj * std::sin(phik);
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void invert_matrix([[maybe_unused]] const ExecutionSpace &space,
                   const Kokkos::View<double **, Layout, MemorySpace> &matrix,
                   const Kokkos::View<double **, Layout, MemorySpace> &matrix_inv) {
  // Check the input sizes
  const size_t matrix_size = matrix.extent(0);
  MUNDY_THROW_ASSERT(matrix.extent(1) == matrix_size, std::invalid_argument, "invert_matrix: matrix must be square.");
  MUNDY_THROW_ASSERT((matrix_inv.extent(0) == matrix_size) && (matrix_inv.extent(1) == matrix_size),
                     std::invalid_argument, "invert_matrix: matrix_inv must be the same size as the matrix to invert.");

  // Create a view to store the pivots
  Kokkos::View<int *, Layout, MemorySpace> pivots("pivots", matrix_size);

  // Fill matrix_inv with the identity matrix
  Kokkos::deep_copy(matrix_inv, 0.0);
  Kokkos::parallel_for(
      "FillIdentity", Kokkos::RangePolicy<>(0, matrix_size), KOKKOS_LAMBDA(const size_t i) { matrix_inv(i, i) = 1.0; });

  // Solve the dense linear equation system M*X = I, which results in X = M^{-1}
  // On exist, M is replaced with its LU decomposition
  //           M_inv is replaced with the solution X = M^{-1}
  KokkosBlas::gesv(matrix, matrix_inv, pivots);
}

/// \brief Write a matrix to a file
///
/// \param[in] filename The filename
/// \param[in] matrix_host The matrix to write (host)
template <typename MatrixDataType, class Layout>
void write_matrix_to_file(const std::string &filename,
                          const Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Perform the write
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write the matrix to the file (using reinterpret_cast to map directly to binary data)
  const size_t num_rows = matrix_host.extent(0);
  const size_t num_columns = matrix_host.extent(1);
  outfile.write(reinterpret_cast<const char *>(&num_rows), sizeof(size_t));
  outfile.write(reinterpret_cast<const char *>(&num_columns), sizeof(size_t));
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      outfile.write(reinterpret_cast<const char *>(&matrix_host(i, j)), sizeof(MatrixDataType));
    }
  }

  // Close the file
  outfile.close();
}

/// \brief Read a matrix from a file
///
/// \param[in] filename The filename
/// \param[out] matrix_host The matrix to read (host)
template <typename MatrixDataType, class Layout>
void read_matrix_from_file(const std::string &filename, const size_t expected_num_rows,
                           const size_t expected_num_columns,
                           const Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Read the matrix from a file
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  size_t num_rows;
  size_t num_columns;
  infile.read(reinterpret_cast<char *>(&num_rows), sizeof(size_t));
  infile.read(reinterpret_cast<char *>(&num_columns), sizeof(size_t));
  if ((num_rows != expected_num_rows) || (num_columns != expected_num_columns)) {
    std::cerr << "Matrix size mismatch: expected (" << expected_num_rows << ", " << expected_num_columns << "), got ("
              << num_rows << ", " << num_columns << ")" << std::endl;
    return;
  }
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      infile.read(reinterpret_cast<char *>(&matrix_host(i, j)), sizeof(MatrixDataType));
    }
  }

  // Close the file
  infile.close();
}

/// \brief Write a vector to a file
///
/// \param[in] filename The filename
/// \param[in] vector_host The vector to write (host)
template <typename VectorDataType, class Layout>
void write_vector_to_file(const std::string &filename,
                          const Kokkos::View<VectorDataType *, Layout, Kokkos::HostSpace> &vector_host) {
  // Perform the write
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write the vector to the file (using reinterpret_cast to map directly to binary data)
  const size_t num_elements = vector_host.extent(0);
  outfile.write(reinterpret_cast<const char *>(&num_elements), sizeof(size_t));
  for (size_t i = 0; i < num_elements; ++i) {
    outfile.write(reinterpret_cast<const char *>(&vector_host(i)), sizeof(VectorDataType));
  }

  // Close the file
  outfile.close();
}

/// \brief Read a vector from a file
///
/// \param[in] filename The filename
/// \param[out] vector_host The vector to read (host)
template <typename VectorDataType, class Layout>
void read_vector_from_file(const std::string &filename, const size_t expected_num_elements,
                           const Kokkos::View<VectorDataType *, Layout, Kokkos::HostSpace> &vector_host) {
  // Read the vector from a file
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  size_t num_elements;
  infile.read(reinterpret_cast<char *>(&num_elements), sizeof(size_t));
  if (num_elements != expected_num_elements) {
    std::cerr << "Vector size mismatch: expected " << expected_num_elements << ", got " << num_elements << std::endl;
    return;
  }
  for (size_t i = 0; i < num_elements; ++i) {
    infile.read(reinterpret_cast<char *>(&vector_host(i)), sizeof(VectorDataType));
  }

  // Close the file
  infile.close();
}

template <class ExecutionSpace, class MemorySpace, class Layout, typename Func>
void penalize_velocity_kernel_over_target_points([[maybe_unused]] const ExecutionSpace &space, int num_target_points,
                                                 int num_source_points, int panel_size,
                                                 Kokkos::View<double *, Layout, MemorySpace> target_velocities,
                                                 const Func &compute_velocity_contribution) {
  int num_panels = (num_target_points + panel_size - 1) / panel_size;

  // Define the team policy with the number of panels
  Kokkos::parallel_for(
      "Penalize_Target_Points", Kokkos::TeamPolicy<>(num_panels, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int panel_start = team_member.league_rank() * panel_size;
        const int panel_end =
            (panel_start + panel_size) > num_target_points ? num_target_points : (panel_start + panel_size);

        // Local accumulation arrays for each target point in the panel
        double local_vx[panel_size] = {0.0};
        double local_vy[panel_size] = {0.0};
        double local_vz[panel_size] = {0.0};

        // Loop over each target point in the panel
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, panel_start, panel_end), [&](const int t) {
          Kokkos::Array<double, 3> v_sum({0.0, 0.0, 0.0});

          // Loop over all source points
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team_member, num_source_points),
              [&](const int s, Kokkos::Array<double, 3> &v_accum) {
                // Call the custom operation to compute the contribution
                compute_velocity_contribution(t, s, v_accum[0], v_accum[1], v_accum[2]);
              },
              v_sum);

          // Store the results in the local arrays
          local_vx[t - panel_start] = v_sum[0];
          local_vy[t - panel_start] = v_sum[1];
          local_vz[t - panel_start] = v_sum[2];
        });

        // After processing, update the global output using a single thread per team
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          for (int t = panel_start; t < panel_end; ++t) {
            Kokkos::atomic_add(&target_velocities(3 * t + 0), local_vx[t - panel_start]);
            Kokkos::atomic_add(&target_velocities(3 * t + 1), local_vy[t - panel_start]);
            Kokkos::atomic_add(&target_velocities(3 * t + 2), local_vz[t - panel_start]);
          }
        });
      });
}

template <class ExecutionSpace, class MemorySpace, class Layout, typename Func>
void panelize_velocity_kernel_over_target_points([[maybe_unused]] const ExecutionSpace &space, int num_target_points,
                                                 int num_source_points, int panel_size,
                                                 Kokkos::View<double *, Layout, MemorySpace> target_velocities,
                                                 const Func &compute_velocity_contribution) {
  int num_panels = (num_target_points + panel_size - 1) / panel_size;

  // Define the team policy with the number of panels
  Kokkos::parallel_for(
      "Panalize_Target_Points", Kokkos::TeamPolicy<>(num_panels, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int panel_start = team_member.league_rank() * panel_size;
        const int panel_end =
            (panel_start + panel_size) > num_target_points ? num_target_points : (panel_start + panel_size);

        // Local accumulation arrays for each target point in the panel
        double local_vx[panel_size] = {0.0};
        double local_vy[panel_size] = {0.0};
        double local_vz[panel_size] = {0.0};

        // Loop over each target point in the panel
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, panel_start, panel_end), [&](const int t) {
          Kokkos::Array<double, 3> v_sum({0.0, 0.0, 0.0});

          // Loop over all source points
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team_member, num_source_points),
              [&](const int s, Kokkos::Array<double, 3> &v_accum) {
                // Call the custom operation to compute the contribution
                compute_velocity_contribution(t, s, v_accum[0], v_accum[1], v_accum[2]);
              },
              v_sum);

          // Store the results in the local arrays
          local_vx[t - panel_start] = v_sum[0];
          local_vy[t - panel_start] = v_sum[1];
          local_vz[t - panel_start] = v_sum[2];
        });

        // After processing, update the global output using a single thread per team
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          for (int t = panel_start; t < panel_end; ++t) {
            Kokkos::atomic_add(&target_velocities(3 * t + 0), local_vx[t - panel_start]);
            Kokkos::atomic_add(&target_velocities(3 * t + 1), local_vy[t - panel_start]);
            Kokkos::atomic_add(&target_velocities(3 * t + 2), local_vz[t - panel_start]);
          }
        });
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_stokes_kernel([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                         const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                         const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                         const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                         const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;

    const double inner_prod = fx * dx + fy * dy + fz * dz;
    const double scale_factor_rinv3 = scale_factor * rinv3;

    // Accumulate velocity contribution to local variables
    vx_accum += scale_factor_rinv3 * (r2 * fx + dx * inner_prod);
    vy_accum += scale_factor_rinv3 * (r2 * fy + dy * inner_prod);
    vz_accum += scale_factor_rinv3 * (r2 * fz + dz * inner_prod);
  };

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_weighted_stokes_kernel([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                                  const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                                  const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                                  const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                                  const Kokkos::View<double *, Layout, MemorySpace> &source_weights,
                                  const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(r2);
    const double rinv2 = rinv * rinv;

    const double inner_prod_rinv2 = (fx * dx + fy * dy + fz * dz) * rinv2;
    const double scale_factor_rinv = scale_factor * rinv;

    // Accumulate velocity contribution to local variables
    vx_accum += scale_factor_rinv * (fx + dx * inner_prod_rinv2);
    vy_accum += scale_factor_rinv * (fy + dy * inner_prod_rinv2);
    vz_accum += scale_factor_rinv * (fz + dz * inner_prod_rinv2);
  };

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_rpy_kernel([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                      const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                      const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                      const Kokkos::View<double *, Layout, MemorySpace> &source_radii,
                      const Kokkos::View<double *, Layout, MemorySpace> &target_radii,
                      const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                      const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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

    const double a2_over_three = (1.0 / 3.0) * a * a;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;
    const double rinv5 = rinv * rinv * rinv3;
    const double fdotr = fx * dx + fy * dy + fz * dz;

    const double three_fdotr_rinv5 = 3 * fdotr * rinv5;
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
    const double lap_coeff = target_radii(t) * target_radii(t) / 6.0;
    vx_accum += v0 + lap_coeff * lap0;
    vy_accum += v1 + lap_coeff * lap1;
    vz_accum += v2 + lap_coeff * lap2;
  };

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
                                              rpy_computation);
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_stokes_double_layer_kernel_ss([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                                         const size_t num_source_points, const size_t num_target_points,
                                         const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                                         const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                                         const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                                         const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                                         const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                                         const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(dr2);
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

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_stokes_double_layer_kernel([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                                      const size_t num_source_points, const size_t num_target_points,
                                      const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                                      const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                                      const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                                      const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                                      const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                                      const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(dr2);
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

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
                                              stokes_double_layer_contribution);
}

/// \brief Apply local drag to the sphere velocities v += 1/(6 pi mu r) f
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_local_drag([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                      Kokkos::View<double *, Layout, MemorySpace> sphere_velocities,
                      Kokkos::View<double *, Layout, MemorySpace> sphere_forces,
                      Kokkos::View<double *, Layout, MemorySpace> sphere_radii) {
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void fill_stokes_double_layer_matrix([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                                     const size_t num_source_points, const size_t num_target_points,
                                     const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                                     const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                                     const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                                     const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                                     const Kokkos::View<double **, Layout, MemorySpace> &T) {
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
        const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : 1.0 / std::sqrt(dr2);
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void add_singularity_subtraction([[maybe_unused]] const ExecutionSpace &space,
                                 const Kokkos::View<double **, Layout, MemorySpace> &T) {
  const size_t num_source_points = T.extent(1) / 3;
  const size_t num_target_points = T.extent(0) / 3;

  // Add the singularity subtraction to T
  // Create a vector that has 1 in the x-component and 0 in the y and z components for each source point
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void add_complementary_matrix([[maybe_unused]] const ExecutionSpace &space,
                              const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                              const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                              const Kokkos::View<double **, Layout, MemorySpace> &T) {
  const size_t num_source_points = T.extent(1) / 3;
  const size_t num_target_points = T.extent(0) / 3;
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void add_complementary_kernel([[maybe_unused]] const ExecutionSpace &space,
                              const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                              const Kokkos::View<double *, Layout, MemorySpace> &target_normals,
                              const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                              const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                              const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void fill_skfie_matrix([[maybe_unused]] const ExecutionSpace &space, const double viscosity,
                       const size_t num_source_points, const size_t num_target_points,
                       const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                       const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                       const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                       const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                       const Kokkos::View<double **, Layout, MemorySpace> &M) {
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
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_skfie([[maybe_unused]] const ExecutionSpace &space, const double viscosity, const size_t num_source_points,
                 const size_t num_target_points, const Kokkos::View<double *, Layout, MemorySpace> &source_positions,
                 const Kokkos::View<double *, Layout, MemorySpace> &target_positions,
                 const Kokkos::View<double *, Layout, MemorySpace> &source_normals,
                 const Kokkos::View<double *, Layout, MemorySpace> &target_normals,
                 const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                 const Kokkos::View<double *, Layout, MemorySpace> &source_forces,
                 const Kokkos::View<double *, Layout, MemorySpace> &target_velocities) {
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
    const double force_t0 = source_forces(3 * t + 0);
    const double force_t1 = source_forces(3 * t + 1);
    const double force_t2 = source_forces(3 * t + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const double rinv = dr2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(dr2);
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

    vx_accum += dx * coeff + scaled_normal_dot_force;
    vy_accum += dy * coeff + scaled_normal_dot_force;
    vz_accum += dz * coeff + scaled_normal_dot_force;
  };

  panelize_velocity_kernel_over_target_points(space, num_target_points, num_source_points, 128, target_velocities,
                                              skfie_contribution);
}

class FastDirectPeriphery {
 public:
  //! \name Types
  //@{

  using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
  using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  FastDirectPeriphery() = delete;

  /// \brief No copy constructor
  FastDirectPeriphery(const FastDirectPeriphery &) = delete;

  /// \brief No copy assignment
  FastDirectPeriphery &operator=(const FastDirectPeriphery &) = delete;

  /// \brief Default move constructor
  FastDirectPeriphery(FastDirectPeriphery &&) = default;

  /// \brief Default move assignment
  FastDirectPeriphery &operator=(FastDirectPeriphery &&) = default;

  /// \brief Destructor
  ~FastDirectPeriphery() = default;

  /// \brief Constructor
  FastDirectPeriphery(const size_t num_surface_nodes, const double viscosity)
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
  FastDirectPeriphery &set_surface_positions(const Kokkos::View<double *, Layout, MemorySpace> &surface_positions) {
    MUNDY_THROW_REQUIRE(surface_positions.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                        "set_surface_positions: surface_positions must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_positions_, surface_positions);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface positions
  ///
  /// \param surface_positions The surface positions (size num_nodes * 3)
  FastDirectPeriphery &set_surface_positions(const double *surface_positions) {
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
  FastDirectPeriphery &set_surface_positions(const std::string &surface_positions_filename) {
    read_vector_from_file(surface_positions_filename, 3 * num_surface_nodes_, surface_positions_host_);
    Kokkos::deep_copy(surface_positions_, surface_positions_host_);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  template <class MemorySpace, class Layout>
  FastDirectPeriphery &set_surface_normals(const Kokkos::View<double *, Layout, MemorySpace> &surface_normals) {
    MUNDY_THROW_REQUIRE(surface_normals.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                        "set_surface_normals: surface_normals must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_normals_, surface_normals);
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  FastDirectPeriphery &set_surface_normals(const double *surface_normals) {
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
  FastDirectPeriphery &set_surface_normals(const std::string &surface_normals_filename) {
    read_vector_from_file(surface_normals_filename, 3 * num_surface_nodes_, surface_normals_host_);
    Kokkos::deep_copy(surface_normals_, surface_normals_host_);
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  template <class MemorySpace, class Layout>
  FastDirectPeriphery &set_quadrature_weights(const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights) {
    MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_surface_nodes_, std::invalid_argument,
                       "set_quadrature_weights: quadrature_weights must have size num_surface_nodes.");
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  FastDirectPeriphery &set_quadrature_weights(const double *quadrature_weights) {
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
  FastDirectPeriphery &set_quadrature_weights(const std::string &quadrature_weights_filename) {
    read_vector_from_file(quadrature_weights_filename, num_surface_nodes_, quadrature_weights_host_);
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights_host_);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param M_inv The precomputed matrix (size 3 * num_nodes x 3 * num_nodes)
  template <class MemorySpace>
  FastDirectPeriphery &set_inverse_self_interaction_matrix(
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
  FastDirectPeriphery &set_inverse_self_interaction_matrix(const double *M_inv_flat) {
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
  FastDirectPeriphery &set_inverse_self_interaction_matrix(
      const std::string &inverse_self_interaction_matrix_filename) {
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
  FastDirectPeriphery &build_inverse_self_interaction_matrix(
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
  FastDirectPeriphery &compute_surface_forces(
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
};  // class FastDirectPeriphery

}  // namespace periphery

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_FASTDIRECTPERIPHERY_HPP_
