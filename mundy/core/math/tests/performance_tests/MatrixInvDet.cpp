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

//! \file OldMatrixInverse.cpp
/// \brief Runtime benchmark of mundy::determinant/inverse

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstdlib>  // for rand, RAND_MAX
#include <string>   // for std::string
#include <vector>   // for std::vector

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>                 // for Kokkos::initialize, Kokkos::finalize
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_math/Matrix.hpp>  // for mundy::Matrix, mundy::determinant/cofactors/adjugate/inverse

/// \brief Fill `data` (num_matrices * N * N flattened doubles) with diagonally-dominant (hence
/// well-conditioned and invertible) N x N matrices.
void randomize_diag_dominant(std::vector<double>& data, size_t N) {
  const size_t num_matrices = data.size() / (N * N);
  for (size_t m = 0; m < num_matrices; ++m) {
    double* block = data.data() + m * N * N;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        block[i * N + j] = (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
      }
      block[i * N + i] += 5.0 * static_cast<double>(N);  // diagonally dominant regardless of N
    }
  }
}

template <size_t N>
void bench_det(size_t num_matrices) {
  std::vector<double> data(num_matrices * N * N);
  randomize_diag_dominant(data, N);

  double checksum = 0.0;  // laundered through doNotOptimizeAway so the batch loop can't be elided

  ankerl::nanobench::Bench det_bench;
  det_bench.title("determinant, Matrix<double," + std::to_string(N) + "," + std::to_string(N) + ">")
      .unit("op")
      .batch(num_matrices)
      .performanceCounters(true)
      .minEpochIterations(20);
  det_bench.run("det", [&] {
    for (size_t m = 0; m < num_matrices; ++m) {
      const auto mat = mundy::get_matrix<double, N, N>(data.data() + m * N * N);
      checksum += mundy::determinant(mat);
    }
    ankerl::nanobench::doNotOptimizeAway(checksum);
  });
}

template <size_t N>
void bench_inv(size_t num_matrices) {
  std::vector<double> data(num_matrices * N * N);
  randomize_diag_dominant(data, N);

  double checksum = 0.0;  // laundered through doNotOptimizeAway so the batch loop can't be elided

  ankerl::nanobench::Bench inv_bench;
  inv_bench.title("inverse, Matrix<double," + std::to_string(N) + "," + std::to_string(N) + ">")
      .unit("op")
      .batch(num_matrices)
      .performanceCounters(true)
      .minEpochIterations(20);
  inv_bench.run("inv", [&] {
    for (size_t m = 0; m < num_matrices; ++m) {
      const auto mat = mundy::get_matrix<double, N, N>(data.data() + m * N * N);
      checksum += mundy::inverse(mat)(0, 0);
    }
    ankerl::nanobench::doNotOptimizeAway(checksum);
  });
}

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  const size_t num_matrices = 5000;

  bench_det<2>(num_matrices);
  bench_det<3>(num_matrices);
  bench_det<4>(num_matrices);
  bench_det<5>(num_matrices);
  bench_det<6>(num_matrices);
  bench_det<7>(num_matrices);
  bench_det<8>(num_matrices);
  bench_det<9>(num_matrices);
  bench_det<10>(num_matrices);
  bench_det<11>(num_matrices);
  bench_det<12>(num_matrices);

  bench_inv<2>(num_matrices);
  bench_inv<3>(num_matrices);
  bench_inv<4>(num_matrices);
  bench_inv<5>(num_matrices);
  bench_inv<6>(num_matrices);
  bench_inv<7>(num_matrices);
  bench_inv<8>(num_matrices);
  bench_inv<9>(num_matrices);
  bench_inv<10>(num_matrices);
  bench_inv<11>(num_matrices);
  bench_inv<12>(num_matrices);

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
