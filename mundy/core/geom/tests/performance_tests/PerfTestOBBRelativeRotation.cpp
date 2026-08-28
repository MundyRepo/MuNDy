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

/// \file PerfTestOBBRelativeRotation.cpp
/// \brief Compare the cost of computing the OBB relative rotation matrix R_{ij} = A_i · B_j
///        under two OBB orientation storage strategies:
///
///   Strategy A — Rotation matrix storage  (15 scalars per OBB: center 3 + axes 9 + half-extents 3)
///     R = Q_A^T * Q_B          (matrix–matrix product, 27 mults + 18 adds = 45 FLOPs)
///
///   Strategy B — Unit quaternion storage  (10 scalars per OBB: center 3 + quat 4 + half-extents 3)
///     q_rel = conj(q_A) * q_B  (quaternion product, 16 mults + 12 adds = 28 FLOPs)
///     R     = mat(q_rel)        (quat->matrix, 12 mults + 12 adds ≈ 24 FLOPs)
///
/// The output R is the same in both cases and is used identically in the OBB–OBB SAT test.
/// This benchmark answers: does the 5-scalar storage saving of quaternions outweigh the extra
/// computation of the quat->matrix conversion?

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstdlib>  // for rand, RAND_MAX
#include <iostream>

// External
#include <nanobench.h>

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy
#include <mundy_math/Matrix3.hpp>     // for mundy::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion, mundy::conjugate, mundy::quaternion_to_rotation_matrix

// ============================================================
// Helpers
// ============================================================

static double rnd01() {
  return static_cast<double>(rand()) / RAND_MAX;
}

/// Return a random unit quaternion via Shoemake's method.
static mundy::Quaternion<double> random_unit_quat() {
  // Shoemake 1992: uniform distribution on SO(3)
  const double u1 = rnd01();
  const double u2 = rnd01() * 6.2831853;
  const double u3 = rnd01() * 6.2831853;
  const double s1 = std::sqrt(1.0 - u1);
  const double s2 = std::sqrt(u1);
  return mundy::Quaternion<double>{s1 * std::cos(u2), s1 * std::sin(u2), s2 * std::cos(u3), s2 * std::sin(u3)};
}

/// Convert a unit quaternion to its rotation matrix.
static mundy::Matrix3<double> quat_to_mat(const mundy::Quaternion<double>& q) {
  return mundy::quaternion_to_rotation_matrix(q);
}

// ============================================================
// Benchmark
// ============================================================

void bench_relative_rotation() {
  ankerl::nanobench::Bench bench;
  bench.title("OBB relative rotation R = Q_A^T Q_B")
      .unit("op")
      .relative(true)
      .performanceCounters(true)
      .warmup(500)
      .epochs(500)
      .minEpochTime(std::chrono::microseconds(200))
      .minEpochIterations(100000);

  // Generate random orientations for two OBBs.
  const mundy::Quaternion<double> q_a = random_unit_quat();
  const mundy::Quaternion<double> q_b = random_unit_quat();
  const mundy::Matrix3<double> R_a = quat_to_mat(q_a);
  const mundy::Matrix3<double> R_b = quat_to_mat(q_b);

  // ---------------------------------------------------------------
  // Strategy A: rotation matrix storage — R = Q_A^T * Q_B
  //
  // view_transpose() returns a zero-copy transposed view; the
  // subsequent matrix–matrix product uses the transposed accessor.
  // ---------------------------------------------------------------
  bench.run("A: matrix storage  R = Q_A^T * Q_B", [&] {
    const auto R = R_a.view_transpose() * R_b;
    ankerl::nanobench::doNotOptimizeAway(R);
  });

  // ---------------------------------------------------------------
  // Strategy B: quaternion storage — q_rel = conj(q_A)*q_B, R = mat(q_rel)
  // ---------------------------------------------------------------
  bench.run("B: quat storage    R = mat(conj(q_A)*q_B)", [&] {
    const auto q_rel = conjugate(q_a) * q_b;
    const auto R = quaternion_to_rotation_matrix(q_rel);
    ankerl::nanobench::doNotOptimizeAway(R);
  });

  // ---------------------------------------------------------------
  // Strategy B split: show cost of each sub-step independently
  // ---------------------------------------------------------------
  bench.run("B (step 1 only)    q_rel = conj(q_A)*q_B", [&] {
    const auto q_rel = conjugate(q_a) * q_b;
    ankerl::nanobench::doNotOptimizeAway(q_rel);
  });

  bench.run("B (step 2 only)    R = mat(q_rel) [from pre-stored quat]", [&] {
    const auto R = quaternion_to_rotation_matrix(q_a);
    ankerl::nanobench::doNotOptimizeAway(R);
  });
}

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    bench_relative_rotation();
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
