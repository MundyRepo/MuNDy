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

//! \file PerfTestGeomMetric.cpp
/// \brief Performance test metrics for geometric distance calculations.

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <fstream>    // for std::ofstream
#include <iostream>   // for std::cout, std::endl
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// External
#include <nanobench.h>

// Trilinos
#include <Kokkos_Core.hpp>                 // for Kokkos::initialize, Kokkos::finalize
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_geom/distance/DistanceMetrics.hpp>     // for mundy::FreeSpaceMetric
#include <mundy_math/Array.hpp>                        // for mundy::Array
#include <mundy_math/Matrix.hpp>                       // for mundy::Matrix
#include <mundy_math/Quaternion.hpp>                   // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>                    // for mundy::get_relaxed_tolerance
#include <mundy_math/Vector.hpp>                       // for mundy::Vector

mundy::Vector3d random_vector() {
  return mundy::Vector3d{static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX,
                         static_cast<double>(rand()) / RAND_MAX};
}

void speed_test() {
  ankerl::nanobench::Bench bench;
  bench.title("Speed")
      .unit("op")
      .relative(true)
      .performanceCounters(true)
      .warmup(500)                                   // a few warmup runs
      .epochs(500)                                   // multiple independent epochs
      .minEpochTime(std::chrono::microseconds(200))  // run long enough per epoch
      .minEpochIterations(100000);

  mundy::Vector3d cell_size = random_vector();
  mundy::Vector3d point1 = random_vector();
  mundy::Vector3d point2 = random_vector();

  // New constructor
  auto new_periodic_space_metric = mundy::periodic_metric_from_unit_cell(cell_size);
  auto new_periodic_scaled_space_metric = mundy::periodic_scaled_metric_from_unit_cell(cell_size);

  bench.run("New Periodic Metric | No Loops", [&] {
    auto sep = new_periodic_space_metric(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
  bench.run("New Periodic Metric | No Loops | Scale only", [&] {
    auto sep = new_periodic_scaled_space_metric(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
}

void construction_test() {
  ankerl::nanobench::Bench bench;
  bench.title("Construction")
      .unit("op")
      .relative(true)
      .performanceCounters(true)
      .warmup(500)                                   // a few warmup runs
      .epochs(500)                                   // multiple independent epochs
      .minEpochTime(std::chrono::microseconds(200))  // run long enough per epoch
      .minEpochIterations(100000);

  mundy::Vector3d cell_size = random_vector();
  mundy::Vector3d point1 = random_vector();
  mundy::Vector3d point2 = random_vector();

  bench.run("New Periodic Metric | No Loops", [&] {
    // New constructor
    auto new_periodic_space_metric = mundy::periodic_metric_from_unit_cell(cell_size);
    auto sep = new_periodic_space_metric(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
  bench.run("New Periodic Metric | No Loops | Scale only", [&] {
    // New constructor | scale only
    auto new_periodic_scaled_space_metric = mundy::periodic_scaled_metric_from_unit_cell(cell_size);
    auto sep = new_periodic_scaled_space_metric(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
}

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    speed_test();
    construction_test();
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
