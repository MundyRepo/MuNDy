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
#include <mundy_geom/periodicity.hpp>  // for mundy::OrthorhombicMetric, mundy::TriclinicMetric
#include <mundy_math/Array.hpp>        // for mundy::Array
#include <mundy_math/Matrix.hpp>       // for mundy::Matrix
#include <mundy_math/Quaternion.hpp>   // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>    // for mundy::get_relaxed_tolerance
#include <mundy_math/Vector.hpp>       // for mundy::Vector

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

  // OrthorhombicMetric uses element-wise multiply/divide (diagonal cell, no matrix ops).
  // TriclinicMetric uses full matrix multiply (general tilted cell).
  mundy::OrthorhombicMetric<mundy::AXIS_XYZ, double> ortho_metric{cell_size};
  mundy::TriclinicMetric<mundy::AXIS_XYZ, double>    tri_metric{mundy::Matrix3<double>::diagonal(cell_size)};

  bench.run("OrthorhombicMetric sep | No Loops", [&] {
    auto sep = ortho_metric.sep(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
  bench.run("TriclinicMetric sep | No Loops (diagonal cell)", [&] {
    auto sep = tri_metric.sep(point1, point2);
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

  bench.run("OrthorhombicMetric construct+sep | No Loops", [&] {
    mundy::OrthorhombicMetric<mundy::AXIS_XYZ, double> m{cell_size};
    auto sep = m.sep(point1, point2);
    ankerl::nanobench::doNotOptimizeAway(sep);
  });
  bench.run("TriclinicMetric construct+sep | No Loops (diagonal cell)", [&] {
    mundy::TriclinicMetric<mundy::AXIS_XYZ, double> m{mundy::Matrix3<double>::diagonal(cell_size)};
    auto sep = m.sep(point1, point2);
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
