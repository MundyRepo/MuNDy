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

#define ANKERL_NANOBENCH_IMPLEMENT

#include "PerfTestSharedComponentsSupport.hpp"
#include "nanobench.h"

namespace mundy::mesh {

namespace {

void run_test() {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    std::cout << "PerfTestSharedComponents requires MPI size 1, skipping." << std::endl;
    return;
  }

  perf_test_shared_components::run_drag_benchmarks();
  perf_test_shared_components::run_body_mobility_benchmarks();
  perf_test_shared_components::run_complex_benchmarks();
}

}  // namespace

}  // namespace mundy::mesh

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  mundy::mesh::run_test();

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
