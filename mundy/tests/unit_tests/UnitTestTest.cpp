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

//! \file UnitTestTest
/// Validate that tests are actually running as expected.
/// You may assume that simply because you ran
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize
#include <iostream>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

TEST(UnitTest, test) {
  std::cout << "EXAMPLE TEST PASSED" << std::endl;
  std::cout << "  If you every don't see this message. There's a problem." << std::endl;
}

int main(int argc, char** argv) {
  // Initialize MPI and Kokkos
  // Note, we mitigate our interaction with MPI through STK's stk::ParallelMachine.
  // If STK is MPI enabled, then we're MPI enabled. As such, Mundy doesn't directly depend on or interact with MPI.
  // However, if tests are to be run in parallel, then TPL_ENABLE_MPI must be set to ON in the TriBITS configuration.

  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  testing::InitGoogleMock(&argc, argv);
  int return_val = RUN_ALL_TESTS();

  std::cout << return_val << std::endl;

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return return_val;
}
