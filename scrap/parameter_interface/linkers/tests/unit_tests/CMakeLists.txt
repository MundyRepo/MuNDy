# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
# Developed under support from the NSF Graduate Research Fellowship Program.
#
# Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

# Here, we attempt to organize unit tests logically by dependencies (not by spelling). Simple classes are tested first 
# and more complex classes are tested later. This is not a hard and fast rule, but it is a good guideline to follow, 
# as it produces more logical error messages when a test fails.
tribits_add_executable_and_test(
  UnitTestLinkers
  SOURCES
    UnitTestLinkers.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 2
)

tribits_add_executable_and_test(
  UnitTestGenerateNeighborLinkers
  SOURCES
    UnitTestGenerateNeighborLinkers.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestDestroyNeighborLinkers
  SOURCES
    UnitTestDestroyNeighborLinkers.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestComputeSignedSeparationDistanceAndContactNormal
  SOURCES
    UnitTestComputeSignedSeparationDistanceAndContactNormal.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 2
)

tribits_add_executable_and_test(
  UnitTestEvaluateLinkerPotentials
  SOURCES
    UnitTestEvaluateLinkerPotentials.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 2
)

tribits_add_executable_and_test(
  UnitTestLinkerPotentialForceReduction
  SOURCES
    UnitTestLinkerPotentialForceReduction.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)
