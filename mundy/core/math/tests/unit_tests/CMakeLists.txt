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
if(Mundy_ENABLE_TESTS AND Mundy_ENABLE_GTest)
  message(STATUS "MundyMath: Setting up unit tests since Mundy_ENABLE_TESTS=ON and Mundy_ENABLE_GTest=ON")
  tribits_add_executable_and_test(
    UnitTestViews
    SOURCES UnitTestViews.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestVector
    SOURCES UnitTestVector.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestMatrix
    SOURCES UnitTestMatrix.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestVector3
    SOURCES UnitTestVector3.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestMatrix3
    SOURCES UnitTestMatrix3.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestQuaternion
    SOURCES UnitTestQuaternion.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestMinimize
    SOURCES UnitTestMinimize.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestHilbert
    SOURCES UnitTestHilbert.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestZMorton
    SOURCES UnitTestZMorton.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestConvex
    SOURCES UnitTestConvex.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1
    INSTALLABLE
  )
else()
  message(STATUS "MundyMath: Skipping unit tests since Mundy_ENABLE_TESTS=OFF or Mundy_ENABLE_GTest=OFF ...")
endif()
