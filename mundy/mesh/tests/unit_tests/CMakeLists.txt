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
if (Mundy_ENABLE_TESTS AND Mundy_ENABLE_GTest)
  message(STATUS "MundyMesh: Setting up unit tests since Mundy_ENABLE_TESTS=ON and Mundy_ENABLE_GTest=ON")
  tribits_add_executable_and_test(
    UnitTestMeshBuilder
    SOURCES UnitTestMeshBuilder.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestBulkData
    SOURCES UnitTestBulkData.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestMetaData
    SOURCES UnitTestMetaData.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestAccessorExpr
    SOURCES UnitTestAccessorExpr.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestComponent
    SOURCES UnitTestComponent.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestComponentDeclaration
    SOURCES UnitTestComponentDeclaration.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestFieldDeclaration
    SOURCES UnitTestFieldDeclaration.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestRngAccessorExpr
    SOURCES UnitTestRngAccessorExpr.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  # tribits_add_executable_and_test(
  # UnitTestLinkData
  # SOURCES
  # UnitTestLinkData.cpp
  # ${Mundy_DEFAULT_UNIT_TEST_MAIN}
  # NUM_MPI_PROCS 1-4
  # )
  tribits_add_executable_and_test(
    UnitTestNgpLinkData
    SOURCES UnitTestNgpLinkData.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestStringToSelector
    SOURCES UnitTestStringToSelector.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestNgpModRequests
    SOURCES UnitTestNgpModRequests.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestDeclareEntities
    SOURCES UnitTestDeclareEntities.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestEntityPool
    SOURCES UnitTestEntityPool.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestFieldBLAS
    SOURCES UnitTestFieldBLAS.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    INSTALLABLE
  )

  tribits_add_executable_and_test(
    UnitTestAggregate
    SOURCES UnitTestAggregate.cpp ${Mundy_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1
  )

  # tribits_add_executable_and_test(
  # UnitTestGenNeighborLinks
  # SOURCES
  # UnitTestGenNeighborLinks.cpp
  # ${Mundy_DEFAULT_UNIT_TEST_MAIN}
  # NUM_MPI_PROCS 1-4
  # )
else()
  message(STATUS "MundyMesh: Skipping unit tests since Mundy_ENABLE_TESTS=OFF or Mundy_ENABLE_GTest=OFF ...")
endif()
