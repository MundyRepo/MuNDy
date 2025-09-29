# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
# Developed under support from the NSF Graduate Research Fellowship Program. ft. Chris Edelmaier
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
    UnitTestSegmentSegment
    SOURCES
      UnitTestSegmentSegment.cpp
      ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
    )

tribits_add_executable_and_test(
  UnitTestEllipsoidEllipsoid
  SOURCES
  UnitTestEllipsoidEllipsoid.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  )

tribits_add_executable_and_test(
  UnitTestComputeAABB
  SOURCES
  UnitTestComputeAABB.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  )

tribits_add_executable_and_test(
  UnitTestComputeBoundingRadius
  SOURCES
  UnitTestComputeBoundingRadius.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  )

tribits_add_executable_and_test(
  UnitTestPeriodicity
  SOURCES
  UnitTestPeriodicity.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  )

  