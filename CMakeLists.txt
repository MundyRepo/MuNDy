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

cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)

# Make CMake set WIN32 with CYGWIN for older CMake versions.  CMake requires
# this to be in the top-level CMakeLists.txt file and not an include file :-(
set(CMAKE_LEGACY_CYGWIN_WIN32
    1
    CACHE BOOL "" FORCE
)

# Get and declare PROJECT_NAME
include("${CMAKE_CURRENT_SOURCE_DIR}/ProjectName.cmake")
project(${PROJECT_NAME} NONE)

set(TRIBITS_HIDE_DEPRECATED_INCLUDE_DIRECTORIES_OVERRIDE TRUE)

# Pull in the TriBITS system
set(${PROJECT_NAME}_TRIBITS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TriBITS/tribits"
    CACHE STRING "TriBITS base directory (default assumes in TriBITS source tree)"
)
include("${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake")

# Pull in our custom TriBITS functions
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MundyTribitsInstallDirectories.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MundyTribitsAddLibrary.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MundyTribitsProject.cmake")

# Mundy only has one project so just enable it :-)
set(${PROJECT_NAME}_ENABLE_Mundy
    ON
    CACHE BOOL "" FORCE
)

# Define some global flags
set(${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT 20)
set(${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED "(20|23)")
advanced_set(
  CMAKE_CXX_STANDARD
  ${${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT}
  CACHE
  STRING
  "C++ standard number with values ${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED} (default ${${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT})"
)

if(NOT CMAKE_CXX_STANDARD MATCHES "^${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED}$")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} is not in the allowed set "
                      "${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED}"
  )
endif()

# Print the CMAKE_CXX_FLAGS
message(STATUS "CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}'")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG='${CMAKE_CXX_FLAGS_DEBUG}'")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE='${CMAKE_CXX_FLAGS_RELEASE}'")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO='${CMAKE_CXX_FLAGS_RELWITHDEBINFO}'")

# Default dependency logic for unit vs performance tests
set(Mundy_ENABLE_TESTS
    OFF
    CACHE BOOL "Enable tests in package Mundy"
)
set(Mundy_TEST_CATEGORIES
    "BASIC;CONTINUOUS;NIGHTLY;HEAVY"
    CACHE
      STRING
      "Semicolon-separated list of test categories to enable (CMake list). Valid categories are BASIC, CONTINUOUS, NIGHTLY, HEAVY, and PERFORMANCE."
)
if(Mundy_ENABLE_TESTS)
  message("")
  message("Mundy: Setting up default dependencies for tests since Mundy_ENABLE_TESTS=ON ...")
  message("")

  set(_mundy_has_performance_tests OFF)
  list(FIND Mundy_TEST_CATEGORIES "PERFORMANCE" _mundy_perf_idx)
  if(NOT _mundy_perf_idx EQUAL -1)
    set(_mundy_has_performance_tests ON)
  endif()

  set(_mundy_has_unit_like_tests OFF)
  foreach(_mundy_test_cat BASIC CONTINUOUS NIGHTLY HEAVY)
    list(FIND Mundy_TEST_CATEGORIES "${_mundy_test_cat}" _mundy_test_cat_idx)
    if(NOT _mundy_test_cat_idx EQUAL -1)
      set(_mundy_has_unit_like_tests ON)
      break()
    endif()
  endforeach()

  if(_mundy_has_performance_tests)
    message(STATUS
            "Enabling performance test defaults since Mundy_ENABLE_TESTS=ON and Mundy_TEST_CATEGORIES contains PERFORMANCE ...")

    if("${Mundy_ENABLE_nanobench}" STREQUAL "")
      set(Mundy_ENABLE_nanobench
          ON
          CACHE STRING "Enable support for the TPL nanobench in package Mundy." FORCE
      )
      message(
        STATUS
          "-- Setting Mundy_ENABLE_nanobench=ON since Mundy_ENABLE_TESTS=ON AND Mundy_TEST_CATEGORIES contains PERFORMANCE"
      )
    else()
      message(
        STATUS
          "-- NOT setting Mundy_ENABLE_nanobench=ON since it is already set to (${Mundy_ENABLE_nanobench}) on input by user"
      )
    endif()
  endif()

  if(_mundy_has_unit_like_tests)
    message(STATUS
            "Enabling unit-like test defaults since Mundy_ENABLE_TESTS=ON and Mundy_TEST_CATEGORIES contains BASIC, CONTINUOUS, NIGHTLY, or HEAVY ...")
    if("${Mundy_ENABLE_GTest}" STREQUAL "")
      set(Mundy_ENABLE_GTest
          ON
          CACHE STRING "Enable support for the TPL GTest in package Mundy." FORCE
      )
      message(
        STATUS
          "-- Setting Mundy_ENABLE_GTest=ON since Mundy_ENABLE_TESTS=ON AND Mundy_TEST_CATEGORIES contains BASIC, CONTINUOUS, NIGHTLY, or HEAVY"
      )
    else()
      message(
        STATUS
          "-- NOT setting Mundy_ENABLE_GTest=ON since it is already set to (${Mundy_ENABLE_GTest}) on input by user"
      )
    endif()
  endif()

  unset(_mundy_perf_idx)
  unset(_mundy_test_cat_idx)
  unset(_mundy_has_performance_tests)
  unset(_mundy_has_unit_like_tests)
  unset(_mundy_test_cat)
  unset(_mundy_perf_pkg)
  unset(_mundy_gtest_pkg)
else()
  message("")
  message("Mundy: Skipping default dependencies for tests since Mundy_ENABLE_TESTS=OFF ...")
  message("")
endif()

# Do all of the processing for this Tribits project
mundy_tribits_project()
