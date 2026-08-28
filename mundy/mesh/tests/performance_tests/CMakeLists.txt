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

if(Mundy_ENABLE_TESTS AND Mundy_ENABLE_nanobench)
  message(STATUS
    "MundyMesh: Setting up performance tests since Mundy_ENABLE_TESTS=ON and Mundy_ENABLE_nanobench=ON")

  tribits_add_executable(
    PerfTestDeclareEntities
    SOURCES PerfTestDeclareEntities.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )
  tribits_copy_files_to_binary_dir(
    PerfTestDeclareEntitiesCopyFiles
    SOURCE_FILES
    2013_03_06_cell08_876_H41_05_Cell2_cell-axon.ply
    bee_hive.ply
    cube.ply
    ellipsoid.ply
    sphere.ply
    EXEDEPS PerfTestDeclareEntities
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestAccessors
    SOURCES PerfTestAccessors.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestSharedComponents
    SOURCES
    PerfTestSharedComponents.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestAggregates
    SOURCES PerfTestAggregates.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestNgpAggregates
    SOURCES PerfTestNgpAggregates.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestNgpLinkData
    SOURCES PerfTestNgpLinkData.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  tribits_add_executable(
    PerfTestNgpModRequests
    SOURCES PerfTestNgpModRequests.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

  # These tests require Trilinos 16.0.0 or later
  # They still get built but will simply print a warning and return
  tribits_add_executable(
    PerfTestFieldBLAS
    SOURCES PerfTestFieldBLAS.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )

# tribits_add_executable(
# KokkosMbodyTest
# SOURCES KokkosMbody.cpp
# INSTALLABLE
# CATEGORIES PERFORMANCE
# )
else()
  message(STATUS
    "MundyMesh: Skipping performance tests since Mundy_ENABLE_TESTS=OFF or Mundy_ENABLE_nanobench=OFF ...")
endif()
