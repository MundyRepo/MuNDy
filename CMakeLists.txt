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
set(CMAKE_LEGACY_CYGWIN_WIN32 1 CACHE BOOL "" FORCE)

# Get and declare PROJECT_NAME
include("${CMAKE_CURRENT_SOURCE_DIR}/ProjectName.cmake")
project(${PROJECT_NAME} NONE)

set(TRIBITS_HIDE_DEPRECATED_INCLUDE_DIRECTORIES_OVERRIDE  TRUE)

# Pull in the TriBITS system
set(${PROJECT_NAME}_TRIBITS_DIR
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TriBITS/tribits"  CACHE  STRING
  "TriBITS base directory (default assumes in TriBITS source tree)")
include("${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake")

# Pull in our custom TriBITS functions
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MundyTribitsInstallDirectories.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MundyTribitsAddLibrary.cmake")

# Mundy only has one project so just enable it :-)
set(${PROJECT_NAME}_ENABLE_Mundy ON CACHE BOOL "" FORCE)

# Define some global flags
set(${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT 20)
set(${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED "(20|23)")
advanced_set(CMAKE_CXX_STANDARD ${${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT}
  CACHE STRING
  "C++ standard number with values ${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED} (default ${${PROJECT_NAME}_CMAKE_CXX_STANDARD_DEFAULT})")
if (NOT CMAKE_CXX_STANDARD MATCHES "^${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED}$")
  MESSAGE(FATAL_ERROR
    "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} is not in the allowed set "
    "${${PROJECT_NAME}_CMAKE_CXX_STANDARDS_ALLOWED}")
endif ()

# Print the CMAKE_CXX_FLAGS
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

# Do all of the processing for this Tribits project
tribits_project()
