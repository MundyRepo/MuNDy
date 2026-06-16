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
    "MundyGeom: Setting up performance tests since Mundy_ENABLE_TESTS=ON and Mundy_ENABLE_nanobench=ON")

  tribits_add_executable(
    PerfTestGeomMetric
    SOURCES PerfTestGeomMetric.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )
  tribits_add_executable(
    PerfTestOBBRelativeRotation
    SOURCES PerfTestOBBRelativeRotation.cpp
    INSTALLABLE
    CATEGORIES PERFORMANCE
  )
else()
  message(STATUS
    "MundyGeom: Skipping performance tests since Mundy_ENABLE_TESTS=OFF or Mundy_ENABLE_nanobench=OFF ...")
endif()
