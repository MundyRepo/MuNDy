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

function(mundy_add_core_tutorial exe_name source_file test_name)
  tribits_add_executable_and_test(
    ${exe_name}
    NOEXEPREFIX
    SOURCES ${source_file}
    NAME ${test_name}
    NUM_MPI_PROCS 1
    CATEGORIES BASIC
    TIMEOUT 30
    INSTALLABLE
  )
endfunction()

add_subdirectory(01_hello_world)
add_subdirectory(02_utilities)
add_subdirectory(03_vectors)
add_subdirectory(04_matrices)
add_subdirectory(05_quaternions)
add_subdirectory(06_views)
add_subdirectory(07_primitives)
add_subdirectory(08_distance)
add_subdirectory(09_periodicity)
add_subdirectory(10_ngp_view)
