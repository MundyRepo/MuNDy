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

# This test had no asserts and takes minutes to run. It outputs convergence numbers which must be read and compared. 
# It is ideal for regression testing, not unit testing.
# tribits_add_executable_and_test(
#   UnitTestPeriphery
#   SOURCES
#   UnitTestPeriphery.cpp
#     ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
#   NUM_MPI_PROCS 1
#   CATEGORIES HEAVY
# )

# tribits_copy_files_to_binary_dir(
#     UnitTestPeripheryCopyFiles
#     SOURCE_FILES
#     sphere_triangle_points_1280.dat
#     sphere_triangle_weights_1280.dat
#     sphere_triangle_normals_1280.dat
#     sphere_triangle_points_3840.dat
#     sphere_triangle_weights_3840.dat
#     sphere_triangle_normals_3840.dat
#     sphere_triangle_points_5120.dat
#     sphere_triangle_weights_5120.dat
#     sphere_triangle_normals_5120.dat
#     EXEDEPS UnitTestPeriphery
#     CATEGORIES HEAVY
#     )
