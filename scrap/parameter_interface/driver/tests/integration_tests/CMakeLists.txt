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

tribits_add_executable_and_test(
    IntegrationTestConfigureDriver
    SOURCES
      IntegrationTestConfigureDriver.cpp
      ${MUNDY_DEFAULT_INTEGRATION_TEST_MAIN}
    NUM_MPI_PROCS 1
    CATEGORIES BASIC
    )

# Copy the test files for the executables...
tribits_copy_files_to_binary_dir(
    CopyTestConfigureDriver
    SOURCE_FILES
    integration_test_configuredriver_basic.yaml
    EXEDEPS IntegrationTestConfigureDriver
    CATEGORIES BASIC
    )
