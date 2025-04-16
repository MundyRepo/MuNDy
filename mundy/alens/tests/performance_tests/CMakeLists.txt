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

# tribits_add_executable(
#     PerformanceTestFilament
#     SOURCES
#     Filament.cpp
#     CATEGORIES PERFORMANCE
#     )
# tribits_copy_files_to_binary_dir(
#     PerformanceTestFilamentCopyFiles
#     SOURCE_FILES
#     filament_params.yaml
#     EXEDEPS PerformanceTestFilament
#     )

# tribits_add_executable(
#     PerformanceTestBacteria
#     SOURCES
#     Bacteria.cpp
#     CATEGORIES PERFORMANCE
#     )
# tribits_copy_files_to_binary_dir(
#     PerformanceTestBacteriaCopyFiles
#     SOURCE_FILES
#     bacteria_params.yaml
#     EXEDEPS PerformanceTestBacteria
#     )

tribits_add_executable(
    PerformanceTestBrownianMotion
    SOURCES
    BrownianMotion.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    PerformanceTestBrownianMotionMSD
    SOURCES
    BrownianMotionMSD.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    PerformanceTestSphereBrownianMotionWithContactNew
    SOURCES
    SphereBrownianMotionWithContactNew.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    PerformanceTestSphereBrownianInteractingIO
    SOURCES
    SphereBrownianInteractingIO.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    Springs
    SOURCES
    Springs.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    SpringsUpdated
    SOURCES
    SpringsUpdated.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    StickySettings
    SOURCES
    StickySettings.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    HP1
    SOURCES
    HP1.cpp
    CATEGORIES PERFORMANCE
    )
tribits_copy_files_to_binary_dir(
    HP1CopyFiles
    SOURCE_FILES
    hp1.yaml
    hp1_periphery_quadrature_normals.dat
    hp1_periphery_quadrature_points.dat
    hp1_periphery_quadrature_weights.dat
    hp1_pos.dat
    hp1_hilbert_centers.dat
    EXEDEPS HP1
    )

tribits_add_executable(
    NonInteractingSperm
    SOURCES
    NonInteractingSperm.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    CollidingSperm
    SOURCES
    CollidingSperm.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    CollidingFrictionalSperm
    SOURCES
    CollidingFrictionalSperm.cpp
    CATEGORIES PERFORMANCE
    )

tribits_add_executable(
    CollidingOverdampedFrictionalSperm
    SOURCES
    CollidingOverdampedFrictionalSperm.cpp
    CATEGORIES PERFORMANCE
    )

# tribits_add_executable(
#     PeriodicCollidingOverdampedFrictionalSperm
#     SOURCES
#     PeriodicCollidingOverdampedFrictionalSperm.cpp
#     CATEGORIES PERFORMANCE
#     )

tribits_add_executable(
    SpheresPerif
    SOURCES
    SpheresPerif.cpp
    CATEGORIES PERFORMANCE
    )
      
# tribits_add_executable(
#     NgpHP1
#     SOURCES
#     NgpHP1.cpp
#     CATEGORIES PERFORMANCE
#     )
# tribits_copy_files_to_binary_dir(
#     NgpHP1CopyFiles
#     SOURCE_FILES
#     ngp_hp1.yaml
#     hp1_periphery_quadrature_normals.dat
#     hp1_periphery_quadrature_points.dat
#     hp1_periphery_quadrature_weights.dat
#     hp1_pos.dat
#     EXEDEPS NgpHP1
#     )

tribits_add_executable(
    NGPSabastion
    SOURCES
    NGPSabastion.cpp
    CATEGORIES PERFORMANCE
    )
