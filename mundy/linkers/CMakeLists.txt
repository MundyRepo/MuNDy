#
# A) Define the subpackage
#
tribits_subpackage(Linker)

# By default we enable all optional packages. They can be disabled by the user if they are not needed.
GLOBAL_SET(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find src/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort
SET(NOINSTALLHEADERS
  src/mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSphereLinker.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderLinker.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderSegmentLinker.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSegmentSpherocylinderSegmentLinker.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderLinker.hpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderSegmentLinker.hpp
  src/mundy_linkers/DestroyNeighborLinkers.hpp
  src/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyBoundNeighbors.hpp
  src/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.hpp
  src/mundy_linkers/EvaluateLinkerPotentials.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderSegmentHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentWCA.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderHertzianContact.hpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderSegmentHertzianContact.hpp
  src/mundy_linkers/GenerateNeighborLinkers.hpp
  src/mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.hpp
  src/mundy_linkers/LinkerPotentialForceReduction.hpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/Sphere.hpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/Spherocylinder.hpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/SpherocylinderSegment.hpp
  src/mundy_linkers/Linkers.hpp
  src/mundy_linkers/NeighborLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SphereSpherocylinderLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SphereSpherocylinderSegmentLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderLinkers.hpp
  src/mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderSegmentLinkers.hpp
  )

SET(SOURCES 
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSphereLinker.cpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderLinker.cpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSpherocylinderSegmentLinker.cpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSegmentSpherocylinderSegmentLinker.cpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderLinker.cpp
  src/mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderSegmentLinker.cpp
  src/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyBoundNeighbors.cpp
  src/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderSegmentHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentWCA.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderHertzianContact.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSpherocylinderSegmentHertzianContact.cpp
  src/mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.cpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/Sphere.cpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/Spherocylinder.cpp
  src/mundy_linkers/linker_potential_force_reduction/kernels/SpherocylinderSegment.cpp
  src/mundy_linkers/Linkers.cpp
)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_linkers
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_linkers
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()