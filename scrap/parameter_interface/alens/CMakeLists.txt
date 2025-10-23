
#
# A) Define the subpackage
#
tribits_subpackage(Alens)

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
  src/mundy_alens/actions_crosslinkers.hpp
  src/mundy_alens/ComputeBrownianVelocity.hpp
  src/mundy_alens/compute_brownian_velocity/kernels/SpheresKernel.hpp
  src/mundy_alens/ComputeMobility.hpp
  src/mundy_alens/compute_mobility/techniques/LocalDragNonOrientableSpheres.hpp
  src/mundy_alens/compute_mobility/techniques/RPYSpheres.hpp
  src/mundy_alens/compute_mobility/techniques/RPYSpheresPerf.hpp
  src/mundy_alens/periphery/Gauss_Legendre_Nodes_and_Weights.hpp
  src/mundy_alens/periphery/Periphery.hpp
  )

SET(SOURCES 
  src/mundy_alens/compute_brownian_velocity/kernels/SpheresKernel.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_alens
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_alens
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()
