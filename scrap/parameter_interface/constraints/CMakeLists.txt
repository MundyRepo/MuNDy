#
# A) Define the subpackage
#
tribits_subpackage(Constraint)

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
  src/mundy_constraints/AngularSprings.hpp
  src/mundy_constraints/ComputeConstraintForcing.hpp
  src/mundy_constraints/compute_constraint_forcing/kernels/AngularSpringsKernel.hpp
  src/mundy_constraints/compute_constraint_forcing/kernels/FENESpringsKernel.hpp
  src/mundy_constraints/compute_constraint_forcing/kernels/FENEWCASpringsKernel.hpp
  src/mundy_constraints/compute_constraint_forcing/kernels/HookeanSpringsKernel.hpp
  src/mundy_constraints/Constraints.hpp
  src/mundy_constraints/DeclareAndInitConstraints.hpp
  src/mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp
  src/mundy_constraints/declare_and_initialize_constraints/techniques/ChainOfSprings.hpp
  src/mundy_constraints/FENESprings.hpp
  src/mundy_constraints/FENEWCASprings.hpp
  src/mundy_constraints/HookeanSprings.hpp
  )

SET(SOURCES 
  src/mundy_constraints/compute_constraint_forcing/kernels/AngularSpringsKernel.cpp
  src/mundy_constraints/compute_constraint_forcing/kernels/FENESpringsKernel.cpp
  src/mundy_constraints/compute_constraint_forcing/kernels/FENEWCASpringsKernel.cpp
  src/mundy_constraints/compute_constraint_forcing/kernels/HookeanSpringsKernel.cpp
  src/mundy_constraints/declare_and_initialize_constraints/techniques/ChainOfSprings.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_constraints
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_constraints
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()