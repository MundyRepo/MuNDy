#
# A) Define the subpackage
#
tribits_subpackage(Meta)

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
  src/mundy_meta/FieldReqsBase.hpp
  src/mundy_meta/FieldReqsFactory.hpp
  src/mundy_meta/FieldReqs.hpp
  src/mundy_meta/FieldReqsRegistry.hpp
  src/mundy_meta/HasMeshReqsAndIsRegisterable.hpp
  src/mundy_meta/MeshReqs.hpp
  src/mundy_meta/MetaFactory.hpp
  src/mundy_meta/MetaKernelDispatcher.hpp
  src/mundy_meta/MetaKernel.hpp
  src/mundy_meta/MetaMethodExecutionInterface.hpp
  src/mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp
  src/mundy_meta/MetaMethodSubsetExecutionInterface.hpp
  src/mundy_meta/MetaRegistry.hpp
  src/mundy_meta/MetaTechniqueDispatcher.hpp
  src/mundy_meta/ParameterValidationHelpers.hpp
  src/mundy_meta/PartReqs.hpp
  src/mundy_meta/utils/MeshGeneration.hpp
  )

SET(SOURCES 
  src/mundy_meta/PartReqs.cpp
  src/mundy_meta/MeshReqs.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_meta
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_meta
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()