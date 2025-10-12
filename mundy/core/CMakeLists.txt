#
# A) Define the subpackage
#
tribits_subpackage(Core)

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
  src/mundy_core/aggregate.hpp
  src/mundy_core/attribute_unused.hpp
  src/mundy_core/MakeStringArray.hpp
  src/mundy_core/NgpPool.hpp
  src/mundy_core/NgpView.hpp
  src/mundy_core/OurAnyNumberParameterEntryValidator.hpp
  src/mundy_core/StringLiteral.hpp
  src/mundy_core/throw_assert.hpp
  src/mundy_core/tuple.hpp
  )

SET(SOURCES 
  src/mundy_core/OurAnyNumberParameterEntryValidator.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_core
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_core
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()