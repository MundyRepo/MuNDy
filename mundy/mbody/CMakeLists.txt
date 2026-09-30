#
# A) Define the subpackage
#
tribits_subpackage(MBody)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use
# find src/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort
set(NOINSTALLHEADERS
    src/mundy_mbody/KokkosMbody.hpp
    src/mundy_mbody/KokkosMbodyImpl.hpp
    src/mundy_mbody/KokkosMbodyTypes.hpp
)

set(SOURCES
   src/mundy_mbody/a_necessary_empty_file.cpp
)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
    mundy_mbody
    HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
    NOINSTALLHEADERS ${NOINSTALLHEADERS}
    DIRECTORIES src/mundy_mbody
    FILES_MATCHING
    PATTERN "*.hpp"
    SOURCES ${SOURCES}
)

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()
