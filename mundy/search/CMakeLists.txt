#
# A) Define the subpackage
#
tribits_subpackage(Search)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use
# find src/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort
set(NOINSTALLHEADERS
    src/mundy_search/ArborX1dNeighborList.hpp
    src/mundy_search/ArborX2dNeighborList.hpp
    src/mundy_search/ArborXNeighborList.hpp
    src/mundy_search/Excluder.hpp
    src/mundy_search/ForEach.hpp
    src/mundy_search/impl/ArborXCallback.hpp
    src/mundy_search/impl/ArborXSearchBoxes.hpp
    src/mundy_search/impl/NeighborListFunctors.hpp
    src/mundy_search/impl/STKSearchBoxes.hpp
    src/mundy_search/NeighborList.hpp
    src/mundy_search/NeighborListBuilder.hpp
    src/mundy_search/NeighborListBuildTraits.hpp
    src/mundy_search/Neighbors.hpp
    src/mundy_search/SearchCandidate.hpp
    src/mundy_search/STKSearchNeighborList.hpp
)

set(SOURCES
    src/mundy_math/a_necessary_empty_file.cpp
)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_search
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_search
  FILES_MATCHING
  PATTERN "*.hpp"
  SOURCES ${SOURCES}
)

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()
