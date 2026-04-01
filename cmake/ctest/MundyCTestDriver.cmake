#
# Common MuNDy CTest/TriBITS driver setup.
#

set(TRIBITS_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(CTEST_SOURCE_NAME "MuNDy")

include("${TRIBITS_PROJECT_ROOT}/ProjectName.cmake")

if(NOT "$ENV{${PROJECT_NAME}_TRIBITS_DIR}" STREQUAL "")
  set(${PROJECT_NAME}_TRIBITS_DIR "$ENV{${PROJECT_NAME}_TRIBITS_DIR}")
endif()

if("${${PROJECT_NAME}_TRIBITS_DIR}" STREQUAL "")
  set(${PROJECT_NAME}_TRIBITS_DIR "${TRIBITS_PROJECT_ROOT}/cmake/TriBITS/tribits")
endif()

include("${${PROJECT_NAME}_TRIBITS_DIR}/ctest_driver/TribitsCTestDriverCore.cmake")

function(mundy_ctest_driver)
  set_default_and_from_env(CTEST_BUILD_FLAGS "-j1")
  set_default_and_from_env(CTEST_PARALLEL_LEVEL "1")
  tribits_ctest_driver()
endfunction()
