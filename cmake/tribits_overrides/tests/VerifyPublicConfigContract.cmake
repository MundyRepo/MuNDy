cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED PROJECT_BINARY_DIR)
  message(FATAL_ERROR "PROJECT_BINARY_DIR is required")
endif()
if(NOT DEFINED PROJECT_NAME)
  message(FATAL_ERROR "PROJECT_NAME is required")
endif()

function(assert_file_contains file_path needle)
  if(NOT EXISTS "${file_path}")
    message(FATAL_ERROR "Expected file does not exist: ${file_path}")
  endif()
  file(READ "${file_path}" file_contents)
  string(FIND "${file_contents}" "${needle}" needle_index)
  if(needle_index EQUAL -1)
    message(FATAL_ERROR "Expected to find '${needle}' in ${file_path}")
  endif()
endfunction()

function(assert_file_not_contains file_path needle)
  if(NOT EXISTS "${file_path}")
    message(FATAL_ERROR "Expected file does not exist: ${file_path}")
  endif()
  file(READ "${file_path}" file_contents)
  string(FIND "${file_contents}" "${needle}" needle_index)
  if(NOT needle_index EQUAL -1)
    message(FATAL_ERROR "Did not expect to find '${needle}' in ${file_path}")
  endif()
endfunction()

set(project_install_config "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake")
set(private_package_install_config "${PROJECT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config_install.cmake")

file(GLOB_RECURSE mundy_math_install_configs LIST_DIRECTORIES FALSE
  "${PROJECT_BINARY_DIR}/*/MundyMathConfig_install.cmake")
list(LENGTH mundy_math_install_configs mundy_math_config_count)
if(mundy_math_config_count EQUAL 0)
  message(FATAL_ERROR "Could not locate MundyMathConfig_install.cmake under ${PROJECT_BINARY_DIR}")
endif()
list(GET mundy_math_install_configs 0 mundy_math_install_config)

# Verify Mundy_config.hpp is scheduled for installation.
# The Mundy top-level package is processed from the build root, so
# tribits_install_headers() registers the header in the root cmake_install.cmake.
# If it is absent the header was never wired into the install tree.
set(root_cmake_install "${PROJECT_BINARY_DIR}/cmake_install.cmake")
assert_file_contains("${root_cmake_install}" "Mundy_config.hpp")

assert_file_contains("${project_install_config}" "MundyPackageConfig.cmake")
assert_file_contains("${project_install_config}" "set(compVarPrefix \"MundyPackage\")")
assert_file_contains("${project_install_config}" "list(APPEND Mundy_LIBRARIES")
assert_file_contains("${project_install_config}" "add_library(Mundy::all_libs IMPORTED INTERFACE GLOBAL)")

assert_file_contains("${private_package_install_config}" "set(MundyPackage_LIBRARIES Mundy::all_libs)")
assert_file_contains("${private_package_install_config}" "MundyPackage_LIBRARIES")
assert_file_not_contains("${private_package_install_config}" "set(Mundy_LIBRARIES Mundy::all_libs)")

assert_file_contains("${mundy_math_install_config}" "set(MundyMath_LIBRARIES MundyMath::all_libs)")
assert_file_not_contains("${mundy_math_install_config}" "MundyPackage_")
