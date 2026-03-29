# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

# Local entry point for project configuration.
#
# We still call the upstream tribits_project_impl() macro, but we make the
# override design explicit here:
# - the same-name TriBITS modules in ${PROJECT_SOURCE_DIR}/cmake are thin shims
# - those shims delegate into ${PROJECT_SOURCE_DIR}/cmake/tribits_overrides
# - MundyPackageConfig.cmake and MundyPackage_* are private implementation
#   details used only to avoid the self-name collision between the Mundy
#   project and the Mundy package
#
# Public API preserved for downstream users:
# - find_package(Mundy)
# - find_package(MundyUtils), find_package(MundyMath), find_package(MundyMesh), ...
# - link with Mundy_LIBRARIES / Mundy::all_libs / MundyMath_LIBRARIES / ...
set(MUNDY_TRIBITS_OVERRIDES_DIR
  "${PROJECT_SOURCE_DIR}/cmake/tribits_overrides")


function(mundy_tribits_register_public_config_contract_test)
  if (NOT DEFINED BUILD_TESTING OR NOT BUILD_TESTING)
    return()
  endif()

  # Match the same effective prerequisites that TriBITS uses before it writes
  # the project-level install config files. If those files are not being
  # generated, this contract test has nothing meaningful to validate.
  if (NOT ${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES)
    return()
  endif()
  if (${PROJECT_NAME}_ENABLE_INSTALLATION_TESTING)
    return()
  endif()
  if (${PROJECT_NAME}_SKIP_INSTALL_PROJECT_CMAKE_CONFIG_FILES)
    return()
  endif()

  set(contractTestScript
    "${MUNDY_TRIBITS_OVERRIDES_DIR}/tests/VerifyPublicConfigContract.cmake")
  if (NOT EXISTS "${contractTestScript}")
    return()
  endif()

  add_test(
    NAME MundyPublicConfigContract
    COMMAND ${CMAKE_COMMAND}
      -DPROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}
      -DPROJECT_NAME=${PROJECT_NAME}
      -P "${contractTestScript}"
    )
endfunction()


macro(mundy_tribits_project)
  tribits_project_impl(${ARGN})
  mundy_tribits_register_public_config_contract_test()
endmacro()
