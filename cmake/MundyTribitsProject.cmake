# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

# Local wrapper around tribits_project() so we can customize behavior without
# patching TriBITS directly.
#
# In this repository, the project name and a package name are both "Mundy".
# When that happens, TriBITS can emit two install rules targeting the same file:
#   <prefix>/lib/cmake/Mundy/MundyConfig.cmake
# If both are emitted, install order may overwrite the package-level config with
# the project-level config, which can recurse through components and fail.
#
# This wrapper sets:
#   <Project>_SKIP_INSTALL_PROJECT_CMAKE_CONFIG_FILES=ON
# by default when it detects a package with the same name as the project.
function(mundy_tribits_project)
  set(_packages_list_file "${CMAKE_CURRENT_SOURCE_DIR}/PackagesList.cmake")
  set(_project_name_matches_package OFF)

  if(EXISTS "${_packages_list_file}")
    file(STRINGS "${_packages_list_file}" _package_name_matches
      REGEX "^[ \t]*${PROJECT_NAME}[ \t]+")
    if(_package_name_matches)
      set(_project_name_matches_package ON)
    endif()
  endif()

  if(_project_name_matches_package)
    set(_skip_var "${PROJECT_NAME}_SKIP_INSTALL_PROJECT_CMAKE_CONFIG_FILES")
    set(_skip_default_var "${PROJECT_NAME}_SKIP_INSTALL_PROJECT_CMAKE_CONFIG_FILES_DEFAULT")
    set(_skip_doc
      "Skip installing project-level ${PROJECT_NAME}Config.cmake when project and package names collide.")

    if(NOT DEFINED ${_skip_var})
      set(${_skip_var} ON CACHE BOOL "${_skip_doc}")
      message(STATUS
        "${PROJECT_NAME}: Setting ${_skip_var}=ON because package name matches project name.")
    elseif(NOT ${${_skip_var}})
      message(WARNING
        "${PROJECT_NAME}: ${_skip_var}=OFF while package '${PROJECT_NAME}' is enabled. "
        "This can overwrite ${PROJECT_NAME}Config.cmake during install and trigger recursion.")
    endif()

    if(NOT DEFINED ${_skip_default_var})
      set(${_skip_default_var} ON CACHE BOOL "${_skip_doc}")
    endif()
  endif()

  tribits_project()
endfunction()

