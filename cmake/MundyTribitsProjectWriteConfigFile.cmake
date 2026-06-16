# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

include_guard(GLOBAL)

# Local override for the TriBITS project-config writer. This file is reached
# through the same-name shim in ${PROJECT_SOURCE_DIR}/cmake/ so TriBITS resolves
# it before the upstream module.
#
# Public contract preserved by this override layer:
# - find_package(Mundy) stays the top-level public entry point.
# - the self-named package remains private and is loaded only through the public
#   project config using MundyPackageConfig.cmake.
include("${${PROJECT_NAME}_TRIBITS_DIR}/core/package_arch/TribitsProjectWriteConfigFile.cmake")


# @FUNCTION: mundy_tribits_get_install_project_config_template()
#
# Returns the install-tree project-config template. Mundy uses an explicit local
# template so the self-named top-level package can remain private while the
# public project config continues to export the usual Mundy_* API.
function(mundy_tribits_get_install_project_config_template outputVar)
  set(projectConfigTemplate
    "${PROJECT_SOURCE_DIR}/cmake/MundyTribitsProjectConfigTemplate.cmake.in")
  if (NOT EXISTS "${projectConfigTemplate}")
    set(projectConfigTemplate
      "${${PROJECT_NAME}_TRIBITS_DIR}/${TRIBITS_CMAKE_INSTALLATION_FILES_DIR}/TribitsProjectConfigTemplate.cmake.in")
  endif()

  set(${outputVar} "${projectConfigTemplate}" PARENT_SCOPE)
endfunction()


# @FUNCTION: tribits_write_project_client_export_files()
#
# Local override of the upstream TriBITS function. The only behavioral delta is
# that the install-tree project config is generated from Mundy's explicit local
# template instead of the upstream template.
function(tribits_write_project_client_export_files)

  set(EXPORT_FILE_VAR_PREFIX ${PROJECT_NAME})

  set(PACKAGE_LIST ${${PROJECT_NAME}_DEFINED_INTERNAL_PACKAGES})
  if (PACKAGE_LIST)
    list(REVERSE PACKAGE_LIST)
  endif()

  set(FULL_PACKAGE_SET "")
  set(FULL_LIBRARY_SET "")
  foreach(TRIBITS_PACKAGE ${PACKAGE_LIST})
    if(${PROJECT_NAME}_ENABLE_${TRIBITS_PACKAGE})
      list(APPEND FULL_PACKAGE_SET ${TRIBITS_PACKAGE})
      list(APPEND FULL_LIBRARY_SET ${${TRIBITS_PACKAGE}_LIBRARIES})
    endif()
  endforeach()

  set(${PROJECT_NAME}_CONFIG_LIBRARIES ${FULL_LIBRARY_SET})

  if (${PROJECT_NAME}_DEFINED_TPLS)
    set(TPL_LIST ${${PROJECT_NAME}_DEFINED_TPLS})
    list(REVERSE TPL_LIST)
  endif()

  set(FULL_TPL_SET "")
  set(FULL_TPL_LIBRARY_SET "")
  foreach(TPL ${TPL_LIST})
    if(TPL_ENABLE_${TPL})
      list(APPEND FULL_TPL_SET ${TPL})
      list(APPEND FULL_TPL_LIBRARY_SET ${TPL_${TPL}_LIBRARIES})
    endif()
  endforeach()

  set(${PROJECT_NAME}_CONFIG_TPL_LIBRARIES ${FULL_TPL_LIBRARY_SET})

  set(DISCOURAGE_EDITING "Do not edit: This file was generated automatically by CMake.")

  if(BUILD_SHARED_LIBS)
    string(REPLACE ";" ":" SHARED_LIB_RPATH_COMMAND
     "${${PROJECT_NAME}_CONFIG_LIBRARY_DIRS}")
    set(SHARED_LIB_RPATH_COMMAND ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG}${SHARED_LIB_RPATH_COMMAND})
  endif()

  set(PROJECT_CONFIG_CODE "")

  set(LOAD_CODE "# Load configurations from enabled packages")
  foreach(TRIBITS_PACKAGE ${FULL_PACKAGE_SET})
    set(LOAD_CODE "${LOAD_CODE}
include(\"${${TRIBITS_PACKAGE}_BINARY_DIR}/${TRIBITS_PACKAGE}Config.cmake\")")
  endforeach()
  set(PROJECT_CONFIG_CODE "${PROJECT_CONFIG_CODE}\n${LOAD_CODE}")

  tribits_set_compiler_vars_for_config_file(INSTALL_DIR)

  set(upstreamProjectConfigTemplate
    "${${PROJECT_NAME}_TRIBITS_DIR}/${TRIBITS_CMAKE_INSTALLATION_FILES_DIR}/TribitsProjectConfigTemplate.cmake.in")
  mundy_tribits_get_install_project_config_template(localInstallProjectConfigTemplate)

  if (${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES)
    set(PDOLLAR "$")
    set(TRIBITS_PROJECT_INSTALL_INCLUDE_DIR "")
    configure_file(
      "${upstreamProjectConfigTemplate}"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" )
  endif()

  string(REPLACE "/" ";" PATH_LIST ${${PROJECT_NAME}_INSTALL_LIB_DIR})
  set(RELATIVE_PATH "../..")
  foreach(PATH ${PATH_LIST})
    set(RELATIVE_PATH "${RELATIVE_PATH}/..")
  endforeach()

  if(BUILD_SHARED_LIBS)
    set(SHARED_LIB_RPATH_COMMAND
       "${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG}${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_LIB_DIR}"
      )
  endif()

  if (${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES)

    tribits_set_compiler_vars_for_config_file(INSTALL_DIR)

    set(PROJECT_CONFIG_CODE "")

    set(PDOLLAR "$")

    if (IS_ABSOLUTE "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}")
      set(TRIBITS_PROJECT_INSTALL_INCLUDE_DIR "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}")
    else()
      set(TRIBITS_PROJECT_INSTALL_INCLUDE_DIR
        "${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}")
    endif()

    configure_file(
      "${localInstallProjectConfigTemplate}"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake"
      )

    install(
      FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake"
      DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/${PROJECT_NAME}"
      RENAME ${PROJECT_NAME}Config.cmake
      )
  endif()

  include(CMakePackageConfigHelpers)
  if ("${${PROJECT_NAME}_VERSION}" STREQUAL "")
    set(${PROJECT_NAME}_VERSION 0.0.0)
  endif()
  write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${${PROJECT_NAME}_VERSION}
    COMPATIBILITY SameMajorVersion
    )
  install(
    FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/${PROJECT_NAME}"
    )

endfunction()
