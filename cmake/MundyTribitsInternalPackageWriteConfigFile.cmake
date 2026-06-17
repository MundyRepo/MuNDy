# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

include_guard(GLOBAL)

# Local overrides for a small subset of the TriBITS internal package-config
# generation path. This file is reached through the same-name shim in
# ${PROJECT_SOURCE_DIR}/cmake/ so TriBITS resolves it before the upstream
# module.
#
# Public contract preserved by this override layer:
# - find_package(Mundy) remains the project-level entry point.
# - find_package(MundyUtils), find_package(MundyMath), etc. remain public
#   package-level entry points.
# - MundyPackageConfig.cmake and the MundyPackage_* variable namespace are
#   private implementation details used only to break the self-name collision
#   between the Mundy project and the Mundy package.
include("${${PROJECT_NAME}_TRIBITS_DIR}/core/package_arch/TribitsInternalPackageWriteConfigFile.cmake")


# @FUNCTION: mundy_tribits_get_install_package_config_file_name()
#
# Returns the installed config filename for a package. For the self-named
# top-level package, install a private helper config beside the project config
# under the name <Project>PackageConfig.cmake so the public <Project>Config.cmake
# can include it without colliding on disk.
function(mundy_tribits_get_install_package_config_file_name packageName outputVar)
  set(packageConfigFileName "${packageName}Config.cmake")

  if ("${packageName}" STREQUAL "${PROJECT_NAME}")
    set(packageConfigFileName "${packageName}PackageConfig.cmake")
  endif()

  set(${outputVar} "${packageConfigFileName}" PARENT_SCOPE)
endfunction()


# @FUNCTION: mundy_tribits_get_package_export_file_var_prefix()
#
# Returns the variable prefix to use in generated package configs. The
# self-named top-level package exports MundyPackage_* variables to avoid
# stomping the project-level Mundy_* variables when it is loaded from the
# public project config.
function(mundy_tribits_get_package_export_file_var_prefix packageName outputVar)
  set(exportFileVarPrefix "${packageName}")

  if ("${packageName}" STREQUAL "${PROJECT_NAME}")
    set(exportFileVarPrefix "${packageName}Package")
  endif()

  set(${outputVar} "${exportFileVarPrefix}" PARENT_SCOPE)
endfunction()


# @FUNCTION: mundy_tribits_get_package_config_template()
#
# Returns the package-config template used for both build-tree and install-tree
# package configs. Mundy uses a local copy so deprecation text can refer to the
# active export-file prefix instead of always hardcoding ${PACKAGE_NAME}_LIBRARIES.
function(mundy_tribits_get_package_config_template outputVar)
  set(packageConfigTemplate
    "${PROJECT_SOURCE_DIR}/cmake/MundyTribitsPackageConfigTemplate.cmake.in")
  if (NOT EXISTS "${packageConfigTemplate}")
    set(packageConfigTemplate
      "${${PROJECT_NAME}_TRIBITS_DIR}/${TRIBITS_CMAKE_INSTALLATION_FILES_DIR}/TribitsPackageConfigTemplate.cmake.in")
  endif()

  set(${outputVar} "${packageConfigTemplate}" PARENT_SCOPE)
endfunction()


# @FUNCTION: tribits_write_package_client_export_files()
#
# Local override of the upstream TriBITS function. The only behavioral delta is
# that the self-named top-level package writes a private MundyPackageConfig.cmake
# file and exports MundyPackage_* variables instead of Mundy_*.
function(tribits_write_package_client_export_files PACKAGE_NAME)

  if(${PROJECT_NAME}_VERBOSE_CONFIGURE)
    message("\nTRIBITS_WRITE_PACKAGE_CLIENT_EXPORT_FILES: ${PACKAGE_NAME}")
  endif()

  set(buildDirCMakePkgsDir
     "${${PROJECT_NAME}_BINARY_DIR}/${${PROJECT_NAME}_BUILD_DIR_CMAKE_PKGS_DIR}")

  mundy_tribits_get_package_export_file_var_prefix(
    ${PACKAGE_NAME} exportFileVarPrefix)
  mundy_tribits_get_install_package_config_file_name(
    ${PACKAGE_NAME} packageConfigInstallFileName)

  set(EXPORT_FILES_ARGS PACKAGE_NAME ${PACKAGE_NAME})
  if (NOT "${exportFileVarPrefix}" STREQUAL "${PACKAGE_NAME}")
    list(APPEND EXPORT_FILES_ARGS EXPORT_FILE_VAR_PREFIX "${exportFileVarPrefix}")
  endif()

  if (${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES)
    if(${PROJECT_NAME}_VERBOSE_CONFIGURE)
      message("For package ${PACKAGE_NAME} creating ${packageConfigInstallFileName}")
    endif()
    set(PACKAGE_CONFIG_FOR_BUILD_BASE_DIR
      "${buildDirCMakePkgsDir}/${PACKAGE_NAME}" )
    set(PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles" )
    append_set(EXPORT_FILES_ARGS
      PACKAGE_CONFIG_FOR_BUILD_BASE_DIR "${PACKAGE_CONFIG_FOR_BUILD_BASE_DIR}"
      PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR "${PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR}"
      )
  endif()

  tribits_write_flexible_package_client_export_files(${EXPORT_FILES_ARGS})

  tribits_write_package_client_export_files_export_and_install_targets(${EXPORT_FILES_ARGS})

endfunction()


# @FUNCTION: tribits_generate_package_config_file_for_build_tree()
#
# Local override of the upstream TriBITS function. The only behavioral delta is
# that Mundy uses its explicit local package-config template so generated files
# stay internally consistent when EXPORT_FILE_VAR_PREFIX is not PACKAGE_NAME.
function(tribits_generate_package_config_file_for_build_tree packageName)

  if (TRIBITS_WRITE_FLEXIBLE_PACKAGE_CLIENT_EXPORT_FILES_DEBUG_DUMP)
    message("tribits_generate_package_config_file_for_build_tree(${ARGV})")
  endif()

  cmake_parse_arguments(
     PARSE  #prefix
     ""    #options
     "EXPORT_FILE_VAR_PREFIX"  #one_value_keywords
     "" #multi_value_keywords
     ${ARGN}
     )

  if (PARSE_EXPORT_FILE_VAR_PREFIX)
    set(EXPORT_FILE_VAR_PREFIX ${PARSE_EXPORT_FILE_VAR_PREFIX})
  else()
    set(EXPORT_FILE_VAR_PREFIX ${packageName})
  endif()

  set(buildDirExtPkgsDir
     "${${PROJECT_NAME}_BINARY_DIR}/${${PROJECT_NAME}_BUILD_DIR_EXTERNAL_PKGS_DIR}")
  set(buildDirCMakePkgsDir
     "${${PROJECT_NAME}_BINARY_DIR}/${${PROJECT_NAME}_BUILD_DIR_CMAKE_PKGS_DIR}")

  if (PARSE_PACKAGE_CONFIG_FOR_BUILD_BASE_DIR
      OR PARSE_PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR
    )
    set(PACKAGE_CONFIG_CODE "")

    tribits_append_dependent_package_config_file_includes_and_enables_str(${packageName}
      EXPORT_FILE_VAR_PREFIX ${EXPORT_FILE_VAR_PREFIX}
      EXT_PKG_CONFIG_FILE_BASE_DIR "${buildDirExtPkgsDir}"
      PKG_CONFIG_FILE_BASE_DIR "${buildDirCMakePkgsDir}"
      CONFIG_FILE_STR_INOUT PACKAGE_CONFIG_CODE )

    if (PARSE_PACKAGE_CONFIG_FOR_BUILD_BASE_DIR)
      tribits_get_package_config_build_dir_targets_file(${packageName}
        "${PACKAGE_CONFIG_FOR_BUILD_BASE_DIR}" packageConfigBuildDirTargetsFile )
      string(APPEND PACKAGE_CONFIG_CODE
        "\n# Import ${packageName} targets\n"
        "include(\"${packageConfigBuildDirTargetsFile}\")\n")
    endif()

    tribits_extpkg_append_tribits_compliant_package_config_vars_str(${packageName}
      PACKAGE_CONFIG_CODE)

    tribits_set_compiler_vars_for_config_file(BUILD_DIR)

    if ("${CMAKE_CXX_FLAGS}" STREQUAL "")
      set(CMAKE_CXX_FLAGS_ESCAPED "")
    else()
      string(REGEX REPLACE "\"" "\\\\\"" CMAKE_CXX_FLAGS_ESCAPED ${CMAKE_CXX_FLAGS})
    endif()

    set(EXPORTED_PACKAGE_LIBS_NAMES ${${packageName}_EXPORTED_PACKAGE_LIBS_NAMES})
    set(PDOLLAR "$")

    mundy_tribits_get_package_config_template(packageConfigTemplate)
    configure_file(
      "${packageConfigTemplate}"
      "${PARSE_PACKAGE_CONFIG_FOR_BUILD_BASE_DIR}/${packageName}Config.cmake"
      )

  endif()

endfunction()


# @FUNCTION: tribits_generate_package_config_file_for_install_tree()
#
# Local override of the upstream TriBITS function. The only behavioral delta is
# that Mundy uses its explicit local package-config template so generated files
# stay internally consistent when EXPORT_FILE_VAR_PREFIX is not PACKAGE_NAME.
function(tribits_generate_package_config_file_for_install_tree packageName)

  if (TRIBITS_WRITE_FLEXIBLE_PACKAGE_CLIENT_EXPORT_FILES_DEBUG_DUMP)
    message("tribits_generate_package_config_file_for_install_tree(${ARGV})")
  endif()

  cmake_parse_arguments(
     PARSE  #prefix
     ""    #options
     "EXPORT_FILE_VAR_PREFIX"  #one_value_keywords
     "" #multi_value_keywords
     ${ARGN}
     )

  if (PARSE_EXPORT_FILE_VAR_PREFIX)
    set(EXPORT_FILE_VAR_PREFIX ${PARSE_EXPORT_FILE_VAR_PREFIX})
  else()
    set(EXPORT_FILE_VAR_PREFIX ${packageName})
  endif()

  set(PACKAGE_CONFIG_CODE "")

  tribits_append_dependent_package_config_file_includes_and_enables_str(${packageName}
    EXPORT_FILE_VAR_PREFIX ${EXPORT_FILE_VAR_PREFIX}
    EXT_PKG_CONFIG_FILE_BASE_DIR
      "\${CMAKE_CURRENT_LIST_DIR}/../../${${PROJECT_NAME}_BUILD_DIR_EXTERNAL_PKGS_DIR}"
    PKG_CONFIG_FILE_BASE_DIR "\${CMAKE_CURRENT_LIST_DIR}/.."
    CONFIG_FILE_STR_INOUT PACKAGE_CONFIG_CODE )

  string(APPEND PACKAGE_CONFIG_CODE
    "\n# Import ${packageName} targets\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/${packageName}Targets.cmake\")\n")

  tribits_extpkg_append_tribits_compliant_package_config_vars_str(${packageName}
    PACKAGE_CONFIG_CODE)

  if (BUILD_SHARED_LIBS)
    set(SHARED_LIB_RPATH_COMMAND
      ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG}${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_LIB_DIR}
      )
  endif()

  tribits_set_compiler_vars_for_config_file(INSTALL_DIR)

  set(EXPORTED_PACKAGE_LIBS_NAMES ${${packageName}_EXPORTED_PACKAGE_LIBS_NAMES})
  set(PDOLLAR "$")

  if (PARSE_PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR)
    mundy_tribits_get_package_config_template(packageConfigTemplate)
    configure_file(
      "${packageConfigTemplate}"
      "${PARSE_PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR}/${packageName}Config_install.cmake"
      )
  endif()

endfunction()


# @FUNCTION: tribits_write_package_client_export_files_export_and_install_targets()
#
# Local override of the upstream TriBITS function. The only behavioral delta is
# the installed filename for the self-named top-level package config.
function(tribits_write_package_client_export_files_export_and_install_targets)

  cmake_parse_arguments(
     PARSE
     ""
     "PACKAGE_NAME;PACKAGE_CONFIG_FOR_BUILD_BASE_DIR;PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR"
     ""
     ${ARGN}
     )

  set(PACKAGE_NAME ${PARSE_PACKAGE_NAME})

  if (PARSE_PACKAGE_CONFIG_FOR_BUILD_BASE_DIR)
    tribits_get_package_config_build_dir_targets_file(${PACKAGE_NAME}
      "${PARSE_PACKAGE_CONFIG_FOR_BUILD_BASE_DIR}" packageConfigBuildDirTargetsFile )
    export(
      EXPORT ${PACKAGE_NAME}
      NAMESPACE ${PACKAGE_NAME}::
      FILE "${packageConfigBuildDirTargetsFile}" )
  endif()

  if (PARSE_PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR)
    mundy_tribits_get_install_package_config_file_name(
      ${PACKAGE_NAME} packageConfigInstallFileName)

    install(
      FILES
        "${PARSE_PACKAGE_CONFIG_FOR_INSTALL_BASE_DIR}/${PACKAGE_NAME}Config_install.cmake"
      DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/${PACKAGE_NAME}"
      RENAME ${packageConfigInstallFileName}
      )
    install(
      EXPORT ${PACKAGE_NAME}
      NAMESPACE ${PACKAGE_NAME}::
      DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/${PACKAGE_NAME}"
      FILE "${PACKAGE_NAME}Targets.cmake" )
  endif()

endfunction()
