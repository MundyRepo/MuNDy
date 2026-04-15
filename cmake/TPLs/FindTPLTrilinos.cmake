if (DEFINED TPL_Trilinos_DIR)
  FIND_PACKAGE(Trilinos REQUIRED
      CONFIG
      NO_DEFAULT_PATH
      PATHS
        ${TPL_Trilinos_DIR}/lib/cmake/Trilinos
        ${TPL_Trilinos_DIR}/lib64/cmake/Trilinos
        ${TPL_Trilinos_DIR}
      COMPONENTS
        ${${PACKAGE_NAME}_Trilinos_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS
        ${${PACKAGE_NAME}_Trilinos_OPTIONAL_COMPONENTS}
  )
else()
  message(FATAL_ERROR "TPL_Trilinos_DIR must be defined before calling FIND_PACKAGE(Trilinos).")
endif()

# Print out where Trilinos was found
message(STATUS "Found Trilinos: ${Trilinos_DIR}")

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  Trilinos
  INNER_FIND_PACKAGE_NAME Trilinos
  IMPORTED_TARGETS_FOR_ALL_LIBS Trilinos::all_libs)
