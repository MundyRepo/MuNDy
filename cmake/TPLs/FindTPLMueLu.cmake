if (DEFINED TPL_MueLu_DIR)
  FIND_PACKAGE(MueLu REQUIRED
      CONFIG
      PATHS
        ${TPL_MueLu_DIR}/lib/cmake/MueLu
        ${TPL_MueLu_DIR}/lib64/cmake/MueLu
        ${TPL_MueLu_DIR}
      COMPONENTS
        ${${PACKAGE_NAME}_MueLu_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS
        ${${PACKAGE_NAME}_MueLu_OPTIONAL_COMPONENTS}
  )
else()
  message(FATAL_ERROR "TPL_MueLu_DIR must be defined before calling FIND_PACKAGE(MueLu).")
endif()

# Print out where MueLu was found
message(STATUS "Found MueLu: ${MueLu_DIR}")

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  MueLu
  INNER_FIND_PACKAGE_NAME MueLu
  IMPORTED_TARGETS_FOR_ALL_LIBS MueLu::all_libs)
