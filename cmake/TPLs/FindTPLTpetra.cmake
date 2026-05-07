if (DEFINED TPL_Tpetra_DIR)
  FIND_PACKAGE(Tpetra REQUIRED
      CONFIG
      PATHS
        ${TPL_Tpetra_DIR}/lib/cmake/Tpetra
        ${TPL_Tpetra_DIR}/lib64/cmake/Tpetra
        ${TPL_Tpetra_DIR}
      COMPONENTS
        ${${PACKAGE_NAME}_Tpetra_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS
        ${${PACKAGE_NAME}_Tpetra_OPTIONAL_COMPONENTS}
  )
else()
  message(FATAL_ERROR "TPL_Tpetra_DIR must be defined before calling FIND_PACKAGE(Tpetra).")
endif()

# Print out where Tpetra was found
message(STATUS "Found Tpetra: ${Tpetra_DIR}")

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  Tpetra
  INNER_FIND_PACKAGE_NAME Tpetra
  IMPORTED_TARGETS_FOR_ALL_LIBS Tpetra::all_libs)
