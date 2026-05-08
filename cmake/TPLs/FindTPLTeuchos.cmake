if (DEFINED TPL_Teuchos_DIR)
  FIND_PACKAGE(Teuchos REQUIRED
      CONFIG
      PATHS
        ${TPL_Teuchos_DIR}/lib/cmake/Teuchos
        ${TPL_Teuchos_DIR}/lib64/cmake/Teuchos
        ${TPL_Teuchos_DIR}
      COMPONENTS
        ${${PACKAGE_NAME}_Teuchos_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS
        ${${PACKAGE_NAME}_Teuchos_OPTIONAL_COMPONENTS}
  )
else()
  message(FATAL_ERROR "TPL_Teuchos_DIR must be defined before calling FIND_PACKAGE(Teuchos).")
endif()

# Print out where Teuchos was found
message(STATUS "Found Teuchos: ${Teuchos_DIR}")

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  Teuchos
  INNER_FIND_PACKAGE_NAME Teuchos
  IMPORTED_TARGETS_FOR_ALL_LIBS Teuchos::all_libs)
