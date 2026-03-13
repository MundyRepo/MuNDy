FIND_PACKAGE(STK REQUIRED
    CONFIG
    HINTS
      ${TPL_STK_DIR}/lib/cmake/STK
      ${TPL_STK_DIR}
    COMPONENTS
      ${${PACKAGE_NAME}_STK_REQUIRED_COMPONENTS}
    OPTIONAL_COMPONENTS
      ${${PACKAGE_NAME}_STK_OPTIONAL_COMPONENTS}
)

# Print out where STK was found
message(STATUS "Found STK: ${STK_DIR}")

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  STK
  INNER_FIND_PACKAGE_NAME STK
  IMPORTED_TARGETS_FOR_ALL_LIBS STK::all_libs)
