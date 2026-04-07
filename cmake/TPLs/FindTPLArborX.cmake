FIND_PACKAGE(ArborX REQUIRED
    CONFIG
    NO_DEFAULT_PATH
    PATHS
      ${TPL_ArborX_DIR}/lib/cmake/ArborX
      ${TPL_ArborX_DIR}/lib64/cmake/ArborX
      ${TPL_ArborX_DIR}
)

# Print out where ArborX was found
message(STATUS "Found ArborX: ${ArborX_DIR}")

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  ArborX
  INNER_FIND_PACKAGE_NAME ArborX
  IMPORTED_TARGETS_FOR_ALL_LIBS ArborX::ArborX )