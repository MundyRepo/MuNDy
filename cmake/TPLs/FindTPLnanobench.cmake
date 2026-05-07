if (DEFINED TPL_nanobench_DIR)
  FIND_PACKAGE(nanobench REQUIRED
      CONFIG
      PATHS
        ${TPL_nanobench_DIR}/lib/cmake/nanobench
        ${TPL_nanobench_DIR}/lib64/cmake/nanobench
        ${TPL_nanobench_DIR}
  )
else()
  message(FATAL_ERROR "TPL_nanobench_DIR must be defined before calling FIND_PACKAGE(nanobench).")
endif()

# Print out where nanobench was found
message(STATUS "Found nanobench: ${nanobench_DIR}")

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  nanobench
  INNER_FIND_PACKAGE_NAME nanobench
  IMPORTED_TARGETS_FOR_ALL_LIBS nanobench::nanobench)