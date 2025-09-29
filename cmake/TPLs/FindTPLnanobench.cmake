FIND_PACKAGE(nanobench REQUIRED
    CONFIG
    HINTS
      ${TPL_nanobench_DIR}/lib/cmake/nanobench
      ${TPL_nanobench_DIR}/lib64/cmake/nanobench
      ${TPL_nanobench_DIR}
)

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  nanobench
  INNER_FIND_PACKAGE_NAME nanobench
  IMPORTED_TARGETS_FOR_ALL_LIBS nanobench::nanobench)