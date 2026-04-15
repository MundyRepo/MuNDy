# tribits_tpl_find_include_dirs_and_libraries( OpenRAND
#   REQUIRED_HEADERS openrand/base_state.h openrand/philox.h openrand/squares.h openrand/threefry.h openrand/tyche.h openrand/util.h
#   MUST_FIND_ALL_HEADERS 
#   )

if (DEFINED TPL_OpenRAND_DIR)
  FIND_PACKAGE(OpenRAND REQUIRED
      CONFIG
      NO_DEFAULT_PATH
      PATHS
        ${TPL_OpenRAND_DIR}/lib/cmake/OpenRAND
        ${TPL_OpenRAND_DIR}/lib64/cmake/OpenRAND
        ${TPL_OpenRAND_DIR}
  )
else()
  message(FATAL_ERROR "TPL_OpenRAND_DIR must be defined before calling FIND_PACKAGE(OpenRAND).")
endif()

# Print out where OpenRAND was found
message(STATUS "Found OpenRAND: ${OpenRAND_DIR}")

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  OpenRAND
  INNER_FIND_PACKAGE_NAME OpenRAND
  IMPORTED_TARGETS_FOR_ALL_LIBS OpenRAND::OpenRAND)