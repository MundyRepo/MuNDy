if (DEFINED TPL_GTest_DIR)
  FIND_PACKAGE(GTest REQUIRED
      NO_DEFAULT_PATH
      PATHS
        ${TPL_GTest_DIR}/lib/cmake/GTest
        ${TPL_GTest_DIR}/lib64/cmake/GTest
        ${TPL_GTest_DIR}
  )
else()
  message(FATAL_ERROR "TPL_GTest_DIR must be defined before calling FIND_PACKAGE(GTest).")
endif()

# Print out where GTest was found
message(STATUS "Found GTest: ${GTest_DIR}")

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  GTest
  INNER_FIND_PACKAGE_NAME GTest
  IMPORTED_TARGETS_FOR_ALL_LIBS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)