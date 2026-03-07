tribits_package_decl(Mundy)

tribits_add_debug_option()

# Set
GLOBAL_SET(Mundy_DEFAULT_UNIT_TEST_MAIN
  "${PACKAGE_SOURCE_DIR}/tests/unit_tests/DefaultUnitTestMain.cpp")
GLOBAL_SET(Mundy_DEFAULT_INTEGRATION_TEST_MAIN
  "${PACKAGE_SOURCE_DIR}/tests/integration_tests/DefaultIntegrationTestMain.cpp")

# For now, we enable all subpackages of Mundy by default. They can be disabled by the user
# if they are not needed.
GLOBAL_SET(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)

tribits_add_test_directories("${PACKAGE_SOURCE_DIR}/tests/unit_tests")

tribits_process_subpackages()

tribits_package_def()

tribits_package_postprocess()
