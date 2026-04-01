# The order of the packages below is important. Mundy subpackages can
# depend on subpackages listed above them, and not below them.
tribits_package_define_dependencies(
  LIB_REQUIRED_PACKAGES
  LIB_OPTIONAL_PACKAGES
  TEST_REQUIRED_PACKAGES
  TEST_OPTIONAL_PACKAGES
  LIB_REQUIRED_TPLS
  LIB_OPTIONAL_TPLS MPI STK
  TEST_REQUIRED_TPLS
  TEST_OPTIONAL_TPLS GTest nanobench
  SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
    Utils core/utils PT OPTIONAL
    Math  core/math PT OPTIONAL
    Geom  core/geom PT OPTIONAL
    Mech  core/mech PT OPTIONAL
    Mesh  mesh PT OPTIONAL
  REGRESSION_EMAIL_LIST brycepalmer96@gmail.com
)
