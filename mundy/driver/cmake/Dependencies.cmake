tribits_package_define_dependencies(
  LIB_REQUIRED_PACKAGES MundyCore MundyMesh MundyMeta
  LIB_OPTIONAL_PACKAGES MundyIo MundyLinkers MundyShapes MundyAgents MundyConstraints
  TEST_REQUIRED_PACKAGES
  TEST_OPTIONAL_PACKAGES
  LIB_REQUIRED_TPLS STK Kokkos Teuchos fmt
  LIB_OPTIONAL_TPLS MPI CUDA
  TEST_REQUIRED_TPLS GTest
  TEST_OPTIONAL_TPLS
  )
