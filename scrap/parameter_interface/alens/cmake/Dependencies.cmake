tribits_package_define_dependencies(
  LIB_REQUIRED_PACKAGES MundyCore MundyMath MundyGeom MundyMesh MundyMeta MundyAgents MundyShapes MundyLinkers MundyIo MundyConstraints
  LIB_OPTIONAL_PACKAGES
  TEST_REQUIRED_PACKAGES
  TEST_OPTIONAL_PACKAGES
  LIB_REQUIRED_TPLS STK Teuchos Kokkos KokkosKernels fmt
  LIB_OPTIONAL_TPLS MPI STKFMM PVFMM CUDA
  TEST_REQUIRED_TPLS GTest OpenRAND
  TEST_OPTIONAL_TPLS
  )
