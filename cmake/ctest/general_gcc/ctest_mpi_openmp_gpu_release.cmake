include("${CMAKE_CURRENT_LIST_DIR}/../MundyCTestDriver.cmake")

set(COMM_TYPE MPI)
set(THREAD_TYPE OPENMP)
set(BUILD_TYPE RELEASE)
set(COMPILER_VERSION GCC)
set(BUILD_DIR_NAME "${COMM_TYPE}_${THREAD_TYPE}_${BUILD_TYPE}")

# --- Dependency prefixes (set by caller, e.g. ci/test_jenkins_gpu_release.sh) ---
# Kokkos/KokkosKernels default to the Trilinos prefix (which ships them) but are
# separate vars: when standalone, they live in different prefixes.
set_default_and_from_env(TRILINOS_ROOT_DIR "")
set_default_and_from_env(TPL_ROOT_DIR "")
set_default_and_from_env(KOKKOS_ROOT_DIR "${TRILINOS_ROOT_DIR}")
set_default_and_from_env(KOKKOS_KERNELS_ROOT_DIR "${TRILINOS_ROOT_DIR}")
set_default_and_from_env(MUNDY_CMAKE_CXX_COMPILER "mpicxx")
set_default_and_from_env(MUNDY_CMAKE_CXX_FLAGS "-O3 -g -fno-omit-frame-pointer -march=native")
set_default_and_from_env(MUNDY_TEST_CATEGORIES "BASIC;CONTINUOUS;NIGHTLY")
set_default_and_from_env(CTEST_BUILD_FLAGS "-j8")
set_default_and_from_env(CTEST_PARALLEL_LEVEL "8")

if(TRILINOS_ROOT_DIR STREQUAL "")
  message(FATAL_ERROR "Set TRILINOS_ROOT_DIR before running this CTest driver.")
endif()

if(TPL_ROOT_DIR STREQUAL "")
  message(FATAL_ERROR "Set TPL_ROOT_DIR before running this CTest driver.")
endif()

# Double-escape ';': ctest_configure re-splits OPTIONS twice, so the list value
# needs two backslashes to survive as a single -D argument.
string(REPLACE ";" "\\\;" MUNDY_TEST_CATEGORIES_ESCAPED "${MUNDY_TEST_CATEGORIES}")

# --- Driver behavior under CI ---------------------------------------------
# DO_UPDATES=FALSE: CI owns the checkout, don't git-update. CTEST_TEST_TYPE =
# CDash track (Experimental for PR/manual, Nightly/Continuous for scheduled).
# Default: wipe+rebuild build dir, submit to the server in CTestConfig.cmake.
set_default_and_from_env(CTEST_TEST_TYPE "Experimental")
set_default_and_from_env(CTEST_DO_UPDATES "FALSE")
set_default_and_from_env(CTEST_DO_SUBMIT "TRUE")
set_default_and_from_env(CTEST_BINARY_DIRECTORY "${TRIBITS_PROJECT_ROOT}/build")

# Mundy has no extra repos. In dev mode the driver would process a
# (nonexistent) cmake/ExtraRepositoriesList.cmake and abort; NONE disables that.
set_default_and_from_env(${PROJECT_NAME}_EXTRAREPOS_FILE "NONE")

set(EXTRA_CONFIGURE_OPTIONS
  "-DCMAKE_BUILD_TYPE:STRING=Release"
  "-DCMAKE_CXX_COMPILER:STRING=${MUNDY_CMAKE_CXX_COMPILER}"
  "-DCMAKE_CXX_FLAGS:STRING=${MUNDY_CMAKE_CXX_FLAGS}"
  "-DBUILD_SHARED_LIBS:BOOL=OFF"
  "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
  "-DTPL_ENABLE_MPI:BOOL=ON"
  "-DMundy_ENABLE_TESTS:BOOL=ON"
  "-DMundy_ENABLE_GTest:BOOL=ON"
  "-DMundy_ENABLE_STKFMM:BOOL=OFF"
  "-DMundy_ENABLE_PVFMM:BOOL=OFF"
  "-DMundy_TEST_CATEGORIES:STRING=${MUNDY_TEST_CATEGORIES_ESCAPED}"
  # TPLs that live with the Mundy dependency install (TPL_ROOT_DIR)
  "-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR}"
  "-DTPL_nanobench_DIR:PATH=${TPL_ROOT_DIR}"
  "-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR}"
  "-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR}"
  "-DTPL_ArborX_DIR:PATH=${TPL_ROOT_DIR}"
  # TPLs that come from the Kokkos / Trilinos installs
  "-DTPL_Kokkos_DIR:PATH=${KOKKOS_ROOT_DIR}"
  "-DTPL_KokkosKernels_DIR:PATH=${KOKKOS_KERNELS_ROOT_DIR}"
  "-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR}"
  "-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR}"
)

set_default_and_from_env(MUNDY_CMAKE_INSTALL_PREFIX "")
if(MUNDY_CMAKE_INSTALL_PREFIX)
  list(APPEND EXTRA_CONFIGURE_OPTIONS
    "-DCMAKE_INSTALL_PREFIX:PATH=${MUNDY_CMAKE_INSTALL_PREFIX}"
  )
endif()

# Launch C++ compiles via ccache if present. The driver wipes the build dir each
# run, but ccache caches objects by content -> repeated/CI builds reuse them.
find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE)
  message("-- Using ccache as the C++ compiler launcher: ${CCACHE_EXECUTABLE}")
  list(APPEND EXTRA_CONFIGURE_OPTIONS
    "-DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CCACHE_EXECUTABLE}"
  )
endif()

mundy_ctest_driver()
