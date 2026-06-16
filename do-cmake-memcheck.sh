set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <TPL_ROOT_DIR> <MUNDY_SOURCE_DIR> <INSTALL_DIR>" >&2
  echo "  TPL_ROOT_DIR    : where mundy_tpl_deps are installed (fmt, gtest, ...)" >&2
  echo "  MUNDY_SOURCE_DIR: path to the MuNDy source tree" >&2
  echo "  INSTALL_DIR     : CMAKE_INSTALL_PREFIX for the MuNDy install" >&2
  echo "" >&2
  echo "Memory-check build. After cmake + make, run from the build directory:" >&2
  echo "  ctest -T memcheck                   # all tests" >&2
  echo "  ctest -T memcheck -L MundyMath      # one package" >&2
  echo "  ctest -T memcheck -R ^<TEST_NAME>$  # one test" >&2
  echo "Results: Testing/Temporary/LastDynamicAnalysis_*.log" >&2
  exit 1
fi

TPL_ROOT_DIR=$(readlink -f "$1")
MUNDY_SOURCE_DIR=$(readlink -f "$2")
INSTALL_DIR=$(readlink -m "$3")

if [ ! -d "$TPL_ROOT_DIR" ]; then
  echo "ERROR: TPL_ROOT_DIR does not exist or is not a directory: $1" >&2
  exit 1
fi
if [ ! -d "$MUNDY_SOURCE_DIR" ]; then
  echo "ERROR: MUNDY_SOURCE_DIR does not exist or is not a directory: $2" >&2
  exit 1
fi

VALGRIND=$(command -v valgrind 2>/dev/null) || {
  echo "ERROR: 'valgrind' not found in PATH." >&2
  exit 1
}

# Trilinos / Kokkos are discovered via spack.
TRILINOS_SPEC=${TRILINOS_SPEC:-trilinos}
KOKKOS_SPEC=${KOKKOS_SPEC:-kokkos}
KOKKOS_KERNELS_SPEC=${KOKKOS_KERNELS_SPEC:-kokkos-kernels}

if ! command -v spack >/dev/null 2>&1; then
  echo "ERROR: 'spack' not found in PATH. Source your spack setup-env.sh first." >&2
  exit 1
fi

TRILINOS_ROOT_DIR=$(spack location -i "${TRILINOS_SPEC}") || {
  echo "ERROR: could not locate spack install for '${TRILINOS_SPEC}'." >&2
  exit 1
}
KOKKOS_ROOT_DIR=$(spack location -i "${KOKKOS_SPEC}") || {
  echo "ERROR: could not locate spack install for '${KOKKOS_SPEC}'." >&2
  exit 1
}
KOKKOS_KERNELS_ROOT_DIR=$(spack location -i "${KOKKOS_KERNELS_SPEC}") || {
  echo "ERROR: could not locate spack install for '${KOKKOS_KERNELS_SPEC}'." >&2
  exit 1
}

echo "Using Trilinos dir:  $TRILINOS_ROOT_DIR"
echo "Using Kokkos dir:    $KOKKOS_ROOT_DIR"
echo "Using TPL dir:       $TPL_ROOT_DIR"
echo "Using MuNDy source:  $MUNDY_SOURCE_DIR"
echo "Using install dir:   $INSTALL_DIR"
echo "Using valgrind:      $VALGRIND"

# Wrapper script that reverses the invocation order for MPI tests.
# CTest normally generates:  valgrind [opts] mpiexec -np N ./test.exe
# The wrapper reassembles as: mpiexec -np N valgrind [opts] ./test.exe
# This prevents valgrind from having to JIT-compile AVX-512 instructions
# from spack-built shared libraries during mpiexec's fork/exec machinery,
# which caused SIGILL on Cascade Lake nodes.
WRAP="${MUNDY_SOURCE_DIR}/cmake/valgrind/mpi-valgrind-wrap.sh"
if [ ! -f "${WRAP}" ]; then
  echo "ERROR: MPI valgrind wrapper not found at ${WRAP}" >&2
  exit 1
fi
chmod +x "${WRAP}"

# Suppression file.
SUPP_FILE="${MUNDY_SOURCE_DIR}/cmake/valgrind/mundy.supp"
if [ -f "${SUPP_FILE}" ]; then
  SUPP_ARGS="--suppressions=${SUPP_FILE}"
  echo "Using suppressions:  ${SUPP_FILE}"
else
  SUPP_ARGS=""
  echo "No suppression file found at ${SUPP_FILE}; running without project suppressions."
fi

# memcheck configuration:
#   NO --trace-children=yes: the wrapper puts valgrind inside mpiexec so
#     tracing into children is not needed and would re-introduce the SIGILL.
#   --leak-check=yes      report reachable and definitely-lost blocks
#   --num-callers=50      deep stack traces aid diagnosis
#   --error-exitcode=1    non-zero exit lets CTest mark the test as failed
MEMCHECK_OPTIONS="-q --tool=memcheck \
--leak-check=yes --num-callers=50 \
--error-exitcode=1 ${SUPP_ARGS}"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-DEBUG} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-g -O0" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-DCTEST_BUILD_FLAGS:STRING="${CTEST_BUILD_FLAGS:--j8}" \
-DCTEST_PARALLEL_LEVEL:STRING="${CTEST_PARALLEL_LEVEL:-8}" \
-DCTEST_BUILD_NAME:STRING="${CTEST_BUILD_NAME:-mundy-cpu-memcheck}" \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DTPL_ENABLE_MPI=ON \
-DMPI_EXEC_MAX_NUMPROCS=${MPI_EXEC_MAX_NUMPROCS:-1} \
-DMundy_ENABLE_MundyCore=ON \
-DMundy_ENABLE_MundyMath=ON \
-DMundy_ENABLE_MundyGeom=ON \
-DMundy_ENABLE_MundyMesh=ON \
-DMundy_ENABLE_MundySearch=ON \
-DMundy_ENABLE_TESTS=ON \
-DMundy_ENABLE_GTest=ON \
-DMundy_ENABLE_ArborX=ON \
-DMundy_ENABLE_STKFMM=OFF \
-DMundy_ENABLE_PVFMM=OFF \
-DMundy_ENABLE_KokkosKernels=ON \
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY" \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_ArborX_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_nanobench_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${KOKKOS_ROOT_DIR} \
-DTPL_KokkosKernels_DIR:PATH=${KOKKOS_KERNELS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DMEMORYCHECK_COMMAND="${WRAP}" \
-DMEMORYCHECK_COMMAND_OPTIONS="${MEMCHECK_OPTIONS}" \
${extra_args} \
${MUNDY_SOURCE_DIR}
