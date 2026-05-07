set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <TPL_ROOT_DIR> <MUNDY_SOURCE_DIR> <INSTALL_DIR>" >&2
  echo "  TPL_ROOT_DIR    : where mundy_tpl_deps are installed (fmt, gtest, ...)" >&2
  echo "  MUNDY_SOURCE_DIR: path to the MuNDy source tree" >&2
  echo "  INSTALL_DIR     : CMAKE_INSTALL_PREFIX for the MuNDy install" >&2
  exit 1
fi

TPL_ROOT_DIR=$1
MUNDY_SOURCE_DIR=$2
INSTALL_DIR=$3

# Trilinos / Kokkos are discovered via spack. Optionally pass a spec
# (e.g. TRILINOS_SPEC="trilinos@16.2.0 %gcc@13") to disambiguate.
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

# Usage:
#   bash ../do-cmake.sh ~/MuNDyScratch/mundy_tpl_deps ../ ~/MuNDyScratch/mundy_install
#
# Override package selection with TRILINOS_SPEC / KOKKOS_SPEC env vars if needed.

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using Kokkos dir: $KOKKOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using MuNDy source dir: $MUNDY_SOURCE_DIR"
echo "Using install dir: $INSTALL_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RELEASE} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O3 -march=native" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-DCTEST_BUILD_FLAGS:STRING="${CTEST_BUILD_FLAGS:--j8}" \
-DCTEST_PARALLEL_LEVEL:STRING="${CTEST_PARALLEL_LEVEL:-8}" \
-DCTEST_BUILD_NAME:STRING="${CTEST_BUILD_NAME:-mundy-cpu-local}" \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DTPL_ENABLE_MPI=ON \
-DMundy_ENABLE_MundyCore=ON \
-DMundy_ENABLE_MundyMath=ON \
-DMundy_ENABLE_MundyMesh=ON \
-DMundy_ENABLE_MundyGeom=ON \
-DMundy_ENABLE_MundyMeta=ON \
-DMundy_ENABLE_TESTS=ON \
-DMundy_ENABLE_GTest=ON \
-DMPI_EXEC_MAX_NUMPROCS=${MPI_EXEC_MAX_NUMPROCS:-1} \
-DMundy_ENABLE_STKFMM=OFF \
-DMundy_ENABLE_PVFMM=OFF \
-DMundy_ENABLE_KokkosKernels=ON \
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_nanobench_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${KOKKOS_ROOT_DIR} \
-DTPL_KokkosKernels_DIR:PATH=${KOKKOS_KERNELS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
${MUNDY_SOURCE_DIR}