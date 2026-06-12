set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <TPL_ROOT_DIR> <MUNDY_SOURCE_DIR> <INSTALL_DIR>" >&2
  echo "  TPL_ROOT_DIR    : where mundy_tpl_deps are installed (fmt, gtest, ...)" >&2
  echo "  MUNDY_SOURCE_DIR: path to the MuNDy source tree" >&2
  echo "  INSTALL_DIR     : CMAKE_INSTALL_PREFIX for the MuNDy install" >&2
  exit 1
fi

TPL_ROOT_DIR=$(readlink -f "$1")
MUNDY_SOURCE_DIR=$(readlink -f "$2")
INSTALL_DIR=$(readlink -m "$3")  # -m: install dir need not exist yet

if [ ! -d "$TPL_ROOT_DIR" ]; then
  echo "ERROR: TPL_ROOT_DIR does not exist or is not a directory: $1" >&2
  exit 1
fi
if [ ! -d "$MUNDY_SOURCE_DIR" ]; then
  echo "ERROR: MUNDY_SOURCE_DIR does not exist or is not a directory: $2" >&2
  exit 1
fi

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

is_positive_integer() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    0) return 1 ;;
    *) return 0 ;;
  esac
}

# TriBITS uses MPI_EXEC_MAX_NUMPROCS as a hard configure-time cap:
# tests requesting more ranks are not added to CTest. MPI_EXEC_DEFAULT_NUMPROCS
# is only used for MPI tests without NUM_MPI_PROCS; TriBITS does not clamp it
# to the max, so keep the default <= max.
#
# Policy: explicit env vars win. Otherwise infer max from Slurm when possible,
# fall back to 1 inside Slurm with no task count, and use nproc outside Slurm.
# If default is not set, use min(4, max) to preserve TriBITS' usual default
# without skipping default-rank MPI tests on small allocations.
if [ -n "${MPI_EXEC_MAX_NUMPROCS:-}" ]; then
  if ! is_positive_integer "$MPI_EXEC_MAX_NUMPROCS"; then
    echo "ERROR: MPI_EXEC_MAX_NUMPROCS must be a positive integer: $MPI_EXEC_MAX_NUMPROCS" >&2
    exit 1
  fi
  MPI_MAX_NUMPROCS="$MPI_EXEC_MAX_NUMPROCS"
elif [ -n "${SLURM_STEP_NUM_TASKS:-}" ] && is_positive_integer "$SLURM_STEP_NUM_TASKS"; then
  MPI_MAX_NUMPROCS="$SLURM_STEP_NUM_TASKS"
elif [ -n "${SLURM_NTASKS:-}" ] && is_positive_integer "$SLURM_NTASKS"; then
  MPI_MAX_NUMPROCS="$SLURM_NTASKS"
elif [ -n "${SLURM_NPROCS:-}" ] && is_positive_integer "$SLURM_NPROCS"; then
  MPI_MAX_NUMPROCS="$SLURM_NPROCS"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  MPI_MAX_NUMPROCS=1
else
  MPI_MAX_NUMPROCS=$(nproc 2>/dev/null || echo 1)
  if ! is_positive_integer "$MPI_MAX_NUMPROCS"; then
    MPI_MAX_NUMPROCS=1
  fi
fi

if [ -n "${MPI_EXEC_DEFAULT_NUMPROCS:-}" ]; then
  if ! is_positive_integer "$MPI_EXEC_DEFAULT_NUMPROCS"; then
    echo "ERROR: MPI_EXEC_DEFAULT_NUMPROCS must be a positive integer: $MPI_EXEC_DEFAULT_NUMPROCS" >&2
    exit 1
  fi
  MPI_DEFAULT_NUMPROCS="$MPI_EXEC_DEFAULT_NUMPROCS"
elif [ "$MPI_MAX_NUMPROCS" -lt 4 ]; then
  MPI_DEFAULT_NUMPROCS="$MPI_MAX_NUMPROCS"
else
  MPI_DEFAULT_NUMPROCS=4
fi

# Usage:
#   bash ../do-cmake.sh ~/MuNDyScratch/mundy_tpl_deps ../ ~/MuNDyScratch/mundy_install
#
# Override package selection with TRILINOS_SPEC / KOKKOS_SPEC env vars if needed.

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using Kokkos dir: $KOKKOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using MuNDy source dir: $MUNDY_SOURCE_DIR"
echo "Using install dir: $INSTALL_DIR"
echo "Using MPI max num procs: $MPI_MAX_NUMPROCS"
echo "Using MPI default num procs: $MPI_DEFAULT_NUMPROCS"

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
-DMPI_EXEC_MAX_NUMPROCS:STRING=${MPI_MAX_NUMPROCS} \
-DMPI_EXEC_DEFAULT_NUMPROCS:STRING=${MPI_DEFAULT_NUMPROCS} \
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
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_ArborX_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_nanobench_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${KOKKOS_ROOT_DIR} \
-DTPL_KokkosKernels_DIR:PATH=${KOKKOS_KERNELS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${extra_args} \
${MUNDY_SOURCE_DIR}
