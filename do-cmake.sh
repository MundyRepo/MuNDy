set -e

ENABLE_COVERAGE=0
POSITIONAL=()

usage() {
  echo "Usage: $0 [--coverage] <TPL_ROOT_DIR> <MUNDY_SOURCE_DIR> <INSTALL_DIR>" >&2
  echo "  --coverage      : configure a gcov-instrumented build (-O0 -g --coverage)." >&2
  echo "                    Forces CMAKE_BUILD_TYPE=Debug and skips ccache -- both" >&2
  echo "                    optimization and cached objects distort line-level coverage" >&2
  echo "                    attribution, and this is meant as a one-off diagnostic build," >&2
  echo "                    not the hot path ccache is there to speed up." >&2
  echo "  TPL_ROOT_DIR    : where mundy_tpl_deps are installed (fmt, gtest, ...)" >&2
  echo "  MUNDY_SOURCE_DIR: path to the MuNDy source tree" >&2
  echo "  INSTALL_DIR     : CMAKE_INSTALL_PREFIX for the MuNDy install" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --coverage)
      ENABLE_COVERAGE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
POSITIONAL+=("$@")  # no-op unless `--` was hit above

if [ "${#POSITIONAL[@]}" -lt 3 ]; then
  usage
fi

TPL_ROOT_DIR=$(readlink -f "${POSITIONAL[0]}")
MUNDY_SOURCE_DIR=$(readlink -f "${POSITIONAL[1]}")
INSTALL_DIR=$(readlink -m "${POSITIONAL[2]}")  # -m: install dir need not exist yet

if [ ! -d "$TPL_ROOT_DIR" ]; then
  echo "ERROR: TPL_ROOT_DIR does not exist or is not a directory: ${POSITIONAL[0]}" >&2
  exit 1
fi
if [ ! -d "$MUNDY_SOURCE_DIR" ]; then
  echo "ERROR: MUNDY_SOURCE_DIR does not exist or is not a directory: ${POSITIONAL[1]}" >&2
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

# TriBITS MPI test-registration policy
#
# TriBITS decides which MPI tests exist at configure time. The relevant knobs
# are defined in TriBITS' MPI/test support modules:
#
#   MPI_EXEC_MAX_NUMPROCS
#     Hard cap for test registration. A test requesting more ranks than this
#     value is not added to CTest.
#
#   MPI_EXEC_DEFAULT_NUMPROCS
#     Rank count for MPI tests that do not specify NUM_MPI_PROCS. This is not
#     a cap, so keep it <= MPI_EXEC_MAX_NUMPROCS.
#
# This script chooses MPI_EXEC_MAX_NUMPROCS from, in order:
#   1. explicit MPI_EXEC_MAX_NUMPROCS
#   2. Slurm task-count variables
#   3. 1 inside Slurm when no task count is visible
#   4. local CPU count outside Slurm
#
# The default rank count is explicit MPI_EXEC_DEFAULT_NUMPROCS when provided;
# otherwise min(4, MPI_EXEC_MAX_NUMPROCS), matching TriBITS' default when the
# allocation can support it.
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
#   bash ../do-cmake.sh --coverage ~/MuNDyScratch/mundy_tpl_deps ../ ~/MuNDyScratch/mundy_install
#
# Override package selection with TRILINOS_SPEC / KOKKOS_SPEC env vars if needed.

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using Kokkos dir: $KOKKOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using MuNDy source dir: $MUNDY_SOURCE_DIR"
echo "Using install dir: $INSTALL_DIR"
echo "Using MPI max num procs: $MPI_MAX_NUMPROCS"
echo "Using MPI default num procs: $MPI_DEFAULT_NUMPROCS"

# Coverage builds need -O0 -g for reliable gcov line attribution (-O3 folds
# and inlines lines together, which silently corrupts per-line hit counts),
# and skip ccache -- coverage's .gcno notes files are tied to a specific
# build-directory path, and a cache hit from a different build dir is a
# real, if intermittent, source of confusing "why is my coverage wrong" bugs.
# This is a one-off diagnostic build; ccache's speed isn't the bottleneck here.
if [ "$ENABLE_COVERAGE" -eq 1 ]; then
  if [ -n "${BUILD_TYPE:-}" ] && [ "${BUILD_TYPE}" != "Debug" ]; then
    echo "NOTE: --coverage forces CMAKE_BUILD_TYPE=Debug, overriding BUILD_TYPE=${BUILD_TYPE}" >&2
  fi
  CMAKE_BUILD_TYPE_VAL="Debug"
  CXX_FLAGS_VAL="-O0 -g --coverage -fprofile-arcs -ftest-coverage"
  EXE_LINKER_FLAGS_VAL="--coverage"
  SHARED_LINKER_FLAGS_VAL="--coverage"  # BUILD_SHARED_LIBS defaults ON; libs need this too
  coverage_args="-DMundy_ENABLE_COVERAGE_TESTING=ON"
  ccache_args=""
  echo "Coverage build: -O0 -g --coverage, CMAKE_BUILD_TYPE=Debug, ccache disabled"
else
  CMAKE_BUILD_TYPE_VAL=${BUILD_TYPE:-RELEASE}
  CXX_FLAGS_VAL="-O3 -march=native"
  EXE_LINKER_FLAGS_VAL=""
  SHARED_LINKER_FLAGS_VAL=""
  coverage_args=""
  if command -v ccache >/dev/null 2>&1; then
    ccache_args="-DCMAKE_CXX_COMPILER_LAUNCHER=$(command -v ccache)"
    echo "Using ccache: $(command -v ccache)"
  else
    ccache_args=""
    echo "ccache not found on PATH; building without it"
  fi
fi

cmake \
-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE_VAL} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="${CXX_FLAGS_VAL}" \
-DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS_VAL}" \
-DCMAKE_SHARED_LINKER_FLAGS="${SHARED_LINKER_FLAGS_VAL}" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-DCTEST_BUILD_FLAGS:STRING="${CTEST_BUILD_FLAGS:--j8}" \
-DCTEST_PARALLEL_LEVEL:STRING="${CTEST_PARALLEL_LEVEL:-8}" \
-DCTEST_BUILD_NAME:STRING="${CTEST_BUILD_NAME:-mundy-cpu-local}" \
-DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-ON} \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
-DTPL_ENABLE_MPI=ON \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_ArborX_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_nanobench_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${KOKKOS_ROOT_DIR} \
-DTPL_KokkosKernels_DIR:PATH=${KOKKOS_KERNELS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Tpetra_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Belos_DIR:PATH=${TRILINOS_ROOT_DIR} \
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
-DMundy_ENABLE_Tpetra=${MUNDY_ENABLE_TPETRA:-OFF} \
-DMundy_ENABLE_Belos=${MUNDY_ENABLE_BELOS:-OFF} \
${coverage_args} \
${ccache_args} \
${MUNDY_SOURCE_DIR}
