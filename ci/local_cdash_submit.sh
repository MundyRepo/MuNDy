#!/bin/bash
# Local CDash submission for Mundy (no Jenkins).
#
# Drives the TriBITS ctest -S script: configure/build/test, then submit to
# https://my.cdash.org/index.php?project=Mundy under the Experimental group.
# Incremental by default (reuses BUILD_DIR); pass --clean for a full rebuild.
#
# Prereq: activate your spack env first (provides trilinos/kokkos/kokkos-kernels),
#   spack env activate <env>      # e.g. tril161
#
# Env overrides:
#   TPL_ROOT_DIR  Mundy dep prefix (gtest/arborx/fmt/openrand/nanobench)  [default $HOME/envs/MundyScratch]
#   BUILD_DIR     persistent ctest build dir                             [default $HOME/mundy_ctest_dev]
#   NO_SUBMIT=1   build+test but skip the CDash submit (dry run)
set -euo pipefail

# Always run from the repo root so the ctest -S path resolves.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg (use --clean / --help)" >&2; exit 1 ;;
  esac
done

command -v spack >/dev/null 2>&1 \
  || { echo "ERROR: spack not on PATH; activate your env first (e.g. spack env activate tril161)." >&2; exit 1; }

locate() {  # spack spec -> install prefix; fail loudly if absent
  spack location -i "$1" 2>/dev/null \
    || { echo "ERROR: spack spec '$1' not installed in the active env." >&2; exit 1; }
}
export TRILINOS_ROOT_DIR=$(locate trilinos)
export KOKKOS_ROOT_DIR=$(locate kokkos)
export KOKKOS_KERNELS_ROOT_DIR=$(locate kokkos-kernels)

export TPL_ROOT_DIR="${TPL_ROOT_DIR:-$HOME/envs/MundyScratch}"
[ -d "${TPL_ROOT_DIR}" ] \
  || { echo "ERROR: TPL_ROOT_DIR not found: ${TPL_ROOT_DIR} (override with TPL_ROOT_DIR=...)." >&2; exit 1; }

# Persistent build dir => incremental rebuilds. Not ./build (avoids cache thrash
# with a hand-built tree; the driver configures with extra CI-only flags).
export CTEST_BINARY_DIRECTORY="${BUILD_DIR:-$HOME/mundy_ctest_dev}"
export CTEST_TEST_TYPE=Experimental
export CTEST_BUILD_NAME="local-$(whoami)"

if [ -n "${NO_SUBMIT:-}" ]; then export CTEST_DO_SUBMIT=FALSE; else export CTEST_DO_SUBMIT=TRUE; fi

nprocs=$(nproc 2>/dev/null || echo 8)
export CTEST_BUILD_FLAGS="-j${nprocs}"
export CTEST_PARALLEL_LEVEL="${nprocs}"

# Incremental unless --clean: keep objects + CMakeCache across runs. Re-run with
# --clean once after changing configure options (TPL dirs, flags).
if [ "${CLEAN}" -eq 1 ]; then
  export CTEST_START_WITH_EMPTY_BINARY_DIRECTORY=TRUE
  export CTEST_WIPE_CACHE=TRUE
else
  export CTEST_START_WITH_EMPTY_BINARY_DIRECTORY=FALSE
  export CTEST_WIPE_CACHE=FALSE
fi

echo "Repo:      ${REPO_ROOT}"
echo "Build dir: ${CTEST_BINARY_DIRECTORY}  (clean=${CLEAN})"
echo "TPL root:  ${TPL_ROOT_DIR}"
echo "Submit:    ${CTEST_DO_SUBMIT}   Track: ${CTEST_TEST_TYPE}   Jobs: ${nprocs}"

ctest -V -S cmake/ctest/general_gcc/ctest_mpi_openmp_release.cmake
