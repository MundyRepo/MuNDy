#!/bin/bash
# Configure, build, test, and submit the GPU release build to CDash.
#
# This drives the TriBITS CTest -S script, which builds Mundy package-by-package
# and submits per-package configure/build/test results to the CDash server
# configured in CTestConfig.cmake (https://my.cdash.org, project "Mundy").
#
# Knobs (override via the environment / Jenkins job):
#   CTEST_TEST_TYPE   CDash track: Experimental (default), Continuous, Nightly
#   CTEST_BUILD_NAME  Build label shown on CDash
#   CTEST_DO_SUBMIT   TRUE (default) to submit; FALSE for a local dry run
set -euo pipefail
. ${SPACK_ROOT}/share/spack/setup-env.sh
spack env activate ${SPACK_TRILINOS}

# Resolve dependency installs from spack, matching do-cmake.sh. Kokkos and
# KokkosKernels are separate standalone specs in different prefixes. Under
# `set -e` a missing spec fails loudly rather than silently substituting.
export TRILINOS_ROOT_DIR=$(spack location -i trilinos)
export KOKKOS_ROOT_DIR=$(spack location -i kokkos)
export KOKKOS_KERNELS_ROOT_DIR=$(spack location -i kokkos-kernels)
export TPL_ROOT_DIR=${MUNDY_DEPS}
export CTEST_BINARY_DIRECTORY=${PWD}/build

export CTEST_TEST_TYPE=${CTEST_TEST_TYPE:-Experimental}
export CTEST_BUILD_NAME=${CTEST_BUILD_NAME:-jenkins-cpu-release}
export CTEST_DO_SUBMIT=${CTEST_DO_SUBMIT:-TRUE}

export OMPI_CXX="$(spack find --format '{prefix}' kokkos | head -1)/bin/nvcc_wrapper"
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"

echo "TRILINOS_ROOT_DIR:            ${TRILINOS_ROOT_DIR}"
echo "KOKKOS_ROOT_DIR:              ${KOKKOS_ROOT_DIR}"
echo "KOKKOS_KERNELS_ROOT_DIR:      ${KOKKOS_KERNELS_ROOT_DIR}"
echo "TPL_ROOT_DIR:                 ${TPL_ROOT_DIR}"
echo "CTEST_TEST_TYPE:              ${CTEST_TEST_TYPE}"
echo "OMPI_CXX:                     ${OMPI_CXX}"
echo "NVCC_WRAPPER_DEFAULT_COMPILER ${NVCC_WRAPPER_DEFAULT_COMPILER}"

# -V for verbose driver output; the script itself runs configure/build/test/submit.
# Don't let a test failure abort the script before we convert results to JUnit;
# capture the driver's exit code and propagate it at the very end. errexit stays
# off for the rest of the script so a failure in the conversion step cannot mask
# the driver's real exit code.
# Chris bumped the file here
set +e
ctest -V -S cmake/ctest/general_gcc/ctest_mpi_openmp_gpu_release.cmake
ctest_rc=$?

# Convert the driver's CTest Test.xml into JUnit for Jenkins (CDash already has
# the native results). The current test day's tag is the first line of TAG.
if [ -f "${CTEST_BINARY_DIRECTORY}/Testing/TAG" ]; then
  tag=$(head -n1 "${CTEST_BINARY_DIRECTORY}/Testing/TAG")
  test_xml="${CTEST_BINARY_DIRECTORY}/Testing/${tag}/Test.xml"
  if [ -f "${test_xml}" ]; then
    python3 ci/ctest_to_junit.py "${test_xml}" "${CTEST_BINARY_DIRECTORY}/ctest_results.xml"
  else
    echo "WARNING: ${test_xml} not found; no JUnit produced." >&2
  fi
else
  echo "WARNING: ${CTEST_BINARY_DIRECTORY}/Testing/TAG not found; no JUnit produced." >&2
fi

exit ${ctest_rc}
