#!/bin/bash
# Prepare the dependencies for the CPU release build.
#
# The actual configure/build/test/submit-to-CDash is driven by the TriBITS
# CTest -S script in ci/test_jenkins_cpu_release.sh, so this stage only needs
# to install the Mundy dependency TPLs and check out the submodules.
set -euo pipefail
. ${SPACK_ROOT}/share/spack/setup-env.sh
spack env activate ${SPACK_TRILINOS}
spack find -p
mkdir -p ${MUNDY_DEPS}
bash dep/install_arborx.sh ${MUNDY_DEPS} $(spack location -i kokkos)
bash dep/install_fmt.sh ${MUNDY_DEPS}
bash dep/install_gtest.sh ${MUNDY_DEPS}
bash dep/install_openrand.sh ${MUNDY_DEPS}
bash dep/install_nanobench.sh ${MUNDY_DEPS}
echo "TRILINOS_ROOT_DIR: $(spack location -i trilinos)"
git submodule update --init --recursive
