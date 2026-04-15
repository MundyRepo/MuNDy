#!/bin/bash
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
export TRILINOS_ROOT_DIR=$(spack location -i trilinos)
echo "TRILINOS_ROOT_DIR: ${TRILINOS_ROOT_DIR}"
git submodule update --init --recursive
cmake -B build . \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-O3 -g -fno-omit-frame-pointer -march=native" \
    -DCMAKE_INSTALL_PREFIX=${MUNDY_INSTALL} \
    -DCTEST_BUILD_FLAGS:STRING="${CTEST_BUILD_FLAGS:--j8}" \
    -DCTEST_PARALLEL_LEVEL:STRING="${CTEST_PARALLEL_LEVEL:-8}" \
    -DCTEST_BUILD_NAME:STRING="${CTEST_BUILD_NAME:-jenkins-cpu-release}" \
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
    -DMundy_ENABLE_STKFMM=OFF \
    -DMundy_ENABLE_PVFMM=OFF \
    -DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY" \
    -DTPL_GTest_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_nanobench_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_OpenRAND_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_fmt_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_Kokkos_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_KokkosKernels_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR}
cd build/ && make
