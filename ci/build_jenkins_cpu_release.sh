#!/bin/bash
. ${SPACK_ROOT}/share/spack/setup-env.sh
spack env activate ${SPACK_TRILINOS}
mkdir -p ${MUNDY_DEPS}
dep/install_fmt.sh ${MUNDY_DEPS}
dep/install_gtest.sh ${MUNDY_DEPS}
dep/install_openrand.sh ${MUNDY_DEPS}
export TRILINOS_ROOT_DIR=$(spack location -i trilinos)
cmake -B build . \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-O3 -g -fno-omit-frame-pointer -march=native" \
    -DCMAKE_INSTALL_PREFIX=${MUNDY_INSTALL} \
    -DTPL_ENABLE_MPI=ON \
    -DKokkos_ENABLE_SERIAL=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DMundy_ENABLE_MundyCore=ON \
    -DMundy_ENABLE_MundyMath=ON \
    -DMundy_ENABLE_MundyMesh=ON \
    -DMundy_ENABLE_MundyGeom=ON \
    -DMundy_ENABLE_MundyMeta=ON \
    -DMundy_ENABLE_MundyAgents=ON \
    -DMundy_ENABLE_MundyShapes=ON \
    -DMundy_ENABLE_MundyLinkers=ON \
    -DMundy_ENABLE_MundyIo=ON \
    -DMundy_ENABLE_MundyConstraints=ON \
    -DMundy_ENABLE_MundyBalance=OFF \
    -DMundy_ENABLE_MundyMotion=OFF \
    -DMundy_ENABLE_MundyAlens=ON \
    -DMundy_ENABLE_MundyDriver=OFF \
    -DMundy_ENABLE_TESTS=ON \
    -DMundy_ENABLE_GTest=ON \
    -DMundy_ENABLE_STKFMM=OFF \
    -DMundy_ENABLE_PVFMM=OFF \
    -DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
    -DTPL_GTest_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_OpenRAND_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_fmt_DIR:PATH=${MUNDY_DEPS} \
    -DTPL_Kokkos_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_KokkosKernels_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
    -DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
    ${ccache_args} \
    ${compiler_flags} \
    ${install_dir} \
    ${extra_args}' 