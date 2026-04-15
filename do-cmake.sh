TRILINOS_ROOT_DIR=$1
TPL_ROOT_DIR=$2
MUNDY_SOURCE_DIR=$3

# bash ../do-cmake.sh /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0 ~/envs/MundyScratch/ ../

# bash ../do-cmake.sh /mnt/ceph/users/bpalmer/envs/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-master-ek7lwb5ilssmazas2p3zhavykp6kiyf4 ~/envs/MundyScratch/ ../

# Using Chris's spack tril16
#   source ~/software/MundyPerformanceTests/apps/conway_prc1_spacing_2026/load_cpu_cje.sh
#   bash ../do-cmake.sh /mnt/home/cedelmaier/Projects/Software/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-16.0.0-jg6itzcs5ms7vsuecbejqfr7l3bbjm2f/ ~/mundyscratch ../

# Using my spack tril16
#   source ~/software/MundyPerformanceTests/apps/conway_prc1_spacing_2026/load_cpu_bp.sh
#   bash ../do-cmake.sh /mnt/home/bpalmer/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-16.0.0-vqcs3hqcerjbv6g3ipewq364pxkjnutn ~/envs/MundyScratch/ ../

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RELEASE} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O3 -march=native" \
-DCMAKE_INSTALL_PREFIX=~/tmp/mundy_install_test/ \
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
-DTPL_Kokkos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_KokkosKernels_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
${MUNDY_SOURCE_DIR}
