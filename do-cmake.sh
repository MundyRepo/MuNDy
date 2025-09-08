TRILINOS_ROOT_DIR=$1
TPL_ROOT_DIR=$2
MUNDY_SOURCE_DIR=$3

# bash ../do-cmake.sh /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0 ~/envs/MundyScratch/ ../

# bash ../do-cmake.sh /mnt/ceph/users/bpalmer/envs/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-master-ek7lwb5ilssmazas2p3zhavykp6kiyf4 ~/envs/MundyScratch/ ../

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RELEASE} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O3 -march=native" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR:-$HOME/envs/MundyScratch} \
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
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
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
