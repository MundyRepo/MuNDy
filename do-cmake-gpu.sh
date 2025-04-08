TRILINOS_ROOT_DIR=$1
TPL_ROOT_DIR=$2
MUNDY_SOURCE_DIR=$3

# bash ../do-cmake-gpu.sh /mnt/sw/nix/store/6wikgk3cr5f1s9dj7rq6ai1ik6f3ncb1-trilinos-14.2.0/ ~/envs/GPUMundyScratch/ ../
# bash ../do-cmake-gpu.sh /mnt/ceph/users/bpalmer/envs/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-16.0.0-wms6rcs5kbxydtaydwbum7ypy3esleak ~/envs/GPUMundyScratch/ ../
echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

# Find the nvcc_wrapper used for Kokkos
export OMPI_CXX=${TRILINOS_ROOT_DIR}/bin/nvcc_wrapper
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1


# Note, MPI_EXEC_MAX_NUMPROCS is an important flag when running on a restricted number of MPI ranks
# such as when you only have one gpu available. Any tests that requests more than this many cores
# will not be run

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-DEBUG} \
-DCMAKE_CXX_COMPILER=${OMPI_CXX} \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR:-$HOME/envs/MundyScratch} \
-DCMAKE_CXX_FLAGS="-O3 -march=native -Wall -Wextra -Wdouble-promotion -Wconversion -lmpi -lcuda" \
-DTPL_ENABLE_MPI=ON \
-DTPL_ENABLE_CUDA=ON \
-DMPI_BASE_DIR=${OPENMPI_BASE} \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_CUDA_UVM:BOOL=ON \
-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
-DMundy_ENABLE_MundyCore=ON \
-DMundy_ENABLE_MundyMath=ON \
-DMundy_ENABLE_MundyGeom=ON \
-DMundy_ENABLE_MundyMesh=ON \
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
-DMPI_EXEC_MAX_NUMPROCS=1\
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
