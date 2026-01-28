# Install mundy
git clone https://github.com/MundyRepo/MuNDy.git --recursive
cd MuNDy

TPL_ROOT_DIR=$PWD/dep/tpls
echo "Using TPL dir: ${TPL_ROOT_DIR}"

# Install third-party libraries (TPLs) | fmt, Kokkos, OpenRAND, GTEST
cd ./dep
bash ./install_fmt.sh ${TPL_ROOT_DIR}
bash ./install_kokkos.sh ${TPL_ROOT_DIR}
bash ./install_openrand.sh ${TPL_ROOT_DIR}
bash ./install_gtest.sh ${TPL_ROOT_DIR}
bash ./install_nanobench.sh ${TPL_ROOT_DIR}

# Build MuNDy
cd ../
mkdir build && cd build

cmake \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O3 -march=native" \
-DCMAKE_INSTALL_PREFIX=${TPL_ROOT_DIR} \
-DMundy_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-DMundy_ENABLE_MundyCore=ON \
-DMundy_ENABLE_MundyMath=ON \
-DMundy_ENABLE_MundyGeom=ON \
-DMundy_ENABLE_MundyMesh=OFF \
-DMundy_ENABLE_MundyMeta=OFF \
-DMundy_ENABLE_MundyAgents=OFF \
-DMundy_ENABLE_MundyShapes=OFF \
-DMundy_ENABLE_MundyLinkers=OFF \
-DMundy_ENABLE_MundyIo=OFF \
-DMundy_ENABLE_MundyConstraints=OFF \
-DMundy_ENABLE_MundyBalance=OFF \
-DMundy_ENABLE_MundyMotion=OFF \
-DMundy_ENABLE_MundyAlens=OFF \
-DMundy_ENABLE_MundyDriver=OFF \
-DMundy_ENABLE_TESTS=ON \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_fmt_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
..

make -j12
ctest -j12