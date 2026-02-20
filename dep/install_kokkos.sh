#!/bin/bash

#./install_kokkos.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where Kokkos will be installed
INSTALL_DIR=$1

# Temporary directory for building Kokkos
BUILD_DIR="tmp_kokkos"
git clone https://github.com/kokkos/kokkos.git $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR

# Create a build directory
mkdir build && cd build

# Configure, build, and install the project with CMake
cmake .. \
  -DCMAKE_PREFIX_PATH=~/envs/kokkos_arborx \
  -DCMAKE_INSTALL_PREFIX=~/envs/kokkos_arborx \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make -j$(nproc)
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "Kokkos has been installed to $INSTALL_DIR"