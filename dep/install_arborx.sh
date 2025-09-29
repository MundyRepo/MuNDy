#!/bin/bash

#./install_arborx.sh /path/to/install/directory
#for FI we used bash ./install_arborx.sh ~/envs/MundyScratch /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/lib/cmake/Kokkos

# Check if an install directory was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <install_directory> <kokkos_dir>"
    exit 1
fi

# The directory where ArborX will be installed
INSTALL_DIR=$1
KOKKOS_DIR=$2

# Temporary directory for building ArborX
BUILD_DIR="tmp_arborx"
git clone https://github.com/arborx/ArborX.git $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR
git checkout e026f82

# Create a build directory
mkdir build && cd build
echo "Building in: $PWD"


# Configure, build, and install the project with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -DKokkos_DIR=$KOKKOS_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make -j$(nproc)
make install

# Cleanup
cd "../../"
echo "Current directory: $PWD"
rm -rf $BUILD_DIR

echo "ArborX has been installed to $INSTALL_DIR"
