#!/bin/bash

#./install_kokkos_tools.sh /path/to/install/directory /path/to/kokkos/root
# Mirrors install_arborx.sh: builds kokkos-tools against the kokkos install
# pointed to by KOKKOS_DIR, then installs alongside the rest of mundy_tpl_deps.
# This keeps the kokkos-tools install in lock-step with the kokkos that
# Mundy is building against.

# Check if an install directory was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <install_directory> <kokkos_dir>"
    exit 1
fi

# The directory where kokkos-tools will be installed
INSTALL_DIR=$1
KOKKOS_DIR=$2

# Temporary directory for building kokkos-tools
BUILD_DIR="tmp_kokkos_tools"
git clone https://github.com/kokkos/kokkos-tools.git $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR
git checkout e222e7b

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

echo "kokkos-tools has been installed to $INSTALL_DIR"