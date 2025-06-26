#!/bin/bash

#./install_openrand.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where OpenRAND will be installed
INSTALL_DIR=$1

# Temporary directory for building OpenRAND
BUILD_DIR="tmp_openrand"
git clone https://github.com/msu-sparta/OpenRAND.git $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR

# Create a build directory
mkdir build && cd build
echo "Building in: $PWD"


# Configure, build, and install the project with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make -j$(nproc)
make install

# Cleanup
cd "../../"
echo "Current directory: $PWD"
rm -rf $BUILD_DIR

echo "OpenRAND has been installed to $INSTALL_DIR"
