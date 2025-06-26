#!/bin/bash

#./install_gtest.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where Google Test will be installed
INSTALL_DIR=$1

# Temporary directory for building Google Test
BUILD_DIR="tmp_gtest"
git clone --depth=1 "https://github.com/google/googletest.git" $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR

# Create a build directory
mkdir build && cd build

# Configure, build, and install the project with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
make -j$(nproc)
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "Google Test has been installed to $INSTALL_DIR"
