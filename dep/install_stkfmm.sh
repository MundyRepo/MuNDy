#!/bin/bash

#./install_stkfmm.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where OpenRAND will be installed
INSTALL_DIR=$1

# Temporary directory for building OpenRAND
BUILD_DIR="tmp_stkfmm"

# Temporary directory for building
git clone https://github.com/wenyan4work/STKFMM.git --recursive $BUILD_DIR

# Proceed to the build directory and touch up the source code
cd $BUILD_DIR
mkdir build && cd build

# Configure, build, and install the project with CMake
cmake .. \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$INSTALL_DIR" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_TEST=ON \
  -D BUILD_DOC=OFF \
  -D BUILD_M2L=OFF \
  -D PyInterface=OFF \

make -j$(nproc)
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "STKFMM has been installed to $INSTALL_DIR"
