#!/bin/bash

#./install_pvfmm.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where OpenRAND will be installed
INSTALL_DIR=$1

# Temporary directory for building OpenRAND
BUILD_DIR="tmp_pvfmm"

# Temporary directory for building
git clone https://github.com/dmalhotra/pvfmm.git --recursive $BUILD_DIR

# Proceed to the build directory and touch up the source code
cd $BUILD_DIR
sed -i 's/Complex<ValueType>(ValueType r=0/Complex(ValueType r=0/g' SCTL/include/sctl/fft_wrapper.hpp
sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 20)/g' CMakeLists.txt
mkdir build && cd build

# Configure, build, and install the project with CMake
cmake .. \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$INSTALL_DIR" \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native" \
  -D PVFMM_EXTENDED_BC:BOOL=ON \
  -D MKL_INCLUDE_DIR:FILEPATH="$MKLROOT/include" \
  -D MKL_FFTW_INCLUDE_DIR:FILEPATH="$MKLROOT/include/fftw" \
  -D MKL_SDL_LIBRARY:STRING="$MKLROOT/lib/intel64/libmkl_rt.so"
make -j$(nproc)
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "PVFMM has been installed to $INSTALL_DIR"
