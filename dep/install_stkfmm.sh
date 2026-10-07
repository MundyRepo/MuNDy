#!/bin/bash

# Build and install STKFMM, the Stokes and Laplace FMM built on PVFMM.
#
#   ./install_stkfmm.sh /path/to/install/directory
#
# Install PVFMM into the same directory first (install_pvfmm.sh), in the same environment, with FFTW loaded
# (module load fftw/3.3.10). STKFMM finds PVFMM there and links the OpenBLAS and FFTW PVFMM recorded.
#
# Optional environment variables:
#   MARCH  CPU target for -march and -mtune (default: cascadelake). Use the value PVFMM was built with.
#   JOBS   Parallel build jobs (default: 8).

set -euo pipefail

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where STKFMM will be installed
INSTALL_DIR=$1

# STKFMM 56bfce3 (2022-03-10, the latest upstream) is known to work with PVFMM 24f02ef.
STKFMM_COMMIT=56bfce3

MARCH=${MARCH:-cascadelake}
JOBS=${JOBS:-8}
if [ ! -f "$INSTALL_DIR/share/pvfmm/pvfmmConfig.cmake" ]; then
    echo "No PVFMM in $INSTALL_DIR: run install_pvfmm.sh $INSTALL_DIR first."
    exit 1
fi
if [ -z "${FFTW_ROOT:-}" ]; then
    echo "FFTW_ROOT is not set: load FFTW first (module load fftw/3.3.10)."
    exit 1
fi

# -O3 and -DNDEBUG come from the Release build type. -fno-math-errno lets sqrt vectorize, as in MundyMath.
CXX_FLAGS="-march=${MARCH} -mtune=${MARCH} -fno-math-errno"

# Temporary directory for building STKFMM
BUILD_DIR="tmp_stkfmm"

# Fetch the pinned commit
git clone https://github.com/wenyan4work/STKFMM.git $BUILD_DIR
cd $BUILD_DIR
git checkout $STKFMM_COMMIT
git submodule update --init --recursive
mkdir build && cd build

# Configure, build, and install the project with CMake. The fftw module puts FFTW on LIBRARY_PATH, which CMake treats
# as an implicit link directory and leaves out of the run path, so it is named explicitly.
cmake .. \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$INSTALL_DIR" \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_CXX_FLAGS:STRING="$CXX_FLAGS" \
  -D CMAKE_PREFIX_PATH:PATH="$INSTALL_DIR" \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON \
  -D CMAKE_INSTALL_RPATH:PATH="$FFTW_ROOT/lib" \
  -D BUILD_TEST=ON \
  -D BUILD_DOC=OFF \
  -D BUILD_M2L=OFF \
  -D PyInterface=OFF
make -j"$JOBS"
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "STKFMM $STKFMM_COMMIT has been installed to $INSTALL_DIR (-march=$MARCH)"
