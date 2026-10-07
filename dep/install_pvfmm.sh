#!/bin/bash

# Build and install PVFMM, the kernel-independent FMM library STKFMM is built on.
#
#   ./install_pvfmm.sh /path/to/install/directory
#
# Load the Mundy environment first, plus FFTW, e.g.
#   source ~cedelmaier/bin/env_loads/load_tril1610_shared_cascadelake_rocky9.sh
#   module load fftw/3.3.10
#
# PVFMM links
#   - the OpenBLAS (with LAPACK) that Trilinos already links, so a Mundy executable carries a single BLAS. It is found
#     with `spack location -i openblas` in the active spack environment; set PVFMM_OPENBLAS_ROOT to override.
#     (Not OPENBLAS_ROOT: the openblas module sets that to its own, different OpenBLAS.)
#   - FFTW from FFTW_ROOT, which the fftw module sets.
# The installed library records both in its run path, so it runs without those modules loaded.
#
# Optional environment variables:
#   MARCH                CPU target for -march and -mtune (default: cascadelake). The library only runs on CPUs
#                        that support it: cascadelake needs AVX-512 (Skylake-SP and newer, AMD Zen 4).
#   JOBS                 Parallel build jobs (default: 8).
#   PVFMM_OPENBLAS_ROOT  OpenBLAS install prefix (default: from spack).

set -euo pipefail

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where PVFMM will be installed
INSTALL_DIR=$1

# PVFMM 24f02ef (2023-08-03), with its SCTL f13f65724, is the version STKFMM 56bfce3 is known to work with. Later
# PVFMM retires the pvfmm::Vector alias STKFMM uses.
PVFMM_COMMIT=24f02ef

MARCH=${MARCH:-cascadelake}
JOBS=${JOBS:-8}
OPENBLAS_PREFIX=${PVFMM_OPENBLAS_ROOT:-$(spack location -i openblas)}
if [ -z "${FFTW_ROOT:-}" ]; then
    echo "FFTW_ROOT is not set: load FFTW first (module load fftw/3.3.10)."
    exit 1
fi

# -O3 and -DNDEBUG come from the Release build type. -fno-math-errno lets sqrt vectorize, as in MundyMath.
CXX_FLAGS="-march=${MARCH} -mtune=${MARCH} -fno-math-errno"

# Temporary directory for building PVFMM
BUILD_DIR="tmp_pvfmm"

# Fetch the pinned commit and the SCTL it records
git clone https://github.com/dmalhotra/pvfmm.git $BUILD_DIR
cd $BUILD_DIR
git checkout $PVFMM_COMMIT
git submodule update --init --recursive

# Touch up the source code so the headers compile as C++20, as Mundy includes them
sed -i 's/Complex<ValueType>(ValueType r=0/Complex(ValueType r=0/g' SCTL/include/sctl/fft_wrapper.hpp
sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 20)/g' CMakeLists.txt
mkdir build && cd build

# Configure, build, and install the project with CMake. MKL is excluded so the OpenBLAS and FFTW above are the only
# BLAS, LAPACK, and FFT candidates. The fftw module puts FFTW on LIBRARY_PATH, which CMake treats as an implicit link
# directory and leaves out of the run path, so it is named explicitly.
cmake .. \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$INSTALL_DIR" \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_CXX_FLAGS:STRING="$CXX_FLAGS" \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON \
  -D CMAKE_INSTALL_RPATH:PATH="$FFTW_ROOT/lib" \
  -D CMAKE_DISABLE_FIND_PACKAGE_MKL:BOOL=ON \
  -D CMAKE_PREFIX_PATH:PATH="$OPENBLAS_PREFIX" \
  -D BLA_VENDOR:STRING=OpenBLAS \
  -D FFTW_ROOT:PATH="$FFTW_ROOT" \
  -D PVFMM_EXTENDED_BC:BOOL=ON
make -j"$JOBS"
make install

# Cleanup
cd "../../"
rm -rf $BUILD_DIR

echo "PVFMM $PVFMM_COMMIT has been installed to $INSTALL_DIR (-march=$MARCH, OpenBLAS from $OPENBLAS_PREFIX)"
