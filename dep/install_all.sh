#!/bin/bash

#./install_all.sh /path/to/install/directory
#for FI we used 
# bash ./install_all.sh ~/envs/MundyScratch /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/lib/cmake/Kokkos
# bash ./install_all.sh ~/mundyscratch /mnt/home/cedelmaier/Projects/Software/spack/opt/spack/linux-rocky8-cascadelake/gcc-11.4.0/trilinos-16.0.0-jg6itzcs5ms7vsuecbejqfr7l3bbjm2f/lib/cmake/Kokkos
# Check if an install directory was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <install_directory> <kokkos_dir>"
    exit 1
fi

INSTALL_DIR=$1
KOKKOS_DIR=$2

bash ./install_arborx.sh $INSTALL_DIR $KOKKOS_DIR
bash ./install_fmt.sh $INSTALL_DIR
bash ./install_gtest.sh $INSTALL_DIR
bash ./install_nanobench.sh $INSTALL_DIR
bash ./install_openrand.sh $INSTALL_DIR
