FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

RUN apt -y update && \
    apt -y install curl gpg software-properties-common && \
    curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-add-repository -y multiverse && \
    apt-add-repository -y universe && \
    apt -y update && \
    apt -y install intel-oneapi-mkl-devel build-essential g++ gfortran make bzip2 tar patch \
                   cmake libeigen3-dev ninja-build git python3-pip vim gdb && \
    apt -y autoclean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

ARG SPACK_TAG=v0.23.1
RUN git clone -c feature.manyFiles=true --depth=2 --branch ${SPACK_TAG} https://github.com/spack/spack.git /opt/spack
ENV SPACK_ROOT=/opt/spack
RUN echo "source $SPACK_ROOT/share/spack/setup-env.sh" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-c"]

ENV SPACK_TRILINOS=/root/spack_trilinos
RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
    mkdir $SPACK_TRILINOS && \
    cd $SPACK_TRILINOS && \
    spack env create -d $SPACK_TRILINOS && \
    spack env activate $SPACK_TRILINOS && \
    spack add trilinos@16.0.0%gcc@11.4.0+belos~boost+exodus+hdf5+kokkos+openmp~shared+stk+zoltan+zoltan2 cxxstd=17 ^kokkos@4.3.01+hwloc+openmp~shared ^openmpi@4.1.2 && \
    spack add cmake@3.23 && \
    spack concretize && \
    spack install -j12 && \
    spack load cmake@3.23 && \
    spack gc -y
