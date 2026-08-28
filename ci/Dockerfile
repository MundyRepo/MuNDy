FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04

# Reset the ubuntu base image ENTRYPOINT so Jenkins' `docker run ... cat`
# keep-alive pattern works without triggering the
# "container started but didn't run the expected command" error.
ENTRYPOINT []

ENV DEBIAN_FRONTEND=noninteractive

# Install base packages including ca-certificates BEFORE any curl calls
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
        ca-certificates \
        curl \
        gpg \
        software-properties-common && \
    apt-add-repository -y multiverse && \
    apt-add-repository -y universe && \
    apt-get -y update && \
    apt-get -y install --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        bzip2 \
        ccache \
        cmake \
        g++ \
        gdb \
        gfortran \
        git \
        libeigen3-dev \
        libncurses-dev \
        libnuma-dev \
        libtool \
        make \
        ninja-build \
        numactl \
        openssh-server \
        patch \
        perl \
        pkg-config \
        python3-pip \
        vim \
        tar && \
    apt-get -y autoclean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Create the testuser while still root. The home directory is made world-traversable
# (and later the spack tree world-readable) so that Jenkins, which launches the
# container as the host agent's uid (not testuser), can still read/source spack.
RUN useradd -m -s /bin/bash testuser && \
    chmod 755 /home/testuser

ENV SPACK_ROOT=/home/testuser/spack

# Make spack available in interactive shells for any user (must be done as root).
RUN echo "source ${SPACK_ROOT}/share/spack/setup-env.sh" >> /etc/bash.bashrc

SHELL ["/bin/bash", "-c"]

# Everything below runs as testuser.
USER testuser
WORKDIR /home/testuser

# ARG SPACK_TAG=v1.2.2
# RUN git clone --depth=2 --branch ${SPACK_TAG} https://github.com/spack/spack.git ${SPACK_ROOT}
RUN mkdir -p ${SPACK_ROOT} && cd ${SPACK_ROOT} && \
    git init -q && \
    git remote add origin https://github.com/spack/spack.git && \
    git fetch --depth=1 origin 24ecb489befa7d8ddf4ac6163c3d30819ad85915 && \
    git checkout -q FETCH_HEAD

# Pin the package recipes
RUN printf '%s\n' \
      'repos:' \
      '  builtin:' \
      '    git: https://github.com/spack/spack-packages.git' \
      '    commit: 1a79dd1ae9ff26bd0dda5a6fdc10d95df46c6ff0' \
      "    destination: ${SPACK_ROOT}/spack-packages" \
      > ${SPACK_ROOT}/etc/spack/repos.yaml && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack repo list

###############################################################
# CPU Build
###############################################################
ENV SPACK_ENV_NAME_CPU=tril1610_cpu
# Concretize first in case there are errors.
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env create ${SPACK_ENV_NAME_CPU} && \
    spack env activate ${SPACK_ENV_NAME_CPU} && \
    spack config add "packages:all:target:[broadwell]" && \
    spack external find autoconf \
        automake \
        bzip2 \
        cmake \
        diffutils \
        gcc \
        gmake \
        findutils \
        libtool \
        m4 \
        ncurses \
        numactl \
        openssh \
        perl \
        pkgconf \
        python \
        tar \
        xz && \
    spack add trilinos@16.1.0 \
            +belos \
            ~boost \
            +exodus \
            +hdf5 \
            +kokkos \
            ~gtest \
            +mpi \
            +openmp \
            +shared \
            +stk \
            +zoltan \
            +zoltan2 \
            cxxstd=17 %gcc@13 \
        ^kokkos@4.5.01 \
            +hwloc \
            +openmp \
            +serial \
            +shared \
        ^kokkos-kernels@4.5.01 \
            +blas \
            +lapack \
            +openmp \
            +serial \
            +shared \
        ^openmpi@4.1.8 && \
    spack concretize -f

# Build in the image only if the concretize succeeded
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate ${SPACK_ENV_NAME_CPU} && \
    spack install && \
    spack gc -y && \
    chmod -R a+rX ${SPACK_ROOT}

###############################################################
# GPU Build
###############################################################
ENV SPACK_ENV_NAME_GPU=tril1610_sm80_cuda1263
# Concretize first in case there are errors.
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env create ${SPACK_ENV_NAME_GPU} && \
    spack env activate ${SPACK_ENV_NAME_GPU} && \
    spack config add "packages:all:target:[broadwell]" && \
    spack external find autoconf \
        automake \
        bzip2 \
        cmake \
        cuda \
        diffutils \
        gcc \
        gmake \
        findutils \
        libtool \
        m4 \
        ncurses \
        numactl \
        openssh \
        perl \
        pkgconf \
        python \
        tar \
        xz && \
    spack add trilinos@16.1.0 \
            +belos \
            ~boost \
            +cuda \
            +cuda_constexpr \
            +cuda_rdc \
            +exodus \
            +hdf5 \
            ~gtest \
            +kokkos \
            +mpi \
            +openmp \
            ~shared \
            +stk \
            ~uvm \
            +wrapper \
            +zoltan \
            +zoltan2 \
            cuda_arch=80 cxxstd=17 %gcc@13 \
        ^kokkos@4.5.01 \
            +cuda \
            +cuda_constexpr \
            +cuda_lambda \
            +cuda_relocatable_device_code \
            ~cuda_uvm \
            +openmp \
            +serial \
            ~shared \
            +wrapper \
            cuda_arch=80 \
        ^kokkos-kernels@4.5.01 \
            +cublas \
            +cuda \
            +cusolver \
            +openmp \
            +serial \
            ~shared \
            cuda_arch=80 \
        ^openmpi@4.1.8 \
            +atomics \
            +cuda \
            cuda_arch=80 && \
    spack concretize -f

# Build in the image only if the concretize succeeded
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate ${SPACK_ENV_NAME_GPU} && \
    spack install && \
    spack gc -y && \
    chmod -R a+rX ${SPACK_ROOT}