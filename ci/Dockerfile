FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Reset the nvidia/cuda base image ENTRYPOINT so Jenkins' `docker run ... cat`
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
    curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-add-repository -y multiverse && \
    apt-add-repository -y universe && \
    apt-get -y update && \
    apt-get -y install --no-install-recommends \
        intel-oneapi-mkl-devel \
        build-essential \
        g++ \
        gfortran \
        make \
        bzip2 \
        tar \
        patch \
        cmake \
        libeigen3-dev \
        ninja-build \
        git \
        python3-pip \
        vim \
        gdb && \
    apt-get -y autoclean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Create the testuser while still root. The home directory is made world-traversable
# (and later the spack tree world-readable) so that Jenkins, which launches the
# container as the host agent's uid (not testuser), can still read/source spack.
RUN useradd -m -s /bin/bash testuser && \
    chmod 755 /home/testuser

ENV SPACK_ROOT=/home/testuser/spack
ENV SPACK_ENV_NAME=tril161_cpu

# Make spack available in interactive shells for any user (must be done as root).
RUN echo "source ${SPACK_ROOT}/share/spack/setup-env.sh" >> /etc/bash.bashrc

SHELL ["/bin/bash", "-c"]

# Everything below runs as testuser.
USER testuser
WORKDIR /home/testuser

ARG SPACK_TAG=releases/v1.1
RUN git clone --depth=2 --branch=${SPACK_TAG} https://github.com/spack/spack.git ${SPACK_ROOT}

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env create ${SPACK_ENV_NAME} && \
    spack env activate ${SPACK_ENV_NAME} && \
    spack add trilinos@16.1.0+belos~boost+exodus+hdf5+kokkos+openmp+shared+stk+zoltan+zoltan2 cxxstd=17 %gcc@11.4.0 ^kokkos@4.5.01+hwloc+openmp+serial+shared ^openmpi@4.1.2 && \
    spack add cmake@3.27.9 && \
    spack concretize && \
    spack install && \
    spack load cmake@3.27.9 && \
    spack gc -y && \
    chmod -R a+rX ${SPACK_ROOT}
