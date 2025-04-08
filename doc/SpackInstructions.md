# Trilinos via Spack

## Pre-spack
Make sure all your modules are loaded, we will need them in a little bit. We are also assuming that you have spack installed.

### Modules
This is current as of 8/6/2024. 
```bash
module -q purge
module load modules/2.3-20240529
module load gcc/11.4.0
module load openmpi/4.0.7
module load hdf5/mpi-1.14.3
module load intel-oneapi-mkl/2024.0
module load netcdf-c/4.9.2
module load flexiblas/3.4.2
module load cmake/3.27.9
module load python/3.10.13
module load eigen/3.4.0
module load boost/1.84.0
module load llvm
module load vscode
module load hwloc
```

## Spack

### Load dependencies. For us, this is 
```bash
module purge
module load modules/2.3-20240529
module load slurm openmpi/4.0.7 boost/1.84.0 eigen/3.4.0 hdf5/mpi-1.12.3 hwloc/2.9.1 netcdf-c/4.9.2 gcc cmake
```

```bash
module purge
module load modules/2.3-20240529
module load slurm cuda/12.3.2 openmpi/cuda-4.0.7 gcc/11.4.0 cmake/3.27.9 hwloc openblas hdf5 netcdf-c


git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh
spack env create tril16_gpu
spack env activate tril16_gpu
spack external find cuda
spack external find cmake
spack external find openmpi
spack external find openblas
spack external find hdf5
spack external find hwloc



spack add kokkos+openmp+cuda+cuda_constexpr+cuda_lambda+cuda_relocatable_device_code~cuda_uvm~shared+wrapper cuda_arch=90 ^cuda@12.3.107
spack add magma+cuda cuda_arch=90 ^cuda@12.3.107
spack add trilinos@16.0.0%gcc@11.4.0+belos~boost+exodus+hdf5+kokkos+openmp++cuda+cuda_rdc+stk+zoltan+zoltan2~shared~uvm+wrapper cuda_arch=90 cxxstd=17 ^cuda@12.3.107 ^openblas@0.3.26
```





### Create an independent spack environment for Trilinos
```bash
spack env create -d .
spack env activate .
```

### Setup the compilers
This may autodetect the compilers, or it may not. If it doesn't, then you run the second command.
```bash
spack compilers
spack compiler find
```

### Setup the external packages
These are things that the external finder can deal with.

```bash
spack external find hdf5
spack external find hwloc
spack external find openmpi
spack external find cmake
```

Edit the YAML configuration directly to add things we know about if need be. Or ignore this and `spack` will handle building the packages.
```yaml
    boost:
      externals:
      - spec: boost@1.84.0
        prefix: /mnt/sw/nix/store/a6ai1053d86p4wwzij3skwcflciqfdm7-boost-1.84.0/
    netcdf-c:
      externals:
      - spec: netcdf-c@4.9.2
        prefix: /mnt/sw/nix/store/hxp8ssyrwwl9qm4g6qx9dig8vh3pqd5j-netcdf-c-4.9.2/
```

### Check the specification of Trilinos
```bash
spack spec trilinos@14.2.0 %gcc@11.4.0 +belos +boost +debug +exodus +hdf5 +openmp +stk +zoltan +zoltan2 cxxstd=17
```

### Add Trilinos to the spack environment
```bash
spack add trilinos@14.2.0 %gcc@11.4.0 +belos +boost +debug +exodus +hdf5 +openmp +stk +zoltan +zoltan2 cxxstd=17
```

### Install
```bash
spack install -j 12
```

## Using Trilinos in DEBUG mode

Mesh consistency checks are enabled via an environment variable.

```bash
export STK_MESH_RUN_CONSISTENCY_CHECK=TRUE
```

To link against the debug version of Trilinos, you can get the full path by making sure you're in the environment you installed into and running.

```bash
spack find -p trilinos
```


# TODO

Think about adding PERL to what is knows about as its a pain to install.
Figure out how to set the openblas to flexiblas.

# Fun with circular dependencies

If we want to install Trilinos 15.1.1 with ArborX, we have to deal with the fact that both want Kokkos. Luckily, we can deal with the circular dependency of ArborX wanting Trilinos by installing ArborX first I guess.

## Try installing ArborX first

Install ArborX first:
```bash
spack spec arborx %gcc@11.4.0 +openmp ^kokkos@4.2.01
spack add arborx %gcc@11.4.0 +openmp ^kokkos@4.2.01
spack install -j 12
```

DEPRECATED:
Didn't have the correct version of Trilinos, need 16.0.0 at least for ArborX to be turned on.
Install Trilinos second:
```bash
spack spec trilinos@15.1.1 %gcc@11.4.0 +belos +boost +exodus +hdf5 +openmp +stk +zoltan +zoltan2 ^kokkos@4.2.01
spack add trilinos@15.1.1 %gcc@11.4.0 +belos +boost +exodus +hdf5 +openmp +stk +zoltan +zoltan2 ^kokkos@4.2.01
```

## Trilinos install via cmake
Try to install Trilinos by compiling it with cmake, but install all the dependencies with spack and then import them. Check the Trilinos install for how this works. This is to get around the fact that we don't seem to see ArborX.